"""Freeze one execution recipe and inherit delivery options on resume."""

import json
import os
import platform
from importlib.util import find_spec
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from geosunpre.artifacts import digest, file_digest, write_frozen_json
from geosunpre.config import CONFIG_FILE
from geosunpre.device import DEVICE_PROFILE
from geosunpre.masks import MODEL, MODEL_REVISION
from geosunpre.initialization import DEFAULT_VOXEL_SIZE, initialization_recipe

TASK_FILE = 'pipeline_task.json'


def configure_offline():
    os.environ.setdefault('TORCH_HOME', str(Path('.cache/torch').resolve()))
    os.environ.setdefault('HF_HUB_CACHE', str(Path('.cache/huggingface').resolve()))
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'


def execution_identity():
    package = Path(__file__).resolve().parent.parent
    packages = {}
    for name in ('geosunpre-native', 'pycolmap', 'pyceres', 'torch', 'torchvision',
                 'lomatch', 'transformers', 'numpy', 'scipy', 'Pillow', 'opencv-python'):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    sources = {p.relative_to(package).as_posix(): file_digest(p)
               for p in sorted(package.rglob('*.py'))}
    for name in ('depth_guard.cpp', 'joint_costs.h'):
        path = package.parent.parent / 'native' / name
        if path.is_file():
            sources['native/' + name] = file_digest(path)
    spec = find_spec('geosunpre_native')
    if spec and spec.origin and Path(spec.origin).is_file():
        sources['native_extension'] = file_digest(spec.origin)
    lock = package.parent.parent / 'uv.lock'
    if lock.is_file():
        sources['uv.lock'] = file_digest(lock)
    return {'source_sha256': sources, 'packages': packages, 'python': platform.python_version(),
            'device_profile': DEVICE_PROFILE, 'matcher': 'flash_cuda_graph', 'depth_backend': 'native'}


def _execution_changes(saved, current):
    changes = []
    for key in sorted(set(saved) | set(current)):
        old, new = saved.get(key), current.get(key)
        if isinstance(old, dict) and isinstance(new, dict):
            changes.extend(f'{key}.{name}' for name in sorted(set(old) | set(new))
                           if old.get(name) != new.get(name))
        elif old != new:
            changes.append(key)
    return changes


def _weights_identity(model_cache, families):
    checkpoints = Path(os.environ['TORCH_HOME']).expanduser().resolve() / 'hub/checkpoints'
    snapshot = Path(model_cache) / ('models--' + MODEL.replace('/', '--')) / 'snapshots' / MODEL_REVISION
    paths = [checkpoints / name for name in ('dad.pth', 'dinov2_vitl14_pretrain.pth', 'loma_B.pt')]
    paths += [snapshot / name for name in
              ('config.json', 'preprocessor_config.json', 'model.safetensors', 'pytorch_model.bin')]
    groups = {'frontend_weights.json': paths[:3], 'segmentation_weights.json': paths[3:]}
    return {name: {str(path): file_digest(path) if path.is_file() else None for path in groups[name]}
            for name in families}


def bind_weights(work, model_cache, *, frontend=False, segmentation=False):
    """Freeze each model family when first used; preparation needs no weights."""
    needed = {'frontend_weights.json': frontend, 'segmentation_weights.json': segmentation}
    families = [name for name, required in needed.items() if required or (Path(work) / name).is_file()]
    if not families:
        return
    for name, identity in _weights_identity(model_cache, families).items():
        path = Path(work) / name
        if path.is_file():
            if json.loads(path.read_text()) != identity:
                raise ValueError('Frozen model weights changed: ' + name +
                                 '; use a fresh work directory')
        else:
            write_frozen_json(path, identity)


def bind_task(work, *, output=None, image_mode=None, include_masks=None,
              include_ply=None, ply_voxel_size=None, model_cache=None):
    work = Path(work)
    path = work / TASK_FILE
    configure_offline()
    explicit = {'image_mode': image_mode, 'include_masks': include_masks,
                'include_ply': include_ply, 'ply_voxel_size': ply_voxel_size}
    if ply_voxel_size is not None:
        initialization_recipe(ply_voxel_size)
    if image_mode is not None and image_mode not in ('copy', 'link'):
        raise ValueError('image_mode must be copy or link')
    for key in ('include_masks', 'include_ply'):
        if explicit[key] is not None and type(explicit[key]) is not bool:
            raise ValueError(f'{key} must be a boolean')
    execution = execution_identity()
    if path.is_file():
        task = json.loads(path.read_text())
        if task.get('version') != 2 or task.get('fingerprint') != digest(
                {k: v for k, v in task.items() if k != 'fingerprint'}):
            raise ValueError('Pipeline task identity changed; use a fresh work directory')
        cache = task['model_cache']
        if model_cache is not None and str(Path(model_cache).expanduser().resolve()) != cache:
            raise ValueError('Frozen model cache changed; use a fresh work directory')
        changed = _execution_changes(task['execution'], execution)
        if task['config_sha256'] != file_digest(work / CONFIG_FILE):
            changed.append(CONFIG_FILE)
        if changed:
            raise ValueError('Frozen task execution changed: ' + ', '.join(changed) +
                             '; use a fresh work directory')
        bind_weights(work, cache)
        delivery = dict(task['delivery'])
    else:
        if (work / 'manifest.json').exists():
            raise ValueError('Missing pipeline_task.json for existing artifacts; use a fresh work directory')
        cache = str(Path(model_cache or os.environ['HF_HUB_CACHE']).expanduser().resolve())
        destination = str(Path(output or work / 'colmap').expanduser().resolve())
        delivery = {'output': destination, 'image_mode': image_mode or 'copy',
                    'include_masks': True if include_masks is None else include_masks,
                    'include_ply': True if include_ply is None else include_ply,
                    'ply_voxel_size': DEFAULT_VOXEL_SIZE if ply_voxel_size is None else float(ply_voxel_size)}
        task = {'version': 2, 'config_sha256': file_digest(work / CONFIG_FILE),
                'execution': execution, 'delivery': delivery, 'model_cache': cache}
        task['fingerprint'] = digest(task)
        write_frozen_json(path, task)
    destination = str(Path(output).expanduser().resolve()) if output else delivery['output']
    for key, value in explicit.items():
        if value is not None:
            if destination == delivery['output'] and value != delivery[key]:
                raise ValueError(f'Frozen delivery option changed: {key}; choose a new --output')
            delivery[key] = value
    delivery['output'] = destination
    return task, delivery, cache
