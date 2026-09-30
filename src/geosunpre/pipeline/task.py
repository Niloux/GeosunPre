"""Freeze the execution recipe and delivery options once per work directory."""

import json
import os
import platform
from datetime import datetime, timezone
from importlib.util import find_spec
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from geosunpre.artifacts import digest, file_digest, write_frozen_json, write_json
from geosunpre.config import CONFIG_FILE
from geosunpre.device import DEVICE_PROFILE

TASK_FILE = 'pipeline_task.json'
# These modules check resources and record identities; they do not fit geometry.
AUDIT_ONLY_SOURCES = {'pipeline/preflight.py', 'pipeline/task.py'}


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
        native_source = package.parent.parent / 'native' / name
        if native_source.is_file():
            sources['native/' + name] = file_digest(native_source)
    spec = find_spec('geosunpre_native')
    if spec and spec.origin and Path(spec.origin).is_file():
        sources['native_extension'] = file_digest(spec.origin)
    lock = package.parent.parent / 'uv.lock'
    if lock.is_file():
        sources['uv.lock'] = file_digest(lock)
    return {'source_sha256': sources, 'packages': packages,
            'python': platform.python_version(), 'device_profile': DEVICE_PROFILE,
            'depth_backend': os.environ.get('GEOSUNPRE_DEPTH_BACKEND', 'native'),
            'matcher_graph': os.environ.get('GEOSUNPRE_MATCH_GRAPH', '1'),
            'matcher_math': os.environ.get('GEOSUNPRE_MATCH_MATH', 'optimized')}


def _execution_changes(saved, current):
    changes = []
    for key in sorted(set(saved) | set(current)):
        old, new = saved.get(key), current.get(key)
        if isinstance(old, dict) and isinstance(new, dict):
            for name in sorted(set(old) | set(new)):
                if key == 'source_sha256' and name in AUDIT_ONLY_SOURCES:
                    continue
                if old.get(name) != new.get(name):
                    changes.append(f'{key}.{name}')
        elif old != new:
            changes.append(key)
    return changes


def _record_control_changes(work, task, execution):
    saved = task['execution'].get('source_sha256', {})
    current = execution.get('source_sha256', {})
    changes = {name: {'frozen_sha256': saved.get(name), 'current_sha256': current.get(name)}
               for name in sorted(AUDIT_ONLY_SOURCES) if saved.get(name) != current.get(name)}
    if not changes:
        return
    path = work / 'pipeline_execution_audit.json'
    audit = (json.loads(path.read_text()) if path.is_file() else
             {'version': 1, 'task_fingerprint': task['fingerprint'], 'changes': []})
    if audit.get('version') != 1 or audit.get('task_fingerprint') != task['fingerprint']:
        raise ValueError('Pipeline execution audit belongs to another task')
    record = {'task_sha256': file_digest(work / TASK_FILE), 'source_changes': changes}
    if not any({k: v for k, v in entry.items() if k != 'recorded_at_utc'} == record
               for entry in audit['changes']):
        audit['changes'].append({'recorded_at_utc': datetime.now(timezone.utc).isoformat(), **record})
        write_json(path, audit)


def bind_task(work, *, output=None, image_mode=None, include_masks=None,
              include_ply=None, model_cache=None, legacy=False):
    """Omitted options inherit the task; alternate exports use a new destination."""
    work = Path(work)
    path = work / TASK_FILE
    explicit = {'image_mode': image_mode, 'include_masks': include_masks, 'include_ply': include_ply}
    for key in ('include_masks', 'include_ply'):
        if explicit[key] is not None and type(explicit[key]) is not bool:
            raise ValueError(f'{key} must be a boolean')
    execution = execution_identity()
    if path.is_file():
        task = json.loads(path.read_text())
        if task.get('version') != 1 or task.get('fingerprint') != digest(
                {k: v for k, v in task.items() if k != 'fingerprint'}):
            raise ValueError('Pipeline task identity changed')
        changed = _execution_changes(task['execution'], execution)
        if task['config_sha256'] != file_digest(work / CONFIG_FILE):
            changed.append(CONFIG_FILE)
        if changed:
            raise ValueError('Frozen task execution changed: ' + ', '.join(changed) +
                             '; use a fresh work directory')
        delivery = dict(task['delivery'])
    else:
        # Old successful exports provide the delivery options their config omitted.
        previous = {}
        result_path = work / 'run_result.json'
        if legacy and result_path.is_file():
            previous = json.loads(result_path.read_text()).get('export') or {}
        destination = Path(output or previous.get('output') or work / 'colmap').expanduser().resolve()
        old_identity = destination / 'colmap_export.json'
        old = json.loads(old_identity.read_text()) if legacy and old_identity.is_file() else {}
        delivery = {'output': str(destination),
                    'image_mode': image_mode if image_mode is not None else old.get('image_mode', 'copy'),
                    'include_masks': include_masks if include_masks is not None else old.get('include_masks', False),
                    'include_ply': include_ply if include_ply is not None else
                        'sparse/0/points3D.ply' in old.get('supplemental_sha256', {})}
        task = {'version': 1, 'config_sha256': file_digest(work / CONFIG_FILE),
                'execution': execution, 'delivery': delivery,
                'model_cache': str(Path(model_cache).expanduser().resolve()) if model_cache else None,
                'legacy_source_identity': 'unrecorded' if legacy else None}
        if delivery['image_mode'] not in ('copy', 'link'):
            raise ValueError('image_mode must be copy or link')
        write_frozen_json(path, {**task, 'fingerprint': digest(task)})
        task = json.loads(path.read_text())
    destination = str(Path(output).expanduser().resolve()) if output else delivery['output']
    for key, value in explicit.items():
        if value is not None:
            if destination == delivery['output'] and value != delivery[key]:
                raise ValueError(f'Frozen delivery option changed: {key}; choose a new --output')
            delivery[key] = value
    delivery['output'] = destination
    cache = str(Path(model_cache).expanduser().resolve()) if model_cache else task['model_cache']
    _record_control_changes(work, task, execution)
    return task, delivery, cache
