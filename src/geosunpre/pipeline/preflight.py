"""Check resources required by pending work before expensive stages start."""

import json
import os
import shutil
from importlib.util import find_spec
from pathlib import Path

from geosunpre.features import feature_input
from geosunpre.features import FRONTEND_CONFIG as CONFIG
from geosunpre.masks import MODEL, MODEL_REVISION, mask_cached
from geosunpre.device import DEVICE_PROFILE


def preflight(work, stages, delivery, *, model_cache=None):
    work = Path(work)
    root = work / 'sfm'
    scene_path = root / 'manifest.json'
    base_path = work / 'manifest.json'
    scene = json.loads((scene_path if scene_path.is_file() else base_path).read_text(encoding='utf-8'))
    images = scene['images']
    missing_masks = 'masks' in stages and any(not mask_cached(r, root / 'masks') for r in images)
    missing_features = False
    if 'features' in stages:
        if not scene_path.is_file():
            missing_features = True
        else:
            missing_features = any(not feature_input(work, scene, CONFIG, r)[3].is_file() for r in images)
    missing_matches = False
    if 'matches' in stages:
        path = root / 'frontend.json'
        data = json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {}
        missing_matches = not data.get('pairs') or any(not Path(p['path']).is_file() for p in data.get('pairs', []))
    gpu_needed = missing_masks or missing_features or missing_matches
    result = {'gpu_needed': gpu_needed, 'device_profile': DEVICE_PROFILE, 'warnings': []}
    if any(s in stages for s in ('register', 'export')):
        from geosunpre.registration.joint_solver import native_costs
        from geosunpre.pipeline.registration import freeze_inputs
        native = native_costs()
        inputs = freeze_inputs(work)
        result['registration'] = {'las': inputs['las'], 'inputs_file': inputs['inputs_file'],
                                  'native_implementation': native.implementation_version,
                                  'target_frame': 'input_las_xyz'}
    if any(s in stages for s in ('calibration', 'sfm', 'finalize')):
        from geosunpre.sfm.depth_guard import depth_cost_class
        cost = depth_cost_class()
        result['depth_backend'] = cost.__module__
    if gpu_needed:
        modules = {'torch'}
        if missing_features or missing_matches:
            modules.add('loma')
        if missing_masks:
            modules.add('transformers')
        for module in sorted(modules):
            if find_spec(module) is None:
                raise RuntimeError(f'Missing {module}; install with uv sync --locked')
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('Pending masks/features/matches require an available CUDA GPU')
        if missing_matches:
            from torch.nn.attention import sdpa_kernel
            from geosunpre.matching import ATTENTION_BACKEND, attention_backend
            # Test the requested kernel with the pinned matcher dtype/head width.
            with torch.inference_mode(), sdpa_kernel(attention_backend()):
                sample = torch.zeros((1, 4, 16, 64), dtype=torch.bfloat16, device='cuda')
                torch.nn.functional.scaled_dot_product_attention(sample, sample, sample)
            del sample
            result['attention_backend'] = ATTENTION_BACKEND
        result['gpu'] = torch.cuda.get_device_name()
    if missing_features or missing_matches:
        checkpoints = Path(os.environ.get('TORCH_HOME', '.cache/torch')).expanduser().resolve() / 'hub/checkpoints'
        missing = [name for name in ('dad.pth', 'dinov2_vitl14_pretrain.pth', 'loma_B.pt')
                   if not (checkpoints / name).is_file()]
        result['loma_weights_cached'] = not missing
        if missing:
            raise RuntimeError('Missing offline LoMa weights in ' + str(checkpoints) + ': ' + ', '.join(missing))
    cache = Path(model_cache or os.environ.get('HF_HUB_CACHE', '.cache/huggingface')).expanduser().resolve()
    if missing_masks:
        snapshot = cache / ('models--' + MODEL.replace('/', '--')) / 'snapshots' / MODEL_REVISION
        cached = (snapshot / 'config.json').is_file() and (snapshot / 'preprocessor_config.json').is_file() and any(
            (snapshot / name).is_file() for name in ('model.safetensors', 'pytorch_model.bin'))
        result['segmentation_weights_cached'] = cached
        if not cached:
            raise RuntimeError(f'Offline segmentation weights are incomplete in {cache}')
    destination = Path(delivery['output'])
    parent = destination.parent
    while not parent.is_dir():
        parent = parent.parent
    free = shutil.disk_usage(parent).free
    minimum = (sum(Path(r['source']).stat().st_size for r in images)
               if 'export' in stages and delivery['image_mode'] == 'copy'
                   and not (destination / 'colmap_export.json').is_file() else 0)
    initialization_bytes = 0
    if ('export' in stages and delivery['include_ply']
            and not (destination / 'colmap_export.json').is_file()):
        from geosunpre.initialization import AGGREGATE, VERTEX
        initialization_bytes = inputs['las_header']['point_count'] * (AGGREGATE.itemsize + 2 * VERTEX.itemsize)
    required = minimum + initialization_bytes
    if free < required:
        raise RuntimeError(f'Insufficient export disk space: need up to {required} bytes, available {free}')
    result['export_disk'] = {'available_bytes': free, 'minimum_image_copy_bytes': minimum,
                             'initialization_peak_bytes_upper_bound': initialization_bytes,
                             'required_bytes': required}
    from geosunpre.pipeline.task import bind_weights
    bind_weights(work, cache, frontend=missing_features or missing_matches,
                 segmentation=missing_masks)
    return result
