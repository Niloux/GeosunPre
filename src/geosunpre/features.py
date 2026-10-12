"""Pinned LoMa frontend with per-image identity and configuration-aware caches."""
from dataclasses import asdict, dataclass
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from geosunpre.camera import Camera
from geosunpre.artifacts import file_metadata

LOMA_COMMIT = '8fb59c458f2a2ef44314f462467b229c27786bf0'


@dataclass(frozen=True)
class FrontendConfig:
    detector_size: int = 1536
    num_keypoints: int = 3072
    oversample: int = 4
    grid_size: int = 8
    match_threshold: float = .1
    loma_commit: str = LOMA_COMMIT


FRONTEND_CONFIG = FrontendConfig()


def _feature_cache_root(work):
    return Path(work) / 'features'


def feature_input(work, scene, config, record):
    """Check prepared inputs and derive the shared extraction identity and path."""
    from geosunpre.scene import prepared_image_metadata

    cam = Camera.from_dict(scene['cameras'][record['camera']])
    identity = {'image_file': prepared_image_metadata(record),
                'mask_file': file_metadata(record['semantic_mask']) if record['semantic_mask'] else None,
                'camera': cam.to_dict(), 'fov': scene['fov_degrees'], 'config': asdict(config)}
    identity = json.loads(json.dumps(identity, allow_nan=False))
    token = record['name']
    return cam, identity, token, _feature_cache_root(work) / f'{token}.npz'


def read_feature_cache(path, token, *, name=None, fields=None, identity=None, expected_file=None):
    """Read selected arrays after checking recorded inputs and file metadata."""
    def check_file():
        if expected_file is not None and file_metadata(path) != expected_file:
            raise ValueError(f'Feature cache changed: {name or path}')

    check_file()
    with np.load(path, allow_pickle=False) as cached:
        if str(cached['token']) != token:
            raise ValueError(f'Cached feature identity changed: {name or path}')
        metadata = json.loads(str(cached['metadata']))
        if name is not None and metadata['name'] != name:
            raise ValueError(f'Global feature identity changed: {name}')
        if identity is not None and metadata['inputs'] != identity:
            raise ValueError(f'Feature extraction inputs changed: {name or path}')
        result = {key: cached[key] for key in (cached.files if fields is None else fields)}
    check_file()
    return result


def balanced_selection(pixels, scores, target, width, height, grid_size=8):
    """Reserve a quota for occupied cells, then fill unused slots by confidence."""
    if not len(pixels):
        return np.empty(0, dtype=np.int64)
    cells = np.minimum((pixels / [width, height] * grid_size).astype(int), grid_size-1)
    ids = cells[:, 1]*grid_size+cells[:, 0]
    active = np.unique(ids)
    quota = max(1, target//len(active))
    ranked = np.argsort(-scores, kind='stable')
    used = np.zeros(len(pixels), dtype=bool)
    for cell in active:
        used[ranked[ids[ranked] == cell][:quota]] = True
    chosen = ranked[used[ranked]][:target]
    if len(chosen) < target:
        chosen = np.concatenate((chosen, ranked[~used[ranked]][:target-len(chosen)]))
    return chosen


class LoMaFrontend:
    def __init__(self, work, scene, config):
        self.work = Path(work)
        self.scene = scene
        self.config = config
        self.model = None
        self.device = None
        self._gpu_features = OrderedDict()
        self._gpu_feature_bytes = 0
        cache_mib = int(os.environ.get('GEOSUNPRE_GPU_FEATURE_CACHE_MIB', '2048'))
        if cache_mib < 0:
            raise ValueError('GEOSUNPRE_GPU_FEATURE_CACHE_MIB must be nonnegative')
        self._gpu_feature_limit = cache_mib * 1024 * 1024
        self._matcher = None
        self.cache_stats = {'feature_hits': 0, 'feature_misses': 0, 'match_hits': 0, 'match_misses': 0}
        self.cache_stats.update(gpu_feature_hits=0, gpu_feature_misses=0,
                                gpu_upload_bytes=0, gpu_feature_peak_bytes=0)
        self.cache = _feature_cache_root(self.work)
        self.cache.mkdir(parents=True, exist_ok=True)

    def load_model(self):
        if self.model is None:
            import torch
            os.environ.setdefault('TORCH_HOME', str(Path('.cache/torch').resolve()))
            if not torch.cuda.is_available():
                raise RuntimeError('LoMa feature extraction requires CUDA; run with GPU access enabled')
            from loma import LoMa, LoMaB
            from loma.device import device
            self.device = device
            self._gpu_feature_limit = min(self._gpu_feature_limit,
                                         torch.cuda.get_device_properties(device).total_memory // 8)
            self.cache_stats['gpu_feature_limit_bytes'] = self._gpu_feature_limit
            self.model = LoMa(LoMaB(compile=False, num_keypoints=self.config.num_keypoints)).eval()
            self.model._detector.resize = self.config.detector_size
            from geosunpre.matching import configure_matcher
            configure_matcher(self.model)
        return self.model

    def prepare_image(self, record):
        """Decode once and reproduce the pinned LoMa PIL preprocessing on CPU."""
        import torch
        with Image.open(record['source']) as source:
            rgb = source.convert('RGB')
        width, height = rgb.size
        size = self.config.detector_size
        if self.model._detector.keep_aspect_ratio:
            scale = size / max(width, height)
            detector_size = (int((scale * width) // 8 * 8), int((scale * height) // 8 * 8))
        else:
            detector_size = (size, size)
        def tensor(size):
            return torch.from_numpy(np.array(rgb.resize(size)) / 255.0).permute(2, 0, 1).float()[None]
        return tensor(detector_size), tensor((784, 784))

    def extract_all(self, records):
        """Prepare only cache misses, with at most two images in flight."""
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix='image-prepare') as pool:
            pending = OrderedDict()
            iterator = iter(enumerate(records))
            for index, record in iterator:
                _, _, _, path = feature_input(self.work, self.scene, self.config, record)
                prepared = None
                if not path.is_file():
                    self.load_model()
                    prepared = pool.submit(self.prepare_image, record)
                pending[index] = (record, prepared)
                if len(pending) == 2:
                    _, (first, prepared) = pending.popitem(last=False)
                    yield self.extract(first, prepared=prepared.result() if prepared else None)
            for record, prepared in pending.values():
                yield self.extract(record, prepared=prepared.result() if prepared else None)

    def extract(self, record, *, prepared=None):
        cam, identity, token, path = feature_input(self.work, self.scene, self.config, record)
        if path.exists():
            self.cache_stats['feature_hits'] += 1
            return read_feature_cache(path, token, name=record['name'], identity=identity)
        self.cache_stats['feature_misses'] += 1
        import torch
        model = self.load_model()
        detector_image, descriptor_image = prepared if prepared is not None else self.prepare_image(record)
        with torch.inference_mode():
            result = model._detector.detect(
                detector_image, num_keypoints=self.config.num_keypoints*self.config.oversample)
            normalized = result['keypoints'][0].float().cpu().numpy()
            probabilities = result['keypoint_probs'][0].float().cpu().numpy().reshape(-1)
            pixels = (normalized+1)*np.array([cam.width, cam.height], dtype=np.float32)/2
            valid = cam.valid(pixels, self.scene['fov_degrees'])
            if record['semantic_mask']:
                with Image.open(record['semantic_mask']) as im:
                    mask = np.asarray(im.convert('L')) == 255
                ij = np.floor(pixels).astype(int)
                ij[:, 0] = ij[:, 0].clip(0, cam.width-1)
                ij[:, 1] = ij[:, 1].clip(0, cam.height-1)
                valid &= mask[ij[:, 1], ij[:, 0]]
            candidates = np.flatnonzero(valid)
            selected = balanced_selection(pixels[candidates], probabilities[candidates],
                                          self.config.num_keypoints, cam.width, cam.height, self.config.grid_size)
            indices = candidates[selected]
            if len(indices) < 32:
                raise RuntimeError(f'{record["name"]}: only {len(indices)} valid keypoints')
            keypoints = result['keypoints'][:, torch.as_tensor(indices, device=self.device)]
            descriptors = model._descriptor.describe_keypoints(
                descriptor_image, keypoints)['descriptions'][0].float().cpu().numpy()
        meta = {'inputs': identity, 'token': token, 'name': record['name'], 'detected': len(pixels),
                'valid_before_quota': len(candidates), 'selected': len(indices),
                'descriptor_resolution': [784, 784], 'pixel_coordinates': self.scene['pixel_coordinates']}
        output = {'keypoints': pixels[indices].astype(np.float32),
                  'normalized_keypoints': normalized[indices].astype(np.float32),
                  'descriptors': descriptors.astype(np.float32), 'scores': probabilities[indices],
                  'metadata': np.array(json.dumps(meta, ensure_ascii=False)), 'token': np.array(token)}
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix('.tmp.npz')
        np.savez(tmp, **output)
        tmp.replace(path)
        return output

    def _matching_tensors(self, features, feature_loader=None):
        import torch
        token = str(features['token'])
        if token in self._gpu_features:
            self.cache_stats['gpu_feature_hits'] += 1
            self._gpu_features.move_to_end(token)
            return self._gpu_features[token]
        self.cache_stats['gpu_feature_misses'] += 1
        if feature_loader is not None:
            features = feature_loader(token)
        value = tuple(torch.from_numpy(features[key])[None].to(self.device)
                      for key in ('normalized_keypoints', 'descriptors'))
        size = sum(t.numel() * t.element_size() for t in value)
        self.cache_stats['gpu_upload_bytes'] += size
        while self._gpu_features and self._gpu_feature_bytes + size > self._gpu_feature_limit:
            _, old = self._gpu_features.popitem(last=False)
            self._gpu_feature_bytes -= sum(t.numel() * t.element_size() for t in old)
        if size <= self._gpu_feature_limit:
            self._gpu_features[token] = value
            self._gpu_feature_bytes += size
            self.cache_stats['gpu_feature_peak_bytes'] = max(
                self.cache_stats['gpu_feature_peak_bytes'], self._gpu_feature_bytes)
        return value

    def match(self, features_a, features_b, *, feature_loader=None):
        a_name, b_name = str(features_a['token']), str(features_b['token'])
        identity = {'a': a_name, 'b': b_name, 'config': asdict(self.config),
                    'a_file': file_metadata(self.cache / f'{a_name}.npz'),
                    'b_file': file_metadata(self.cache / f'{b_name}.npz'),
                    'attention_backend': 'flash'}
        token = f'{Path(a_name).name}__{Path(b_name).name}'
        path = self.work/'matches'/f'{token}.npz'
        if path.exists():
            self.cache_stats['match_hits'] += 1
            with np.load(path, allow_pickle=False) as f:
                if str(f['token']) != token or json.loads(str(f['identity'])) != identity:
                    raise ValueError(f'Match inputs changed: {a_name}, {b_name}')
                return {k: f[k] for k in f.files}
        self.cache_stats['match_misses'] += 1
        import torch
        model = self.load_model()
        if self._matcher is None:
            from geosunpre.matching import MatcherGraph
            self._matcher = MatcherGraph(model, self.config.match_threshold)
        with torch.inference_mode():
            a, da = self._matching_tensors(features_a, feature_loader)
            b, db = self._matching_tensors(features_b, feature_loader)
            m0, scores = self._matcher(a, b, da, db)
            ia = torch.where(m0[0] >= 0)[0]
            indices = torch.stack((ia, m0[0, ia]), dim=-1).cpu().numpy().astype(np.uint32)
            confidence = scores[0, ia].float().cpu().numpy()
        output = {'matches': indices, 'scores': confidence, 'token': np.array(token),
                  'identity': np.array(json.dumps(identity, ensure_ascii=False))}
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix('.tmp.npz')
        np.savez_compressed(tmp, **output)
        tmp.replace(path)
        return output
