"""Pinned LoMa frontend with per-image identity and configuration-aware caches."""
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from .camera import Camera
from .scene import digest, file_digest

LOMA_COMMIT = '8fb59c458f2a2ef44314f462467b229c27786bf0'
FRONTEND_VERSION = 1


@dataclass(frozen=True)
class FrontendConfig:
    detector_size: int = 1024
    num_keypoints: int = 2048
    oversample: int = 4
    grid_size: int = 8
    match_threshold: float = .1
    loma_commit: str = LOMA_COMMIT
    implementation_version: int = FRONTEND_VERSION


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
        self.cache_stats = {'feature_hits': 0, 'feature_misses': 0, 'match_hits': 0, 'match_misses': 0}
        self.cache = self.work/'features'/digest(asdict(config))[:20]
        self.cache.mkdir(parents=True, exist_ok=True)

    def load_model(self):
        if self.model is None:
            import torch
            os.environ.setdefault('TORCH_HOME', str(Path(__file__).resolve().parents[2]/'.cache/torch'))
            if not torch.cuda.is_available():
                raise RuntimeError('LoMa diagnostic requires CUDA; run with GPU access enabled')
            from loma import LoMa, LoMaB
            from loma.device import device
            self.device = device
            self.model = LoMa(LoMaB(compile=False, num_keypoints=self.config.num_keypoints)).eval()
            self.model._detector.resize = self.config.detector_size
        return self.model

    def extract(self, record):
        cam = Camera.from_dict(self.scene['cameras'][record['camera']])
        stat = Path(record['source']).stat()
        if stat.st_size != record['size_bytes'] or stat.st_mtime_ns != record['mtime_ns']:
            raise ValueError(f'{record["name"]}: image changed since prepare; prepare the scene again')
        source_hash = file_digest(record['source'])
        mask_hash = file_digest(record['semantic_mask']) if record['semantic_mask'] else None
        identity = {'image_sha256': source_hash, 'mask_sha256': mask_hash,
                    'camera': cam.to_dict(), 'fov': self.scene['fov_degrees'], 'config': asdict(self.config)}
        token = digest(identity)
        path = self.cache/f'{token}.npz'
        if path.exists():
            self.cache_stats['feature_hits'] += 1
            with np.load(path, allow_pickle=False) as f:
                return {k: f[k] for k in f.files}
        self.cache_stats['feature_misses'] += 1
        import torch
        model = self.load_model()
        with torch.inference_mode():
            result = model._detector.detect_from_path(
                record['source'], num_keypoints=self.config.num_keypoints*self.config.oversample)
            normalized = result['keypoints'][0].float().cpu().numpy()
            probabilities = result['keypoint_probs'][0].float().cpu().numpy().reshape(-1)
            pixels = (normalized+1)*np.array([cam.width, cam.height], dtype=np.float32)/2
            valid = cam.valid(pixels, self.scene['fov_degrees'])
            if record['semantic_mask']:
                with Image.open(record['semantic_mask']) as im:
                    mask = np.asarray(im.convert('L')) >= 128
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
            descriptors = model._descriptor.describe_keypoints_from_path(
                record['source'], keypoints)['descriptions'][0].float().cpu().numpy()
        meta = {**identity, 'token': token, 'name': record['name'], 'detected': len(pixels),
                'valid_before_quota': len(candidates), 'selected': len(indices),
                'descriptor_resolution': [784, 784], 'pixel_coordinates': self.scene['pixel_coordinates']}
        output = {'keypoints': pixels[indices].astype(np.float32),
                  'normalized_keypoints': normalized[indices].astype(np.float32),
                  'descriptors': descriptors.astype(np.float32), 'scores': probabilities[indices],
                  'metadata': np.array(json.dumps(meta, ensure_ascii=False)), 'token': np.array(token)}
        tmp = path.with_suffix('.tmp.npz')
        np.savez_compressed(tmp, **output)
        tmp.replace(path)
        return output

    def match(self, features_a, features_b):
        token = digest({'a': str(features_a['token']), 'b': str(features_b['token']), 'config': asdict(self.config)})
        path = self.work/'matches'/f'{token}.npz'
        if path.exists():
            self.cache_stats['match_hits'] += 1
            with np.load(path, allow_pickle=False) as f:
                return {k: f[k] for k in f.files}
        self.cache_stats['match_misses'] += 1
        import torch
        from loma.loma import filter_matches
        model = self.load_model()
        with torch.inference_mode():
            a = torch.from_numpy(features_a['normalized_keypoints'])[None].to(self.device)
            b = torch.from_numpy(features_b['normalized_keypoints'])[None].to(self.device)
            da = torch.from_numpy(features_a['descriptors'])[None].to(self.device)
            db = torch.from_numpy(features_b['descriptors'])[None].to(self.device)
            m0, _, scores, _ = filter_matches(model(a, b, da, db)['scores'], self.config.match_threshold)
            ia = torch.where(m0[0] >= 0)[0]
            indices = torch.stack((ia, m0[0, ia]), dim=-1).cpu().numpy().astype(np.uint32)
            confidence = scores[0, ia].float().cpu().numpy()
        output = {'matches': indices, 'scores': confidence, 'token': np.array(token)}
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix('.tmp.npz')
        np.savez_compressed(tmp, **output)
        tmp.replace(path)
        return output
