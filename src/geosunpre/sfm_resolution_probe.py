"""Bounded LoMa detector-resolution probe on four same-camera image pairs.

The 2048-pixel features have separate cache identities. This measures pair
geometry and original-image flow repeatability; it never replaces global SfM.
"""

from datetime import datetime, timezone
import json
from pathlib import Path

import cv2
import numpy as np
import pycolmap

from .camera import Camera
from .frontend import FrontendConfig, LoMaFrontend
from .geometry import fit_pair
from .scene import digest, file_digest, load_scene, write_json
from .sfm_lk_probe import _flow, _frame


OUTPUT = 'sfm_resolution_probe'
PAIR_FRAMES = (('R', 65), ('R', 70), ('L', 120), ('L', 150))
HIGH_RES = FrontendConfig(detector_size=2048, num_keypoints=3072)
MATCH_CONFIDENCE = .9


def _pick_pairs(frontend):
    picked = []
    for camera, frame in PAIR_FRAMES:
        found = [entry for entry in frontend['pairs']
                 if entry['a'][0] == camera and entry['b'][0] == camera
                 and _frame(entry['a']) == frame and _frame(entry['b']) == frame + 1]
        if len(found) != 1:
            raise ValueError(f'Expected one adjacent pair for {camera} frame {frame}')
        picked.append(found[0])
    return picked


def _cached_feature(entry):
    with np.load(entry['path'], allow_pickle=False) as raw:
        if str(raw['token']) != entry['token']:
            raise ValueError(f'Changed LoMa feature cache: {entry["path"]}')
        return {key: raw[key] for key in raw.files}


def _cached_match(entry):
    with np.load(entry['path'], allow_pickle=False) as raw:
        if str(raw['token']) != entry['token']:
            raise ValueError(f'Changed LoMa match cache: {entry["path"]}')
        return {key: raw[key] for key in raw.files}


def _lk_summary(gray_a, gray_b, xy_a, xy_b, take):
    xy_a, xy_b = xy_a[take], xy_b[take]
    if not len(xy_a):
        return {'matches': 0, 'repeatable': 0, 'disagreement_median_px': None,
                'disagreement_p90_px': None, 'fraction_gt_2px': None}
    a = (xy_a / 2 - .5).astype(np.float32).reshape(-1, 1, 2)
    b = (xy_b / 2 - .5).astype(np.float32).reshape(-1, 1, 2)
    moved_b, ok_ab, _ = _flow(gray_a, gray_b, a)
    returned_a, ok_ba, _ = _flow(gray_b, gray_a, moved_b)
    moved_a, ok_ba_direct, _ = _flow(gray_b, gray_a, b)
    returned_b, ok_ab_direct, _ = _flow(gray_a, gray_b, moved_a)
    fb_a = np.linalg.norm((returned_a - a).reshape(-1, 2), axis=1) * 2
    fb_b = np.linalg.norm((returned_b - b).reshape(-1, 2), axis=1) * 2
    repeatable = (ok_ab.ravel().astype(bool) & ok_ba.ravel().astype(bool)
                  & ok_ba_direct.ravel().astype(bool) & ok_ab_direct.ravel().astype(bool)
                  & (fb_a < 1) & (fb_b < 1))
    delta_a = np.linalg.norm((moved_a - a).reshape(-1, 2), axis=1) * 2
    delta_b = np.linalg.norm((moved_b - b).reshape(-1, 2), axis=1) * 2
    disagreement = np.maximum(delta_a, delta_b)[repeatable]
    return {'matches': len(xy_a), 'repeatable': int(repeatable.sum()),
            'disagreement_median_px': float(np.median(disagreement)) if len(disagreement) else None,
            'disagreement_p90_px': float(np.quantile(disagreement, .9)) if len(disagreement) else None,
            'fraction_gt_2px': float(np.mean(disagreement > 2)) if len(disagreement) else None}


def _analyze_pair(entry, features, match, cameras, gray):
    a, b = entry['a'], entry['b']
    matches, scores = match['matches'], match['scores']
    xy_a = features[a]['keypoints'][matches[:, 0]].astype(float)
    xy_b = features[b]['keypoints'][matches[:, 1]].astype(float)
    camera = cameras[a[0]]
    metrics, fit = fit_pair(camera, camera, xy_a, xy_b, max_error=3, seed=271828)
    high = scores >= MATCH_CONFIDENCE
    verified = fit['verified_mask']
    rows = {}
    for name, take in (('all_high_confidence', high),
                       ('geometry_verified_high_confidence', high & verified)):
        selected = xy_a[take]
        cells = np.floor(selected / [camera.width, camera.height] * 8).astype(int)
        rows[name] = {'matches': int(take.sum()),
                      'occupied_8x8_cells': len({tuple(cell) for cell in cells}),
                      'bottom_half_fraction': float(np.mean(selected[:, 1] >= camera.height / 2))
                      if len(selected) else None,
                      'flow': _lk_summary(gray[a], gray[b], xy_a, xy_b, take)}
    return {'raw_matches': len(matches), 'fitted_pair_success': metrics['success'],
            'fit_train_inliers': metrics.get('ransac_train_inliers'),
            'fit_holdout_epipolar_error_px_all_raw': metrics.get('holdout_epipolar_error_px_all_raw'),
            'fit_verified_matches': int(verified.sum()), 'groups': rows}


def run_resolution_probe(work):
    cv2.setNumThreads(4)
    work = Path(work).resolve()
    root = work / 'global_sfm'
    scene = load_scene(root)
    records = {record['name']: record for record in scene['images']}
    baseline = json.loads((root / 'frontend.json').read_text())
    chosen = _pick_pairs(baseline)
    names = sorted({entry[key] for entry in chosen for key in ('a', 'b')})
    source = pycolmap.Reconstruction(root / 'ground_joint_ba/equal_control/sparse/0')
    cameras = {label: Camera(label, source.cameras[1 if label == 'L' else 2].width,
                             source.cameras[1 if label == 'L' else 2].height,
                             tuple(source.cameras[1 if label == 'L' else 2].params), (0, 0))
               for label in ('L', 'R')}
    protocol = {'version': 1, 'pairs': [(entry['a'], entry['b']) for entry in chosen],
                'image_sha256': {name: file_digest(records[name]['source']) for name in names},
                'mask_sha256': {name: file_digest(records[name]['semantic_mask']) for name in names},
                'baseline_frontend_sha256': file_digest(root / 'frontend.json'),
                'camera_model_hashes': {p.name: file_digest(p) for p in sorted(
                    (root / 'ground_joint_ba/equal_control/sparse/0').glob('*.bin'))},
                'candidate_config': HIGH_RES.__dict__,
                'high_match_confidence': MATCH_CONFIDENCE,
                'pair_geometry': 'Same image-only camera-aware essential fit, 3 native-pixel threshold',
                'lk': 'Half-resolution native JPEG, 21x21 window, 3 pyramid levels, both directions forward/backward <1 original px',
                'uses_navigation_las_or_manual_labels': False,
                'scope': 'Four exploratory adjacent same-camera pairs; not a registered SfM or 3DGS quality comparison.'}
    fingerprint = digest(protocol)
    output = root / OUTPUT
    output.mkdir(exist_ok=True)
    protocol_path = output / 'protocol.json'
    if protocol_path.exists():
        if json.loads(protocol_path.read_text())['fingerprint'] != fingerprint:
            raise ValueError('Existing detector-resolution probe belongs to changed inputs')
    else:
        write_json(protocol_path, {**protocol, 'fingerprint': fingerprint,
                                   'frozen_at_utc': datetime.now(timezone.utc).isoformat()})

    old_features = {name: _cached_feature(baseline['features'][name]) for name in names}
    new_frontend = LoMaFrontend(work, scene, HIGH_RES)
    new_features = {}
    for index, name in enumerate(names, 1):
        new_features[name] = new_frontend.extract(records[name])
        print(f'High-resolution LoMa features {index}/{len(names)} {name}', flush=True)
    gray = {}
    for name in names:
        original = cv2.imread(records[name]['source'], cv2.IMREAD_GRAYSCALE)
        if original is None or original.shape != (4000, 3000):
            raise ValueError(f'Missing native JPEG: {name}')
        gray[name] = cv2.resize(original, (1500, 2000), interpolation=cv2.INTER_AREA)
    rows = []
    for index, entry in enumerate(chosen, 1):
        old_match = _cached_match(entry)
        new_match = new_frontend.match(new_features[entry['a']], new_features[entry['b']])
        row = {'camera': entry['a'][0], 'frame': _frame(entry['a']),
               'a': entry['a'], 'b': entry['b'],
               'baseline': _analyze_pair(entry, old_features, old_match, cameras, gray),
               'detector_2048': _analyze_pair(entry, new_features, new_match, cameras, gray),
               'new_feature_tokens': [str(new_features[entry[key]]['token']) for key in ('a', 'b')],
               'new_match_token': str(new_match['token'])}
        rows.append(row)
        print(f'High-resolution LoMa pair comparison {index}/{len(chosen)}', flush=True)
    result = {'protocol_fingerprint': fingerprint, 'rows': rows,
              'cache_stats': new_frontend.cache_stats,
              'guard': 'Different detector resolutions choose different keypoints. Flow success and pairwise E fit are proxies, not identical-point accuracy or downstream SfM. No production cache, model, manual holdout, or LFS data was changed.'}
    write_json(output / 'results.json', result)
    return result
