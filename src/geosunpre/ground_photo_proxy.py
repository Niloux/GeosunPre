"""Held-out local planar appearance alignment on native fisheye JPEGs."""
from collections import OrderedDict
from datetime import datetime, timezone
import json
from pathlib import Path

import cv2
import numpy as np
import pycolmap

from .camera import Camera
from .ground_joint_ba import OUTPUT as BA_OUTPUT
from .ground_pose_probe import SOURCE, fit_floor_from_training
from .local_evaluation import triangulate_support
from .local_reconstruction import assert_holdout_isolated
from .scene import digest, file_digest, load_scene, write_json
from .static_validation import checked_annotations


OUTPUT = 'ground_photo_proxy'
STAGES = ('equal_control', 'near_floor_2p0')
PATCH_RADIUS_PX = 20
PATCH_STEP_PX = 2
MIN_COMMON_SAMPLES = 250
MIN_SOURCE_STD = 5.
MAX_CACHED_IMAGES = 8


def _grid(center):
    values = np.arange(-PATCH_RADIUS_PX, PATCH_RADIUS_PX + 1, PATCH_STEP_PX)
    xx, yy = np.meshgrid(values, values)
    return np.column_stack((center[0] + xx.ravel(), center[1] + yy.ravel()))


def _sample(image, coords):
    """COLMAP corner coordinates to OpenCV array-center coordinates."""
    x = np.asarray(coords[:, 0] - .5, dtype=np.float32).reshape(1, -1)
    y = np.asarray(coords[:, 1] - .5, dtype=np.float32).reshape(1, -1)
    return cv2.remap(image, x, y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT).reshape(-1)


def _warp_ground_patch(model, names, support_index, support_coords, grid, normal):
    images = {im.name: im for im in model.images.values() if im.has_pose}
    ims = [images[name] for name in names]
    cameras = [Camera(str(im.camera_id), model.cameras[im.camera_id].width,
                      model.cameras[im.camera_id].height,
                      tuple(model.cameras[im.camera_id].params), (0, 0)) for im in ims]
    poses = [im.cam_from_world().matrix() for im in ims]
    center, support_error, angle = triangulate_support(cameras[:2], poses[:2],
                                                       support_coords[:2])
    if center is None or support_error > 3 or angle < 1:
        return None
    image = ims[support_index]
    camera = cameras[support_index]
    pose = poses[support_index]
    origin = image.projection_center()
    rays = camera.rays(grid) @ pose[:, :3]
    denom = rays @ normal
    distance = float((center - origin) @ normal)
    positive = np.abs(denom) > 1e-9
    steps = np.full(len(grid), np.nan)
    steps[positive] = distance / denom[positive]
    positive &= steps > 0
    world = origin[None] + steps[:, None] * rays
    target = poses[2]
    local = world @ target[:, :3].T + target[:, 3]
    positive &= local[:, 2] > 1e-8
    safe_local = np.where(np.isfinite(local), local, np.array([0., 0., 1.]))
    projected = cameras[2].project(safe_local)
    valid = positive & camera.valid(grid) & cameras[2].valid(projected)
    valid &= ((grid[:, 0] >= 1.5) & (grid[:, 0] < camera.width - .5)
              & (grid[:, 1] >= 1.5) & (grid[:, 1] < camera.height - .5)
              & (projected[:, 0] >= 1.5) & (projected[:, 0] < cameras[2].width - .5)
              & (projected[:, 1] >= 1.5) & (projected[:, 1] < cameras[2].height - .5))
    return projected, valid, {'support_max_error_px': support_error,
                              'support_angle_deg': angle}


def _zncc(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a -= a.mean(); b -= b.mean()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denominator) if denominator > 1e-12 else None


def _metric_summary(rows):
    scores = [r['zncc'] for r in rows if r['zncc'] is not None]
    return {'pairs': len(rows), 'scored': len(scores),
            'median_zncc': float(np.median(scores)) if scores else None,
            'p10_zncc': float(np.quantile(scores, .1)) if scores else None,
            'mean_zncc': float(np.mean(scores)) if scores else None}


def run_ground_photo_proxy(work):
    cv2.setNumThreads(1)
    root = Path(work).resolve() / 'global_sfm'
    output = root / OUTPUT
    output.mkdir(exist_ok=True)
    scene = load_scene(root)
    labels_path = root / 'static_validation/basketball-static-annotations.json'
    labels = json.loads(labels_path.read_text())
    candidates = json.loads((root / 'static_validation/candidates.json').read_text())
    accepted, _ = checked_annotations(labels, candidates, scene)
    classes = {row['id']: row['class'] for row in json.loads(
        (root / 'ground_pose_probe/target_analysis.json').read_text())['manual_proxy_details']}
    near_rows = [(row, coords) for row, _, coords in accepted
                 if classes[row['id']] == 'near_floor']
    source = pycolmap.Reconstruction(root / SOURCE)
    split = json.loads((root / 'split.json').read_text())
    assert_holdout_isolated(source, split)
    _, _, plane = fit_floor_from_training(source)
    normal = np.asarray(plane['floor_normal'])
    joint_protocol = json.loads((root / BA_OUTPUT / 'protocol.json').read_text())
    comparison = json.loads((root / BA_OUTPUT / 'comparison.json').read_text())
    if joint_protocol['fingerprint'] != comparison['protocol_fingerprint']:
        raise ValueError('Joint BA protocol/results mismatch')
    if joint_protocol['official_annotations_sha256'] != file_digest(labels_path):
        raise ValueError('Official annotations changed')
    models = {name: pycolmap.Reconstruction(root / BA_OUTPUT / name / 'sparse/0')
              for name in STAGES}
    model_hashes = {name: {p.name: file_digest(p)
                           for p in sorted((root / BA_OUTPUT / name / 'sparse/0').glob('*.bin'))}
                    for name in STAGES}
    for name in STAGES:
        if model_hashes[name] != comparison['variants'][name]['model_hashes']:
            raise ValueError(f'Joint BA model changed: {name}')
        assert_holdout_isolated(models[name], split)
    records = {record['name']: record for record in scene['images']}
    image_names = sorted({obs['name'] for row, _ in near_rows
                          for obs in row['observations']})
    protocol = {'version': 1, 'scene_fingerprint': scene['fingerprint'],
                'native_jpeg_sha256': {name: file_digest(records[name]['source'])
                                       for name in image_names},
                'official_annotations_sha256': file_digest(labels_path),
                'manual_proxy_sha256': file_digest(root / 'ground_pose_probe/target_analysis.json'),
                'joint_protocol_fingerprint': joint_protocol['fingerprint'],
                'model_hashes': model_hashes,
                'near_floor_manual_ids': sorted(row['id'] for row, _ in near_rows),
                'pair_rule': 'Each manually confirmed near-floor proxy: both support images to the query image. Support pair triangulates a point in each saved model; fixed source-model floor normal defines a local tangent plane through that point. Query coordinate and JPEG appearance do not fit geometry.',
                'patch_radius_px': PATCH_RADIUS_PX, 'patch_step_px': PATCH_STEP_PX,
                'min_common_samples': MIN_COMMON_SAMPLES,
                'min_source_gray_std': MIN_SOURCE_STD,
                'metric': 'Grayscale ZNCC after native-fisheye tangent-plane warping; shared valid sample mask across models. Not linear radiance or 3DGS rendering.'}
    fingerprint = digest(protocol)
    path = output / 'protocol.json'
    if path.exists():
        if json.loads(path.read_text()).get('fingerprint') != fingerprint:
            raise ValueError('Photometric proxy protocol changed; use fresh output')
    else:
        write_json(path, {**protocol, 'fingerprint': fingerprint,
                   'frozen_at_utc': datetime.now(timezone.utc).isoformat()})

    cache = OrderedDict()

    def gray(name):
        if name in cache:
            cache.move_to_end(name)
            return cache[name]
        image = cv2.imread(records[name]['source'], cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f'Image unavailable: {name}')
        if image.shape != (4000, 3000):
            raise ValueError(f'Unexpected native image size: {name}')
        cache[name] = image.astype(np.float32)
        if len(cache) > MAX_CACHED_IMAGES:
            cache.popitem(last=False)
        return cache[name]

    rows = []
    for ordinal, (row, coords) in enumerate(near_rows, 1):
        names = [obs['name'] for obs in row['observations']]
        for support_index in (0, 1):
            grid = _grid(coords[support_index])
            warps = {name: _warp_ground_patch(model, names, support_index,
                                               coords, grid, normal)
                     for name, model in models.items()}
            entry = {'id': row['id'], 'support_index': support_index,
                     'support_name': names[support_index], 'query_name': names[2],
                     'camera_mix': 'cross_LR' if len({records[n]['camera'] for n in names}) > 1
                     else 'single_camera', 'samples_total': len(grid),
                     'samples_common_valid': 0, 'source_std_gray': None,
                     'stages': {}, 'reason': None}
            if any(value is None for value in warps.values()):
                entry['reason'] = 'support_triangulation'
                rows.append(entry)
                continue
            valid = np.logical_and.reduce([value[1] for value in warps.values()])
            entry['samples_common_valid'] = int(valid.sum())
            if valid.sum() < MIN_COMMON_SAMPLES:
                entry['reason'] = 'insufficient_common_valid_pixels'
                rows.append(entry)
                continue
            source_gray = _sample(gray(names[support_index]), grid[valid])
            entry['source_std_gray'] = float(np.std(source_gray))
            if entry['source_std_gray'] < MIN_SOURCE_STD:
                entry['reason'] = 'low_texture_source'
                rows.append(entry)
                continue
            query_gray = gray(names[2])
            for name, (projected, _, support) in warps.items():
                values = _sample(query_gray, projected[valid])
                entry['stages'][name] = {'zncc': _zncc(source_gray.copy(), values),
                                         'target_std_gray': float(np.std(values)),
                                         **support}
            rows.append(entry)
        if ordinal % 10 == 0 or ordinal == len(near_rows):
            print(f'Near-floor photometric proxy {ordinal}/{len(near_rows)}', flush=True)
    stage_rows = {name: [{'zncc': row['stages'][name]['zncc']}
                         for row in rows if name in row['stages']]
                  for name in STAGES}
    pairs = [(row['stages'][STAGES[0]]['zncc'], row['stages'][STAGES[1]]['zncc'])
             for row in rows if all(name in row['stages'] for name in STAGES)
             and all(row['stages'][name]['zncc'] is not None for name in STAGES)]
    deltas = np.asarray([b - a for a, b in pairs])
    track_deltas = {}
    for row in rows:
        if all(name in row['stages'] and row['stages'][name]['zncc'] is not None
               for name in STAGES):
            track_deltas.setdefault(row['id'], []).append(
                row['stages'][STAGES[1]]['zncc'] - row['stages'][STAGES[0]]['zncc'])
    track_means = np.asarray([np.mean(values) for _, values in sorted(track_deltas.items())])
    rng = np.random.default_rng(314159)
    bootstrap = track_means[rng.integers(0, len(track_means),
                                         size=(10000, len(track_means)))]
    by_camera_mix = {}
    for kind in ('single_camera', 'cross_LR'):
        values = [row['stages'][STAGES[1]]['zncc'] - row['stages'][STAGES[0]]['zncc']
                  for row in rows if row['camera_mix'] == kind and
                  all(name in row['stages'] and row['stages'][name]['zncc'] is not None
                      for name in STAGES)]
        by_camera_mix[kind] = {'pairs': len(values),
                               'median_delta_zncc': float(np.median(values)) if values else None,
                               'improved': int(np.sum(np.asarray(values) > 0))}
    result = {'protocol_fingerprint': fingerprint,
              'near_floor_manual_tracks': len(near_rows),
              'support_to_query_pairs': len(rows),
              'reason_counts': {reason: sum(row['reason'] == reason for row in rows)
                                for reason in sorted({row['reason'] for row in rows if row['reason']})},
              'stages': {name: _metric_summary(stage_rows[name]) for name in STAGES},
              'paired': {'count': len(deltas),
                         'median_weighted_minus_control_zncc': float(np.median(deltas)) if len(deltas) else None,
                         'mean_weighted_minus_control_zncc': float(np.mean(deltas)) if len(deltas) else None,
                         'improved': int(np.sum(deltas > 0)),
                         'worsened': int(np.sum(deltas < 0)),
                         'by_camera_mix': by_camera_mix,
                         'track_level': {'tracks': len(track_means),
                                         'improved': int(np.sum(track_means > 0)),
                                         'worsened': int(np.sum(track_means < 0)),
                                         'median_two_support_mean_delta_zncc': float(np.median(track_means)),
                                         'mean_two_support_mean_delta_zncc': float(np.mean(track_means)),
                                         'descriptive_bootstrap_mean_95pct': np.quantile(
                                             bootstrap.mean(axis=1), [.025, .975]).tolist(),
                                         'bootstrap_seed': 314159,
                                         'bootstrap_replicates': 10000}},
              'rows': rows,
              'limitation': 'Local planar and grayscale JPEG alignment proxy only; exposure, shadows, specularity, ground nonplanarity and JPEG processing remain confounders. Pairs from same 30 tracks are correlated and do not substitute for 3DGS render evaluation.'}
    write_json(output / 'results.json', result)
    print('Ground photometric proxy:', result['stages'], result['paired'], flush=True)
    return result
