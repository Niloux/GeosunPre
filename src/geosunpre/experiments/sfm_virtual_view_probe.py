"""Bounded LoMa front-end probe on virtual pinhole views of native fisheye JPEGs.

Virtual views only propose image features. Their detections are mapped back to
the original OPENCV_FISHEYE image, then matched and verified in native pixels.
No reconstruction or production feature cache is modified.
"""

from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path

import cv2
import numpy as np
import pycolmap
from PIL import Image
from scipy.spatial import cKDTree

from ..camera import Camera
from ..frontend import FrontendConfig, LoMaFrontend, balanced_selection
from ..geometry import epipolar_pixel_error, skew, stats
from ..scene import digest, file_digest, load_scene, write_json
from .sfm_resolution_probe import PAIR_FRAMES, _analyze_pair, _cached_feature, _cached_match, _pick_pairs
from ..static_masks import virtual_views


OUTPUT = 'sfm_virtual_view_probe'
VIEW_SIZE = 1024
VIEW_FOV_DEG = 100
VIEW_KEYPOINTS = 1536
NATIVE_KEYPOINTS = 3072
DEDUP_RADIUS_ORIGINAL_PX = 3.
CONFIG = FrontendConfig(detector_size=VIEW_SIZE, num_keypoints=NATIVE_KEYPOINTS)


def _virtual_token(record, protocol_fingerprint):
    return digest({'protocol': protocol_fingerprint,
                   'name': record['name'],
                   'source_sha256': file_digest(record['source']),
                   'mask_sha256': file_digest(record['semantic_mask'])})


def _deduplicate_native(pixels, scores, radius):
    """Keep the highest-score representative from nearby overlapping views."""
    cell_size = radius
    order = np.argsort(-scores, kind='stable')
    accepted = []
    cells = defaultdict(list)
    for index in order:
        xy = pixels[index]
        cell = tuple(np.floor(xy / cell_size).astype(int))
        if any(np.linalg.norm(xy - pixels[other]) < radius
               for dx in (-1, 0, 1) for dy in (-1, 0, 1)
               for other in cells[(cell[0] + dx, cell[1] + dy)]):
            continue
        accepted.append(int(index))
        cells[cell].append(int(index))
    return np.asarray(accepted, dtype=int)


def _virtual_feature(out, record, camera, scene, frontend, protocol_fingerprint):
    token = _virtual_token(record, protocol_fingerprint)
    cache = out / 'features' / f'{token}.npz'
    if cache.exists():
        frontend.cache_stats['feature_hits'] += 1
        with np.load(cache, allow_pickle=False) as data:
            if str(data['token']) != token:
                raise ValueError(f'Changed virtual-feature cache: {record["name"]}')
            return {key: data[key] for key in data.files}

    frontend.cache_stats['feature_misses'] += 1
    model = frontend.load_model()
    import torch
    with Image.open(record['source']) as image:
        original = np.asarray(image.convert('RGB'))
    with Image.open(record['semantic_mask']) as image:
        valid_mask = np.asarray(image.convert('L')) >= 128
    all_pixels, all_descriptors, all_scores, by_view = [], [], [], []
    view_dir = out / 'views' / record['name']
    view_dir.mkdir(parents=True, exist_ok=True)
    for index, (rotation, focal, source) in enumerate(
            virtual_views(camera, size=VIEW_SIZE, fov=VIEW_FOV_DEG)):
        rgb = cv2.remap(original, source[..., 0], source[..., 1], cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT)
        path = view_dir / f'view_{index}.png'
        Image.fromarray(rgb).save(path)
        with torch.inference_mode():
            detected = model._detector.detect_from_path(path, num_keypoints=VIEW_KEYPOINTS)
            normalized = detected['keypoints'][0].float().cpu().numpy()
            score = detected['keypoint_probs'][0].float().cpu().numpy().reshape(-1)
            virtual_xy = (normalized + 1) * VIEW_SIZE / 2
            rays = np.column_stack(((virtual_xy[:, 0] - VIEW_SIZE / 2) / focal,
                                    (virtual_xy[:, 1] - VIEW_SIZE / 2) / focal,
                                    np.ones(len(virtual_xy)))) @ rotation.T
            native_xy = camera.project(rays)
            finite = np.isfinite(native_xy).all(axis=1)
            check_xy = np.where(finite[:, None], native_xy, 0)
            inside = (finite & camera.valid(check_xy, scene['fov_degrees'])
                      & (virtual_xy[:, 0] >= 8) & (virtual_xy[:, 0] < VIEW_SIZE - 8)
                      & (virtual_xy[:, 1] >= 8) & (virtual_xy[:, 1] < VIEW_SIZE - 8))
            ij = np.floor(check_xy).astype(int)
            ij[:, 0] = np.clip(ij[:, 0], 0, camera.width - 1)
            ij[:, 1] = np.clip(ij[:, 1], 0, camera.height - 1)
            inside &= valid_mask[ij[:, 1], ij[:, 0]]
            indices = np.flatnonzero(inside)
            if not len(indices):
                continue
            selected = detected['keypoints'][:, torch.as_tensor(indices, device=frontend.device)]
            descriptors = model._descriptor.describe_keypoints_from_path(
                path, selected)['descriptions'][0].float().cpu().numpy()
        all_pixels.append(native_xy[indices].astype(np.float32))
        all_descriptors.append(descriptors.astype(np.float32))
        all_scores.append(score[indices].astype(np.float32))
        by_view.append({'view_index': index, 'detected': len(normalized),
                        'native_valid': len(indices), 'source_sha256': file_digest(path)})
    if not all_pixels:
        raise RuntimeError(f'No valid virtual features: {record["name"]}')
    pixels = np.concatenate(all_pixels)
    descriptors = np.concatenate(all_descriptors)
    scores = np.concatenate(all_scores)
    dedup = _deduplicate_native(pixels, scores, DEDUP_RADIUS_ORIGINAL_PX)
    chosen = dedup[balanced_selection(pixels[dedup], scores[dedup],
                                      NATIVE_KEYPOINTS, camera.width, camera.height)]
    native = pixels[chosen]
    result = {'keypoints': native,
              'normalized_keypoints': (native / [camera.width, camera.height] * 2 - 1).astype(np.float32),
              'descriptors': descriptors[chosen], 'scores': scores[chosen],
              'view_audit': np.array(json.dumps(by_view, ensure_ascii=False)),
              'pre_dedup': np.array(len(pixels)),
              'after_dedup': np.array(len(dedup)),
              'token': np.array(token)}
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix('.tmp.npz')
    np.savez_compressed(tmp, **result)
    tmp.replace(cache)
    return result


def run_virtual_view_probe(work):
    cv2.setNumThreads(4)
    work = Path(work).resolve()
    root = work / 'global_sfm'
    scene = load_scene(root)
    records = {record['name']: record for record in scene['images']}
    baseline = json.loads((root / 'frontend.json').read_text())
    chosen_pairs = _pick_pairs(baseline)
    if tuple((entry['a'][0], int(records[entry['a']]['frame_index']))
             for entry in chosen_pairs) != PAIR_FRAMES:
        raise ValueError('Virtual-view pair selection drifted')
    names = sorted({entry[key] for entry in chosen_pairs for key in ('a', 'b')})
    source_model = root / 'ground_joint_ba/equal_control/sparse/0'
    reconstruction = pycolmap.Reconstruction(source_model)
    cameras = {label: Camera(label,
                             reconstruction.cameras[1 if label == 'L' else 2].width,
                             reconstruction.cameras[1 if label == 'L' else 2].height,
                             tuple(reconstruction.cameras[1 if label == 'L' else 2].params), (0, 0))
               for label in ('L', 'R')}
    protocol = {'version': 1,
                'pairs': [(entry['a'], entry['b']) for entry in chosen_pairs],
                'native_image_sha256': {name: file_digest(records[name]['source']) for name in names},
                'native_mask_sha256': {name: file_digest(records[name]['semantic_mask']) for name in names},
                'baseline_frontend_sha256': file_digest(root / 'frontend.json'),
                'camera_model_hashes': {p.name: file_digest(p) for p in sorted(source_model.glob('*.bin'))},
                'virtual_view_size': VIEW_SIZE, 'virtual_view_fov_degrees': VIEW_FOV_DEG,
                'virtual_orientations': ['front', 'yaw_plus_45', 'yaw_minus_45',
                                         'pitch_plus_45', 'pitch_minus_45'],
                'virtual_keypoints_per_view': VIEW_KEYPOINTS,
                'native_keypoints_target': NATIVE_KEYPOINTS,
                'native_dedup_radius_px': DEDUP_RADIUS_ORIGINAL_PX,
                'loma_config': CONFIG.__dict__,
                'uses_navigation_las_or_manual_labels': False,
                'calibration_role': 'Current visual candidate only maps virtual detections back to native rays; not assumed truth.',
                'scope': 'Four exploratory adjacent same-camera pairs. No production SfM/LFS replacement.'}
    fingerprint = digest(protocol)
    output = root / OUTPUT
    output.mkdir(exist_ok=True)
    protocol_path = output / 'protocol.json'
    if protocol_path.exists():
        if json.loads(protocol_path.read_text())['fingerprint'] != fingerprint:
            raise ValueError('Existing virtual-view probe belongs to changed inputs')
    else:
        write_json(protocol_path, {**protocol, 'fingerprint': fingerprint,
                                   'frozen_at_utc': datetime.now(timezone.utc).isoformat()})
    frontend = LoMaFrontend(work, scene, CONFIG)
    new_features = {}
    for index, name in enumerate(names, 1):
        new_features[name] = _virtual_feature(output, records[name], cameras[name[0]],
                                              scene, frontend, fingerprint)
        print(f'Virtual-view LoMa features {index}/{len(names)} {name}: '
              f'{len(new_features[name]["keypoints"])}', flush=True)
    old_features = {name: _cached_feature(baseline['features'][name]) for name in names}
    gray = {}
    for name in names:
        with Image.open(records[name]['source']) as image:
            original = np.asarray(image.convert('L'))
        gray[name] = cv2.resize(original, (1500, 2000), interpolation=cv2.INTER_AREA)
    rows = []
    for index, entry in enumerate(chosen_pairs, 1):
        old_match = _cached_match(entry)
        new_match = frontend.match(new_features[entry['a']], new_features[entry['b']])
        row = {'camera': entry['a'][0], 'frame': int(records[entry['a']]['frame_index']),
               'a': entry['a'], 'b': entry['b'],
               'baseline': _analyze_pair(entry, old_features, old_match, cameras, gray),
               'virtual_views': _analyze_pair(entry, new_features, new_match, cameras, gray),
               'virtual_feature_counts': [len(new_features[entry[key]]['keypoints'])
                                          for key in ('a', 'b')],
               'virtual_match_token': str(new_match['token'])}
        rows.append(row)
        print(f'Virtual-view LoMa pair comparison {index}/{len(chosen_pairs)}', flush=True)
    result = {'protocol_fingerprint': fingerprint, 'rows': rows,
              'cache_stats': frontend.cache_stats,
              'guard': 'Virtual views are temporary front-end sampling only. Different features are not pointwise comparable; E and LK are proxies. Original OPENCV_FISHEYE coordinates and independent exposure poses remain the reconstruction domain.'}
    write_json(output / 'results.json', result)
    return result


def _fixed_pose_match_summary(a, b, features, match, images_by_name, camera):
    pose_a = images_by_name[a].cam_from_world().matrix()
    pose_b = images_by_name[b].cam_from_world().matrix()
    relative_rotation = pose_b[:, :3] @ pose_a[:, :3].T
    relative_translation = pose_b[:, 3] - relative_rotation @ pose_a[:, 3]
    essential = skew(relative_translation) @ relative_rotation
    pairs = match['matches']
    xy_a = features[a]['keypoints'][pairs[:, 0]].astype(float)
    xy_b = features[b]['keypoints'][pairs[:, 1]].astype(float)
    error = epipolar_pixel_error(camera, camera, xy_a, xy_b, essential)
    high = match['scores'] >= .9
    groups = {}
    for label, take in [('all', high),
                        ('lower_native_half', high & (xy_a[:, 1] >= camera.height / 2)),
                        ('upper_native_half', high & (xy_a[:, 1] < camera.height / 2))]:
        values = error[take]
        groups[label] = {'matches': int(take.sum()), 'fixed_pose_epipolar_px': stats(values),
                         'fraction_le_2px': float(np.mean(values <= 2)) if len(values) else None,
                         'fraction_le_3px': float(np.mean(values <= 3)) if len(values) else None}
    return groups


def run_virtual_view_audit(work):
    """Compare both frontends against one frozen visual model and feature layout."""
    work = Path(work).resolve()
    root = work / 'global_sfm'
    output = root / OUTPUT
    probe = json.loads((output / 'results.json').read_text())
    protocol = json.loads((output / 'protocol.json').read_text())
    if probe['protocol_fingerprint'] != protocol['fingerprint']:
        raise ValueError('Virtual-view result/protocol mismatch')
    scene = load_scene(root)
    records = {record['name']: record for record in scene['images']}
    baseline = json.loads((root / 'frontend.json').read_text())
    if file_digest(root / 'frontend.json') != protocol['baseline_frontend_sha256']:
        raise ValueError('Baseline frontend changed since virtual-view probe')
    source_model = root / 'ground_joint_ba/equal_control/sparse/0'
    if {p.name: file_digest(p) for p in sorted(source_model.glob('*.bin'))} != protocol['camera_model_hashes']:
        raise ValueError('Fixed visual model changed since virtual-view probe')
    reconstruction = pycolmap.Reconstruction(source_model)
    images_by_name = {image.name: image for image in reconstruction.images.values()}
    cameras = {label: Camera(label,
                             reconstruction.cameras[1 if label == 'L' else 2].width,
                             reconstruction.cameras[1 if label == 'L' else 2].height,
                             tuple(reconstruction.cameras[1 if label == 'L' else 2].params), (0, 0))
               for label in ('L', 'R')}
    names = sorted({row[key] for row in probe['rows'] for key in ('a', 'b')})
    for name in names:
        if (file_digest(records[name]['source']) != protocol['native_image_sha256'][name]
                or file_digest(records[name]['semantic_mask']) != protocol['native_mask_sha256'][name]):
            raise ValueError(f'Virtual-view input changed since pair probe: {name}')
    old = {name: _cached_feature(baseline['features'][name]) for name in names}
    virtual = {}
    layout = {}
    for name in names:
        token = _virtual_token(records[name], protocol['fingerprint'])
        with np.load(output / 'features' / f'{token}.npz', allow_pickle=False) as data:
            if str(data['token']) != token:
                raise ValueError(f'Changed virtual feature cache: {name}')
            virtual[name] = {key: data[key] for key in data.files}
        camera = cameras[name[0]]
        old_xy = old[name]['keypoints']
        virtual_xy = virtual[name]['keypoints']
        distance = cKDTree(old_xy).query(virtual_xy, k=1)[0]
        layout[name] = {'baseline_bottom_half': float(np.mean(old_xy[:, 1] >= camera.height / 2)),
                        'virtual_bottom_half': float(np.mean(virtual_xy[:, 1] >= camera.height / 2)),
                        'virtual_farther_than_3px_from_baseline': int(np.sum(distance > 3)),
                        'virtual_farther_than_6px_from_baseline': int(np.sum(distance > 6)),
                        'virtual_pre_dedup': int(virtual[name]['pre_dedup']),
                        'virtual_after_dedup': int(virtual[name]['after_dedup'])}
    rows = []
    for row in probe['rows']:
        a, b = row['a'], row['b']
        entry = next(e for e in baseline['pairs'] if e['a'] == a and e['b'] == b)
        old_match = _cached_match(entry)
        with np.load(work / 'matches' / f'{row["virtual_match_token"]}.npz', allow_pickle=False) as data:
            if str(data['token']) != row['virtual_match_token']:
                raise ValueError(f'Changed virtual match cache: {a}, {b}')
            virtual_match = {key: data[key] for key in data.files}
        rows.append({'camera': row['camera'], 'frame': row['frame'], 'a': a, 'b': b,
                     'baseline': _fixed_pose_match_summary(a, b, old, old_match,
                                                           images_by_name, cameras[row['camera']]),
                     'virtual_views': _fixed_pose_match_summary(a, b, virtual, virtual_match,
                                                                images_by_name, cameras[row['camera']])})
    result = {'probe_protocol_fingerprint': protocol['fingerprint'],
              'fixed_visual_model': str(source_model),
              'fixed_visual_model_hashes': protocol['camera_model_hashes'],
              'feature_layout': layout, 'rows': rows,
              'interpretation_guard': 'The fixed camera poses were fitted from the original frontend and therefore favor it. Same model does not make differently sampled matches identical or ground truth. Lower image half is only a screen-coordinate proxy, not a measured ground mask.'}
    write_json(output / 'fixed_model_audit.json', result)
    return result
