"""Freeze image-reviewed near/far LoMa cycles before inspecting visual poses."""

from datetime import datetime, timezone
import itertools
import json
from pathlib import Path

import cv2
import numpy as np
import pycolmap

from .camera import Camera
from .scene import digest, file_digest, load_scene, write_json
from .sfm_court_anchor_probe import _triangulate
from .sfm_court_native_anchor_probe import _project, _world_ray


OUTPUT = 'sfm_r_static_depth_probe'
TRIPLETS = ((277, 278, 279), (279, 280, 281))
R277_OUTPUT = 'sfm_r277_static_probe'
R277_TRIPLET = (276, 277, 278)
R277_REVIEWED = (
    ('facade_edge', 'far_building', 'primary', (1135, 3006, 3053),
     'Fixed grey facade edge; the same feature also appears in the later frozen triplet.'),
    ('basket_frame_joint', 'near_hoop', 'primary', (1547, 449, 175),
     'Red support and diagonal red brace form a visible fixed joint.'),
    ('lamp_foot', 'near_ground', 'primary', (793, 519, 208),
     'Black lamp-post foot on the blue paving.'),
    ('lamp_head', 'elevated_fixture', 'primary', (1710, 1563, 1266),
     'Upper right corner of the black two-headed lamp.'),
    ('window_grid', 'far_building', 'context_only', (2966, 868, 668),
     'Glass facade grid repeats; exact cell identity is less certain.'),
)

# Chosen from native crops before querying pose-based residuals. The tree trunk
# is context only: its base touches hedge foliage and is not a precise pixel.
REVIEWED = (
    ('early_facade_edge', 'far_building', 'primary', (277, 278, 279), (3006, 3053, 879),
     'Visible facade edge beside the darker building; windows repeat.'),
    ('early_building_outline', 'far_building', 'primary', (277, 278, 279), (2993, 794, 763),
     'Building outer silhouette against the sky.'),
    ('late_tower_ledge', 'far_building', 'primary', (279, 280, 281), (1178, 2696, 677),
     'Ledge meeting the dark vertical strip of the glass tower.'),
    ('late_roof_unit', 'far_building', 'primary', (279, 280, 281), (20, 20, 41),
     'Distinct roof unit silhouette against the sky.'),
    ('late_basket_base', 'near_hoop', 'primary', (279, 280, 281), (2606, 868, 430),
     'Red support meets blue basket base at a manufactured corner.'),
    ('late_basket_brace', 'near_hoop', 'primary', (279, 280, 281), (1087, 3071, 1427),
     'Red vertical support meets its slanted red brace.'),
    ('late_tree_trunk', 'tree_trunk', 'context_only', (279, 280, 281), (333, 1105, 1050),
     'Trunk is static, but its picked base lies against moving hedge foliage.'),
)


def _write_protocol(path, protocol):
    fingerprint = digest(protocol)
    if path.exists():
        if json.loads(path.read_text())['fingerprint'] != fingerprint:
            raise ValueError(f'Changed static-depth inputs: {path}')
    else:
        write_json(path, {**protocol, 'fingerprint': fingerprint,
                          'frozen_at_utc': datetime.now(timezone.utc).isoformat()})
    return fingerprint


def _inputs(root, triplets=TRIPLETS):
    frontend_path = root / 'frontend.json'
    frontend = json.loads(frontend_path.read_text())
    records = {row['name']: row for row in load_scene(root)['images']}
    frames = sorted({frame for triple in triplets for frame in triple})
    names = {frame: next(name for name in frontend['features']
                         if name.startswith('R/') and f'_{frame:05d}-R' in name)
             for frame in frames}
    required = {pair for triple in triplets
                for pair in itertools.combinations(triple, 2)}
    pairs = {}
    for row in frontend['pairs']:
        if row['a'] not in names.values() or row['b'] not in names.values():
            continue
        first = next(frame for frame, name in names.items() if name == row['a'])
        second = next(frame for frame, name in names.items() if name == row['b'])
        if (first, second) not in required:
            continue
        with np.load(row['path'], allow_pickle=False) as data:
            matches, scores = data['matches'], data['scores']
        pairs[first, second] = {
            'row': row, 'forward': {int(a): int(b) for a, b in matches},
            'confidence': {int(a): float(score)
                           for (a, _), score in zip(matches, scores)}}
    if set(pairs) != required:
        raise ValueError(f'Missing R-frame LoMa pairs: {required-set(pairs)}')
    features = {}
    for frame, name in names.items():
        with np.load(frontend['features'][name]['path'], allow_pickle=False) as data:
            features[frame] = data['keypoints'].copy()
    return frontend_path, frontend, records, names, pairs, features


def _cycles(frames, pairs):
    first, middle, last = frames
    forward = pairs[first, middle]['forward']
    direct = pairs[first, last]['forward']
    remaining = pairs[middle, last]['forward']
    for i, j in forward.items():
        k = direct.get(i)
        if k is not None and remaining.get(j) == k:
            yield i, j, k


def _review_sheet(output, records, rows):
    canvas = np.full((len(rows)*224, 3*224, 3), 24, np.uint8)
    images = {observation['name']: cv2.imread(records[observation['name']]['source'])
              for row in rows for observation in row['observations']}
    for row_index, row in enumerate(rows):
        for column, observation in enumerate(row['observations']):
            x, y = np.rint(observation['native_xy']).astype(int)
            crop = images[observation['name']][y-96:y+96, x-96:x+96].copy()
            if crop.shape[:2] != (192, 192):
                raise ValueError(f'{row["id"]}: native review crop meets border')
            cv2.drawMarker(crop, (96, 96), (0, 0, 255), cv2.MARKER_CROSS, 14, 2)
            x0, y0 = column*224, row_index*224
            canvas[y0:y0+192, x0:x0+192] = crop
            cv2.putText(canvas, f'{row["id"]} R{observation["frame"]}',
                        (x0+2, y0+211), cv2.FONT_HERSHEY_SIMPLEX,
                        .43, (245, 245, 245), 1)
    review = output / 'review'
    review.mkdir(exist_ok=True)
    path = review / 'selected_native_static_cycles.jpg'
    cv2.imwrite(str(path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return {'path': str(path), 'sha256': file_digest(path)}


def run_r_static_depth_selection(work):
    root = Path(work).resolve() / 'global_sfm'
    output = root / OUTPUT
    output.mkdir(exist_ok=True)
    frontend_path, frontend, records, names, pairs, features = _inputs(root)
    model_path = root / 'ground_joint_ba/equal_control/sparse/0'
    protocol = {
        'version': 1, 'frontend_sha256': file_digest(frontend_path),
        'source_sha256': {names[f]: file_digest(records[names[f]]['source']) for f in names},
        'feature_sha256': {names[f]: file_digest(frontend['features'][names[f]]['path'])
                           for f in names},
        'pair_sha256': {f'{a}-{b}': file_digest(row['row']['path'])
                        for (a, b), row in pairs.items()},
        'visual_model_hashes': {path.name: file_digest(path)
                                for path in sorted(model_path.glob('*.bin'))},
        'triplets': TRIPLETS, 'reviewed_selection': REVIEWED,
        'selection': 'Exact three-edge LoMa cycles, native crop review, and no mapped 3D feature in any of the three images',
        'upper_region_max_native_y': 2100,
        'uses_navigation_pose_values_intrinsics_or_las': False,
        'scope': 'Freeze pixel identities before pose audit; no SfM change.'}
    fingerprint = _write_protocol(output / 'selection_protocol.json', protocol)
    result_path = output / 'selection.json'
    if result_path.exists():
        old = json.loads(result_path.read_text())
        if old['protocol_fingerprint'] != fingerprint:
            raise ValueError('Changed R static-depth selection inputs')
        return old
    model = pycolmap.Reconstruction(model_path)
    images = {image.name: image for image in model.images.values()}
    inventory = []
    for frames in TRIPLETS:
        cycles = list(_cycles(frames, pairs))
        upper = [indices for indices in cycles
                 if all(features[frame][index, 1] < 2100
                        for frame, index in zip(frames, indices))]
        unmapped = [indices for indices in upper
                    if all(not images[names[frame]].points2D[index].has_point3D()
                           for frame, index in zip(frames, indices))]
        inventory.append({'frames': frames, 'exact_cycle_count': len(cycles),
                          'upper_count': len(upper),
                          'upper_all_unmapped_count': len(unmapped)})
    rows = []
    for identity, scene_type, role, frames, indices, note in REVIEWED:
        if frames not in TRIPLETS or indices not in set(_cycles(frames, pairs)):
            raise ValueError(f'{identity}: no exact LoMa cycle in frozen triplet')
        if any(features[frame][index, 1] >= 2100
               for frame, index in zip(frames, indices)):
            raise ValueError(f'{identity}: outside upper image region')
        if any(images[names[frame]].points2D[index].has_point3D()
               for frame, index in zip(frames, indices)):
            raise ValueError(f'{identity}: already mapped in saved visual model')
        confidence = [pairs[a, b]['confidence'][indices[frames.index(a)]]
                      for a, b in itertools.combinations(frames, 2)]
        rows.append({'id': identity, 'scene_type': scene_type, 'role': role,
                     'image_review_note': note, 'frames': frames,
                     'min_loma_pair_confidence': min(confidence),
                     'pair_confidences': confidence,
                     'observations': [{'frame': frame, 'name': names[frame],
                                       'feature_index': index,
                                       'native_xy': features[frame][index].tolist()}
                                      for frame, index in zip(frames, indices)]})
    review = _review_sheet(output, records, rows)
    result = {'protocol_fingerprint': fingerprint, 'inventory': inventory,
              'selected': rows, 'review': review,
              'guard': 'High LoMa score and cycle closure do not prove subpixel identity. Leaf/hedge tracks are excluded from primary evidence.'}
    write_json(result_path, result)
    return result


def run_r277_static_selection(work):
    root = Path(work).resolve() / 'global_sfm'
    output = root / R277_OUTPUT
    output.mkdir(exist_ok=True)
    triplets = (R277_TRIPLET,)
    frontend_path, frontend, records, names, pairs, features = _inputs(root, triplets)
    model_path = root / 'ground_joint_ba/equal_control/sparse/0'
    protocol = {
        'version': 1, 'frontend_sha256': file_digest(frontend_path),
        'source_sha256': {names[f]: file_digest(records[names[f]]['source']) for f in names},
        'feature_sha256': {names[f]: file_digest(frontend['features'][names[f]]['path'])
                           for f in names},
        'pair_sha256': {f'{a}-{b}': file_digest(row['row']['path'])
                        for (a, b), row in pairs.items()},
        'visual_model_hashes': {path.name: file_digest(path)
                                for path in sorted(model_path.glob('*.bin'))},
        'triplet': R277_TRIPLET, 'reviewed_selection': R277_REVIEWED,
        'selection': 'Exact three-edge LoMa cycle, native crop review, and no mapped 3D feature in any of the three images',
        'upper_region_max_native_y': 2100,
        'uses_navigation_pose_values_intrinsics_or_las': False,
        'scope': 'Freeze R277 static identities before pose audit; no SfM change.'}
    fingerprint = _write_protocol(output / 'selection_protocol.json', protocol)
    result_path = output / 'selection.json'
    if result_path.exists():
        old = json.loads(result_path.read_text())
        if old['protocol_fingerprint'] != fingerprint:
            raise ValueError('Changed R277 static selection inputs')
        return old
    model = pycolmap.Reconstruction(model_path)
    images = {image.name: image for image in model.images.values()}
    cycles = list(_cycles(R277_TRIPLET, pairs))
    upper = [indices for indices in cycles
             if all(features[frame][index, 1] < 2100
                    for frame, index in zip(R277_TRIPLET, indices))]
    unmapped = [indices for indices in upper
                if all(not images[names[frame]].points2D[index].has_point3D()
                       for frame, index in zip(R277_TRIPLET, indices))]
    rows = []
    for identity, scene_type, role, indices, note in R277_REVIEWED:
        if indices not in set(unmapped):
            raise ValueError(f'{identity}: not an upper-region, fully unmapped LoMa cycle')
        confidence = [pairs[a, b]['confidence'][indices[R277_TRIPLET.index(a)]]
                      for a, b in itertools.combinations(R277_TRIPLET, 2)]
        rows.append({'id': identity, 'scene_type': scene_type, 'role': role,
                     'image_review_note': note, 'frames': R277_TRIPLET,
                     'min_loma_pair_confidence': min(confidence),
                     'pair_confidences': confidence,
                     'observations': [{'frame': frame, 'name': names[frame],
                                       'feature_index': index,
                                       'native_xy': features[frame][index].tolist()}
                                      for frame, index in zip(R277_TRIPLET, indices)]})
    review = _review_sheet(output, records, rows)
    result = {'protocol_fingerprint': fingerprint,
              'inventory': {'frames': R277_TRIPLET, 'exact_cycle_count': len(cycles),
                            'upper_count': len(upper),
                            'upper_all_unmapped_count': len(unmapped)},
              'selected': rows, 'review': review,
              'guard': 'Four primary static structures were chosen in native crops before reading R277 pose residuals. Repetitive window grid is context only.'}
    write_json(result_path, result)
    return result


def run_r_static_depth_audit(work):
    root = Path(work).resolve() / 'global_sfm'
    output = root / OUTPUT
    selection_path = output / 'selection.json'
    if not selection_path.exists():
        raise FileNotFoundError(f'Run static-depth selection first: {selection_path}')
    selection = json.loads(selection_path.read_text())
    model_path = root / 'ground_joint_ba/equal_control/sparse/0'
    protocol = {
        'version': 1, 'selection_sha256': file_digest(selection_path),
        'visual_model_hashes': {path.name: file_digest(path)
                                for path in sorted(model_path.glob('*.bin'))},
        'support_rule': 'First and third exposure of each frozen exact LoMa 3-cycle; middle exposure held out',
        'pose_and_calibration': 'Saved equal-control visual SfM and candidate R OpenCV fisheye intrinsics, not truth',
        'uses_navigation_pose_values_or_las': False,
        'scope': 'Diagnostic only; no BA or LFS changes.'}
    fingerprint = _write_protocol(output / 'audit_protocol.json', protocol)
    result_path = output / 'audit.json'
    if result_path.exists():
        old = json.loads(result_path.read_text())
        if old['protocol_fingerprint'] != fingerprint:
            raise ValueError('Changed R static-depth audit inputs')
        return old
    model = pycolmap.Reconstruction(model_path)
    images = {image.name: image for image in model.images.values()}
    camera_data = model.cameras[2]
    camera = Camera('R', camera_data.width, camera_data.height,
                    tuple(camera_data.params), (0, 0))
    rows = []
    for selected in selection['selected']:
        observations = selected['observations']
        rays = [_world_ray(images[row['name']], camera, np.asarray(row['native_xy']))
                for row in (observations[0], observations[2])]
        world = _triangulate(*rays)
        angle = float(np.rad2deg(np.arccos(np.clip(rays[0][1] @ rays[1][1], -1., 1.))))
        predictions = []
        for observation in observations:
            projected, depth = _project(images[observation['name']], camera, world)
            residual = (projected-np.asarray(observation['native_xy'])
                        if projected is not None else None)
            predictions.append({'frame': observation['frame'], 'camera_depth': depth,
                                'predicted_native_xy': projected.tolist()
                                if projected is not None else None,
                                'residual_xy_native_px': residual.tolist()
                                if residual is not None else None,
                                'error_native_px': float(np.linalg.norm(residual))
                                if residual is not None else None})
        rows.append({'id': selected['id'], 'scene_type': selected['scene_type'],
                     'role': selected['role'], 'support_ray_angle_degrees': angle,
                     'predictions': predictions,
                     'heldout_middle_error_native_px': predictions[1]['error_native_px'],
                     'heldout_middle_residual_xy_native_px': predictions[1]['residual_xy_native_px']})
    summary = {}
    for triplet in TRIPLETS:
        key = '-'.join(str(frame) for frame in triplet)
        primary_ids = {row['id'] for row in selection['selected']
                       if tuple(row['frames']) == triplet and row['role'] == 'primary'}
        errors = [row['heldout_middle_error_native_px'] for row in rows
                  if row['id'] in primary_ids and row['heldout_middle_error_native_px'] is not None]
        summary[key] = {'primary_count': len(errors),
                        'median_error_native_px': float(np.median(errors)) if errors else None,
                        'max_error_native_px': float(np.max(errors)) if errors else None}
    for scene_type in ('far_building', 'near_hoop'):
        subset = [row for row in rows if row['role'] == 'primary'
                  and row['scene_type'] == scene_type]
        errors = [row['heldout_middle_error_native_px'] for row in subset
                  if row['heldout_middle_error_native_px'] is not None]
        summary[scene_type] = {'count': len(errors),
                               'median_error_native_px': float(np.median(errors))
                               if errors else None,
                               'max_error_native_px': float(np.max(errors)) if errors else None}
    result = {'protocol_fingerprint': fingerprint, 'rows': rows, 'summary': summary,
              'guard': 'Saved visual poses and fitted fisheye calibration set support rays. Sparse LoMa coordinates cannot uniquely separate pose, calibration, frontend location error, and scene motion.'}
    write_json(result_path, result)
    write_json(output / 'DIAGNOSTIC_ONLY.json', {
        'status': 'DIAGNOSTIC_ONLY', 'audit_sha256': file_digest(result_path),
        'reason': 'Near/far heldouts require consistent independent evidence before BA or calibration update.'})
    return result


def run_r277_static_audit(work):
    root = Path(work).resolve() / 'global_sfm'
    output = root / R277_OUTPUT
    selection_path = output / 'selection.json'
    if not selection_path.exists():
        raise FileNotFoundError(f'Run R277 static selection first: {selection_path}')
    selection = json.loads(selection_path.read_text())
    model_path = root / 'ground_joint_ba/equal_control/sparse/0'
    protocol = {
        'version': 1, 'selection_sha256': file_digest(selection_path),
        'visual_model_hashes': {path.name: file_digest(path)
                                for path in sorted(model_path.glob('*.bin'))},
        'support_rule': 'R276 and R278 rays from each frozen LoMa 3-cycle; R277 held out',
        'pose_and_calibration': 'Saved equal-control visual SfM and candidate R OpenCV fisheye intrinsics, not truth',
        'uses_navigation_pose_values_or_las': False,
        'scope': 'Diagnostic only; no BA or LFS changes.'}
    fingerprint = _write_protocol(output / 'audit_protocol.json', protocol)
    result_path = output / 'audit.json'
    if result_path.exists():
        old = json.loads(result_path.read_text())
        if old['protocol_fingerprint'] != fingerprint:
            raise ValueError('Changed R277 static audit inputs')
        return old
    model = pycolmap.Reconstruction(model_path)
    images = {image.name: image for image in model.images.values()}
    camera_data = model.cameras[2]
    camera = Camera('R', camera_data.width, camera_data.height,
                    tuple(camera_data.params), (0, 0))
    rows = []
    for selected in selection['selected']:
        observations = selected['observations']
        rays = [_world_ray(images[row['name']], camera, np.asarray(row['native_xy']))
                for row in (observations[0], observations[2])]
        world = _triangulate(*rays)
        angle = float(np.rad2deg(np.arccos(np.clip(rays[0][1] @ rays[1][1], -1., 1.))))
        predictions = []
        for observation in observations:
            projected, depth = _project(images[observation['name']], camera, world)
            residual = (projected-np.asarray(observation['native_xy'])
                        if projected is not None else None)
            predictions.append({'frame': observation['frame'], 'camera_depth': depth,
                                'predicted_native_xy': projected.tolist()
                                if projected is not None else None,
                                'residual_xy_native_px': residual.tolist()
                                if residual is not None else None,
                                'error_native_px': float(np.linalg.norm(residual))
                                if residual is not None else None})
        rows.append({'id': selected['id'], 'scene_type': selected['scene_type'],
                     'role': selected['role'], 'support_ray_angle_degrees': angle,
                     'predictions': predictions,
                     'heldout_middle_error_native_px': predictions[1]['error_native_px'],
                     'heldout_middle_residual_xy_native_px': predictions[1]['residual_xy_native_px']})
    primary = [row['heldout_middle_error_native_px'] for row in rows
               if row['role'] == 'primary' and row['heldout_middle_error_native_px'] is not None]
    result = {'protocol_fingerprint': fingerprint, 'rows': rows,
              'summary': {'primary_count': len(primary),
                          'median_error_native_px': float(np.median(primary)),
                          'max_error_native_px': float(np.max(primary))},
              'guard': 'The saved model supplies support rays; LoMa feature pixels are not ground truth. Four manually reviewed structures cannot exclude smaller local calibration or pose errors.'}
    write_json(result_path, result)
    write_json(output / 'DIAGNOSTIC_ONLY.json', {
        'status': 'DIAGNOSTIC_ONLY', 'audit_sha256': file_digest(result_path),
        'reason': 'R277 static heldouts do not authorize BA or calibration changes alone.'})
    return result
