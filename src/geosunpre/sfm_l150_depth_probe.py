"""Image-first audit of LoMa ground coverage around the L150 court views.

The selection stage never reads pose values. The audit stage uses a saved visual
SfM solution only to predict the middle exposure of frozen three-image cycles.
"""

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


OUTPUT = 'sfm_l150_depth_probe'
TRIPLET = (151, 152, 153)
INVENTORY_FRAMES = tuple(range(149, 156))
LOWER_SCREEN_Y = 2300

# IDs came from image-only three-edge cycles and native-image crop inspection.
# Ground rows are deliberately not promoted: repeated tile/paint patterns and
# the white object do not establish the same physical pixel in every view.
REVIEWED = (
    ('tower_sky_edge', 'far_building', 'control', 8,
     'Building silhouette at a specific facade band; one-dimensional edge localization remains possible.'),
    ('facade_vertical_joint', 'far_building', 'control', 568,
     'Fixed facade joint beside the tree, away from foliage.'),
    ('canopy_post_foot', 'near_structure', 'control', 68,
     'Manufactured canopy post meeting the paving.'),
    ('black_post_foot', 'near_structure', 'control', 218,
     'Black fixed post at the blue/green paving boundary.'),
    ('red_tile_patch', 'court_ground', 'ambiguous', 460,
     'Repeated tile texture; no unique physical mark at the three LoMa centers.'),
    ('red_white_stripe', 'court_ground', 'ambiguous', 1539,
     'Feature lies near a broad white stripe; along-stripe pixel identity is uncertain.'),
    ('red_stripe_end', 'court_ground', 'ambiguous', 296,
     'Three crops show the stripe, but not a stable single point on its edge.'),
    ('white_object', 'court_ground', 'suspect', 265,
     'Feature moves from plain red tile to a white object across views; object itself may move.'),
)


def _frozen_protocol(path, payload):
    fingerprint = digest(payload)
    if path.exists():
        if json.loads(path.read_text())['fingerprint'] != fingerprint:
            raise ValueError(f'Frozen probe inputs changed: {path}')
    else:
        write_json(path, {**payload, 'fingerprint': fingerprint,
                          'frozen_at_utc': datetime.now(timezone.utc).isoformat()})
    return fingerprint


def _inputs(root):
    frontend_path = root / 'frontend.json'
    frontend = json.loads(frontend_path.read_text())
    records = {record['name']: record for record in load_scene(root)['images']}
    names = {frame: next(name for name in frontend['features']
                         if name.startswith('L/') and f'_{frame:05d}-L' in name)
             for frame in INVENTORY_FRAMES}
    required = {pair for pair in itertools.combinations(INVENTORY_FRAMES, 2)
                if pair[1] - pair[0] <= 3}
    pair_rows = {}
    for row in frontend['pairs']:
        if row['a'] not in names.values() or row['b'] not in names.values():
            continue
        a = next(frame for frame, name in names.items() if name == row['a'])
        b = next(frame for frame, name in names.items() if name == row['b'])
        if (a, b) in required:
            pair_rows[a, b] = row
    if set(pair_rows) != required:
        raise ValueError(f'Missing LoMa image pairs: {required - set(pair_rows)}')
    features = {}
    for frame, name in names.items():
        with np.load(frontend['features'][name]['path'], allow_pickle=False) as data:
            features[frame] = data['keypoints'].copy()
    matches, confidences = {}, {}
    for pair, row in pair_rows.items():
        with np.load(row['path'], allow_pickle=False) as data:
            index_pairs, scores = data['matches'], data['scores']
        mapping = {int(a): int(b) for a, b in index_pairs}
        if len(mapping) != len(index_pairs) or len(set(mapping.values())) != len(mapping):
            raise ValueError(f'Non-unique LoMa pair: {pair}')
        matches[pair] = mapping
        confidences[pair] = {int(a): float(score)
                             for (a, _), score in zip(index_pairs, scores)}
    return frontend_path, frontend, records, names, pair_rows, features, matches, confidences


def _cycles(frames, matches):
    first, middle, last = frames
    ab, ac, bc = (matches[first, middle], matches[first, last],
                  matches[middle, last])
    return [(i, j, ac[i]) for i, j in ab.items()
            if i in ac and bc.get(j) == ac[i]]


def _four_cycles(frames, matches):
    first = frames[0]
    result = []
    for index in matches[first, frames[1]]:
        indices = {first: index}
        for frame in frames[1:]:
            if index not in matches[first, frame]:
                break
            indices[frame] = matches[first, frame][index]
        if len(indices) == len(frames) and all(
                matches[a, b].get(indices[a]) == indices[b]
                for a, b in itertools.combinations(frames, 2)):
            result.append(tuple(indices[frame] for frame in frames))
    return result


def _inventory(features, matches, confidences):
    triplets = []
    for start in range(149, 154):
        frames = (start, start + 1, start + 2)
        cycles = _cycles(frames, matches)
        lower = [indices for indices in cycles
                 if features[start][indices[0], 1] > LOWER_SCREEN_Y]
        high_lower = [indices for indices in lower if min(
            confidences[a, b][indices[frames.index(a)]]
            for a, b in itertools.combinations(frames, 2)) >= .9]
        triplets.append({'frames': frames, 'exact_three_edge_cycles': len(cycles),
                         'lower_screen_cycles': len(lower),
                         'lower_screen_min_confidence_ge_0p9': len(high_lower)})
    fours = []
    for start in range(149, 153):
        frames = tuple(range(start, start + 4))
        cycles = _four_cycles(frames, matches)
        fours.append({'frames': frames, 'exact_six_edge_cycles': len(cycles),
                      'lower_screen_cycles': int(sum(
                          features[start][indices[0], 1] > LOWER_SCREEN_Y
                          for indices in cycles))})
    keypoints = [{'frame': frame, 'all_keypoints': len(features[frame]),
                  'lower_screen_keypoints': int((features[frame][:, 1] > LOWER_SCREEN_Y).sum()),
                  'below_y3000_keypoints': int((features[frame][:, 1] > 3000).sum())}
                 for frame in INVENTORY_FRAMES]
    return {'triplets': triplets, 'four_view_windows': fours,
            'keypoint_coverage': keypoints,
            'lower_screen_rule': f'first-image native y > {LOWER_SCREEN_Y}; screen location only, not court semantics'}


def _review_sheet(output, records, selected):
    side, crop_side, stride = 304, 290, 338
    canvas = np.full((len(selected) * stride, len(TRIPLET) * side, 3), 24, np.uint8)
    images = {row['name']: cv2.imread(records[row['name']]['source'])
              for item in selected for row in item['observations']}
    for row_index, item in enumerate(selected):
        for col, obs in enumerate(item['observations']):
            x, y = np.rint(obs['native_xy']).astype(int)
            crop = images[obs['name']][y-145:y+145, x-145:x+145]
            if crop.shape[:2] != (crop_side, crop_side):
                raise ValueError(f'Crop meets image border: {item["id"]}')
            x0, y0 = col * side, row_index * stride
            canvas[y0:y0+crop_side, x0:x0+crop_side] = crop
            cv2.drawMarker(canvas, (x0+145, y0+145), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
            cv2.putText(canvas, f'L{obs["frame"]} i{obs["feature_index"]} {x},{y}',
                        (x0+2, y0+316), cv2.FONT_HERSHEY_SIMPLEX, .53, (245, 245, 245), 1)
    path = output / 'review/frozen_native_cycles.jpg'
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return {'path': str(path), 'sha256': file_digest(path)}


def _mapped_track_sheet(output, records, model, point_id):
    point = model.points3D[point_id]
    observations = []
    for element in point.track.elements:
        image = model.images[element.image_id]
        if image.name not in records:
            continue
        xy = image.points2D[element.point2D_idx].xy
        source = cv2.imread(records[image.name]['source'])
        x, y = np.rint(xy).astype(int)
        crop = source[y-120:y+120, x-120:x+120]
        if crop.shape[:2] != (240, 240):
            continue
        tile = np.full((270, 270, 3), 24, np.uint8)
        tile[:240, :240] = crop
        cv2.drawMarker(tile, (120, 120), (0, 0, 255), cv2.MARKER_CROSS, 22, 2)
        cv2.putText(tile, f'{records[image.name]["camera"]}{records[image.name]["frame_index"]} '
                         f'i{element.point2D_idx}', (4, 257),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (245, 245, 245), 1)
        observations.append(tile)
    cols = 4
    canvas = np.full((((len(observations)+cols-1)//cols)*270, cols*270, 3),
                     24, np.uint8)
    for index, tile in enumerate(observations):
        row, col = divmod(index, cols)
        canvas[row*270:(row+1)*270, col*270:(col+1)*270] = tile
    path = output / f'review/mapped_point_{point_id}_all_views.jpg'
    cv2.imwrite(str(path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return {'point3D_id': point_id, 'track_length': len(point.track.elements),
            'saved_model_mean_reprojection_error_px': float(point.error),
            'review_path': str(path), 'review_sha256': file_digest(path)}


def run_l150_selection(work):
    root = Path(work).resolve() / 'global_sfm'
    output = root / OUTPUT
    output.mkdir(exist_ok=True)
    frontend_path, frontend, records, names, pair_rows, features, matches, scores = _inputs(root)
    protocol = {
        'version': 1, 'frontend_sha256': file_digest(frontend_path),
        'image_sha256': {names[f]: file_digest(records[names[f]]['source']) for f in names},
        'feature_sha256': {names[f]: file_digest(frontend['features'][names[f]]['path']) for f in names},
        'match_sha256': {f'{a}-{b}': file_digest(row['path']) for (a, b), row in pair_rows.items()},
        'triplet': TRIPLET, 'inventory_frames': INVENTORY_FRAMES,
        'lower_screen_y': LOWER_SCREEN_Y, 'reviewed': REVIEWED,
        'selection': 'Exact native LoMa cycles and original JPEG crop inspection, before pose audit',
        'uses_pose_values_intrinsics_or_las': False,
        'scope': 'Image-only diagnostic selection; no model or LFS input change.'}
    fingerprint = _frozen_protocol(output / 'selection_protocol.json', protocol)
    result_path = output / 'selection.json'
    if result_path.exists():
        previous = json.loads(result_path.read_text())
        if previous['protocol_fingerprint'] != fingerprint:
            raise ValueError('Existing L150 selection belongs to changed inputs')
        return previous
    all_cycles = {indices[0]: indices for indices in _cycles(TRIPLET, matches)}
    selected = []
    for identity, scene_type, role, index, note in REVIEWED:
        if index not in all_cycles:
            raise ValueError(f'{identity}: no exact three-edge cycle')
        indices = all_cycles[index]
        confidence = [scores[a, b][indices[TRIPLET.index(a)]]
                      for a, b in itertools.combinations(TRIPLET, 2)]
        selected.append({'id': identity, 'scene_type': scene_type, 'role': role,
                         'image_review_note': note, 'pair_confidences': confidence,
                         'min_pair_confidence': min(confidence),
                         'observations': [{'frame': frame, 'name': names[frame],
                                           'feature_index': idx,
                                           'native_xy': features[frame][idx].tolist()}
                                          for frame, idx in zip(TRIPLET, indices)]})
    result = {'protocol_fingerprint': fingerprint,
              'inventory': _inventory(features, matches, scores),
              'selected': selected,
              'review': _review_sheet(output, records, selected),
              'guard': 'Lower image location is not a court mask. Three-edge closure does not establish exact physical pixel identity; ground rows remain ambiguous or suspect.'}
    write_json(result_path, result)
    return result


def run_l150_audit(work):
    root = Path(work).resolve() / 'global_sfm'
    output = root / OUTPUT
    selection_path = output / 'selection.json'
    if not selection_path.exists():
        raise FileNotFoundError(f'Freeze image selection first: {selection_path}')
    selected = json.loads(selection_path.read_text())
    model_path = root / 'ground_joint_ba/equal_control/sparse/0'
    protocol = {
        'version': 1, 'selection_sha256': file_digest(selection_path),
        'visual_model_hashes': {path.name: file_digest(path)
                                for path in sorted(model_path.glob('*.bin'))},
        'support_rule': 'L151 and L153 triangulate; L152 native feature pixel held out',
        'camera_model': 'Saved equal-control visual SfM and candidate L OPENCV_FISHEYE, not truth',
        'uses_navigation_pose_values_or_las': False,
        'scope': 'Diagnostic only; no BA, calibration or LFS change.'}
    fingerprint = _frozen_protocol(output / 'audit_protocol.json', protocol)
    result_path = output / 'audit.json'
    if result_path.exists():
        previous = json.loads(result_path.read_text())
        if previous['protocol_fingerprint'] != fingerprint:
            raise ValueError('Existing L150 audit belongs to changed inputs')
        if 'suspect_model_tracks' in previous:
            return previous
    model = pycolmap.Reconstruction(model_path)
    images = {image.name: image for image in model.images.values()}
    cam_data = model.cameras[1]
    camera = Camera('L', cam_data.width, cam_data.height, tuple(cam_data.params), (0, 0))
    rows = []
    for item in selected['selected']:
        observations = item['observations']
        for obs in observations:
            saved_xy = images[obs['name']].points2D[obs['feature_index']].xy
            if not np.allclose(saved_xy, obs['native_xy'], atol=.02):
                raise ValueError(f'Model and frozen LoMa coordinate disagree: {item["id"]}')
        rays = [_world_ray(images[obs['name']], camera, np.asarray(obs['native_xy']))
                for obs in (observations[0], observations[2])]
        angle = float(np.rad2deg(np.arccos(np.clip(rays[0][1] @ rays[1][1], -1., 1.))))
        world = _triangulate(*rays)
        predictions = []
        mapped = []
        for obs in observations:
            point = images[obs['name']].points2D[obs['feature_index']]
            mapped.append(int(point.point3D_id) if point.has_point3D() else None)
            projected, depth = _project(images[obs['name']], camera, world)
            residual = projected - np.asarray(obs['native_xy']) if projected is not None else None
            predictions.append({'frame': obs['frame'], 'camera_depth': depth,
                                'predicted_native_xy': projected.tolist() if projected is not None else None,
                                'residual_xy_native_px': residual.tolist() if residual is not None else None,
                                'error_native_px': float(np.linalg.norm(residual))
                                if residual is not None else None})
        rows.append({'id': item['id'], 'scene_type': item['scene_type'], 'role': item['role'],
                     'min_pair_confidence': item['min_pair_confidence'],
                     'mapped_point3D_ids': mapped,
                     'support_ray_angle_degrees': angle, 'predictions': predictions,
                     'heldout_middle_error_native_px': predictions[1]['error_native_px'],
                     'heldout_middle_residual_xy_native_px': predictions[1]['residual_xy_native_px']})
    control = [row['heldout_middle_error_native_px'] for row in rows
               if row['role'] == 'control' and row['heldout_middle_error_native_px'] is not None]
    records = {record['name']: record for record in load_scene(root)['images']}
    suspect_ids = sorted({point_id for row in rows if row['role'] == 'suspect'
                          for point_id in row['mapped_point3D_ids'] if point_id is not None})
    result = {'protocol_fingerprint': fingerprint, 'rows': rows,
              'summary': {'control_count': len(control),
                          'control_median_error_native_px': float(np.median(control)) if control else None,
                          'control_max_error_native_px': float(np.max(control)) if control else None},
              'suspect_model_tracks': [_mapped_track_sheet(output, records, model, point_id)
                                       for point_id in suspect_ids],
              'guard': 'Only image-reviewed fixed structures are controls. Ground LoMa cycles are not precise validation points; reprojection alone cannot distinguish pose, calibration, and match error.'}
    write_json(result_path, result)
    write_json(output / 'DIAGNOSTIC_ONLY.json', {
        'status': 'DIAGNOSTIC_ONLY', 'audit_sha256': file_digest(result_path),
        'reason': 'No stable, unambiguous near-ground multi-view pixel track was frozen for BA.'})
    return result
