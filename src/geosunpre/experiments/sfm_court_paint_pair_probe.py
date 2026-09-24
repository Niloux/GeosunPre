"""Find paired painted-court junctions across native L/R exposure sequences.

Pair identity comes from two three-color junctions touching one red court
component and from contiguous image order. Saved visual poses are consulted
only after native-pixel localization, as an audit rather than an association
prior. The resulting tracks are proposals, not BA constraints.
"""

from datetime import datetime, timezone
import json
from pathlib import Path

import cv2
import numpy as np
import pycolmap

from ..camera import Camera
from ..scene import digest, file_digest, load_scene, write_json
from .sfm_court_anchor_probe import _triangulate
from .sfm_court_native_anchor_probe import (_color_junction_native,
                                            _project, _world_ray)
from .sfm_court_trajectory_probe import _hue_masks


OUTPUT = 'sfm_court_paint_pair_probe'
QUARTER_SHAPE = (1000, 750)
TRIPLE_DILATION = 15
RED_COMPONENT_MIN_AREA = 2000
TRIPLE_AREA_RANGE = (4, 400)
CLUSTER_DISTANCE = 35.
PAIR_SEPARATION_RANGE = (40., 300.)
PAIR_MAX_Y_SEPARATION = 140.
MAX_INTERFRAME_MOTION = 150.
MIN_RUN_LENGTH = 5


def _write_protocol(path, protocol):
    fingerprint = digest(protocol)
    if path.exists():
        if json.loads(path.read_text())['fingerprint'] != fingerprint:
            raise ValueError(f'Changed court paint-pair inputs: {path}')
    else:
        write_json(path, {**protocol, 'fingerprint': fingerprint,
                          'frozen_at_utc': datetime.now(timezone.utc).isoformat()})
    return fingerprint


def _image_pair(record):
    bgr = cv2.imread(record['source'], cv2.IMREAD_REDUCED_COLOR_4)
    if bgr is None or bgr.shape[:2] != QUARTER_SHAPE:
        raise ValueError(f'Missing quarter-resolution image: {record["source"]}')
    masks = _hue_masks(bgr, 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (TRIPLE_DILATION, TRIPLE_DILATION))
    triple = np.logical_and.reduce([cv2.dilate(mask, kernel) > 0
                                    for mask in masks]).astype(np.uint8)
    triple[:350] = 0
    count, _, stats, centers = cv2.connectedComponentsWithStats(triple, 8)
    _, red_ids, red_stats, _ = cv2.connectedComponentsWithStats(masks[0], 8)
    candidates = []
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if not TRIPLE_AREA_RANGE[0] <= area <= TRIPLE_AREA_RANGE[1]:
            continue
        x, y = centers[index]
        ix, iy = int(round(x)), int(round(y))
        neighborhood = red_ids[max(0, iy - 12):iy + 13,
                               max(0, ix - 12):ix + 13]
        labels = np.bincount(neighborhood.ravel())
        labels[0] = 0
        red_id = int(np.argmax(labels))
        if not red_id or red_stats[red_id, cv2.CC_STAT_AREA] < RED_COMPONENT_MIN_AREA:
            continue
        candidates.append({'quarter_xy': [float(x), float(y)],
                           'triple_area_quarter_px': area, 'red_id': red_id,
                           'red_area_quarter_px': int(red_stats[red_id, cv2.CC_STAT_AREA])})
    clusters = []
    visited = set()
    for index, candidate in enumerate(candidates):
        if index in visited:
            continue
        members = [index]
        visited.add(index)
        changed = True
        while changed:
            changed = False
            for other_index, other in enumerate(candidates):
                if other_index in visited or other['red_id'] != candidate['red_id']:
                    continue
                xy = np.asarray(other['quarter_xy'])
                if any(np.linalg.norm(xy - candidates[i]['quarter_xy']) <= CLUSTER_DISTANCE
                       for i in members):
                    members.append(other_index)
                    visited.add(other_index)
                    changed = True
        clusters.append(max((candidates[i] for i in members),
                            key=lambda row: row['triple_area_quarter_px']))
    pairs = []
    for index, first in enumerate(clusters):
        for second in clusters[index + 1:]:
            if first['red_id'] != second['red_id']:
                continue
            left, right = sorted((first, second), key=lambda row: row['quarter_xy'][0])
            separation = right['quarter_xy'][0] - left['quarter_xy'][0]
            if (PAIR_SEPARATION_RANGE[0] <= separation <= PAIR_SEPARATION_RANGE[1]
                    and abs(right['quarter_xy'][1] - left['quarter_xy'][1])
                    <= PAIR_MAX_Y_SEPARATION):
                score = (np.sqrt(left['triple_area_quarter_px'] *
                                 right['triple_area_quarter_px'])
                         * min(separation / 80., 2.))
                pairs.append({'left': left, 'right': right,
                              'quarter_separation_px': float(separation),
                              'image_pair_score': float(score)})
    return {'name': record['name'], 'camera': record['camera'],
            'frame_index': record['frame_index'],
            'triple_components': count - 1, 'qualified_candidates': len(candidates),
            'pair': max(pairs, key=lambda row: row['image_pair_score'])
            if pairs else None}


def _continuous_runs(rows):
    runs = []
    for camera in ('L', 'R'):
        sequence = sorted((row for row in rows if row['camera'] == camera),
                          key=lambda row: row['frame_index'])
        current = []
        for row in sequence:
            previous = current[-1] if current else None
            split = row['pair'] is None
            if previous is not None and not split:
                split |= row['frame_index'] != previous['frame_index'] + 1
                split |= any(np.linalg.norm(
                    np.asarray(row['pair'][side]['quarter_xy']) -
                    previous['pair'][side]['quarter_xy']) > MAX_INTERFRAME_MOTION
                    for side in ('left', 'right'))
            if split and current:
                if len(current) >= MIN_RUN_LENGTH:
                    runs.append({'camera': camera,
                                 'frames': [item['frame_index'] for item in current],
                                 'images': [item['name'] for item in current]})
                current = []
            if row['pair'] is not None:
                current.append(row)
        if len(current) >= MIN_RUN_LENGTH:
            runs.append({'camera': camera,
                         'frames': [item['frame_index'] for item in current],
                         'images': [item['name'] for item in current]})
    return runs


def run_court_paint_pair_scan(work):
    cv2.setNumThreads(4)
    root = Path(work).resolve() / 'global_sfm'
    scene = load_scene(root)
    output = root / OUTPUT
    output.mkdir(exist_ok=True)
    protocol = {'version': 1, 'scene_manifest_sha256': file_digest(root / 'manifest.json'),
                'source_size_and_mtime': {r['name']: [r['size_bytes'], r['mtime_ns']]
                                          for r in scene['images']},
                'read_reduction': 4, 'hsv_classes': 'Same court HSV classes as trajectory probe',
                'hsv_open_size': 3, 'triple_dilation': TRIPLE_DILATION,
                'triple_area_range': TRIPLE_AREA_RANGE,
                'red_component_min_area': RED_COMPONENT_MIN_AREA,
                'candidate_cluster_distance_quarter_px': CLUSTER_DISTANCE,
                'pair_separation_range_quarter_px': PAIR_SEPARATION_RANGE,
                'pair_max_y_separation_quarter_px': PAIR_MAX_Y_SEPARATION,
                'max_interframe_corner_motion_quarter_px': MAX_INTERFRAME_MOTION,
                'min_contiguous_run': MIN_RUN_LENGTH,
                'uses_pose_values_intrinsics_or_las_for_selection': False}
    fingerprint = _write_protocol(output / 'scan_protocol.json', protocol)
    result_path = output / 'scan.json'
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if result['protocol_fingerprint'] != fingerprint:
            raise ValueError('Changed court paint-pair scan inputs')
        return result
    rows = []
    for index, record in enumerate(scene['images'], 1):
        rows.append(_image_pair(record))
        if index % 100 == 0:
            print(f'court paint-pair scan {index}/{len(scene["images"])}', flush=True)
    runs = _continuous_runs(rows)
    result = {'protocol_fingerprint': fingerprint, 'rows': rows, 'runs': runs,
              'guard': 'Same red image region and temporal continuity are image-side proposals, not proof of one physical pair. Repeated court ends and adjacent red structures can still swap identities.'}
    write_json(result_path, result)
    return result


def _refined_segments(run, rows_by_name, records, camera):
    attempts = []
    current = []
    segments = []
    for frame, name in zip(run['frames'], run['images']):
        row = rows_by_name[name]
        image = cv2.imread(records[name]['source'])
        attempt = {'frame': frame, 'name': name}
        try:
            for side in ('left', 'right'):
                seed = row['pair'][side]['quarter_xy']
                detected = _color_junction_native(image, seed)
                xy = np.asarray(detected['native_xy'])
                detected['valid_160_fov'] = bool(camera.valid(xy[None], 160)[0])
                attempt[side] = detected
            mask = cv2.imread(records[name]['semantic_mask'], cv2.IMREAD_UNCHANGED)
            for side in ('left', 'right'):
                x, y = np.floor(attempt[side]['native_xy']).astype(int)
                attempt[side]['saved_mask_value'] = mask[y, x].tolist()
            attempt['status'] = 'located'
            current.append(attempt)
        except (ValueError, IndexError) as exc:
            attempt['status'] = str(exc)
            if len(current) >= MIN_RUN_LENGTH:
                segments.append(current)
            current = []
        attempts.append(attempt)
    if len(current) >= MIN_RUN_LENGTH:
        segments.append(current)
    return attempts, segments


def _corner_audit(segment, side, images, camera):
    first_index = (len(segment) - 1) // 4
    second_index = 3 * (len(segment) - 1) // 4
    support = (segment[first_index], segment[second_index])
    xy = {row['name']: np.asarray(row[side]['native_xy']) for row in segment}
    result = {'support_frames': [row['frame'] for row in support]}
    try:
        rays = [_world_ray(images[row['name']], camera, xy[row['name']])
                for row in support]
        world = _triangulate(*rays)
        result['support_ray_angle_degrees'] = float(np.rad2deg(np.arccos(
            np.clip(rays[0][1] @ rays[1][1], -1., 1.))))
        predictions = []
        for row in segment:
            predicted, depth = _project(images[row['name']], camera, world)
            predictions.append({'frame': row['frame'], 'camera_depth': depth,
                                'error_native_px': None if predicted is None else
                                float(np.linalg.norm(predicted - xy[row['name']]))})
        held = [entry['error_native_px'] for entry in predictions
                if entry['frame'] not in result['support_frames']
                and entry['error_native_px'] is not None]
        result['predictions'] = predictions
        result['holdout_count'] = len(held)
        result['holdout_median_native_px'] = float(np.median(held)) if held else None
        result['holdout_p90_native_px'] = float(np.percentile(held, 90)) if held else None
        result['holdout_max_native_px'] = float(np.max(held)) if held else None
    except (ValueError, np.linalg.LinAlgError) as exc:
        result['error'] = str(exc)
    coverage = []
    for row in segment:
        p2d = images[row['name']].points2D
        dist = np.linalg.norm(np.asarray([p.xy for p in p2d]) - xy[row['name']], axis=1)
        valid = np.asarray([p.has_point3D() for p in p2d], dtype=bool)
        coverage.append({'frame': row['frame'],
                         'nearest_feature_distance_native_px': float(np.min(dist)),
                         'nearest_registered_distance_native_px':
                         float(np.min(dist[valid])) if valid.any() else None,
                         'registered_within_16_native_px': bool(np.any(valid & (dist <= 16)))})
    result['existing_model_coverage'] = coverage
    result['no_registered_neighbor_fraction'] = float(np.mean(
        [not row['registered_within_16_native_px'] for row in coverage]))
    result['all_saved_mask_valid_and_160_fov'] = all(
        row[side]['saved_mask_value'] == 255 and row[side]['valid_160_fov']
        for row in segment)
    return result


def run_court_paint_pair_audit(work):
    work = Path(work).resolve()
    root = work / 'global_sfm'
    output = root / OUTPUT
    scan = run_court_paint_pair_scan(work)
    scene = load_scene(root)
    records = {r['name']: r for r in scene['images']}
    rows_by_name = {row['name']: row for row in scan['rows']}
    selected_names = {name for run in scan['runs'] for name in run['images']}
    model_path = root / 'ground_joint_ba/equal_control/sparse/0'
    protocol = {'version': 1, 'scan_sha256': file_digest(output / 'scan.json'),
                'selected_source_sha256': {name: file_digest(records[name]['source'])
                                           for name in sorted(selected_names)},
                'selected_mask_sha256': {name: file_digest(records[name]['semantic_mask'])
                                         for name in sorted(selected_names)},
                'visual_model_hashes': {p.name: file_digest(p)
                                        for p in sorted(model_path.glob('*.bin'))},
                'native_refiner': 'Same three-HSV-interior minimax method as native anchor probe v2, 120px ROI and 80px search',
                'support_choice': 'Indices floor((n-1)/4) and floor(3*(n-1)/4) of each image-selected contiguous refined segment',
                'coverage_radius_native_px': 16,
                'pose_use': 'Visual SfM pose and fitted candidate fisheye intrinsics used only after image-side pair selection',
                'uses_navigation_pose_values_or_las': False,
                'scope': 'Diagnostic ranking only; no SfM/LFS input changes.'}
    fingerprint = _write_protocol(output / 'audit_protocol.json', protocol)
    result_path = output / 'audit.json'
    if result_path.exists():
        old = json.loads(result_path.read_text())
        if old['protocol_fingerprint'] != fingerprint:
            raise ValueError('Changed court paint-pair audit inputs')
        return old
    reconstruction = pycolmap.Reconstruction(model_path)
    images = {image.name: image for image in reconstruction.images.values()}
    cameras = {label: Camera(label, reconstruction.cameras[camera_id].width,
                             reconstruction.cameras[camera_id].height,
                             tuple(reconstruction.cameras[camera_id].params), (0, 0))
               for label, camera_id in (('L', 1), ('R', 2))}
    audited = []
    for run in scan['runs']:
        attempts, segments = _refined_segments(
            run, rows_by_name, records, cameras[run['camera']])
        rows = []
        for segment in segments:
            row = {'camera': run['camera'],
                   'frames': [item['frame'] for item in segment],
                   'observations': segment}
            row['left'] = _corner_audit(segment, 'left', images, cameras[run['camera']])
            row['right'] = _corner_audit(segment, 'right', images, cameras[run['camera']])
            rows.append(row)
        audited.append({'camera': run['camera'], 'scan_frames': run['frames'],
                        'attempts': attempts, 'refined_segments': rows})
    result = {'protocol_fingerprint': fingerprint, 'audited_runs': audited,
              'guard': 'Image-side red components can change physical identity, and HSV border minima are not exact pixel truth. Holdout checks use candidate visual poses/intrinsics and must not be used to select matches. No candidate is promoted to BA.'}
    write_json(result_path, result)
    write_json(output / 'DIAGNOSTIC_ONLY.json', {
        'result_sha256': file_digest(result_path), 'status': 'DIAGNOSTIC_ONLY',
        'reason': 'All candidates require image review and independent validation before any SfM optimization.'})
    return result


def run_court_paint_pair_review(work):
    work = Path(work).resolve()
    root = work / 'global_sfm'
    output = root / OUTPUT
    audited = run_court_paint_pair_audit(work)
    protocol = {'version': 1, 'audit_sha256': file_digest(output / 'audit.json'),
                'selected_views': 'First, middle, last refined exposure per segment',
                'markers': 'Red left corner, blue right corner in native image coordinates',
                'scope': 'Visual review only'}
    fingerprint = _write_protocol(output / 'review_protocol.json', protocol)
    result_path = output / 'review.json'
    if result_path.exists():
        old = json.loads(result_path.read_text())
        if old['protocol_fingerprint'] != fingerprint:
            raise ValueError('Changed court paint-pair review inputs')
        return old
    records = {r['name']: r for r in load_scene(root)['images']}
    review = output / 'review'
    review.mkdir(exist_ok=True)
    groups = [segment for run in audited['audited_runs']
              for segment in run['refined_segments']]
    sheets = {}
    for camera in ('L', 'R'):
        selected = [group for group in groups if group['camera'] == camera]
        if not selected:
            continue
        canvas = np.full((len(selected) * 270, 3 * 385, 3), 24, np.uint8)
        for row_index, segment in enumerate(selected):
            observations = segment['observations']
            for column, index in enumerate((0, len(observations)//2, len(observations)-1)):
                observation = observations[index]
                image = cv2.imread(records[observation['name']]['source'],
                                   cv2.IMREAD_REDUCED_COLOR_4)
                for side, color in (('left', (0, 0, 255)), ('right', (255, 0, 0))):
                    xy = np.rint(np.asarray(observation[side]['native_xy']) / 4).astype(int)
                    cv2.circle(image, tuple(xy), 8, color, 2)
                tile = cv2.resize(image[350:800], (375, 225),
                                  interpolation=cv2.INTER_AREA)
                x0, y0 = column * 385, row_index * 270
                canvas[y0:y0+225, x0:x0+375] = tile
                cv2.putText(canvas, f'{camera}{observation["frame"]}',
                            (x0+5, y0+250), cv2.FONT_HERSHEY_SIMPLEX,
                            .65, (245, 245, 245), 2)
        path = review / f'{camera}_segments_contact.jpg'
        cv2.imwrite(str(path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 91])
        sheets[camera] = {'path': str(path), 'sha256': file_digest(path),
                          'segments': len(selected)}
    crops = {}
    for segment in groups:
        first, last = segment['frames'][0], segment['frames'][-1]
        name = f'{segment["camera"]}{first}_{last}'
        observations = segment['observations']
        canvas = np.full((3 * 288, 2 * 288, 3), 24, np.uint8)
        for row_index, index in enumerate((0, len(observations)//2, len(observations)-1)):
            observation = observations[index]
            image = cv2.imread(records[observation['name']]['source'])
            padded = cv2.copyMakeBorder(image, 128, 128, 128, 128,
                                        cv2.BORDER_CONSTANT)
            for column, side in enumerate(('left', 'right')):
                x, y = np.rint(observation[side]['native_xy']).astype(int)
                crop = padded[y:y+256, x:x+256].copy()
                cv2.drawMarker(crop, (128, 128), (0, 0, 255) if column == 0
                               else (255, 0, 0), cv2.MARKER_CROSS, 14, 1)
                x0, y0 = column * 288, row_index * 288
                canvas[y0:y0+256, x0:x0+256] = crop
                cv2.putText(canvas, f'{segment["camera"]}{observation["frame"]} {side}',
                            (x0+5, y0+278), cv2.FONT_HERSHEY_SIMPLEX,
                            .52, (245, 245, 245), 1)
        path = review / f'{name}_native_crops.jpg'
        cv2.imwrite(str(path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 94])
        crops[name] = {'path': str(path), 'sha256': file_digest(path)}
    result = {'protocol_fingerprint': fingerprint, 'contact_sheets': sheets,
              'native_crops': crops}
    write_json(result_path, result)
    return result


def _threshold_variants(bgr, canonical_xy):
    x, y = np.rint(canonical_xy).astype(int)
    if x < 100 or y < 100 or x + 100 >= bgr.shape[1] or y + 100 >= bgr.shape[0]:
        return {'status': '200px local window meets image border'}
    roi = bgr[y-100:y+100, x-100:x+100]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    hue_classes = (((hue <= 10) | (hue >= 170)),
                   ((hue >= 74) & (hue <= 95)),
                   ((hue >= 98) & (hue <= 115)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    variants = []
    for min_saturation in (60, 80, 100):
        for min_value in (50, 60, 80):
            chromatic = (saturation >= min_saturation) & (value >= min_value)
            masks = [cv2.morphologyEx((chromatic & class_mask).astype(np.uint8),
                                      cv2.MORPH_OPEN, kernel)
                     for class_mask in hue_classes]
            row = {'min_saturation': min_saturation, 'min_value': min_value}
            if min(int(mask.sum()) for mask in masks) < 200:
                row['status'] = 'insufficient color interior'
                variants.append(row)
                continue
            distance = np.stack([cv2.distanceTransform((mask == 0).astype(np.uint8),
                                                        cv2.DIST_L2, 3)
                                 for mask in masks])
            local = np.max(distance, axis=0)[75:125, 75:125]
            index_y, index_x = np.unravel_index(np.argmin(local), local.shape)
            if index_x in (0, 49) or index_y in (0, 49):
                row['status'] = 'minimum at 50px local search boundary'
            else:
                row['status'] = 'located'
                row['native_xy'] = [float(x + index_x - 25 + .5),
                                    float(y + index_y - 25 + .5)]
            variants.append(row)
    baseline = next(row for row in variants
                    if row['min_saturation'] == 80 and row['min_value'] == 60)
    valid = [row for row in variants if row['status'] == 'located']
    result = {'status': 'located' if baseline['status'] == 'located' else baseline['status'],
              'variants': variants, 'located_count': len(valid)}
    if baseline['status'] == 'located':
        xy = np.asarray(baseline['native_xy'])
        displacement = [float(np.linalg.norm(np.asarray(row['native_xy']) - xy))
                        for row in valid]
        result.update({'local_baseline_native_xy': baseline['native_xy'],
                       'baseline_vs_frozen_native_px':
                       float(np.linalg.norm(xy - canonical_xy)),
                       'variant_median_shift_native_px':
                       float(np.median(displacement)),
                       'variant_max_shift_native_px':
                       float(np.max(displacement))})
    return result


def run_court_paint_pair_stability(work):
    """Measure HSV-threshold sensitivity without changing saved corner tracks."""
    work = Path(work).resolve()
    root = work / 'global_sfm'
    output = root / OUTPUT
    audit = run_court_paint_pair_audit(work)
    protocol = {'version': 1, 'audit_sha256': file_digest(output / 'audit.json'),
                'min_saturation': [60, 80, 100], 'min_value': [50, 60, 80],
                'hues': 'Original fixed red/green/blue ranges',
                'open_kernel': '3px ellipse', 'local_roi_size_native_px': 200,
                'local_search_size_native_px': 50,
                'baseline': 'S>=80,V>=60 in the same fixed 200px local window',
                'scope': 'Detector sensitivity only; does not measure true physical point error.'}
    fingerprint = _write_protocol(output / 'stability_protocol.json', protocol)
    result_path = output / 'stability.json'
    if result_path.exists():
        old = json.loads(result_path.read_text())
        if old['protocol_fingerprint'] != fingerprint:
            raise ValueError('Changed court paint-pair stability inputs')
        return old
    records = {r['name']: r for r in load_scene(root)['images']}
    segments = []
    for run in audit['audited_runs']:
        for segment in run['refined_segments']:
            observations = []
            for observation in segment['observations']:
                image = cv2.imread(records[observation['name']]['source'])
                row = {'frame': observation['frame']}
                for side in ('left', 'right'):
                    row[side] = _threshold_variants(
                        image, np.asarray(observation[side]['native_xy']))
                observations.append(row)
            segments.append({'camera': segment['camera'], 'frames': segment['frames'],
                             'observations': observations})
    result = {'protocol_fingerprint': fingerprint, 'segments': segments,
              'guard': 'Small HSV-threshold spread cannot prove pixel truth or physical identity. Large spread makes high weighting especially unsafe.'}
    write_json(result_path, result)
    return result
