"""Image-only white-paint junction refinement of frozen court-color proposals.

The color corner is only a search center. Pose and candidate fisheye intrinsics
are read after line fitting, for a fixed-support holdout diagnostic. A white
line intersection does not establish physical identity across exposures.
"""

from collections import Counter
import json
from pathlib import Path

import cv2
import numpy as np
import pycolmap

from .camera import Camera
from .scene import file_digest, load_scene, write_json
from .sfm_court_anchor_probe import _triangulate
from .sfm_court_native_anchor_probe import _project, _world_ray
from .sfm_court_paint_pair_probe import _write_protocol
from .sfm_court_trajectory_probe import _hue_masks


OUTPUT = 'sfm_court_white_line_probe'
PAIRS = {'red_green': (0, 1), 'red_blue': (0, 2), 'green_blue': (1, 2)}
HALF = 128
WHITE_MAX_S = 85
WHITE_MIN_V = 125
COLOR_DILATION = 17
RANSAC_TRIALS = 500
RANSAC_RESIDUAL = 2.5
MIN_INLIERS = 35
MIN_EXTENT = 40.
MIN_CROSS_ANGLE = 15.
MAX_CROSS_SPREAD = 8.
MAX_SEED_SHIFT = 70.
MAX_NEAREST_SUPPORT_DISTANCE = 45.


def _fit_line(mask, center, seed):
    ys, xs = np.where(mask)
    points = np.column_stack((xs, ys)).astype(float)
    distance = np.linalg.norm(points - center, axis=1)
    points = points[(distance >= 15) & (distance <= 108)]
    if len(points) < MIN_INLIERS:
        return None
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(RANSAC_TRIALS):
        first, second = rng.choice(len(points), 2, replace=False)
        delta = points[second] - points[first]
        length = np.linalg.norm(delta)
        if length < 25:
            continue
        tangent = delta / length
        normal = np.array([-tangent[1], tangent[0]])
        if abs(normal @ (center - points[first])) > 35:
            continue
        inliers = np.abs((points - points[first]) @ normal) < RANSAC_RESIDUAL
        count = int(inliers.sum())
        if count < MIN_INLIERS:
            continue
        extent = float(np.ptp(points[inliers] @ tangent))
        score = (count, extent)
        if best is None or score > best[0]:
            best = (score, inliers)
    if best is None:
        return None
    support = points[best[1]]
    centroid = support.mean(axis=0)
    _, _, vectors = np.linalg.svd(support - centroid, full_matrices=False)
    tangent = vectors[0]
    normal = np.array([-tangent[1], tangent[0]])
    residual = np.abs((support - centroid) @ normal)
    return {'normal': normal, 'constant': float(normal @ centroid),
            'support': support, 'inliers': len(support), 'all_pixels': len(points),
            'median_residual_px': float(np.median(residual)),
            'p90_residual_px': float(np.percentile(residual, 90)),
            'extent_px': float(np.ptp(support @ tangent))}


def _angle_degrees(first, second):
    cosine = abs(float(first['normal'] @ second['normal']))
    return float(np.rad2deg(np.arccos(np.clip(cosine, 0., 1.))))


def _fit_corner(image, seed_xy, seed, white_max_s=WHITE_MAX_S,
                white_min_v=WHITE_MIN_V, color_dilation=COLOR_DILATION):
    x, y = np.rint(seed_xy).astype(int)
    roi = image[y-HALF:y+HALF, x-HALF:x+HALF]
    if roi.shape[:2] != (2*HALF, 2*HALF):
        return {'status': 'crop_meets_image_border'}
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    white = (hsv[:, :, 1] < white_max_s) & (hsv[:, :, 2] > white_min_v)
    color = _hue_masks(roi, 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (color_dilation, color_dilation))
    near = [cv2.dilate(mask, kernel) > 0 for mask in color]
    center = np.array([float(HALF), float(HALF)])
    lines = {}
    for index, (name, (first, second)) in enumerate(PAIRS.items()):
        line = _fit_line(white & near[first] & near[second], center, seed+index)
        if line is not None:
            lines[name] = line
    public = {name: {key: value.tolist() if isinstance(value, np.ndarray)
                     else value for key, value in line.items() if key != 'support'}
              for name, line in lines.items()}
    result = {'status': 'unclassified', 'lines': public}
    if 'red_green' not in lines or not any(
            name in lines for name in ('red_blue', 'green_blue')):
        result['status'] = 'missing_independent_color_boundary'
        return result
    crossing = {name: _angle_degrees(lines['red_green'], lines[name])
                for name in ('red_blue', 'green_blue') if name in lines}
    result['red_green_cross_angles_deg'] = crossing
    usable = ['red_green'] + [name for name, angle in crossing.items()
                              if angle >= MIN_CROSS_ANGLE]
    if len(usable) < 2:
        result['status'] = 'small_cross_angle'
        return result
    # Two blue-adjacent masks often describe the same straight paint arm.
    # Their near-parallel intersection is intentionally excluded.
    if any(lines[name]['extent_px'] < MIN_EXTENT for name in usable):
        result['status'] = 'short_line_support'
        return result
    normals = np.array([lines[name]['normal'] for name in usable])
    constants = np.array([lines[name]['constant'] for name in usable])
    estimate = np.linalg.lstsq(normals, constants, rcond=None)[0]
    crossings = []
    for name in usable[1:]:
        pair = np.array([lines['red_green']['normal'], lines[name]['normal']])
        rhs = np.array([lines['red_green']['constant'], lines[name]['constant']])
        crossings.append(np.linalg.solve(pair, rhs))
    spread = max(float(np.linalg.norm(point-estimate)) for point in crossings)
    shift = float(np.linalg.norm(estimate-center))
    nearest = {name: float(np.min(np.linalg.norm(lines[name]['support']-estimate,
                                                 axis=1))) for name in usable}
    result.update({'used_lines': usable, 'local_xy': estimate.tolist(),
                   'native_xy': (estimate + [x-HALF, y-HALF]).tolist(),
                   'shift_from_hsv_native_px': shift,
                   'cross_spread_native_px': spread,
                   'nearest_support_native_px': nearest})
    if spread > MAX_CROSS_SPREAD:
        result['status'] = 'inconsistent_line_crossings'
    elif shift > MAX_SEED_SHIFT:
        result['status'] = 'junction_far_from_color_seed'
    elif max(nearest.values()) > MAX_NEAREST_SUPPORT_DISTANCE:
        result['status'] = 'line_extrapolation_too_far'
    else:
        result['status'] = 'image_geometry_pass'
    return result


def _fixed_support_holdout(segment, side, observations, images, camera):
    support_frames = segment[side]['support_frames']
    located = {row['frame']: row for row in observations
               if row[side]['status'] == 'image_geometry_pass'
               and row[side]['saved_mask_valid'] and row[side]['valid_160_fov']}
    result = {'support_frames': support_frames, 'valid_frames': sorted(located)}
    if not all(frame in located for frame in support_frames):
        result['status'] = 'frozen_support_not_both_valid'
        return result
    support = [located[frame] for frame in support_frames]
    try:
        rays = [_world_ray(images[row['name']], camera,
                           np.asarray(row[side]['native_xy'])) for row in support]
        world = _triangulate(*rays)
        result['support_ray_angle_degrees'] = float(np.rad2deg(np.arccos(
            np.clip(rays[0][1] @ rays[1][1], -1., 1.))))
        predictions = []
        for frame, row in sorted(located.items()):
            prediction, depth = _project(images[row['name']], camera, world)
            predictions.append({'frame': frame, 'camera_depth': depth,
                                'error_native_px': None if prediction is None else
                                float(np.linalg.norm(prediction-row[side]['native_xy']))})
        baseline_by_frame = {row['frame']: row['error_native_px']
                             for row in segment[side]['predictions']}
        held = [row for row in predictions if row['frame'] not in support_frames
                and row['error_native_px'] is not None
                and baseline_by_frame.get(row['frame']) is not None]
        result['status'] = 'holdout_available' if held else 'no_common_holdout'
        result['predictions'] = predictions
        result['common_holdout_frames'] = [row['frame'] for row in held]
        for label, values in (
            ('white_line', [row['error_native_px'] for row in held]),
            ('hsv', [baseline_by_frame[row['frame']] for row in held])):
            result[f'{label}_median_native_px'] = float(np.median(values)) if values else None
            result[f'{label}_p90_native_px'] = float(np.percentile(values, 90)) if values else None
            result[f'{label}_max_native_px'] = float(np.max(values)) if values else None
    except (ValueError, np.linalg.LinAlgError) as exc:
        result['status'] = 'holdout_failed'
        result['error'] = str(exc)
    return result


def _render_pages(review, segment, observations, records):
    paths = []
    label = f'{segment["camera"]}{segment["frames"][0]}_{segment["frames"][-1]}'
    for page, start in enumerate(range(0, len(observations), 5), 1):
        selected = observations[start:start+5]
        canvas = np.full((len(selected)*285, 2*285, 3), 24, np.uint8)
        for row_index, row in enumerate(selected):
            image = cv2.imread(records[row['name']]['source'])
            for column, side in enumerate(('left', 'right')):
                record = row[side]
                x, y = np.rint(row[side]['hsv_native_xy']).astype(int)
                tile = image[y-HALF:y+HALF, x-HALF:x+HALF].copy()
                if tile.shape[:2] != (256, 256):
                    continue
                for name, line in record.get('lines', {}).items():
                    normal = np.asarray(line['normal'])
                    direction = np.array([normal[1], -normal[0]])
                    midpoint = normal * line['constant']
                    a = tuple(np.rint(midpoint - 130*direction).astype(int))
                    b = tuple(np.rint(midpoint + 130*direction).astype(int))
                    color = {'red_green': (0, 255, 255), 'red_blue': (255, 0, 255),
                             'green_blue': (255, 255, 0)}[name]
                    cv2.line(tile, a, b, color, 1)
                cv2.drawMarker(tile, (128, 128), (0, 0, 255), cv2.MARKER_CROSS, 16, 2)
                if 'local_xy' in record:
                    target = tuple(np.rint(record['local_xy']).astype(int))
                    cv2.drawMarker(tile, target, (0, 255, 255), cv2.MARKER_CROSS, 16, 2)
                x0, y0 = column*285, row_index*285
                canvas[y0:y0+256, x0:x0+256] = tile
                cv2.putText(canvas, f'{segment["camera"]}{row["frame"]} {side} {record["status"]}',
                            (x0+2, y0+276), cv2.FONT_HERSHEY_SIMPLEX,
                            .42, (245, 245, 245), 1)
        path = review / f'{label}_page_{page:02d}.jpg'
        cv2.imwrite(str(path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 94])
        paths.append({'path': str(path), 'sha256': file_digest(path)})
    return paths


def run_court_white_line_probe(work):
    cv2.setNumThreads(4)
    root = Path(work).resolve() / 'global_sfm'
    parent = root / 'sfm_court_paint_pair_probe'
    audit_path = parent / 'audit.json'
    if not audit_path.exists():
        raise FileNotFoundError(f'Run the paint-pair audit first: {audit_path}')
    audit = json.loads(audit_path.read_text())
    scene = load_scene(root)
    records = {row['name']: row for row in scene['images']}
    segments = [segment for run in audit['audited_runs']
                for segment in run['refined_segments']]
    names = sorted({row['name'] for segment in segments
                    for row in segment['observations']})
    model_path = root / 'ground_joint_ba/equal_control/sparse/0'
    output = root / OUTPUT
    output.mkdir(exist_ok=True)
    protocol = {
        'version': 1, 'paint_pair_audit_sha256': file_digest(audit_path),
        'selected_source_sha256': {name: file_digest(records[name]['source']) for name in names},
        'selected_mask_sha256': {name: file_digest(records[name]['semantic_mask']) for name in names},
        'visual_model_hashes': {p.name: file_digest(p) for p in sorted(model_path.glob('*.bin'))},
        'native_roi_px': 256, 'white_hsv': {'saturation_below': WHITE_MAX_S,
                                           'value_above': WHITE_MIN_V},
        'color_classes': 'Fixed red/green/blue HSV classes in court trajectory probe',
        'color_open_px': 3, 'color_dilation_px': COLOR_DILATION,
        'pairwise_white_masks': list(PAIRS), 'fit_annulus_px': [15, 108],
        'ransac': {'trials': RANSAC_TRIALS, 'min_pair_spacing_px': 25,
                   'max_seed_to_line_px': 35, 'inlier_residual_px': RANSAC_RESIDUAL,
                   'min_inliers': MIN_INLIERS, 'score': 'inlier count then extent',
                   'seed': 'frame_index*10 + side_index*3 + pair_index'},
        'image_quality': {'min_line_extent_px': MIN_EXTENT,
                          'min_red_green_to_blue_angle_deg': MIN_CROSS_ANGLE,
                          'max_crossing_spread_px': MAX_CROSS_SPREAD,
                          'max_hsv_seed_shift_px': MAX_SEED_SHIFT,
                          'max_nearest_line_support_px': MAX_NEAREST_SUPPORT_DISTANCE},
        'holdout': 'Use unchanged 1/4 and 3/4 support frames from paint-pair audit; compare only common image-valid held-out frames',
        'uses_navigation_pose_values_or_las': False,
        'pose_and_intrinsics_use': 'Saved visual SfM model only after image-line selection; candidate calibration and pose are not truth',
        'scope': 'Diagnostic only; no new SfM tracks, BA, COLMAP, or LFS input changes.'}
    fingerprint = _write_protocol(output / 'protocol.json', protocol)
    result_path = output / 'results.json'
    if result_path.exists():
        old = json.loads(result_path.read_text())
        if old['protocol_fingerprint'] != fingerprint:
            raise ValueError('Changed court white-line probe inputs')
        return old
    reconstruction = pycolmap.Reconstruction(model_path)
    images = {image.name: image for image in reconstruction.images.values()}
    cameras = {label: Camera(label, reconstruction.cameras[camera_id].width,
                             reconstruction.cameras[camera_id].height,
                             tuple(reconstruction.cameras[camera_id].params), (0, 0))
               for label, camera_id in (('L', 1), ('R', 2))}
    review = output / 'review'
    review.mkdir(exist_ok=True)
    rows = []
    for segment in segments:
        observations = []
        for source in segment['observations']:
            image = cv2.imread(records[source['name']]['source'])
            mask = cv2.imread(records[source['name']]['semantic_mask'], cv2.IMREAD_UNCHANGED)
            row = {'frame': source['frame'], 'name': source['name']}
            for side_index, side in enumerate(('left', 'right')):
                xy = source[side]['native_xy']
                fitted = _fit_corner(image, xy, source['frame']*10+side_index*3)
                fitted['hsv_native_xy'] = xy
                if 'native_xy' in fitted:
                    target = np.asarray(fitted['native_xy'])
                    px, py = np.floor(target).astype(int)
                    fitted['saved_mask_valid'] = bool(mask[py, px] == 255)
                    fitted['valid_160_fov'] = bool(cameras[segment['camera']].valid(
                        target[None], 160)[0])
                row[side] = fitted
            observations.append(row)
        row = {'camera': segment['camera'], 'frames': segment['frames'],
               'observations': observations}
        for side in ('left', 'right'):
            row[side] = _fixed_support_holdout(segment, side, observations, images,
                                               cameras[segment['camera']])
        row['review_pages'] = _render_pages(review, segment, observations, records)
        rows.append(row)
        print(f'white-line probe {row["camera"]}{row["frames"][0]}–{row["frames"][-1]}',
              flush=True)
    result = {'protocol_fingerprint': fingerprint, 'segments': rows,
              'guard': 'White paint fitted in one image can be a different or partly occluded physical corner in another. A lower saved-model holdout error does not validate the old pose or prove SfM improvement. No constraint promoted.'}
    write_json(result_path, result)
    write_json(output / 'DIAGNOSTIC_ONLY.json', {
        'result_sha256': file_digest(result_path), 'status': 'DIAGNOSTIC_ONLY',
        'reason': 'Image-only line geometry and fixed-model holdouts cannot alone establish physical identity or justify BA.'})
    return result


def run_court_white_line_stability(work):
    """Perturb only image segmentation, without consulting pose residuals."""
    cv2.setNumThreads(4)
    root = Path(work).resolve() / 'global_sfm'
    output = root / OUTPUT
    result_path = output / 'results.json'
    if not result_path.exists():
        raise FileNotFoundError(f'Run the white-line probe first: {result_path}')
    baseline = json.loads(result_path.read_text())
    selected = [segment for segment in baseline['segments']
                if (segment['camera'], segment['frames'][0])
                in (('L', 233), ('R', 267))]
    if len(selected) != 2:
        raise ValueError('Expected the frozen L233 and R267 segments')
    variants = [
        {'name': 'white_strict', 'white_max_s': 70, 'white_min_v': 145,
         'color_dilation': 17},
        {'name': 'white_loose', 'white_max_s': 100, 'white_min_v': 105,
         'color_dilation': 17},
        {'name': 'color_near_13', 'white_max_s': 85, 'white_min_v': 125,
         'color_dilation': 13},
        {'name': 'color_near_21', 'white_max_s': 85, 'white_min_v': 125,
         'color_dilation': 21}]
    protocol = {'version': 1, 'baseline_results_sha256': file_digest(result_path),
                'selected_segments': ['L233-243', 'R267-281'],
                'variants': variants,
                'comparison': 'Native-pixel Euclidean shift from baseline white-line intersection when both pass image-only quality gates',
                'uses_pose_values_intrinsics_or_las': False,
                'scope': 'Detector sensitivity; not accuracy or physical identity.'}
    fingerprint = _write_protocol(output / 'stability_protocol.json', protocol)
    stability_path = output / 'stability.json'
    if stability_path.exists():
        old = json.loads(stability_path.read_text())
        if old['protocol_fingerprint'] != fingerprint:
            raise ValueError('Changed court white-line stability inputs')
        return old
    records = {row['name']: row for row in load_scene(root)['images']}
    segments = []
    for segment in selected:
        observations = []
        for source in segment['observations']:
            image = cv2.imread(records[source['name']]['source'])
            row = {'frame': source['frame']}
            for side_index, side in enumerate(('left', 'right')):
                original = source[side]
                attempts = []
                for variant in variants:
                    fitted = _fit_corner(image, original['hsv_native_xy'],
                                         source['frame']*10+side_index*3,
                                         variant['white_max_s'], variant['white_min_v'],
                                         variant['color_dilation'])
                    attempt = {'name': variant['name'], 'status': fitted['status']}
                    if 'native_xy' in fitted:
                        attempt['native_xy'] = fitted['native_xy']
                        attempt['shift_from_baseline_native_px'] = float(
                            np.linalg.norm(np.asarray(fitted['native_xy']) -
                                           original['native_xy'])) if 'native_xy' in original else None
                    attempts.append(attempt)
                row[side] = {'baseline_status': original['status'], 'variants': attempts}
            observations.append(row)
        summary = {}
        for side in ('left', 'right'):
            summary[side] = {}
            for variant in variants:
                selected_attempts = [next(item for item in row[side]['variants']
                                          if item['name'] == variant['name'])
                                     for row in observations]
                comparable = [item['shift_from_baseline_native_px']
                              for source, item in zip(segment['observations'], selected_attempts)
                              if source[side]['status'] == 'image_geometry_pass'
                              and item['status'] == 'image_geometry_pass']
                summary[side][variant['name']] = {
                    'variant_pass_count': sum(item['status'] == 'image_geometry_pass'
                                              for item in selected_attempts),
                    'comparable_count': len(comparable),
                    'median_shift_native_px': float(np.median(comparable)) if comparable else None,
                    'max_shift_native_px': float(np.max(comparable)) if comparable else None}
        segments.append({'camera': segment['camera'], 'frames': segment['frames'],
                         'observations': observations, 'summary': summary})
        print(f'white-line stability {segment["camera"]}{segment["frames"][0]}',
              flush=True)
    result = {'protocol_fingerprint': fingerprint, 'segments': segments,
              'guard': 'Small threshold sensitivity does not prove localization truth. A variant crossing a gate changes which observations can be compared.'}
    write_json(stability_path, result)
    return result


def run_court_white_line_coverage(work):
    """Find already mapped features near white junctions and hold out their views."""
    root = Path(work).resolve() / 'global_sfm'
    output = root / OUTPUT
    results_path = output / 'results.json'
    if not results_path.exists():
        raise FileNotFoundError(f'Run the white-line probe first: {results_path}')
    results = json.loads(results_path.read_text())
    model_path = root / 'ground_joint_ba/equal_control/sparse/0'
    protocol = {
        'version': 1, 'white_line_results_sha256': file_digest(results_path),
        'visual_model_hashes': {p.name: file_digest(p) for p in sorted(model_path.glob('*.bin'))},
        'neighbor_radius_native_px': 16,
        'R267_point_selection': 'Most frequent nearest registered 3D point within 16px among image-geometry-pass junctions, separately per side; tie by smallest point ID',
        'point_holdout': 'Project each selected existing 3D point into R267-281 images where that exact point has no registered observation; compare against image-only white junction',
        'uses_navigation_pose_values_or_las': False,
        'scope': 'Existing-model coverage and projection diagnostic; not BA or geometry truth.'}
    fingerprint = _write_protocol(output / 'coverage_protocol.json', protocol)
    result_path = output / 'coverage.json'
    if result_path.exists():
        old = json.loads(result_path.read_text())
        if old['protocol_fingerprint'] != fingerprint:
            raise ValueError('Changed court white-line coverage inputs')
        return old
    reconstruction = pycolmap.Reconstruction(model_path)
    images = {image.name: image for image in reconstruction.images.values()}
    camera_data = reconstruction.cameras[2]
    camera = Camera('R', camera_data.width, camera_data.height,
                    tuple(camera_data.params), (0, 0))
    segment_rows = []
    for segment in results['segments']:
        observations = []
        for source in segment['observations']:
            image = images[source['name']]
            points = image.points2D
            coords = np.asarray([point.xy for point in points])
            mapped = np.asarray([point.has_point3D() for point in points])
            row = {'frame': source['frame'], 'name': source['name']}
            for side in ('left', 'right'):
                proposed = source[side]
                if proposed['status'] != 'image_geometry_pass':
                    row[side] = {'status': proposed['status']}
                    continue
                distance = np.linalg.norm(coords-np.asarray(proposed['native_xy']), axis=1)
                nearest = int(np.argmin(distance))
                registered = np.flatnonzero(mapped)
                nearest_mapped = int(registered[np.argmin(distance[registered])])
                point = points[nearest_mapped]
                row[side] = {
                    'status': 'located',
                    'nearest_feature_index': nearest,
                    'nearest_feature_distance_native_px': float(distance[nearest]),
                    'nearest_registered_feature_index': nearest_mapped,
                    'nearest_registered_distance_native_px': float(distance[nearest_mapped]),
                    'nearest_registered_point3D_id': int(point.point3D_id),
                    'registered_within_16_native_px': bool(distance[nearest_mapped] <= 16)}
            observations.append(row)
        summary = {}
        for side in ('left', 'right'):
            located = [row[side] for row in observations if row[side]['status'] == 'located']
            summary[side] = {'located_count': len(located),
                             'registered_within_16_count': sum(
                                 row['registered_within_16_native_px'] for row in located)}
        segment_rows.append({'camera': segment['camera'], 'frames': segment['frames'],
                             'observations': observations, 'summary': summary})
    target = next((row for row in segment_rows
                   if row['camera'] == 'R' and row['frames'][0] == 267), None)
    target_source = next((row for row in results['segments']
                          if row['camera'] == 'R' and row['frames'][0] == 267), None)
    if target is None or target_source is None:
        raise ValueError('Missing frozen R267-281 segment')
    existing_points = {}
    for side in ('left', 'right'):
        counts = Counter(row[side]['nearest_registered_point3D_id']
                         for row in target['observations']
                         if row[side].get('registered_within_16_native_px'))
        if not counts:
            existing_points[side] = {'status': 'no_nearby_mapped_point'}
            continue
        point_id = min(counts, key=lambda point: (-counts[point], point))
        point = reconstruction.points3D[point_id]
        registered_image_ids = {item.image_id for item in point.track.elements}
        predictions = []
        for source in target_source['observations']:
            if source[side]['status'] != 'image_geometry_pass':
                continue
            if not source[side]['saved_mask_valid'] or not source[side]['valid_160_fov']:
                continue
            image = images[source['name']]
            projected, depth = _project(image, camera, point.xyz)
            if projected is None:
                continue
            residual = projected-np.asarray(source[side]['native_xy'])
            predictions.append({'frame': source['frame'], 'name': source['name'],
                                'in_existing_track': image.image_id in registered_image_ids,
                                'predicted_native_xy': projected.tolist(),
                                'camera_depth': depth,
                                'white_line_residual_xy_native_px': residual.tolist(),
                                'white_line_error_native_px': float(np.linalg.norm(residual))})
        heldout = [row['white_line_error_native_px'] for row in predictions
                   if not row['in_existing_track']]
        existing_points[side] = {
            'status': 'projected', 'point3D_id': point_id,
            'nearby_image_count': counts[point_id],
            'point_track_length': len(point.track.elements),
            'point_reprojection_error_native_px': float(point.error),
            'predictions': predictions, 'holdout_count': len(heldout),
            'holdout_median_native_px': float(np.median(heldout)) if heldout else None,
            'holdout_p90_native_px': float(np.percentile(heldout, 90)) if heldout else None,
            'holdout_max_native_px': float(np.max(heldout)) if heldout else None}
    result = {'protocol_fingerprint': fingerprint, 'segments': segment_rows,
              'R267_existing_points': existing_points,
              'guard': 'Nearby mapped features may occupy the same paint border without being the exact white-line junction. Their 3D positions and camera poses come from the saved SfM model, not ground truth.'}
    write_json(result_path, result)
    return result


def run_court_white_line_model_review(work):
    """Show candidate color point, white junction, and old 3D projection."""
    root = Path(work).resolve() / 'global_sfm'
    output = root / OUTPUT
    results_path, coverage_path = output / 'results.json', output / 'coverage.json'
    if not results_path.exists() or not coverage_path.exists():
        raise FileNotFoundError('Run white-line geometry and coverage before model review')
    protocol = {'version': 1, 'results_sha256': file_digest(results_path),
                'coverage_sha256': file_digest(coverage_path),
                'selected_segment': 'R267-281',
                'markers': {'red': 'original HSV point', 'yellow': 'white-line fit',
                            'cyan': 'saved visual-model 3D point projection'},
                'scope': 'Visual review only; all displayed model projections are candidates, not truth.'}
    fingerprint = _write_protocol(output / 'model_review_protocol.json', protocol)
    result_path = output / 'model_review.json'
    if result_path.exists():
        old = json.loads(result_path.read_text())
        if old['protocol_fingerprint'] != fingerprint:
            raise ValueError('Changed white-line model review inputs')
        return old
    results = json.loads(results_path.read_text())
    coverage = json.loads(coverage_path.read_text())
    segment = next(row for row in results['segments']
                   if row['camera'] == 'R' and row['frames'][0] == 267)
    records = {row['name']: row for row in load_scene(root)['images']}
    review = output / 'review'
    review.mkdir(exist_ok=True)
    pages = {}
    for side in ('left', 'right'):
        predicted = {row['frame']: row
                     for row in coverage['R267_existing_points'][side]['predictions']}
        paths = []
        for page, start in enumerate(range(0, len(segment['observations']), 5), 1):
            selected = segment['observations'][start:start+5]
            canvas = np.full((len(selected)*285, 285, 3), 24, np.uint8)
            for index, row in enumerate(selected):
                observation = row[side]
                x, y = np.rint(observation['hsv_native_xy']).astype(int)
                image = cv2.imread(records[row['name']]['source'])
                tile = image[y-HALF:y+HALF, x-HALF:x+HALF].copy()
                if tile.shape[:2] != (256, 256):
                    continue
                origin = np.array([x-HALF, y-HALF])
                cv2.drawMarker(tile, (128, 128), (0, 0, 255), cv2.MARKER_CROSS, 15, 2)
                if 'native_xy' in observation:
                    point = tuple(np.rint(np.asarray(observation['native_xy'])-origin).astype(int))
                    cv2.drawMarker(tile, point, (0, 255, 255), cv2.MARKER_CROSS, 15, 2)
                model = predicted.get(row['frame'])
                if model is not None:
                    point = tuple(np.rint(np.asarray(model['predicted_native_xy'])-origin).astype(int))
                    cv2.drawMarker(tile, point, (255, 255, 0), cv2.MARKER_CROSS, 15, 2)
                y0 = index*285
                canvas[y0:y0+256, :256] = tile
                error = (f'{model["white_line_error_native_px"]:.1f}px'
                         if model is not None else observation['status'])
                cv2.putText(canvas, f'R{row["frame"]} {side} {error}',
                            (3, y0+276), cv2.FONT_HERSHEY_SIMPLEX,
                            .52, (245, 245, 245), 1)
            path = review / f'R267_281_model_{side}_{page:02d}.jpg'
            cv2.imwrite(str(path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 94])
            paths.append({'path': str(path), 'sha256': file_digest(path)})
        pages[side] = paths
    result = {'protocol_fingerprint': fingerprint, 'pages': pages}
    write_json(result_path, result)
    return result


def run_court_white_line_pair_mask_review(work):
    """Expose the color-pair white support near the shallow R right junction."""
    root = Path(work).resolve() / 'global_sfm'
    output = root / OUTPUT
    results_path = output / 'results.json'
    if not results_path.exists():
        raise FileNotFoundError(f'Run the white-line probe first: {results_path}')
    selected_frames = (270, 277, 278, 279, 280, 281)
    protocol = {'version': 1, 'results_sha256': file_digest(results_path),
                'selected_camera': 'R', 'selected_frames': selected_frames,
                'white_hsv': {'saturation_below': WHITE_MAX_S, 'value_above': WHITE_MIN_V},
                'color_dilation_px': COLOR_DILATION,
                'pair_colors_bgr': {'red_green': [0, 255, 255],
                                    'red_blue': [255, 0, 255],
                                    'green_blue': [255, 255, 0]},
                'scope': 'Visualize image-only paint support; no pose or point evaluation.'}
    fingerprint = _write_protocol(output / 'pair_mask_review_protocol.json', protocol)
    result_path = output / 'pair_mask_review.json'
    if result_path.exists():
        old = json.loads(result_path.read_text())
        if old['protocol_fingerprint'] != fingerprint:
            raise ValueError('Changed court pair-mask review inputs')
        return old
    results = json.loads(results_path.read_text())
    segment = next(row for row in results['segments']
                   if row['camera'] == 'R' and row['frames'][0] == 267)
    by_frame = {row['frame']: row for row in segment['observations']}
    records = {row['name']: row for row in load_scene(root)['images']}
    canvas = np.full((len(selected_frames)*280, 2*280, 3), 24, np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (COLOR_DILATION, COLOR_DILATION))
    for index, frame in enumerate(selected_frames):
        row = by_frame[frame]
        candidate = row['right']
        x, y = np.rint(candidate['hsv_native_xy']).astype(int)
        image = cv2.imread(records[row['name']]['source'])
        roi = image[y-HALF:y+HALF, x-HALF:x+HALF].copy()
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        white = (hsv[:, :, 1] < WHITE_MAX_S) & (hsv[:, :, 2] > WHITE_MIN_V)
        near = [cv2.dilate(mask, kernel) > 0 for mask in _hue_masks(roi, 3)]
        mask_image = np.zeros_like(roi)
        for (first, second), color in (((0, 1), (0, 255, 255)),
                                        ((0, 2), (255, 0, 255)),
                                        ((1, 2), (255, 255, 0))):
            mask_image[white & near[first] & near[second]] = color
        for tile in (roi, mask_image):
            cv2.drawMarker(tile, (HALF, HALF), (0, 0, 255),
                           cv2.MARKER_CROSS, 17, 2)
            if 'local_xy' in candidate:
                point = tuple(np.rint(candidate['local_xy']).astype(int))
                cv2.drawMarker(tile, point, (0, 255, 255),
                               cv2.MARKER_CROSS, 17, 2)
        y0 = index*280
        canvas[y0:y0+256, :256] = roi
        canvas[y0:y0+256, 280:536] = mask_image
        cv2.putText(canvas, f'R{frame} right', (3, y0+275),
                    cv2.FONT_HERSHEY_SIMPLEX, .52, (245, 245, 245), 1)
    review = output / 'review'
    review.mkdir(exist_ok=True)
    path = review / 'R270_281_right_pair_masks.jpg'
    cv2.imwrite(str(path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
    result = {'protocol_fingerprint': fingerprint, 'path': str(path),
              'sha256': file_digest(path)}
    write_json(result_path, result)
    return result
