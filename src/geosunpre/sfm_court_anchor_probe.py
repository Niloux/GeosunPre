"""Reviewable court-color junctions, with image-only association and pose holdout.

A three-color junction is a structural proposal. Basketball-court ends can
repeat exactly the same junction topology, so geometry or color alone cannot
establish physical point identity.
"""

from datetime import datetime, timezone
import json
from pathlib import Path

import cv2
import numpy as np
import pycolmap
from scipy.spatial.transform import Rotation

from .camera import Camera
from .scene import digest, file_digest, load_scene, write_json
from .sfm_court_plane_probe import _virtual_pose
from .sfm_court_trajectory_probe import (OUTPUT as TRAJECTORY_OUTPUT,
                                         VIEW_SIZE, VIEW_FOV_DEG, PITCH_DEG,
                                         _view_classes,
                                         run_court_trajectory_graph_cycles)


OUTPUT = 'sfm_court_anchor_probe'
TRIPLE_DILATION_SIZE = 31
TRIPLE_COMPONENT_MIN = 5
TRIPLE_COMPONENT_MAX = 10000
PATCH_SIZE = 120
HIGHPASS_SIGMA = 5.


def _junctions(bgr):
    labels, _ = _view_classes(bgr, bgr.any(axis=2))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (TRIPLE_DILATION_SIZE,
                                        TRIPLE_DILATION_SIZE))
    nearby = [cv2.dilate((labels == label).astype(np.uint8), kernel) > 0
              for label in (1, 2, 3)]
    triple = (nearby[0] & nearby[1] & nearby[2]).astype(np.uint8)
    count, components, regions, centroids = cv2.connectedComponentsWithStats(triple, 8)
    # Minimax distance to the three color interiors is more stable than the
    # centroid of a potentially clipped dilation intersection.
    distance = np.stack([cv2.distanceTransform((labels != label).astype(np.uint8),
                                                cv2.DIST_L2, 3)
                         for label in (1, 2, 3)])
    cost = np.max(distance, axis=0)
    rows = []
    for index in range(1, count):
        area = int(regions[index, cv2.CC_STAT_AREA])
        if not TRIPLE_COMPONENT_MIN <= area <= TRIPLE_COMPONENT_MAX:
            continue
        ys, xs = np.where(components == index)
        best = int(np.argmin(cost[ys, xs]))
        refined = np.array([xs[best] + .5, ys[best] + .5], dtype=float)
        rows.append({'centroid_view_xy': centroids[index].tolist(),
                     'junction_view_xy': refined.tolist(),
                     'component_pixels': area,
                     'color_max_distance_view_px': float(cost[ys[best], xs[best]]),
                     'view_border_distance_px': float(min(refined[0], refined[1],
                                                          VIEW_SIZE - refined[0],
                                                          VIEW_SIZE - refined[1]))})
    return rows


def _native_xy(camera, view_xy):
    focal = VIEW_SIZE / (2 * np.tan(np.deg2rad(VIEW_FOV_DEG / 2)))
    local = np.array([[(view_xy[0] - VIEW_SIZE / 2) / focal,
                       (view_xy[1] - VIEW_SIZE / 2) / focal, 1.]])
    rotation = Rotation.from_euler('x', PITCH_DEG, degrees=True).as_matrix()
    return camera.project(local @ rotation.T)[0]


def _transform(H, xy):
    return cv2.perspectiveTransform(np.asarray([[xy]], np.float32), H)[0, 0]


def _highpass_ncc(source, target, H, center):
    warped = cv2.warpPerspective(source, H, (VIEW_SIZE, VIEW_SIZE))
    x, y = np.rint(center).astype(int)
    half = PATCH_SIZE // 2
    x0, y0, x1, y1 = max(0, x-half), max(0, y-half), min(VIEW_SIZE, x+half), min(VIEW_SIZE, y+half)
    if (x1-x0) < PATCH_SIZE * .7 or (y1-y0) < PATCH_SIZE * .7:
        return None
    a, b = warped[y0:y1, x0:x1], target[y0:y1, x0:x1]
    valid = (a.max(axis=2) > 0) & (b.max(axis=2) > 0)
    if valid.sum() < .7 * valid.size:
        return None
    gray_a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float32)
    high_a = gray_a - cv2.GaussianBlur(gray_a, (0, 0), HIGHPASS_SIGMA)
    high_b = gray_b - cv2.GaussianBlur(gray_b, (0, 0), HIGHPASS_SIGMA)
    xa, xb = high_a[valid].astype(float), high_b[valid].astype(float)
    xa -= xa.mean()
    xb -= xb.mean()
    norm = float(np.linalg.norm(xa) * np.linalg.norm(xb))
    return float(xa @ xb / norm) if norm > 1e-12 else None


def _ray(pose, xy):
    focal = VIEW_SIZE / (2 * np.tan(np.deg2rad(VIEW_FOV_DEG / 2)))
    rotation, translation = pose[:3, :3], pose[:3, 3]
    center = -rotation.T @ translation
    direction = rotation.T @ np.array([(xy[0] - VIEW_SIZE / 2) / focal,
                                       (xy[1] - VIEW_SIZE / 2) / focal, 1.])
    direction /= np.linalg.norm(direction)
    return center, direction


def _triangulate(first, second):
    matrix = np.zeros((3, 3))
    rhs = np.zeros(3)
    for center, direction in (first, second):
        projection = np.eye(3) - np.outer(direction, direction)
        matrix += projection
        rhs += projection @ center
    if np.linalg.cond(matrix) > 1e8:
        raise ValueError('Poor two-ray triangulation condition')
    return np.linalg.solve(matrix, rhs)


def _project(pose, world):
    camera = pose[:3, :3] @ world + pose[:3, 3]
    if camera[2] <= 0:
        raise ValueError('Anchor behind virtual camera')
    focal = VIEW_SIZE / (2 * np.tan(np.deg2rad(VIEW_FOV_DEG / 2)))
    return np.array([focal * camera[0] / camera[2] + VIEW_SIZE / 2,
                     focal * camera[1] / camera[2] + VIEW_SIZE / 2])


def _render_review(output, records, views, detections, groups, pair_by_key):
    review = output / 'review'
    review.mkdir(exist_ok=True)
    rows = [row for row in groups if row['frame_start'] in (50, 103, 223, 233)]
    canvas = np.zeros((len(rows) * 340, 3 * 320, 3), np.uint8)
    for row_index, group in enumerate(rows):
        for column, (name, candidate) in enumerate(zip(group['images'], group['candidate_indices'])):
            bgr = views[name].copy()
            xy = np.rint(detections[name][candidate]['junction_view_xy']).astype(int)
            cv2.circle(bgr, tuple(xy), 10, (0, 0, 255), 3)
            tile = cv2.resize(bgr, (320, 320), interpolation=cv2.INTER_AREA)
            canvas[row_index * 340:row_index * 340 + 320,
                   column * 320:(column + 1) * 320] = tile
            cv2.putText(canvas, f'{name[0]}{records[name]["frame_index"]}',
                        (column * 320 + 8, row_index * 340 + 334),
                        cv2.FONT_HERSHEY_SIMPLEX, .55, (245, 245, 245), 1)
    track_path = review / 'junction_triplets.jpg'
    cv2.imwrite(str(track_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
    native_frames = (('L', 209), ('L', 210), ('L', 289))
    native = np.zeros((1050, 3 * 750, 3), np.uint8)
    by_camera_frame = {(r['camera'], r['frame_index']): r for r in records.values()}
    for column, key in enumerate(native_frames):
        bgr = cv2.imread(by_camera_frame[key]['source'], cv2.IMREAD_REDUCED_COLOR_4)
        native[:1000, column * 750:(column + 1) * 750] = bgr
        cv2.putText(native, f'{key[0]}{key[1]}', (column * 750 + 8, 1035),
                    cv2.FONT_HERSHEY_SIMPLEX, .8, (245, 245, 245), 2)
    native_path = review / 'L209_L210_L289_native.jpg'
    cv2.imwrite(str(native_path), native, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return {'junction_triplets': str(track_path),
            'junction_triplets_sha256': file_digest(track_path),
            'L210_L289_native_context': str(native_path),
            'L210_L289_native_context_sha256': file_digest(native_path)}


def run_court_anchor_probe(work):
    cv2.setNumThreads(4)
    work = Path(work).resolve()
    root, trajectory = work / 'global_sfm', work / 'global_sfm' / TRAJECTORY_OUTPUT
    run_court_trajectory_graph_cycles(work)
    scan = json.loads((trajectory / 'scan.json').read_text())
    fitted = json.loads((trajectory / 'pairs.json').read_text())
    scene = load_scene(root)
    records = {r['name']: r for r in scene['images']}
    names = sorted({name for group in scan['groups'] for name in group['images']})
    view_paths = {name: trajectory / 'views'
                  / f'{records[name]["camera"]}{records[name]["frame_index"]:05d}.jpg'
                  for name in names}
    model_path = root / 'ground_joint_ba/equal_control/sparse/0'
    protocol = {'version': 1, 'scan_sha256': file_digest(trajectory / 'scan.json'),
                'pairs_sha256': file_digest(trajectory / 'pairs.json'),
                'view_sha256': {name: file_digest(path) for name, path in view_paths.items()},
                'visual_model_hashes': {p.name: file_digest(p)
                                        for p in sorted(model_path.glob('*.bin'))},
                'color_classes': 'Same HSV court-color classes as trajectory probe',
                'triple_dilation_size': TRIPLE_DILATION_SIZE,
                'component_area_range': [TRIPLE_COMPONENT_MIN, TRIPLE_COMPONENT_MAX],
                'junction_refinement': 'Minimize maximum distance to red, green and blue class interiors inside each triple-dilation component',
                'pair_association': 'Nearest image-only LoMa homography projection; never saved pose',
                'patch_size_view_px': PATCH_SIZE,
                'patch_highpass_sigma_view_px': HIGHPASS_SIGMA,
                'pose_use': 'Only triangulate first two junction rays and predict a third after image-only association',
                'uses_navigation_pose_values_or_las': False,
                'scope': 'Structural identity and holdout diagnostic only; no SfM or LFS input changed.'}
    fingerprint = digest(protocol)
    output = root / OUTPUT
    output.mkdir(exist_ok=True)
    protocol_path = output / 'protocol.json'
    if protocol_path.exists():
        if json.loads(protocol_path.read_text())['fingerprint'] != fingerprint:
            raise ValueError('Existing court anchor protocol belongs to changed inputs')
    else:
        write_json(protocol_path, {**protocol, 'fingerprint': fingerprint,
                                   'frozen_at_utc': datetime.now(timezone.utc).isoformat()})
    result_path = output / 'results.json'
    if result_path.exists():
        old = json.loads(result_path.read_text())
        if old['protocol_fingerprint'] != fingerprint:
            raise ValueError('Changed court anchor input')
        return old
    reconstruction = pycolmap.Reconstruction(model_path)
    visual_images = {image.name: image for image in reconstruction.images.values()}
    cameras = {label: Camera(label, reconstruction.cameras[camera_id].width,
                             reconstruction.cameras[camera_id].height,
                             tuple(reconstruction.cameras[camera_id].params), (0, 0))
               for label, camera_id in (('L', 1), ('R', 2))}
    views = {}
    detections = {}
    for name in names:
        bgr = cv2.imread(str(view_paths[name]), cv2.IMREAD_COLOR)
        if bgr is None or bgr.shape[:2] != (VIEW_SIZE, VIEW_SIZE):
            raise ValueError(f'Missing trajectory view: {view_paths[name]}')
        views[name] = bgr
        detections[name] = _junctions(bgr)
        for candidate in detections[name]:
            native = _native_xy(cameras[records[name]['camera']],
                                candidate['junction_view_xy'])
            candidate['candidate_native_xy'] = native.tolist()
            candidate['native_within_179_fov'] = bool(
                cameras[records[name]['camera']].valid(native[None], 179)[0])
    pair_by_key = {tuple(row['images']): row for row in fitted['pairs']
                   if row['status'] == 'fitted'}
    associations = []
    for (a, b), pair in pair_by_key.items():
        if not detections[a] or not detections[b]:
            continue
        source = np.asarray([row['junction_view_xy'] for row in detections[a]], np.float32)
        target = np.asarray([row['junction_view_xy'] for row in detections[b]], np.float32)
        H = np.asarray(pair['homography'])
        projected = cv2.perspectiveTransform(source[None], H)[0]
        distances = np.linalg.norm(projected[:, None, :] - target[None, :, :], axis=2)
        i, j = np.unravel_index(np.argmin(distances), distances.shape)
        associations.append({'images': [a, b], 'candidate_indices': [int(i), int(j)],
                             'frame_gap': pair['frame_gap'],
                             'homography_inliers': pair['homography_inliers'],
                             'junction_transfer_error_view_px': float(distances[i, j]),
                             'court_patch_highpass_ncc':
                             _highpass_ncc(views[a], views[b], H, target[j])})
    groups = []
    for group in scan['groups']:
        names3 = group['images']
        if any(not detections[name] for name in names3):
            continue
        first = np.asarray(detections[names3[0]][0]['junction_view_xy'])
        chosen = [0]
        for name in names3[1:]:
            candidates = detections[name]
            edge = pair_by_key.get((names3[0], name))
            if edge:
                predicted = _transform(np.asarray(edge['homography']), first)
                chosen.append(int(np.argmin([
                    np.linalg.norm(np.asarray(row['junction_view_xy']) - predicted)
                    for row in candidates])))
            elif len(candidates) == 1:
                chosen.append(0)
            else:
                chosen = []
                break
        if len(chosen) != 3:
            continue
        poses = [_virtual_pose(visual_images[name]) for name in names3]
        xy = [np.asarray(detections[name][index]['junction_view_xy'])
              for name, index in zip(names3, chosen)]
        try:
            world = _triangulate(_ray(poses[0], xy[0]), _ray(poses[1], xy[1]))
            predicted = _project(poses[2], world)
            error = float(np.linalg.norm(predicted - xy[2]))
            rays = [_ray(pose, point)[1] for pose, point in zip(poses, xy)]
            angles = [float(np.rad2deg(np.arccos(np.clip(rays[i] @ rays[j], -1, 1))))
                      for i, j in ((0, 1), (0, 2), (1, 2))]
            status = 'triangulated'
        except (ValueError, np.linalg.LinAlgError) as exc:
            error, angles, status = None, None, str(exc)
        groups.append({'images': names3, 'camera': group['camera'],
                       'frame_start': group['first_frame'],
                       'candidate_indices': chosen,
                       'candidate_view_xy': [point.tolist() for point in xy],
                       'border_distance_view_px': [
                           detections[name][index]['view_border_distance_px']
                           for name, index in zip(names3, chosen)],
                       'visual_pose_first_two_predict_third_error_view_px': error,
                       'ray_angle_degrees_ab_ac_bc': angles, 'status': status})
    pair_lookup = {(row['images'][0], row['images'][1]): row for row in associations}
    example = {}
    by_camera_frame = {(row['camera'], row['frame_index']): row['name']
                       for row in records.values()}
    for label, first, second in (('nearby', ('L', 209), ('L', 210)),
                                 ('long_range', ('L', 210), ('L', 289)),
                                 ('other_similar_corner', ('R', 150), ('L', 289))):
        key = (by_camera_frame[first], by_camera_frame[second])
        example[label] = pair_lookup.get(key)
    review = _render_review(output, records, views, detections, groups, pair_by_key)
    result = {'protocol_fingerprint': fingerprint,
              'images_scanned': len(names),
              'images_with_junction': sum(bool(rows) for rows in detections.values()),
              'detections': detections, 'pair_associations': associations,
              'three_view_groups': groups, 'lookalike_examples': example,
              'review': review,
              'guard': 'A color-topology match is not a physical point identity. Junction localization is image-derived and can move several view pixels, especially at view borders. Pose holdout uses fitted candidate intrinsics/visual poses, never input navigation truth. No anchors have been promoted to BA.'}
    write_json(result_path, result)
    write_json(output / 'DIAGNOSTIC_ONLY.json', {
        'result_sha256': file_digest(result_path), 'status': 'DIAGNOSTIC_ONLY',
        'reason': 'Only a few near-ground junction tracks have strong three-image support. Repeated basketball-court ends can produce nearly exact false image-plane junction transfer.'})
    return result


def _native_holdout(images, cameras, names, points):
    rays = []
    for name, xy in zip(names, points):
        pose = images[name].cam_from_world().matrix()
        direction = pose[:3, :3].T @ cameras[name[0]].rays(np.asarray(xy)[None])[0]
        center = -pose[:3, :3].T @ pose[:3, 3]
        rays.append((center, direction))
    world = _triangulate(*rays[:2])
    pose = images[names[2]].cam_from_world().matrix()
    camera_point = pose[:3, :3] @ world + pose[:3, 3]
    if camera_point[2] <= 0:
        raise ValueError('Anchor behind native camera')
    predicted = cameras[names[2][0]].project(camera_point[None])[0]
    return {'predicted_third_native_xy': predicted.tolist(),
            'third_error_native_px': float(np.linalg.norm(predicted - points[2]))}


def _model_coverage(images, names, points):
    per_image = []
    nearby_3d_ids = []
    for name, xy in zip(names, points):
        observations = images[name].points2D
        coordinates = np.asarray([observation.xy for observation in observations])
        distance = np.linalg.norm(coordinates - xy, axis=1)
        closest_any = int(np.argmin(distance))
        # INVALID_POINT3D_ID is UINT64_MAX. NumPy would turn a mixed list of
        # valid IDs and that sentinel into float64, rounding the sentinel and
        # making the nearest-track query meaningless.
        ids = [int(observation.point3D_id) for observation in observations]
        registered = np.asarray([observation.has_point3D()
                                 for observation in observations], dtype=bool)
        nearest_registered = int(np.argmin(np.where(registered, distance, np.inf)))
        close_ids = {ids[index] for index, dist in enumerate(distance)
                     if dist <= 16 and registered[index]}
        nearby_3d_ids.append(close_ids)
        per_image.append({
            'image': name, 'candidate_native_xy': np.asarray(xy).tolist(),
            'nearest_feature_distance_native_px': float(distance[closest_any]),
            'nearest_registered_distance_native_px': float(distance[nearest_registered]),
            'nearest_registered_point3d_id': ids[nearest_registered],
            'registered_points_within_16_native_px': len(close_ids)})
    common = set.intersection(*nearby_3d_ids)
    return {'per_image': per_image,
            'existing_3d_ids_within_16_native_px_in_all_three': sorted(common),
            'existing_3d_tracks_near_junction_in_all_three': len(common)}


def _native_crop_review(output, records, groups, detections):
    crop_size, tile_size = 256, 300
    canvas = np.full((len(groups) * tile_size, 3 * tile_size, 3), 24, np.uint8)
    for row_index, group in enumerate(groups):
        for column, (name, index) in enumerate(zip(group['images'], group['candidate_indices'])):
            image = cv2.imread(records[name]['source'])
            xy = np.asarray(detections[name][index]['candidate_native_xy'])
            x, y = np.rint(xy).astype(int)
            padded = cv2.copyMakeBorder(image, crop_size, crop_size, crop_size,
                                        crop_size, cv2.BORDER_CONSTANT)
            crop = padded[y + crop_size - crop_size//2:y + crop_size + crop_size//2,
                          x + crop_size - crop_size//2:x + crop_size + crop_size//2].copy()
            cv2.circle(crop, (crop_size//2, crop_size//2), 5, (0, 0, 255), 1)
            cv2.line(crop, (crop_size//2 - 10, crop_size//2),
                     (crop_size//2 + 10, crop_size//2), (0, 0, 255), 1)
            cv2.line(crop, (crop_size//2, crop_size//2 - 10),
                     (crop_size//2, crop_size//2 + 10), (0, 0, 255), 1)
            x0, y0 = column * tile_size, row_index * tile_size
            canvas[y0:y0 + crop_size, x0:x0 + crop_size] = crop
            cv2.putText(canvas, f'{name[0]}{records[name]["frame_index"]} ({x},{y})',
                        (x0 + 5, y0 + crop_size + 25), cv2.FONT_HERSHEY_SIMPLEX,
                        .55, (245, 245, 245), 1)
    path = output / 'review' / 'junction_native_crops.jpg'
    cv2.imwrite(str(path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 94])
    return {'path': str(path), 'sha256': file_digest(path)}


def run_court_anchor_coverage(work):
    """Check whether candidate ground junctions add any missing visual tracks."""
    work = Path(work).resolve()
    result = run_court_anchor_probe(work)
    output = work / 'global_sfm' / OUTPUT
    model_path = work / 'global_sfm/ground_joint_ba/equal_control/sparse/0'
    protocol = {'version': 2, 'anchor_result_sha256': file_digest(output / 'results.json'),
                'visual_model_hashes': {p.name: file_digest(p)
                                        for p in sorted(model_path.glob('*.bin'))},
                'nearby_registered_track_radius_native_px': 16,
                'native_holdout': 'Triangulate first two native fish-eye rays with existing visual poses; predict third image',
                'scope': 'Coverage audit only; no tracks, poses, or LFS input changed.'}
    fingerprint = digest(protocol)
    protocol_path = output / 'coverage_v2_protocol.json'
    if protocol_path.exists():
        if json.loads(protocol_path.read_text())['fingerprint'] != fingerprint:
            raise ValueError('Existing anchor coverage protocol belongs to changed inputs')
    else:
        write_json(protocol_path, {**protocol, 'fingerprint': fingerprint,
                                   'frozen_at_utc': datetime.now(timezone.utc).isoformat()})
    coverage_path = output / 'coverage_v2.json'
    if coverage_path.exists():
        old = json.loads(coverage_path.read_text())
        if old['protocol_fingerprint'] != fingerprint:
            raise ValueError('Changed anchor coverage input')
        return old
    scene = load_scene(work / 'global_sfm')
    records = {r['name']: r for r in scene['images']}
    reconstruction = pycolmap.Reconstruction(model_path)
    images = {image.name: image for image in reconstruction.images.values()}
    cameras = {label: Camera(label, reconstruction.cameras[camera_id].width,
                             reconstruction.cameras[camera_id].height,
                             tuple(reconstruction.cameras[camera_id].params), (0, 0))
               for label, camera_id in (('L', 1), ('R', 2))}
    rows = []
    for group in result['three_view_groups']:
        names, indices = group['images'], group['candidate_indices']
        points = [result['detections'][name][index]['candidate_native_xy']
                  for name, index in zip(names, indices)]
        row = {'camera': group['camera'], 'frame_start': group['frame_start'],
               'images': names, **_model_coverage(images, names, points)}
        try:
            row.update(_native_holdout(images, cameras, names, points))
        except (ValueError, np.linalg.LinAlgError) as exc:
            row['native_holdout_error'] = str(exc)
        rows.append(row)
    review = _native_crop_review(output, records, result['three_view_groups'],
                                 result['detections'])
    coverage = {'protocol_fingerprint': fingerprint, 'groups': rows,
                'review_native_crops': review,
                'guard': 'Nearest existing SfM feature is not proof of same physical junction. 16-pixel shared-track search is a coverage diagnostic, not a new match. Current visual poses and candidate intrinsics are untrusted.'}
    write_json(coverage_path, coverage)
    return coverage
