"""Image-selected court corners followed in native fisheye pixels.

This is an exploratory audit of two visible painted-court corners. The coarse
ROIs identify a corner in each image but are not pixel measurements; the
native-pixel location is derived afresh from the three HSV color interiors.
Saved visual poses are used only after this image-side selection.
"""

from datetime import datetime, timezone
import json
from pathlib import Path

import cv2
import numpy as np
import pycolmap
from scipy.optimize import least_squares

from ..camera import Camera
from ..scene import digest, file_digest, load_scene, write_json
from .sfm_court_anchor_probe import _triangulate
from .sfm_court_trajectory_probe import _hue_masks


OUTPUT = 'sfm_court_native_anchor_probe_v2'
ROI_HALF_SIZE_NATIVE = 120
SEARCH_HALF_SIZE_NATIVE = 80
FRAMES = tuple(range(231, 243))
COARSE_QUARTER_SEEDS = {
    'paint_left_corner': {
        231: (146.9, 554.1), 232: (259.6, 529.7),
        233: (340.1, 520.6), 234: (332.4, 525.3),
        235: (305.5, 530.1), 236: (283.8, 531.3),
        237: (263.9, 532.3), 238: (238.7, 542.7),
        239: (219.1, 550.3), 240: (183.9, 565.7),
        241: (151.5, 560.8), 242: (117.7, 572.5),
    },
    'paint_right_corner': {
        231: (220.5, 516.2), 232: (351.1, 499.1),
        233: (441.3, 495.6), 234: (444.2, 497.5),
        235: (427.3, 508.7), 236: (411.1, 511.3),
        237: (393.9, 510.1), 238: (370.5, 522.3),
        239: (348.5, 527.1), 240: (306.3, 541.5),
        241: (266.6, 545.6), 242: (221.2, 559.0),
    },
}
SUPPORT_PAIRS = ((233, 239), (231, 242))


def _color_junction_native(bgr, quarter_seed):
    x, y = np.rint(np.asarray(quarter_seed) * 4).astype(int)
    h = ROI_HALF_SIZE_NATIVE
    if x < h or y < h or x + h >= bgr.shape[1] or y + h >= bgr.shape[0]:
        raise ValueError('Native junction ROI meets image border')
    roi = bgr[y-h:y+h, x-h:x+h]
    masks = _hue_masks(roi, 3)
    areas = [int(mask.sum()) for mask in masks]
    if min(areas) < 500:
        raise ValueError(f'Insufficient three-color interior: {areas}')
    distance = np.stack([cv2.distanceTransform((mask == 0).astype(np.uint8),
                                                cv2.DIST_L2, 3)
                         for mask in masks])
    cost = np.max(distance, axis=0)
    s = SEARCH_HALF_SIZE_NATIVE
    inner = cost[h-s:h+s, h-s:h+s]
    row, column = np.unravel_index(np.argmin(inner), inner.shape)
    point = np.array([x + column - s + .5, y + row - s + .5])
    if np.any(np.abs(point - np.array([x, y])) >= s - 1):
        raise ValueError('Three-color minimum at search boundary')
    return {'native_xy': point.tolist(), 'max_color_distance_native_px':
            float(inner[row, column]), 'local_color_areas_native_px': areas,
            'coarse_seed_quarter_xy': list(quarter_seed)}


def _world_ray(image, camera, xy):
    pose = image.cam_from_world().matrix()
    center = -pose[:3, :3].T @ pose[:3, 3]
    direction = pose[:3, :3].T @ camera.rays(np.asarray(xy)[None])[0]
    return center, direction


def _project(image, camera, world):
    pose = image.cam_from_world().matrix()
    camera_point = pose[:3, :3] @ world + pose[:3, 3]
    if camera_point[2] <= 0:
        return None, float(camera_point[2])
    return camera.project(camera_point[None])[0], float(camera_point[2])


def _audit_visual_pose(points, images, names, camera):
    audits = []
    for first, second in SUPPORT_PAIRS:
        world = _triangulate(_world_ray(images[names[first]], camera, points[first]),
                             _world_ray(images[names[second]], camera, points[second]))
        rays = [_world_ray(images[names[f]], camera, points[f])[1]
                for f in (first, second)]
        angle = float(np.rad2deg(np.arccos(np.clip(rays[0] @ rays[1], -1., 1.))))
        predictions = []
        for frame in FRAMES:
            predicted, depth = _project(images[names[frame]], camera, world)
            predictions.append({'frame': frame, 'predicted_native_xy':
                                None if predicted is None else predicted.tolist(),
                                'camera_depth': depth,
                                'error_native_px': None if predicted is None else
                                float(np.linalg.norm(predicted - points[frame]))})
        holdout = [row['error_native_px'] for row in predictions
                   if row['frame'] not in (first, second) and row['error_native_px'] is not None]
        audits.append({'support_frames': [first, second],
                       'support_ray_angle_degrees': angle,
                       'holdout_count': len(holdout),
                       'holdout_median_native_px': float(np.median(holdout)),
                       'holdout_max_native_px': float(np.max(holdout)),
                       'predictions': predictions})
    world0 = _triangulate(_world_ray(images[names[233]], camera, points[233]),
                          _world_ray(images[names[239]], camera, points[239]))

    def residual(world):
        errors = []
        for frame in FRAMES:
            projected, _ = _project(images[names[frame]], camera, world)
            if projected is None:
                return np.full(2 * len(FRAMES), 1e6)
            errors.extend(projected - points[frame])
        return np.asarray(errors)

    fit = least_squares(residual, world0, max_nfev=100)
    fit_errors = np.linalg.norm(residual(fit.x).reshape(-1, 2), axis=1)
    return {'two_view_holdouts': audits,
            'all_view_3d_point_fit': {
                'success': bool(fit.success),
                'median_native_px': float(np.median(fit_errors)),
                'max_native_px': float(np.max(fit_errors)),
                'per_frame_native_px': {str(frame): float(error)
                                        for frame, error in zip(FRAMES, fit_errors)}}}


def _model_coverage(points, images, names):
    rows = []
    close_ids = []
    for frame in FRAMES:
        image = images[names[frame]]
        p2d = image.points2D
        distances = np.linalg.norm(np.asarray([p.xy for p in p2d]) - points[frame], axis=1)
        valid = np.asarray([p.has_point3D() for p in p2d], dtype=bool)
        nearest_any = int(np.argmin(distances))
        nearest_3d = int(np.argmin(np.where(valid, distances, np.inf)))
        nearby = {int(p2d[i].point3D_id) for i in range(len(p2d))
                  if valid[i] and distances[i] <= 16}
        close_ids.append(nearby)
        rows.append({'frame': frame,
                     'nearest_feature_distance_native_px': float(distances[nearest_any]),
                     'nearest_3d_track_distance_native_px': float(distances[nearest_3d]),
                     'nearest_3d_track_id': int(p2d[nearest_3d].point3D_id),
                     '3d_tracks_within_16_native_px': len(nearby)})
    common = set.intersection(*close_ids)
    return {'per_frame': rows, 'common_existing_3d_track_ids_within_16_native_px':
            sorted(common)}


def _render_crops(output, source, rows):
    output.mkdir(exist_ok=True)
    files = {}
    for label, positions in rows.items():
        canvas = np.full((3 * 296, 4 * 296, 3), 24, np.uint8)
        for index, frame in enumerate(FRAMES):
            bgr = cv2.imread(source[frame])
            x, y = np.rint(positions[frame]['native_xy']).astype(int)
            crop = bgr[y-128:y+128, x-128:x+128].copy()
            cv2.drawMarker(crop, (128, 128), (0, 0, 255), cv2.MARKER_CROSS, 14, 1)
            row, column = divmod(index, 4)
            x0, y0 = column * 296, row * 296
            canvas[y0:y0+256, x0:x0+256] = crop
            cv2.putText(canvas, f'L{frame} ({x},{y})', (x0+5, y0+281),
                        cv2.FONT_HERSHEY_SIMPLEX, .55, (245, 245, 245), 1)
        path = output / f'{label}_native_crops.jpg'
        cv2.imwrite(str(path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 94])
        files[label] = {'path': str(path), 'sha256': file_digest(path)}
    return files


def run_court_native_anchor_probe(work):
    cv2.setNumThreads(4)
    work = Path(work).resolve()
    root = work / 'global_sfm'
    scene = load_scene(root)
    records = {r['frame_index']: r for r in scene['images']
               if r['camera'] == 'L' and r['frame_index'] in FRAMES}
    if len(records) != len(FRAMES):
        raise ValueError('Missing L231-L242 pose-selected images')
    model_path = root / 'ground_joint_ba/equal_control/sparse/0'
    protocol = {'version': 2,
                'source_sha256': {str(frame): file_digest(records[frame]['source'])
                                  for frame in FRAMES},
                'visual_model_hashes': {p.name: file_digest(p)
                                        for p in sorted(model_path.glob('*.bin'))},
                'coarse_quarter_roi_seeds': COARSE_QUARTER_SEEDS,
                'seed_origin': 'Exploratory image-only selection of the red paint left/right corners in consecutive L views; approximate ROI centers, not pixel annotations or ground truth',
                'native_roi_half_size_px': ROI_HALF_SIZE_NATIVE,
                'search_half_size_px': SEARCH_HALF_SIZE_NATIVE,
                'native_location': 'Minimum of maximum distance to the three HSV color interiors within ROI; OpenCV distanceTransform L2 mask3',
                'support_frame_pairs': SUPPORT_PAIRS,
                'pose_use': 'Saved visual SfM pose and candidate fitted fisheye intrinsics used only after corner identification to audit 3D compatibility',
                'uses_navigation_pose_values_or_las': False,
                'scope': 'Diagnostic candidate tracks only; no SfM or LFS model changed.'}
    fingerprint = digest(protocol)
    output = root / OUTPUT
    output.mkdir(exist_ok=True)
    protocol_path = output / 'protocol.json'
    if protocol_path.exists():
        if json.loads(protocol_path.read_text())['fingerprint'] != fingerprint:
            raise ValueError('Existing native anchor protocol belongs to changed inputs')
    else:
        write_json(protocol_path, {**protocol, 'fingerprint': fingerprint,
                                   'frozen_at_utc': datetime.now(timezone.utc).isoformat()})
    result_path = output / 'results.json'
    if result_path.exists():
        old = json.loads(result_path.read_text())
        if old['protocol_fingerprint'] != fingerprint:
            raise ValueError('Changed native anchor inputs')
        return old
    reconstruction = pycolmap.Reconstruction(model_path)
    images = {image.name: image for image in reconstruction.images.values()}
    names = {frame: records[frame]['name'] for frame in FRAMES}
    raw = reconstruction.cameras[1]
    camera = Camera('L', raw.width, raw.height, tuple(raw.params), (0, 0))
    locations = {}
    for frame in FRAMES:
        bgr = cv2.imread(records[frame]['source'])
        for label, seeds in COARSE_QUARTER_SEEDS.items():
            locations.setdefault(label, {})[frame] = _color_junction_native(
                bgr, seeds[frame])
    tracks = {}
    for label, detections in locations.items():
        points = {frame: np.asarray(detections[frame]['native_xy']) for frame in FRAMES}
        tracks[label] = {
            'detections': {str(frame): detections[frame] for frame in FRAMES},
            'all_within_candidate_179_fov': all(camera.valid(point[None], 179)[0]
                                                  for point in points.values()),
            'visual_pose_compatibility': _audit_visual_pose(points, images, names, camera),
            'existing_model_coverage': _model_coverage(points, images, names)}
    review = _render_crops(output / 'review',
                           {f: records[f]['source'] for f in FRAMES}, locations)
    result = {'protocol_fingerprint': fingerprint, 'frames': list(FRAMES),
              'tracks': tracks, 'review': review,
              'guard': 'ROI associations were chosen from image appearance during exploration, not a blinded selection. Native color minima can be several pixels off the physical line junction. A good fixed-pose 3D fit establishes consistency for this track only, not correct global poses or 3DGS image consistency.'}
    write_json(result_path, result)
    write_json(output / 'DIAGNOSTIC_ONLY.json', {
        'result_sha256': file_digest(result_path), 'status': 'DIAGNOSTIC_ONLY',
        'reason': 'One red-court corner is a strong near-ground candidate, but one short track cannot justify changing the full SfM solution; the other corner has larger fixed-pose residuals.'})
    return result


def run_court_native_anchor_eligibility(work):
    """Separate mask/FOV eligibility from feature scarcity for both corners."""
    work = Path(work).resolve()
    root = work / 'global_sfm'
    output = root / OUTPUT
    result = run_court_native_anchor_probe(work)
    scene = load_scene(root)
    records = {r['frame_index']: r for r in scene['images']
               if r['camera'] == 'L' and r['frame_index'] in FRAMES}
    model_path = root / 'ground_joint_ba/equal_control/sparse/0'
    protocol = {'version': 1, 'anchor_result_sha256': file_digest(output / 'results.json'),
                'semantic_mask_sha256': {str(frame): file_digest(records[frame]['semantic_mask'])
                                         for frame in FRAMES},
                'visual_model_hashes': {p.name: file_digest(p)
                                        for p in sorted(model_path.glob('*.bin'))},
                'sampling': 'Nearest original mask array element at detected native corner; fitted visual-camera FOV checked separately',
                'uses_navigation_pose_values_or_las': False}
    fingerprint = digest(protocol)
    protocol_path = output / 'eligibility_protocol.json'
    if protocol_path.exists():
        if json.loads(protocol_path.read_text())['fingerprint'] != fingerprint:
            raise ValueError('Changed native anchor eligibility inputs')
    else:
        write_json(protocol_path, {**protocol, 'fingerprint': fingerprint,
                                   'frozen_at_utc': datetime.now(timezone.utc).isoformat()})
    result_path = output / 'eligibility.json'
    if result_path.exists():
        old = json.loads(result_path.read_text())
        if old['protocol_fingerprint'] != fingerprint:
            raise ValueError('Changed native anchor eligibility inputs')
        return old
    reconstruction = pycolmap.Reconstruction(model_path)
    raw = reconstruction.cameras[1]
    camera = Camera('L', raw.width, raw.height, tuple(raw.params), (0, 0))
    images = {image.name: image for image in reconstruction.images.values()}
    rows = []
    for frame in FRAMES:
        mask = cv2.imread(records[frame]['semantic_mask'], cv2.IMREAD_UNCHANGED)
        row = {'frame': frame, 'existing_model_feature_count':
               len(images[records[frame]['name']].points2D)}
        for label, track in result['tracks'].items():
            xy = np.asarray(track['detections'][str(frame)]['native_xy'])
            x, y = np.floor(xy).astype(int)
            value = mask[y, x]
            row[label] = {'mask_value': value.tolist(),
                          'valid_160_fov': bool(camera.valid(xy[None], 160)[0]),
                          'valid_179_fov': bool(camera.valid(xy[None], 179)[0])}
        rows.append(row)
    data = {'protocol_fingerprint': fingerprint, 'rows': rows,
            'guard': 'Mask/FOV eligibility does not prove the detector would select or accurately locate a new feature.'}
    write_json(result_path, data)
    return data
