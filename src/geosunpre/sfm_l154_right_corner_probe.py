"""Image-first right-court painted-corner check around L154."""

import itertools
import json
from pathlib import Path

import cv2
import numpy as np
import pycolmap
from scipy.spatial.transform import Rotation

from .camera import Camera
from .scene import file_digest, load_scene, write_json
from .sfm_court_anchor_probe import _triangulate
from .sfm_court_native_anchor_probe import _project, _world_ray
from .sfm_l150_depth_probe import _frozen_protocol


OUTPUT = 'sfm_l154_right_corner_probe'
FRAMES = (151, 152, 153, 154, 155)
# Coarse centers are from the native photos, before consulting saved pose values.
COARSE = {151: (2595, 2010), 152: (2600, 2055), 153: (2650, 2080),
          154: (2735, 2140), 155: (2785, 2160)}
BASELINE = {151: 'moderate', 152: 'moderate', 153: 'balanced',
            154: 'balanced', 155: 'balanced'}
VARIANTS = {
    'strict': (70, 130), 'balanced': (85, 125),
    'moderate': (100, 115), 'loose': (120, 105),
}
ROI = (220, 80, 70, 230)  # left, right, up, down from coarse center


def _fit(image, center, max_s, min_v):
    cx, cy = center
    left, right, up, down = ROI
    roi = image[cy-up:cy+down, cx-left:cx+right]
    if roi.shape[:2] != (up+down, left+right):
        return {'status': 'crop_meets_image_border'}
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    white = ((hsv[:, :, 1] < max_s) & (hsv[:, :, 2] > min_v)).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(white, 8)
    components = []
    for index in range(1, count):
        if stats[index, cv2.CC_STAT_AREA] < 150:
            continue
        mask = labels == index
        arm_score = int(mask[40:125, 155:270].sum() + mask[115:280, 165:280].sum())
        components.append((arm_score, index))
    if not components:
        return {'status': 'no_painted_component'}
    component = max(components)[1]
    mask = labels == component
    horizontal_x, horizontal_y = [], []
    for x in range(30, 180):
        rows = np.flatnonzero(mask[35:125, x]) + 35
        if len(rows) >= 2:
            horizontal_x.append(x)
            horizontal_y.append(float(np.median(rows)))
    vertical_x, vertical_y = [], []
    for y in range(110, 285):
        columns = np.flatnonzero(mask[y, 165:285]) + 165
        if len(columns) >= 2:
            vertical_x.append(float(np.median(columns)))
            vertical_y.append(y)
    if len(horizontal_x) < 25 or len(vertical_y) < 25:
        return {'status': 'short_arm_support',
                'horizontal_columns': len(horizontal_x), 'vertical_rows': len(vertical_y)}
    horizontal_x, horizontal_y = np.asarray(horizontal_x), np.asarray(horizontal_y)
    vertical_x, vertical_y = np.asarray(vertical_x), np.asarray(vertical_y)
    # Fit the center of the painted stripe, rather than one HSV edge.
    horizontal = np.polyfit(horizontal_x-220, horizontal_y, 2)
    vertical = np.polyfit(vertical_y-70, vertical_x, 2)
    x, y = 220., 70.
    for _ in range(40):
        next_y = float(np.polyval(horizontal, x-220))
        next_x = float(np.polyval(vertical, next_y-70))
        if not np.isfinite(next_x+next_y):
            return {'status': 'nonfinite_intersection'}
        if abs(next_x-x)+abs(next_y-y) < 1e-7:
            x, y = next_x, next_y
            break
        x, y = next_x, next_y
    native = [float(cx-left+x), float(cy-up+y)]
    median_horizontal = float(np.median(np.abs(np.polyval(horizontal, horizontal_x-220)-horizontal_y)))
    median_vertical = float(np.median(np.abs(np.polyval(vertical, vertical_y-70)-vertical_x)))
    if np.linalg.norm(np.asarray(native)-np.asarray(center)) > 70:
        return {'status': 'intersection_far_from_seed', 'native_xy': native}
    return {'status': 'ok', 'native_xy': native,
            'horizontal_columns': len(horizontal_x), 'vertical_rows': len(vertical_y),
            'median_horizontal_fit_px': median_horizontal,
            'median_vertical_fit_px': median_vertical,
            'component_area_px': int(stats[component, cv2.CC_STAT_AREA]),
            'mask': mask}


def _public(fit):
    return {key: value for key, value in fit.items() if key != 'mask'}


def _image_gate(row):
    return (row['status'] == 'ok' and row['horizontal_columns'] >= 60
            and row['vertical_rows'] >= 100
            and row['median_horizontal_fit_px'] < 1.
            and row['median_vertical_fit_px'] < 1.)


def _review(output, records, selected):
    tile, stride = 320, 342
    canvas = np.full((2*stride, len(FRAMES)*stride, 3), 24, np.uint8)
    for col, frame in enumerate(FRAMES):
        row = selected[frame]
        image = cv2.imread(records[row['name']]['source'])
        cx, cy = COARSE[frame]
        crop = image[cy-150:cy+170, cx-150:cx+170]
        xx = col*stride
        canvas[:tile, xx:xx+tile] = crop
        px, py = np.rint(row['native_xy']).astype(int)
        cv2.drawMarker(canvas, (xx+px-(cx-150), py-(cy-150)),
                       (0, 0, 255), cv2.MARKER_CROSS, 24, 2)
        cv2.putText(canvas, f'L{frame} {px},{py}', (xx+3, 334),
                    cv2.FONT_HERSHEY_SIMPLEX, .54, (245, 245, 245), 1)
        fit = _fit(image, COARSE[frame], *VARIANTS[BASELINE[frame]])
        mask = fit['mask'].astype(np.uint8)*255
        left, right, up, down = ROI
        canvas[stride+10:stride+10+up+down, xx+10:xx+10+left+right] = (
            cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))
        cv2.drawMarker(canvas, (xx+10+px-(cx-left), stride+10+py-(cy-up)),
                       (0, 0, 255), cv2.MARKER_CROSS, 24, 2)
        cv2.putText(canvas, BASELINE[frame], (xx+3, 2*stride-9),
                    cv2.FONT_HERSHEY_SIMPLEX, .54, (245, 245, 245), 1)
    path = output / 'review/frozen_corner_and_mask.jpg'
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return {'path': str(path), 'sha256': file_digest(path)}


def run_right_corner_selection(work):
    root = Path(work).resolve() / 'global_sfm'
    output = root / OUTPUT
    output.mkdir(exist_ok=True)
    records = {row['name']: row for row in load_scene(root)['images']}
    names = {frame: next(name for name in records
                         if name.startswith('L/') and f'_{frame:05d}-L' in name)
             for frame in FRAMES}
    protocol = {
        'version': 1, 'frames': FRAMES, 'coarse_native_xy': COARSE,
        'source_sha256': {names[frame]: file_digest(records[names[frame]]['source'])
                          for frame in FRAMES},
        'baseline_variant_by_frame': BASELINE, 'white_hsv_variants': VARIANTS,
        'roi_left_right_up_down_px': ROI,
        'detector': 'Largest nearby connected white component; quadratic stripe centerlines on separate horizontal and vertical arms; intersection',
        'seed_shift_check_native_px': [-10, 0, 10],
        'uses_pose_values_intrinsics_navigation_or_las': False,
        'scope': 'Image-side candidate only; no SfM or LFS input change.',
    }
    fingerprint = _frozen_protocol(output / 'selection_protocol.json', protocol)
    path = output / 'selection.json'
    if path.exists():
        previous = json.loads(path.read_text())
        if previous['protocol_fingerprint'] != fingerprint:
            raise ValueError('Existing right-corner selection belongs to changed inputs')
        return previous
    selected = {}
    for frame in FRAMES:
        image = cv2.imread(records[names[frame]]['source'])
        attempts = {label: _public(_fit(image, COARSE[frame], *threshold))
                    for label, threshold in VARIANTS.items()}
        baseline = attempts[BASELINE[frame]]
        if not _image_gate(baseline):
            raise ValueError(f'L{frame}: baseline painted corner fails image gate: {baseline}')
        base_xy = np.asarray(baseline['native_xy'])
        threshold_shifts = {label: float(np.linalg.norm(np.asarray(row['native_xy'])-base_xy))
                            for label, row in attempts.items() if _image_gate(row)}
        seeds = []
        for dx in (-10, 0, 10):
            for dy in (-10, 0, 10):
                candidate = _public(_fit(image, (COARSE[frame][0]+dx, COARSE[frame][1]+dy),
                                         *VARIANTS[BASELINE[frame]]))
                seeds.append({'offset_xy': [dx, dy], 'status': candidate['status'],
                              'shift_from_baseline_px': float(np.linalg.norm(
                                  np.asarray(candidate['native_xy'])-base_xy))
                              if _image_gate(candidate) else None})
        valid_seeds = [row['shift_from_baseline_px'] for row in seeds
                       if row['shift_from_baseline_px'] is not None]
        selected[frame] = {'frame': frame, 'name': names[frame],
                           'coarse_native_xy': list(COARSE[frame]),
                           'baseline_variant': BASELINE[frame],
                           'native_xy': baseline['native_xy'],
                           'baseline': baseline, 'threshold_attempts': attempts,
                           'valid_threshold_shifts_px': threshold_shifts,
                           'seed_perturbations': seeds,
                           'max_valid_seed_shift_px': max(valid_seeds) if valid_seeds else None}
    result = {'protocol_fingerprint': fingerprint, 'selected': list(selected.values()),
              'review': _review(output, records, selected),
              'guard': 'A white-stripe centerline crossing is a physically identifiable court mark, but the detector can shift with white threshold, seed ROI and worn paint. These coordinates are candidates, not calibrated truth.'}
    write_json(path, result)
    return result


def run_right_corner_audit(work):
    root = Path(work).resolve() / 'global_sfm'
    output = root / OUTPUT
    selection_path = output / 'selection.json'
    if not selection_path.exists():
        raise FileNotFoundError(f'Freeze image selection first: {selection_path}')
    selection = json.loads(selection_path.read_text())
    model_path = root / 'ground_joint_ba/equal_control/sparse/0'
    protocol = {
        'version': 1, 'selection_sha256': file_digest(selection_path),
        'visual_model_hashes': {path.name: file_digest(path)
                                for path in sorted(model_path.glob('*.bin'))},
        'support_frames': [153, 155], 'holdout_frame': 154,
        'camera_model': 'Saved equal-control visual SfM and candidate L OPENCV_FISHEYE, not truth',
        'uses_navigation_pose_values_or_las': False,
        'scope': 'Fixed-model diagnostic only; no BA, calibration or LFS change.',
    }
    fingerprint = _frozen_protocol(output / 'audit_protocol.json', protocol)
    path = output / 'audit.json'
    if path.exists():
        previous = json.loads(path.read_text())
        if previous['protocol_fingerprint'] != fingerprint:
            raise ValueError('Existing right-corner audit belongs to changed inputs')
        return previous
    model = pycolmap.Reconstruction(model_path)
    images = {image.name: image for image in model.images.values()}
    cam_data = model.cameras[1]
    camera = Camera('L', cam_data.width, cam_data.height, tuple(cam_data.params), (0, 0))
    observations = {row['frame']: row for row in selection['selected']}
    rays = [_world_ray(images[observations[frame]['name']], camera,
                       np.asarray(observations[frame]['native_xy']))
            for frame in (153, 155)]
    angle = float(np.rad2deg(np.arccos(np.clip(rays[0][1] @ rays[1][1], -1., 1.))))
    world = _triangulate(*rays)
    predictions = []
    for frame in FRAMES:
        obs = observations[frame]
        projected, depth = _project(images[obs['name']], camera, world)
        residual = projected-np.asarray(obs['native_xy']) if projected is not None else None
        mapped = [(index, point) for index, point in enumerate(images[obs['name']].points2D)
                  if point.has_point3D()]
        positions = np.asarray([point.xy for _, point in mapped])
        distances = np.linalg.norm(positions-np.asarray(obs['native_xy']), axis=1)
        nearest = int(np.argmin(distances))
        predictions.append({'frame': frame, 'used_as_support': frame in (153, 155),
                            'predicted_native_xy': projected.tolist() if projected is not None else None,
                            'camera_depth': depth,
                            'residual_xy_native_px': residual.tolist() if residual is not None else None,
                            'error_native_px': float(np.linalg.norm(residual))
                            if residual is not None else None,
                            'nearest_mapped_distance_native_px': float(distances[nearest]),
                            'nearest_mapped_feature_index': int(mapped[nearest][0]),
                            'nearest_point3D_id': int(mapped[nearest][1].point3D_id)})
    result = {'protocol_fingerprint': fingerprint, 'support_ray_angle_degrees': angle,
              'predictions': predictions,
              'guard': 'The corner is image-selected and L154 is held out only from its triangulation. Threshold and seed sensitivity bound detector stability, not absolute accuracy. The saved camera model is a candidate, not truth.'}
    write_json(path, result)
    write_json(output / 'DIAGNOSTIC_ONLY.json', {
        'status': 'DIAGNOSTIC_ONLY', 'audit_sha256': file_digest(path),
        'reason': 'One additional painted-corner track cannot determine pose versus intrinsics or justify BA.'})
    return result


def run_right_corner_sensitivity(work):
    """Propagate image-only accepted thresholds through the frozen holdout."""
    root = Path(work).resolve() / 'global_sfm'
    output = root / OUTPUT
    selection_path = output / 'selection.json'
    audit_path = output / 'audit.json'
    if not selection_path.exists() or not audit_path.exists():
        raise FileNotFoundError('Run right-corner selection and audit first')
    selection = json.loads(selection_path.read_text())
    audit = json.loads(audit_path.read_text())
    model_path = root / 'ground_joint_ba/equal_control/sparse/0'
    protocol = {
        'version': 1, 'selection_sha256': file_digest(selection_path),
        'audit_sha256': file_digest(audit_path),
        'visual_model_hashes': {path.name: file_digest(path)
                                for path in sorted(model_path.glob('*.bin'))},
        'threshold_rule': 'Only image-gated threshold variants; Cartesian product at L153/L154/L155',
        'support_frames': [153, 155], 'holdout_frame': 154,
        'scope': 'Detector sensitivity and saved-model proximity only; no BA or LFS change.',
    }
    fingerprint = _frozen_protocol(output / 'sensitivity_protocol.json', protocol)
    path = output / 'sensitivity.json'
    if path.exists():
        previous = json.loads(path.read_text())
        if previous['protocol_fingerprint'] != fingerprint:
            raise ValueError('Existing right-corner sensitivity belongs to changed inputs')
        return previous
    model = pycolmap.Reconstruction(model_path)
    images = {image.name: image for image in model.images.values()}
    cam_data = model.cameras[1]
    camera = Camera('L', cam_data.width, cam_data.height, tuple(cam_data.params), (0, 0))
    rows = {row['frame']: row for row in selection['selected']}
    choices = {frame: {name: row['threshold_attempts'][name]['native_xy']
                       for name in row['valid_threshold_shifts_px']}
               for frame, row in rows.items() if frame in (153, 154, 155)}
    combinations = []
    for variants in itertools.product(*(choices[frame].items()
                                        for frame in (153, 154, 155))):
        names = [name for name, _ in variants]
        coords = [np.asarray(xy) for _, xy in variants]
        rays = [_world_ray(images[rows[frame]['name']], camera, xy)
                for frame, xy in ((153, coords[0]), (155, coords[2]))]
        world = _triangulate(*rays)
        projected, _ = _project(images[rows[154]['name']], camera, world)
        residual = projected-coords[1] if projected is not None else None
        combinations.append({
            'thresholds_L153_L154_L155': names,
            'heldout_L154_residual_xy_native_px': residual.tolist() if residual is not None else None,
            'heldout_L154_error_native_px': float(np.linalg.norm(residual))
            if residual is not None else None,
        })
    valid = [row['heldout_L154_residual_xy_native_px'] for row in combinations
             if row['heldout_L154_residual_xy_native_px'] is not None]
    nearest_ids = [row['nearest_point3D_id'] for row in audit['predictions']
                   if row['frame'] in (151, 152, 153, 154)]
    shared_id = nearest_ids[0] if len(set(nearest_ids)) == 1 else None
    mapped = None
    if shared_id is not None:
        point = model.points3D[shared_id]
        mapped = {
            'point3D_id': shared_id, 'track_length': len(point.track.elements),
            'saved_model_mean_reprojection_error_px': float(point.error),
            'observed_camera_frames': sorted([{
                'camera': model.images[element.image_id].name.split('/')[0],
                'frame_index': int(model.images[element.image_id].name.split('_')[-1].split('-')[0]),
            } for element in point.track.elements],
                key=lambda item: (item['camera'], item['frame_index'])),
        }
    result = {
        'protocol_fingerprint': fingerprint, 'combinations': combinations,
        'residual_xy_range_native_px': {
            'x': [float(min(x[0] for x in valid)), float(max(x[0] for x in valid))],
            'y': [float(min(x[1] for x in valid)), float(max(x[1] for x in valid))],
        },
        'shared_nearest_saved_3D_point_L151_to_L154': mapped,
        'guard': 'Threshold stability is not true-point accuracy. The nearby saved 3D track uses this same physical corner in earlier pose fitting, so this detector is not independent scene evidence even though its pixels were selected before reading pose values.',
    }
    write_json(path, result)
    write_json(output / 'DIAGNOSTIC_ONLY.json', {
        'status': 'DIAGNOSTIC_ONLY', 'audit_sha256': file_digest(audit_path),
        'sensitivity_sha256': file_digest(path),
        'reason': 'Right corner already participates in saved visual BA; no independent ground correction is established.',
    })
    return result


def run_right_corner_pose_hypotheses(work):
    """Probe two minimal L154 pose perturbations without modifying the model."""
    root = Path(work).resolve() / 'global_sfm'
    output = root / OUTPUT
    crop_path = root / 'sfm_l150_crop_loma_probe/selection.json'
    static_path = root / 'sfm_l154_static_control_probe/selection.json'
    corner_path = output / 'selection.json'
    sensitivity_path = output / 'sensitivity.json'
    inputs = (crop_path, static_path, corner_path, sensitivity_path)
    if not all(path.exists() for path in inputs):
        raise FileNotFoundError('Run crop, static and corner selections first')
    model_path = root / 'ground_joint_ba/equal_control/sparse/0'
    protocol = {
        'version': 1, 'input_sha256': {path.name + ':' + path.parent.name: file_digest(path)
                                      for path in inputs},
        'visual_model_hashes': {path.name: file_digest(path)
                                for path in sorted(model_path.glob('*.bin'))},
        'target': 'White-patch candidate A only; candidate B, six static structures and right painted corner evaluated afterwards',
        'parameterizations': ['minimum-norm rotation-only', 'minimum-norm translation-only'],
        'linearization_step': 1e-5,
        'support_frames': [153, 155], 'evaluation_frame': 154,
        'scope': 'Pose existence/sensitivity check only; no reconstruction or LFS input change.',
    }
    fingerprint = _frozen_protocol(output / 'pose_hypotheses_protocol.json', protocol)
    path = output / 'pose_hypotheses.json'
    if path.exists():
        previous = json.loads(path.read_text())
        if previous['protocol_fingerprint'] != fingerprint:
            raise ValueError('Existing pose hypotheses belong to changed inputs')
        return previous
    model = pycolmap.Reconstruction(model_path)
    images = {image.name: image for image in model.images.values()}
    cam_data = model.cameras[1]
    camera = Camera('L', cam_data.width, cam_data.height, tuple(cam_data.params), (0, 0))
    selected = []
    for item in json.loads(crop_path.read_text())['selected']:
        if item['id'] in ('white_patch_edge_a', 'white_patch_edge_b'):
            selected.append((item['id'], 'patch_candidate',
                             {row['frame']: row for row in item['observations']}, 'image'))
    for item in json.loads(static_path.read_text())['selected']:
        selected.append((item['id'], item['scene_type'],
                         {row['frame']: row for row in item['observations']}, 'name'))
    corner = json.loads(corner_path.read_text())['selected']
    selected.append(('right_white_corner', 'painted_corner',
                     {row['frame']: row for row in corner}, 'name'))
    tracks = []
    for identity, group, obs, name_key in selected:
        rays = [_world_ray(images[obs[frame][name_key]], camera,
                           np.asarray(obs[frame]['native_xy']))
                for frame in (153, 155)]
        world = _triangulate(*rays)
        pose = images[obs[154][name_key]].cam_from_world().matrix()
        camera_xyz = pose[:3, :3] @ world + pose[:3, 3]
        observed = np.asarray(obs[154]['native_xy'])
        tracks.append({'id': identity, 'group': group,
                       'camera_xyz': camera_xyz, 'observed_xy': observed})

    def projection(track, delta, mode):
        point = track['camera_xyz']
        shifted = Rotation.from_rotvec(delta).apply(point) if mode == 'rotation' else point+delta
        return camera.project(shifted[None])[0]

    target = next(track for track in tracks if track['id'] == 'white_patch_edge_a')
    hypotheses = []
    for mode in ('rotation', 'translation'):
        columns = []
        for dimension in range(3):
            step = np.zeros(3)
            step[dimension] = 1e-5
            columns.append((projection(target, step, mode)-projection(target, -step, mode))
                           / (2e-5))
        jacobian = np.stack(columns, axis=1)
        target_residual = projection(target, np.zeros(3), mode)-target['observed_xy']
        delta = np.linalg.lstsq(jacobian, -target_residual, rcond=None)[0]
        evaluations = []
        for track in tracks:
            before = projection(track, np.zeros(3), mode)-track['observed_xy']
            after = projection(track, delta, mode)-track['observed_xy']
            evaluations.append({
                'id': track['id'], 'group': track['group'],
                'before_residual_xy_native_px': before.tolist(),
                'after_residual_xy_native_px': after.tolist(),
                'before_error_native_px': float(np.linalg.norm(before)),
                'after_error_native_px': float(np.linalg.norm(after)),
            })
        hypotheses.append({
            'mode': mode,
            'camera_frame_delta': delta.tolist(),
            'rotation_magnitude_degrees': float(np.rad2deg(np.linalg.norm(delta)))
            if mode == 'rotation' else None,
            'evaluations': evaluations,
        })
    result = {
        'protocol_fingerprint': fingerprint, 'hypotheses': hypotheses,
        'guard': 'One uncertain patch observation supplies only two equations for a three-parameter perturbation. The minimum-norm choice is arbitrary and does not identify which exposure, intrinsics or track is wrong. Candidate B is correlated with A. These are counterfactual projections, not BA or an accepted pose.',
    }
    write_json(path, result)
    write_json(output / 'DIAGNOSTIC_ONLY.json', {
        'status': 'DIAGNOSTIC_ONLY',
        'sensitivity_sha256': file_digest(sensitivity_path),
        'pose_hypotheses_sha256': file_digest(path),
        'reason': 'Neither counterfactual is an independently validated SfM correction.',
    })
    return result


def run_right_corner_mapped_impact(work):
    """Check the counterfactual L154 poses against the saved long court track."""
    root = Path(work).resolve() / 'global_sfm'
    output = root / OUTPUT
    hypotheses_path = output / 'pose_hypotheses.json'
    sensitivity_path = output / 'sensitivity.json'
    if not hypotheses_path.exists() or not sensitivity_path.exists():
        raise FileNotFoundError('Run right-corner sensitivity and pose hypotheses first')
    sensitivity = json.loads(sensitivity_path.read_text())
    mapped = sensitivity['shared_nearest_saved_3D_point_L151_to_L154']
    if mapped is None:
        raise ValueError('No shared mapped court corner across L151-L154')
    model_path = root / 'ground_joint_ba/equal_control/sparse/0'
    protocol = {
        'version': 1, 'hypotheses_sha256': file_digest(hypotheses_path),
        'sensitivity_sha256': file_digest(sensitivity_path),
        'visual_model_hashes': {path.name: file_digest(path)
                                for path in sorted(model_path.glob('*.bin'))},
        'point3D_id': mapped['point3D_id'], 'evaluation_frame': 154,
        'scope': 'Saved-track counterfactual only; no model or LFS input change.',
    }
    fingerprint = _frozen_protocol(output / 'mapped_impact_protocol.json', protocol)
    path = output / 'mapped_impact.json'
    if path.exists():
        previous = json.loads(path.read_text())
        if previous['protocol_fingerprint'] != fingerprint:
            raise ValueError('Existing mapped impact belongs to changed inputs')
        return previous
    model = pycolmap.Reconstruction(model_path)
    point = model.points3D[mapped['point3D_id']]
    image, index = next((model.images[element.image_id], element.point2D_idx)
                        for element in point.track.elements
                        if model.images[element.image_id].name.startswith('L/')
                        and '_00154-L' in model.images[element.image_id].name)
    cam_data = model.cameras[1]
    camera = Camera('L', cam_data.width, cam_data.height, tuple(cam_data.params), (0, 0))
    pose = image.cam_from_world().matrix()
    camera_xyz = pose[:3, :3] @ point.xyz + pose[:3, 3]
    observed = np.asarray(image.points2D[index].xy)
    trials = [{'mode': 'saved_model', 'camera_frame_delta': [0., 0., 0.],
               'camera_xyz': camera_xyz}]
    for row in json.loads(hypotheses_path.read_text())['hypotheses']:
        delta = np.asarray(row['camera_frame_delta'])
        shifted = (Rotation.from_rotvec(delta).apply(camera_xyz)
                   if row['mode'] == 'rotation' else camera_xyz+delta)
        trials.append({'mode': row['mode'], 'camera_frame_delta': delta.tolist(),
                       'camera_xyz': shifted})
    evaluations = []
    for trial in trials:
        predicted = camera.project(trial['camera_xyz'][None])[0]
        residual = predicted-observed
        evaluations.append({'mode': trial['mode'],
                            'camera_frame_delta': trial['camera_frame_delta'],
                            'residual_xy_native_px': residual.tolist(),
                            'error_native_px': float(np.linalg.norm(residual))})
    result = {'protocol_fingerprint': fingerprint,
              'point3D_id': mapped['point3D_id'],
              'track_length': len(point.track.elements),
              'saved_model_mean_reprojection_error_px': float(point.error),
              'L154_feature_index': index, 'L154_observed_native_xy': observed.tolist(),
              'evaluations': evaluations,
              'guard': 'This old 3D point helped fit the saved pose; it is a consistency constraint for counterfactual changes, not independent camera truth. The physical white-corner pixel has a few-pixel localization convention difference.'}
    write_json(path, result)
    write_json(output / 'DIAGNOSTIC_ONLY.json', {
        'status': 'DIAGNOSTIC_ONLY', 'mapped_impact_sha256': file_digest(path),
        'reason': 'White-patch-only pose perturbations lack independent support and worsen existing court constraints.',
    })
    return result
