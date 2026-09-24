"""Frozen fixed-structure controls for the held-out L154 exposure."""

import itertools
import json
from pathlib import Path

import cv2
import numpy as np
import pycolmap

from .camera import Camera
from .scene import file_digest, load_scene, write_json
from .sfm_court_anchor_probe import _triangulate
from .sfm_court_native_anchor_probe import _project, _world_ray
from .sfm_l150_depth_probe import _cycles, _frozen_protocol, _inputs


OUTPUT = 'sfm_l154_static_control_probe'
FRAMES = (153, 154, 155)
REVIEWED = (
    ('tower_outer_band', 'far_building', 164,
     'Outer tower silhouette at a facade band; fixed but partly edge-localized.'),
    ('tower_facade_joint', 'far_building', 878,
     'Junction of two manufactured facade faces, away from tree foliage.'),
    ('walkway_facade_grid', 'far_building', 698,
     'Crossing of fixed facade grid lines on the right-hand building.'),
    ('black_post_foot', 'near_structure', 131,
     'Fixed black post meeting blue court-side paving.'),
    ('gray_post_foot', 'near_structure', 133,
     'Fixed gray canopy post meeting blue court-side paving.'),
    ('walkway_roof_corner', 'far_building', 306,
     'Fixed walkway roofline at building corner; edge localization remains possible.'),
)


def _review_sheet(output, records, selected):
    side, stride = 320, 342
    canvas = np.full((len(selected) * stride, len(FRAMES) * stride, 3), 24, np.uint8)
    images = {obs['name']: cv2.imread(records[obs['name']]['source'])
              for row in selected for obs in row['observations']}
    for row_index, row in enumerate(selected):
        for col, obs in enumerate(row['observations']):
            x, y = np.rint(obs['native_xy']).astype(int)
            crop = images[obs['name']][y-160:y+160, x-160:x+160]
            if crop.shape[:2] != (side, side):
                raise ValueError(f'Review crop meets image border: {row["id"]}')
            xx, yy = col * stride, row_index * stride
            canvas[yy:yy+side, xx:xx+side] = crop
            cv2.drawMarker(canvas, (xx+160, yy+160), (0, 0, 255), cv2.MARKER_CROSS, 22, 2)
            cv2.putText(canvas, f'L{obs["frame"]} i{obs["feature_index"]} ({x},{y})',
                        (xx+2, yy+335), cv2.FONT_HERSHEY_SIMPLEX, .51, (245, 245, 245), 1)
    path = output / 'review/frozen_static_controls.jpg'
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return {'path': str(path), 'sha256': file_digest(path)}


def run_l154_static_selection(work):
    root = Path(work).resolve() / 'global_sfm'
    output = root / OUTPUT
    output.mkdir(exist_ok=True)
    frontend_path, frontend, records, names, pairs, features, matches, scores = _inputs(root)
    protocol = {
        'version': 1, 'frames': FRAMES,
        'frontend_sha256': file_digest(frontend_path),
        'source_sha256': {names[frame]: file_digest(records[names[frame]]['source'])
                          for frame in FRAMES},
        'feature_sha256': {names[frame]: file_digest(frontend['features'][names[frame]]['path'])
                           for frame in FRAMES},
        'match_sha256': {f'{a}-{b}': file_digest(pairs[a, b]['path'])
                         for a, b in itertools.combinations(FRAMES, 2)},
        'reviewed': REVIEWED,
        'selection': 'Original-image fixed geometry in three-edge native LoMa cycles, selected before pose audit',
        'uses_pose_values_intrinsics_navigation_or_las': False,
        'scope': 'Freeze image observations only; no SfM or LFS input change.',
    }
    fingerprint = _frozen_protocol(output / 'selection_protocol.json', protocol)
    path = output / 'selection.json'
    if path.exists():
        previous = json.loads(path.read_text())
        if previous['protocol_fingerprint'] != fingerprint:
            raise ValueError('Existing L154 static selection belongs to changed inputs')
        return previous
    cycles = {row[0]: row for row in _cycles(FRAMES, matches)}
    selected = []
    for identity, scene_type, first_index, note in REVIEWED:
        if first_index not in cycles:
            raise ValueError(f'{identity}: missing exact three-edge cycle')
        indices = cycles[first_index]
        pair_scores = {f'L{a}-L{b}': scores[a, b][indices[FRAMES.index(a)]]
                       for a, b in itertools.combinations(FRAMES, 2)}
        selected.append({
            'id': identity, 'scene_type': scene_type,
            'image_review_note': note, 'pair_confidences': pair_scores,
            'min_pair_confidence': min(pair_scores.values()),
            'observations': [{'frame': frame, 'name': names[frame],
                              'feature_index': index,
                              'native_xy': features[frame][index].tolist()}
                             for frame, index in zip(FRAMES, indices)],
        })
    result = {
        'protocol_fingerprint': fingerprint, 'exact_three_edge_cycle_count': len(cycles),
        'selected': selected, 'review': _review_sheet(output, records, selected),
        'guard': 'Static classification does not establish subpixel identity. Building edges can have one-dimensional localization uncertainty.',
    }
    write_json(path, result)
    return result


def run_l154_static_audit(work):
    root = Path(work).resolve() / 'global_sfm'
    output = root / OUTPUT
    selection_path = output / 'selection.json'
    if not selection_path.exists():
        raise FileNotFoundError(f'Freeze image observations first: {selection_path}')
    selected = json.loads(selection_path.read_text())
    model_path = root / 'ground_joint_ba/equal_control/sparse/0'
    protocol = {
        'version': 1, 'selection_sha256': file_digest(selection_path),
        'visual_model_hashes': {path.name: file_digest(path)
                                for path in sorted(model_path.glob('*.bin'))},
        'support_rule': 'L153 and L155 triangulate; L154 native feature pixel held out',
        'camera_model': 'Saved equal-control visual SfM and candidate L OPENCV_FISHEYE, not truth',
        'uses_navigation_pose_values_or_las': False,
        'scope': 'Fixed-pose diagnostic only; no BA, calibration or LFS change.',
    }
    fingerprint = _frozen_protocol(output / 'audit_protocol.json', protocol)
    path = output / 'audit.json'
    if path.exists():
        previous = json.loads(path.read_text())
        if previous['protocol_fingerprint'] != fingerprint:
            raise ValueError('Existing L154 static audit belongs to changed inputs')
        return previous
    model = pycolmap.Reconstruction(model_path)
    images = {image.name: image for image in model.images.values()}
    cam_data = model.cameras[1]
    camera = Camera('L', cam_data.width, cam_data.height, tuple(cam_data.params), (0, 0))
    rows = []
    for item in selected['selected']:
        observations = item['observations']
        mapped = []
        for obs in observations:
            point = images[obs['name']].points2D[obs['feature_index']]
            if not np.allclose(point.xy, obs['native_xy'], atol=.02):
                raise ValueError(f'Model and frozen LoMa coordinate disagree: {item["id"]}')
            mapped.append(int(point.point3D_id) if point.has_point3D() else None)
        rays = [_world_ray(images[obs['name']], camera, np.asarray(obs['native_xy']))
                for obs in (observations[0], observations[2])]
        angle = float(np.rad2deg(np.arccos(np.clip(rays[0][1] @ rays[1][1], -1., 1.))))
        world = _triangulate(*rays)
        predictions = []
        for obs in observations:
            projected, depth = _project(images[obs['name']], camera, world)
            residual = projected - np.asarray(obs['native_xy']) if projected is not None else None
            predictions.append({
                'frame': obs['frame'], 'used_as_support': obs['frame'] in (153, 155),
                'camera_depth': depth,
                'predicted_native_xy': projected.tolist() if projected is not None else None,
                'residual_xy_native_px': residual.tolist() if residual is not None else None,
                'error_native_px': float(np.linalg.norm(residual)) if residual is not None else None,
            })
        rows.append({
            'id': item['id'], 'scene_type': item['scene_type'],
            'min_pair_confidence': item['min_pair_confidence'],
            'mapped_point3D_ids': mapped, 'support_ray_angle_degrees': angle,
            'predictions': predictions,
            'heldout_L154_residual_xy_native_px': predictions[1]['residual_xy_native_px'],
            'heldout_L154_error_native_px': predictions[1]['error_native_px'],
        })
    valid = [row for row in rows if row['heldout_L154_error_native_px'] is not None]
    by_scene = {}
    for scene_type in ('far_building', 'near_structure'):
        group = [row for row in valid if row['scene_type'] == scene_type]
        by_scene[scene_type] = {
            'count': len(group),
            'median_error_native_px': float(np.median([row['heldout_L154_error_native_px']
                                                        for row in group])) if group else None,
            'median_signed_residual_xy_native_px': np.median(
                [row['heldout_L154_residual_xy_native_px'] for row in group], axis=0
            ).tolist() if group else None,
        }
    result = {
        'protocol_fingerprint': fingerprint, 'rows': rows, 'summary_by_scene_type': by_scene,
        'guard': 'L154 is held out from each triangulation, not from the saved visual SfM pose estimate as a whole. Far building ray intersection may be ill-conditioned. Residuals combine pose, intrinsics, geometry and frontend localization.',
    }
    write_json(path, result)
    write_json(output / 'DIAGNOSTIC_ONLY.json', {
        'status': 'DIAGNOSTIC_ONLY', 'audit_sha256': file_digest(path),
        'reason': 'Static-control comparison does not itself license pose or calibration BA.',
    })
    return result


def _patch_alignment(source_image, target_image, homography, native_xy, side, margin=12):
    x, y = np.rint(native_xy).astype(int)
    x0, y0 = x-side//2-margin, y-side//2-margin
    patch_transform = np.array([[1., 0., -x0], [0., 1., -y0], [0., 0., 1.]])
    warped = cv2.warpPerspective(source_image, patch_transform @ homography,
                                 (side+2*margin, side+2*margin))
    target = target_image[y-side//2:y+side//2, x-side//2:x+side//2]
    if target.shape[:2] != (side, side):
        raise ValueError('Photometric target patch meets image border')
    source_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY).astype(np.float32)
    target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY).astype(np.float32)
    result = {}
    for mode in ('gray', 'gradient'):
        src, dst = source_gray, target_gray
        if mode == 'gradient':
            src = cv2.Laplacian(cv2.GaussianBlur(src, (3, 3), 0), cv2.CV_32F)
            dst = cv2.Laplacian(cv2.GaussianBlur(dst, (3, 3), 0), cv2.CV_32F)
        response = cv2.matchTemplate(src, dst, cv2.TM_CCOEFF_NORMED)
        _, peak, _, location = cv2.minMaxLoc(response)
        result[mode] = {'best_translation_xy_native_px': [int(location[0]-margin),
                                                        int(location[1]-margin)],
                        'normalized_correlation': float(peak)}
    return result


def run_l154_patch_comparison(work):
    """Compare patch and fixed controls with identical L153/L155 support frames."""
    root = Path(work).resolve() / 'global_sfm'
    output = root / OUTPUT
    crop_root = root / 'sfm_l150_crop_loma_probe'
    crop_selection_path = crop_root / 'selection.json'
    crop_results_path = crop_root / 'results.json'
    static_audit_path = output / 'audit.json'
    if not all(path.exists() for path in (crop_selection_path, crop_results_path,
                                         static_audit_path)):
        raise FileNotFoundError('Run crop selection and L154 static selection/audit first')
    crop_selection = json.loads(crop_selection_path.read_text())
    crop_results = json.loads(crop_results_path.read_text())
    records = {row['name']: row for row in load_scene(root)['images']}
    names = {obs['frame']: obs['image']
             for obs in crop_selection['selected'][0]['observations']
             if obs['frame'] in FRAMES}
    pair_paths = {}
    for a, b in ((153, 154), (154, 155)):
        token = next(row['match_token'] for row in crop_results['pairs']
                     if row['frames'] == [a, b])
        pair_paths[a, b] = crop_root / 'isolated_frontend_cache/matches' / f'{token}.npz'
    feature_paths = {frame: crop_root / 'features' / f'L{frame:05d}.npz'
                     for frame in FRAMES}
    model_path = root / 'ground_joint_ba/equal_control/sparse/0'
    protocol = {
        'version': 1, 'static_audit_sha256': file_digest(static_audit_path),
        'crop_selection_sha256': file_digest(crop_selection_path),
        'crop_results_sha256': file_digest(crop_results_path),
        'source_sha256': {names[frame]: file_digest(records[names[frame]]['source'])
                          for frame in FRAMES},
        'feature_sha256': {str(frame): file_digest(path)
                           for frame, path in feature_paths.items()},
        'crop_match_sha256': {f'{a}-{b}': file_digest(path)
                              for (a, b), path in pair_paths.items()},
        'visual_model_hashes': {path.name: file_digest(path)
                                for path in sorted(model_path.glob('*.bin'))},
        'support_rule': 'L153 and L155 triangulate; L154 held out for both static and patch rows',
        'homography_rule': 'Crop LoMa matches within 120 native px of coarse patch center, score >0.6; exclude three reviewed patch/boundary tracks within 20 px; RANSAC 3 px',
        'photometric_rule': 'Gray and Laplacian normalized correlation over 96/160 px original-image patches, translation search +/-12 px',
        'scope': 'Image and saved-pose diagnostic only; no SfM or LFS changes.',
    }
    fingerprint = _frozen_protocol(output / 'comparison_protocol.json', protocol)
    path = output / 'comparison.json'
    if path.exists():
        previous = json.loads(path.read_text())
        if previous['protocol_fingerprint'] != fingerprint:
            raise ValueError('Existing L154 comparison belongs to changed inputs')
        write_json(output / 'DIAGNOSTIC_ONLY.json', {
            'status': 'DIAGNOSTIC_ONLY', 'audit_sha256': file_digest(static_audit_path),
            'comparison_sha256': file_digest(path),
            'reason': 'Correlated white-patch candidates do not establish a safe pose or calibration update.',
        })
        return previous

    model = pycolmap.Reconstruction(model_path)
    images = {image.name: image for image in model.images.values()}
    cam_data = model.cameras[1]
    camera = Camera('L', cam_data.width, cam_data.height, tuple(cam_data.params), (0, 0))
    ground_rows = []
    for item in crop_selection['selected']:
        obs = {row['frame']: row for row in item['observations']}
        rays = [_world_ray(images[obs[frame]['image']], camera,
                           np.asarray(obs[frame]['native_xy'])) for frame in (153, 155)]
        angle = float(np.rad2deg(np.arccos(np.clip(rays[0][1] @ rays[1][1], -1., 1.))))
        world = _triangulate(*rays)
        projected, depth = _project(images[obs[154]['image']], camera, world)
        residual = projected-np.asarray(obs[154]['native_xy']) if projected is not None else None
        ground_rows.append({
            'id': item['id'], 'role': item['role'], 'support_ray_angle_degrees': angle,
            'heldout_L154_camera_depth': depth,
            'heldout_L154_residual_xy_native_px': residual.tolist() if residual is not None else None,
            'heldout_L154_error_native_px': float(np.linalg.norm(residual))
            if residual is not None else None,
        })

    observations = {item['id']: {row['frame']: np.asarray(row['native_xy'])
                                 for row in item['observations']}
                    for item in crop_selection['selected']}
    features = {}
    for frame, feature_path in feature_paths.items():
        with np.load(feature_path, allow_pickle=False) as data:
            features[frame] = data['native_xy'].copy()
    source_images = {frame: cv2.imread(records[names[frame]]['source'])
                     for frame in FRAMES}
    alignments = []
    for source_frame, pair in ((153, (153, 154)), (155, (154, 155))):
        with np.load(pair_paths[pair], allow_pickle=False) as data:
            matches, scores = data['matches'], data['scores']
        if source_frame == 153:
            source_xy, target_xy = features[153][matches[:, 0]], features[154][matches[:, 1]]
            center = np.array([1735., 2358.])
        else:
            source_xy, target_xy = features[155][matches[:, 1]], features[154][matches[:, 0]]
            center = np.array([1055., 2700.])
        keep = (np.linalg.norm(source_xy-center, axis=1) < 120) & (scores > .6)
        for identity in ('white_patch_edge_a', 'white_patch_edge_b', 'red_green_boundary'):
            keep &= np.linalg.norm(source_xy-observations[identity][source_frame],
                                   axis=1) > 20
        cv2.setRNGSeed(0)
        homography, inliers = cv2.findHomography(source_xy[keep], target_xy[keep],
                                                 cv2.RANSAC, 3.)
        if homography is None or int(inliers.sum()) < 10:
            raise ValueError(f'Insufficient independent local LoMa support L{source_frame}->L154')
        rows = []
        for identity in ('white_patch_edge_a', 'white_patch_edge_b'):
            source = observations[identity][source_frame].astype(np.float32)
            transferred = cv2.perspectiveTransform(source[None, None], homography)[0, 0]
            target = observations[identity][154]
            rows.append({
                'id': identity,
                'transferred_minus_L154_LoMa_xy_native_px': (transferred-target).tolist(),
                'photometric': {str(side): _patch_alignment(source_images[source_frame],
                                                            source_images[154], homography,
                                                            target, side)
                                for side in (96, 160)},
            })
        alignments.append({'source_frame': source_frame, 'target_frame': 154,
                           'input_match_count': int(keep.sum()),
                           'ransac_inlier_count': int(inliers.sum()), 'rows': rows})
    result = {
        'protocol_fingerprint': fingerprint, 'support_frames': [153, 155],
        'ground_rows': ground_rows, 'local_image_alignments': alignments,
        'guard': 'The two white-patch points remain correlated candidates, not precise ground truth. Local homographies and pixel correlation reuse the same images and some neighboring LoMa matches; they can expose large inconsistencies but do not identify pose versus intrinsics or certify physical point identity.',
    }
    write_json(path, result)
    write_json(output / 'DIAGNOSTIC_ONLY.json', {
        'status': 'DIAGNOSTIC_ONLY', 'audit_sha256': file_digest(static_audit_path),
        'comparison_sha256': file_digest(path),
        'reason': 'Correlated white-patch candidates do not establish a safe pose or calibration update.',
    })
    return result
