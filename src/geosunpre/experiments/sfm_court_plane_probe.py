"""Image-only plane-homography diagnostic from ground-focused LoMa matches.

Native camera poses and LAS are never used to estimate a homography. Court
color boundaries, rather than the matches used for fitting, check alignment.
"""

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path

import cv2
import numpy as np
import pycolmap
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from ..geometry import stats
from ..scene import digest, file_digest, write_json
from .sfm_court_down_loma_probe import (CONFIG, OUTPUT as DOWN_OUTPUT,
                                        VIEW_SIZE, VIEW_FOV_DEG, PITCH_DEG,
                                        MIN_MATCH_SCORE)


OUTPUT = 'sfm_court_plane_probe'
RANSAC_THRESHOLD_VIEW_PX = 4.
RANSAC_MAX_ITERS = 10000
RANSAC_CONFIDENCE = .999
BOUNDARY_TOLERANCE_VIEW_PX = 5.
HSV_S_MIN = 80
HSV_V_MIN = 60
CLASS_OPEN_KERNEL = 9


def _features_and_matches(work, ground, protocol, views, names):
    features = {}
    for name in names:
        token = digest({'protocol': protocol['fingerprint'], 'name': name,
                        'view_sha256': views[name]['view_sha256']})
        path = ground / 'features' / f'{token}.npz'
        with np.load(path, allow_pickle=False) as data:
            if str(data['token']) != token:
                raise ValueError(f'Changed ground LoMa feature: {name}')
            features[name] = {key: data[key] for key in data.files}
    pairs = {}
    for a, b in [(names[0], names[1]), (names[0], names[2]),
                 (names[1], names[2]), (names[3], names[4]),
                 (names[3], names[5]), (names[4], names[5])]:
        token = digest({'a': str(features[a]['token']),
                        'b': str(features[b]['token']),
                        'config': asdict(CONFIG)})
        path = work / 'matches' / f'{token}.npz'
        with np.load(path, allow_pickle=False) as data:
            if str(data['token']) != token:
                raise ValueError(f'Changed ground LoMa match: {a}, {b}')
            pairs[(a, b)] = {key: data[key] for key in data.files}
    return features, pairs


def _class_map(path):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None or bgr.shape[:2] != (VIEW_SIZE, VIEW_SIZE):
        raise ValueError(f'Missing ground view: {path}')
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    chromatic = (s >= HSV_S_MIN) & (v >= HSV_V_MIN)
    rules = {1: (h <= 10) | (h >= 170),
             2: (h >= 74) & (h <= 95),
             3: (h >= 98) & (h <= 115)}
    labels = np.zeros((VIEW_SIZE, VIEW_SIZE), dtype=np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (CLASS_OPEN_KERNEL, CLASS_OPEN_KERNEL))
    for label, hue in rules.items():
        mask = cv2.morphologyEx((hue & chromatic).astype(np.uint8),
                                cv2.MORPH_OPEN, kernel) > 0
        labels[mask] = label
    return bgr, labels


def _warp_mask(mask, H, shape):
    return cv2.warpPerspective(mask.astype(np.uint8), H, shape,
                               flags=cv2.INTER_NEAREST) > 0


def _boundary(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT, kernel) > 0


def _boundary_f1(source, target, valid):
    first, second = _boundary(source) & valid, _boundary(target) & valid
    if not first.any() or not second.any():
        return {'source_boundary_pixels': int(first.sum()),
                'target_boundary_pixels': int(second.sum()),
                'precision': None, 'recall': None, 'f1': None}
    dist_target = cv2.distanceTransform((~second).astype(np.uint8),
                                        cv2.DIST_L2, 3)
    dist_source = cv2.distanceTransform((~first).astype(np.uint8),
                                        cv2.DIST_L2, 3)
    precision = float(np.mean(dist_target[first] <= BOUNDARY_TOLERANCE_VIEW_PX))
    recall = float(np.mean(dist_source[second] <= BOUNDARY_TOLERANCE_VIEW_PX))
    return {'source_boundary_pixels': int(first.sum()),
            'target_boundary_pixels': int(second.sum()),
            'precision': precision, 'recall': recall,
            'f1': float(2 * precision * recall / (precision + recall))
            if precision + recall else 0.}


def _color_alignment(source, target, H):
    valid = _warp_mask(np.ones_like(source, dtype=np.uint8), H,
                       (VIEW_SIZE, VIEW_SIZE))
    transformed = cv2.warpPerspective(source, H, (VIEW_SIZE, VIEW_SIZE),
                                      flags=cv2.INTER_NEAREST)
    groups = {}
    for label, name in ((1, 'red'), (2, 'green'), (3, 'blue')):
        a = (transformed == label) & valid
        b = (target == label) & valid
        union = a | b
        groups[name] = {'source_pixels': int(a.sum()), 'target_pixels': int(b.sum()),
                        'iou': float((a & b).sum() / union.sum()) if union.any() else None,
                        'boundary': _boundary_f1(a, b, valid)}
    return {'overlap_view_pixels': int(valid.sum()), 'classes': groups}


def _fit_pair(a, b, features, matches, classes, views, output, label):
    match = matches[(a, b)]
    high = match['scores'] >= MIN_MATCH_SCORE
    pairs = match['matches'][high]
    if len(pairs) < 4:
        raise RuntimeError(f'Insufficient high-score ground LoMa pairs: {a}, {b}')
    xy_a = features[a]['view_keypoints'][pairs[:, 0]].astype(np.float32)
    xy_b = features[b]['view_keypoints'][pairs[:, 1]].astype(np.float32)
    H, inlier = cv2.findHomography(xy_a, xy_b, cv2.USAC_MAGSAC,
                                   RANSAC_THRESHOLD_VIEW_PX,
                                   maxIters=RANSAC_MAX_ITERS,
                                   confidence=RANSAC_CONFIDENCE)
    if H is None or inlier is None:
        raise RuntimeError(f'Ground LoMa homography fit failed: {a}, {b}')
    inlier = inlier.ravel().astype(bool)
    projected = cv2.perspectiveTransform(xy_a[None], H)[0]
    error = np.linalg.norm(projected - xy_b, axis=1)
    alignment = _color_alignment(classes[a], classes[b], H)
    identity = _color_alignment(classes[a], classes[b], np.eye(3))
    source_bgr = cv2.imread(views[a]['view'], cv2.IMREAD_COLOR)
    target_bgr = cv2.imread(views[b]['view'], cv2.IMREAD_COLOR)
    warped = cv2.warpPerspective(source_bgr, H, (VIEW_SIZE, VIEW_SIZE))
    overlay = cv2.addWeighted(warped, .5, target_bgr, .5, 0)
    review_path = output / 'review' / f'{label}.jpg'
    review_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(review_path), cv2.resize(overlay, (800, 800),
                                             interpolation=cv2.INTER_AREA))
    return H, {'images': [a, b], 'high_score_matches': int(len(pairs)),
               'homography_inliers': int(inlier.sum()),
               'inlier_fraction': float(np.mean(inlier)),
               'inlier_error_view_px': stats(error[inlier]),
               'outlier_error_view_px': stats(error[~inlier]),
               'homography': H.tolist(), 'alignment': alignment,
               'identity_alignment': identity,
               'review': str(review_path)}


def _cycle_error(H_ab, H_ac, H_bc, first_class):
    ys, xs = np.mgrid[80:VIEW_SIZE-80:80, 80:VIEW_SIZE-80:80]
    xy = np.column_stack((xs.ravel(), ys.ravel())).astype(np.float32)
    take = first_class[xy[:, 1].astype(int), xy[:, 0].astype(int)] > 0
    xy = xy[take]
    if not len(xy):
        return stats([])
    direct = cv2.perspectiveTransform(xy[None], H_ac)[0]
    via = cv2.perspectiveTransform(xy[None], H_bc @ H_ab)[0]
    finite = np.isfinite(direct).all(axis=1) & np.isfinite(via).all(axis=1)
    inside = ((direct[:, 0] >= 0) & (direct[:, 0] < VIEW_SIZE)
              & (direct[:, 1] >= 0) & (direct[:, 1] < VIEW_SIZE)
              & (via[:, 0] >= 0) & (via[:, 0] < VIEW_SIZE)
              & (via[:, 1] >= 0) & (via[:, 1] < VIEW_SIZE))
    return stats(np.linalg.norm(direct[finite & inside] - via[finite & inside],
                                axis=1))


def run_court_plane_probe(work):
    cv2.setNumThreads(4)
    work = Path(work).resolve()
    root = work / 'global_sfm'
    ground = root / DOWN_OUTPUT
    ground_protocol = json.loads((ground / 'protocol.json').read_text())
    ground_result = json.loads((ground / 'results.json').read_text())
    if ground_result['protocol_fingerprint'] != ground_protocol['fingerprint']:
        raise ValueError('Ground-view LoMa result/protocol mismatch')
    names = [name for row in ground_result['rows'] for name in row['images']]
    view_rows = json.loads((ground / 'views.json').read_text())['rows']
    views = {row['name']: row for row in view_rows}
    features, matches = _features_and_matches(work, ground, ground_protocol,
                                              views, names)
    protocol = {'version': 1,
                'ground_loma_protocol_fingerprint': ground_protocol['fingerprint'],
                'ground_loma_result_sha256': file_digest(ground / 'results.json'),
                'view_sha256': {name: views[name]['view_sha256'] for name in names},
                'feature_tokens': {name: str(features[name]['token']) for name in names},
                'match_tokens': {f'{a}|{b}': str(match['token'])
                                 for (a, b), match in matches.items()},
                'minimum_loma_score': MIN_MATCH_SCORE,
                'homography_method': 'USAC_MAGSAC',
                'ransac_threshold_view_px': RANSAC_THRESHOLD_VIEW_PX,
                'ransac_max_iters': RANSAC_MAX_ITERS,
                'ransac_confidence': RANSAC_CONFIDENCE,
                'hsv_s_min': HSV_S_MIN, 'hsv_v_min': HSV_V_MIN,
                'hue_classes_cv8': {'red': '0..10 or 170..179',
                                    'green': '74..95', 'blue': '98..115'},
                'class_open_kernel': CLASS_OPEN_KERNEL,
                'boundary_tolerance_view_px': BOUNDARY_TOLERANCE_VIEW_PX,
                'uses_visual_pose_navigation_las_or_manual_labels': False,
                'scope': 'Image-only planar homography pilot on two L triplets; visual color-boundary and three-homography cycle audit. No SfM or LFS change.'}
    fingerprint = digest(protocol)
    output = root / OUTPUT
    output.mkdir(exist_ok=True)
    protocol_path = output / 'protocol.json'
    if protocol_path.exists():
        if json.loads(protocol_path.read_text())['fingerprint'] != fingerprint:
            raise ValueError('Existing court-plane probe belongs to changed inputs')
    else:
        write_json(protocol_path, {**protocol, 'fingerprint': fingerprint,
                                   'frozen_at_utc': datetime.now(timezone.utc).isoformat()})
    classes = {name: _class_map(views[name]['view'])[1] for name in names}
    rows = []
    for triple in [tuple(names[:3]), tuple(names[3:])]:
        a, b, c = triple
        H_ab, ab = _fit_pair(a, b, features, matches, classes, views,
                             output, f'L{int(Path(a).stem.split("_")[-1].split("-")[0]):05d}_ab')
        H_ac, ac = _fit_pair(a, c, features, matches, classes, views,
                             output, f'L{int(Path(a).stem.split("_")[-1].split("-")[0]):05d}_ac')
        H_bc, bc = _fit_pair(b, c, features, matches, classes, views,
                             output, f'L{int(Path(a).stem.split("_")[-1].split("-")[0]):05d}_bc')
        row = {'images': triple, 'pairs': {'a_b': ab, 'a_c': ac, 'b_c': bc},
               'homography_three_way_cycle_error_view_px':
               _cycle_error(H_ab, H_ac, H_bc, classes[a])}
        rows.append(row)
        print(f'Court-plane L{Path(a).stem}: {row["homography_three_way_cycle_error_view_px"]["median"]} px median cycle', flush=True)
    result = {'protocol_fingerprint': fingerprint, 'rows': rows,
              'guard': 'A low homography residual does not certify correct pixel identities, camera calibration, or 3D pose. Repeated paving can support a coherent false shift. Color classes are rough image cues. No plane or pose has been injected into SfM or LFS.'}
    write_json(output / 'results.json', result)
    return result


def _virtual_pose(image):
    native = image.cam_from_world().matrix()
    view_from_native = Rotation.from_euler('x', PITCH_DEG, degrees=True).as_matrix().T
    return view_from_native @ native


def _relative_pose(a, b):
    ra, rb = a[:, :3], b[:, :3]
    relative_rotation = rb @ ra.T
    relative_translation = b[:, 3] - relative_rotation @ a[:, 3]
    return relative_rotation, relative_translation


def _pose_plane_fit(H, source_class, source_pose, target_pose,
                    view_size=VIEW_SIZE, view_fov_deg=VIEW_FOV_DEG):
    """Best single-plane warp possible for two frozen old visual poses."""
    rotation, translation = _relative_pose(source_pose, target_pose)
    focal = view_size / (2 * np.tan(np.deg2rad(view_fov_deg / 2)))
    K = np.array([[focal, 0, view_size / 2],
                  [0, focal, view_size / 2], [0, 0, 1.]])
    inv_K = np.linalg.inv(K)
    Hn = inv_K @ H @ K
    translation_sq = float(np.dot(translation, translation))
    if translation_sq < 1e-12:
        raise ValueError('Saved visual poses have zero translation')
    orthogonal = np.eye(3) - np.outer(translation, translation) / translation_sq
    projected_h = orthogonal @ Hn
    projected_r = orthogonal @ rotation
    scale = float(np.sum(projected_h * projected_r)
                  / np.sum(projected_h * projected_h))
    initial = translation @ (scale * Hn - rotation) / translation_sq
    margin = max(20, round(view_size / 16))
    stride = max(20, round(view_size * 60 / VIEW_SIZE))
    ys, xs = np.mgrid[margin:view_size-margin:stride,
                      margin:view_size-margin:stride]
    xy = np.column_stack((xs.ravel(), ys.ravel())).astype(np.float32)
    xy = xy[source_class[xy[:, 1].astype(int), xy[:, 0].astype(int)] > 0]
    measured = cv2.perspectiveTransform(xy[None], H)[0]
    inside = ((measured[:, 0] >= 0) & (measured[:, 0] < view_size)
              & (measured[:, 1] >= 0) & (measured[:, 1] < view_size))
    xy, measured = xy[inside], measured[inside]
    if len(xy) < 12:
        raise ValueError('Insufficient common court grid for pose-plane comparison')

    def warp(vector):
        matrix = K @ (rotation + np.outer(translation, vector)) @ inv_K
        transformed = cv2.perspectiveTransform(xy[None], matrix)[0]
        return transformed, matrix

    def residual(vector):
        warped, _ = warp(vector)
        difference = warped - measured
        return np.nan_to_num(difference, nan=1e4, posinf=1e4,
                             neginf=-1e4).ravel()

    fitted = least_squares(residual, initial, loss='soft_l1', f_scale=2.,
                           max_nfev=300)
    predicted, H_predicted = warp(fitted.x)
    error = np.linalg.norm(predicted - measured, axis=1)
    return H_predicted, {'common_court_grid_points': int(len(xy)),
                         'old_pose_baseline_translation_norm': float(np.linalg.norm(translation)),
                         'best_plane_vector_in_source_view': fitted.x.tolist(),
                         'fit_success': bool(fitted.success),
                         'fitted_plane_warp_disagreement_view_px': stats(error)}


def run_court_plane_pose_audit(work):
    """Compare image-only court homographies to the best old-pose plane warp."""
    cv2.setNumThreads(4)
    work = Path(work).resolve()
    root = work / 'global_sfm'
    output = root / OUTPUT
    plane_protocol = json.loads((output / 'protocol.json').read_text())
    plane_result = json.loads((output / 'results.json').read_text())
    if plane_result['protocol_fingerprint'] != plane_protocol['fingerprint']:
        raise ValueError('Court-plane result/protocol mismatch')
    ground = root / DOWN_OUTPUT
    views = {row['name']: row for row in
             json.loads((ground / 'views.json').read_text())['rows']}
    names = [name for row in plane_result['rows'] for name in row['images']]
    model_path = root / 'ground_joint_ba/equal_control/sparse/0'
    protocol = {'version': 1,
                'court_plane_protocol_fingerprint': plane_protocol['fingerprint'],
                'court_plane_result_sha256': file_digest(output / 'results.json'),
                'saved_visual_model_hashes': {p.name: file_digest(p)
                                              for p in sorted(model_path.glob('*.bin'))},
                'view_sha256': {name: views[name]['view_sha256'] for name in names},
                'virtual_view_size': VIEW_SIZE, 'virtual_view_fov_degrees': VIEW_FOV_DEG,
                'virtual_view_pitch_degrees': PITCH_DEG,
                'plane_fit_loss': 'soft_l1', 'plane_fit_f_scale_view_px': 2.,
                'grid_stride_view_px': 60,
                'scope': 'Read-only diagnostic: best planar homography representable by each pair of saved visual poses. Camera intrinsics/poses are candidate values, not ground truth.'}
    fingerprint = digest(protocol)
    protocol_path = output / 'pose_audit_protocol.json'
    if protocol_path.exists():
        if json.loads(protocol_path.read_text())['fingerprint'] != fingerprint:
            raise ValueError('Existing pose audit belongs to changed plane or visual inputs')
    else:
        write_json(protocol_path, {**protocol, 'fingerprint': fingerprint,
                                   'frozen_at_utc': datetime.now(timezone.utc).isoformat()})
    reconstruction = pycolmap.Reconstruction(model_path)
    images = {image.name: image for image in reconstruction.images.values()}
    poses = {name: _virtual_pose(images[name]) for name in names}
    classes = {name: _class_map(views[name]['view'])[1] for name in names}
    rows = []
    for row in plane_result['rows']:
        comparisons = {}
        for label, pair in row['pairs'].items():
            a, b = pair['images']
            H = np.asarray(pair['homography'], dtype=float)
            H_old, summary = _pose_plane_fit(H, classes[a], poses[a], poses[b])
            color = _color_alignment(classes[a], classes[b], H_old)
            comparisons[label] = {'images': [a, b], **summary,
                                  'best_old_pose_color_alignment': color,
                                  'image_only_color_alignment': pair['alignment']}
        rows.append({'images': row['images'], 'pairs': comparisons})
    result = {'protocol_fingerprint': fingerprint, 'rows': rows,
              'guard': 'A large residual means the current visual poses plus any one plane do not reproduce the image-only court warp under the current fitted intrinsics. It does not identify whether poses, calibration, plane approximation, or imagery cause the discrepancy. No SfM or LFS input changed.'}
    write_json(output / 'pose_audit.json', result)
    return result


def _common_pose_plane_fit(triple, image_only_pairs, classes, poses, initial):
    a, b, c = triple
    pair_names = {'a_b': (a, b), 'a_c': (a, c), 'b_c': (b, c)}
    relative = {label: _relative_pose(poses[src], poses[dst])
                for label, (src, dst) in pair_names.items()}
    focal = VIEW_SIZE / (2 * np.tan(np.deg2rad(VIEW_FOV_DEG / 2)))
    K = np.array([[focal, 0, VIEW_SIZE / 2],
                  [0, focal, VIEW_SIZE / 2], [0, 0, 1.]])
    inv_K = np.linalg.inv(K)
    grids = {}
    for label, (src, _) in pair_names.items():
        H = np.asarray(image_only_pairs[label]['homography'], dtype=float)
        ys, xs = np.mgrid[80:VIEW_SIZE-80:60, 80:VIEW_SIZE-80:60]
        xy = np.column_stack((xs.ravel(), ys.ravel())).astype(np.float32)
        xy = xy[classes[src][xy[:, 1].astype(int), xy[:, 0].astype(int)] > 0]
        measured = cv2.perspectiveTransform(xy[None], H)[0]
        inside = ((measured[:, 0] >= 0) & (measured[:, 0] < VIEW_SIZE)
                  & (measured[:, 1] >= 0) & (measured[:, 1] < VIEW_SIZE))
        grids[label] = (xy[inside], measured[inside])

    def matrices(vector_a):
        r_ab, t_ab = relative['a_b']
        direction_b = r_ab @ vector_a
        denominator_b = 1 + direction_b @ t_ab
        if abs(denominator_b) < 1e-6:
            raise ValueError('Common-plane transfer crosses camera center')
        vector_b = direction_b / denominator_b
        matrices_by_label = {}
        for label in pair_names:
            rotation, translation = relative[label]
            vector = vector_b if label == 'b_c' else vector_a
            matrices_by_label[label] = K @ (rotation + np.outer(translation, vector)) @ inv_K
        return matrices_by_label

    def residual(vector_a):
        try:
            Hs = matrices(vector_a)
        except ValueError:
            return np.full(sum(len(xy) * 2 for xy, _ in grids.values()), 1e4)
        pieces = []
        for label, (xy, measured) in grids.items():
            predicted = cv2.perspectiveTransform(xy[None], Hs[label])[0]
            pieces.append(np.nan_to_num((predicted - measured).ravel(),
                                        nan=1e4, posinf=1e4, neginf=-1e4))
        return np.concatenate(pieces)

    fitted = least_squares(residual, initial, loss='soft_l1', f_scale=2.,
                           max_nfev=300)
    Hs = matrices(fitted.x)
    rows = {}
    for label, (src, dst) in pair_names.items():
        xy, measured = grids[label]
        predicted = cv2.perspectiveTransform(xy[None], Hs[label])[0]
        rows[label] = {'common_court_grid_points': int(len(xy)),
                       'shared_plane_warp_disagreement_view_px':
                       stats(np.linalg.norm(predicted - measured, axis=1)),
                       'shared_plane_color_alignment':
                       _color_alignment(classes[src], classes[dst], Hs[label])}
    return {'fit_success': bool(fitted.success),
            'plane_vector_in_first_view': fitted.x.tolist(), 'pairs': rows}


def run_court_plane_common_pose_audit(work):
    """Fit one plane to all three saved-pose pair warps in each L triplet."""
    cv2.setNumThreads(4)
    work = Path(work).resolve()
    root = work / 'global_sfm'
    output = root / OUTPUT
    plane_result = json.loads((output / 'results.json').read_text())
    pose_audit = json.loads((output / 'pose_audit.json').read_text())
    names = [name for row in plane_result['rows'] for name in row['images']]
    model_path = root / 'ground_joint_ba/equal_control/sparse/0'
    views = {row['name']: row for row in json.loads(
        (root / DOWN_OUTPUT / 'views.json').read_text())['rows']}
    protocol = {'version': 1,
                'court_plane_result_sha256': file_digest(output / 'results.json'),
                'per_pair_pose_audit_sha256': file_digest(output / 'pose_audit.json'),
                'saved_visual_model_hashes': {p.name: file_digest(p)
                                              for p in sorted(model_path.glob('*.bin'))},
                'view_sha256': {name: views[name]['view_sha256'] for name in names},
                'fit': 'one source-view plane vector jointly across AB, AC, BC; soft_l1 2px',
                'grid_stride_view_px': 60,
                'scope': 'Read-only single-plane consistency check of saved visual poses against image-only court homographies.'}
    fingerprint = digest(protocol)
    protocol_path = output / 'common_pose_audit_protocol.json'
    if protocol_path.exists():
        if json.loads(protocol_path.read_text())['fingerprint'] != fingerprint:
            raise ValueError('Existing common-plane audit belongs to changed inputs')
    else:
        write_json(protocol_path, {**protocol, 'fingerprint': fingerprint,
                                   'frozen_at_utc': datetime.now(timezone.utc).isoformat()})
    reconstruction = pycolmap.Reconstruction(model_path)
    images = {image.name: image for image in reconstruction.images.values()}
    poses = {name: _virtual_pose(images[name]) for name in names}
    classes = {name: _class_map(views[name]['view'])[1] for name in names}
    rows = []
    for plane_row, pose_row in zip(plane_result['rows'], pose_audit['rows']):
        if plane_row['images'] != pose_row['images']:
            raise ValueError('Plane and saved-pose audit triplets differ')
        first = np.asarray(pose_row['pairs']['a_b']['best_plane_vector_in_source_view'])
        second = np.asarray(pose_row['pairs']['a_c']['best_plane_vector_in_source_view'])
        fitted = _common_pose_plane_fit(plane_row['images'], plane_row['pairs'],
                                        classes, poses, (first + second) / 2)
        rows.append({'images': plane_row['images'], **fitted})
    result = {'protocol_fingerprint': fingerprint, 'rows': rows,
              'guard': 'One plane for all three pairs is a stronger consistency test, but the saved visual poses and fitted fisheye intrinsics remain uncertain. A small residual is local compatibility, not proof of global pose or 3DGS image consistency.'}
    write_json(output / 'common_pose_audit.json', result)
    return result
