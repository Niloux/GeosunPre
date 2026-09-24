"""Training-only floor weighting probe with fixed 3D points and calibration.

This tests whether finite extra weight on reliable nearby floor tracks helps
independent-exposure poses. It is not a replacement for full weighted BA.
"""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import cv2
import numpy as np
import pycolmap
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from ..camera import Camera
from ..local_evaluation import error_summary, triangulate_support
from ..local_reconstruction import assert_holdout_isolated, cheirality_health
from ..scene import digest, file_digest, load_scene, write_json
from .static_validation import _evaluate_stage, _stage_summary, checked_annotations


OUTPUT = 'ground_pose_probe'
SOURCE = 'stages/shared_calibration/sparse/0'
REFERENCE_FLOOR_SIDE = np.array([-0.17629417041114237, 0.9793657897849057,
                                 0.09880797173322986])
PLANE_SEED_BAND = .06
PLANE_HUBER_SCALE = .01
PLANE_INLIER_DISTANCE = .03
MAX_POINT_ERROR_PX = 2.0
MIN_TRACK_LENGTH = 4
HUBER_PX = 2.0
POSE_IRLS_ROUNDS = 3
MAX_INNER_EVALUATIONS = 30
VARIANTS = {
    'equal_weight': {'near': 1.0, 'far': 1.0},
    'floor_1p5': {'near': 1.5, 'far': 1.5},
    'near_floor_2p0': {'near': 2.0, 'far': 1.25},
}


def fit_floor_from_training(rec):
    """Fit the thin plane on the visually checked floor side of the trajectory."""
    images = [rec.images[i] for i in sorted(rec.images) if rec.images[i].has_pose]
    centers = np.asarray([im.projection_center() for im in images], dtype=float)
    center = centers.mean(axis=0)
    values, axes = np.linalg.eigh(np.cov(centers.T))
    normal = axes[:, 0]
    if normal @ REFERENCE_FLOOR_SIDE < 0:
        normal = -normal
    tangent = axes[:, 1:]
    point_ids = np.asarray(sorted(rec.points3D), dtype=np.int64)
    xyz = np.asarray([rec.points3D[int(pid)].xyz for pid in point_ids], dtype=float)
    uv = (xyz - center) @ tangent
    height = (xyz - center) @ normal
    peak = float(np.median(height[height > np.quantile(height, .6)]))
    seed = np.abs(height - peak) < PLANE_SEED_BAND
    if seed.sum() < 1000:
        raise ValueError('Insufficient floor-side plane candidates')
    design = np.column_stack((uv[seed], np.ones(int(seed.sum()))))
    fit = least_squares(lambda p: design @ p - height[seed],
                        [0., 0., peak], loss='huber',
                        f_scale=PLANE_HUBER_SCALE, max_nfev=30)
    if not fit.success:
        raise RuntimeError(f'Floor-side plane fit failed: {fit.message}')
    residual = height - (np.column_stack((uv, np.ones(len(uv)))) @ fit.x)
    plane_points = np.abs(residual) <= PLANE_INLIER_DISTANCE
    errors = np.asarray([rec.points3D[int(pid)].error for pid in point_ids])
    lengths = np.asarray([rec.points3D[int(pid)].track.length() for pid in point_ids])
    reliable = plane_points & (errors <= MAX_POINT_ERROR_PX) & (lengths >= MIN_TRACK_LENGTH)
    if reliable.sum() < 1000:
        raise ValueError('Insufficient reliable training-floor points')
    selected_ids = {int(pid) for pid in point_ids[reliable]}
    by_id = {int(i): im for i, im in rec.images.items() if im.has_pose}
    distances = []
    for pid in selected_ids:
        point = rec.points3D[pid]
        for element in point.track.elements:
            im = by_id[int(element.image_id)]
            distances.append(float(np.linalg.norm(point.xyz - im.projection_center())))
    near_cutoff = float(np.median(distances))
    floor_normal = normal - tangent @ fit.x[:2]
    floor_normal /= np.linalg.norm(floor_normal)
    info = {'camera_center_pca_variances': values.tolist(),
            'camera_center': center.tolist(), 'floor_normal': floor_normal.tolist(),
            'floor_offset_from_mean_camera_center': float(fit.x[2]),
            'plane_seed_peak': peak, 'plane_seed_candidates': int(seed.sum()),
            'plane_inlier_points': int(plane_points.sum()),
            'reliable_floor_points': int(reliable.sum()),
            'reliable_floor_observations': len(distances),
            'near_cutoff_world_units': near_cutoff,
            'reliable_observation_distance_quantiles': np.quantile(distances,[.1,.25,.5,.75,.9]).tolist(),
            'floor_plane_is_visual_model_estimate': True,
            'note': 'The normal sign and thin plane were checked in native images. They are not navigation/LAS truth; point error and track length are training-only gates.'}
    return selected_ids, near_cutoff, info


def pose_projection(q, xyz, camera):
    """Original-pixel OpenCV fisheye projection and 2N-by-6 pose Jacobian."""
    projected, jacobian = cv2.fisheye.projectPoints(
        np.ascontiguousarray(xyz, dtype=np.float64).reshape(-1, 1, 3),
        np.asarray(q[:3], dtype=np.float64), np.asarray(q[3:], dtype=np.float64),
        camera['K'], camera['D'])
    return projected.reshape(-1, 2), jacobian[:, 8:14]


def fit_image_pose(q0, xyz, observed, camera, prior_weights):
    """Three IRLS rounds approximate per-2D-observation weighted Huber BA."""
    q = q0.copy()
    solver = []
    for _ in range(POSE_IRLS_ROUNDS):
        predicted, _ = pose_projection(q, xyz, camera)
        norm = np.linalg.norm(predicted - observed, axis=1)
        huber_weight = np.minimum(1., HUBER_PX / np.maximum(norm, 1e-12))
        sqrt_weight = np.sqrt(prior_weights * huber_weight)
        scale = np.repeat(sqrt_weight, 2)

        def residual(value):
            projected, _ = pose_projection(value, xyz, camera)
            return ((projected - observed).ravel() * scale)

        def jacobian(value):
            _, jac = pose_projection(value, xyz, camera)
            return jac * scale[:, None]

        result = least_squares(residual, q, jac=jacobian, method='lm',
                               max_nfev=MAX_INNER_EVALUATIONS,
                               ftol=1e-8, xtol=1e-8, gtol=1e-8)
        if not np.isfinite(result.x).all():
            raise RuntimeError('Nonfinite pose refinement')
        q = result.x
        solver.append({'status': int(result.status), 'nfev': result.nfev,
                       'cost': float(result.cost)})
    return q, solver


def camera_data(cam):
    fx, fy, cx, cy, *distortion = map(float, cam.params)
    return {'K': np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float),
            'D': np.asarray(distortion, dtype=float)}


def image_training_observations(rec, im, floor_ids, near_cutoff, variant,
                                reference_center=None):
    rows = [(int(p.point3D_id), np.asarray(p.xy, dtype=float))
            for p in im.points2D if p.has_point3D()]
    ids = np.asarray([pid for pid, _ in rows], dtype=np.int64)
    xyz = np.asarray([rec.points3D[int(pid)].xyz for pid in ids], dtype=float)
    observed = np.asarray([xy for _, xy in rows], dtype=float)
    ground = np.fromiter((int(pid) in floor_ids for pid in ids), count=len(ids), dtype=bool)
    center = im.projection_center() if reference_center is None else reference_center
    distance = np.linalg.norm(xyz - np.asarray(center)[None], axis=1)
    near = ground & (distance <= near_cutoff)
    far = ground & ~near
    weights = np.ones(len(ids), dtype=float)
    weights[near] = VARIANTS[variant]['near']
    weights[far] = VARIANTS[variant]['far']
    return xyz, observed, weights, near, far


def _training_error_summary(parts):
    return error_summary(np.concatenate(parts) if parts else np.array([]))


def _manual_comparison(base_rows, candidate_rows):
    before = {r['id']: r for r in base_rows}
    after = {r['id']: r for r in candidate_rows}
    if set(before) != set(after):
        raise RuntimeError('Manual candidates changed during pose probe')
    pairs = [(before[k]['query_error_px'], after[k]['query_error_px'])
             for k in sorted(before)
             if before[k]['query_error_px'] is not None and after[k]['query_error_px'] is not None]
    deltas = np.asarray([b - a for a, b in pairs])
    return {'common_finite': len(deltas),
            'median_delta_px': float(np.median(deltas)) if len(deltas) else None,
            'fraction_improved': float(np.mean(deltas < 0)) if len(deltas) else None,
            'improved': int(np.count_nonzero(deltas < 0)),
            'worsened': int(np.count_nonzero(deltas > 0))}


def run_ground_pose_probe(work):
    cv2.setNumThreads(1)
    root = Path(work).resolve() / 'global_sfm'
    out = root / OUTPUT
    out.mkdir(exist_ok=True)
    model = root / SOURCE
    scene = load_scene(root)
    split = json.loads((root / 'split.json').read_text())
    labels_path = root / 'static_validation/basketball-static-annotations.json'
    earlier_result = out / 'comparison.json'
    if earlier_result.exists():
        earlier_labels = json.loads(earlier_result.read_text()).get('official_annotations_sha256')
        if earlier_labels != file_digest(labels_path):
            raise ValueError('Official annotations changed; use a fresh pose probe output')
    labels = json.loads(labels_path.read_text())
    candidates = json.loads((root / 'static_validation/candidates.json').read_text())
    accepted, annotation_counts = checked_annotations(labels, candidates, scene)
    rec = pycolmap.Reconstruction(model)
    assert_holdout_isolated(rec, split)
    floor_ids, near_cutoff, plane = fit_floor_from_training(rec)
    protocol = {'version': 1, 'source_model_hashes': {p.name: file_digest(p)
                for p in sorted(model.glob('*.bin'))},
                'scene_fingerprint': scene['fingerprint'],
                'split_sha256': file_digest(root / 'split.json'),
                'floor_reference_side': REFERENCE_FLOOR_SIDE.tolist(),
                'plane_seed_band': PLANE_SEED_BAND,
                'plane_huber_scale': PLANE_HUBER_SCALE,
                'plane_inlier_distance': PLANE_INLIER_DISTANCE,
                'max_training_point_error_px': MAX_POINT_ERROR_PX,
                'min_training_track_length': MIN_TRACK_LENGTH,
                'near_distance': 'Median camera-to-point distance of selected training-floor observations',
                'variants': VARIANTS, 'pose_irls_rounds': POSE_IRLS_ROUNDS,
                'pose_huber_px': HUBER_PX, 'max_inner_evaluations': MAX_INNER_EVALUATIONS,
                'heldout_queries_used_for_weights_or_optimization': False,
                'optimization': 'Independent image pose only; all 3D points and both fisheye calibrations fixed',
                'scope': 'Exploratory pose weighting probe, not full weighted BA or 3DGS acceptance.'}
    fingerprint = digest(protocol)
    protocol_path = out / 'protocol.json'
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()).get('fingerprint') != fingerprint:
            raise ValueError('Ground pose probe protocol changed; use a fresh output')
    else:
        write_json(protocol_path, {**protocol, 'fingerprint': fingerprint,
                   'frozen_at_utc': datetime.now(timezone.utc).isoformat()})
    write_json(out / 'floor_plane.json', plane)
    base_rows = _evaluate_stage(rec, accepted, scene)
    baseline = {'manual': _stage_summary(base_rows), 'training': {},
                'registered': rec.num_reg_images(), 'points': rec.num_points3D(),
                'observations': rec.compute_num_observations()}
    all_training = []
    for im in rec.images.values():
        if not im.has_pose:
            continue
        xyz, observed, _, _, _ = image_training_observations(rec, im, floor_ids,
                                                               near_cutoff, 'equal_weight')
        pose = im.cam_from_world().matrix()
        q = np.r_[cv2.Rodrigues(pose[:, :3])[0].ravel(), pose[:, 3]]
        predicted, _ = pose_projection(q, xyz, camera_data(rec.cameras[im.camera_id]))
        all_training.append(np.linalg.norm(predicted - observed, axis=1))
    baseline['training']['all'] = _training_error_summary(all_training)
    results = {'protocol_fingerprint': fingerprint,
               'official_annotations_sha256': file_digest(labels_path),
               'annotation_counts': annotation_counts,
               'floor_plane': plane, 'baseline': baseline, 'variants': {},
               'note': 'All manual queries stay out of optimization. Their values are exploratory because this same capture and its earlier results are already known.'}
    write_json(out / 'comparison.json', results)
    for variant in VARIANTS:
        started = time.monotonic()
        candidate = pycolmap.Reconstruction(model)
        changes = []
        training = []
        near_total = far_total = 0
        for ordinal, image_id in enumerate(sorted(candidate.reg_image_ids()), 1):
            im = candidate.images[image_id]
            xyz, observed, weights, near_mask, far_mask = image_training_observations(
                candidate, im, floor_ids, near_cutoff, variant)
            near_count, far_count = int(near_mask.sum()), int(far_mask.sum())
            near_total += near_count; far_total += far_count
            pose = im.cam_from_world().matrix()
            q0 = np.r_[cv2.Rodrigues(pose[:, :3])[0].ravel(), pose[:, 3]]
            q, solver = fit_image_pose(q0, xyz, observed,
                                       camera_data(candidate.cameras[im.camera_id]), weights)
            new_R = cv2.Rodrigues(q[:3])[0]
            new_pose = np.column_stack((new_R, q[3:]))
            im.frame.set_cam_from_world(im.camera_id, pycolmap.Rigid3d(new_pose))
            old_center = -pose[:, :3].T @ pose[:, 3]
            new_center = im.projection_center()
            rotation_deg = float(np.rad2deg(Rotation.from_matrix(new_R @ pose[:, :3].T).magnitude()))
            translation = float(np.linalg.norm(new_center - old_center))
            changes.append({'name': im.name, 'camera': 'L' if im.camera_id == 1 else 'R',
                            'rotation_deg': rotation_deg, 'translation_world': translation,
                            'near_floor_observations': near_count,
                            'far_floor_observations': far_count,
                            'training_observations': len(xyz),
                            'solver': solver})
            projected, _ = pose_projection(q, xyz, camera_data(candidate.cameras[im.camera_id]))
            training.append(np.linalg.norm(projected - observed, axis=1))
            if ordinal % 100 == 0 or ordinal == candidate.num_reg_images():
                print(f'{variant}: pose {ordinal}/{candidate.num_reg_images()}', flush=True)
        assert_holdout_isolated(candidate, split)
        depth = cheirality_health(candidate)
        height = abs(plane['floor_offset_from_mean_camera_center'])
        translation_limit = .1 * height
        rotation_limit = 1.
        changed_t = np.asarray([r['translation_world'] for r in changes])
        changed_r = np.asarray([r['rotation_deg'] for r in changes])
        over_guard = [r['name'] for r in changes
                      if r['translation_world'] > translation_limit or r['rotation_deg'] > rotation_limit]
        result = {'registered': candidate.num_reg_images(),
                  'points': candidate.num_points3D(),
                  'observations': candidate.compute_num_observations(),
                  'elapsed_seconds': time.monotonic() - started,
                  'near_floor_observations': near_total,
                  'far_floor_observations': far_total,
                  'training': {'all': _training_error_summary(training)},
                  'pose_change': {
                      'translation_world_median': float(np.median(changed_t)),
                      'translation_world_p90': float(np.quantile(changed_t, .9)),
                      'translation_world_max': float(np.max(changed_t)),
                      'rotation_deg_median': float(np.median(changed_r)),
                      'rotation_deg_p90': float(np.quantile(changed_r, .9)),
                      'rotation_deg_max': float(np.max(changed_r)),
                      'translation_guard_world': translation_limit,
                      'rotation_guard_deg': rotation_limit,
                      'images_over_guard': over_guard},
                  'cheirality': depth,
                  'passes_pose_health_guard': not over_guard and not depth['invalid_depth']}
        stage_dir = out / variant
        stage_dir.mkdir(exist_ok=True)
        model_dir = stage_dir / 'sparse/0'
        model_dir.mkdir(parents=True, exist_ok=True)
        candidate.write(model_dir)
        result['model_hashes'] = {p.name: file_digest(p)
                                  for p in sorted(model_dir.glob('*.bin'))}
        write_json(stage_dir / 'pose_changes.json', changes)
        rows = _evaluate_stage(candidate, accepted, scene)
        result['manual'] = _stage_summary(rows)
        result['manual_paired_vs_source'] = _manual_comparison(base_rows, rows)
        write_json(stage_dir / 'manual_rows.json', rows)
        write_json(stage_dir / 'result.json', result)
        results['variants'][variant] = result
        write_json(out / 'comparison.json', results)
        print(variant, 'manual median', result['manual']['all']['median'],
              'P90', result['manual']['all']['p90'],
              'pose guard', result['passes_pose_health_guard'], flush=True)
    return results


def analyze_ground_pose_probe(work):
    """Read-only group analysis on the saved models and frozen manual labels."""
    root = Path(work).resolve() / 'global_sfm'
    out = root / OUTPUT
    comparison = json.loads((out / 'comparison.json').read_text())
    labels_path = root / 'static_validation/basketball-static-annotations.json'
    if comparison['official_annotations_sha256'] != file_digest(labels_path):
        raise ValueError('Official annotations changed after the pose probe')
    scene = load_scene(root)
    labels = json.loads(labels_path.read_text())
    candidates = json.loads((root / 'static_validation/candidates.json').read_text())
    accepted, _ = checked_annotations(labels, candidates, scene)
    rec = pycolmap.Reconstruction(root / SOURCE)
    split = json.loads((root / 'split.json').read_text())
    assert_holdout_isolated(rec, split)
    frozen = json.loads((out / 'protocol.json').read_text())
    if frozen['fingerprint'] != comparison['protocol_fingerprint']:
        raise ValueError('Pose probe protocol does not match the saved results')
    if frozen['source_model_hashes'] != {p.name: file_digest(p)
                                         for p in sorted((root / SOURCE).glob('*.bin'))}:
        raise ValueError('Source reconstruction changed after the pose probe')
    floor_ids, near_cutoff, plane = fit_floor_from_training(rec)
    center = np.asarray(plane['camera_center'])
    normal = np.asarray(plane['floor_normal'])
    offset = plane['floor_offset_from_mean_camera_center']
    images = {im.name: im for im in rec.images.values() if im.has_pose}
    cameras = {cid: Camera(str(cid), c.width, c.height, tuple(c.params), (0, 0))
               for cid, c in rec.cameras.items()}
    manual_classes = {}
    details = []
    for row, _, coords in accepted:
        names = [obs['name'] for obs in row['observations']]
        ims = [images[name] for name in names]
        cs = [cameras[im.camera_id] for im in ims[:2]]
        ps = [im.cam_from_world().matrix() for im in ims[:2]]
        point, support_error, angle = triangulate_support(cs, ps, coords[:2])
        if point is None or support_error > 3 or angle < 1:
            kind, plane_distance, camera_distance = 'unsupported', None, None
        else:
            plane_distance = float(abs((point - center) @ normal - offset))
            camera_distance = float(np.linalg.norm(point - ims[2].projection_center()))
            kind = ('near_floor' if camera_distance <= near_cutoff else 'far_floor') \
                   if plane_distance <= .05 else 'off_plane'
        manual_classes[row['id']] = kind
        details.append({'id': row['id'], 'class': kind,
                        'distance_to_floor_plane': plane_distance,
                        'query_camera_distance': camera_distance,
                        'support_error_px': float(support_error) if np.isfinite(support_error) else None,
                        'support_angle_deg': float(angle)})

    source_images = {im.name: im for im in rec.images.values() if im.has_pose}

    def training_groups(model):
        parts = {key: [] for key in ('all', 'near_floor', 'far_floor', 'other')}
        for im in model.images.values():
            if not im.has_pose:
                continue
            xyz, observed, _, near, far = image_training_observations(
                model, im, floor_ids, near_cutoff, 'equal_weight',
                reference_center=source_images[im.name].projection_center())
            pose = im.cam_from_world().matrix()
            q = np.r_[cv2.Rodrigues(pose[:, :3])[0].ravel(), pose[:, 3]]
            predicted, _ = pose_projection(q, xyz, camera_data(model.cameras[im.camera_id]))
            errors = np.linalg.norm(predicted - observed, axis=1)
            parts['all'].append(errors)
            parts['near_floor'].append(errors[near])
            parts['far_floor'].append(errors[far])
            parts['other'].append(errors[~(near | far)])
        return {key: _training_error_summary(value) for key, value in parts.items()}

    baseline_rows = _evaluate_stage(rec, accepted, scene)
    result = {'source_protocol_fingerprint': comparison['protocol_fingerprint'],
              'official_annotations_sha256': comparison['official_annotations_sha256'],
              'manual_proxy_rule': 'Two manually confirmed supports triangulated only with source poses; max support error <=3px and angle >=1deg. Absolute distance <=0.05 source-world units to source floor plane; near if query-camera distance <= training-only cutoff. No query error in membership.',
              'manual_proxy_counts': dict(Counter(manual_classes.values())),
              'manual_proxy_details': details,
              'stages': {'source': {'training': training_groups(rec),
                                    'manual': {group: _stage_summary([r for r in baseline_rows
                                        if manual_classes[r['id']] == group])['all']
                                               for group in ('near_floor','far_floor','off_plane','unsupported')}}},
              'paired_vs_source': {},
              'limitation': 'Manual floor groups are source-model geometric proxies, not independent semantic labels. Both training and manual groups are from the same capture.'}
    for variant in VARIANTS:
        stage = out / variant
        saved_hashes = comparison['variants'][variant]['model_hashes']
        actual_hashes = {p.name: file_digest(p)
                         for p in sorted((stage / 'sparse/0').glob('*.bin'))}
        if saved_hashes != actual_hashes:
            raise ValueError(f'Saved {variant} model changed after the pose probe')
        model = pycolmap.Reconstruction(stage / 'sparse/0')
        assert_holdout_isolated(model, split)
        rows = json.loads((stage / 'manual_rows.json').read_text())
        result['stages'][variant] = {'training': training_groups(model),
                                      'manual': {group: _stage_summary([r for r in rows
                                          if manual_classes[r['id']] == group])['all']
                                                 for group in ('near_floor','far_floor','off_plane','unsupported')}}
        result['paired_vs_source'][variant] = {
            group: _manual_comparison(
                [r for r in baseline_rows if manual_classes[r['id']] == group],
                [r for r in rows if manual_classes[r['id']] == group])
            for group in ('near_floor','far_floor','off_plane','unsupported')}
    write_json(out / 'target_analysis.json', result)
    print('Ground pose target analysis:', result['manual_proxy_counts'], flush=True)
    return result
