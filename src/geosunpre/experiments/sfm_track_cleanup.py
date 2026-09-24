"""Training-only SfM track cleanup against a frozen geometric baseline."""

from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path

import cv2
import numpy as np
import pycolmap
from scipy.spatial.transform import Rotation

from ..pipeline.global_reconstruction import load_global
from ..local_evaluation import evaluate_reconstruction, paired_common_support
from ..local_reconstruction import (assert_holdout_isolated, bounded_bundle_adjustment,
                                   cheirality_health)
from ..scene import digest, file_digest, load_scene, write_json
from .static_validation import _evaluate_stage, _stage_summary, checked_annotations
from ..track_consistency import reconstruction_coverage


OUTPUT = 'sfm_track_cleanup'
SOURCE = 'ground_joint_ba/equal_control/sparse/0'
MIN_MAX_TRIANGULATION_ANGLE_DEG = 2.0
MAX_TRAIN_REPROJECTION_ERROR_PX = 3.0
MIN_POINT_OBSERVATIONS = 3
MIN_IMAGE_OBSERVATIONS = 100
MIN_IMAGE_CELLS = 12
MAX_BA_ITERATIONS = 100


def _model_hashes(path):
    return {file.name: file_digest(file) for file in sorted(Path(path).glob('*.bin'))}


def _max_ray_angle_degrees(point, centers):
    rays = np.asarray([point.xyz - centers[int(element.image_id)]
                       for element in point.track.elements], dtype=float)
    rays /= np.linalg.norm(rays, axis=1)[:, None]
    return float(np.rad2deg(np.arccos(np.clip(np.min(rays @ rays.T), -1., 1.))))


def _image_residuals(rec, excluded_points):
    """Original-pixel reprojection errors for all still-eligible observations."""
    by_point = defaultdict(list)
    for image_id in sorted(rec.reg_image_ids()):
        image = rec.images[image_id]
        observations = [(index, observation)
                        for index, observation in enumerate(image.points2D)
                        if observation.has_point3D()
                        and int(observation.point3D_id) not in excluded_points]
        if not observations:
            continue
        xyz = np.asarray([rec.points3D[observation.point3D_id].xyz
                          for _, observation in observations], dtype=np.float64).reshape(-1, 1, 3)
        observed = np.asarray([observation.xy for _, observation in observations],
                              dtype=np.float64)
        fx, fy, cx, cy, *distortion = rec.cameras[image.camera_id].params
        K = np.asarray([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        pose = image.cam_from_world().matrix()
        rvec = cv2.Rodrigues(pose[:, :3])[0]
        projected = cv2.fisheye.projectPoints(
            xyz, rvec, pose[:, 3], K, np.asarray(distortion, dtype=np.float64)
        )[0].reshape(-1, 2)
        for (index, observation), error in zip(observations,
                                                np.linalg.norm(projected - observed, axis=1)):
            by_point[int(observation.point3D_id)].append(
                (int(image_id), int(index), float(error)))
    return by_point


def filter_training_tracks(rec):
    """Remove weak or conflicting training tracks without consulting validation."""
    before = {'points': rec.num_points3D(), 'observations': rec.compute_num_observations(),
              'coverage': reconstruction_coverage(rec)}
    centers = {int(image_id): rec.images[image_id].projection_center()
               for image_id in rec.reg_image_ids()}
    remove_points = {}
    for point_id in sorted(rec.points3D):
        point = rec.points3D[point_id]
        ids = [int(element.image_id) for element in point.track.elements]
        if len(ids) != len(set(ids)):
            remove_points[int(point_id)] = 'duplicate_image_association'
        elif _max_ray_angle_degrees(point, centers) < MIN_MAX_TRIANGULATION_ANGLE_DEG:
            remove_points[int(point_id)] = 'low_parallax'

    errors = _image_residuals(rec, remove_points)
    remove_observations = defaultdict(list)
    for point_id, rows in errors.items():
        rejected = [(image_id, index) for image_id, index, error in rows
                    if error > MAX_TRAIN_REPROJECTION_ERROR_PX]
        if len(rows) - len(rejected) < MIN_POINT_OBSERVATIONS:
            remove_points[point_id] = 'too_few_after_residual_filter'
        else:
            remove_observations[point_id] = rejected

    for point_id in sorted(remove_points):
        rec.delete_point3D(point_id)
    removed_observation_count = 0
    for point_id in sorted(remove_observations):
        if point_id not in rec.points3D:
            continue
        for image_id, index in remove_observations[point_id]:
            rec.delete_observation(image_id, index)
            removed_observation_count += 1
    rec.update_point_3d_errors()
    after = {'points': rec.num_points3D(), 'observations': rec.compute_num_observations(),
             'coverage': reconstruction_coverage(rec),
             'cheirality': cheirality_health(rec)}
    coverage = after['coverage']
    accepted = (rec.num_reg_images() == 536 and len(coverage['image_components']) == 1
                and coverage['min_observations'] >= MIN_IMAGE_OBSERVATIONS
                and coverage['min_occupied_cells'] >= MIN_IMAGE_CELLS
                and after['cheirality']['invalid_depth'] == 0)
    if not accepted:
        raise RuntimeError('Track cleanup violates registration, coverage or depth guard')
    return {'before': before, 'after': after,
            'removed_points_by_reason': dict(Counter(remove_points.values())),
            'removed_observations_from_retained_points': removed_observation_count,
            'total_removed_points': before['points'] - after['points'],
            'total_removed_observations': before['observations'] - after['observations'],
            'coverage_accepted': accepted,
            'decision_inputs': 'Training point tracks, current SfM rays, and native-fisheye training residuals only; no manual or frozen holdout observations.'}


def _protocol(root):
    source = root / SOURCE
    return {'version': 1, 'source_model_hashes': _model_hashes(source),
            'split_sha256': file_digest(root / 'split.json'),
            'triangle_split_sha256': file_digest(root / 'triangle_validation_split.json'),
            'manual_annotations_sha256': file_digest(
                root / 'static_validation/basketball-static-annotations.json'),
            'geometry_protocol_sha256': file_digest(root / 'geometry_protocol.json'),
            'min_max_triangulation_angle_deg': MIN_MAX_TRIANGULATION_ANGLE_DEG,
            'max_training_reprojection_error_original_px': MAX_TRAIN_REPROJECTION_ERROR_PX,
            'minimum_point_observations': MIN_POINT_OBSERVATIONS,
            'minimum_image_observations': MIN_IMAGE_OBSERVATIONS,
            'minimum_occupied_8x8_cells': MIN_IMAGE_CELLS,
            'max_ba_iterations': MAX_BA_ITERATIONS,
            'camera_model': 'OPENCV_FISHEYE',
            'independent_exposure_poses': True,
            'navigation_pose_or_las_used': False,
            'manual_or_frozen_holdout_used_for_filtering_or_ba': False,
            'scope': 'Development SfM candidate; original model and LFS outputs remain untouched.'}


def _prepare(root, output, split):
    initial = output / 'filtered_initial/sparse/0'
    audit_path = output / 'filter_audit.json'
    if initial.exists() or audit_path.exists():
        if not initial.exists() or not audit_path.exists():
            raise RuntimeError('Incomplete saved cleanup audit; inspect before resuming')
        audit = json.loads(audit_path.read_text())
        if _model_hashes(initial) != audit['initial_model_hashes']:
            raise ValueError('Filtered initial model changed')
        return audit
    rec = pycolmap.Reconstruction(root / SOURCE)
    assert_holdout_isolated(rec, split)
    audit = filter_training_tracks(rec)
    assert_holdout_isolated(rec, split)
    initial.mkdir(parents=True)
    rec.write(initial)
    audit['initial_model_hashes'] = _model_hashes(initial)
    write_json(audit_path, audit)
    return audit


def _pose_change(source, candidate):
    centers, rotations = [], []
    for image_id in sorted(source.reg_image_ids()):
        first, second = source.images[image_id], candidate.images[image_id]
        centers.append(float(np.linalg.norm(first.projection_center()
                                            - second.projection_center())))
        old = first.cam_from_world().matrix()[:, :3]
        new = second.cam_from_world().matrix()[:, :3]
        rotations.append(float(np.rad2deg(Rotation.from_matrix(new @ old.T).magnitude())))
    return {'center_world_median': float(np.median(centers)),
            'center_world_p90': float(np.quantile(centers, .9)),
            'center_world_max': float(np.max(centers)),
            'rotation_deg_median': float(np.median(rotations)),
            'rotation_deg_p90': float(np.quantile(rotations, .9)),
            'rotation_deg_max': float(np.max(rotations))}


def run_sfm_track_cleanup(work, stage='all'):
    cv2.setNumThreads(1)
    root = Path(work).resolve() / 'global_sfm'
    output = root / OUTPUT
    output.mkdir(exist_ok=True)
    protocol = _protocol(root)
    fingerprint = digest(protocol)
    path = output / 'protocol.json'
    if path.exists():
        if json.loads(path.read_text())['fingerprint'] != fingerprint:
            raise ValueError('SfM cleanup protocol changed; preserve the existing run')
    else:
        write_json(path, {**protocol, 'fingerprint': fingerprint,
                          'frozen_at_utc': datetime.now(timezone.utc).isoformat()})
    split = json.loads((root / 'split.json').read_text())
    audit = _prepare(root, output, split)
    if stage == 'audit':
        return {'output': str(output), 'filter_audit': audit}

    initial = output / 'filtered_initial/sparse/0'
    candidate_path = output / 'candidate/sparse/0'
    ba_path = output / 'candidate/ba.json'
    if candidate_path.exists() or ba_path.exists():
        if not candidate_path.exists() or not ba_path.exists():
            raise RuntimeError('Incomplete saved cleanup BA; inspect before resuming')
        ba = json.loads(ba_path.read_text())
        if _model_hashes(candidate_path) != ba['model_hashes']:
            raise ValueError('Saved cleanup candidate changed')
        rec = pycolmap.Reconstruction(candidate_path)
    else:
        rec = pycolmap.Reconstruction(initial)
        assert_holdout_isolated(rec, split)
        geometry = json.loads((root / 'geometry_protocol.json').read_text())
        initial_cameras = {int(key): np.asarray(value, dtype=float)
                           for key, value in geometry['calibration'].items()}
        print('SfM cleanup BA start', rec.num_points3D(), rec.compute_num_observations(),
              flush=True)
        ba = bounded_bundle_adjustment(
            rec, 'principal', initial_cameras, protect_depth=True,
            show_progress=True, max_iterations=MAX_BA_ITERATIONS)
        assert_holdout_isolated(rec, split)
        candidate_path.mkdir(parents=True)
        rec.write(candidate_path)
        ba['model_hashes'] = _model_hashes(candidate_path)
        write_json(ba_path, ba)
    if stage == 'solve':
        return {'output': str(output), 'ba': ba['termination_type']}

    comparison_path = output / 'comparison.json'
    if comparison_path.exists():
        result = json.loads(comparison_path.read_text())
        if result.get('protocol_fingerprint') != fingerprint:
            raise ValueError('Saved cleanup comparison protocol changed')
        return result
    source = pycolmap.Reconstruction(root / SOURCE)
    assert_holdout_isolated(source, split)
    assert_holdout_isolated(rec, split)
    scene = load_scene(root)
    labels = json.loads((root / 'static_validation/basketball-static-annotations.json').read_text())
    candidates = json.loads((root / 'static_validation/candidates.json').read_text())
    accepted, counts = checked_annotations(labels, candidates, scene)
    manual = {}
    for name, model in [('equal_control', source), ('cleaned', rec)]:
        rows = _evaluate_stage(model, accepted, scene)
        write_json(output / f'{name}_manual_rows.json', rows)
        manual[name] = _stage_summary(rows)

    validation = output / 'validation'
    _, scene, _, features = load_global(work)
    triangle = json.loads((root / 'triangle_validation_split.json').read_text())
    triangle_summary = {}
    for name, model in [('equal_control', source), ('cleaned', rec)]:
        destination = validation / 'stages' / name / 'triangle_validation'
        destination.mkdir(parents=True, exist_ok=True)
        triangle_summary[name] = evaluate_reconstruction(
            model, scene, features, triangle, destination)['all_candidate_queries']
    common, _, _ = paired_common_support(
        validation, ['equal_control', 'cleaned'], 'triangle_validation',
        baseline_stage='equal_control')
    result = {'protocol_fingerprint': fingerprint, 'filter_audit': audit,
              'ba': ba, 'manual_annotation_counts': counts,
              'manual': manual,
              'manual_paired': _manual_paired(
                  output / 'equal_control_manual_rows.json',
                  output / 'cleaned_manual_rows.json'),
              'triangle': triangle_summary, 'triangle_common_support': common,
              'pose_change': _pose_change(source, rec),
              'intrinsics_delta': {str(cid): (rec.cameras[cid].params
                                               - source.cameras[cid].params).tolist()
                                   for cid in source.cameras},
              'candidate_is_not_promoted': True,
              'note': 'Validation is reused development evidence from the same capture. No LFS training or input changes.'}
    write_json(comparison_path, result)
    return result


def _manual_paired(first, second):
    old = {row['id']: row for row in json.loads(first.read_text())}
    new = {row['id']: row for row in json.loads(second.read_text())}
    if set(old) != set(new):
        raise ValueError('Manual validation membership changed')
    rows = [(old[key]['query_error_px'], new[key]['query_error_px']) for key in old
            if old[key]['query_error_px'] is not None
            and new[key]['query_error_px'] is not None]
    delta = np.asarray([after - before for before, after in rows])
    return {'paired_finite': len(delta),
            'median_delta_original_px': float(np.median(delta)) if len(delta) else None,
            'mean_delta_original_px': float(np.mean(delta)) if len(delta) else None,
            'fraction_improved': float(np.mean(delta < 0)) if len(delta) else None}
