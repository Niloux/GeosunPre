"""Controlled full-scene native BA continuation with finite ground weights."""
from datetime import datetime, timezone
import json
from pathlib import Path

import cv2
import numpy as np
import pycolmap
from scipy.spatial.transform import Rotation

from .ground_pose_probe import (SOURCE, _manual_comparison, camera_data,
                                fit_floor_from_training, pose_projection)
from ..local_evaluation import error_summary
from ..local_reconstruction import assert_holdout_isolated, bounded_bundle_adjustment
from ..scene import digest, file_digest, load_scene, write_json
from .static_validation import _evaluate_stage, _stage_summary, checked_annotations
from ..weighted_bundle import add_weighted_ground_factors


OUTPUT = 'ground_joint_ba'
MAX_ITERATIONS = 40
WEIGHTS = {'equal_control': {'near': 1., 'far': 1.},
           'near_floor_2p0': {'near': 2., 'far': 1.25}}
GROUPS = ('near_floor', 'far_floor', 'off_plane', 'unsupported')


def _selected_observations(source, floor_ids, near_cutoff):
    near, far = set(), set()
    for image_id in sorted(source.reg_image_ids()):
        image = source.images[image_id]
        center = image.projection_center()
        for observation_index, observation in enumerate(image.points2D):
            if not observation.has_point3D() or int(observation.point3D_id) not in floor_ids:
                continue
            key = (image_id, observation_index)
            point = source.points3D[observation.point3D_id]
            (near if np.linalg.norm(point.xyz - center) <= near_cutoff else far).add(key)
    return near, far


def _training_groups(model, near, far):
    parts = {key: [] for key in ('all', 'near_floor', 'far_floor', 'other')}
    for image_id in sorted(model.reg_image_ids()):
        image = model.images[image_id]
        rows = [(j, int(p.point3D_id), np.asarray(p.xy, dtype=float))
                for j, p in enumerate(image.points2D) if p.has_point3D()]
        xyz = np.asarray([model.points3D[pid].xyz for _, pid, _ in rows], dtype=float)
        observed = np.asarray([xy for _, _, xy in rows], dtype=float)
        pose = image.cam_from_world().matrix()
        q = np.r_[cv2.Rodrigues(pose[:, :3])[0].ravel(), pose[:, 3]]
        predicted, _ = pose_projection(q, xyz, camera_data(model.cameras[image.camera_id]))
        errors = np.linalg.norm(predicted - observed, axis=1)
        near_mask = np.fromiter(((image_id, j) in near for j, _, _ in rows),
                                count=len(rows), dtype=bool)
        far_mask = np.fromiter(((image_id, j) in far for j, _, _ in rows),
                               count=len(rows), dtype=bool)
        parts['all'].append(errors)
        parts['near_floor'].append(errors[near_mask])
        parts['far_floor'].append(errors[far_mask])
        parts['other'].append(errors[~(near_mask | far_mask)])
    return {key: error_summary(np.concatenate(value)) for key, value in parts.items()}


def _manual_groups(rows, classes):
    return {group: _stage_summary([row for row in rows if classes[row['id']] == group])['all']
            for group in GROUPS}


def _model_change(source, candidate, floor_height):
    translations, rotations, point_movements = [], [], []
    for image_id in sorted(source.reg_image_ids()):
        old = source.images[image_id]
        new = candidate.images[image_id]
        translations.append(float(np.linalg.norm(new.projection_center() - old.projection_center())))
        old_R = old.cam_from_world().matrix()[:, :3]
        new_R = new.cam_from_world().matrix()[:, :3]
        rotations.append(float(np.rad2deg(Rotation.from_matrix(new_R @ old_R.T).magnitude())))
    for pid in sorted(source.points3D):
        point_movements.append(float(np.linalg.norm(candidate.points3D[pid].xyz
                                                   - source.points3D[pid].xyz)))
    values = {'pose_center_world': translations, 'pose_rotation_deg': rotations,
              'point_world': point_movements}
    summary = {key: {'median': float(np.median(v)), 'p90': float(np.quantile(v, .9)),
                     'max': float(np.max(v))} for key, v in values.items()}
    summary['pose_guard'] = {'max_center_world': .1 * floor_height,
                             'max_rotation_deg': 1.,
                             'passes': summary['pose_center_world']['max'] <= .1 * floor_height
                             and summary['pose_rotation_deg']['max'] <= 1.}
    return summary


def run_ground_joint_ba(work):
    cv2.setNumThreads(1)
    root = Path(work).resolve() / 'global_sfm'
    output = root / OUTPUT
    output.mkdir(exist_ok=True)
    model_path = root / SOURCE
    labels_path = root / 'static_validation/basketball-static-annotations.json'
    scene = load_scene(root)
    labels = json.loads(labels_path.read_text())
    candidates = json.loads((root / 'static_validation/candidates.json').read_text())
    accepted, annotation_counts = checked_annotations(labels, candidates, scene)
    split_path = root / 'split.json'
    split = json.loads(split_path.read_text())
    source = pycolmap.Reconstruction(model_path)
    assert_holdout_isolated(source, split)
    floor_ids, near_cutoff, floor_info = fit_floor_from_training(source)
    near, far = _selected_observations(source, floor_ids, near_cutoff)
    pose_result = json.loads((root / 'ground_pose_probe/comparison.json').read_text())
    expected = pose_result['variants']['near_floor_2p0']
    if (len(near), len(far)) != (expected['near_floor_observations'],
                                   expected['far_floor_observations']):
        raise ValueError('Training-floor membership changed from the frozen pose probe')
    proxy_path = root / 'ground_pose_probe/target_analysis.json'
    proxies = json.loads(proxy_path.read_text())
    classes = {row['id']: row['class'] for row in proxies['manual_proxy_details']}
    if set(classes) != {row['id'] for row, _, _ in accepted}:
        raise ValueError('Manual proxy classes do not match official annotations')
    geometry = json.loads((root / 'geometry_protocol.json').read_text())
    initial_cameras = {int(cid): np.asarray(values, dtype=float)
                       for cid, values in geometry['calibration'].items()}
    protocol = {'version': 1,
                'source_model_hashes': {p.name: file_digest(p)
                                        for p in sorted(model_path.glob('*.bin'))},
                'scene_fingerprint': scene['fingerprint'],
                'geometry_protocol_sha256': file_digest(root / 'geometry_protocol.json'),
                'split_sha256': file_digest(split_path),
                'official_annotations_sha256': file_digest(labels_path),
                'manual_proxy_sha256': file_digest(proxy_path),
                'ground_pose_protocol_sha256': file_digest(root / 'ground_pose_probe/protocol.json'),
                'ground_membership': {'near': len(near), 'far': len(far),
                                      'near_cutoff_world': near_cutoff},
                'weights': WEIGHTS, 'max_iterations': MAX_ITERATIONS,
                'ba_options': 'Native pycolmap Ceres BA with original 2px Huber factors, bounded shared OPENCV_FISHEYE intrinsics, independent exposure poses, variable 3D points, two-pose gauge and positive-depth barriers. Added robust factors contribute weight minus one only for reliable ground observations.',
                'heldout_queries_used_for_optimization': False,
                'navigation_pose_or_las_used': False,
                'scope': 'Development comparison; not independent 3DGS acceptance'}
    fingerprint = digest(protocol)
    protocol_path = output / 'protocol.json'
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()).get('fingerprint') != fingerprint:
            raise ValueError('Joint ground BA protocol changed; use a fresh output directory')
    else:
        write_json(protocol_path, {**protocol, 'fingerprint': fingerprint,
                   'frozen_at_utc': datetime.now(timezone.utc).isoformat()})

    base_rows = _evaluate_stage(source, accepted, scene)
    comparison = {'protocol_fingerprint': fingerprint,
                  'annotation_counts': annotation_counts,
                  'floor_plane': floor_info,
                  'baseline': {'registered': source.num_reg_images(),
                               'points': source.num_points3D(),
                               'observations': source.compute_num_observations(),
                               'manual': _stage_summary(base_rows),
                               'manual_ground_proxy': _manual_groups(base_rows, classes),
                               'training': _training_groups(source, near, far)},
                  'variants': {},
                  'note': 'All human-checked points remain out of BA. They are reused development validation from the same capture.'}
    write_json(output / 'comparison.json', comparison)
    for name, weights in WEIGHTS.items():
        candidate = pycolmap.Reconstruction(model_path)
        assert_holdout_isolated(candidate, split)

        def factor_hook(problem, rec):
            costs, losses, counts = add_weighted_ground_factors(
                problem, rec, floor_ids, near_cutoff, weights,
                near_membership=near)
            if (counts['near'], counts['far']) != (len(near), len(far)):
                raise RuntimeError('Added BA factors do not match frozen ground membership')
            print(name, 'extra factors', len(costs), 'selected', counts, flush=True)
            return (costs, losses), {'count': len(costs), 'selected': counts,
                                     'weights': weights}

        print('Joint ground BA', name, 'start', flush=True)
        ba = bounded_bundle_adjustment(candidate, 'principal', initial_cameras,
                                       protect_depth=True, show_progress=True,
                                       factor_hook=factor_hook,
                                       max_iterations=MAX_ITERATIONS)
        assert_holdout_isolated(candidate, split)
        if (candidate.num_reg_images(), candidate.num_points3D(), candidate.compute_num_observations()) != (
                source.num_reg_images(), source.num_points3D(), source.compute_num_observations()):
            raise RuntimeError('Joint ground BA changed reconstruction topology')
        stage = output / name
        model_out = stage / 'sparse/0'
        model_out.mkdir(parents=True, exist_ok=True)
        candidate.write(model_out)
        rows = _evaluate_stage(candidate, accepted, scene)
        result = {'ba': ba, 'model_hashes': {p.name: file_digest(p)
                                            for p in sorted(model_out.glob('*.bin'))},
                  'registered': candidate.num_reg_images(),
                  'points': candidate.num_points3D(),
                  'observations': candidate.compute_num_observations(),
                  'change_from_source': _model_change(
                      source, candidate, abs(floor_info['floor_offset_from_mean_camera_center'])),
                  'manual': _stage_summary(rows),
                  'manual_ground_proxy': _manual_groups(rows, classes),
                  'manual_paired_vs_source': _manual_comparison(base_rows, rows),
                  'training': _training_groups(candidate, near, far)}
        write_json(stage / 'manual_rows.json', rows)
        write_json(stage / 'result.json', result)
        comparison['variants'][name] = result
        write_json(output / 'comparison.json', comparison)
        print(name, 'manual', result['manual']['all']['median'],
              result['manual']['all']['p90'], 'BA', ba['termination_type'], flush=True)
    control_rows = json.loads((output / 'equal_control/manual_rows.json').read_text())
    weighted_rows = json.loads((output / 'near_floor_2p0/manual_rows.json').read_text())
    comparison['weighted_paired_vs_equal_control'] = _manual_comparison(control_rows, weighted_rows)
    write_json(output / 'comparison.json', comparison)
    return comparison


def analyze_ground_joint_ba(work):
    """Audit saved full BA candidates without changing any reconstruction."""
    root = Path(work).resolve() / 'global_sfm'
    output = root / OUTPUT
    protocol = json.loads((output / 'protocol.json').read_text())
    comparison = json.loads((output / 'comparison.json').read_text())
    if protocol['fingerprint'] != comparison['protocol_fingerprint']:
        raise ValueError('Joint BA protocol does not match results')
    model_path = root / SOURCE
    if protocol['source_model_hashes'] != {p.name: file_digest(p)
                                           for p in sorted(model_path.glob('*.bin'))}:
        raise ValueError('Joint BA source model changed')
    source = pycolmap.Reconstruction(model_path)
    floor_ids, _, plane = fit_floor_from_training(source)
    classes = {row['id']: row['class'] for row in json.loads(
        (root / 'ground_pose_probe/target_analysis.json').read_text())['manual_proxy_details']}
    control_rows = json.loads((output / 'equal_control/manual_rows.json').read_text())
    weighted_rows = json.loads((output / 'near_floor_2p0/manual_rows.json').read_text())
    analysis = {'protocol_fingerprint': protocol['fingerprint'],
                'manual_paired_weighted_vs_control': {
                    group: _manual_comparison(
                        [r for r in control_rows if classes[r['id']] == group],
                        [r for r in weighted_rows if classes[r['id']] == group])
                    for group in GROUPS},
                'point_movement': {}, 'camera_intrinsics_delta': {},
                'recommendation': 'Do not promote weighted candidate as 3DGS input: near-floor manual median and <=2px rate did not improve, while non-floor training and single-camera/edge manual groups worsened. Preserve the candidate for further geometry and photometric diagnosis.'}
    center = np.asarray(plane['camera_center'])
    normal = np.asarray(plane['floor_normal'])
    offset = plane['floor_offset_from_mean_camera_center']
    for name in WEIGHTS:
        stage = output / name
        saved_hashes = comparison['variants'][name]['model_hashes']
        actual_hashes = {p.name: file_digest(p)
                         for p in sorted((stage / 'sparse/0').glob('*.bin'))}
        if saved_hashes != actual_hashes:
            raise ValueError(f'Saved joint BA model changed: {name}')
        model = pycolmap.Reconstruction(stage / 'sparse/0')
        movements = [(float(np.linalg.norm(model.points3D[pid].xyz - source.points3D[pid].xyz)),
                      pid) for pid in source.points3D]
        movements.sort(reverse=True)
        all_values = np.asarray([value for value, _ in movements])
        floor_values = np.asarray([value for value, pid in movements if pid in floor_ids])
        other_values = np.asarray([value for value, pid in movements if pid not in floor_ids])

        def quantiles(values):
            return {str(q): float(np.quantile(values, q)) for q in (.5, .9, .99, .999, 1.)}

        old_xyz = np.asarray([source.points3D[pid].xyz for pid in floor_ids])
        new_xyz = np.asarray([model.points3D[pid].xyz for pid in floor_ids])
        old_distance = np.abs((old_xyz - center) @ normal - offset)
        new_distance = np.abs((new_xyz - center) @ normal - offset)
        analysis['point_movement'][name] = {
            'all_world_quantiles': quantiles(all_values),
            'reliable_floor_world_quantiles': quantiles(floor_values),
            'other_world_quantiles': quantiles(other_values),
            'counts_over_world_distance': {str(t): int(np.sum(all_values > t))
                                           for t in (.1, 1., 10., 100.)},
            'reliable_floor_plane_distance_source': quantiles(old_distance),
            'reliable_floor_plane_distance_candidate': quantiles(new_distance),
            'reliable_floor_points_outside_source_0p03_plane': int(np.sum(new_distance > .03)),
            'largest_movements': [{
                'point3D_id': int(pid), 'movement_world': value,
                'source_xyz': source.points3D[pid].xyz.tolist(),
                'candidate_xyz': model.points3D[pid].xyz.tolist(),
                'source_distance_from_mean_camera_world': float(
                    np.linalg.norm(source.points3D[pid].xyz - center)),
                'track_length': source.points3D[pid].track.length(),
                'source_error_px': float(source.points3D[pid].error),
                'candidate_error_px': float(model.points3D[pid].error),
                'is_reliable_floor': pid in floor_ids}
                for value, pid in movements[:12]]}
        analysis['camera_intrinsics_delta'][name] = {
            str(cid): (model.cameras[cid].params - source.cameras[cid].params).tolist()
            for cid in source.cameras}
    write_json(output / 'analysis.json', analysis)
    write_json(output / 'NOT_PROMOTED.json', {
        'source_protocol_fingerprint': protocol['fingerprint'],
        'reason': analysis['recommendation'],
        'comparison': 'comparison.json', 'analysis': 'analysis.json',
        'does_not_change_existing_global_model': True})
    print('Joint ground BA analysis:', analysis['recommendation'], flush=True)
    return analysis
