"""Image-led fixed-intrinsics BA with an independently parameterized global Sim3.

Image gates use the existing fitted observations. They protect this candidate
against regression; they are not independent validation or a 3DGS quality test.
"""

import time

import numpy as np
import pyceres
import pycolmap
from scipy.spatial.transform import Rotation

from geosunpre.registration.geometry import _huber, quantiles, transform_points, validate_similarity
from geosunpre.registration.structural import (RECIPE as STRUCTURAL_RECIPE, balanced_weights,
    point_precision, structural_matches, visual_planes)
from geosunpre.camera import Camera
from geosunpre.sfm.depth_guard import add_positive_depth_guards


RECIPE = {
    'fixed_intrinsics': True, 'image_huber_cutoff_px': 2.,
    'pose_std_m': .5, 'las_std_m': .05, 'prior_huber_cutoff_sigma': 2.,
    'pose_weight': .001, 'las_weight': .005,
    'outer_iterations': 2, 'iterations_per_solve': 20, 'workers': 4,
    'maximum_local_camera_displacement_m': .15,
    'maximum_local_camera_rotation_deg': .5,
    'point_coordinate_bound_m': .2, 'maximum_local_point_displacement_m': .5,
    'maximum_global_rotation_deg': 1., 'maximum_global_log_scale_change': .005,
    'maximum_global_pivot_displacement_m': .15,
    'global_translation_coordinate_bound_m': .3,
    'backtrack_fractions': [1., .5, .25, .125, .0625, .03125],
    'pose_rotations_used': False, 'las_points_optimized': False,
    'gauge': 'TWO_CAMS_FROM_WORLD in the visual frame; global Sim3 stays variable',
    'noise_scope': 'Uncalibrated conservative noise proxies; not measured sensor uncertainty',
}

IMAGE_GATES = {
    'all': {'median': .02, 'p90': .05, 'p99': .1},
    'camera': {'median': .03, 'p90': .08, 'p99': .15},
    'camera_zone': {'median': .03, 'p90': .08, 'p99': .15},
    'image': {'median': .05, 'p90': .1, 'p99': .2},
    'tile': {'median': .1, 'p90': .2, 'p99': .3},
    'minimum_group_observations': 30,
    'scope': 'Absolute pixel increases on frozen fitted observations; no failed observations discarded',
}


class JointConstraintsUnavailable(ValueError):
    """Reliable LAS matches are insufficient; the registered visual model is usable."""


def native_costs():
    import geosunpre_native as native
    if (getattr(native, 'implementation_version', None) != 2
            or not all(hasattr(native, name) for name in ('Sim3PointPlaneCost', 'Sim3PositionPriorCost'))):
        raise RuntimeError('Rebuild geosunpre-native with uv sync --locked --reinstall-package geosunpre-native')
    return native


def similarity_parameters(matrix):
    matrix = validate_similarity(matrix)
    scale = np.cbrt(np.linalg.det(matrix[:3, :3]))
    return (Rotation.from_matrix(matrix[:3, :3] / scale).as_quat().copy(),
            np.r_[matrix[:3, 3], np.log(scale)].astype(np.float64))


def similarity_matrix(quaternion, translation_scale):
    matrix = np.eye(4)
    matrix[:3, :3] = np.exp(translation_scale[3]) * Rotation.from_quat(quaternion).as_matrix()
    matrix[:3, 3] = translation_scale[:3]
    return validate_similarity(matrix)


def assert_identity(before, after):
    """Geometry may change, camera parameters and all observation identities may not."""
    if (not after.is_valid() or set(before.points3D) != set(after.points3D)
            or set(before.images) != set(after.images) or set(before.frames) != set(after.frames)
            or set(before.cameras) != set(after.cameras)
            or before.compute_num_observations() != after.compute_num_observations()):
        raise ValueError('Joint BA changed model membership or topology')
    for cid, camera in before.cameras.items():
        other = after.cameras[cid]
        if ((camera.model, camera.width, camera.height) != (other.model, other.width, other.height)
                or not np.array_equal(camera.params, other.params)):
            raise ValueError('Fixed-intrinsics joint BA changed the camera model')
    for pid, point in before.points3D.items():
        if [(e.image_id, e.point2D_idx) for e in point.track.elements] != [
                (e.image_id, e.point2D_idx) for e in after.points3D[pid].track.elements]:
            raise ValueError('Joint BA changed observation tracks')
    for iid, image in before.images.items():
        other = after.images[iid]
        if ((image.name, image.camera_id, image.frame_id, image.num_points2D()) !=
                (other.name, other.camera_id, other.frame_id, other.num_points2D())
                or any(a.point3D_id != b.point3D_id or not np.array_equal(a.xy, b.xy)
                       for a, b in zip(image.points2D, other.points2D))):
            raise ValueError('Joint BA changed image or feature identity')


def pixel_summary(errors):
    errors = np.asarray(errors)
    finite = np.isfinite(errors)
    if not len(errors):
        return {'count': 0, 'finite_count': 0, 'median': None, 'p90': None, 'p99': None, 'huber_cost': 0.}
    values = np.quantile(errors, [.5, .9, .99]) if finite.all() else [np.inf] * 3
    return {'count': len(errors), 'finite_count': int(finite.sum()),
            **{name: float(value) if np.isfinite(value) else None
               for name, value in zip(('median', 'p90', 'p99'), values)},
            'huber_cost': float(np.sum(_huber(errors, 2.))) if finite.all() else None}


def _groups(values):
    unique, inverse, counts = np.unique(values, axis=0, return_inverse=True, return_counts=True)
    order = np.argsort(inverse, kind='stable')
    indices = np.split(order, np.cumsum(counts)[:-1])
    return [(np.atleast_1d(value).tolist(), index) for value, index in zip(unique, indices)]


class ImageObservations:
    """Freeze every feature identity and grouping before any candidate solve."""
    def __init__(self, rec, initial_matrix):
        self.images = []
        ids, image_ids, camera_ids, feature_ids, pixels, zones = [], [], [], [], [], []
        self.initial_intrinsics = {cid: camera.params.copy() for cid, camera in rec.cameras.items()}
        for image in sorted(rec.images.values(), key=lambda im: im.image_id):
            indices = np.array([i for i, p in enumerate(image.points2D) if p.has_point3D()], dtype=np.int64)
            if not len(indices):
                continue
            point_ids = np.array([image.points2D[int(i)].point3D_id for i in indices], dtype=np.uint64)
            observed = np.array([image.points2D[int(i)].xy for i in indices])
            camera = rec.cameras[image.camera_id]
            if camera.model_name != 'OPENCV_FISHEYE':
                raise ValueError('Joint BA currently expects native OPENCV_FISHEYE cameras')
            frame = rec.frames[image.frame_id]
            if (len(frame.image_ids) != 1 or not np.allclose(image.cam_from_world().matrix(),
                    frame.rig_from_world.matrix(), atol=1e-12, rtol=0)):
                raise ValueError('Joint BA expects one independent reference-camera exposure per frame')
            theta = np.deg2rad(80.)
            radius = theta * (1 + sum(camera.params[4 + k] * theta ** (2 * (k + 1)) for k in range(4)))
            rho = np.linalg.norm((observed - camera.params[2:4]) / camera.params[:2], axis=1) / radius
            zones.append(np.where(rho < .5, 0, np.where(rho < .8, 1, 2)))
            self.images.append((image.image_id, point_ids, indices, observed, sum(map(len, ids))))
            ids.append(point_ids); feature_ids.append(indices); pixels.append(observed)
            image_ids.append(np.full(len(indices), image.image_id)); camera_ids.append(np.full(len(indices), image.camera_id))
        if not ids:
            raise ValueError('No visual observations for joint BA')
        self.point_ids = np.concatenate(ids)
        self.image_ids, self.camera_ids = np.concatenate(image_ids), np.concatenate(camera_ids)
        self.feature_ids, self.pixels, self.zones = np.concatenate(feature_ids), np.concatenate(pixels), np.concatenate(zones)
        xyz = np.array([rec.points3D[int(pid)].xyz for pid in self.point_ids])
        self.tiles = np.floor(transform_points(initial_matrix, xyz) / 4.).astype(np.int64)
        self.groups = [('all', 'all', np.arange(len(self.point_ids)))]
        for label, values in [('camera', self.camera_ids), ('image', self.image_ids),
                              ('camera_zone', np.column_stack((self.camera_ids, self.zones))), ('tile', self.tiles)]:
            self.groups.extend((label, label + '/' + ','.join(map(str, key)), indices)
                               for key, indices in _groups(values))

    def errors(self, rec):
        errors, depths = np.empty(len(self.point_ids)), np.empty(len(self.point_ids))
        for iid, point_ids, indices, observed, offset in self.images:
            image = rec.images[iid]
            pose = image.cam_from_world().matrix()
            xyz = np.array([rec.points3D[int(pid)].xyz for pid in point_ids])
            local = xyz @ pose[:, :3].T + pose[:, 3]
            projected = rec.cameras[image.camera_id].img_from_cam(local)
            values = np.linalg.norm(projected - observed, axis=1)
            values[(local[:, 2] <= 1e-9) | ~np.isfinite(local).all(1) | ~np.isfinite(projected).all(1)] = np.inf
            errors[offset:offset + len(values)], depths[offset:offset + len(values)] = values, local[:, 2]
        return errors, depths

    def summary(self, errors):
        return {name: pixel_summary(errors[indices]) for _, name, indices in self.groups}


def image_gate(observations, baseline, candidate):
    if len(baseline) != len(candidate) or len(candidate) != len(observations.point_ids):
        raise ValueError('Image gate requires identical frozen observations')
    failures, maximum = [], {}
    if not np.isfinite(candidate).all():
        failures.append({'group': 'all', 'reason': 'nonfinite_or_nonpositive_depth',
                         'observations': int((~np.isfinite(candidate)).sum())})
    for kind, name, indices in observations.groups:
        if kind != 'all' and len(indices) < IMAGE_GATES['minimum_group_observations']:
            continue
        old, new = pixel_summary(baseline[indices]), pixel_summary(candidate[indices])
        for metric, tolerance in IMAGE_GATES[kind].items():
            if new[metric] is None or old[metric] is None:
                failures.append({'group': name, 'metric': metric, 'reason': 'nonfinite_quantile'})
                continue
            delta = new[metric] - old[metric]
            key = kind + '/' + metric
            if key not in maximum or delta > maximum[key]['delta_px']:
                maximum[key] = {'group': name, 'delta_px': delta, 'tolerance_px': tolerance}
            if delta > tolerance:
                failures.append({'group': name, 'metric': metric, 'baseline_px': old[metric],
                                 'candidate_px': new[metric], 'delta_px': delta, 'tolerance_px': tolerance})
    return {'passed': not failures, 'scope': IMAGE_GATES['scope'], 'thresholds': IMAGE_GATES,
            'failed_checks': len(failures), 'failures': failures, 'maximum_changes': maximum}


def las_correspondences(rec, matrix, fit_ids, tiles_fit, surface, *, maximum=.25, workers=4):
    """All point precision, visual neighbours and LAS queries use fit IDs only."""
    source = np.array([rec.points3D[int(pid)].xyz for pid in fit_ids])
    points = transform_points(matrix, source)
    normals, available = visual_planes(points, workers=workers)
    covariance, valid, _ = point_precision(rec, fit_ids)
    covariance = matrix[:3, :3] @ covariance @ matrix[:3, :3].T
    std = np.sqrt(np.maximum(np.linalg.eigvalsh(covariance)[:, -1], 0))
    reliable = valid & (std <= STRUCTURAL_RECIPE['maximum_point_precision_proxy_std'])
    point_only = reliable & ~available & (std <= STRUCTURAL_RECIPE['maximum_point_only_precision_proxy_std'])
    keep, nearest, plane, ambiguous = structural_matches(points, normals,
        reliable & (available | point_only), surface, maximum, workers=workers, normal_available=available)
    if keep.sum() < 100:
        raise JointConstraintsUnavailable('Too few reliable LAS correspondences for joint BA')
    n = surface['normals'][nearest[keep]]
    normal_std = np.sqrt(np.maximum(np.einsum('ni,nij,nj->n', n, covariance[keep], n), 0))
    confidence = (np.minimum(1., (.02 / np.maximum(normal_std, 1e-12))**2)
                  * np.minimum(1., (.01 / np.maximum(surface['noise'][nearest[keep]], 1e-12))**2))
    weight, orientation = balanced_weights(tiles_fit[keep], n, confidence)
    return {'point_ids': np.asarray(fit_ids)[keep], 'centres': surface['centres'][nearest[keep]],
            'normals': n, 'weights': weight, 'nearest': nearest[keep],
            'summary': {'fit_points': len(fit_ids), 'matches': int(keep.sum()),
                'precise_point_only_matches': int((keep & point_only).sum()),
                'ambiguous_parallel_faces': int(ambiguous.sum()), 'holdout_used': False,
                'initial_plane_error': quantiles(np.abs(plane[keep])),
                'orientation_groups': {name: {'points': int((orientation == i).sum()),
                    'weight': float(weight[orientation == i].sum())}
                    for i, name in enumerate(('horizontal', 'vertical', 'inclined'))}}}


def mixed_objective(rec, matrix, errors, positions, matches, recipe):
    value = float(np.sum(_huber(errors, recipe['image_huber_cutoff_px'])))
    if positions and recipe['pose_weight']:
        centres = np.array([rec.images[iid].projection_center() for iid in positions])
        delta = (transform_points(matrix, centres) - np.array(list(positions.values()))) / recipe['pose_std_m']
        value += len(errors) * recipe['pose_weight'] / (3 * len(positions)) * float(
            np.sum(_huber(np.linalg.norm(delta, axis=1), recipe['prior_huber_cutoff_sigma'])))
    if matches is not None and recipe['las_weight']:
        xyz = np.array([rec.points3D[int(pid)].xyz for pid in matches['point_ids']])
        delta = np.einsum('ni,ni->n', transform_points(matrix, xyz) - matches['centres'], matches['normals'])
        value += len(errors) * recipe['las_weight'] * float(np.sum(matches['weights'] *
            _huber(delta / recipe['las_std_m'], recipe['prior_huber_cutoff_sigma'])))
    return value


def solve_fixed(rec, matrix, base, initial_matrix, positions, matches, recipe):
    """One fixed-correspondence Ceres solve; acceptance is evaluated separately."""
    native = native_costs()
    options = pycolmap.BundleAdjustmentOptions()
    options.refine_focal_length = options.refine_principal_point = options.refine_extra_params = False
    options.refine_sensor_from_rig = False
    options.ceres.loss_function_type = pycolmap.LossFunctionType.HUBER
    options.ceres.loss_function_scale = recipe['image_huber_cutoff_px']
    options.ceres.solver_options.num_threads = recipe['workers']
    options.ceres.solver_options.max_num_iterations = recipe['iterations_per_solve']
    options.ceres.solver_options.initial_trust_region_radius = 10.
    options.ceres.solver_options.max_trust_region_radius = 10000.
    options.ceres.solver_options.function_tolerance = 1e-7
    options.print_summary = False
    config = pycolmap.BundleAdjustmentConfig()
    for iid in sorted(rec.reg_image_ids()):
        config.add_image(iid)
    config.fix_gauge(pycolmap.BundleAdjustmentGauge.TWO_CAMS_FROM_WORLD)
    adjuster = pycolmap.create_default_ceres_bundle_adjuster(options, config, rec)
    problem = adjuster.problem
    costs = add_positive_depth_guards(problem, rec, cost_class=native.PositiveDepthCost)
    quaternion, translation_scale = similarity_parameters(matrix)
    fixed_frames = [fid for fid, frame in rec.frames.items()
        if problem.has_parameter_block(frame.rig_from_world.params)
        and problem.is_parameter_block_constant(frame.rig_from_world.params)]
    losses, manifolds = [], []
    observations = rec.compute_num_observations()
    if positions and recipe['pose_weight']:
        mass = observations * recipe['pose_weight'] / (3 * len(positions))
        loss = pyceres.HuberLoss(np.sqrt(mass) * recipe['prior_huber_cutoff_sigma'])
        losses.append(loss)
        for iid, position in positions.items():
            pose = rec.frames[rec.images[iid].frame_id].rig_from_world.params
            cost = native.Sim3PositionPriorCost(position.tolist(), np.sqrt(mass) / recipe['pose_std_m'])
            costs.append(cost)
            problem.add_residual_block(cost, loss, [pose, quaternion, translation_scale])
    if matches is not None and recipe['las_weight']:
        for pid, centre, normal, weight in zip(matches['point_ids'], matches['centres'], matches['normals'], matches['weights']):
            mass = observations * recipe['las_weight'] * weight
            if mass <= 0:
                continue
            cost = native.Sim3PointPlaneCost(centre.tolist(), normal.tolist(), np.sqrt(mass) / recipe['las_std_m'])
            loss = pyceres.HuberLoss(np.sqrt(mass) * recipe['prior_huber_cutoff_sigma'])
            costs.append(cost); losses.append(loss)
            problem.add_residual_block(cost, loss, [rec.points3D[int(pid)].xyz, quaternion, translation_scale])
    if problem.has_parameter_block(quaternion):
        manifold = pyceres.EigenQuaternionManifold()
        manifolds.append(manifold); problem.set_manifold(quaternion, manifold)
        _, initial_parameters = similarity_parameters(initial_matrix)
        for k in range(3):
            bound = recipe['global_translation_coordinate_bound_m']
            problem.set_parameter_lower_bound(translation_scale, k, initial_parameters[k] - bound)
            problem.set_parameter_upper_bound(translation_scale, k, initial_parameters[k] + bound)
        bound = recipe['maximum_global_log_scale_change']
        problem.set_parameter_lower_bound(translation_scale, 3, initial_parameters[3] - bound)
        problem.set_parameter_upper_bound(translation_scale, 3, initial_parameters[3] + bound)
    initial_scale = np.cbrt(np.linalg.det(initial_matrix[:3, :3]))
    bound = recipe['point_coordinate_bound_m'] / initial_scale
    for pid, point in rec.points3D.items():
        if not problem.has_parameter_block(point.xyz) or problem.is_parameter_block_constant(point.xyz):
            continue
        original = base.points3D[pid].xyz
        for k in range(3):
            problem.set_parameter_lower_bound(point.xyz, k, float(original[k] - bound))
            problem.set_parameter_upper_bound(point.xyz, k, float(original[k] + bound))
    started = time.monotonic()
    summary = adjuster.solve()
    elapsed = time.monotonic() - started
    rec.update_point_3d_errors()
    output = similarity_matrix(quaternion, translation_scale)
    return output, {'solver': summary.brief_report(), 'solution_usable': summary.is_solution_usable(),
        'termination_type': str(summary.termination_type), 'elapsed_seconds': elapsed,
        'initial_cost': summary.ceres_summary.initial_cost, 'final_cost': summary.ceres_summary.final_cost,
        'successful_steps': summary.ceres_summary.num_successful_steps,
        'unsuccessful_steps': summary.ceres_summary.num_unsuccessful_steps,
        'fixed_visual_frame_ids': fixed_frames, 'depth_barriers': observations,
        'position_priors': len(positions) if recipe['pose_weight'] else 0,
        'las_constraints': len(matches['point_ids']) if matches is not None and recipe['las_weight'] else 0,
        'global_sim3_variable': bool(problem.has_parameter_block(quaternion)),
        'native_implementation_version': native.implementation_version,
        'full_solver_report': summary.ceres_summary.FullReport()}


def geometry_gate(base, rec, initial_matrix, matrix, recipe):
    assert_identity(base, rec)
    scale = np.cbrt(np.linalg.det(initial_matrix[:3, :3]))
    image_ids = sorted(base.images)
    before = np.array([base.images[i].projection_center() for i in image_ids])
    after = np.array([rec.images[i].projection_center() for i in image_ids])
    camera_displacements = scale * np.linalg.norm(after - before, axis=1)
    camera_rotations = np.array([np.rad2deg(Rotation.from_matrix(
        rec.images[i].cam_from_world().matrix()[:, :3] @ base.images[i].cam_from_world().matrix()[:, :3].T).magnitude())
        for i in image_ids])
    point_displacements = scale * np.array([np.linalg.norm(rec.points3D[pid].xyz - point.xyz)
                                           for pid, point in base.points3D.items()])
    total = matrix @ np.linalg.inv(initial_matrix)
    total_scale = np.cbrt(np.linalg.det(total[:3, :3]))
    rotation = float(np.rad2deg(Rotation.from_matrix(total[:3, :3] / total_scale).magnitude()))
    pivot = transform_points(initial_matrix, before).mean(0)
    displacement = float(np.linalg.norm(transform_points(total, pivot[None])[0] - pivot))
    tests = {
        'local_camera_displacement_m': (float(camera_displacements.max()), recipe['maximum_local_camera_displacement_m']),
        'local_camera_rotation_deg': (float(camera_rotations.max()), recipe['maximum_local_camera_rotation_deg']),
        'local_point_displacement_m': (float(point_displacements.max()), recipe['maximum_local_point_displacement_m']),
        'global_rotation_deg': (rotation, recipe['maximum_global_rotation_deg']),
        'global_log_scale_change': (abs(float(np.log(total_scale))), recipe['maximum_global_log_scale_change']),
        'global_pivot_displacement_m': (displacement, recipe['maximum_global_pivot_displacement_m']),
    }
    failures = [name for name, (value, limit) in tests.items() if not np.isfinite(value) or value > limit + 1e-10]
    for cid, cam in rec.cameras.items():
        camera = Camera('L' if cid == 1 else 'R', cam.width, cam.height, tuple(cam.params), (0, 0))
        try:
            camera.distorted_radius(160)
        except ValueError:
            failures.append(f'nonmonotonic_camera_{cid}')
    return {'passed': not failures, 'failures': failures,
            'checks': {name: {'value': value, 'maximum': limit} for name, (value, limit) in tests.items()},
            'local_camera_displacement': quantiles(camera_displacements),
            'local_camera_rotation_deg': quantiles(camera_rotations),
            'local_point_displacement': quantiles(point_displacements),
            'intrinsics_and_all_observation_tracks_unchanged': True}


def interpolate(before, after, first_matrix, second_matrix, fraction):
    """Backtrack rotations on SO(3), optical centres and points linearly."""
    rec = pycolmap.Reconstruction(before)
    for fid, frame in rec.frames.items():
        a, b = before.frames[fid].rig_from_world, after.frames[fid].rig_from_world
        ra, rb = Rotation.from_matrix(a.matrix()[:, :3]), Rotation.from_matrix(b.matrix()[:, :3])
        rotation = (Rotation.from_rotvec(fraction * (rb * ra.inv()).as_rotvec()) * ra).as_matrix()
        ca, cb = -a.matrix()[:, :3].T @ a.translation, -b.matrix()[:, :3].T @ b.translation
        centre = (1 - fraction) * ca + fraction * cb
        frame.rig_from_world = pycolmap.Rigid3d(pycolmap.Rotation3d(rotation), -rotation @ centre)
    for pid, point in rec.points3D.items():
        point.xyz = (1 - fraction) * before.points3D[pid].xyz + fraction * after.points3D[pid].xyz
    qa, pa = similarity_parameters(first_matrix)
    qb, pb = similarity_parameters(second_matrix)
    ra, rb = Rotation.from_quat(qa), Rotation.from_quat(qb)
    quaternion = (Rotation.from_rotvec(fraction * (rb * ra.inv()).as_rotvec()) * ra).as_quat()
    rec.update_point_3d_errors()
    return rec, similarity_matrix(quaternion, (1 - fraction) * pa + fraction * pb)
