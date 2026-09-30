"""Conservative similarity refinement from coherent surfaces and visual geometry.

Point precision is conditional on fixed visual cameras/intrinsics and a pixel
noise proxy. It supplies relative weights, not a calibrated accuracy estimate.
"""

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from geosunpre.alignment.geometry import (_huber, quantiles, surfaces, transform_points,
                                         validate_similarity)


RECIPE = {
    'method': 'structural', 'evaluation_voxel_size': .1, 'plane_scales': [.1, .2],
    'maximum_surface_noise': .025, 'maximum_scale_normal_angle_deg': 15.,
    'maximum_scale_plane_separation': .025, 'visual_neighbours': [16, 32],
    'maximum_visual_neighbourhood_radius': 1.5, 'maximum_visual_surface_noise': .05,
    'maximum_visual_plane_ratio': .025, 'maximum_visual_normal_angle_deg': 25.,
    'maximum_point_precision_proxy_std': .3, 'pixel_noise_proxy_floor': 1.,
    'maximum_point_only_precision_proxy_std': .1,
    'candidate_neighbours': 8, 'correspondence_distances': [.4, .25],
    'parallel_face_ambiguity_distance_margin': .1, 'parallel_face_separation': .05,
    'normal_precision_weight_floor': .02, 'surface_noise_weight_floor': .01,
    'orientation_total_weights': {'horizontal': .45, 'vertical': .45, 'inclined': .1},
    'minimum_vertical_matches_for_sim3': 100, 'minimum_vertical_tiles_for_sim3': 5,
    'minimum_vertical_normal_spread_ratio': .2, 'minimum_horizontal_matches': 100,
    'minimum_horizontal_tiles': 5,
    'huber_cutoff': .025, 'iterations_per_distance': 15,
    'maximum_total_rotation_deg': 1., 'maximum_total_scale_log_change': .005,
    'maximum_total_pivot_displacement': .15,
    'semantic_classes_used': False, 'pose_residuals_used': False,
}


def point_precision(reconstruction, point_ids):
    """Inverse sum(J.T J / sigma_px^2), using native camera projections.

    Cameras and intrinsics stay fixed. No bundle adjustment or camera covariance
    calculation is performed. Singular viewing geometry is flagged explicitly.
    """
    point_ids = list(map(int, point_ids))
    positions = np.array([reconstruction.points3D[pid].xyz for pid in point_ids])
    index = {pid: i for i, pid in enumerate(point_ids)}
    noise = np.array([max(RECIPE['pixel_noise_proxy_floor'], reconstruction.points3D[pid].error)
                      for pid in point_ids])
    information = np.zeros((len(point_ids), 3, 3))
    observations = np.zeros(len(point_ids), dtype=np.int32)
    for image in reconstruction.images.values():
        selected = [index[p.point3D_id] for p in image.points2D
                    if p.has_point3D() and p.point3D_id in index]
        if not selected:
            continue
        selected = np.array(selected)
        pose = image.cam_from_world().matrix()
        local = positions[selected] @ pose[:, :3].T + pose[:, 3]
        camera = reconstruction.cameras[image.camera_id]
        step = np.maximum(np.linalg.norm(local, axis=1), .1) * 1e-6
        jacobian = np.empty((len(local), 2, 3))
        for axis in range(3):
            offset = np.zeros_like(local)
            offset[:, axis] = step
            jacobian[:, :, axis] = (camera.img_from_cam(local + offset)
                                    - camera.img_from_cam(local - offset)) / (2 * step[:, None])
        jacobian = (jacobian @ pose[:, :3]) / noise[selected, None, None]
        valid = np.isfinite(jacobian).all((1, 2)) & (local[:, 2] > 0)
        np.add.at(information, selected[valid],
                  np.einsum('nki,nkj->nij', jacobian[valid], jacobian[valid]))
        np.add.at(observations, selected[valid], 1)
    values, vectors = np.linalg.eigh(information)
    valid = ((observations >= 2) & (values[:, 0] > np.maximum(values[:, 2] * 1e-8, 1e-12)))
    covariance = np.zeros_like(information)
    covariance[valid] = np.einsum('nik,nk,njk->nij', vectors[valid], 1 / values[valid], vectors[valid])
    return covariance, valid, observations


def visual_planes(points, *, workers=4):
    tree = cKDTree(points)
    masks, normals = [], []
    for k in RECIPE['visual_neighbours']:
        if len(points) < k:
            return np.zeros_like(points), np.zeros(len(points), dtype=bool)
        normal = np.empty_like(points)
        valid = np.zeros(len(points), dtype=bool)
        for start in range(0, len(points), 8192):
            stop = min(start + 8192, len(points))
            distance, neighbours = tree.query(points[start:stop], k=k, workers=workers)
            patch = points[neighbours]
            centred = patch - patch.mean(1)[:, None]
            values, vectors = np.linalg.eigh(np.einsum('nki,nkj->nij', centred, centred) / k)
            values = np.maximum(values, 0)
            normal[start:stop] = vectors[:, :, 0]
            valid[start:stop] = (
                (values[:, 0] / np.maximum(values.sum(1), 1e-12) < RECIPE['maximum_visual_plane_ratio'])
                & (values[:, 1] > .04 * values[:, 2])
                & (np.sqrt(values[:, 0]) <= RECIPE['maximum_visual_surface_noise'])
                & (distance[:, -1] <= RECIPE['maximum_visual_neighbourhood_radius']))
        masks.append(valid)
        normals.append(normal)
    valid = (masks[0] & masks[1]
             & (np.abs(np.einsum('ni,ni->n', normals[0], normals[1])) >= np.cos(np.deg2rad(15))))
    return normals[1], valid


def coherent_las(target, *, workers=4):
    fine = surfaces(target, voxel_size=.1, workers=workers)
    coarse = surfaces(target, voxel_size=.2, workers=workers)
    _, nearest = coarse['tree'].query(fine['points'], workers=workers)
    normal_dot = np.abs(np.einsum('ni,ni->n', fine['normals'], coarse['normals'][nearest]))
    separation = np.abs(np.einsum('ni,ni->n', fine['centres'] - coarse['centres'][nearest], fine['normals']))
    fine['coherent'] = (
        fine['valid'] & coarse['valid'][nearest]
        & (fine['noise'] <= RECIPE['maximum_surface_noise'])
        & (coarse['noise'][nearest] <= RECIPE['maximum_surface_noise'])
        & (normal_dot >= np.cos(np.deg2rad(RECIPE['maximum_scale_normal_angle_deg'])))
        & (separation <= RECIPE['maximum_scale_plane_separation']))
    return fine


def structural_matches(points, normals, visual_valid, surface, maximum, *, workers=4, normal_available=None):
    distances, neighbours = surface['tree'].query(points, k=RECIPE['candidate_neighbours'], workers=workers)
    target_normals = surface['normals'][neighbours]
    cosine = np.abs(np.einsum('nki,ni->nk', target_normals, normals))
    plane = np.einsum('nki,nki->nk', points[:, None] - surface['centres'][neighbours], target_normals)
    if normal_available is None:
        normal_available = np.ones(len(points), dtype=bool)
    normal_compatible = ((cosine >= np.cos(np.deg2rad(RECIPE['maximum_visual_normal_angle_deg'])))
                         | ~normal_available[:, None])
    admissible = (surface['coherent'][neighbours] & visual_valid[:, None]
                  & (distances <= maximum) & (np.abs(plane) <= maximum)
                  & normal_compatible)
    choice = np.argmin(np.where(admissible, distances, np.inf), axis=1)
    rows = np.arange(len(points))
    nearest = neighbours[rows, choice]
    best_normal = surface['normals'][nearest]
    best_distance = distances[rows, choice]
    parallel = np.abs(np.einsum('nki,ni->nk', target_normals, best_normal)) >= np.cos(np.deg2rad(20))
    separation = np.abs(np.einsum('nki,ni->nk',
                                  surface['centres'][neighbours] - surface['centres'][nearest, None], best_normal))
    ambiguous = np.any(admissible & parallel
                       & (distances <= best_distance[:, None] + RECIPE['parallel_face_ambiguity_distance_margin'])
                       & (separation > RECIPE['parallel_face_separation']), axis=1)
    keep = admissible.any(1) & ~ambiguous
    return keep, nearest, plane[rows, choice], ambiguous


def balanced_weights(tiles, normals, confidence):
    """Equal density weights per tile within each orientation, scaled by confidence."""
    z = np.abs(normals[:, 2])
    orientation = np.where(z >= np.cos(np.deg2rad(25)), 0,
                           np.where(z <= np.sin(np.deg2rad(25)), 1, 2))
    weights = np.zeros(len(tiles))
    for group, mass in enumerate(RECIPE['orientation_total_weights'].values()):
        mask = orientation == group
        if not mask.any():
            continue
        unique, inverse, counts = np.unique(tiles[mask], axis=0, return_inverse=True, return_counts=True)
        weights[mask] = mass * confidence[mask] / (len(unique) * counts[inverse])
    return weights, orientation


def _coverage(keep, nearest, tiles, surface):
    normals = surface['normals'][nearest]
    vertical = keep & (np.abs(normals[:, 2]) <= np.sin(np.deg2rad(25)))
    horizontal = keep & (np.abs(normals[:, 2]) >= np.cos(np.deg2rad(25)))
    spread = np.linalg.svd(normals[vertical], compute_uv=False) if vertical.any() else np.zeros(3)
    value = {'vertical_points': int(vertical.sum()),
             'vertical_tiles': len(np.unique(tiles[vertical], axis=0)),
             'vertical_normal_spread_ratio': float(spread[1] / max(spread[0], 1e-12)) if len(spread) > 1 else 0.,
             'horizontal_points': int(horizontal.sum()),
             'horizontal_tiles': len(np.unique(tiles[horizontal], axis=0))}
    sim3_supported = (value['vertical_points'] >= RECIPE['minimum_vertical_matches_for_sim3']
                      and value['vertical_tiles'] >= RECIPE['minimum_vertical_tiles_for_sim3']
                      and value['vertical_normal_spread_ratio'] >= RECIPE['minimum_vertical_normal_spread_ratio'])
    return value, sim3_supported


def refine_structural(source_fit, target, initial, covariance_fit, precision_valid_fit, *,
                      tiles_fit, workers=4):
    """Fit-only refinement; all camera/point geometry remains a single Sim3."""
    initial = validate_similarity(initial)
    source_fit = np.asarray(source_fit, dtype=np.float64)
    if len(source_fit) < 100 or len(target) < 30:
        raise ValueError('Too few points for structural refinement')
    if (np.asarray(covariance_fit).shape != (len(source_fit), 3, 3)
            or np.asarray(precision_valid_fit).shape != (len(source_fit),)
            or np.asarray(tiles_fit).shape != source_fit.shape
            or not np.isfinite(covariance_fit).all()):
        raise ValueError('Structural fit attributes must correspond to fit points only')
    initial_points = transform_points(initial, source_fit)
    initial_normals, normal_available = visual_planes(initial_points, workers=workers)
    covariance = initial[:3, :3] @ covariance_fit @ initial[:3, :3].T
    maximum_std = np.sqrt(np.maximum(np.linalg.eigvalsh(covariance)[:, -1], 0))
    reliable = precision_valid_fit & (maximum_std <= RECIPE['maximum_point_precision_proxy_std'])
    point_only = reliable & ~normal_available & (maximum_std <= RECIPE['maximum_point_only_precision_proxy_std'])
    visual_valid = reliable & (normal_available | point_only)
    surface = coherent_las(target, workers=workers)
    matrix, log, ranks = initial.copy(), [], []
    initial_keep, initial_nearest, _, _ = structural_matches(initial_points, initial_normals, visual_valid,
        surface, RECIPE['correspondence_distances'][0], workers=workers, normal_available=normal_available)
    coverage, seven_dofs = _coverage(initial_keep, initial_nearest, tiles_fit, surface)
    ground_supported = (coverage['horizontal_points'] >= RECIPE['minimum_horizontal_matches']
                        and coverage['horizontal_tiles'] >= RECIPE['minimum_horizontal_tiles'])
    mode = 'sim3' if seven_dofs else 'ground_tilt_z_only' if ground_supported else 'retain_reference'
    columns = np.arange(7) if mode == 'sim3' else np.array([0, 1, 5])
    pivot_initial = initial_points.mean(0)
    for maximum in RECIPE['correspondence_distances']:
        if mode == 'retain_reference':
            break
        for iteration in range(RECIPE['iterations_per_distance']):
            delta_frame = matrix[:3, :3] @ np.linalg.inv(initial[:3, :3])
            scale = np.cbrt(np.linalg.det(delta_frame))
            points = transform_points(matrix, source_fit)
            normals = initial_normals @ (delta_frame / scale).T
            keep, nearest, plane, ambiguous = structural_matches(points, normals, visual_valid,
                surface, maximum, workers=workers, normal_available=normal_available)
            if mode == 'ground_tilt_z_only':
                keep &= np.abs(surface['normals'][nearest, 2]) >= np.cos(np.deg2rad(25))
            elif not _coverage(keep, nearest, tiles_fit, surface)[1]:
                log.append({'maximum_distance': maximum, 'iteration': iteration,
                            'status': 'lost_vertical_structure_coverage', 'matches': int(keep.sum())})
                break
            if keep.sum() < 100:
                log.append({'maximum_distance': maximum, 'iteration': iteration,
                            'status': 'too_few_structural_matches', 'matches': int(keep.sum())})
                break
            p, n, residual = points[keep], surface['normals'][nearest[keep]], plane[keep]
            current_covariance = delta_frame @ covariance[keep] @ delta_frame.T
            normal_std = np.sqrt(np.maximum(np.einsum('ni,nij,nj->n', n, current_covariance, n), 0))
            confidence = (np.minimum(1., (RECIPE['normal_precision_weight_floor'] / np.maximum(normal_std, 1e-12))**2)
                          * np.minimum(1., (RECIPE['surface_noise_weight_floor']
                                            / np.maximum(surface['noise'][nearest[keep]], 1e-12))**2))
            base_weight, orientations = balanced_weights(tiles_fit[keep], n, confidence)
            pivot = p.mean(0)
            centred = p - pivot
            jacobian = np.column_stack((np.cross(centred, n), n, np.einsum('ni,ni->n', centred, n)))
            jacobian = jacobian[:, columns]
            cutoff = RECIPE['huber_cutoff']
            weight = base_weight * np.minimum(1., cutoff / np.maximum(np.abs(residual), 1e-12))
            a = jacobian * np.sqrt(weight[:, None])
            column_norm = np.linalg.norm(a, axis=0)
            singular = np.linalg.svd(a / np.maximum(column_norm, 1e-12), compute_uv=False)
            rank = int(np.count_nonzero(singular > singular[0] * 1e-3))
            ranks.append(rank)
            groups = {name: {'points': int((orientations == group).sum()),
                             'weight': float(base_weight[orientations == group].sum())}
                      for group, name in enumerate(RECIPE['orientation_total_weights'])}
            if rank < len(columns):
                log.append({'maximum_distance': maximum, 'iteration': iteration,
                            'status': 'unobservable_similarity', 'matches': int(keep.sum()),
                            'rank': rank, 'singular_values': singular.tolist(), 'orientation_groups': groups})
                break
            step = np.zeros(7)
            step[columns] = np.linalg.lstsq(a, -residual * np.sqrt(weight), rcond=1e-8)[0]
            limit = max(1., np.linalg.norm(step[:3]) / np.deg2rad(.25),
                        np.linalg.norm(step[3:6]) / .1, abs(step[6]) / .002)
            step /= limit
            old_cost = float(np.sum(base_weight * _huber(residual, cutoff)))
            accepted = False
            for backtrack in range(10):
                delta = step * .5**backtrack
                linear = np.exp(delta[6]) * Rotation.from_rotvec(delta[:3]).as_matrix()
                increment = np.eye(4)
                increment[:3, :3], increment[:3, 3] = linear, pivot + delta[3:6] - linear @ pivot
                candidate = increment @ matrix
                total = candidate @ np.linalg.inv(initial)
                total_scale = np.cbrt(np.linalg.det(total[:3, :3]))
                rotation_change = np.rad2deg(Rotation.from_matrix(total[:3, :3] / total_scale).magnitude())
                pivot_displacement = np.linalg.norm(transform_points(total, pivot_initial[None])[0] - pivot_initial)
                if (abs(np.log(total_scale)) > RECIPE['maximum_total_scale_log_change']
                        or rotation_change > RECIPE['maximum_total_rotation_deg']
                        or pivot_displacement > RECIPE['maximum_total_pivot_displacement']):
                    continue
                changed = transform_points(candidate, source_fit[keep])
                new_residual = np.einsum('ni,ni->n', changed - surface['centres'][nearest[keep]], n)
                new_cost = float(np.sum(base_weight * _huber(new_residual, cutoff)))
                if new_cost <= old_cost:
                    if mode == 'sim3':
                        candidate_frame = candidate[:3, :3] @ np.linalg.inv(initial[:3, :3])
                        candidate_normals = initial_normals @ (candidate_frame / np.cbrt(np.linalg.det(candidate_frame))).T
                        candidate_keep, candidate_nearest, *_ = structural_matches(
                            transform_points(candidate, source_fit), candidate_normals, visual_valid,
                            surface, maximum, workers=workers, normal_available=normal_available)
                        if not _coverage(candidate_keep, candidate_nearest, tiles_fit, surface)[1]:
                            continue
                    matrix, accepted = validate_similarity(candidate), True
                    break
            log.append({'maximum_distance': maximum, 'iteration': iteration,
                        'status': 'updated' if accepted else 'no_descent_within_trust_bounds',
                        'matches': int(keep.sum()), 'ambiguous_parallel_faces': int(ambiguous.sum()),
                        'rank': rank, 'singular_values': singular.tolist(), 'orientation_groups': groups,
                        'weighted_cost_before': old_cost,
                        'weighted_cost_after': new_cost if accepted else old_cost})
            if not accepted or (np.linalg.norm(delta[:3]) < 1e-5
                                and np.linalg.norm(delta[3:6]) < 1e-4 and abs(delta[6]) < 1e-5):
                break
    delta_frame = matrix[:3, :3] @ np.linalg.inv(initial[:3, :3])
    keep, nearest, _, ambiguous = structural_matches(transform_points(matrix, source_fit),
        initial_normals @ (delta_frame / np.cbrt(np.linalg.det(delta_frame))).T,
        visual_valid, surface, RECIPE['correspondence_distances'][-1], workers=workers, normal_available=normal_available)
    if mode == 'ground_tilt_z_only':
        keep &= np.abs(surface['normals'][nearest, 2]) >= np.cos(np.deg2rad(25))
    elif mode == 'retain_reference':
        keep[:] = False
    report = {'recipe': RECIPE, 'iterations': log,
              'mode': mode, 'initial_coverage': coverage,
              'estimated_dofs': len(columns) if mode != 'retain_reference' else 0,
              'all_seven_similarity_dofs_reestimated': mode == 'sim3',
              'minimum_rank': min(ranks) if ranks else None,
              'full_rank_at_all_solved_iterations': bool(ranks) and min(ranks) == len(columns),
              'pose_residuals_used_for_refinement': False, 'holdout_used_for_optimization': False,
              'fit_points': len(source_fit), 'precision_reliable_fit_points': int(reliable.sum()),
              'visual_planar_fit_points': int((normal_available & reliable).sum()),
              'precise_point_only_fit_points': int(point_only.sum()),
              'coherent_las_patches': int(surface['coherent'].sum()), 'final_fit_matches': int(keep.sum()),
              'fit_point_precision_proxy_std': quantiles(maximum_std[precision_valid_fit]),
              'precision_scope': 'fixed visual cameras and intrinsics; relative weighting only; not full BA covariance'}
    diagnostic = {'maximum_precision_proxy_std': maximum_std, 'precision_reliable': reliable,
                  'visual_planar': normal_available & reliable, 'precise_point_only': point_only,
                  'final_structural_matches': keep,
                  'final_las_patch': nearest, 'ambiguous_parallel_faces': ambiguous}
    return matrix, report, surface, diagnostic
