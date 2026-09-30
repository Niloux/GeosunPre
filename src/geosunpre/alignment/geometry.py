"""Similarity registration against LAS surface patches, with spatial holdout."""

import numpy as np
import pycolmap
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def transform_points(matrix, points):
    matrix = np.asarray(matrix, dtype=np.float64)
    return np.asarray(points) @ matrix[:3, :3].T + matrix[:3, 3]


def validate_similarity(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all() or not np.allclose(matrix[3], [0, 0, 0, 1]):
        raise ValueError('Expected a finite 4x4 similarity matrix')
    scale = np.cbrt(np.linalg.det(matrix[:3, :3]))
    if scale <= 0 or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3],
                                     np.eye(3) * scale**2, rtol=1e-7, atol=1e-10):
        raise ValueError('Expected a proper rotation and positive uniform scale')
    return matrix


def matrix4(similarity):
    matrix = np.eye(4)
    matrix[:3] = similarity.matrix()
    return validate_similarity(matrix)


def robust_similarity(source, target, *, threshold=.5, seed=0):
    source, target = np.asarray(source), np.asarray(target)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3 or len(source) < 3:
        raise ValueError('Expected at least three corresponding 3D positions')
    if not np.isfinite(source).all() or not np.isfinite(target).all() or threshold <= 0:
        raise ValueError('Expected finite correspondences and a positive threshold')
    spread = np.linalg.svd(source - source.mean(0), compute_uv=False)
    if spread[1] < max(spread[0] * 1e-4, 1e-8):
        raise ValueError('Camera trajectory is collinear; similarity orientation is unobservable')
    result = pycolmap.estimate_sim3d_robust(
        source, target, pycolmap.RANSACOptions(max_error=threshold, random_seed=seed,
                                               min_inlier_ratio=.25))
    if result is None or result['num_inliers'] < 3:
        raise ValueError('Pose correspondences did not support a similarity initialization')
    matrix = matrix4(result['tgt_from_src'])
    errors = np.linalg.norm(transform_points(matrix, source) - target, axis=1)
    return matrix, {'threshold': threshold, 'inliers': int(result['num_inliers']),
                    'positions': len(source), 'position_error': quantiles(errors),
                    'trajectory_singular_values': spread.tolist()}


def voxel_downsample(points, size):
    if size <= 0:
        raise ValueError('Voxel size must be positive')
    points = np.asarray(points, dtype=np.float64)
    _, inverse, counts = np.unique(np.floor(points / size).astype(np.int64), axis=0,
                                   return_inverse=True, return_counts=True)
    return np.column_stack([np.bincount(inverse, weights=points[:, i]) / counts for i in range(3)])


def spatial_split(points, *, tile_size=4., fraction=.2, seed=0):
    if tile_size <= 0 or not 0 < fraction < 1:
        raise ValueError('Expected positive tiles and a holdout fraction between zero and one')
    tiles, inverse = np.unique(np.floor(np.asarray(points) / tile_size).astype(np.int64),
                               axis=0, return_inverse=True)
    if len(tiles) < 5:
        raise ValueError('Too few spatial tiles for registration holdout')
    chosen = np.random.default_rng(seed).permutation(len(tiles))[:max(1, int(round(len(tiles) * fraction)))]
    holdout = np.isin(inverse, chosen)
    return holdout, tiles[inverse], {'tile_size': tile_size, 'fraction_requested': fraction,
                                     'seed': seed, 'tiles': len(tiles), 'heldout_tiles': len(chosen),
                                     'fit_points': int((~holdout).sum()), 'holdout_points': int(holdout.sum()),
                                     'frozen_in_initial_frame': True}


def surfaces(points, *, voxel_size, workers=4, diagnostics=False):
    """Estimate plane centres/normals in bounded batches; reject line-like patches."""
    points = voxel_downsample(points, voxel_size)
    if len(points) < 20:
        raise ValueError('Too few spatially distinct LAS points for plane estimation')
    tree = cKDTree(points)
    centres, normals = np.zeros_like(points), np.zeros_like(points)
    valid = np.zeros(len(points), dtype=bool)
    noise = np.full(len(points), np.inf)
    if diagnostics:
        eigenvalues_all = np.empty_like(points)
        radii = np.empty(len(points))
    k = min(20, len(points))
    for start in range(0, len(points), 8192):
        stop = min(start + 8192, len(points))
        distances, neighbours = tree.query(points[start:stop], k=k, workers=workers)
        patch = points[neighbours]
        centre = patch.mean(axis=1)
        centred = patch - centre[:, None]
        covariance = np.einsum('nki,nkj->nij', centred, centred) / k
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        eigenvalues = np.maximum(eigenvalues, 0)
        if diagnostics:
            eigenvalues_all[start:stop] = eigenvalues
            radii[start:stop] = distances[:, -1]
        centres[start:stop] = centre
        normals[start:stop] = eigenvectors[:, :, 0]
        noise[start:stop] = np.sqrt(eigenvalues[:, 0])
        valid[start:stop] = ((eigenvalues[:, 0] / np.maximum(eigenvalues.sum(1), 1e-12) < .04)
                            & (eigenvalues[:, 1] > .04 * eigenvalues[:, 2])
                            & (distances[:, -1] <= max(.35, voxel_size * 5))
                            & (noise[start:stop] <= .08))
    result = {'points': points, 'tree': tree, 'centres': centres, 'normals': normals,
              'valid': valid, 'noise': noise, 'voxel_size': voxel_size}
    if diagnostics:
        result.update(eigenvalues=eigenvalues_all, radius=radii)
    return result


def quantiles(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if not len(values):
        return {'count': 0, 'median': None, 'p90': None, 'max': None}
    return {'count': len(values), 'median': float(np.median(values)),
            'p90': float(np.quantile(values, .9)), 'max': float(values.max())}


def residuals(points, surface, *, workers=4):
    distances, nearest = surface['tree'].query(points, workers=workers)
    plane = np.einsum('ni,ni->n', points - surface['centres'][nearest], surface['normals'][nearest])
    return distances, plane, nearest


def error_summary(points, surface, *, support_distance=.3, workers=4):
    distances, plane, nearest = residuals(points, surface, workers=workers)
    supported = (distances <= support_distance) & surface['valid'][nearest]
    return {'points': len(points), 'nearest_distance': quantiles(distances),
            'surface_support_count': int(supported.sum()),
            'surface_support_fraction': float(supported.mean()) if len(points) else None,
            'support_distance': support_distance,
            'point_to_plane_on_supported': quantiles(np.abs(plane[supported])),
            'fraction_all_points_supported_within_5cm': float(np.mean(supported & (np.abs(plane) <= .05)))
                if len(points) else None,
            'fraction_all_points_supported_within_10cm': float(np.mean(supported & (np.abs(plane) <= .1)))
                if len(points) else None}


def _huber(residual, cutoff):
    absolute = np.abs(residual)
    return np.where(absolute <= cutoff, .5 * residual**2, cutoff * (absolute - .5 * cutoff))


def refine_similarity(source_fit, target, initial, *, tiles_fit=None, workers=4,
                      voxel_sizes=(.4, .2, .1), correspondence_distances=(.8, .4, .2), iterations=12):
    """Only fit points enter this optimizer; holdout points cannot be consulted.

    Each accepted update has a proper rotation and positive uniform scale.
    Surface correspondences use robust residuals and equal total tile weights.
    """
    matrix = validate_similarity(initial).copy()
    source_fit = np.asarray(source_fit)
    if len(source_fit) < 100 or len(target) < 30:
        raise ValueError('Too few points for surface refinement')
    if len(voxel_sizes) != len(correspondence_distances) or not voxel_sizes:
        raise ValueError('Expected one correspondence distance per voxel scale')
    if tiles_fit is None:
        tiles_fit = np.floor(transform_points(matrix, source_fit) / 4.).astype(np.int64)
    log, ranks = [], []
    last_surface = None
    for voxel, maximum in zip(voxel_sizes, correspondence_distances):
        surface = surfaces(target, voxel_size=voxel, workers=workers)
        last_surface = surface
        for iteration in range(iterations):
            points = transform_points(matrix, source_fit)
            distances, plane, nearest = residuals(points, surface, workers=workers)
            keep = (distances <= maximum) & surface['valid'][nearest] & (np.abs(plane) <= maximum)
            if keep.sum() < 100:
                log.append({'voxel': voxel, 'iteration': iteration, 'status': 'too_few_surface_matches',
                            'matches': int(keep.sum())})
                break
            p, n, residual = points[keep], surface['normals'][nearest[keep]], plane[keep]
            pivot = p.mean(0)
            centred = p - pivot
            jacobian = np.column_stack((np.cross(centred, n), n, np.einsum('ni,ni->n', centred, n)))
            _, inverse, counts = np.unique(tiles_fit[keep], axis=0, return_inverse=True, return_counts=True)
            base_weight = 1. / counts[inverse]
            cutoff = max(.015, min(.08, maximum / 4))
            weight = base_weight * np.minimum(1., cutoff / np.maximum(np.abs(residual), 1e-12))
            a = jacobian * np.sqrt(weight[:, None])
            column_norm = np.linalg.norm(a, axis=0)
            singular = np.linalg.svd(a / np.maximum(column_norm, 1e-12), compute_uv=False)
            rank = int(np.count_nonzero(singular > singular[0] * 1e-3))
            ranks.append(rank)
            if rank < 7:
                log.append({'voxel': voxel, 'iteration': iteration, 'status': 'unobservable_similarity',
                            'matches': int(keep.sum()), 'rank': rank, 'singular_values': singular.tolist()})
                break
            delta = np.linalg.lstsq(a, -residual * np.sqrt(weight), rcond=1e-8)[0]
            limit = max(1., np.linalg.norm(delta[:3]) / np.deg2rad(5),
                        np.linalg.norm(delta[3:6]) / maximum, abs(delta[6]) / .03)
            delta /= limit
            old_cost = float(np.sum(base_weight * _huber(residual, cutoff)))
            accepted = False
            for backtrack in range(8):
                step = delta * .5**backtrack
                rotation = Rotation.from_rotvec(step[:3]).as_matrix()
                linear = np.exp(step[6]) * rotation
                increment = np.eye(4)
                increment[:3, :3] = linear
                increment[:3, 3] = pivot + step[3:6] - linear @ pivot
                candidate = increment @ matrix
                changed = transform_points(candidate, source_fit[keep])
                new_residual = np.einsum('ni,ni->n', changed - surface['centres'][nearest[keep]], n)
                new_cost = float(np.sum(base_weight * _huber(new_residual, cutoff)))
                if new_cost <= old_cost:
                    matrix = validate_similarity(candidate)
                    accepted = True
                    break
            log.append({'voxel': voxel, 'iteration': iteration, 'status': 'updated' if accepted else 'no_descent',
                        'matches': int(keep.sum()), 'rank': rank, 'singular_values': singular.tolist(),
                        'weighted_cost_before': old_cost, 'weighted_cost_after': new_cost,
                        'step': step.tolist()})
            if not accepted or (np.linalg.norm(step[:3]) < 1e-5
                                and np.linalg.norm(step[3:6]) < 1e-4 and abs(step[6]) < 1e-5):
                break
    return matrix, {'iterations': log, 'full_rank_at_all_solved_iterations': bool(ranks) and min(ranks) == 7,
                    'minimum_rank': min(ranks) if ranks else None,
                    'pose_residuals_used_for_refinement': False}, last_surface
