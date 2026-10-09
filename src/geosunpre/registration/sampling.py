"""Derive solve resolution and frozen regions from observed scene geometry."""

import numpy as np
from scipy.spatial import cKDTree


def spatial_regions(points):
    """Partition all eligible points, without withholding fit observations."""
    points = np.asarray(points, dtype=np.float64)
    origin = points.min(0)
    span = float(np.max(np.ptp(np.quantile(points, [.02, .98], axis=0), axis=0)))
    if not np.isfinite(span) or span <= 0:
        raise ValueError('Registration points have no spatial extent')
    requested = int(np.clip(len(points) // 100, 8, 256))
    low, high = span / 128, span * 2
    best = None
    for _ in range(12):
        size = (low + high) / 2
        tiles = np.floor((points - origin) / size).astype(np.int64)
        count = len(np.unique(tiles, axis=0))
        proposal = (abs(count - requested), size, tiles, count)
        if best is None or proposal[:2] < best[:2]:
            best = proposal
        if count > requested:
            low = size
        else:
            high = size
    _, size, tiles, count = best
    return tiles, {'method': 'occupied regions from robust extent and point count',
        'tile_size': size, 'origin': origin.tolist(), 'regions': count,
        'requested_regions': requested, 'points': len(points), 'frozen_in_initial_frame': True}


def sampling_scales(target, header, settings, initialization, *, workers=4):
    """Use sampled LAS spacing and initial pose residuals to set solve scales.

    Delivery limits retain their physical units; estimated density does not
    relax those limits. Neighbour spacing is a sampling proxy, not LAS noise.
    """
    target = np.asarray(target)
    tree = cKDTree(target)
    indices = np.linspace(0, len(target) - 1, min(len(target), 4096), dtype=int)
    distances, _ = tree.query(target[indices], k=min(8, len(target)), workers=workers)
    positive = np.where(distances > 0, distances, np.inf)
    nearest = positive.min(axis=1)
    nearest = nearest[np.isfinite(nearest)]
    if not len(nearest):
        raise ValueError('LAS samples have no distinct spatial support')
    spacing = float(np.median(nearest))
    quantization = float(max(header['scales']))
    fine = max(2 * quantization, min(2 * spacing, settings['registration_max_median_m']))
    pose_p90 = initialization['camera_position_fit']['position_error']['p90']
    coarse_distance = max(8 * fine, 2 * (pose_p90 or 0.))
    return {'method': 'LAS spacing, coordinate resolution and initial position residuals',
        'nearest_spacing_median_m': spacing, 'coordinate_resolution_m': quantization,
        'voxel_sizes': [4 * fine, 2 * fine, fine],
        'correspondence_distances': [coarse_distance, max(4 * fine, coarse_distance / 2), 2 * fine],
        'support_distance': settings['registration_max_p90_m'],
        'scope': 'Data-dependent solve parameters; not measured registration accuracy.'}
