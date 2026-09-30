"""Read-only geometric diagnostics for a frozen registration candidate."""

from collections import Counter

import numpy as np
from scipy.spatial import cKDTree

from geosunpre.alignment.geometry import quantiles, residuals


SUPPORT_REASONS = ('supported', 'no_close_las', 'sparse_neighbourhood',
                   'line_like', 'nonplanar', 'thick_patch')


def support_diagnostics(points, surface, *, support_distance=.3, workers=4):
    """Explain the existing support gate, without relaxing it or moving points.

    Primary reasons are exclusive, in SUPPORT_REASONS priority order. Rejection
    flags overlap and preserve every failed criterion (including distant points).
    """
    distance, plane, nearest = residuals(points, surface, workers=workers)
    eigen = surface['eigenvalues'][nearest]
    flags = np.column_stack((
        distance > support_distance,
        surface['radius'][nearest] > max(.35, surface['voxel_size'] * 5),
        eigen[:, 1] <= .04 * eigen[:, 2],
        eigen[:, 0] / np.maximum(eigen.sum(1), 1e-12) >= .04,
        surface['noise'][nearest] > .08))
    reason = np.zeros(len(points), dtype=np.uint8)
    for index in reversed(range(flags.shape[1])):
        reason[flags[:, index]] = index + 1
    if not np.array_equal(reason == 0, (distance <= support_distance) & surface['valid'][nearest]):
        raise ValueError('Surface diagnostics disagree with the registration support gate')
    return {'reason': reason, 'rejection_flags': flags, 'nearest_distance': distance,
            'signed_plane_error': plane, 'nearest': nearest,
            'normals': surface['normals'][nearest]}


def track_geometry(reconstruction, point_ids):
    """Maximum pairwise viewing-ray angle, native error, and track length.

    These are diagnostics of visual geometry, not a calibrated depth variance.
    """
    angles, lengths, errors = [], [], []
    camera_centres = {iid: im.projection_center() for iid, im in reconstruction.images.items()}
    for pid in point_ids:
        point = reconstruction.points3D[int(pid)]
        centres = np.array([camera_centres[e.image_id] for e in point.track.elements])
        rays = centres - point.xyz
        rays /= np.maximum(np.linalg.norm(rays, axis=1)[:, None], 1e-12)
        angle = np.rad2deg(np.arccos(np.clip((rays @ rays.T).min(), -1., 1.)))
        angles.append(angle)
        lengths.append(point.track.length())
        errors.append(point.error)
    return {'max_triangulation_angle_deg': np.array(angles),
            'track_length': np.array(lengths), 'pixel_error': np.array(errors)}


def support_summary(diagnostic, tracks, holdout):
    return {
        'primary_reasons': {
            name: {'points': int((diagnostic['reason'] == index).sum()),
                   'holdout_points': int(((diagnostic['reason'] == index) & holdout).sum()),
                   'nearest_distance': quantiles(diagnostic['nearest_distance'][diagnostic['reason'] == index]),
                   **{key: quantiles(value[diagnostic['reason'] == index]) for key, value in tracks.items()}}
            for index, name in enumerate(SUPPORT_REASONS)},
        'overlapping_rejection_counts': {
            name: int(diagnostic['rejection_flags'][:, index].sum())
            for index, name in enumerate(SUPPORT_REASONS[1:])},
        'primary_reason_priority': list(SUPPORT_REASONS),
    }


def _plane(points):
    centre = points.mean(0)
    centred = points - centre
    eigen, basis = np.linalg.eigh(centred.T @ centred / len(points))
    eigen = np.maximum(eigen, 0)
    normal = basis[:, 0]
    normal *= 1 if normal[np.argmax(np.abs(normal))] >= 0 else -1
    valid = (eigen[0] / max(eigen.sum(), 1e-12) < .04
             and eigen[1] > .04 * eigen[2] and np.sqrt(eigen[0]) <= .03)
    return centre, normal, basis[:, 1:], valid


def compare_visits(early, late, *, max_tangent_distance=.08, min_pairs=10):
    """Compare repeat observations only where their tangential footprints overlap.

    Two nearby surfaces may belong to different objects. Even a repeat offset
    with matching plane normals does not establish sensor drift or ground truth.
    """
    if len(early) < min_pairs or len(late) < min_pairs:
        return {'status': 'insufficient_visit_samples'}
    centre, normal, tangent, valid_early = _plane(early)
    _, normal_late, _, valid_late = _plane(late)
    if not valid_early or not valid_late:
        return {'status': 'nonplanar_visit'}
    normal_angle = float(np.rad2deg(np.arccos(np.clip(abs(normal @ normal_late), 0, 1))))
    if normal_angle > 15:
        return {'status': 'incompatible_visit_normals', 'normal_angle_deg': normal_angle}
    a, b = (early - centre) @ tangent, (late - centre) @ tangent
    d_b, i_b = cKDTree(a).query(b)
    d_a, i_a = cKDTree(b).query(a)
    keep_a, keep_b = d_a <= max_tangent_distance, d_b <= max_tangent_distance
    pairs = np.unique(np.vstack((np.column_stack((np.flatnonzero(keep_a), i_a[keep_a])),
                                  np.column_stack((i_b[keep_b], np.flatnonzero(keep_b))))), axis=0)
    if (len(pairs) < min_pairs or len(np.unique(pairs[:, 0])) < min_pairs
            or len(np.unique(pairs[:, 1])) < min_pairs):
        return {'status': 'no_shared_tangential_footprint', 'pairs': len(pairs)}
    differences = (late[pairs[:, 1]] - early[pairs[:, 0]]) @ normal
    median = float(np.median(differences))
    return {'status': 'compared', 'pairs': len(pairs), 'normal': normal.tolist(),
            'normal_angle_deg': normal_angle, 'signed_median_separation': median,
            'abs_median_separation': abs(median),
            'median_absolute_deviation': float(np.median(np.abs(differences - median))),
            'pair_abs_separation': quantiles(np.abs(differences)),
            'orientation': ('horizontal' if abs(normal[2]) >= np.cos(np.deg2rad(25)) else
                            'vertical' if abs(normal[2]) <= np.sin(np.deg2rad(25)) else 'inclined')}


def temporal_consistency(target, times, centres, *, workers=4, radius=.45,
                         cell_size=.5, max_patches=4000):
    """Compare earliest/latest planar scan visits near visual/LAS overlap.

    A visit is separated by a >3-second local timestamp gap. The selected visits
    must be >=10 seconds apart. Invalid timestamps never get fabricated.
    """
    recipe = {'radius': radius, 'cell_size': cell_size, 'max_patches': max_patches,
              'visit_gap_seconds': 3., 'minimum_visit_separation_seconds': 10.,
              'minimum_samples_per_visit': 10, 'maximum_tangent_distance': .08,
              'maximum_visit_plane_noise': .03, 'maximum_normal_angle_deg': 15.,
              'selection': f'one supplied centre per {cell_size:g}m cell, deterministic ordinal subsample',
              'scope': 'LAS repeat geometry; frozen registration and no image timestamp pairing'}
    if times is None:
        return {'status': 'gps_time_unavailable', 'recipe': recipe, 'patches': [],
                'abs_median_separation_by_orientation': {}}
    finite = np.isfinite(times)
    target, times = target[finite], times[finite]
    if len(target) < 30:
        return {'status': 'insufficient_finite_gps_times', 'recipe': recipe, 'patches': [],
                'abs_median_separation_by_orientation': {}}
    _, index = np.unique(np.floor(centres / cell_size).astype(np.int64), axis=0, return_index=True)
    index = index[::max(1, int(np.ceil(len(index) / max_patches)))]
    tree = cKDTree(target)
    patches, statuses = [], Counter()
    for start in range(0, len(index), 256):
        selected = centres[index[start:start + 256]]
        neighbours = tree.query_ball_point(selected, r=radius, workers=workers)
        for centre, nearby in zip(selected, neighbours):
            nearby = np.asarray(nearby, dtype=np.int64)
            if len(nearby) < 20:
                statuses['insufficient_patch_samples'] += 1
                continue
            nearby = nearby[np.argsort(times[nearby], kind='stable')]
            visits = np.split(nearby, np.flatnonzero(np.diff(times[nearby]) > 3.) + 1)
            visits = [visit for visit in visits if len(visit) >= 10]
            if len(visits) < 2 or np.median(times[visits[-1]]) - np.median(times[visits[0]]) < 10.:
                statuses['no_separated_repeat_visits'] += 1
                continue
            early, late = visits[0], visits[-1]
            value = compare_visits(target[early], target[late])
            statuses[value['status']] += 1
            if value['status'] == 'compared':
                patches.append({'centre': centre.tolist(),
                                'early_time': float(np.median(times[early])),
                                'late_time': float(np.median(times[late])),
                                'early_samples': len(early), 'late_samples': len(late),
                                'visits': len(visits), **value})
    summary = {label: quantiles([p['abs_median_separation'] for p in patches if p['orientation'] == label])
               for label in ('horizontal', 'vertical', 'inclined')}
    return {'status': 'diagnostic_only', 'recipe': recipe, 'candidate_patches': len(index),
            'nonfinite_times_excluded': int((~finite).sum()), 'patch_status_counts': dict(statuses),
            'abs_median_separation_by_orientation': summary, 'patches': patches}


def viewing_side_proxy(temporal, camera_times, camera_centres, *, max_time_offset=.6, side_margin=1.):
    """Flag opposing observation sides using time-near visual camera centres.

    Camera centres approximate the scanner origin. No extrinsic or timing offset
    is inferred: this can flag ambiguity, never certify identical LAS surfaces.
    """
    times, centres = np.asarray(camera_times), np.asarray(camera_centres)
    if (times.ndim != 1 or centres.shape != (len(times), 3) or not len(times)
            or not np.isfinite(times).all() or not np.isfinite(centres).all()):
        raise ValueError('Expected finite camera times and corresponding 3D centres')
    for patch in temporal['patches']:
        indices = [int(np.argmin(np.abs(times - patch[key]))) for key in ('early_time', 'late_time')]
        offsets = [float(abs(times[index] - patch[key]))
                   for index, key in zip(indices, ('early_time', 'late_time'))]
        sides = [(centres[index] - patch['centre']) @ patch['normal'] for index in indices]
        status = ('time_proxy_unavailable' if max(offsets) > max_time_offset else
                  'near_plane_ambiguous' if min(abs(value) for value in sides) <= side_margin else
                  'same_side_proxy' if np.sign(sides[0]) == np.sign(sides[1]) else 'opposite_sides_proxy')
        patch['viewing_side_proxy'] = {'status': status, 'time_offsets_seconds': offsets,
                                       'signed_camera_to_plane_distances': list(map(float, sides))}
    temporal['viewing_side_proxy'] = {
        'kind': 'time_near_visual_camera_centre_not_measured_scanner_origin',
        'maximum_time_offset_seconds': max_time_offset, 'minimum_plane_side_margin': side_margin,
        'status_counts': dict(Counter(p['viewing_side_proxy']['status'] for p in temporal['patches'])),
        'same_side_abs_median_separation_by_orientation': {
            label: quantiles([p['abs_median_separation'] for p in temporal['patches']
                             if p['orientation'] == label and p['viewing_side_proxy']['status'] == 'same_side_proxy'])
            for label in ('horizontal', 'vertical', 'inclined')},
        'limitation': 'Weak viewing-side diagnostic; camera/LiDAR extrinsics and time synchronization unverified',
    }
    return temporal
