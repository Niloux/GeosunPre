"""Reliable surface correspondences and weights for joint registration.

Point precision is conditional on fixed visual cameras/intrinsics and a pixel
noise proxy. It supplies relative weights, not a calibrated accuracy estimate.
"""

import numpy as np
from scipy.spatial import cKDTree

from geosunpre.registration.geometry import surfaces


RECIPE = {
    'maximum_surface_noise': .025, 'maximum_scale_normal_angle_deg': 15.,
    'maximum_scale_plane_separation': .025, 'visual_neighbours': [16, 32],
    'maximum_visual_neighbourhood_radius': 1.5, 'maximum_visual_surface_noise': .05,
    'maximum_visual_plane_ratio': .025, 'maximum_visual_normal_angle_deg': 25.,
    'maximum_point_precision_proxy_std': .3, 'pixel_noise_proxy_floor': 1.,
    'maximum_point_only_precision_proxy_std': .1,
    'candidate_neighbours': 8,
    'parallel_face_ambiguity_distance_margin': .1, 'parallel_face_separation': .05,
    'normal_direction_grid': 4, 'minimum_direction_support': 30,
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


def coherent_las(target, *, voxel_size, workers=4):
    fine = surfaces(target, voxel_size=voxel_size, workers=workers)
    coarse = surfaces(target, voxel_size=2 * voxel_size, workers=workers)
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
    """Balance actual normal directions and regions, with support and confidence.

    Sparse directions receive proportionally less mass. Horizontal/vertical
    labels are descriptive only and no longer define optimization weights.
    """
    z = np.abs(normals[:, 2])
    orientation = np.where(z >= np.cos(np.deg2rad(25)), 0,
                           np.where(z <= np.sin(np.deg2rad(25)), 1, 2))
    weights = np.zeros(len(tiles))
    axis = np.argmax(np.abs(normals), axis=1)
    sign = np.where(normals[np.arange(len(normals)), axis] < 0, -1., 1.)
    directions = np.rint(normals * sign[:, None] * RECIPE['normal_direction_grid']).astype(int)
    _, groups, support = np.unique(directions, axis=0, return_inverse=True, return_counts=True)
    masses = np.minimum(1., support / RECIPE['minimum_direction_support'])
    masses /= masses.sum()
    for group, mass in enumerate(masses):
        mask = groups == group
        unique, inverse, counts = np.unique(tiles[mask], axis=0, return_inverse=True, return_counts=True)
        weights[mask] = mass * confidence[mask] / (len(unique) * counts[inverse])
    return weights, orientation
