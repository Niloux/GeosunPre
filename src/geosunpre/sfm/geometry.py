"""Image-only ray-space estimation and original-pixel diagnostics."""
from itertools import combinations

import numpy as np
import cv2
import pycolmap
from scipy.spatial.transform import Rotation


def stats(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if not len(finite):
        return {'count': int(len(values)), 'finite_count': 0, 'median': None, 'p90': None, 'p99': None}
    q = np.quantile(finite, [.5, .9, .99])
    return {'count': int(len(values)), 'finite_count': int(len(finite)),
            'median': float(q[0]), 'p90': float(q[1]), 'p99': float(q[2])}


def skew(t):
    x,y,z = t
    return np.array([[0,-z,y],[z,0,-x],[-y,x,0]], dtype=float)


def epipolar_pixel_error(cam_a, cam_b, a, b, E):
    """First-order Sampson distance using the fisheye bearing Jacobian.

    Both image derivatives are in ORIGINAL image pixels, including the nonlinear
    unprojection. This is an epipolar distance, not a 3-D reprojection error.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if not len(a):
        return np.empty(0)
    ra, rb = cam_a.rays(a), cam_b.rays(b)
    residual = np.einsum('ni,ij,nj->n', rb, E, ra)
    gradient_squared = np.zeros(len(a))
    for axis in range(2):
        offset = np.zeros(2)
        offset[axis] = .25
        ja = (cam_a.rays(a+offset)-cam_a.rays(a-offset))/.5
        jb = (cam_b.rays(b+offset)-cam_b.rays(b-offset))/.5
        ga = np.einsum('ni,ij,nj->n', rb, E, ja)
        gb = np.einsum('ni,ij,nj->n', jb, E, ra)
        gradient_squared += ga**2+gb**2
    return np.divide(np.abs(residual), np.sqrt(gradient_squared),
                     out=np.full(len(a), np.inf), where=gradient_squared > 1e-24)


def triangulation_diagnostics(cam_a, cam_b, a, b, pose):
    ra, rb = cam_a.rays(a), cam_b.rays(b)
    R, t = pose[:, :3], pose[:, 3]
    if not len(a):
        return np.empty(0, dtype=bool), np.empty(0), np.empty(0)
    pa = np.column_stack((np.eye(3), np.zeros(3)))
    hom = cv2.triangulatePoints(pa, pose, (ra[:, :2]/ra[:, 2:]).T, (rb[:, :2]/rb[:, 2:]).T).T
    xyz = np.divide(hom[:, :3], hom[:, 3:], out=np.full((len(hom), 3), np.nan), where=np.abs(hom[:, 3:]) > 1e-14)
    xyz_b = xyz @ R.T+t
    positive = np.isfinite(xyz).all(axis=1) & (xyz[:, 2] > 0) & (xyz_b[:, 2] > 0)
    error = np.full(len(a), np.inf)
    if positive.any():
        ea = np.linalg.norm(cam_a.project(xyz[positive])-a[positive], axis=1)
        eb = np.linalg.norm(cam_b.project(xyz_b[positive])-b[positive], axis=1)
        error[positive] = np.maximum(ea, eb)
    angles = np.rad2deg(np.arccos(np.clip(np.sum(ra*(rb @ R), axis=1), -1, 1)))
    return positive, error, angles


def fit_pair(cam_a, cam_b, a, b, max_error=3.0, seed=42, fov=160):
    """Fit on 80% of raw matches, leaving the remaining correspondences out."""
    n = len(a)
    blank = {'success': False, 'reason': 'fewer than 24 matches', 'raw_matches': n,
             'verified': 0, 'train_count': 0, 'holdout_count': 0}
    if n < 24:
        return blank, {'verified_mask': np.zeros(n, dtype=bool), 'heldout_mask': np.zeros(n, dtype=bool),
                       'epipolar_error_px': np.full(n, np.inf)}
    shuffled = np.random.default_rng(seed).permutation(n)
    heldout = np.zeros(n, dtype=bool)
    heldout[shuffled[:max(5, n//5)]] = True
    train = np.flatnonzero(~heldout).astype(np.uint32)
    # The generic "calibrated two-view geometry" API can still select raw-pixel
    # F over E. That F is invalid on distorted fisheye pixels. Use the 4.2
    # camera-aware E-only API, whose ray Jacobians retain native pixel units.
    options = pycolmap.RANSACOptions(max_error=max_error, min_num_trials=100,
                                    max_num_trials=10000, confidence=.9999, random_seed=seed)
    result = pycolmap.estimate_relative_pose(
        cam_a.pycolmap(), np.asarray(a, dtype=np.float64)[train],
        cam_b.pycolmap(), np.asarray(b, dtype=np.float64)[train], options)
    if result is None:
        return {**blank, 'reason': 'relative pose estimation failed', 'train_count': len(train),
                'holdout_count': int(heldout.sum())}, {
                    'verified_mask': np.zeros(n, dtype=bool), 'heldout_mask': heldout,
                    'epipolar_error_px': np.full(n, np.inf)}
    pose = result['cam2_from_cam1'].matrix()
    E = skew(pose[:, 3]) @ pose[:, :3]
    errors = epipolar_pixel_error(cam_a, cam_b, a, b, E)
    positive, reproj, angles = triangulation_diagnostics(cam_a, cam_b, a, b, pose)
    # Threshold the same native-pixel metric on fitting and held-out observations.
    epipolar = errors <= max_error
    verified = epipolar & positive & (reproj <= max_error*2)
    rho_a = np.linalg.norm((a-np.array(cam_a.params[2:4]))/np.array(cam_a.params[:2]), axis=1)
    rho_b = np.linalg.norm((b-np.array(cam_b.params[2:4]))/np.array(cam_b.params[:2]), axis=1)
    peripheral = (rho_a > .8*cam_a.distorted_radius(fov)) | (rho_b > .8*cam_b.distorted_radius(fov))
    median_angle = float(np.median(angles[verified])) if verified.any() else 0.
    summary = {'success': int(verified.sum()) >= 15, 'reason': 'image-only diagnostic fit',
               'raw_matches': n, 'train_count': len(train), 'holdout_count': int(heldout.sum()),
               'ransac_train_inliers': result['num_inliers'], 'epipolar_inliers': int(epipolar.sum()),
               'verified': int(verified.sum()), 'estimator': 'camera-aware E-only tangent Sampson LO-RANSAC',
               'low_parallax': median_angle < .5,
               'seed_candidate': bool(verified.sum() >= 30 and median_angle >= 1.),
               'holdout_epipolar_error_px_all_raw': stats(errors[heldout]),
               'holdout_fraction_within_threshold': float(epipolar[heldout].mean()),
               'holdout_verified_fraction': float(verified[heldout].mean()),
               'peripheral_holdout_error_px_all_raw': stats(errors[heldout & peripheral]),
               'verified_two_view_reprojection_max_px': stats(reproj[verified]),
               'verified_triangulation_angle_degrees': stats(angles[verified]),
               'native_pixel_threshold': max_error,
               'note': 'Pairwise holdout only; not independent multi-view/3DGS validation.'}
    arrays = {'pose_cam_b_from_a': pose, 'essential': E, 'verified_mask': verified,
              'heldout_mask': heldout, 'epipolar_error_px': errors,
              'two_view_reprojection_max_px': reproj, 'triangulation_angle_degrees': angles}
    return summary, arrays




def rotation_graph_diagnostics(edges, image_names, review_threshold_degrees=5.0):
    """Check triangle closures of image-derived R_b_from_a, without pose priors.

    A bad cycle identifies an inconsistent triangle, not its erroneous edge.
    Small closure errors cannot certify translation, tracks or scene rigidity.
    """
    adjacency = {name: set() for name in image_names}
    rotations, labels = {}, {}
    for a, b, rotation, label in edges:
        adjacency[a].add(b)
        adjacency[b].add(a)
        rotations[a, b] = np.asarray(rotation)
        rotations[b, a] = np.asarray(rotation).T
        labels[tuple(sorted((a, b)))] = label
    remaining = set(adjacency)
    components = []
    while remaining:
        seed = min(remaining)
        reached, pending = {seed}, [seed]
        while pending:
            for neighbor in adjacency[pending.pop()]:
                if neighbor not in reached:
                    reached.add(neighbor)
                    pending.append(neighbor)
        remaining -= reached
        components.append(sorted(reached))
    cycles, per_edge = [], {key: [] for key in labels}
    for a in sorted(adjacency):
        for b, c in combinations(sorted(n for n in adjacency[a] if n > a), 2):
            if c not in adjacency[b]:
                continue
            closure = rotations[c, a] @ rotations[b, c] @ rotations[a, b]
            error = float(np.rad2deg(Rotation.from_matrix(closure).magnitude()))
            cycles.append({'images': [a, b, c], 'rotation_closure_degrees': error})
            for key in (tuple(sorted((a, b))), tuple(sorted((b, c))), tuple(sorted((c, a)))):
                per_edge[key].append(error)
    values = [c['rotation_closure_degrees'] for c in cycles]
    pairs = []
    for key, errors in per_edge.items():
        pairs.append({'label': labels[key], 'images': list(key), 'triangle_count': len(errors),
                      'rotation_closure_degrees': stats(errors),
                      'triangles_above_review_threshold': sum(e > review_threshold_degrees for e in errors)})
    return {'analysis_version': 1,
            'scope': 'Rotation consistency of pairwise image-only estimates; no global SfM or pose prior.',
            'review_threshold_degrees': review_threshold_degrees,
            'summary': {'components': len(components), 'component_sizes': [len(c) for c in components],
                        'triangles': len(cycles), 'rotation_closure_degrees': stats(values),
                        'triangles_above_review_threshold': sum(e > review_threshold_degrees for e in values)},
            'components': components, 'pairs': pairs, 'triangles': cycles,
            'automatically_filtered': False,
            'limitation': 'A failed cycle does not identify which edge is wrong. The review threshold is not a 3DGS acceptance threshold.'}


def build_tracks(pair_data, images):
    """Union stable feature IDs, rejecting merges with two features in one image."""
    parent, image_sets, sizes = {}, {}, {}
    def find(node):
        if node not in parent:
            parent[node], image_sets[node], sizes[node] = node, {node[0]}, 1
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != node:
            next_node = parent[node]
            parent[node] = root
            node = next_node
        return root
    edges = []
    for a,b,matches,scores in pair_data:
        edges.extend((float(score), (a,int(ia)), (b,int(ib))) for (ia,ib),score in zip(matches,scores))
    rejected, accepted = 0, 0
    for _,a,b in sorted(edges, key=lambda e: (-e[0], e[1], e[2])):
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        if image_sets[ra] & image_sets[rb]:
            rejected += 1
            continue
        if sizes[ra] < sizes[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        image_sets[ra] |= image_sets.pop(rb)
        sizes[ra] += sizes.pop(rb)
        accepted += 1
    grouped = {}
    for node in parent:
        grouped.setdefault(find(node), []).append(node)
    tracks = [sorted(nodes) for nodes in grouped.values() if len(nodes) >= 2]
    labels = {im['name']: im['camera'] for im in images}
    histogram = {}
    for nodes in tracks:
        histogram[str(len(nodes))] = histogram.get(str(len(nodes)), 0)+1
    return {'tracks': len(tracks), 'tracks_length_ge3': sum(len(t)>=3 for t in tracks),
            'cross_camera_tracks': sum(len({labels[n] for n,k in t})>1 for t in tracks),
            'length_histogram': histogram, 'conflicting_edge_merges_rejected': rejected,
            'accepted_merges': accepted, 'triangulated_multiview_points': 0,
            'note': 'Feature tracks only; no multi-view SfM/BA has been run.'}, tracks
