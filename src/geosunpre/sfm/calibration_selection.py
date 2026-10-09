"""Choose connected, informative calibration views without scene frame ranges."""

from collections import Counter, defaultdict
import math

import numpy as np

from geosunpre.camera import Camera
from geosunpre.sfm.geometry import fit_pair


def _angular_bins(camera, pixels, fov):
    rays = camera.rays(pixels)
    theta = np.arccos(np.clip(rays[:, 2] / np.linalg.norm(rays, axis=1), -1, 1))
    radial = np.minimum((theta / np.deg2rad(fov / 2) * 4).astype(int), 3)
    sector = ((np.arctan2(rays[:, 1], rays[:, 0]) + np.pi) / (2 * np.pi) * 8).astype(int) % 8
    return radial * 8 + sector


def select_calibration(scene, entries, features):
    """Return a bounded connected subset and its reproducible selection report.

    Match support proposes views; native fisheye verification, parallax and
    angular coverage decide which proposals enter the subset. The 64-view
    budget bounds the local solve, not a frame interval or scene category.
    Information scores are selection proxies, not calibration covariance.
    """
    records = {row['name']: row for row in scene['images']}
    cameras = {label: Camera.from_dict(value) for label, value in scene['cameras'].items()}
    ranks = {label: {row['name']: i for i, row in enumerate(
        r for r in scene['images'] if r['camera'] == label)} for label in cameras}
    bins = {name: _angular_bins(cameras[row['camera']], features[name]['keypoints'], scene['fov_degrees'])
            for name, row in records.items()}
    edges, adjacency, seed_groups = {}, defaultdict(list), defaultdict(list)
    for entry in entries:
        a, b = entry['a'], entry['b']
        with np.load(entry['path'], allow_pickle=False) as raw:
            matches = raw['matches'][raw['scores'] >= .5]
        if len(matches) < 30:
            continue
        angular_fraction = min(len(np.unique(bins[a][matches[:, 0]])),
                               len(np.unique(bins[b][matches[:, 1]]))) / 32
        score = len(matches) * math.sqrt(angular_fraction)
        key = (a, b)
        edges[key] = (entry, score)
        adjacency[a].append(key)
        adjacency[b].append(key)
        if records[a]['camera'] == records[b]['camera']:
            label = records[a]['camera']
            gap = abs(ranks[label][a] - ranks[label][b])
            seed_groups[label, int(math.log2(max(1, gap)))].append(key)

    evaluated = {}

    def verify(key):
        if key not in evaluated:
            a, b = key
            with np.load(edges[key][0]['path'], allow_pickle=False) as raw:
                matches = raw['matches']
            metrics, arrays = fit_pair(cameras[records[a]['camera']], cameras[records[b]['camera']],
                features[a]['keypoints'][matches[:, 0]], features[b]['keypoints'][matches[:, 1]],
                seed=271828, fov=scene['fov_degrees'])
            verified = matches[arrays['verified_mask']]
            angle = metrics.get('verified_triangulation_angle_degrees', {}).get('median') or 0.
            covered = [set(map(int, bins[name][verified[:, i]])) for i, name in enumerate(key)]
            evaluated[key] = (len(verified), float(angle), covered)
        return evaluated[key]

    seeds = []
    # Search different temporal distances in each physical camera, so near
    # duplicates cannot occupy the entire seed shortlist.
    for group in seed_groups.values():
        for key in sorted(group, key=lambda k: (-edges[k][1], k))[:3]:
            count, angle, covered = verify(key)
            if count >= 100 and angle >= 2:
                score = count * min(angle, 10) * math.sqrt(min(map(len, covered)) / 32)
                seeds.append((score, key))
    if not seeds:
        raise RuntimeError('No calibration seed with sufficient verified support and parallax')
    _, seed = min(seeds, key=lambda item: (-item[0], item[1]))
    selected = set(seed)
    counts = Counter(records[name]['camera'] for name in selected)
    coverage = {label: set() for label in cameras}
    for name, covered in zip(seed, verify(seed)[2]):
        coverage[records[name]['camera']].update(covered)
    budget = min(64, len(records))
    while len(selected) < budget:
        frontier = defaultdict(list)
        for name in sorted(selected):
            for key in adjacency[name]:
                other = key[1] if key[0] == name else key[0]
                if other not in selected:
                    frontier[other].append(key)
        if not frontier:
            break
        ordered = sorted(frontier, key=lambda name: (
            -sum(edges[k][1] for k in frontier[name]) / (1 + counts[records[name]['camera']]), name))
        proposals = []
        for name in ordered:
            label = records[name]['camera']
            has_support, gain, support = False, set(), 0.
            for key in sorted(frontier[name], key=lambda k: (-edges[k][1], k))[:3]:
                count, angle, covered = verify(key)
                if count < 30 or angle < .5:
                    continue
                has_support = True
                gain.update(covered[key.index(name)])
                support += count * min(angle, 5)
            if has_support:
                score = support * (1 + len(gain - coverage[label]) / 8) / (1 + counts[label])
                proposals.append((score, name, gain))
                if len(proposals) == 8:
                    break
        if not proposals:
            break
        _, name, gain = min(proposals, key=lambda item: (-item[0], item[1]))
        selected.add(name)
        label = records[name]['camera']
        counts[label] += 1
        coverage[label].update(gain)
    if set(counts) != set(cameras) or min(counts.values()) < 3:
        raise RuntimeError('Calibration needs a connected, verified subset with at least three views per camera')
    images = [row for row in scene['images'] if row['name'] in selected]
    count, angle, _ = verify(seed)
    report = {'version': 1, 'method': 'verified connected support with parallax and angular coverage',
        'image_names': [row['name'] for row in images], 'image_budget': budget,
        'seed': {'images': list(seed), 'verified_matches': count, 'median_triangulation_angle_deg': angle},
        'images_per_camera': dict(counts), 'occupied_angular_bins_per_camera': {
            label: len(value) for label, value in coverage.items()},
        'verified_proposals': len(evaluated), 'input_pose_or_las_used': False,
        'scope': 'Selection proxies under input intrinsics; not independent calibration accuracy.'}
    return images, report
