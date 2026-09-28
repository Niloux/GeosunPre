"""Training-only closed-triangle support for mapped feature tracks.

Closed matching cycles are a consistency heuristic, not static-point truth.
Validation observations must be removed before building the support graph.
"""
from collections import defaultdict
from pathlib import Path

import numpy as np
import pycolmap

from geosunpre.sfm.tracks import train_edge_mask
from geosunpre.artifacts import write_json

CONFIG = {'match_confidence': .5, 'minimum_component_images': 3,
          'minimum_training_observations_per_image': 100,
          'minimum_occupied_8x8_cells_per_image': 12}


def training_adjacency(entries, held, threshold=.5, mapped_points=None):
    adjacency = defaultdict(dict)
    edges = 0
    for entry in entries:
        with np.load(entry['path']) as raw:
            take = train_edge_mask(entry['a'], entry['b'], raw['matches'], held)
            take &= raw['scores'] >= threshold
            for (i, j), score in zip(raw['matches'][take], raw['scores'][take]):
                a, b = (entry['a'], int(i)), (entry['b'], int(j))
                # The later filter only asks about edges within one mapped
                # training point. Discard every other edge before building the
                # full-scene graph; this leaves its triangle decisions intact.
                if mapped_points is not None:
                    point_id = mapped_points.get(a)
                    if point_id is None or point_id != mapped_points.get(b):
                        continue
                adjacency[a][b] = float(score)
                adjacency[b][a] = float(score)
                edges += 1
    return adjacency, edges


def triangle_components(observations, adjacency):
    """Split a mapped track into connected components of supported triangles.

    Two disjoint triangles joined only by one edge remain separate. A path of
    edges without a closed three-view triangle does not certify a point.
    """
    nodes = set(observations)
    graph = defaultdict(set)
    triangles = 0
    for a in sorted(nodes):
        neighbors = nodes.intersection(adjacency.get(a, {}))
        for b in sorted(n for n in neighbors if n > a):
            for c in sorted(n for n in neighbors.intersection(adjacency.get(b, {})) if n > b):
                if len({a[0], b[0], c[0]}) != 3:
                    continue
                triangles += 1
                graph[a].update([b, c]); graph[b].update([a, c]); graph[c].update([a, b])
    remaining = set(graph)
    components = []
    while remaining:
        start = min(remaining)
        reached, pending = {start}, [start]
        while pending:
            for neighbor in graph[pending.pop()] - reached:
                reached.add(neighbor); pending.append(neighbor)
        remaining -= reached
        components.append(sorted(reached))
    return sorted(components), triangles


def reconstruction_coverage(rec):
    rows = {}
    adjacency = {int(i): set() for i in rec.reg_image_ids()}
    for point in rec.points3D.values():
        ids = {int(e.image_id) for e in point.track.elements}
        for image_id in ids:
            adjacency[image_id].update(ids - {image_id})
    for im in rec.images.values():
        if not im.has_pose:
            continue
        xy = np.array([p.xy for p in im.points2D if p.has_point3D()])
        c = rec.cameras[im.camera_id]
        cells = len(np.unique(np.floor(xy / [c.width, c.height] * 8).astype(int), axis=0)) if len(xy) else 0
        rows[im.name] = {'observations': len(xy), 'occupied_8x8_cells': cells}
    remaining = set(adjacency); components = []
    while remaining:
        start = min(remaining); reached, pending = {start}, [start]
        while pending:
            for n in adjacency[pending.pop()] - reached:
                reached.add(n); pending.append(n)
        remaining -= reached; components.append(sorted(reached))
    return {'per_image': rows, 'image_components': components,
            'min_observations': min((r['observations'] for r in rows.values()), default=0),
            'min_occupied_cells': min((r['occupied_8x8_cells'] for r in rows.values()), default=0)}


def filter_mapped_tracks(rec, adjacency, directory):
    names = {i: im.name for i, im in rec.images.items()}
    ids = {im.name: i for i, im in rec.images.items()}
    before = {'points': rec.num_points3D(), 'observations': rec.compute_num_observations(),
              'coverage': reconstruction_coverage(rec)}
    decisions = []; supported_triangles = 0
    for pid in sorted(rec.points3D):
        point = rec.points3D[pid]
        observations = sorted((names[e.image_id], int(e.point2D_idx)) for e in point.track.elements)
        components, triangles = triangle_components(observations, adjacency)
        supported_triangles += triangles
        if len(components) == 1 and components[0] == observations:
            continue
        xyz, color = point.xyz.copy(), point.color.copy()
        rec.delete_point3D(pid)
        created = []
        for nodes in components:
            if len(nodes) < CONFIG['minimum_component_images']:
                continue
            track = pycolmap.Track([pycolmap.TrackElement(ids[n], i) for n, i in nodes])
            new_id = rec.add_point3D(xyz.copy(), track, color.copy())
            created.append({'point_id': int(new_id), 'observations': nodes})
        decisions.append({'original_point_id': int(pid), 'original_observations': observations,
                          'replacement_components': created, 'triangles': triangles})
    after = {'points': rec.num_points3D(), 'observations': rec.compute_num_observations(),
             'coverage': reconstruction_coverage(rec)}
    admissible = (len(after['coverage']['image_components']) == 1
                  and after['coverage']['min_observations'] >= CONFIG['minimum_training_observations_per_image']
                  and after['coverage']['min_occupied_cells'] >= CONFIG['minimum_occupied_8x8_cells_per_image'])
    result = {'before': before, 'after': after, 'changed_original_tracks': len(decisions),
              'split_original_tracks': sum(len(d['replacement_components']) > 1 for d in decisions),
              'supported_triangles': supported_triangles, 'coverage_admissible': admissible,
              'note': 'No pose or query error used for support decisions; matching cycles can still be wrong or view dependent.'}
    write_json(Path(directory) / 'track_filter.json', result)
    write_json(Path(directory) / 'track_decisions.json', decisions)
    return result
