"""Training-only closed-triangle support for mapped feature tracks.

Closed matching cycles are a consistency heuristic, not static-point truth.
Validation observations must be removed before building the support graph.
"""
from collections import defaultdict
from datetime import datetime, timezone
import html
import json
from pathlib import Path

import numpy as np
import pycolmap

from .local_data import load_local, train_edge_mask
from .local_evaluation import evaluate_reconstruction, paired_common_support
from .local_reconstruction import assert_holdout_isolated, bounded_bundle_adjustment
from .scene import digest, file_digest, write_json
from .transfer import paired_track_bootstrap

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


def run_track_probe(work):
    work = Path(work).resolve(); out = work / 'track_consistency'; out.mkdir(exist_ok=True)
    local, ls, ld, lf = load_local(work)
    transfer, ts, td, tf = load_local(work, output_name='transfer_sfm')
    candidate = pycolmap.Reconstruction(local / 'normal_070_094/stages/principal/sparse/0')
    params = {i: c.params.copy() for i, c in candidate.cameras.items()}
    cases = [(local, ls, ld, lf, 'normal_070_094', 'principal'),
             (local, ls, ld, lf, 'turn_190_214', 'principal'),
             (transfer, ts, td, tf, 'transfer_145_169', 'from_normal')]
    protocol = {'version': 1, 'config': CONFIG, 'fixed_calibration': {i: p.tolist() for i, p in params.items()},
                'source_models': {label: {p.name: file_digest(p) for p in (base / label / 'stages' / source / 'sparse/0').glob('*.bin')}
                                  for base, _, _, _, label, source in cases},
                'split_hashes': {label: {f: file_digest(base / label / f) for f in ['split.json', 'triangle_validation_split.json']}
                                 for base, _, _, _, label, _ in cases},
                'comparison': 'Fixed shared candidate calibration. Refit baseline poses/points; copy baseline and retain only closed-triangle-supported connected point components; refit poses/points. Validation unchanged.',
                'solver': {'initial_trust_region_radius': 10000, 'max_trust_region_radius': 1000000, 'max_iterations': 400},
                'selection_uses_validation_or_pose_values': False,
                'scope': 'Exploratory training-track diagnostic on previously inspected segments, not an independent final acceptance.'}
    fingerprint = digest(protocol)
    if (out / 'protocol.json').exists():
        if json.loads((out / 'protocol.json').read_text())['fingerprint'] != fingerprint:
            raise ValueError('Track probe inputs changed; use a fresh output')
    else:
        write_json(out / 'protocol.json', {**protocol, 'fingerprint': fingerprint,
                   'frozen_at_utc': datetime.now(timezone.utc).isoformat()})
    results = {}
    for base, scene, data, features, label, source in cases:
        dest = out / label; dest.mkdir(exist_ok=True)
        split = json.loads((base / label / 'split.json').read_text())
        triangle = json.loads((base / label / 'triangle_validation_split.json').read_text())
        held = {(n, int(i)) for t in split['tracks'] for n, i in t['observations']}
        adjacency, count = training_adjacency(data['segments'][label]['pairs'], held, CONFIG['match_confidence'])
        stages = {}
        for stage in ['baseline', 'triangle_supported']:
            path = dest / 'stages' / stage; path.mkdir(parents=True, exist_ok=True)
            if (path / 'ba.json').exists():
                rec = pycolmap.Reconstruction(path / 'sparse/0')
                ba = json.loads((path / 'ba.json').read_text())
            else:
                src = base / label / 'stages' / source if stage == 'baseline' else dest / 'stages/baseline'
                rec = pycolmap.Reconstruction(src / 'sparse/0')
                for cid, p in params.items(): rec.cameras[cid].params = p.copy()
                if stage == 'triangle_supported':
                    filtration = filter_mapped_tracks(rec, adjacency, path)
                    if not filtration['coverage_admissible']:
                        write_json(path / 'REJECTED.json', {'reason': 'Insufficient training coverage before validation', **filtration['after']})
                        raise RuntimeError(f'{label}: triangle filter reduced coverage too far')
                print(f'{label} {stage}: {rec.num_points3D()} points, {rec.compute_num_observations()} observations', flush=True)
                assert_holdout_isolated(rec, split)
                ba = bounded_bundle_adjustment(rec, 'fixed', params, protect_depth=True, trust_region=(10000, 1000000))
                model = path / 'sparse/0'; model.mkdir(parents=True, exist_ok=True)
                rec.write(model); rec.write_text(model)
                write_json(path / 'ba.json', ba)
            assert_holdout_isolated(rec, split)
            for cid, p in params.items(): np.testing.assert_array_equal(rec.cameras[cid].params, p)
            for im in rec.images.values():
                np.testing.assert_array_equal(np.array([p.xy for p in im.points2D]), features[im.name]['keypoints'])
            raw = evaluate_reconstruction(rec, scene, features, split, path)
            tri = path / 'triangle_validation'; tri.mkdir(exist_ok=True)
            validation = evaluate_reconstruction(rec, scene, features, triangle, tri)
            stages[stage] = {'ba': ba, 'raw': raw, 'triangle': validation, 'points': rec.num_points3D(),
                             'observations': rec.compute_num_observations(), 'registered': rec.num_reg_images()}
            print(label, stage, 'triangle queries:', validation['all_candidate_queries'], flush=True)
        common, values, take = paired_common_support(dest, list(stages), 'triangle_validation', baseline_stage='baseline')
        results[label] = {'stages': stages, 'triangle_common_support': common,
                          'paired_track_bootstrap': paired_track_bootstrap(values, take, baseline='baseline'),
                          'training_graph_edges': count}
        write_json(dest / 'comparison.json', results[label]); write_json(out / 'comparison.json', results)
        write_probe_review(out, results)
    return results


def write_probe_review(out, results):
    rows = []
    for label, result in results.items():
        for stage, s in result['stages'].items():
            a = s['triangle']['all_candidate_queries']; b = s['raw']['all_candidate_queries']
            rows.append(f'<tr><td>{html.escape(label)}</td><td>{stage}</td><td>{s["registered"]}</td><td>{s["points"]} / {s["observations"]}</td>'
                        f'<td>{a["median"]:.3f} / {a["p90"]:.3f}</td><td>{a["count"]-a["finite_count"]} / {a["count"]}</td>'
                        f'<td>{b["median"]:.3f} / {b["p90"]:.3f}</td><td>{html.escape(s["ba"]["termination_type"])}</td></tr>')
    (out / 'index.html').write_text('''<!doctype html><html lang="zh"><meta charset="utf-8"><title>训练轨迹一致性对照</title>
<style>body{font:16px system-ui;background:#171b23;color:#ddd;margin:24px}a{color:#8bc5ff}p{line-height:1.7}table{border-collapse:collapse}td,th{border:1px solid #576070;padding:10px}</style>
<h1>闭合三图支持的训练轨迹检查</h1><p>三段统一固定 70–94 候选内参，每张曝光独立优化姿态。仅用训练匹配的闭合三角形支持观测，三个匹配置信度均 ≥0.5；拆开互不连接的部分并删除没有三图支持的观测，再优化姿态与三维点。没有移动特征点；留出支持点和查询点均未改变或进入训练图。</p>
<p>这是已有片段上的探索性诊断。闭合匹配仍可能对应重复结构或遮挡边界，不能当作静态实体真值。删减训练观测后的训练误差不可直接当作质量提升。以下误差均为原图像素，失败保留分母。</p>
<table><tr><th>片段</th><th>训练轨迹</th><th>注册</th><th>点 / 观测</th><th>全部高置信查询 median / P90</th><th>预测失败</th><th>全部原始查询 median / P90</th><th>求解状态</th></tr>'''+''.join(rows)+'''
</table><p><a href="protocol.json">试验约定</a> · <a href="comparison.json">完整统计与共同支持对照</a></p></html>''')
