"""Image-only, asynchronous full-sequence SfM with frozen validation."""
from datetime import datetime, timezone
import html
import json
from pathlib import Path

import numpy as np
import pycolmap

from .local_data import CONFIG, split_tracks
from .local_evaluation import evaluate_reconstruction, paired_common_support, triangle_validation_split
from .local_reconstruction import (assert_holdout_isolated, bounded_bundle_adjustment,
                                   cheirality_health, database_and_seed, mapper_options)
from .scene import digest, file_digest, load_scene, write_json
from .track_consistency import CONFIG as TRACK_CONFIG, filter_mapped_tracks, training_adjacency

OUTPUT = 'global_sfm'


def load_global(work):
    work = Path(work).resolve(); out = work / OUTPUT; scene = load_scene(out)
    data = json.loads((out / 'frontend.json').read_text())
    if data['scene_fingerprint'] != scene['fingerprint'] or data['config'] != CONFIG.__dict__:
        raise ValueError('Global frontend does not match current scene/configuration')
    features = {}
    for record in scene['images']:
        name = record['name']; row = data['features'][name]
        with np.load(row['path'], allow_pickle=False) as cached:
            metadata = json.loads(str(cached['metadata']))
            expected = digest({'image_sha256': file_digest(record['source']),
                               'mask_sha256': file_digest(record['semantic_mask']),
                               'camera': scene['cameras'][record['camera']], 'fov': scene['fov_degrees'],
                               'config': data['config']})
            if expected != row['token'] or str(cached['token']) != expected or metadata['name'] != name:
                raise ValueError(f'Global feature identity changed: {name}')
            features[name] = {'keypoints': cached['keypoints'], 'scores': cached['scores']}
    return out, scene, data, features


def freeze_global_geometry(work):
    out, scene, data, features = load_global(work)
    work = Path(work).resolve()
    local = work / 'local_sfm/normal_070_094/stages/principal/sparse/0'
    calibration = pycolmap.Reconstruction(local)
    params = {i: camera.params.tolist() for i, camera in calibration.cameras.items()}
    protocol = {'version': 1, 'scene_fingerprint': scene['fingerprint'],
                'frontend_fingerprint': data['fingerprint'],
                'pair_protocol_sha256': file_digest(out / 'pair_protocol.json'),
                'calibration_source_model_sha256': {p.name: file_digest(p) for p in local.glob('*.bin')},
                'calibration': params, 'holdout': {'whole_raw_track_before_geometry': True,
                                                   'fraction': .2, 'seed': 314159,
                                                   'triangle_direct_match_confidence': .9},
                'track_filter': TRACK_CONFIG,
                'bundle_adjustment_stages': ['fixed_calibration', 'triangle_supported', 'shared_calibration'],
                'shared_calibration_limits': 'Same bounded visual-only principal-stage limits as local SfM; final parameters must pass 160-degree monotonicity and positive-depth checks.',
                'independent_exposure_poses': True, 'input_pose_values_used': False, 'las_used': False,
                'scope': 'Full selected L/R sequence, candidate shared calibration from local visual fit. No navigation/LAS alignment or photometric acceptance.'}
    fingerprint = digest(protocol)
    path = out / 'geometry_protocol.json'
    if path.exists():
        if json.loads(path.read_text())['fingerprint'] != fingerprint:
            raise ValueError('Full geometry protocol changed; preserve the existing run')
    else: write_json(path, {**protocol, 'fingerprint': fingerprint,
                           'frozen_at_utc': datetime.now(timezone.utc).isoformat()})
    records = {r['name']: r for r in scene['images']}
    pair_data = []
    for i, entry in enumerate(data['pairs'], 1):
        with np.load(entry['path'], allow_pickle=False) as raw:
            if str(raw['token']) != entry['token']: raise ValueError(f'Match cache token changed: {i}')
            pair_data.append((entry['a'], entry['b'], raw['matches'], raw['scores']))
        if i % 500 == 0: print(f'global split read {i}/{len(data["pairs"])}', flush=True)
    split, held = split_tracks(pair_data, scene['images'])
    split_path = out / 'split.json'
    if split_path.exists() and digest(json.loads(split_path.read_text())) != digest(split):
        raise ValueError('Global heldout split changed; preserve the existing run')
    write_json(split_path, split)
    triangle = triangle_validation_split(split, data['pairs'], records, threshold=.9)
    triangle['selection_frozen_before_geometry'] = True
    triangle_path = out / 'triangle_validation_split.json'
    if triangle_path.exists() and digest(json.loads(triangle_path.read_text())) != digest(triangle):
        raise ValueError('Global triangle validation split changed; preserve the existing run')
    write_json(triangle_path, triangle)
    prepared = {'raw_heldout_tracks': len(split['tracks']), 'heldout_observations': len(held),
                'raw_queries': sum(len(t['queries']) for t in split['tracks']),
                'direct_triangle_tracks': len(triangle['tracks']),
                'triangle_queries': sum(len(t['queries']) for t in triangle['tracks']),
                'frontend_pairs': len(data['pairs']),
                'split_sha256': file_digest(out / 'split.json'),
                'triangle_split_sha256': file_digest(out / 'triangle_validation_split.json')}
    write_json(out / 'split_summary.json', prepared)
    print('Frozen full-sequence validation:', prepared, flush=True)
    return out, scene, data, features, protocol, params


def geometric_scene(scene, params):
    result = {**scene, 'cameras': {label: {**camera, 'params': params[1 if label == 'L' else 2]}
                                    for label, camera in scene['cameras'].items()}}
    result['geometry_calibration_is_candidate'] = True
    return result


def build_global_mapping(work):
    out, scene, data, features, protocol, params = freeze_global_geometry(work)
    split = json.loads((out / 'split.json').read_text())
    mapping = out / 'mapping'
    if (mapping / 'cameras.bin').exists():
        rec = pycolmap.Reconstruction(mapping)
    else:
        db, seed = out / 'database.db', out / 'seed'
        if not (db.exists() and (seed / 'cameras.bin').exists()):
            if db.exists() and not (seed / 'cameras.bin').exists():
                archive = out / f'database_interrupted_{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")}.db'
                db.rename(archive)
            db, seed = database_and_seed(out, geometric_scene(scene, params), scene['images'],
                                         data['pairs'], features, split)
        opts = mapper_options()
        maps = pycolmap.incremental_mapping(db, Path(scene['input_root']) / 'Image',
                                            out / 'mapper_output', options=opts, input_path=seed)
        if not maps: raise RuntimeError('Full mapping did not reconstruct a component')
        rec = max(maps.values(), key=lambda r: r.num_reg_images())
        assert_holdout_isolated(rec, split)
        if cheirality_health(rec)['invalid_depth']: raise RuntimeError('Global mapper model has invalid depths')
        for cid, values in params.items():
            np.testing.assert_array_equal(rec.cameras[cid].params, np.asarray(values))
        mapping.mkdir(exist_ok=True); rec.write(mapping); rec.write_text(mapping)
    names = {i.name for i in rec.images.values() if i.has_pose}
    summary = {'selected': len(scene['images']), 'registered': rec.num_reg_images(),
               'registered_names': sorted(names), 'unregistered_names': sorted(set(features) - names),
               'points': rec.num_points3D(), 'observations': rec.compute_num_observations(),
               'cheirality': cheirality_health(rec), 'pose_prior_used': False, 'LAS_used': False,
               'independent_frame_for_each_exposure': True}
    write_json(out / 'mapping_summary.json', summary)
    write_global_report(out, scene, data)
    print('Full mapping:', {k: v for k, v in summary.items() if k not in ['registered_names', 'unregistered_names']}, flush=True)
    return rec, summary


def evaluate_global_mapping(work):
    """Score the saved mapper result on the same frozen queries as later BA."""
    out, scene, data, features = load_global(work)
    rec = pycolmap.Reconstruction(out / 'mapping')
    split = json.loads((out / 'split.json').read_text())
    triangle = json.loads((out / 'triangle_validation_split.json').read_text())
    assert_holdout_isolated(rec, split)
    path = out / 'mapping_validation'
    path.mkdir(parents=True, exist_ok=True)
    raw = evaluate_reconstruction(rec, scene, features, split, path)
    tri = path / 'triangle_validation'; tri.mkdir(exist_ok=True)
    high = evaluate_reconstruction(rec, scene, features, triangle, tri)
    result = {'ba': {'termination_type': 'incremental_mapping'},
              'registered': rec.num_reg_images(), 'points': rec.num_points3D(),
              'observations': rec.compute_num_observations(), 'raw': raw, 'triangle': high,
              'camera_params': {cid: c.params.tolist() for cid, c in rec.cameras.items()}}
    write_json(path / 'summary.json', result)
    comparison_path = out / 'comparison.json'
    comparison = json.loads(comparison_path.read_text()) if comparison_path.exists() else {'stages': {}}
    comparison['stages'] = {'mapping': result, **comparison['stages']}
    comparison['scope'] = 'Full-scene observation holdout, not independent capture or 3DGS acceptance.'
    write_json(comparison_path, comparison)
    write_global_report(out, scene, data)
    print('Global heldout mapping', high['all_candidate_queries'], flush=True)
    return result


def solve_global(work, *, stop_after='shared_calibration'):
    out, scene, data, features = load_global(work)
    protocol = json.loads((out / 'geometry_protocol.json').read_text())
    mapping = pycolmap.Reconstruction(out / 'mapping')
    split = json.loads((out / 'split.json').read_text())
    triangle = json.loads((out / 'triangle_validation_split.json').read_text())
    params = {int(i): np.asarray(p, dtype=float) for i, p in protocol['calibration'].items()}
    comparison_path = out / 'comparison.json'
    existing = json.loads(comparison_path.read_text()) if comparison_path.exists() else {'stages': {}}
    stages = {'mapping': existing['stages']['mapping']} if 'mapping' in existing['stages'] else {}
    for stage in ['fixed_calibration', 'triangle_supported', 'shared_calibration']:
        path = out / 'stages' / stage; path.mkdir(parents=True, exist_ok=True)
        if (stage in existing['stages'] and (path / 'ba.json').exists()
                and (path / 'validation.json').exists()
                and (path / 'triangle_validation/validation.json').exists()):
            rec = pycolmap.Reconstruction(path / 'sparse/0')
            assert_holdout_isolated(rec, split)
            if stage != 'shared_calibration':
                for cid, p in params.items(): np.testing.assert_array_equal(rec.cameras[cid].params, p)
            stages[stage] = existing['stages'][stage]
            print('Global saved stage', stage, rec.num_reg_images(), rec.num_points3D(), flush=True)
            if stage == stop_after:
                break
            continue
        if (path / 'ba.json').exists():
            rec = pycolmap.Reconstruction(path / 'sparse/0')
            ba = json.loads((path / 'ba.json').read_text())
        else:
            previous = {'fixed_calibration': out / 'mapping',
                        'triangle_supported': out / 'stages/fixed_calibration/sparse/0',
                        'shared_calibration': out / 'stages/triangle_supported/sparse/0'}[stage]
            rec = pycolmap.Reconstruction(previous)
            if stage == 'triangle_supported':
                held = {(n, int(i)) for t in split['tracks'] for n, i in t['observations']}
                mapped_points = {(im.name, idx): int(p.point3D_id)
                                 for im in rec.images.values() if im.has_pose
                                 for idx, p in enumerate(im.points2D) if p.has_point3D()}
                if held.intersection(mapped_points):
                    raise RuntimeError('Heldout observation leaked into mapped training points')
                adjacency, count = training_adjacency(data['pairs'], held,
                    TRACK_CONFIG['match_confidence'], mapped_points=mapped_points)
                filtration = filter_mapped_tracks(rec, adjacency, path)
                filtration['input_graph_edges'] = count
                write_json(path / 'track_filter.json', filtration)
                if not filtration['coverage_admissible']:
                    write_json(path / 'REJECTED.json', {'reason': 'Training coverage fell below frozen threshold', 'coverage': filtration['after']['coverage']})
                    raise RuntimeError('Full-scene track filter rejected by training coverage')
            print('Global BA', stage, rec.num_reg_images(), rec.num_points3D(), rec.compute_num_observations(), flush=True)
            ba = bounded_bundle_adjustment(rec, 'principal' if stage == 'shared_calibration' else 'fixed',
                                           params, protect_depth=True, show_progress=True)
            model = path / 'sparse/0'; model.mkdir(parents=True, exist_ok=True)
            rec.write(model); rec.write_text(model)
            write_json(path / 'ba.json', ba)
        assert_holdout_isolated(rec, split)
        if stage != 'shared_calibration':
            for cid, p in params.items(): np.testing.assert_array_equal(rec.cameras[cid].params, p)
        raw = evaluate_reconstruction(rec, scene, features, split, path)
        tri = path / 'triangle_validation'; tri.mkdir(exist_ok=True)
        high = evaluate_reconstruction(rec, scene, features, triangle, tri)
        stages[stage] = {'ba': ba, 'registered': rec.num_reg_images(), 'points': rec.num_points3D(),
                         'observations': rec.compute_num_observations(), 'raw': raw, 'triangle': high,
                         'camera_params': {cid: c.params.tolist() for cid, c in rec.cameras.items()}}
        write_json(comparison_path, {'stages': stages, 'scope': 'Full-scene observation holdout, not independent capture or 3DGS acceptance.'})
        write_global_report(out, scene, data)
        print('Global heldout', stage, high['all_candidate_queries'], flush=True)
        if stage == stop_after:
            break
    if 'triangle_supported' in stages:
        compare_global_stages(work)
    return stages


def compare_global_stages(work):
    """Compare identical high-confidence queries with valid support in all saved stages."""
    out = Path(work).resolve() / OUTPUT
    path = out / 'comparison.json'
    comparison = json.loads(path.read_text())
    stages = [s for s in ['fixed_calibration', 'triangle_supported', 'shared_calibration']
              if s in comparison['stages']]
    if stages[:2] != ['fixed_calibration', 'triangle_supported']:
        raise ValueError('Need fixed and triangle-supported stages for paired comparison')
    common, values, selected = paired_common_support(
        out, stages, 'triangle_validation', baseline_stage='fixed_calibration')
    from .transfer import paired_track_bootstrap
    comparison['triangle_common_support'] = common
    comparison['paired_track_bootstrap'] = paired_track_bootstrap(
        values, selected, baseline='fixed_calibration', replicates=500)
    if 'shared_calibration' in stages:
        comparison['paired_vs_triangle'] = paired_track_bootstrap(
            values, selected, baseline='triangle_supported', replicates=500)
    write_json(path, comparison)
    print('Global paired high-confidence queries', common['queries'], flush=True)
    return common


def write_global_report(out, scene, data):
    mapping = json.loads((out / 'mapping_summary.json').read_text())
    geo = json.loads((out / 'geometry_summary.json').read_text())
    split = json.loads((out / 'split_summary.json').read_text())
    comparison = json.loads((out / 'comparison.json').read_text()) if (out / 'comparison.json').exists() else {'stages': {}}
    registered = set(mapping['registered_names'])
    by_frame = {(r['frame_index'], r['camera']): r for r in scene['images']}
    frames = sorted({r['frame_index'] for r in scene['images']})
    timeline = []
    for camera in ['L', 'R']:
        timeline.append(f'<p>{camera}：'+''.join(
            f'<span class="{"registered" if by_frame[(frame,camera)]["name"] in registered else "missing"}" '
            f'title="{html.escape(by_frame[(frame,camera)]["name"])} · {"registered" if by_frame[(frame,camera)]["name"] in registered else "not registered"}"></span>'
            for frame in frames)+'</p>')
    def fmt(value): return f'{value:.3f}' if value is not None else '—'
    stage_rows = []
    for name, result in comparison['stages'].items():
        raw = result['raw']['all_candidate_queries']; tri = result['triangle']['all_candidate_queries']
        model_path = 'mapping/' if name == 'mapping' else f'stages/{html.escape(name)}/sparse/0/'
        stage_rows.append(f'<tr><td><a href="{model_path}">{html.escape(name)}</a></td>'
            f'<td>{result["registered"]}</td><td>{result["points"]} / {result["observations"]}</td>'
            f'<td>{fmt(tri["median"])} / {fmt(tri["p90"])}</td>'
            f'<td>{tri["count"]-tri["finite_count"]} / {tri["count"]}</td>'
            f'<td>{fmt(raw["median"])} / {fmt(raw["p90"])}</td>'
            f'<td>{html.escape(result["ba"]["termination_type"])}</td></tr>')
    head = '''<!doctype html><html lang="zh"><meta charset="utf-8"><title>篮球场全量 L/R SfM</title>
<style>body{font:16px system-ui;background:#171b23;color:#ddd;margin:24px}a{color:#8bc5ff}p{line-height:1.7}table{border-collapse:collapse}td,th{border:1px solid #576070;padding:10px}.timeline{overflow:auto;max-width:100%;white-space:nowrap}.timeline p{display:flex;gap:1px;margin:4px 0}.timeline span{display:inline-block;flex:0 0 5px;height:22px}.registered{background:#56bf85}.missing{background:#eb755f}</style>
<h1>篮球场：全部入选 L/R 图像的视觉重建</h1>'''
    intro = (f'<p>仅精确同名 CameraPos 记录的 L/R 共 {len(scene["images"])} 张，逐张独立曝光；'
             '不使用输入 pose 数值或 LAS 来拟合。完整原始轨迹在拟合前整条留出，'
             '闭合三图子集与全部原始查询同时报告。内参来自此前局部视觉候选，不认定为真值。</p>'
             f'<p>候选图像对 {len(data["pairs"])}；E 成功 {geo["successful_E_pairs"]}；'
             f'写入数据库 {geo["database_pairs"]}。留出原始轨迹 {split["raw_heldout_tracks"]} 条、'
             f'查询 {split["raw_queries"]} 个；直接闭合三图查询 {split["triangle_queries"]} 个。</p>'
             f'<p>全场增量建图注册 <strong>{mapping["registered"]}/{mapping["selected"]}</strong> 张，'
             f'{mapping["points"]} 个点、{mapping["observations"]} 个训练观测；'
             f'非正深度训练观测 {mapping["cheirality"]["invalid_depth"]} 个。</p>')
    coverage = '''<h2>注册覆盖</h2><p>每一短线代表一张图，绿色为注册、红色为未注册；横向按帧 33–300 排列。上下两行仅为显示，不表示 L/R 同时曝光。悬停可查看原图文件名。</p><div class="timeline">'''
    table = '''</div><h2>冻结留出集上的阶段对照</h2><p>误差为 3000×4000 原图像素；失败保留分母中。已运行阶段才出现在表内；训练观测数变化后，不应直接用训练残差评价筛选。</p>
<table><tr><th>模型</th><th>注册</th><th>点 / 训练观测</th><th>全部高置信查询 median / P90</th><th>高置信失败</th><th>全部原始查询 median / P90</th><th>求解状态</th></tr>'''
    links = '''</table><p><a href="mapping_summary.json">注册与未注册清单</a> · <a href="geometry_summary.json">几何筛选</a> · <a href="split_summary.json">留出划分</a> · <a href="comparison.json">完整及共同支持对照</a> · <a href="mask_review.html">536 张掩码预览</a></p>
<p>图像检索和两视图 E 成功均只提出约束候选；还需检查跨段回环、外圈和静态实体原图对应、图像曝光/颜色一致性，才能决定 LichtFeld-Studio 的 3DGS 输入。</p></html>'''
    (out / 'index.html').write_text(head + intro + coverage + ''.join(timeline) + table + ''.join(stage_rows) + links)
