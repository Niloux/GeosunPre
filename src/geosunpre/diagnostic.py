"""A bounded, reproducible LoMa probe; never writes a production SfM model."""
from dataclasses import asdict
import csv
import html
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw

from .camera import Camera
from .frontend import LoMaFrontend
from .geometry import (build_tracks, epipolar_pixel_error, fit_pair, prior_relative_pose,
                       rotation_graph_diagnostics, skew, stats)
from .scene import digest, load_scene, write_json

DEFAULT_FRAMES = (33,34,35,79,80,81,139,140,141,199,200,201,202,298,299,300)
DIAGNOSTIC_VERSION = 2


def make_pairs(images):
    pairs = []
    low, high = min(im['frame_index'] for im in images), max(im['frame_index'] for im in images)
    for i,a in enumerate(images):
        for b in images[i+1:]:
            gap = abs(a['frame_index']-b['frame_index'])
            tags = []
            if a['camera'] == b['camera'] and gap in (1,2):
                tags.append('temporal')
            if a['camera'] != b['camera'] and gap <= 1:
                tags.append('cross_camera')
            if high-low > 10 and min(a['frame_index'],b['frame_index']) <= low+2 and max(a['frame_index'],b['frame_index']) >= high-2:
                tags.append('loop_candidate')
            if tags:
                pairs.append((a,b,tags))
    return pairs


def label(record):
    return f'{record["camera"]}{record["frame_index"]:05d}'


def preview(path, a, b, xy_a, xy_b, arrays, threshold):
    pictures = []
    scales = []
    for record in (a,b):
        with Image.open(record['source']) as im:
            scale = min(360/im.width, 480/im.height)
            pictures.append(im.resize((round(im.width*scale),round(im.height*scale))).convert('RGB'))
            scales.append(scale)
    width = sum(p.width for p in pictures)
    canvas = Image.new('RGB',(width, max(p.height for p in pictures)+32),(24,24,24))
    canvas.paste(pictures[0],(0,32)); canvas.paste(pictures[1],(pictures[0].width,32))
    draw = ImageDraw.Draw(canvas)
    draw.text((6,8), f'{label(a)} / {label(b)} | raw {len(xy_a)} | verified {int(arrays["verified_mask"].sum())}', fill='white')
    selected = np.arange(len(xy_a))
    if len(selected) > 100:
        selected = np.random.default_rng(1).choice(selected,100,replace=False)
    for idx in selected:
        p = (xy_a[idx]-.5)*scales[0]+[0,32]
        q = (xy_b[idx]-.5)*scales[1]+[pictures[0].width,32]
        color = (70,220,100) if arrays['verified_mask'][idx] else (215,80,70)
        if arrays['heldout_mask'][idx] and arrays['verified_mask'][idx]:
            color = (70,190,255)
        draw.line((tuple(p),tuple(q)),fill=color,width=1)
        draw.ellipse((p[0]-2,p[1]-2,p[0]+2,p[1]+2),fill=color)
        draw.ellipse((q[0]-2,q[1]-2,q[0]+2,q[1]+2),fill=color)
    canvas.save(path, quality=88)


def summarize_prior_conventions(pair_results, scene):
    groups = {}
    cameras = {k:Camera.from_dict(v) for k,v in scene['cameras'].items()}
    for a,b,raw,xy_a,xy_b,metrics,arrays in pair_results:
        if a['camera'] != b['camera'] or not metrics['success']:
            continue
        take = arrays['verified_mask']
        if take.sum() < 30:
            continue
        for order in ('xyzw','wxyz'):
            for direction in ('camera_to_world','world_to_camera'):
                pose = prior_relative_pose(a['prior'],b['prior'],order,direction)
                if pose is None:
                    continue
                errors = epipolar_pixel_error(cameras[a['camera']],cameras[b['camera']],
                                              xy_a[take],xy_b[take],skew(pose[:,3]) @ pose[:,:3])
                groups.setdefault(f'{order}/{direction}',[]).append({
                    'pair': [label(a),label(b)], 'error_px': stats(errors)})
    return {'scope': 'Vendor convention comparison on image-verified same-camera correspondences; no prior fitted.',
            'candidates': {key:{'pairs':values, 'median_of_pair_medians_px':float(np.median([
                v['error_px']['median'] for v in values if v['error_px']['median'] is not None]))}
                for key,values in groups.items()}, 'automatically_adopted': False}


def compare_principal_conventions(pair_results, scene, threshold, seed):
    signs = {'x_right_y_up': (1,-1), 'x_right_y_down': (1,1),
             'x_left_y_up': (-1,-1), 'x_left_y_down': (-1,1)}
    cameras = {k:Camera.from_dict(v) for k,v in scene['cameras'].items()}
    chosen = [r for r in pair_results if r[0]['camera'] == r[1]['camera']
              and abs(r[0]['frame_index']-r[1]['frame_index']) == 1
              and min(r[0]['frame_index'],r[1]['frame_index']) in (33,79,139,199,200,201,299)
              and r[5]['success'] and len(r[3]) >= 80]
    results = []
    for a,b,raw,pa,pb,_,_ in chosen:
        common = np.ones(len(pa), dtype=bool)
        variants = {name:cameras[a['camera']].with_principal_signs(*s) for name,s in signs.items()}
        for cam in variants.values():
            common &= cam.valid(pa,scene['fov_degrees']) & cam.valid(pb,scene['fov_degrees'])
        for name,cam in variants.items():
            metrics,_ = fit_pair(cam,cam,pa[common],pb[common],max_error=threshold,seed=seed,fov=scene['fov_degrees'])
            results.append({'pair':[label(a),label(b)], 'camera':a['camera'], 'convention':name,
                            'common_raw_matches':int(common.sum()), 'metrics':metrics})
    summary = []
    for camera in cameras:
        for convention in signs:
            rr = [r for r in results if r['camera']==camera and r['convention']==convention and r['metrics']['success']]
            medians = [r['metrics']['holdout_epipolar_error_px_all_raw']['median'] for r in rr]
            fractions = [r['metrics']['holdout_fraction_within_threshold'] for r in rr]
            summary.append({'camera':camera, 'convention':convention, 'successful_pairs':len(rr),
                            'median_of_pair_holdout_medians_px':float(np.median(medians)) if medians else None,
                            'mean_pair_holdout_inlier_fraction':float(np.mean(fractions)) if fractions else None})
    return {'scope':'Same cached correspondences, common FOV, identical pairwise fit/holdout split.',
            'automatically_adopted':False, 'summary':summary, 'pairs':results,
            'limitation':'Pairwise self-calibration evidence cannot by itself certify the vendor convention.'}


def report_html(out, report):
    cards = []
    for row in report['pairs']:
        h = row.get('holdout_epipolar_error_px_all_raw',{})
        fmt = lambda v: f'{v:.2f}' if v is not None else '—'
        cycles = row.get('rotation_cycles', {})
        cycle_text = ''
        if cycles:
            cycle_text = (f" · 三图旋转闭合中位数 {fmt(cycles['rotation_closure_degrees']['median'])}°"
                          f" / {cycles['triangle_count']} 个闭环")
        cards.append(f'''<article data-kind="{html.escape(' '.join(row['tags']))}">
<h3>{html.escape(row['label'])}</h3><p>{html.escape(', '.join(row['tags']))}
 · raw {row['raw_matches']} → verified {row['verified']}
 · 留出极线距离（原图像素）median {fmt(h.get('median'))} / P90 {fmt(h.get('p90'))}{cycle_text}</p>
<a href="pairs/{row['artifact']}.json">逐对统计</a>
<a href="previews/{row['artifact']}.jpg"><img loading="lazy" src="previews/{row['artifact']}.jpg"></a></article>''')
    title = '篮球场 LoMa / 原生鱼眼诊断'
    summary = report['summary']
    graph = report.get('rotation_graph', {})
    graph_text = ''
    if graph:
        graph_text = (f"<p>抽样图共有 {graph['components']} 个连通分量；"
                      f"{graph['triangles']} 个三图旋转闭环中，"
                      f"{graph['triangles_above_review_threshold']} 个误差超过诊断阈值 5°。"
                      '<a href="rotation_cycles.json">查看闭环统计</a>。'
                      '闭环异常不能单独确定哪条边出错；5° 不是 3DGS 验收标准。</p>')
    body = f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>{title}</title>
<style>body{{font:15px system-ui;background:#16191e;color:#ddd;margin:24px}}a{{color:#80bdff}}h1{{font-size:24px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(560px,1fr));gap:16px}}article{{background:#242933;padding:12px}}img{{width:100%}}select{{padding:7px}}p{{line-height:1.6}}</style>
<h1>{title}</h1><p>{summary['images']} 张图 / {summary['pairs']} 对 · LoMa-B · 独立曝光 · 原生鱼眼几何 · pose 未参与拟合</p>
<p>绿色：几何通过；蓝色：留出且通过；红色：未通过。每对最多显示 100 条随机抽样匹配。
留出统计包含全部原始留出匹配，不先按几何误差筛除。两视图诊断尚不能证明多视图或 3DGS 精度。</p>
<p>本轮语义掩码：{'已提供' if report['semantic_masks_available'] else '未提供；仅使用硬 FOV，天空和动态内容仍需进一步处理'}。
<a href="report.json">完整统计</a> · <a href="principal_conventions.json">主点约定对照</a> · <a href="prior_conventions.json">pose 约定对照</a></p>
<label>图像对类型 <select id="filter"><option value="">全部</option><option>temporal</option><option>cross_camera</option><option>loop_candidate</option></select></label>
<p>Raw matches: {summary['raw_matches']}; pairwise verified: {summary['verified_matches']}; length ≥ 3 candidate tracks: {report['tracks']['tracks_length_ge3']}.</p>{graph_text}
<div class="grid">{''.join(cards)}</div><script>document.getElementById('filter').onchange=function(){{document.querySelectorAll('article').forEach(e=>e.hidden=!!this.value&&!e.dataset.kind.split(' ').includes(this.value))}};</script></html>'''
    (out/'index.html').write_text(body)


def diagnose(work, config, frames=DEFAULT_FRAMES, max_error=3.0, seed=42):
    started = time.monotonic()
    work = Path(work).resolve()
    scene = load_scene(work)
    frames = sorted(set(frames))
    images = [im for im in scene['images'] if im['frame_index'] in frames]
    present = {(im['camera'],im['frame_index']) for im in images}
    missing = [(c,f) for c in ('L','R') for f in frames if (c,f) not in present]
    if missing:
        raise ValueError(f'Probe requested excluded/missing frames: {missing}')
    pairs = make_pairs(images)
    fingerprint = digest({'scene':scene['fingerprint'], 'frontend':asdict(config), 'frames':frames,
                          'max_error':max_error,'seed':seed,'diagnostic_version':DIAGNOSTIC_VERSION})
    out = work/'diagnostics'/fingerprint[:16]
    for folder in ['pairs','previews']:
        (out/folder).mkdir(parents=True,exist_ok=True)
    frontend = LoMaFrontend(work,scene,config)
    features = {}
    for i,im in enumerate(images,1):
        features[im['name']] = frontend.extract(im)
        print(f'features {i}/{len(images)} {label(im)}: {len(features[im["name"]]["keypoints"])}',flush=True)
    cameras = {k:Camera.from_dict(v) for k,v in scene['cameras'].items()}
    rows, pair_results, track_edges = [], [], []
    for i,(a,b,tags) in enumerate(pairs,1):
        fa,fb = features[a['name']],features[b['name']]
        raw = frontend.match(fa,fb)
        xy_a = fa['keypoints'][raw['matches'][:,0]].astype(float)
        xy_b = fb['keypoints'][raw['matches'][:,1]].astype(float)
        metrics,arrays = fit_pair(cameras[a['camera']],cameras[b['camera']],xy_a,xy_b,max_error,seed,fov=scene['fov_degrees'])
        artifact = label(a)+'__'+label(b)
        row = {**metrics, 'a':a['name'],'b':b['name'],'label':artifact,'artifact':artifact,'tags':tags,
               'feature_tokens':[str(fa['token']),str(fb['token'])], 'match_token':str(raw['token'])}
        write_json(out/'pairs'/f'{artifact}.json',row)
        np.savez_compressed(out/'pairs'/f'{artifact}.npz',matches=raw['matches'],scores=raw['scores'],**arrays)
        preview(out/'previews'/f'{artifact}.jpg',a,b,xy_a,xy_b,arrays,max_error)
        rows.append(row)
        pair_results.append((a,b,raw,xy_a,xy_b,metrics,arrays))
        if metrics['success']:
            take = arrays['verified_mask']
            track_edges.append((a['name'],b['name'],raw['matches'][take],raw['scores'][take]))
        print(f'pairs {i}/{len(pairs)} {artifact}: {len(xy_a)} raw / {metrics["verified"]} verified',flush=True)
    track_summary,tracks = build_tracks(track_edges,images)
    write_json(out/'tracks.json',{'summary':track_summary,'tracks':tracks})
    graph = rotation_graph_diagnostics([
        (a['name'], b['name'], arrays['pose_cam_b_from_a'][:, :3], label(a)+'__'+label(b))
        for a,b,raw,xy_a,xy_b,metrics,arrays in pair_results if metrics['success']
    ], [im['name'] for im in images])
    write_json(out/'rotation_cycles.json', graph)
    by_label = {p['label']: p for p in graph['pairs']}
    for row in rows:
        if row['label'] in by_label:
            row['rotation_cycles'] = by_label[row['label']]
            write_json(out/'pairs'/f'{row["artifact"]}.json', row)
    write_json(out/'prior_conventions.json',summarize_prior_conventions(pair_results,scene))
    print('Comparing principal-point sign conventions on held-out pair matches...',flush=True)
    write_json(out/'principal_conventions.json',compare_principal_conventions(pair_results,scene,max_error,seed))
    report = {'fingerprint':fingerprint,'scene_fingerprint':scene['fingerprint'],'frames':frames,
              'pose_model':scene['pose_model'],'pose_used_for_fitting':False,'fov_degrees':scene['fov_degrees'],
              'semantic_masks_available':scene['semantic_masks_available'],'frontend':asdict(config),
              'frontend_cache': dict(frontend.cache_stats),
              'timing_note': 'Invocation wall time, affected by cache reuse; not an inference benchmark.',
              'versions':{name:importlib.metadata.version(name) for name in ['numpy','pycolmap','torch','lomatch']},
              'summary':{'images':len(images),'pairs':len(pairs), 'successful_pairs':sum(r['success'] for r in rows),
                         'raw_matches':sum(r['raw_matches'] for r in rows),'verified_matches':sum(r['verified'] for r in rows),
                         'elapsed_seconds':time.monotonic()-started},'tracks':track_summary,'pairs':rows,
              'rotation_graph': graph['summary'],
              'scope':'Frontend probe only. No production reconstruction, LAS alignment or 3DGS training.',
              'status':'diagnostic only; principal convention and static-scene masks still require validation'}
    write_json(out/'report.json',report)
    with (out/'pairs.csv').open('w',newline='') as f:
        columns=['label','raw_matches','train_count','holdout_count','verified','success']
        writer=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    report_html(out,report)
    write_json(work/'latest_diagnostic.json',{'path':str(out),'fingerprint':fingerprint})
    return out,report
