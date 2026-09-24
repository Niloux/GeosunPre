"""Native JPEG appearance differences on geometry-verified LoMa correspondences."""
from collections import defaultdict
import html
import json
from pathlib import Path

import cv2
import numpy as np
import pycolmap

from .scene import file_digest, load_scene, write_json


def sampled_keypoint_colors(record, feature, directory):
    directory = Path(directory)
    token = str(feature['token'])
    path = directory / f'{token}.npz'
    if path.exists():
        with np.load(path, allow_pickle=False) as old:
            if str(old['image_sha256']) != file_digest(record['source']):
                raise ValueError(f'Image changed since photometric sampling: {record["name"]}')
            return old['rgb']
    image = cv2.imread(record['source'], cv2.IMREAD_COLOR)
    if image is None: raise ValueError(record['source'])
    if image.shape[:2] != (4000, 3000): raise ValueError(f'Unexpected native size: {record["name"]}')
    smooth = cv2.blur(image, (5, 5))
    xy = feature['keypoints']
    ij = np.floor(xy).astype(int)
    ij[:, 0] = ij[:, 0].clip(0, image.shape[1]-1)
    ij[:, 1] = ij[:, 1].clip(0, image.shape[0]-1)
    rgb = smooth[ij[:, 1], ij[:, 0], ::-1].copy()
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, rgb=rgb, image_sha256=np.array(file_digest(record['source'])),
                        feature_token=np.array(token))
    return rgb


def inspect_photometric_pairs(work):
    work = Path(work).resolve(); source = work / 'global_sfm'; out = source / 'photometric_review'
    scene = load_scene(source)
    catalog = json.loads((source / 'frontend.json').read_text())
    records = {r['name']: r for r in scene['images']}
    names = {r['image_id']: r['name'] for r in scene['images']}
    colors = {}
    for i, record in enumerate(scene['images'], 1):
        with np.load(catalog['features'][record['name']]['path'], allow_pickle=False) as feature:
            if str(feature['token']) != catalog['features'][record['name']]['token']:
                raise ValueError(f'Feature token changed: {record["name"]}')
            colors[record['name']] = sampled_keypoint_colors(record, feature, out / 'sampled_colors')
        if i % 50 == 0 or i == len(scene['images']):
            print(f'photometric keypoint samples {i}/{len(scene["images"])}', flush=True)
    by_pair = {(a['a'], a['b']): a for a in catalog['pairs']}
    groups = defaultdict(list); rows = []
    with pycolmap.Database.open(source / 'database.db') as db:
        pair_ids, matches = db.read_all_matches()
        for idx, (pair_id, m) in enumerate(zip(pair_ids, matches), 1):
            ia, ib = pycolmap.pair_id_to_image_pair(pair_id)
            a, b = names[ia], names[ib]
            entry = by_pair.get((a, b))
            if entry is None:
                raise ValueError(f'Database pair absent from frozen frontend: {a}, {b}')
            with np.load(entry['path'], allow_pickle=False) as raw:
                keys = (raw['matches'][:, 0].astype(np.uint64) << 32) | raw['matches'][:, 1]
                order = np.argsort(keys)
                sorted_keys = keys[order]
                query = (m[:, 0].astype(np.uint64) << 32) | m[:, 1]
                locations = np.searchsorted(sorted_keys, query)
                located = locations < len(sorted_keys)
                located[located] &= sorted_keys[locations[located]] == query[located]
                if not located.all(): raise ValueError('Database match missing from raw match cache')
                score = raw['scores'][order[locations]]
            take = score >= .9
            if not take.any(): continue
            ca = colors[a][m[take, 0]].astype(np.float32) / 255
            cb = colors[b][m[take, 1]].astype(np.float32) / 255
            luma_a = ca @ np.array([.2126, .7152, .0722], dtype=np.float32)
            luma_b = cb @ np.array([.2126, .7152, .0722], dtype=np.float32)
            luminance = float(np.median(np.abs(luma_a - luma_b)))
            rgb = float(np.median(np.mean(np.abs(ca - cb), axis=1)))
            gap = abs(records[a]['frame_index'] - records[b]['frame_index'])
            kind = ('same_camera' if records[a]['camera'] == records[b]['camera'] else 'cross_camera')
            kind += '_near' if gap <= 2 else '_far'
            ea, eb = records[a]['exposure_seconds'], records[b]['exposure_seconds']
            exposure_ratio = max(ea / eb, eb / ea) if ea and eb else None
            row = {'a': a, 'b': b, 'group': kind, 'frame_gap': gap,
                   'verified_high_confidence_matches': int(take.sum()),
                   'median_abs_luma_difference_0to1': luminance,
                   'median_mean_abs_rgb_difference_0to1': rgb,
                   'exif_exposure_ratio': exposure_ratio}
            rows.append(row); groups[kind].append(row)
            if idx % 1000 == 0: print(f'photometric pair comparison {idx}/{len(matches)}', flush=True)
    report = {'images_sampled': len(colors), 'database_pairs': len(matches),
              'pairs_with_verified_high_confidence_matches': len(rows), 'groups': {},
              'units': 'Absolute sRGB JPEG channel values normalized by 255; 5x5 mean at original LoMa keypoints',
              'scope': 'Appearance diagnostic on geometry-verified training matches. Not static-point ground truth, linear radiance, exposure correction or 3DGS render validation.'}
    for kind, group in sorted(groups.items()):
        luma = np.array([r['median_abs_luma_difference_0to1'] for r in group])
        rgb = np.array([r['median_mean_abs_rgb_difference_0to1'] for r in group])
        ex = np.array([r['exif_exposure_ratio'] for r in group if r['exif_exposure_ratio'] is not None])
        report['groups'][kind] = {'pairs': len(group), 'matches': sum(r['verified_high_confidence_matches'] for r in group),
             'pair_median_abs_luma_difference': {'median': float(np.median(luma)), 'p90': float(np.quantile(luma, .9))},
             'pair_median_abs_rgb_difference': {'median': float(np.median(rgb)), 'p90': float(np.quantile(rgb, .9))},
             'exif_exposure_ratio': {'median': float(np.median(ex)), 'p90': float(np.quantile(ex, .9))} if len(ex) else None}
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / 'summary.json', report)
    write_json(out / 'pairs.json', rows)
    html_rows = []
    for name, result in report['groups'].items():
        a = result['pair_median_abs_luma_difference']; b = result['pair_median_abs_rgb_difference']
        exposure = result['exif_exposure_ratio']
        html_rows.append(f'<tr><td>{html.escape(name)}</td><td>{result["pairs"]}</td><td>{result["matches"]}</td>'
                         f'<td>{a["median"]:.3f} / {a["p90"]:.3f}</td><td>{b["median"]:.3f} / {b["p90"]:.3f}</td>'
                         f'<td>{exposure["median"]:.3f} / {exposure["p90"]:.3f}</td></tr>')
    (out / 'index.html').write_text('''<!doctype html><html lang="zh"><meta charset="utf-8"><title>原图外观一致性检查</title>
<style>body{font:16px system-ui;background:#171b23;color:#ddd;margin:24px}a{color:#8bc5ff}p{line-height:1.7}table{border-collapse:collapse}td,th{border:1px solid #576070;padding:10px}</style>
<h1>LoMa 对应点的原图外观差异</h1><p>只使用全场数据库中已通过鱼眼几何的对应，再要求原始 LoMa 匹配置信度 ≥0.9。从 3000×4000 原生 JPEG 上对每个点取 5×5 RGB 平均，计算匹配位置的亮度与颜色绝对差。每对图像先取匹配中位数，再统计图像对的分布；分数在 0–1 的 JPEG 像素尺度。EXIF 曝光比仅作元数据对照。</p>
<table><tr><th>图像对</th><th>对数</th><th>匹配数</th><th>每对亮度差 median / P90</th><th>每对 RGB 差 median / P90</th><th>EXIF 曝光比 median / P90</th></tr>'''+''.join(html_rows)+'''</table>
<p>近距离为帧号差 ≤2；远距离包括检索提出的候选。原始 JPEG 的 sRGB 差异不等于线性辐射误差；错误匹配、视角、遮挡、动态物体和局部定位均会影响结果。本页不据此自动校正曝光或认定 3DGS 的渲染精度。</p>
<p><a href="summary.json">完整统计</a> · <a href="pairs.json">逐对结果</a> · <a href="../index.html">全场 SfM 总览</a></p></html>''')
    return report
