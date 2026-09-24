"""Selected native-image crops for full-scene heldout geometry diagnosis."""
import html
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .residual_review import case_images
from .scene import digest, file_digest, load_scene, write_json


def build_global_residual_review(work, candidate='triangle_supported'):
    root = Path(work).resolve() / 'global_sfm'
    fixed = root / 'stages/fixed_calibration/triangle_validation/validation_observations.npz'
    later = root / 'stages' / candidate / 'triangle_validation/validation_observations.npz'
    if candidate not in ('triangle_supported', 'shared_calibration') or not later.exists():
        raise ValueError('Candidate must be a completed saved global stage')
    with np.load(fixed) as data:
        baseline = {key: data[key] for key in data.files}
    with np.load(later) as data:
        refined = {key: data[key] for key in data.files}
    for key in ('track_id', 'name', 'feature_index', 'camera', 'zone'):
        np.testing.assert_array_equal(baseline[key], refined[key])
    scene = load_scene(root)
    catalog = json.loads((root / 'frontend.json').read_text())
    records = {record['name']: record for record in scene['images']}
    tracks = {str(track['id']): track for track in
              json.loads((root / 'triangle_validation_split.json').read_text())['tracks']}
    valid = (baseline['support_ok'] & refined['support_ok']
             & np.isfinite(baseline['error']) & np.isfinite(refined['error']))
    out = root / 'residual_review' / candidate
    folder = out / 'cases'; folder.mkdir(parents=True, exist_ok=True)
    selected = []; used = set()
    for camera in ('L', 'R'):
        for zone in ('center', 'middle', 'edge'):
            mask = valid & (refined['camera'] == camera) & (refined['zone'] == zone)
            for band in ('large_error', 'worsened', 'control_le_2px'):
                ids = [int(i) for i in np.flatnonzero(mask)
                       if str(refined['track_id'][i]) not in used]
                if band == 'large_error':
                    ids = [i for i in ids if refined['error'][i] >= 10]
                    key = lambda i: (-min(refined['error'][i], 300), str(refined['track_id'][i]))
                elif band == 'worsened':
                    ids = [i for i in ids if refined['error'][i] - baseline['error'][i] >= 1]
                    key = lambda i: (-(refined['error'][i] - baseline['error'][i]),
                                     str(refined['track_id'][i]))
                else:
                    ids = [i for i in ids if refined['error'][i] <= 2]
                    key = lambda i: digest([str(refined['track_id'][i]),
                                            str(refined['name'][i]), int(refined['feature_index'][i])])
                if ids:
                    idx = min(ids, key=key)
                    used.add(str(refined['track_id'][idx])); selected.append((idx, band))
    cases = []; names = set()
    for idx, band in selected:
        track_id = str(refined['track_id'][idx]); track = tracks[track_id]
        case = {'id': f'case_{len(cases)+1:03d}', 'segment': 'all_536',
                'protocol': 'direct_triangle', 'band': band, 'track_id': track_id,
                'name': str(refined['name'][idx]),
                'feature_index': int(refined['feature_index'][idx]),
                'camera': str(refined['camera'][idx]), 'zone': str(refined['zone'][idx]),
                'error_px': float(refined['error'][idx]),
                'fixed_error_px': float(baseline['error'][idx]),
                'support_error_px': float(refined['support_error'][idx]),
                'support_angle_deg': float(refined['support_angle'][idx]),
                'support': track['support'],
                'observed': refined['observed'][idx].tolist(),
                'predicted': refined['predicted'][idx].tolist(),
                'fixed_predicted': baseline['predicted'][idx].tolist()}
        cases.append(case)
        names.update([case['name'], *(n for n, _ in case['support'])])
    features = {}
    for name in names:
        row = catalog['features'][name]
        with np.load(row['path'], allow_pickle=False) as feature:
            if str(feature['token']) != row['token']:
                raise ValueError(f'Feature token changed: {name}')
            features[name] = {'keypoints': feature['keypoints']}
    write_json(out / 'cases.json', {
        'source_stage': 'fixed_calibration', 'candidate_stage': candidate,
        'selection': 'Diagnostic camera/radius strata with large residual, strongest worsening and <=2px control. Only finite common-support high-confidence queries; one query per raw track. Not an accuracy-rate sample.',
        'source_sha256': {'fixed_validation': file_digest(fixed),
                          'candidate_validation': file_digest(later),
                          'triangle_split': file_digest(root / 'triangle_validation_split.json')},
        'cases': cases, 'units': 'native 3000x4000 pixels; zoom uses nearest-neighbor 2x display'})
    tiles = [case_images(case, records, features, folder) for case in cases]
    sheets = []
    for start in range(0, len(tiles), 4):
        chunk = tiles[start:start+4]
        sheet = Image.new('RGB', (chunk[0].width, sum(tile.height for tile in chunk)))
        y = 0
        for tile in chunk:
            sheet.paste(tile, (0, y)); y += tile.height
        name = f'sheet_{start//4+1:02d}.jpg'; sheet.save(out / name, quality=94)
        sheets.append(name)
    cards = []
    for case in cases:
        label = f'{case["camera"]}/{case["zone"]} · {case["band"]}'
        cards.append(f'<details id="{case["id"]}"><summary>{case["id"]} · {html.escape(label)} · '
                     f'{case["fixed_error_px"]:.2f} → {case["error_px"]:.2f} px</summary>'
                     f'<p>固定内参基线预测为黄色；{html.escape(candidate)} 预测为红色；LoMa 观测为绿色。'
                     f'<a href="cases/{case["id"]}_plain.jpg">无标记原图局部</a> · '
                     f'<a href="cases/{case["id"]}.jpg">全图位置与局部</a></p>'
                     f'<img loading="lazy" src="cases/{case["id"]}.jpg"></details>')
    (out / 'index.html').write_text('''<!doctype html><html lang="zh"><meta charset="utf-8"><title>全场原图残差复核</title>
<style>body{font:16px system-ui;background:#171b23;color:#ddd;margin:24px}a{color:#8bc5ff}p{line-height:1.7}img{max-width:100%}details{border:1px solid #576070;margin:12px 0;padding:10px}summary{cursor:pointer}</style>
<h1>全场高置信留出对应的原图局部</h1><p>每例左、中为两个留出支持观测，右为查询。样例刻意覆盖大误差、相对基线变差和 ≤2 px 对照；不是随机样本，也不能估计总体错误率。三图直接 LoMa 匹配均 ≥0.9，仍可能对应视角相关轮廓或重复纹理。图片只用于人工判断误差来源，不改动几何拟合。</p>
<p><a href="cases.json">样例与输入哈希</a> · <a href="../../index.html">全场结果</a></p>'''
        + ''.join(cards) + '</html>')
    print(f'Global residual review: {len(cases)} cases, {len(sheets)} sheets in {out}', flush=True)
    return out
