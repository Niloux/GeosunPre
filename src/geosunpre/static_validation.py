"""Geometry-blind candidate sampling and manual static-correspondence review.

Selection never opens a reconstruction or any validation residual. Candidates
come only from the frozen, pre-geometry LoMa holdout and original keypoints.
"""
from collections import Counter, defaultdict
from functools import lru_cache
from hashlib import sha256
import html
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .scene import digest, file_digest, load_scene, write_json


TIME_BINS = ((33, 99), (100, 166), (167, 233), (234, 300))
ZONES = ('center', 'middle', 'edge')
CAMERAS = ('L', 'R')
PER_CELL = 10
SEED = 'static-review-2026-09-23-v1'
OUTPUT = 'static_validation'


def screen_zone(xy, width, height):
    """Stratify screen position without using any uncertain intrinsics."""
    x, y = map(float, xy)
    rho = float(np.hypot((x - width / 2) / (width / 2),
                         (y - height / 2) / (height / 2)))
    return 'center' if rho < .5 else 'middle' if rho < .8 else 'edge'


def time_bin(frame):
    for lo, hi in TIME_BINS:
        if lo <= frame <= hi:
            return f'{lo:03d}-{hi:03d}'
    raise ValueError(f'Frame outside frozen basketball range: {frame}')


def rank_candidate(row):
    data = f'{SEED}|{row["track_id"]}|{row["query"][0]}|{row["query"][1]}'
    return sha256(data.encode()).hexdigest()


def choose_candidates(pool, per_cell=PER_CELL):
    """Rare cells first; balance near/far when available; one raw track once."""
    by_cell = defaultdict(list)
    for row in pool:
        by_cell[(row['camera'], row['time_bin'], row['zone'])].append(row)
    cells = [(camera, f'{lo:03d}-{hi:03d}', zone)
             for camera in CAMERAS for lo, hi in TIME_BINS for zone in ZONES]
    if any(cell not in by_cell for cell in cells):
        raise ValueError(f'Empty frozen sampling cells: {[c for c in cells if c not in by_cell]}')
    cells.sort(key=lambda cell: (len({r['track_id'] for r in by_cell[cell]}), cell))
    used_tracks = set(); selected = []
    for cell in cells:
        available = sorted(by_cell[cell], key=rank_candidate)
        chosen = []
        for gap in ('near', 'far'):
            for row in available:
                if len([r for r in chosen if r['gap_kind'] == gap]) >= per_cell // 2:
                    break
                if row['gap_kind'] != gap or row['track_id'] in used_tracks:
                    continue
                chosen.append(row); used_tracks.add(row['track_id'])
        for row in available:
            if len(chosen) >= per_cell:
                break
            if row['track_id'] not in used_tracks:
                chosen.append(row); used_tracks.add(row['track_id'])
        if len(chosen) != per_cell:
            raise ValueError(f'Only {len(chosen)}/{per_cell} independent tracks in {cell}')
        selected.extend(chosen)
    selected.sort(key=lambda row: (row['camera'], row['time_bin'],
                                   ZONES.index(row['zone']), rank_candidate(row)))
    return selected


def build_candidate_pool(scene, frontend, heldout):
    records = {record['name']: record for record in scene['images']}
    used_names = {name for track in heldout['tracks']
                  for name, _ in track['support'] + track['queries']}
    keypoints = {}
    for name in sorted(used_names):
        row = frontend['features'][name]
        record = records[name]
        expected = digest({'image_sha256': file_digest(record['source']),
                           'mask_sha256': file_digest(record['semantic_mask']),
                           'camera': scene['cameras'][record['camera']],
                           'fov': scene['fov_degrees'], 'config': frontend['config']})
        if row['token'] != expected:
            raise ValueError(f'Image, mask, calibration or LoMa config changed: {name}')
        with np.load(row['path'], allow_pickle=False) as data:
            if str(data['token']) != row['token']:
                raise ValueError(f'Changed LoMa feature cache: {name}')
            keypoints[name] = data['keypoints']
    pool = []
    for track in heldout['tracks']:
        support = track['support']
        for name, index in track['queries']:
            record = records[name]
            if record['camera'] not in CAMERAS:
                raise ValueError(f'Unexpected camera in heldout query: {name}')
            query_xy = keypoints[name][index].astype(float).tolist()
            camera = scene['cameras'][record['camera']]
            zone = screen_zone(query_xy, camera['width'], camera['height'])
            frame = int(record['frame_index'])
            gap = max(abs(frame - records[n]['frame_index']) for n, _ in support)
            row = {'track_id': track['id'], 'support': support, 'query': [name, int(index)],
                   'camera': record['camera'], 'frame_index': frame,
                   'time_bin': time_bin(frame), 'zone': zone,
                   'gap_kind': 'near' if gap <= 10 else 'far',
                   'support_frame_gap_max': gap,
                   'observations': [{'name': n, 'feature_index': int(i),
                                     'xy': keypoints[n][i].astype(float).tolist(),
                                     'role': 'support' if j < 2 else 'query'}
                                    for j, (n, i) in enumerate(support + [[name, index]])]}
            row['id'] = digest([row['track_id'], row['query']])[:16]
            pool.append(row)
    return pool


def freeze_candidates(work):
    root = Path(work).resolve() / 'global_sfm'
    out = root / OUTPUT
    scene = load_scene(root)
    frontend = json.loads((root / 'frontend.json').read_text())
    heldout = json.loads((root / 'triangle_validation_split.json').read_text())
    if not (heldout['selection_frozen_before_geometry']
            and not heldout['selection_uses_pose_or_reprojection_error']):
        raise ValueError('Input holdout was not frozen before geometry')
    protocol = {
        'version': 1,
        'scene_fingerprint': scene['fingerprint'],
        'frontend_fingerprint': frontend['fingerprint'],
        'holdout_sha256': file_digest(root / 'triangle_validation_split.json'),
        'selection': {'camera': list(CAMERAS), 'time_bins': [list(x) for x in TIME_BINS],
                      'zone': 'Normalized screen radius from image center: center <0.5, middle <0.8, edge >=0.8',
                      'near_support_gap_frames': 10, 'per_camera_time_zone': PER_CELL,
                      'near_far_target_each': PER_CELL // 2,
                      'raw_track_used_at_most_once': True, 'rare_cells_first': True,
                      'within_cell_rank': 'SHA256(seed|raw_track_id|query_name|query_feature_index)',
                      'seed': SEED},
        'selection_uses_geometry_or_validation_errors': False,
        'candidate_source': 'Frozen direct three-view LoMa holdout, confidence >=0.9; query coordinates from original feature cache',
        'scope': 'Geometry-blind review candidates from one capture, not manually verified ground truth or an independent capture.'}
    protocol['fingerprint'] = digest(protocol)
    out.mkdir(parents=True, exist_ok=True)
    path = out / 'protocol.json'
    if path.exists():
        if json.loads(path.read_text()) != protocol:
            raise ValueError('Frozen manual-review selection protocol changed')
    else:
        write_json(path, protocol)
    pool = build_candidate_pool(scene, frontend, heldout)
    selected = choose_candidates(pool)
    candidates = {'protocol_fingerprint': protocol['fingerprint'],
                  'rows': selected, 'fingerprint': digest(selected)}
    path = out / 'candidates.json'
    if path.exists():
        if json.loads(path.read_text()) != candidates:
            raise ValueError('Frozen manual-review candidate list changed')
    else:
        write_json(path, candidates)
    all_obs = {tuple(obs) for track in heldout['tracks'] for obs in track['observations']}
    if any((obs['name'], obs['feature_index']) not in all_obs
           for row in selected for obs in row['observations']):
        raise RuntimeError('Selected query/support was not wholly withheld')
    cell_counts = Counter((r['camera'], r['time_bin'], r['zone']) for r in selected)
    audit = {'selected': len(selected), 'available_queries': len(pool),
             'unique_raw_tracks': len({r['track_id'] for r in selected}),
             'unique_query_images': len({r['query'][0] for r in selected}),
             'cell_counts': {'/'.join(cell): cell_counts[cell] for cell in sorted(cell_counts)},
             'near': sum(r['gap_kind'] == 'near' for r in selected),
             'far': sum(r['gap_kind'] == 'far' for r in selected),
             'all_observations_in_pre_geometry_holdout': True,
             'no_current_pose_or_error_used_for_selection': True,
             'manual_labels_completed': 0,
             'note': 'These are candidates only. No point has been verified as the same visible static entity.'}
    submitted = out / 'basketball-static-annotations.json'
    if submitted.exists():
        accepted, annotation_counts = checked_annotations(
            json.loads(submitted.read_text()), candidates, scene)
        audit['manual_labels_completed'] = len(selected) - annotation_counts['unreviewed']
        audit['eligible_same_static'] = len(accepted)
        audit['submitted_annotations_sha256'] = file_digest(submitted)
        audit['note'] = ('Candidate selection was geometry-blind; submitted labels are stored '
                         'separately and evaluated without changing the SfM training data.')
    write_json(out / 'audit.json', audit)
    return out, scene, candidates, audit


@lru_cache(maxsize=8)
def source_image(path):
    with Image.open(path) as image:
        return image.convert('RGB')


def render_crops(out, scene, candidates):
    records = {r['name']: r for r in scene['images']}
    crops = out / 'crops'; crops.mkdir(parents=True, exist_ok=True)
    contexts = out / 'contexts'; contexts.mkdir(parents=True, exist_ok=True)
    used = {}
    for row in candidates['rows']:
        for obs in row['observations']:
            name = obs['name']; record = records[name]
            if name in used:
                continue
            image = source_image(record['source'])
            thumb = image.resize((150, 200))
            # Contexts stay unmarked; the selected seed belongs to one case,
            # while the same image can appear in several review cards.
            context_name = f'{record["image_id"]:04d}.jpg'
            thumb.save(contexts / context_name, quality=86)
            used[name] = f'contexts/{context_name}'
    for pos, row in enumerate(candidates['rows'], 1):
        for slot, obs in enumerate(row['observations']):
            image = source_image(records[obs['name']]['source'])
            x, y = obs['xy']
            left, top = int(np.floor(x - 96)), int(np.floor(y - 96))
            crop = image.crop((left, top, left + 192, top + 192))
            crop = crop.resize((384, 384), Image.Resampling.NEAREST)
            crop_name = f'{row["id"]}_{slot}.jpg'
            crop.save(crops / crop_name, quality=94)
            obs['crop'] = f'crops/{crop_name}'
            obs['context'] = used[obs['name']]
            obs['crop_origin'] = [left, top]
        if pos % 40 == 0 or pos == len(candidates['rows']):
            print(f'Static review crops {pos}/{len(candidates["rows"])}', flush=True)
    rendered = {'protocol_fingerprint': candidates['protocol_fingerprint'],
                'candidate_fingerprint': candidates['fingerprint'],
                'rows': candidates['rows']}
    write_json(out / 'rendered_candidates.json', rendered)
    return rendered


def write_review_ui(out, rendered):
    rows = rendered['rows']
    template = {
        'schema_version': 1,
        'protocol_fingerprint': rendered['protocol_fingerprint'],
        'candidate_fingerprint': rendered['candidate_fingerprint'],
        'annotations': {
            row['id']: {'label': 'unreviewed', 'scene_type': None, 'note': '',
                        'coords': [obs['xy'] for obs in row['observations']],
                        'verified': [False, False, False]}
            for row in rows}}
    write_json(out / 'annotations.template.json', template)
    embedded = json.dumps(rows, ensure_ascii=False, separators=(',', ':')).replace('</', '<\\/')
    page = r'''<!doctype html><html lang="zh"><meta charset="utf-8">
<title>篮球场静态点盲审</title>
<style>
body{font:16px system-ui;background:#171b23;color:#e4e7ed;margin:18px;line-height:1.5}
a{color:#8bc5ff}button,select,input,textarea{font:inherit;background:#293341;color:#eee;border:1px solid #647184;border-radius:4px;padding:6px}
button{cursor:pointer;margin:3px}button.active{background:#18684e;border-color:#80e4b1}
.bar{display:flex;gap:12px;flex-wrap:wrap;align-items:center;margin:12px 0}
.panels{display:grid;grid-template-columns:repeat(3,minmax(300px,1fr));gap:10px}
.panel{border:1px solid #526070;padding:8px;background:#222a35}
.context{position:relative;width:150px;height:200px;display:inline-block;vertical-align:top;margin-right:8px}
.context img{width:150px;height:200px}.context .seed,.crop .seed,.crop .adjusted{position:absolute;transform:translate(-50%,-50%);pointer-events:none}
.seed{width:11px;height:11px;border:2px solid #50ee9b;border-radius:50%;box-sizing:border-box}
.adjusted{width:10px;height:10px;border:2px solid #ff716f;transform:translate(-50%,-50%) rotate(45deg)!important}
.crop{position:relative;width:min(100%,384px);aspect-ratio:1;cursor:crosshair;margin-top:8px}
.crop img{width:100%;height:100%;display:block;image-rendering:pixelated}
.panel small{display:block;word-break:break-all}.hint{color:#aebbd0}.status{font-weight:700}
textarea{width:min(100%,900px);min-height:58px}.danger{color:#ff9c91}
@media(max-width:1050px){.panels{grid-template-columns:1fr}}
</style>
<h1>篮球场：冻结留出轨迹的静态点盲审</h1>
<p>这 240 个候选按 L/R、四段时间、原图中心/中圈/外圈各取 10 个；只使用几何拟合前留出的 LoMa 对应和原图位置。<strong>页面不显示任何 SfM 预测或残差。</strong>三个绿色标记是 LoMa 原始点，红色标记是你核对或点击修正后的点。只有确认三张图中可见同一个静态实体且逐点核对，才标记“同一静态点”。重复纹理、遮挡边界和无法确证的树叶应选“不确定”或对应类别。</p>
<p class="hint">点击放大图可修正该图点位；也可点“原点确认”。浏览器本地保存进度。完成后下载 JSON 并用 <code>static-evaluate --annotations 文件</code> 作几何对照；场景类型可以留空，会单列为“未分类”。未人工核对的候选不会自动变成真值。当前页面只来自同一次采集，不能替代独立拍摄的验收。</p>
<div class="bar">
<label>镜头 <select id="camera"><option value="">全部</option><option>L</option><option>R</option></select></label>
<label>时间 <select id="time"><option value="">全部</option><option>033-099</option><option>100-166</option><option>167-233</option><option>234-300</option></select></label>
<label>区域 <select id="zone"><option value="">全部</option><option>center</option><option>middle</option><option>edge</option></select></label>
<label>标注 <select id="statusfilter"><option value="">全部</option><option value="unreviewed">未审</option><option value="same_static">同一静态点</option><option value="ambiguous">不确定</option><option value="wrong_match">错配</option><option value="occluded">遮挡</option><option value="dynamic">动态</option></select></label>
<button id="prev">← 上一条</button><button id="next">下一条 →</button>
<input id="jump" placeholder="输入候选 ID 前缀"><button id="jumpbutton">跳转</button>
<span id="position"></span>
</div>
<div id="meta"></div><div class="panels" id="panels"></div>
<div class="bar" id="labels"></div>
<div class="bar"><label>场景类型 <select id="scenetype"><option value="">未选</option><option value="court_line">球场标线</option><option value="court_surface">球场地面</option><option value="building">建筑</option><option value="hardscape">其他硬质结构</option><option value="vegetation">植被</option><option value="other">其他</option></select></label>
<span id="verification"></span></div>
<label>备注<br><textarea id="note" placeholder="例如：可辨识的裂纹、树枝摆动、两视图遮挡不同"></textarea></label>
<div class="bar"><button id="export">下载标注 JSON</button><label>导入既有标注 <input type="file" id="import" accept=".json,application/json"></label><span id="progress"></span></div>
<p><a href="protocol.json">冻结抽样规则</a> · <a href="audit.json">覆盖与隔离审计</a> · <a href="candidates.json">候选清单</a> · <a href="annotations.template.json">空白标注模板</a> · <a href="../index.html">全场 SfM</a></p>
<script>
const ROWS=__ROWS__;
const PROTOCOL=__PROTOCOL__;
const CANDIDATES=__CANDIDATES__;
const KEY='basketball-static-review-'+CANDIDATES;
const LABELS=[['same_static','1 同一静态点'],['ambiguous','2 不确定'],['wrong_match','3 错配'],['occluded','4 遮挡'],['dynamic','5 动态']];
let saved={};
try{saved=JSON.parse(localStorage.getItem(KEY)||'{}')}catch(e){saved={}}
let currentId=ROWS[0].id;
const byId=Object.fromEntries(ROWS.map(r=>[r.id,r]));
function esc(s){return String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
function original(row){return {label:'unreviewed',scene_type:null,note:'',coords:row.observations.map(o=>o.xy.slice()),verified:[false,false,false]}}
function state(row){if(!saved[row.id])saved[row.id]=original(row);return saved[row.id]}
function persist(){localStorage.setItem(KEY,JSON.stringify(saved));progress()}
function shown(){const c=document.getElementById('camera').value,t=document.getElementById('time').value,z=document.getElementById('zone').value,s=document.getElementById('statusfilter').value;
return ROWS.filter(r=>(!c||r.camera===c)&&(!t||r.time_bin===t)&&(!z||r.zone===z)&&(!s||(saved[r.id]?.label||'unreviewed')===s))}
function marker(x,y,klass){return '<span class="'+klass+'" style="left:'+x.toFixed(2)+'%;top:'+y.toFixed(2)+'%"></span>'}
function draw(){let list=shown();if(!list.length){document.getElementById('meta').textContent='当前筛选无候选';document.getElementById('panels').innerHTML='';return}
if(!list.some(r=>r.id===currentId))currentId=list[0].id;
let row=byId[currentId],a=state(row),at=list.findIndex(r=>r.id===currentId);
document.getElementById('position').textContent=(at+1)+' / '+list.length+'；ID '+row.id;
document.getElementById('meta').innerHTML='<p><strong>'+esc(row.camera)+' · '+esc(row.time_bin)+' · '+esc(row.zone)+'</strong>；查询帧 '+row.frame_index+'；支持帧距最大 '+row.support_frame_gap_max+'；'+esc(row.gap_kind)+'。原始轨迹 '+esc(row.track_id)+'</p>';
document.getElementById('panels').innerHTML=row.observations.map((o,j)=>{let seedX=(o.xy[0]-o.crop_origin[0])/192*100,seedY=(o.xy[1]-o.crop_origin[1])/192*100,adjX=(a.coords[j][0]-o.crop_origin[0])/192*100,adjY=(a.coords[j][1]-o.crop_origin[1])/192*100;
let cx=o.xy[0]/3000*100,cy=o.xy[1]/4000*100;
return '<div class="panel"><strong>'+(j<2?'支持 '+(j+1):'查询')+'</strong><small>'+esc(o.name)+' · feature '+o.feature_index+'</small><div class="context"><img src="'+esc(o.context)+'">'+marker(cx,cy,'seed')+'</div><div class="crop" data-slot="'+j+'"><img src="'+esc(o.crop)+'">'+marker(seedX,seedY,'seed')+(a.verified[j]?marker(adjX,adjY,'adjusted'):'')+'</div><small>原始 ('+o.xy.map(v=>v.toFixed(2)).join(', ')+')；当前 ('+a.coords[j].map(v=>v.toFixed(2)).join(', ')+')</small><button data-verify="'+j+'">确认原点</button><button data-reset="'+j+'">清除核对</button></div>'}).join('');
document.querySelectorAll('.crop').forEach(el=>el.onclick=e=>{let j=Number(el.dataset.slot),rect=el.getBoundingClientRect(),o=row.observations[j];a.coords[j]=[o.crop_origin[0]+192*(e.clientX-rect.left)/rect.width,o.crop_origin[1]+192*(e.clientY-rect.top)/rect.height];a.verified[j]=true;persist();draw()});
document.querySelectorAll('[data-verify]').forEach(el=>el.onclick=()=>{let j=Number(el.dataset.verify);a.coords[j]=row.observations[j].xy.slice();a.verified[j]=true;persist();draw()});
document.querySelectorAll('[data-reset]').forEach(el=>el.onclick=()=>{let j=Number(el.dataset.reset);a.coords[j]=row.observations[j].xy.slice();a.verified[j]=false;persist();draw()});
document.getElementById('labels').innerHTML=LABELS.map(([id,label])=>'<button data-label="'+id+'" class="'+(a.label===id?'active':'')+'">'+label+'</button>').join('')+'<button data-label="unreviewed">清除标签</button>';
document.querySelectorAll('[data-label]').forEach(el=>el.onclick=()=>{a.label=el.dataset.label;persist();draw()});
document.getElementById('scenetype').value=a.scene_type||'';
document.getElementById('note').value=a.note||'';
document.getElementById('verification').innerHTML='逐点确认 '+a.verified.filter(Boolean).length+'/3'+(a.label==='same_static'&&!a.verified.every(Boolean)?' <span class="danger">未完成点位核对，不会进入几何验收</span>':'');
progress()}
function move(d){let list=shown();if(!list.length)return;let i=list.findIndex(r=>r.id===currentId);currentId=list[Math.max(0,Math.min(list.length-1,i+d))].id;draw();window.scrollTo(0,0)}
function progress(){let labeled=ROWS.filter(r=>saved[r.id]&&saved[r.id].label!=='unreviewed').length;
let approved=ROWS.filter(r=>saved[r.id]?.label==='same_static'&&saved[r.id].verified?.every(Boolean)).length;
document.getElementById('progress').textContent='已标注 '+labeled+'/'+ROWS.length+'；三点确认的同一静态点 '+approved}
for(let id of ['camera','time','zone','statusfilter'])document.getElementById(id).onchange=draw;
document.getElementById('prev').onclick=()=>move(-1);document.getElementById('next').onclick=()=>move(1);
document.getElementById('jumpbutton').onclick=()=>{let p=document.getElementById('jump').value.trim().toLowerCase(),r=ROWS.find(x=>x.id.startsWith(p));if(!p||!r){alert('没有此 ID 前缀');return}for(let id of ['camera','time','zone','statusfilter'])document.getElementById(id).value='';currentId=r.id;draw()};
document.getElementById('scenetype').onchange=e=>{state(byId[currentId]).scene_type=e.target.value||null;persist()};
document.getElementById('note').oninput=e=>{state(byId[currentId]).note=e.target.value;persist()};
document.getElementById('export').onclick=()=>{let data={schema_version:1,protocol_fingerprint:PROTOCOL,candidate_fingerprint:CANDIDATES,exported_at_utc:new Date().toISOString(),annotations:saved};let blob=new Blob([JSON.stringify(data,null,2)],{type:'application/json'});let link=document.createElement('a');link.href=URL.createObjectURL(blob);link.download='basketball-static-annotations.json';link.click();setTimeout(()=>URL.revokeObjectURL(link.href),1000)};
document.getElementById('import').onchange=async e=>{let file=e.target.files[0];if(!file)return;try{let data=JSON.parse(await file.text());if(data.protocol_fingerprint!==PROTOCOL||data.candidate_fingerprint!==CANDIDATES)throw Error('候选或抽样规则指纹不匹配');if(Object.keys(data.annotations||{}).some(id=>!byId[id]))throw Error('含有未知候选 ID');saved=data.annotations;persist();draw()}catch(err){alert('导入失败：'+err.message)}};
document.onkeydown=e=>{if(['TEXTAREA','INPUT','SELECT'].includes(document.activeElement.tagName))return;if(e.key==='ArrowRight')move(1);if(e.key==='ArrowLeft')move(-1);let i=Number(e.key)-1;if(i>=0&&i<LABELS.length){state(byId[currentId]).label=LABELS[i][0];persist();draw()}};
draw();
</script></html>'''
    page = page.replace('__ROWS__', embedded)
    page = page.replace('__PROTOCOL__', json.dumps(rendered['protocol_fingerprint']))
    page = page.replace('__CANDIDATES__', json.dumps(rendered['candidate_fingerprint']))
    (out / 'index.html').write_text(page)
    return out / 'index.html'


def prepare_static_review(work):
    out, scene, candidates, audit = freeze_candidates(work)
    model = Path(work).resolve() / 'global_sfm/stages/shared_calibration/sparse/0'
    if model.exists():
        import pycolmap
        from .local_reconstruction import assert_holdout_isolated
        rec = pycolmap.Reconstruction(model)
        split = json.loads((Path(work).resolve() / 'global_sfm/split.json').read_text())
        assert_holdout_isolated(rec, split)
        audit['saved_shared_model_holdout_isolated'] = True
        write_json(out / 'audit.json', audit)
    rendered_path = out / 'rendered_candidates.json'
    if rendered_path.exists():
        rendered = json.loads(rendered_path.read_text())
        if (rendered['protocol_fingerprint'] != candidates['protocol_fingerprint']
                or rendered['candidate_fingerprint'] != candidates['fingerprint']):
            raise ValueError('Existing review crops belong to another frozen selection')
        if not all((out / obs['crop']).exists()
                   for row in rendered['rows'] for obs in row['observations']):
            rendered = render_crops(out, scene, candidates)
    else:
        rendered = render_crops(out, scene, candidates)
    write_review_ui(out, rendered)
    print('Static blind-review candidates:', audit['selected'],
          'from', audit['unique_raw_tracks'], 'independent heldout tracks in', out, flush=True)
    return out


LABELS = frozenset(('unreviewed', 'same_static', 'ambiguous',
                    'wrong_match', 'occluded', 'dynamic'))
SCENE_TYPES = frozenset(('court_line', 'court_surface', 'building',
                         'hardscape', 'vegetation', 'other', 'unspecified'))


def checked_annotations(annotations, candidates, scene):
    """Accept only identity-matched, image-bounded, explicitly reviewed points."""
    if annotations.get('schema_version') != 1:
        raise ValueError('Unsupported manual annotation schema')
    if (annotations.get('protocol_fingerprint') != candidates['protocol_fingerprint']
            or annotations.get('candidate_fingerprint') != candidates['fingerprint']):
        raise ValueError('Manual annotations do not match the frozen candidate set')
    rows = {row['id']: row for row in candidates['rows']}
    records = {row['name']: row for row in scene['images']}
    raw = annotations.get('annotations')
    if not isinstance(raw, dict) or set(raw) - set(rows):
        raise ValueError('Annotations must be an object with only known candidate IDs')
    accepted = []
    counts = Counter()
    cells = defaultdict(Counter)
    for id_, row in rows.items():
        value = raw.get(id_, {'label': 'unreviewed'})
        label = value.get('label')
        if label not in LABELS:
            raise ValueError(f'Invalid label for {id_}: {label}')
        counts[label] += 1
        cell = '/'.join((row['camera'], row['time_bin'], row['zone']))
        cells[cell][label] += 1
        if label != 'same_static':
            continue
        verified = value.get('verified')
        if not isinstance(verified, list) or len(verified) != 3 or any(type(x) is not bool for x in verified):
            raise ValueError(f'Invalid verification flags for {id_}')
        if not all(verified):
            counts['incomplete_same_static'] += 1
            cells[cell]['incomplete_same_static'] += 1
            continue
        scene_type = value.get('scene_type') or 'unspecified'
        if scene_type not in SCENE_TYPES:
            raise ValueError(f'Invalid scene type for {id_}: {scene_type}')
        if scene_type == 'unspecified':
            counts['eligible_missing_scene_type'] += 1
            cells[cell]['eligible_missing_scene_type'] += 1
        coords = value.get('coords')
        if not isinstance(coords, list) or len(coords) != 3:
            raise ValueError(f'Expected three reviewed coordinates for {id_}')
        clean = []
        for obs, xy in zip(row['observations'], coords):
            if not isinstance(xy, list) or len(xy) != 2:
                raise ValueError(f'Invalid reviewed coordinates for {id_}')
            xy = np.asarray(xy, dtype=float)
            record = records[obs['name']]
            camera = scene['cameras'][record['camera']]
            if not (np.isfinite(xy).all() and 0 <= xy[0] < camera['width']
                    and 0 <= xy[1] < camera['height']):
                raise ValueError(f'Reviewed coordinate outside image for {id_}')
            clean.append(xy)
        accepted.append((row, scene_type, clean))
        cells[cell]['eligible_same_static'] += 1
    counts['eligible_same_static'] = len(accepted)
    counts['candidate_total'] = len(rows)
    result = dict(counts)
    result['by_cell'] = {key: dict(value) for key, value in sorted(cells.items())}
    result['reviewed_fraction'] = 1 - counts['unreviewed'] / len(rows)
    result['complete_24_cells'] = all(sum(n for label, n in value.items()
                                           if label in LABELS and label != 'unreviewed') == PER_CELL
                                      for value in cells.values()) and len(cells) == 24
    return accepted, result


def _evaluate_stage(rec, accepted, scene):
    from .camera import Camera
    from .local_evaluation import triangulate_support
    cameras = {cid: Camera(str(cid), c.width, c.height, tuple(c.params), (0, 0))
               for cid, c in rec.cameras.items()}
    images = {im.name: im for im in rec.images.values() if im.has_pose}
    records = {record['name']: record for record in scene['images']}
    output = []
    for row, scene_type, coords in accepted:
        names = [obs['name'] for obs in row['observations']]
        camera_pattern = ''.join(records[name]['camera'] for name in names)
        result = {'id': row['id'], 'camera': row['camera'],
                  'time_bin': row['time_bin'], 'zone': row['zone'],
                  'scene_type': scene_type, 'gap_kind': row['gap_kind'],
                  'camera_pattern': camera_pattern,
                  'camera_mix': 'cross_LR' if len(set(camera_pattern)) > 1 else 'single_camera',
                  'observed_xy': coords[2].tolist(), 'predicted_xy': None,
                  'query_error_px': None, 'support_max_error_px': None,
                  'support_angle_deg': None, 'support_ok': False,
                  'failure': None}
        if any(name not in images for name in names):
            result['failure'] = 'unregistered_image'
            output.append(result)
            continue
        ims = [images[name] for name in names]
        cs = [cameras[im.camera_id] for im in ims]
        ps = [im.cam_from_world().matrix() for im in ims]
        if not all(bool(cam.valid(xy[None], scene['fov_degrees'])[0])
                   for cam, xy in zip(cs, coords)):
            result['failure'] = 'outside_candidate_fov'
            output.append(result)
            continue
        try:
            point, support_error, angle = triangulate_support(cs[:2], ps[:2], coords[:2])
        except (ValueError, FloatingPointError, RuntimeError):
            result['failure'] = 'support_triangulation'
            output.append(result)
            continue
        result['support_max_error_px'] = float(support_error) if np.isfinite(support_error) else None
        result['support_angle_deg'] = float(angle) if np.isfinite(angle) else None
        result['support_ok'] = bool(point is not None and np.isfinite(support_error)
                                    and support_error <= 3 and angle >= 1)
        if point is None:
            result['failure'] = 'nonpositive_support_depth'
        elif not result['support_ok']:
            result['failure'] = 'support_quality'
        else:
            local = ps[2][:, :3] @ point + ps[2][:, 3]
            if local[2] <= 1e-9:
                result['failure'] = 'nonpositive_query_depth'
            else:
                predicted = cs[2].project(local[None])[0]
                error = float(np.linalg.norm(predicted - coords[2]))
                if np.isfinite(error):
                    result['predicted_xy'] = predicted.tolist()
                    result['query_error_px'] = error
                else:
                    result['failure'] = 'nonfinite_projection'
        output.append(result)
    return output


def _stage_summary(rows):
    from .local_evaluation import error_summary
    def summarize(subset):
        errors = [r['query_error_px'] if r['query_error_px'] is not None else np.inf
                  for r in subset]
        return {**error_summary(errors),
                'failures': len(subset) - sum(r['query_error_px'] is not None for r in subset)}
    groups = {'all': summarize(rows)}
    for field in ('camera', 'time_bin', 'zone', 'scene_type', 'gap_kind',
                  'camera_mix', 'camera_pattern'):
        for value in sorted({r[field] for r in rows}):
            groups[f'{field}/{value}'] = summarize([r for r in rows if r[field] == value])
    groups['failure_reasons'] = dict(Counter(r['failure'] for r in rows if r['failure']))
    return groups


def write_manual_error_gallery(out, report, annotations):
    """Post-label visual audit; never feeds labels or geometry selection back to BA."""
    rendered = json.loads((out / 'rendered_candidates.json').read_text())
    candidates = {r['id']: r for r in rendered['rows']}
    stages = {name: {r['id']: r for r in value['rows']}
              for name, value in report['stages'].items()}
    if not stages:
        return None
    final = stages['shared_calibration']
    ids = sorted(final, key=lambda id_: (
        final[id_]['query_error_px'] is not None,
        -(final[id_]['query_error_px'] or 0), id_))
    stage_labels = [('fixed_calibration', '固定内参', '#71c6ff'),
                    ('triangle_supported', '训练轨迹筛选', '#f4b45e'),
                    ('shared_calibration', '共享内参', '#ec73dc')]
    cards = []
    for id_ in ids:
        row = candidates[id_]
        label = annotations['annotations'][id_]
        panels = []
        for j, obs in enumerate(row['observations']):
            def marker(xy, klass, color):
                x = (xy[0] - obs['crop_origin'][0]) / 192 * 100
                y = (xy[1] - obs['crop_origin'][1]) / 192 * 100
                return (f'<span class="mark {klass}" style="left:{x:.3f}%;top:{y:.3f}%;'
                        f'border-color:{color}"></span>')
            marks = marker(obs['xy'], 'seed', '#42e68f')
            marks += marker(label['coords'][j], 'confirmed', '#ff7373')
            if j == 2:
                for name, _, color in stage_labels:
                    predicted = stages[name][id_]['predicted_xy']
                    if predicted is not None:
                        marks += marker(predicted, 'prediction', color)
            panels.append(f'<div class="panel"><b>{"支持" if j < 2 else "查询"} {j+1}</b>'
                          f'<small>{html.escape(obs["name"])}</small>'
                          f'<div class="crop"><img loading="lazy" src="../{html.escape(obs["crop"])}">'
                          f'{marks}</div></div>')
        values = []
        for name, title, color in stage_labels:
            stage_row = stages[name][id_]
            error = stage_row['query_error_px']
            value = f'{error:.2f} px' if error is not None else f'失败：{stage_row["failure"]}'
            values.append(f'<span style="color:{color}">{title} {html.escape(value)}</span>')
        cards.append(f'<article id="{id_}"><h2>{html.escape(id_)} · '
                     f'{row["camera"]} / {row["time_bin"]} / {row["zone"]}</h2>'
                     f'<p>{"；".join(values)} · 支持间隔 {row["gap_kind"]}</p>'
                     f'<div class="panels">{"".join(panels)}</div></article>')
    page = '''<!doctype html><html lang="zh"><meta charset="utf-8">
<title>人工静态点几何误差复核</title><style>
body{font:15px system-ui;background:#171b23;color:#e5eaf1;margin:18px;line-height:1.45}
a{color:#8bc5ff}article{margin:22px 0;padding:12px;border:1px solid #536173;background:#222a35}
h2{font-size:18px;margin:0 0 7px}p{margin:5px 0 12px}.panels{display:grid;grid-template-columns:repeat(3,minmax(240px,1fr));gap:10px}
.panel{min-width:0}.panel small{display:block;word-break:break-all}.crop{position:relative;width:min(100%,384px);aspect-ratio:1;overflow:hidden}
.crop img{width:100%;height:100%;display:block;image-rendering:pixelated}
.mark{position:absolute;width:10px;height:10px;border:2px solid;border-radius:50%;box-sizing:border-box;transform:translate(-50%,-50%);pointer-events:none}
.confirmed{width:8px;height:8px;transform:translate(-50%,-50%) rotate(45deg);border-radius:0}
.prediction{width:9px;height:9px}.hint{color:#b3bdcb}
@media(max-width:850px){.panels{grid-template-columns:1fr}}
</style><h1>人工确认静态点：事后几何复核</h1>
<p class="hint">本页仅包含先前确认的静态点，按共享内参阶段的失败及误差从大到小排列；排序发生在标签冻结之后，不参与训练或候选抽样。绿色为 LoMa 原点，红色为人工确认点；查询图中的蓝/橙/紫分别为固定内参、训练轨迹筛选、共享内参预测。预测超出裁片时只显示数值。它是误差归因页，不可把大误差样例比例当作全场比例。</p>
<p><a href="../index.html">返回盲审页</a> · <a href="../manual_evaluation.json">完整数值</a></p>
''' + ''.join(cards) + '</html>'
    destination = out / 'manual_review'
    destination.mkdir(exist_ok=True)
    path = destination / 'index.html'
    path.write_text(page)
    return path


def evaluate_static_review(work, annotations_path):
    """Evaluate fixed identities; no review point or query goes back into BA."""
    import pycolmap
    from .local_reconstruction import assert_holdout_isolated
    root = Path(work).resolve() / 'global_sfm'
    out = root / OUTPUT
    scene = load_scene(root)
    candidates = json.loads((out / 'candidates.json').read_text())
    annotations_path = Path(annotations_path).resolve()
    annotations = json.loads(annotations_path.read_text())
    accepted, annotation_counts = checked_annotations(annotations, candidates, scene)
    report = {'protocol_fingerprint': candidates['protocol_fingerprint'],
              'candidate_fingerprint': candidates['fingerprint'],
              'annotations_sha256': file_digest(annotations_path),
              'annotation_counts': annotation_counts,
              'stages': {}, 'paired_common_finite': {},
              'scope': 'Manually confirmed static correspondences from the original capture; two support coordinates triangulate, the third is query only. Failed support/query predictions remain in the stage denominators. Paired bootstrap resamples one candidate per raw track but does not account for shared-scene correlation. Not independent capture or 3DGS rendering acceptance.'}
    if accepted:
        split = json.loads((root / 'split.json').read_text())
        for stage in ('fixed_calibration', 'triangle_supported', 'shared_calibration'):
            model = root / 'stages' / stage / 'sparse/0'
            rec = pycolmap.Reconstruction(model)
            assert_holdout_isolated(rec, split)
            rows = _evaluate_stage(rec, accepted, scene)
            report['stages'][stage] = {'summary': _stage_summary(rows), 'rows': rows}
        stage_pairs = (('fixed_calibration', 'triangle_supported'),
                       ('fixed_calibration', 'shared_calibration'),
                       ('triangle_supported', 'shared_calibration'))
        for baseline_stage, stage in stage_pairs:
            baseline = report['stages'][baseline_stage]['rows']
            other = report['stages'][stage]['rows']
            if [r['id'] for r in other] != [r['id'] for r in baseline]:
                raise RuntimeError(f'Candidate order changed in {stage}')
            pairs = [(a['query_error_px'], b['query_error_px']) for a, b in zip(baseline, other)
                     if a['query_error_px'] is not None and b['query_error_px'] is not None]
            deltas = np.asarray([b - a for a, b in pairs])
            if len(deltas):
                rng = np.random.default_rng(20260923)
                draws = rng.integers(0, len(deltas), (10000, len(deltas)))
                bootstrap = np.median(deltas[draws], axis=1)
                ci = np.quantile(bootstrap, [.025, .975]).tolist()
            else:
                ci = None
            report['paired_common_finite'][f'{stage}_vs_{baseline_stage}'] = {
                'count': len(deltas),
                'median_delta_px': float(np.median(deltas)) if len(deltas) else None,
                'median_delta_bootstrap_95pct_px': ci,
                'fraction_improved': float(np.mean(deltas < 0)) if len(deltas) else None,
                'improved': int(np.count_nonzero(deltas < 0)),
                'worsened': int(np.count_nonzero(deltas > 0)),
                'bootstrap_note': 'Descriptive 10,000 resamples of unique raw-track candidates; same-scene correlation is not modeled.'}
        gallery = write_manual_error_gallery(out, report, annotations)
        report['review_gallery'] = str(gallery.relative_to(out))
    write_json(out / 'manual_evaluation.json', report)
    print('Manually confirmed static points:', len(accepted), '/', len(candidates['rows']),
          '; evaluation:', out / 'manual_evaluation.json', flush=True)
    return report
