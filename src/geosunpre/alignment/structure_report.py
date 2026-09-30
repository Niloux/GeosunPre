"""Native-observation galleries and an offline structural inspection page."""

from collections import Counter
import html
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from geosunpre.artifacts import file_digest
from geosunpre.alignment.diagnostics import SUPPORT_REASONS
from geosunpre.alignment.geometry import transform_points


LABELS = {
    'supported': '获得平面支持', 'no_close_las': '30 cm 内没有 LAS',
    'sparse_neighbourhood': '邻域采样不足', 'line_like': '邻域近似线状',
    'nonplanar': '邻域不满足平面条件', 'thick_patch': '平面厚度超限',
    'vertical_over_5cm': '竖直支持点误差超过 5 cm', 'vertical_within_2cm': '竖直支持点误差不超过 2 cm',
    'nearby_nonplanar': '附近存在 LAS，但邻域不满足平面条件',
    'far_gap_over_2m': '最近 LAS 距离超过 2 m', 'las_repeat_separation': 'LAS 不同时段局部差异',
}


def region_gallery(dataset, reconstruction, matrix, target, points, regions, image_hashes, output):
    output = Path(output)
    output.mkdir()
    inverse = np.linalg.inv(matrix)
    gallery = []
    for region in regions:
        ids = set(map(int, points['point_ids'][region['indices']]))
        counts = Counter(e.image_id for pid in ids for e in reconstruction.points3D[pid].track.elements)
        images = []
        for camera_id in sorted(reconstruction.cameras):
            candidates = [iid for iid in counts if reconstruction.images[iid].camera_id == camera_id]
            if candidates:
                images.append(min(candidates, key=lambda iid: (-counts[iid], reconstruction.images[iid].name)))
        selected = points['final_xyz'][region['indices']]
        low, high = selected.min(0) - .5, selected.max(0) + .5
        local_target = target[np.all((target >= low) & (target <= high), axis=1)]
        local_target = local_target[::max(1, int(np.ceil(len(local_target) / 8000)))]
        for iid in images:
            im = reconstruction.images[iid]
            path = Path(dataset) / 'images' / im.name
            source_hash = file_digest(path)
            if source_hash != image_hashes[im.name]:
                raise ValueError('Gallery source image differs from the COLMAP delivery')
            camera = reconstruction.cameras[im.camera_id]
            pose = im.cam_from_world().matrix()
            camera_points = transform_points(inverse, local_target) @ pose[:, :3].T + pose[:, 3]
            camera_points = camera_points[camera_points[:, 2] > 0]
            pixels = camera.img_from_cam(camera_points)
            keep = (np.isfinite(pixels).all(1) & (pixels[:, 0] >= 0) & (pixels[:, 0] < camera.width)
                    & (pixels[:, 1] >= 0) & (pixels[:, 1] < camera.height)
                    & (np.arctan2(np.linalg.norm(camera_points[:, :2], axis=1), camera_points[:, 2])
                       <= np.deg2rad(80)))
            pixels = pixels[keep]
            obs = np.array([point.xy for point in im.points2D
                            if point.has_point3D() and point.point3D_id in ids])
            with Image.open(path) as source:
                original = source.convert('RGB')
            preview = original.resize((600, 800))
            overlay = preview.copy()
            draw = ImageDraw.Draw(overlay)
            scaled_las = pixels * [600 / camera.width, 800 / camera.height]
            for x, y in scaled_las:
                draw.point((int(x), int(y)), fill=(255, 205, 45))
            scaled = obs * [600 / camera.width, 800 / camera.height]
            for x, y in scaled:
                draw.ellipse((x - 3, y - 3, x + 3, y + 3), outline=(0, 255, 255), width=2)
            canvas = Image.new('RGB', (1200, 826), (20, 24, 32))
            canvas.paste(preview, (0, 26))
            canvas.paste(overlay, (600, 26))
            draw = ImageDraw.Draw(canvas)
            draw.text((6, 5), f'{region["id"]}: {region["group"]} | {Path(im.name).name}', fill='white')
            draw.text((608, 5), 'Cyan: observed visual features | Yellow: local LAS', fill='white')
            name = f'{region["id"]}_camera_{im.camera_id}.jpg'
            canvas.save(output / name, quality=92)
            # A native-resolution crop identifies far objects that shrink in the full preview.
            low_pixel = np.maximum(np.floor(obs.min(0) - 100), [0, 0]).astype(int)
            high_pixel = np.minimum(np.ceil(obs.max(0) + 100), [camera.width, camera.height]).astype(int)
            crop = original.crop((*low_pixel, *high_pixel))
            crop.thumbnail((600, 600))
            marked = crop.copy()
            scale = np.array(crop.size) / (high_pixel - low_pixel)
            draw = ImageDraw.Draw(marked)
            for x, y in (pixels - low_pixel) * scale:
                if 0 <= x < crop.width and 0 <= y < crop.height:
                    draw.point((int(x), int(y)), fill=(255, 205, 45))
            for x, y in (obs - low_pixel) * scale:
                draw.ellipse((x - 3, y - 3, x + 3, y + 3), outline=(0, 255, 255), width=2)
            pair = Image.new('RGB', (crop.width * 2, crop.height + 26), (20, 24, 32))
            pair.paste(crop, (0, 26))
            pair.paste(marked, (crop.width, 26))
            ImageDraw.Draw(pair).text((6, 5), 'Original crop | Observations + local LAS', fill='white')
            crop_name = name.removesuffix('.jpg') + '_crop.jpg'
            pair.save(output / crop_name, quality=94)
            gallery.append({'region': region['id'], 'image': im.name, 'camera_id': im.camera_id,
                            'observations': len(obs), 'file': f'gallery/{name}',
                            'crop_file': f'gallery/{crop_name}', 'source_sha256': source_hash})
    return gallery


def write_structure_inspection(path, report, points, diagnostic, tracks, target, regions, *, output):
    def number(value):
        return '—' if value is None else f'{value:.4f}'

    support_rows = []
    for name, value in report['support']['primary_reasons'].items():
        support_rows.append(f'<tr><td>{LABELS[name]}</td><td>{value["points"]}</td>'
                            f'<td>{value["holdout_points"]}</td><td>{number(value["nearest_distance"]["median"])}</td>'
                            f'<td>{number(value["max_triangulation_angle_deg"]["median"])}</td></tr>')
    repeat_rows = ''.join(
        f'<tr><td>{label}</td><td>{value["count"]}</td><td>{number(value["median"])}</td>'
        f'<td>{number(value["p90"])}</td><td>{number(value["max"])}</td></tr>'
        for label, value in report['las_repeat_geometry'].get('abs_median_separation_by_orientation', {}).items())
    side = report['las_repeat_geometry'].get('viewing_side_proxy')
    side_rows = (''.join(
        f'<tr><td>{label}</td><td>{value["count"]}</td><td>{number(value["median"])}</td>'
        f'<td>{number(value["p90"])}</td><td>{number(value["max"])}</td></tr>'
        for label, value in side.get('same_side_abs_median_separation_by_orientation', {}).items()) if side else '')
    gallery = []
    for region in regions:
        caption = (f'{LABELS[region["group"]]} · {region["points"]} 点 · '
                   f'留出 {region["holdout_points"]} 点 · 最近距离 median '
                   f'{number(region["nearest_distance"]["median"])} m')
        if region.get('repeat_patch'):
            patch = region['repeat_patch']
            caption += f' · LAS 时段间法向偏移 {patch["abs_median_separation"]:.4f} m'
            proxy = patch.get('viewing_side_proxy', {}).get('status', 'unavailable')
            caption += ' · 观察侧代理：' + {'same_side_proxy': '同侧', 'opposite_sides_proxy': '相反侧',
                                           'near_plane_ambiguous': '接近平面，无法判断',
                                           'time_proxy_unavailable': '时刻不覆盖'}.get(proxy, '未计算')
        items = []
        for row in report['gallery']:
            if row['region'] == region['id']:
                items.append(f'<figure><figcaption>{html.escape(row["image"])} · '
                             f'{row["observations"]} 个该组观测</figcaption><img loading="lazy" src="{row["file"]}">'
                             f'<details><summary>查看局部原图与标点</summary><img loading="lazy" '
                             f'src="{row["crop_file"]}"></details></figure>')
        gallery.append(f'<details><summary>{html.escape(caption)}</summary>'
                       f'<button data-region="{region["id"]}">在点云中定位</button>' + ''.join(items) + '</details>')
    index = np.arange(0, len(points['final_xyz']), max(1, int(np.ceil(len(points['final_xyz']) / 18000))))
    las_index = np.arange(0, len(target), max(1, int(np.ceil(len(target) / 16000))))
    payload = {
        'las': target[las_index].round(4).tolist(), 'xyz': points['final_xyz'][index].round(4).tolist(),
        'reason': diagnostic['reason'][index].tolist(), 'holdout': points['holdout'][index].astype(int).tolist(),
        'angle': tracks['max_triangulation_angle_deg'][index].round(3).tolist(),
        'error': np.abs(diagnostic['signed_plane_error'][index]).round(4).tolist(),
        'vertical': (np.abs(diagnostic['normals'][index, 2]) <= np.sin(np.deg2rad(25))).astype(int).tolist(),
        'repeat': [[*p['centre'], p['abs_median_separation']] for p in report['las_repeat_geometry']['patches']],
        'regions': {r['id']: r['centre'] for r in regions},
        'centre': np.median(points['final_xyz'][diagnostic['reason'] == 0], axis=0).round(4).tolist()
            if np.any(diagnostic['reason'] == 0) else np.median(points['final_xyz'], axis=0).round(4).tolist(),
    }
    data = json.dumps(payload, separators=(',', ':'), allow_nan=False).replace('<', '\\u003c')
    options = ''.join(f'<option value="{i}">{LABELS[name]}</option>' for i, name in enumerate(SUPPORT_REASONS))
    warnings = ''.join(f'<li>{html.escape(value)}</li>' for value in report['limitations'])
    baseline = html.escape(os.path.relpath(Path(report['alignment']) / 'inspection.html', output))
    template = '''<!doctype html><html lang="zh"><meta charset="utf-8"><title>配准结构检查</title>
<style>body{font:16px system-ui;background:#151922;color:#e3e7ef;max-width:1400px;margin:24px auto;padding:0 20px}
p,li{line-height:1.7}a{color:#85c9ff}table{border-collapse:collapse}td,th{border:1px solid #465163;padding:8px}
button,select{margin:6px;padding:8px;background:#293345;color:white;border:1px solid #637085}
button,summary{cursor:pointer}canvas{background:#0a0f17;width:100%;height:600px;touch-action:none}
img{max-width:100%}figure{margin:16px 0}details{padding:12px;border:1px solid #465163;margin:12px 0}
.panel{background:#1d2634;padding:16px;border-radius:8px}</style>
<h1>配准后的结构与低支持区域</h1>
<p class="panel">已有变换保持冻结，原有支持标记与残差已重现。这是进一步的结构诊断，尚未验收绝对精度。
下表的主要原因互斥，按“距离、采样、线状、平面、厚度”的顺序归类。几何标签须结合原图判断物体。</p>
<p><a href="__BASELINE__">原配准报告</a> · <a href="structure_report.json">完整结构报告</a> ·
<a href="support_points.csv">逐点诊断</a> · <a href="repeat_patches.csv">扫描时段比较</a></p>
<table><tr><th>支持或拒绝原因</th><th>全部点</th><th>留出点</th><th>最近距离 median / m</th>
<th>最大三角化角 median / °</th></tr>__ROWS__</table>
<p>最大三角化角来自一个点的全部观测相机。角度描述视觉深度约束的几何条件，不能换算成这里的深度精度。</p>
<h2>LAS 不同时段的局部一致性</h2>
<p>仅比较有足够重复观测、法向兼容、切向范围重叠的局部平面；比较范围有限。
以下数值为每块局部表面“时段间法向差的中位数”的绝对值，再按方向汇总。
它与视觉到 LAS 的点到面指标不同；相邻表面或物体变化也可能产生差异。</p>
<table><tr><th>方向</th><th>局部表面数</th><th>median / m</th><th>P90 / m</th><th>max / m</th></tr>__REPEATS__</table>
<p>下表仅列观察侧代理为同侧的局部表面。代理使用 LAS 时刻附近的视觉相机中心，
相机与雷达外参及时间同步仍未验证。相反侧的薄板或构件可能显示不同表面；同侧也不能认证同一表面。</p>
<table><tr><th>同侧代理 / 方向</th><th>局部表面数</th><th>median / m</th><th>P90 / m</th><th>max / m</th></tr>__SIDES__</table>
<h2>点云定位</h2><p>LAS 灰色。支持点青色、距离不足橙色、采样不足紫色、线状黄色、非平面红色、厚度超限粉色。
可筛选竖直支持点或扫描时段比较位置，切换视角、拖动旋转、滚轮缩放；按住 Shift 拖动平移。
点云显示经过采样，表格统计全部点。</p>
<div><select id="filter"><option value="all">全部视觉点</option>__OPTIONS__
<option value="vertical">竖直支持点</option><option value="vertical_bad">竖直支持点误差超过 5 cm</option>
<option value="repeat">扫描时段比较位置</option></select><label><input id="error" type="checkbox">按误差着色</label>
<button data-view="xy">XY</button><button data-view="xz">XZ</button><button data-view="yz">YZ</button>
<button data-view="3d">三维</button><button id="reset">全场</button></div><canvas id="cloud"></canvas>
<h2>定位到原图</h2><p>每个区域从 L/R 相机分别选择该组点观测最多的图像，不要求同步曝光。
左侧原图，右侧青色为该组真实二维特征观测，黄色为冻结变换下局部 LAS 投影。展开局部图可看远处物体。</p>
__GALLERY__<h2>解释边界</h2><ul>__WARNINGS__</ul>
<script id="data" type="application/json">__DATA__</script><script>
const d=JSON.parse(document.getElementById('data').textContent),c=document.getElementById('cloud'),ctx=c.getContext('2d');
let centre=d.centre.slice(),mode='xy',zoom=8,yaw=0,pitch=0,pan=[0,0],last=null;
const colors=['#32d2d0','#ffa844','#b58aff','#ffe66a','#ff6c70','#eaa0cf'];
function draw(){const ratio=devicePixelRatio||1;c.width=c.clientWidth*ratio;c.height=c.clientHeight*ratio;ctx.scale(ratio,ratio);
ctx.clearRect(0,0,c.clientWidth,c.clientHeight);const filter=document.getElementById('filter').value,err=document.getElementById('error').checked;
function pos(p){let [x,y,z]=p.map((v,k)=>v-centre[k]);if(mode==='xz')return[x,z];if(mode==='yz')return[y,z];if(mode==='xy')return[x,y];
let a=Math.cos(yaw)*x-Math.sin(yaw)*y,b=Math.sin(yaw)*x+Math.cos(yaw)*y;return[a,Math.cos(pitch)*b-Math.sin(pitch)*z]}
function dot(p,color,size){const [x,y]=pos(p),u=c.clientWidth/2+pan[0]+x*zoom,v=c.clientHeight/2+pan[1]-y*zoom;
if(u<0||v<0||u>c.clientWidth||v>c.clientHeight)return;ctx.fillStyle=color;ctx.fillRect(u,v,size,size)}
d.las.forEach(p=>dot(p,'#596677',1));let count=0;
if(filter==='repeat'){d.repeat.forEach(p=>{dot(p,`hsl(${Math.max(0,120-120*p[3]/.05)},85%,60%)`,5);count++})}
else d.xyz.forEach((p,i)=>{if(filter!=='all'&&filter!=='vertical'&&filter!=='vertical_bad'&&d.reason[i]!==Number(filter))return;
if((filter==='vertical'||filter==='vertical_bad')&&(!d.vertical[i]||d.reason[i]!==0))return;
if(filter==='vertical_bad'&&d.error[i]<=.05)return;let color=colors[d.reason[i]];
if(err&&d.reason[i]===0)color=`hsl(${Math.max(0,120-120*d.error[i]/.1)},85%,60%)`;dot(p,color,2);count++});
ctx.fillStyle='#d9e2ef';ctx.fillText(`当前采样标点 ${count} | 100 px = ${(100/zoom).toFixed(2)} m`,12,20)}
c.addEventListener('pointerdown',e=>{last=[e.clientX,e.clientY];c.setPointerCapture(e.pointerId)});
c.addEventListener('pointerup',()=>last=null);c.addEventListener('pointercancel',()=>last=null);
c.addEventListener('pointermove',e=>{if(!last)return;if(e.shiftKey){pan[0]+=e.clientX-last[0];pan[1]+=e.clientY-last[1]}
else{mode='3d';yaw+=(e.clientX-last[0])*.008;pitch+=(e.clientY-last[1])*.008}last=[e.clientX,e.clientY];draw()});
c.addEventListener('wheel',e=>{e.preventDefault();zoom=Math.max(.5,Math.min(1000,zoom*Math.exp(-e.deltaY*.001)));draw()},{passive:false});
document.querySelectorAll('button[data-view]').forEach(b=>b.onclick=()=>{mode=b.dataset.view;draw()});
document.querySelectorAll('button[data-region]').forEach(b=>b.onclick=()=>{centre=d.regions[b.dataset.region].slice();zoom=60;pan=[0,0];draw();c.scrollIntoView({behavior:'smooth'})});
document.getElementById('reset').onclick=()=>{centre=d.centre.slice();zoom=8;pan=[0,0];draw()};
document.getElementById('filter').onchange=draw;document.getElementById('error').onchange=draw;window.addEventListener('resize',draw);draw();
</script></html>'''
    replacements = {'BASELINE': baseline, 'ROWS': ''.join(support_rows), 'REPEATS': repeat_rows,
                    'SIDES': side_rows, 'OPTIONS': options, 'GALLERY': ''.join(gallery), 'WARNINGS': warnings, 'DATA': data}
    for key, value in replacements.items():
        template = template.replace(f'__{key}__', value)
    Path(path).write_text(template, encoding='utf-8')
