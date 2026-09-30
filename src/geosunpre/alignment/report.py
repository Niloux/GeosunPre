"""Offline inspection artifacts for a registration candidate."""

import html
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def write_ply(path, points, color=(140, 150, 165)):
    points = np.asarray(points)
    records = np.empty(len(points), dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                                          ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    for index, name in enumerate(('x', 'y', 'z')):
        records[name] = points[:, index]
    for index, name in enumerate(('red', 'green', 'blue')):
        records[name] = color[index]
    with Path(path).open('wb') as stream:
        stream.write(('ply\nformat binary_little_endian 1.0\n'
                      f'element vertex {len(points)}\nproperty float x\nproperty float y\nproperty float z\n'
                      'property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n').encode())
        records.tofile(stream)


def projection_overlays(dataset, before, after, target, output, *,
                        initial_label='Pose initialization + LAS', refined_label='Surface refinement + LAS'):
    """Native camera projections, with a coarse nearest-depth pixel buffer."""
    output = Path(output)
    output.mkdir()
    images = []
    target = target[::max(1, int(np.ceil(len(target) / 200_000)))]
    for camera_id in sorted(before.cameras):
        group = sorted((image for image in before.images.values() if image.camera_id == camera_id),
                       key=lambda image: image.name)
        selected = np.unique(np.linspace(0, len(group) - 1, min(4, len(group))).astype(int))
        for index in selected:
            image = group[index]
            with Image.open(Path(dataset) / 'images' / image.name) as original:
                original = original.convert('RGB')
                original.thumbnail((600, 800))
            result = Image.new('RGB', (original.width * 2, original.height + 24), (20, 24, 32))
            for column, rec in enumerate((before, after)):
                camera = rec.cameras[camera_id]
                matrix = rec.images[image.image_id].cam_from_world().matrix()
                local = target @ matrix[:, :3].T + matrix[:, 3]
                local = local[local[:, 2] > 0]
                pixels = camera.img_from_cam(local)
                keep = (np.isfinite(pixels).all(1) & (pixels[:, 0] >= 0) & (pixels[:, 0] < camera.width)
                        & (pixels[:, 1] >= 0) & (pixels[:, 1] < camera.height)
                        & (np.arctan2(np.linalg.norm(local[:, :2], axis=1), local[:, 2]) <= np.deg2rad(80)))
                pixels = pixels[keep] * np.array([original.width / camera.width, original.height / camera.height])
                depth = np.linalg.norm(local[keep], axis=1)
                order = np.argsort(depth, kind='stable')
                pixels = np.floor(pixels[order]).astype(int)
                key = (pixels[:, 1] // 2) * (original.width // 2 + 1) + pixels[:, 0] // 2
                _, first = np.unique(key, return_index=True)
                overlay = original.convert('RGBA')
                marks = Image.new('RGBA', overlay.size)
                draw = ImageDraw.Draw(marks)
                for x, y in pixels[first]:
                    draw.point((int(x), int(y)), fill=(255, 205, 45, 170))
                overlay = Image.alpha_composite(overlay, marks).convert('RGB')
                result.paste(overlay, (column * original.width, 24))
            draw = ImageDraw.Draw(result)
            draw.text((6, 5), initial_label, fill='white')
            draw.text((original.width + 6, 5), refined_label, fill='white')
            name = f'camera_{camera_id}_{index:04d}.jpg'
            result.save(output / name, quality=90)
            images.append({'name': image.name, 'file': f'overlays/{name}'})
    return images


def write_inspection(path, report, *, target, source_initial, source_final, holdout, errors, errors_initial,
                     cameras, cameras_initial, overlays):
    def sample(points, limit):
        return np.arange(0, len(points), max(1, int(np.ceil(len(points) / limit))))

    indices = sample(source_final, 12_000)
    data = {'las': target[sample(target, 16_000)].round(4).tolist(),
            'initial': source_initial[indices].round(4).tolist(),
            'final': source_final[indices].round(4).tolist(),
            'holdout': holdout[indices].astype(int).tolist(),
            'errors': np.minimum(errors[indices], 1.).round(4).tolist(),
            'errors_initial': np.minimum(errors_initial[indices], 1.).round(4).tolist(),
            'cameras': cameras.round(4).tolist(), 'cameras_initial': cameras_initial.round(4).tolist()}
    payload = json.dumps(data, separators=(',', ':'), allow_nan=False).replace('<', '\\u003c')
    rows = []
    for label in ('fit', 'holdout'):
        for stage in ('initial', 'refined'):
            value = report['errors'][label][stage]
            plane = value['point_to_plane_on_supported']
            number = lambda x: '—' if x is None else f'{x:.4f}'
            rows.append(f'<tr><td>{label}</td><td>{stage}</td><td>{value["points"]}</td>'
                        f'<td>{value["surface_support_fraction"]:.1%}</td><td>{number(plane["median"])}</td>'
                        f'<td>{number(plane["p90"])}</td><td>{number(value["nearest_distance"]["median"])}</td>'
                        f'<td>{number(value["nearest_distance"]["p90"])}</td></tr>')
    gallery = ''.join(f'<figure><figcaption>{html.escape(row["name"])}</figcaption>'
                      f'<img loading="lazy" src="{row["file"]}"></figure>' for row in overlays)
    warnings = ''.join(f'<li>{html.escape(value)}</li>' for value in report['limitations'])
    orientations = ''.join(
        f'<tr><td>{html.escape(label)}</td><td>{value["supported_points"]}</td><td>{value["holdout"]["count"]}</td>'
        f'<td>{number(value["holdout"]["median"])}</td><td>{number(value["holdout"]["p90"])}</td></tr>'
        for label, value in report['surface_orientations'].items())
    common = report['holdout_common_surface_support']
    change = common.get('median_change_spatial_bootstrap')
    confidence = ('无足够空间瓦片做区间估计' if not change else
                  f'共同支持点 median 改变量为 {change["estimate"]:.4f} m，空间 bootstrap 95% 区间 '
                  f'[{change["ci95"][0]:.4f}, {change["ci95"][1]:.4f}] m。负数表示下降。')
    template = '''<!doctype html><html lang="zh"><meta charset="utf-8"><title>LAS 配准诊断</title>
<style>body{font:16px system-ui;background:#151922;color:#e3e7ef;max-width:1400px;margin:24px auto;padding:0 20px}
p,li{line-height:1.7}a{color:#85c9ff}table{border-collapse:collapse}td,th{border:1px solid #465163;padding:8px}
button{margin:8px;padding:8px;background:#293345;color:white;border:1px solid #637085;cursor:pointer}
canvas{background:#0a0f17;width:100%;height:620px;touch-action:none}img{max-width:100%}figure{margin:24px 0}
.muted{color:#aeb7c7}.panel{background:#1d2634;padding:16px;border-radius:8px}</style>
<h1>视觉 / LAS 全局配准诊断</h1>
<p class="panel">状态：候选配准，尚未进行绝对精度验收。单位按米假设。点到面统计仅覆盖获得稳定表面支持的点，
支持比例单独列出；它不能替代整场三维位置误差。空间留出只隔离 LAS 表面优化，pose 初始化使用全部对应相机。</p>
<p><a href="report.json">完整诊断 JSON</a> · <a href="transform.json">变换</a> · <a href="regions.csv">分区统计</a> ·
<a href="las_sample.ply">LAS 采样</a> · <a href="aligned_colmap/sparse/0/points3D.ply">变换后的视觉点云</a></p>
<table><tr><th>区域</th><th>阶段</th><th>点数</th><th>表面支持</th><th>点到面 median / m</th><th>点到面 P90 / m</th>
<th>最近点 median / m</th><th>最近点 P90 / m</th></tr>__ROWS__</table>
<p>__CONFIDENCE__</p><h2>按表面方向检查</h2>
<p>以下点到面误差来自细化结果。horizontal / vertical 按 LAS 的 Z 轴定义；地面方向仍依赖坐标假设。
支持点的总体中位数容易被大量水平表面主导，应结合竖直结构及未获支持点判断。</p>
<table><tr><th>表面方向</th><th>全部支持点</th><th>留出支持点</th><th>留出 median / m</th>
<th>留出 P90 / m</th></tr>__ORIENTATIONS__</table>
<p>LAS 灰色；拟合视觉点青色；留出视觉点橙色；相机紫色。拖动旋转，滚轮缩放。
误差着色中，绿色接近表面、红色偏离，灰色表示没有稳定表面支持。视图为正交投影。</p>
<div><button data-view="xy">俯视 XY</button><button data-view="xz">侧视 XZ</button><button data-view="yz">侧视 YZ</button>
<button data-view="3d">三维</button><label><input id="initial" type="checkbox">显示 pose 初值</label>
<label><input id="error" type="checkbox">按表面误差着色</label></div><canvas id="cloud"></canvas>
<h2>原生鱼眼投影检查</h2><p>左侧为 pose 初值，右侧为表面细化。黄色是降采样 LAS 的投影；遮挡由粗像素深度缓冲近似，
图中只检查结构一致性，不计算逐像素精度。</p>__GALLERY__
<h2>当前诊断边界</h2><ul>__WARNINGS__</ul>
<script id="data" type="application/json">__DATA__</script><script>
const d=JSON.parse(document.getElementById('data').textContent),c=document.getElementById('cloud'),ctx=c.getContext('2d');
const centre=[0,1,2].map(k=>d.cameras.reduce((s,p)=>s+p[k],0)/d.cameras.length);
let yaw=0,pitch=0,zoom=8,last=null,mode='xy';
function draw(){const ratio=devicePixelRatio||1;c.width=c.clientWidth*ratio;c.height=c.clientHeight*ratio;ctx.scale(ratio,ratio);
ctx.clearRect(0,0,c.clientWidth,c.clientHeight);ctx.strokeStyle='#1c2533';ctx.beginPath();
for(let x=0;x<c.clientWidth;x+=100){ctx.moveTo(x,0);ctx.lineTo(x,c.clientHeight)}
for(let y=0;y<c.clientHeight;y+=100){ctx.moveTo(0,y);ctx.lineTo(c.clientWidth,y)}ctx.stroke();
const initial=document.getElementById('initial').checked,err=document.getElementById('error').checked;
function pos(p){let [x,y,z]=p.map((v,k)=>v-centre[k]);if(mode==='xz')return[x,z];if(mode==='yz')return[y,z];if(mode==='xy')return[x,y];
let a=Math.cos(yaw)*x-Math.sin(yaw)*y,b=Math.sin(yaw)*x+Math.cos(yaw)*y;return[a,Math.cos(pitch)*b-Math.sin(pitch)*z]}
function point(p,color,size){const [x,y]=pos(p),u=c.clientWidth/2+x*zoom,v=c.clientHeight/2-y*zoom;
if(u<0||v<0||u>c.clientWidth||v>c.clientHeight)return;ctx.fillStyle=color;ctx.fillRect(u,v,size,size)}
d.las.forEach(p=>point(p,'#687586',1));(initial?d.initial:d.final).forEach((p,i)=>{let color=d.holdout[i]?'#ffa844':'#32d2d0';
if(err){let e=(initial?d.errors_initial:d.errors)[i];color=e>=1?'#9a9a9a':`hsl(${Math.max(0,120-120*e/.1)},85%,60%)`}point(p,color,2)});
(initial?d.cameras_initial:d.cameras).forEach(p=>point(p,'#d47dff',3));ctx.fillStyle='#d9e2ef';ctx.fillText(`单格 = ${Math.round(100/zoom*10)/10} m`,12,20)}
c.addEventListener('pointerdown',e=>{last=[e.clientX,e.clientY];c.setPointerCapture(e.pointerId)});
c.addEventListener('pointerup',()=>last=null);c.addEventListener('pointermove',e=>{if(!last)return;mode='3d';
yaw+=(e.clientX-last[0])*.008;pitch+=(e.clientY-last[1])*.008;last=[e.clientX,e.clientY];draw()});
c.addEventListener('wheel',e=>{e.preventDefault();zoom=Math.max(.5,Math.min(1000,zoom*Math.exp(-e.deltaY*.001)));draw()},{passive:false});
document.querySelectorAll('button[data-view]').forEach(b=>b.onclick=()=>{mode=b.dataset.view;draw()});
document.getElementById('initial').onchange=draw;document.getElementById('error').onchange=draw;window.addEventListener('resize',draw);draw();
</script></html>'''
    for key, value in [('ROWS', ''.join(rows)), ('GALLERY', gallery), ('WARNINGS', warnings), ('DATA', payload),
                       ('ORIENTATIONS', orientations), ('CONFIDENCE', html.escape(confidence))]:
        template = template.replace(f'__{key}__', value)
    if report.get('validation_scope') == 'exploratory_reuses_previously_reviewed_data':
        comparison_rows = []
        for label, value in report['reference_comparison']['groups'].items():
            before, after = value['initial_absolute_plane_error'], value['refined_absolute_plane_error']
            comparison_rows.append(
                f'<tr><td>{html.escape(label)}</td><td>{value["points"]}</td>'
                f'<td>{number(before["median"])}</td><td>{number(after["median"])}</td>'
                f'<td>{number(before["p90"])}</td><td>{number(after["p90"])}</td></tr>')
        optimization = report['optimization']
        coverage = optimization['initial_coverage']
        refinement = (
            '<h2>结构细化与上一版候选</h2><p>多尺度 LAS 平面用于筛选。视觉法向可用时检查法向一致性；'
            '法向不可用时，仅允许精度代理标准差较小的视觉点参与。水平/竖直结构分组控制总权重。'
            f'本轮模式为 {html.escape(optimization["mode"])}，优化 {optimization["estimated_dofs"]} 个自由度。'
            f'初始可靠竖直对应 {coverage["vertical_points"]} 点，覆盖 {coverage["vertical_tiles"]} 个空间瓦片。'
            '相机与内参始终固定，精度代理不是完整 BA 协方差或真实精度。</p>'
            '<p>以下只比较两版都获表面支持的相同留出点，方向分组固定于参考候选。'
            '旧留出区域已用于方案分析，本表是探索性对照；不能作为独立精度验收。</p>'
            '<table><tr><th>方向</th><th>共同支持点</th><th>上一版 median / m</th>'
            '<th>本版 median / m</th><th>上一版 P90 / m</th><th>本版 P90 / m</th></tr>'
            + ''.join(comparison_rows) + '</table><h2>按表面方向检查</h2>')
        template = template.replace('pose 初值', '上一版候选')
        template = template.replace('表面细化', '结构细化')
        template = template.replace('空间留出只隔离 LAS 表面优化，pose 初始化使用全部对应相机。',
            '本轮重用已检查过的旧留出区域，仅作探索性对照。优化器仍只收到 fit 点；这些结果不是独立精度验收。')
        template = template.replace('<h2>按表面方向检查</h2>', refinement)
    Path(path).write_text(template, encoding='utf-8')
