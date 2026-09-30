"""Offline comparison of image gates and geometry for fixed-intrinsics joint BA."""

import html
import json
from pathlib import Path


def write_joint_report(path, report):
    def number(value, places=3):
        return '—' if value is None else f'{value:.{places}f}'

    labels = {'pure_image_ba': '固定内参纯图像 BA', 'pose_only': '图像 + pose', 'pose_las': '图像 + pose + LAS'}
    reference = report['reference_image_statistics']['all']
    rows = [f'<tr><td>输入纯图像模型</td><td>工作参考</td><td>{number(reference["median"])}</td>'
            f'<td>{number(reference["p90"])}</td><td>{number(reference["p99"])}</td>'
            f'<td>{number(report["reference_position_residual_m"]["median"],4)}</td>'
            f'<td>{report["reference_las_holdout"]["surface_support_fraction"]:.2%}</td></tr>']
    comparisons = []
    for name, stage in report['stages'].items():
        image = stage['image_statistics']['all']
        status = '通过图像及几何门槛' if stage['accepted_for_candidate_review'] else '未通过 / 未更新'
        rows.append(f'<tr><td><a href="stages/{name}/diagnostic.json">{labels[name]}</a></td><td>{status}</td>'
            f'<td>{number(image["median"])}</td><td>{number(image["p90"])}</td><td>{number(image["p99"])}</td>'
            f'<td>{number(stage["input_position_residual_m"]["median"],4)}</td>'
            f'<td>{stage["las"]["las_holdout"]["surface_support_fraction"]:.2%}</td></tr>')
        for group, value in stage['las']['fixed_reference_orientation_groups'].items():
            old, new = value['initial_absolute_plane_error'], value['refined_absolute_plane_error']
            comparisons.append(f'<tr><td>{labels[name]}</td><td>{group}</td><td>{value["points"]}</td>'
                f'<td>{number(None if old["median"] is None else old["median"]*1000)}</td>'
                f'<td>{number(None if new["median"] is None else new["median"]*1000)}</td>'
                f'<td>{number(None if old["p90"] is None else old["p90"]*1000)}</td>'
                f'<td>{number(None if new["p90"] is None else new["p90"]*1000)}</td></tr>')
    data = []
    for iid, name in sorted(report['image_names'].items(), key=lambda row: row[1]):
        key = 'image/' + iid
        if key not in report['reference_image_statistics']:
            continue
        row = {'id': iid, 'name': name, 'input': report['reference_image_statistics'][key], 'stages': {}}
        for label, stage in report['stages'].items():
            failures = [failure for failure in stage['image_gate_vs_pure_ba']['failures'] if failure['group'] == key]
            row['stages'][label] = {**stage['image_statistics'][key], 'passed': not failures}
        data.append(row)
    payload = json.dumps(data, ensure_ascii=False, separators=(',', ':'), allow_nan=False).replace('<', '\\u003c')
    limits = ''.join(f'<li>{html.escape(value)}</li>' for value in report['limitations'])
    gallery = ''.join(f'<figure><figcaption>{html.escape(item["name"])}</figcaption>'
                      f'<img loading="lazy" src="{item["file"]}"></figure>' for item in report['overlays'])
    gate_rows = ''.join(
        f'<tr><td>{labels[name]}</td><td>{stage["accepted_updates"]}</td>'
        f'<td>{stage["image_gate_vs_pure_ba"]["failed_checks"]}</td>'
        f'<td>{number(stage["geometry_gate"]["local_camera_displacement"]["max"],4)}</td>'
        f'<td>{number(stage["geometry_gate"]["local_camera_rotation_deg"]["max"],4)}</td>'
        f'<td>{stage["invalid_depth_observations"]}</td>'
        f'<td><a href="stages/{name}/colmap/colmap_export.json">候选 COLMAP</a></td></tr>'
        for name, stage in report['stages'].items())
    page = '''<!doctype html><html lang="zh"><meta charset="utf-8"><title>图像主导联合 BA 对照</title>
<style>body{font:16px system-ui;background:#151922;color:#e3e7ef;max-width:1500px;margin:24px auto;padding:0 20px}
p,li{line-height:1.7}a{color:#85c9ff}table{border-collapse:collapse;width:100%}td,th{border:1px solid #465163;padding:8px;text-align:left}
select,input{padding:7px;margin:8px;background:#293345;color:white;border:1px solid #637085}img{max-width:100%}
.panel{background:#1d2634;padding:16px;border-radius:8px}.bad{color:#ff9999}.good{color:#80d9bd}figure{margin:24px 0}
.scroll{max-height:650px;overflow:auto}thead{position:sticky;top:0;background:#293345}</style>
<h1>固定内参：图像 / pose / LAS 联合 BA 对照</h1>
<p class="panel">__STATUS__。各优化分支从相同纯图像模型和全局配准候选开始，使用相同图像观测及求解预算。
相机内参和 LAS 点始终固定。位置先验仅使用按图像名对应的相机中心。
图像门槛使用已拟合的观测；下表是回归检查与探索性几何对照，不是独立精度或 3DGS 渲染验收。</p>
<p><a href="report.json">完整报告</a> · <a href="protocol.json">冻结配方和接受门槛</a> · <a href="result.json">本次结果</a></p>
<h2>整体图像与几何统计</h2>
<table><tr><th>方案</th><th>候选状态</th><th>图像 median / px</th><th>图像 P90 / px</th><th>图像 P99 / px</th>
<th>输入位置差 median / m</th><th>LAS 留出支持</th></tr>__ROWS__</table>
<p>位置差越小只表示越接近输入 pose，不表示绝对位置更准确。图像误差覆盖全部既有特征观测，没有按更新后的误差删除观测。</p>
<h2>更新幅度及图像门槛</h2>
<table><tr><th>方案</th><th>接受的外循环更新</th><th>相对纯 BA 失败检查数</th><th>局部相机最大位移 / m</th>
<th>局部相机最大旋转 / deg</th><th>无效深度观测</th><th>独立导出</th></tr>__GATES__</table>
<p>局部位移与旋转排除独立的全局配准变化。全局旋转、尺度和枢轴位移另受限制，详见各方案 diagnostic.json。
未通过门槛的方案仍保存为诊断候选，默认纯图像模型保持原样。</p>
<h2>逐图像查看</h2><p>选择方案后，与相同预算的固定内参纯图像 BA 比较，按误差增加量降序排列。
图像名链接到原始图像；红色表示该图至少一个统计量超过门槛。</p>
<div><select id="stage">__OPTIONS__</select>
<select id="metric"><option value="p90">P90</option><option value="median">median</option><option value="p99">P99</option></select>
<input id="query" placeholder="筛选图像名称"><label><input id="failed" type="checkbox">仅显示失败图像</label><span id="count"></span></div>
<div class="scroll"><table><thead><tr><th>图像</th><th>观测数</th><th>纯 BA / px</th><th>所选方案 / px</th><th>变化 / px</th><th>单图门槛</th></tr></thead><tbody id="images"></tbody></table></div>
<h2>固定方向分组的 LAS 共同支持对照</h2>
<p>每一行只比较参考候选和该方案都获得稳定表面支持的相同旧留出点，方向按参考对应固定。
LAS 留出点不加入点到面残差，但其图像观测仍参与 BA；这些区域已用于方案分析，因此只作探索性对照。</p>
<table><tr><th>方案</th><th>方向</th><th>共同支持点数</th><th>参考 median / mm</th><th>候选 median / mm</th>
<th>参考 P90 / mm</th><th>候选 P90 / mm</th></tr>__LAS__</table>
<h2>原生鱼眼 LAS 投影</h2><p>左侧为输入视觉模型在参考配准下的投影，右侧为图像 + pose + LAS 候选。
黄色 LAS 经过粗像素深度缓冲，仅帮助核对结构，不能从图上量取逐像素精度。</p>__GALLERY__
<h2>诊断边界</h2><ul>__LIMITS__</ul>
<script id="data" type="application/json">__DATA__</script><script>
const rows=JSON.parse(document.getElementById('data').textContent);
function draw(){const mode=document.getElementById('stage').value,metric=document.getElementById('metric').value;
const query=document.getElementById('query').value.toLowerCase(),failed=document.getElementById('failed').checked;
const chosen=rows.filter(r=>r.name.toLowerCase().includes(query)&&(!failed||!r.stages[mode].passed));
chosen.sort((a,b)=>(b.stages[mode][metric]-b.stages.pure_image_ba[metric])-(a.stages[mode][metric]-a.stages.pure_image_ba[metric]));
const body=document.getElementById('images');body.replaceChildren();
const fmt=v=>v===null?'—':v.toFixed(4);
for(const row of chosen){const selected=row.stages[mode],base=row.stages.pure_image_ba;
const tr=document.createElement('tr'),link=document.createElement('a');
link.textContent=row.name;link.href=`stages/${mode}/colmap/images/${row.name.split('/').map(encodeURIComponent).join('/')}`;
const name=document.createElement('td');name.appendChild(link);tr.appendChild(name);
for(const value of [selected.count,fmt(base[metric]),fmt(selected[metric]),fmt(selected[metric]===null||base[metric]===null?null:selected[metric]-base[metric]),selected.passed?'通过':'失败']){
const td=document.createElement('td');td.textContent=value;tr.appendChild(td)}tr.className=selected.passed?'good':'bad';body.appendChild(tr)}
document.getElementById('count').textContent=`${chosen.length} / ${rows.length} 张图像`}
for(const id of ['stage','metric','query','failed'])document.getElementById(id).addEventListener('input',draw);draw();
</script></html>'''
    status = '联合候选通过图像及几何回归门槛' if report['status'] == 'candidate_image_gate_passed' else '联合候选未通过接受门槛'
    options = ''.join(f'<option value="{name}">{labels[name]}</option>'
                      for name in ('pose_las', 'pose_only', 'pure_image_ba') if name in report['stages'])
    for key, value in [('STATUS', status), ('ROWS', ''.join(rows)), ('GATES', gate_rows),
                       ('LAS', ''.join(comparisons)), ('GALLERY', gallery), ('LIMITS', limits), ('DATA', payload),
                       ('OPTIONS', options)]:
        page = page.replace(f'__{key}__', value)
    Path(path).write_text(page, encoding='utf-8')
