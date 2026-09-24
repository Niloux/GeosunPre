"""Visual, scale-free checks of an image-only reconstructed camera path."""
import html
import json
from pathlib import Path

import numpy as np
import pycolmap

from ..scene import load_scene, write_json


def build_trajectory_review(work, stage='mapping'):
    work = Path(work).resolve()
    out = work / 'global_sfm'
    model = out / ('mapping' if stage == 'mapping' else f'stages/{stage}/sparse/0')
    rec = pycolmap.Reconstruction(model)
    scene = load_scene(out)
    records = {r['name']: r for r in scene['images']}
    rows = {'L': [], 'R': []}
    for im in rec.images.values():
        if not im.has_pose: continue
        pose = im.cam_from_world().matrix()
        center = -pose[:3, :3].T @ pose[:, 3]
        record = records[im.name]
        rows[record['camera']].append({'name': im.name, 'frame': record['frame_index'],
                                       'time': record['seconds'], 'center': center.tolist(),
                                       'rotation': pose[:3, :3].tolist()})
    for values in rows.values(): values.sort(key=lambda r: r['frame'])
    centers = np.array([r['center'] for values in rows.values() for r in values])
    centered = centers - np.median(centers, axis=0)
    _, _, axes = np.linalg.svd(centered, full_matrices=False)
    axes = axes[:2].T
    projected = centered @ axes
    lower = np.min(projected, axis=0); upper = np.max(projected, axis=0)
    span = np.max(np.maximum(upper - lower, 1e-6))
    def pixel(xyz):
        uv = (np.asarray(xyz) - np.median(centers, axis=0)) @ axes
        return tuple((80 + (uv - lower) / span * 800).tolist())
    summary = {'stage': stage, 'registered': rec.num_reg_images(), 'unit': 'arbitrary SfM scale',
               'axes': 'First two principal components of camera centers; projection for display, not map alignment',
               'by_camera': {}, 'outliers': []}
    paths = []; labels = []
    for camera, values in rows.items():
        pts = np.array([r['center'] for r in values])
        steps = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        rotations = np.array([r['rotation'] for r in values])
        angles = np.rad2deg(np.arccos(np.clip((np.einsum('nij,nij->n', rotations[1:], rotations[:-1]) - 1) / 2, -1, 1)))
        median = float(np.median(steps)) if len(steps) else 0.
        breaks = np.flatnonzero((steps > 5 * median) & (steps > 0)) if median else []
        summary['by_camera'][camera] = {'images': len(values),
            'adjacent_step': {k: float(fn(steps)) for k, fn in [('median', np.median), ('p95', lambda a: np.quantile(a, .95)), ('max', np.max)]} if len(steps) else None,
            'adjacent_rotation_degrees': {k: float(fn(angles)) for k, fn in [('median', np.median), ('p95', lambda a: np.quantile(a, .95)), ('max', np.max)]} if len(angles) else None,
            'step_gt_5x_median': len(breaks)}
        for i in breaks:
            summary['outliers'].append({'camera': camera, 'from_frame': values[i]['frame'], 'to_frame': values[i+1]['frame'],
                                        'step': float(steps[i]), 'median_ratio': float(steps[i]/median)})
        coords = [pixel(r['center']) for r in values]
        color = '#65b9ff' if camera == 'L' else '#ffbf70'
        path = ' '.join(f'{x:.2f},{y:.2f}' for x, y in coords)
        paths.append(f'<polyline points="{path}" fill="none" stroke="{color}" stroke-width="1.5" opacity=".8"/>')
        for i in breaks:
            x1, y1 = coords[i]; x2, y2 = coords[i+1]
            paths.append(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="#ff4d65" stroke-width="3"/>')
        for i, (r, (x, y)) in enumerate(zip(values, coords)):
            if i == 0 or i == len(values)-1 or r['frame'] % 25 == 0:
                labels.append(f'<text x="{x+4:.2f}" y="{y-4:.2f}" fill="{color}" font-size="11">{camera}{r["frame"]}</text>')
        for r, (x, y) in [(values[0], coords[0]), (values[-1], coords[-1])]:
            paths.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5" fill="{color}"/>')
    dest = out / 'trajectory_review' / stage; dest.mkdir(parents=True, exist_ok=True)
    write_json(dest / 'summary.json', summary)
    write_json(dest / 'camera_centers.json', {camera: [{k: v for k, v in r.items() if k != 'rotation'} for r in values]
                                                for camera, values in rows.items()})
    detail = ''.join(f'<li>{camera}：相邻帧位移中位数 {data["adjacent_step"]["median"]:.4f}、P95 {data["adjacent_step"]["p95"]:.4f}，'
                     f'大于中位数 5 倍 {data["step_gt_5x_median"]} 次；转角 P95 {data["adjacent_rotation_degrees"]["p95"]:.2f}°。</li>'
                     for camera, data in summary['by_camera'].items())
    (dest / 'index.html').write_text('''<!doctype html><html lang="zh"><meta charset="utf-8"><title>全量相机路径检查</title>
<style>body{font:16px system-ui;background:#171b23;color:#ddd;margin:24px}a{color:#8bc5ff}p{line-height:1.7}svg{background:#252c37;max-width:100%;height:auto}li{line-height:1.8}</style>
<h1>全量图像求解路径的尺度无关检查</h1><p>蓝色为 L，橙色为 R，红段标记同一相机连续帧之间超过本相机中位步长 5 倍的跳变。显示的是重建相机中心的前两个主成分投影，单位为任意 SfM 尺度；曲线相近不表示同步曝光或已与 LAS 对齐。跳变是复核线索，不自动判错。</p>
<svg viewBox="0 0 960 960" role="img" aria-label="L/R SfM camera trajectories">'''+''.join(paths)+''.join(labels)+'''</svg>
<ul>'''+detail+'''</ul><p><a href="summary.json">数值统计</a> · <a href="camera_centers.json">逐图相机中心</a> · <a href="../../index.html">全场 SfM 总览</a></p></html>''')
    return summary
