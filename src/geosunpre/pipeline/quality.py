"""Production diagnostics on fitted data; these are not held-out accuracy estimates."""

import html
import json
from pathlib import Path

import numpy as np

from geosunpre.camera import Camera
from geosunpre.sfm.statistics import error_summary
from geosunpre.sfm.reconstruction import cheirality_health
from geosunpre.sfm.track_filter import CONFIG, reconstruction_coverage
from geosunpre.device import FOV_DEGREES
from geosunpre.pipeline.observations import SCOPE


def production_quality(rec):
    cameras = {}
    for cid, cam in rec.cameras.items():
        camera = Camera('L' if cid == 1 else 'R', cam.width, cam.height, tuple(cam.params), (0, 0))
        camera.distorted_radius(FOV_DEGREES)
        cameras[cid] = camera
    depth = cheirality_health(rec)
    coverage = reconstruction_coverage(rec)
    if (depth['invalid_depth'] or len(coverage['image_components']) != 1
            or coverage['min_observations'] < CONFIG['minimum_training_observations_per_image']
            or coverage['min_occupied_cells'] < CONFIG['minimum_occupied_8x8_cells_per_image']):
        raise ValueError('Production model failed depth or observation coverage checks')
    errors = []
    by_camera = {cid: [] for cid in cameras}
    for im in rec.images.values():
        if not im.has_pose:
            continue
        points = [p for p in im.points2D if p.has_point3D()]
        if not points:
            continue
        xyz = np.array([rec.points3D[p.point3D_id].xyz for p in points])
        observed = np.array([p.xy for p in points])
        pose = im.cam_from_world().matrix()
        local = xyz @ pose[:, :3].T + pose[:, 3]
        residuals = np.linalg.norm(cameras[im.camera_id].project(local) - observed, axis=1)
        residuals[~np.isfinite(local).all(axis=1) | (local[:, 2] <= 1e-9)] = np.inf
        errors.append(residuals)
        by_camera[im.camera_id].append(residuals)
    summarize = lambda rows: error_summary(np.concatenate(rows) if rows else [])
    return {'observation_scope': SCOPE, 'accuracy_scope': 'fitted_observations',
            'registered': rec.num_reg_images(), 'points': rec.num_points3D(),
            'observations': rec.compute_num_observations(), 'cheirality': depth,
            'camera_health': {c.label: {'params': list(c.params), 'monotonic_in_160deg': True}
                              for c in cameras.values()},
            'coverage': coverage, 'training_reprojection': summarize(errors),
            'training_reprojection_by_camera': {cameras[cid].label: summarize(rows)
                                                 for cid, rows in by_camera.items()},
            'note': 'Reprojection residuals use fitted observations, not a test set or independent geometry accuracy.'}


def production_acceptance(report, config):
    """Apply explicit fitted-data limits to both the whole model and each lens."""
    errors = []
    summaries = {'all': report['training_reprojection'],
                 **report.get('training_reprojection_by_camera', {})}
    for label, values in summaries.items():
        if not values['count'] or values['finite_count'] != values['count']:
            errors.append(f'{label}: nonfinite or empty fitted reprojection residuals')
        for metric, limit in (('median', config.quality_max_median_px),
                              ('p90', config.quality_max_p90_px)):
            if values.get(metric) is None or not np.isfinite(values[metric]) or values[metric] > limit:
                errors.append(f'{label}: fitted {metric} {values.get(metric)} exceeds {limit} px')
    for camera, health in report.get('camera_parameter_bounds', {}).items():
        if health.get('parameters_at_bounds'):
            message = f'Camera {camera}: parameters at optimization bounds {health["parameters_at_bounds"]}'
            errors.append(message)
    if errors:
        raise ValueError('Production acceptance rejected: ' + '; '.join(errors))
    return {'status': 'passed',
            'limits': {'median_px': config.quality_max_median_px, 'p90_px': config.quality_max_p90_px,
                       'bounds_policy': 'reject'},
            'scope': 'Fitted-data delivery checks; not independent geometry or 3DGS accuracy.'}


def write_production_report(out, scene):
    out = Path(out)
    mapping = json.loads((out / 'mapping_summary.json').read_text())
    path = out / 'optimization_summary.json'
    stages = json.loads(path.read_text())['stages'] if path.exists() else {}
    rows = ''.join(f'<tr><td>{html.escape(stage)}</td><td>{value["registered"]}</td>'
                   f'<td>{value["points"]}</td><td>{value["observations"]}</td></tr>'
                   for stage, value in stages.items())
    document = f'''<!doctype html><html lang="zh"><meta charset="utf-8">
<title>生产重建报告</title><style>body{{font:16px system-ui;margin:32px}}td,th{{padding:8px;border:1px solid #bbb}}table{{border-collapse:collapse}}</style>
<h1>生产重建报告</h1><p>使用全部通过质量筛选的观测，不划分测试集。</p>
<p>增量建图注册 {mapping['registered']}/{len(scene['images'])} 张图像。</p>
<table><tr><th>阶段</th><th>注册图像</th><th>稀疏点</th><th>观测</th></tr>{rows}</table>
<p><a href="mapping_summary.json">建图记录</a> · <a href="optimization_summary.json">优化记录</a> ·
<a href="final/quality_summary.json">最终质量诊断</a></p>
<p>重投影残差属于拟合数据诊断，不代表测试集精度或下游渲染质量。</p></html>'''
    (out / 'index.html').write_text(document)
