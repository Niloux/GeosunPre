"""Paired LFS render diagnostics on the same all-training camera views."""

import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import pycolmap

from ..camera import Camera
from .ground_pose_probe import fit_floor_from_training
from ..scene import load_scene, write_json


STEPS = (7000, 15000, 30000)
BRANCHES = ('equal_control', 'near_floor_2p0')
NEAR_PATCH_RADIUS_HALF_RES_PX = 12
OUTER_FOV_RADIUS_FRACTION = 0.70


def _rows(path):
    with path.open(newline='') as handle:
        return {row['image_name']: {field: (float(row[field]) if row[field] else None)
                                    for field in ('psnr', 'ssim', 'lpips')}
                for row in csv.DictReader(handle)}


def _summary(values, better_sign):
    arr = np.asarray(values, dtype=float)
    if not len(arr):
        return {'count': 0}
    return {'count': len(arr), 'median': float(np.median(arr)),
            'mean': float(np.mean(arr)), 'p10': float(np.quantile(arr, .1)),
            'p90': float(np.quantile(arr, .9)),
            'weighted_better_fraction': float(np.mean(arr * better_sign > 0))}


def _absolute_stats(values):
    arr = np.asarray([value for value in values if value is not None], dtype=float)
    if not len(arr):
        return {'count': 0}
    return {'count': len(arr), 'mean': float(np.mean(arr)),
            'median': float(np.median(arr)), 'p10': float(np.quantile(arr, .1)),
            'p90': float(np.quantile(arr, .9))}


def _near_ground_pixels(source, image_name, ground_ids, near_cutoff, width, height):
    image = next(im for im in source.images.values() if im.name == image_name)
    center = image.projection_center()
    roi = np.zeros((height, width), np.uint8)
    count = 0
    for observation in image.points2D:
        if not observation.has_point3D() or int(observation.point3D_id) not in ground_ids:
            continue
        point = source.points3D[observation.point3D_id]
        if np.linalg.norm(point.xyz - center) > near_cutoff:
            continue
        x = int(np.floor(observation.xy[0] * width / 3000))
        y = int(np.floor(observation.xy[1] * height / 4000))
        if 0 <= x < width and 0 <= y < height:
            cv2.circle(roi, (x, y), NEAR_PATCH_RADIUS_HALF_RES_PX, 1, -1)
            count += 1
    return roi.astype(bool), count


def _outer_pixels(cam, fov_degrees, width, height):
    xx = ((np.arange(width, dtype=np.float32) + .5) * cam.width / width - cam.params[2]) / cam.params[0]
    yy = ((np.arange(height, dtype=np.float32) + .5) * cam.height / height - cam.params[3]) / cam.params[1]
    radius_squared = xx[None, :] ** 2 + yy[:, None] ** 2
    limit = cam.distorted_radius(fov_degrees)
    return (radius_squared >= (OUTER_FOV_RADIUS_FRACTION * limit) ** 2) & (radius_squared <= limit ** 2)


def _render_error(path, image_name, scene, source, ground_ids, near_cutoff, shared):
    with Image.open(path) as image:
        rgb = np.asarray(image.convert('RGB'))
    height, total_width = rgb.shape[:2]
    if (total_width - 4) % 2:
        raise ValueError(f'Unexpected LFS comparison PNG width: {path}')
    width = (total_width - 4) // 2
    if (width, height) != (1500, 2000):
        raise ValueError(f'Unexpected LFS evaluation resolution: {path}')
    truth = rgb[:, :width, :]
    render = rgb[:, width + 4:, :]
    if render.shape != truth.shape:
        raise ValueError(f'Invalid GT/render panes: {path}')
    with Image.open(shared / 'masks' / (image_name + '.png')) as image:
        valid = cv2.resize(np.asarray(image), (width, height), interpolation=cv2.INTER_NEAREST) > 0
    label = image_name.split('/', 1)[0]
    cam = Camera.from_dict(scene['cameras'][label])
    near, near_point_count = _near_ground_pixels(source, image_name, ground_ids,
                                                  near_cutoff, width, height)
    outer = _outer_pixels(cam, scene['fov_degrees'], width, height)
    error = np.mean(np.abs(truth.astype(np.int16) - render.astype(np.int16)), axis=2)
    result = {'valid_pixels': int(np.count_nonzero(valid)),
              'near_floor_points': near_point_count}
    for key, roi in [('all_valid', valid), ('near_floor', valid & near),
                     ('outer_fov', valid & outer)]:
        result[key + '_pixels'] = int(np.count_nonzero(roi))
        result[key + '_mae_8bit'] = float(np.mean(error[roi])) if np.any(roi) else None
    return result, truth


def analyze_lfs_ab(work):
    root = Path(work).resolve() / 'global_sfm'
    out = root / 'lfs_ab'
    protocol = json.loads((out / 'protocol.json').read_text())
    scene = load_scene(root)
    source = pycolmap.Reconstruction(root / 'ground_joint_ba' / 'equal_control' / 'sparse' / '0')
    ground_ids, near_cutoff, _ = fit_floor_from_training(source)
    runs = out / 'runs'
    report = {'input_protocol_fingerprint': protocol['fingerprint'],
              'evaluation_split': 'training_views', 'held_out_images': 0,
              'steps': {}, 'roi_protocol': {
                  'source': 'frozen equal-control visual SfM reliable floor points',
                  'near_ground_cutoff_visual_units': near_cutoff,
                  'near_patch_radius_at_half_resolution_px': NEAR_PATCH_RADIUS_HALF_RES_PX,
                  'outer_fov_radius_fraction': OUTER_FOV_RADIUS_FRACTION,
                  'image_resolution': [1500, 2000],
                  'metric': '8-bit GT-versus-render MAE in the LFS saved comparison PNG',
                  'limitation': 'All camera images are trained; these are paired diagnostics, not independent validation.'}}
    text_lines = ['# 篮球场 LFS 3DGS A/B：训练视图对照', '',
                  '两组使用相同原始 L/R 图像、合成有效掩码和视觉点云初始化；536 张全部训练。',
                  '下面是训练视图的配对渲染指标，不能当成独立测试集精度。',
                  'PSNR / SSIM 的正差、MAE 的负差表示加权组更好。',
                  '本次离线运行未下载 LPIPS 权重，LFS 没有输出 LPIPS 数值。', '']
    for step in STEPS:
        paths = {branch: runs / branch / f'eval_step_{step}' for branch in BRANCHES}
        if not all((path / 'per_image_metrics.csv').exists() and
                   (path / 'evaluation_manifest.json').exists()
                   for path in paths.values()):
            report['steps'][str(step)] = {'status': 'incomplete'}
            continue
        manifest = {branch: json.loads((path / 'evaluation_manifest.json').read_text())
                    for branch, path in paths.items()}
        if any(value['evaluation_split'] != 'training_views' for value in manifest.values()):
            raise ValueError('LFS unexpectedly used a held-out split')
        ordered = {branch: [(row['image_name'], row['output']) for row in value['images']]
                   for branch, value in manifest.items()}
        if ordered[BRANCHES[0]] != ordered[BRANCHES[1]]:
            raise ValueError(f'LFS A/B evaluation image membership differs at {step}')
        metrics = {branch: _rows(path / 'per_image_metrics.csv')
                   for branch, path in paths.items()}
        names = [name for name, _ in ordered[BRANCHES[0]]]
        if len(names) != len(set(names)) or any(set(rows) != set(names)
                                               for rows in metrics.values()):
            raise ValueError(f'LFS A/B metric rows do not cover the same views at {step}')
        data = {'status': 'complete', 'expected_views': len(names),
                'paired_views': len(names), 'by_group': {},
                'absolute_by_group': {}, 'per_image': []}
        for name, filename in ordered[BRANCHES[0]]:
            values = {branch: metrics[branch][name] for branch in BRANCHES}
            for branch in BRANCHES:
                roi, truth = _render_error(paths[branch] / filename, name, scene,
                                           source, ground_ids, near_cutoff, out / 'shared')
                values[branch].update(roi)
                if branch == BRANCHES[0]:
                    baseline_truth = truth
                elif not np.array_equal(baseline_truth, truth):
                    raise ValueError(f'A/B GT comparison panes differ: {step} {name}')
            data['per_image'].append({'name': name, 'lens': name.split('/', 1)[0],
                                      'equal': values[BRANCHES[0]],
                                      'weighted': values[BRANCHES[1]]})
        for group in ('all', 'L', 'R'):
            selected = [row for row in data['per_image']
                        if group == 'all' or row['lens'] == group]
            group_data = {}
            for key, better_sign in [('psnr', 1), ('ssim', 1), ('lpips', -1),
                                     ('all_valid_mae_8bit', -1),
                                     ('near_floor_mae_8bit', -1),
                                     ('outer_fov_mae_8bit', -1)]:
                deltas = [row['weighted'][key] - row['equal'][key]
                          for row in selected if row['equal'][key] is not None and
                          row['weighted'][key] is not None]
                group_data[key] = _summary(deltas, better_sign)
            data['by_group'][group] = group_data
            data['absolute_by_group'][group] = {
                branch: {key: _absolute_stats([row[value_key][key]
                                               for row in selected])
                         for key in ('psnr', 'ssim', 'lpips',
                                     'all_valid_mae_8bit', 'near_floor_mae_8bit',
                                     'outer_fov_mae_8bit')}
                for branch, value_key in ((BRANCHES[0], 'equal'),
                                          (BRANCHES[1], 'weighted'))}
        report['steps'][str(step)] = data
        text_lines.append(f'## 第 {step} 步')
        text_lines.append('')
        text_lines.append(f'配对 {len(data["per_image"])} / {len(names)} 张训练图像；以下为“加权 − 等权”。')
        text_lines.append('')
        text_lines.append('|范围|PSNR 中位差 dB|SSIM 中位差|近地点 MAE 中位差 8-bit|鱼眼外圈 MAE 中位差 8-bit|')
        text_lines.append('|---|---:|---:|---:|---:|')
        for group, values in data['by_group'].items():
            median = lambda key: (f'{values[key]["median"]:+.4f}'
                                  if values[key]['count'] else '—')
            text_lines.append(f'|{group}|{median("psnr")}|{median("ssim")}|'
                              f'{median("near_floor_mae_8bit")}|{median("outer_fov_mae_8bit")}|')
        text_lines.append('')
        text_lines.append('|范围|等权 PSNR 均值 dB|加权 PSNR 均值 dB|等权 SSIM 均值|加权 SSIM 均值|')
        text_lines.append('|---|---:|---:|---:|---:|')
        for group, values in data['absolute_by_group'].items():
            equal, weighted = values[BRANCHES[0]], values[BRANCHES[1]]
            text_lines.append(f'|{group}|{equal["psnr"]["mean"]:.4f}|'
                              f'{weighted["psnr"]["mean"]:.4f}|'
                              f'{equal["ssim"]["mean"]:.4f}|'
                              f'{weighted["ssim"]["mean"]:.4f}|')
        text_lines.append('')
    write_json(out / 'lfs_comparison.json', report)
    (out / 'lfs_comparison.md').write_text('\n'.join(text_lines) + '\n')
    return {'output': str(out / 'lfs_comparison.json'),
            'complete_steps': [int(key) for key, value in report['steps'].items()
                               if value['status'] == 'complete']}
