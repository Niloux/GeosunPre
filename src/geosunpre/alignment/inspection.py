"""Inspect structural residuals without fitting or replacing a transform."""

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pycolmap

from geosunpre.artifacts import digest_cache, file_digest, write_json
from geosunpre.colmap import MODEL_FILES, verify_colmap_export
from geosunpre.alignment.diagnostics import (SUPPORT_REASONS, support_diagnostics,
                                            support_summary, temporal_consistency, track_geometry, viewing_side_proxy)
from geosunpre.alignment.geometry import quantiles, spatial_split, surfaces, transform_points, validate_similarity
from geosunpre.alignment.structure_report import region_gallery, write_structure_inspection
from geosunpre.alignment.las import sample_las_records
from geosunpre.scene import parse_name


def structural_regions(points, diagnostic, tracks, temporal, *, per_group=2):
    supported = diagnostic['reason'] == 0
    vertical = supported & (np.abs(diagnostic['normals'][:, 2]) <= np.sin(np.deg2rad(25)))
    plane = np.abs(diagnostic['signed_plane_error'])
    groups = {
        'vertical_over_5cm': vertical & (plane > .05),
        'vertical_within_2cm': vertical & (plane <= .02),
        'nearby_nonplanar': diagnostic['reason'] == SUPPORT_REASONS.index('nonplanar'),
        'sparse_neighbourhood': diagnostic['reason'] == SUPPORT_REASONS.index('sparse_neighbourhood'),
        'no_close_las': diagnostic['reason'] == SUPPORT_REASONS.index('no_close_las'),
        'far_gap_over_2m': diagnostic['nearest_distance'] > 2.,
    }
    regions = []
    for label, mask in groups.items():
        tiles, counts = np.unique(points['tiles'][mask], axis=0, return_counts=True)
        order = np.argsort(-counts, kind='stable')[:per_group]
        for index in order:
            members = mask & np.all(points['tiles'] == tiles[index], axis=1)
            regions.append({'group': label, 'tile': tiles[index].tolist(),
                            'indices': np.flatnonzero(members),
                            'centre': np.median(points['final_xyz'][members], axis=0).tolist()})
    for patch in sorted(temporal['patches'], key=lambda p: -p['abs_median_separation'])[:3]:
        distance = np.linalg.norm(points['final_xyz'] - patch['centre'], axis=1)
        members = distance <= 1.
        if not members.any():
            members = distance <= max(2., distance.min() + .1)
        regions.append({'group': 'las_repeat_separation', 'tile': None,
                        'indices': np.flatnonzero(members), 'centre': patch['centre'],
                        'repeat_patch': patch})
    for index, region in enumerate(regions):
        members = region['indices']
        keep = supported[members]
        region.update(id=f'region_{index:02d}', points=len(members),
                      holdout_points=int(points['holdout'][members].sum()),
                      supported_points=int(keep.sum()),
                      nearest_distance=quantiles(diagnostic['nearest_distance'][members]),
                      point_to_plane_on_supported=quantiles(plane[members][keep]),
                      max_triangulation_angle_deg=quantiles(tracks['max_triangulation_angle_deg'][members]))
    return regions


def _csv(path, rows):
    rows = list(rows)
    if not rows:
        Path(path).write_text('', encoding='utf-8')
        return
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def inspect_alignment(alignment, output, *, workers=4):
    """Replay the frozen support gate and publish an independent diagnostic."""
    alignment, output = Path(alignment).resolve(), Path(output).resolve()
    if alignment.is_relative_to(output) or output.is_relative_to(alignment):
        raise ValueError('Choose an inspection output separate from the alignment')
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError('Inspection output already exists; choose a fresh directory')
    if not 1 <= workers <= 16:
        raise ValueError('Expected 1--16 workers')
    with digest_cache():
        input_hashes = {name: file_digest(alignment / name)
                        for name in ('report.json', 'transform.json', 'registration_points.npz')}
        baseline = json.loads((alignment / 'report.json').read_text())
        transform = json.loads((alignment / 'transform.json').read_text())
        if (baseline.get('kind') != 'visual_to_las_registration_diagnostic'
                or transform['report_sha256'] != input_hashes['report.json']):
            raise ValueError('Alignment report/transform identity mismatch')
        matrix = validate_similarity(transform['matrix'])
        initial = validate_similarity(transform['initial_matrix'])
        dataset, las = Path(baseline['inputs']['dataset']), Path(baseline['inputs']['las'])
        protected = [alignment, dataset, las]
        for path in baseline['inputs']['initialization']:
            protected.append(Path(path))
        if any(output.is_relative_to(path.resolve()) or path.resolve().is_relative_to(output)
               for path in protected):
            raise ValueError('Choose an inspection output separate from all source data')
        source_hashes = {name: file_digest(dataset / 'sparse/0' / name) for name in MODEL_FILES}
        if (source_hashes != baseline['inputs']['visual_model_sha256']
                or file_digest(dataset / 'colmap_export.json') != baseline['inputs']['colmap_export_sha256']
                or file_digest(las) != baseline['inputs']['las_sha256']):
            raise ValueError('Source geometry changed since the alignment')
        source_verification = verify_colmap_export(dataset)
        identity = json.loads((dataset / 'colmap_export.json').read_text())
        image_hashes = {row['name']: row['sha256'] for row in identity['images']}
        with np.load(alignment / 'registration_points.npz', allow_pickle=False) as archive:
            points = {name: archive[name] for name in archive.files}
        split = baseline['split']
        partition_xyz = points.get('partition_xyz', points['initial_xyz'])
        partition_matrix = validate_similarity(transform.get('partition_matrix', transform['initial_matrix']))
        holdout, tiles, _ = spatial_split(partition_xyz, tile_size=split['tile_size'],
                                         fraction=split['fraction_requested'], seed=split['seed'])
        if (not np.array_equal(holdout, points['holdout']) or not np.array_equal(tiles, points['tiles'])
                or len(np.unique(points['point_ids'])) != baseline['visual_points']['eligible']):
            raise ValueError('Stored points or spatial split differ from the frozen alignment')
        rec = pycolmap.Reconstruction(dataset / 'sparse/0')
        source = np.array([rec.points3D[int(pid)].xyz for pid in points['point_ids']])
        if (not np.allclose(transform_points(matrix, source), points['final_xyz'], atol=1e-10, rtol=0)
                or not np.allclose(transform_points(initial, source), points['initial_xyz'], atol=1e-10, rtol=0)
                or not np.allclose(transform_points(partition_matrix, source), partition_xyz, atol=1e-10, rtol=0)):
            raise ValueError('Stored diagnostic points do not match the frozen transform/source')
        target, metadata, times = sample_las_records(
            las, max_points=baseline['recipe']['max_las_points'],
            seed=baseline['las']['sample_seed'])
        if metadata != baseline['las']:
            raise ValueError('LAS sample recipe no longer reproduces the alignment')
        surface = surfaces(target, voxel_size=baseline['recipe']['voxel_sizes'][-1],
                           workers=workers, diagnostics=True)
        diagnostic = support_diagnostics(points['final_xyz'], surface,
            support_distance=baseline['recipe']['surface_support_distance'], workers=workers)
        if (not np.array_equal(diagnostic['reason'] == 0, points['stable_surface_supported'])
                or not np.allclose(diagnostic['nearest_distance'], points['nearest_distance'], atol=1e-10, rtol=0)
                or not np.allclose(diagnostic['signed_plane_error'], points['signed_plane_error'], atol=1e-10, rtol=0)):
            raise ValueError('Surface replay changed the frozen registration residuals')
        print(f'Inspection: {len(source)} frozen visual points; support gate reproduced exactly', flush=True)
        tracks = track_geometry(rec, points['point_ids'])
        summary = support_summary(diagnostic, tracks, points['holdout'])
        close = diagnostic['nearest_distance'] <= baseline['recipe']['surface_support_distance']
        centres = surface['centres'][np.unique(diagnostic['nearest'][close])]
        temporal = temporal_consistency(target, times, centres, workers=workers)
        cameras = sorted(rec.images.values(), key=lambda im: im.name)
        gps = metadata['gps_time']
        if temporal['patches'] and gps and gps['encoding'] == 'gps_week':
            try:
                exposures = [parse_name(Path(im.name).name) for im in cameras]
            except ValueError:
                exposures = []
            if exposures and len({exposure[0] for exposure in exposures}) == 1:
                temporal = viewing_side_proxy(temporal, [exposure[1] for exposure in exposures],
                    transform_points(matrix, np.array([im.projection_center() for im in cameras])))
            else:
                temporal['viewing_side_proxy'] = {'status': 'image_gps_week_time_unavailable'}
        regions = structural_regions(points, diagnostic, tracks, temporal)
        output.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=f'.{output.name}-', dir=output.parent) as temporary:
            stage = Path(temporary)
            gallery = region_gallery(dataset, rec, matrix, surface['points'], points,
                                     regions, image_hashes, stage / 'gallery')
            public_regions = [{key: value for key, value in region.items() if key != 'indices'}
                              for region in regions]
            report = {
                'schema_version': 1, 'kind': 'frozen_alignment_structural_inspection',
                'status': 'diagnostic_only', 'created_utc': datetime.now(timezone.utc).isoformat(),
                'absolute_accuracy_validated': False, 'transform_updated': False,
                'sfm_rerun': False, 'alignment': str(alignment),
                'alignment_sha256': input_hashes, 'source_inputs': baseline['inputs'],
                'source_verification': source_verification, 'las_sample_reproduced': True,
                'support_and_residuals_reproduced': True,
                'implementation_sha256': {name: file_digest(Path(__file__).parent / name) for name in
                                         ('inspection.py', 'diagnostics.py', 'structure_report.py', 'geometry.py', 'las.py')},
                'versions': {'numpy': np.__version__, 'pycolmap': pycolmap.__version__},
                'points': len(source), 'holdout_points': int(points['holdout'].sum()),
                'support': summary, 'las_repeat_geometry': temporal,
                'regions': public_regions, 'gallery': gallery,
                'limitations': [
                    '只诊断已有变换，未修改、选择或替换配准候选。',
                    '单位及 LAS 坐标约定继承原配准假设，未验收绝对精度。',
                    '空间留出保持原划分；本报告检查了留出数据，后续据此调参应视为探索，不能沿用其独立验收结论。',
                    '几何平面与法向不代表语义类别；墙体、围栏、植被须结合原图核对。',
                    '最近距离同时受范围、采样、遮挡、物体几何与视觉深度影响，不能直接解释成 LAS 漂移。',
                    '最大三角化角描述观测几何，不是深度误差或精度保证。',
                    '扫描时段比较只覆盖足够采样、法向兼容且切向范围重叠的局部平面。',
                    '不同时段差异也可能来自物体运动、相邻表面、局部曲率或入射角；不等价于轨迹误差。',
                    '观察侧使用时间邻近的视觉相机中心作弱代理；没有验证相机与雷达外参或时间同步，同侧也不保证同一表面。',
                    '原图标点来自真实 COLMAP 二维观测；黄色 LAS 为局部投影，未进行完整遮挡判断。'],
            }
            write_json(stage / 'structure_report.json', report)
            _csv(stage / 'support_points.csv', (
                {'point_id': int(pid), 'holdout': bool(points['holdout'][index]),
                 'reason': SUPPORT_REASONS[diagnostic['reason'][index]],
                 'nearest_distance': float(diagnostic['nearest_distance'][index]),
                 'supported_abs_plane_error': float(abs(diagnostic['signed_plane_error'][index]))
                    if diagnostic['reason'][index] == 0 else '',
                 **{key: float(value[index]) for key, value in tracks.items()}}
                for index, pid in enumerate(points['point_ids'])))
            _csv(stage / 'repeat_patches.csv', (
                {'x': p['centre'][0], 'y': p['centre'][1], 'z': p['centre'][2],
                 'orientation': p['orientation'], 'early_time': p['early_time'], 'late_time': p['late_time'],
                 'early_samples': p['early_samples'], 'late_samples': p['late_samples'],
                 'pairs': p['pairs'], 'abs_median_separation': p['abs_median_separation'],
                 'viewing_side_proxy': p.get('viewing_side_proxy', {}).get('status', 'unavailable'),
                 'median_absolute_deviation': p['median_absolute_deviation']}
                for p in temporal['patches']))
            np.savez_compressed(stage / 'structure_points.npz', point_ids=points['point_ids'],
                                xyz=points['final_xyz'], holdout=points['holdout'], **diagnostic, **tracks)
            write_structure_inspection(stage / 'inspection.html', report, points, diagnostic,
                                       tracks, surface['points'], regions, output=output)
            for name, expected in input_hashes.items():
                if file_digest(alignment / name) != expected:
                    raise ValueError('Alignment changed during inspection')
            if (file_digest(las) != baseline['inputs']['las_sha256']
                    or {name: file_digest(dataset / 'sparse/0' / name) for name in MODEL_FILES} != source_hashes):
                raise ValueError('Source geometry changed during inspection')
            for row in gallery:
                if file_digest(dataset / 'images' / row['image']) != row['source_sha256']:
                    raise ValueError('Source image changed during inspection')
            result = {'output': str(output), 'report': str(output / 'inspection.html'),
                      'status': report['status'], 'transform_updated': False,
                      'source_model_unchanged': True, 'support': summary['primary_reasons'],
                      'las_repeat_geometry': temporal['abs_median_separation_by_orientation'],
                      'viewing_side_proxy': temporal.get('viewing_side_proxy'),
                      'regions': len(regions), 'gallery_images': len(gallery)}
            write_json(stage / 'result.json', result)
            stage.rename(output)
    return result
