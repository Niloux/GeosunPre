"""Produce an immutable, inspectable LAS registration candidate after SfM."""

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time

import numpy as np
import pycolmap

from geosunpre.artifacts import digest_cache, file_digest, write_json
from geosunpre.colmap import MODEL_FILES, export_colmap, verify_colmap_export
from geosunpre.scene import load_poses, parse_name
from geosunpre.alignment.geometry import (error_summary, matrix4, quantiles, refine_similarity,
                                         residuals, robust_similarity, spatial_split,
                                         transform_points, validate_similarity)
from geosunpre.alignment.las import sample_las
from geosunpre.alignment.report import projection_overlays, write_inspection, write_ply


def navigation_initialization(input_root, reconstruction):
    """Test an explicit vendor-frame hypothesis without shifting timestamps.

    ECEF/ENH rows are used ONLY to fit a coordinate conversion. Their differing
    timestamps are recorded, not reconciled or used to pair camera exposures.
    ENH minus its first position is a candidate LAS frame, never a verified CRS.
    """
    root = Path(input_root).resolve()
    files = sorted(root.glob('*_CameraPos_C2E.txt'))
    if len(files) != 1:
        raise ValueError('Expected exactly one CameraPos_C2E file for pose initialization')
    pose_path = files[0]
    stem = pose_path.name.removesuffix('_CameraPos_C2E.txt')
    nav_path, enh_path = root / f'{stem}_pos.txt', root / f'{stem}_pos_ENH.txt'
    nav, enh = np.loadtxt(nav_path), np.loadtxt(enh_path)
    if (nav.shape != enh.shape or nav.ndim != 2 or nav.shape[1] < 5 or len(nav) < 3
            or not np.isfinite(nav).all() or not np.isfinite(enh).all()
            or not np.array_equal(nav[:, 0], enh[:, 0])
            or len(np.unique(nav[:, 0])) != 1 or np.any(np.diff(nav[:, 1]) <= 0)
            or np.any(np.diff(enh[:, 1]) <= 0)):
        raise ValueError('Navigation rows cannot support the coordinate-conversion hypothesis')
    timestamp_difference = nav[:, 1] - enh[:, 1]
    if np.ptp(timestamp_difference) > .02:
        raise ValueError('Navigation row time differences vary too much; supply an explicit initial transform')
    origin = enh[0, 2:5]
    nav_centre = nav[:, 2:5].mean(0)
    spread = np.linalg.svd(nav[:, 2:5] - nav_centre, compute_uv=False)
    if spread[1] < spread[0] * 1e-4:
        raise ValueError('Navigation coordinates are collinear; frame conversion is unobservable')
    conversion = pycolmap.estimate_sim3d(nav[:, 2:5] - nav_centre, enh[:, 2:5] - origin)
    if conversion is None:
        raise ValueError('Failed to estimate navigation coordinate conversion')
    centred_matrix = matrix4(conversion)
    conversion_matrix = centred_matrix.copy()
    conversion_matrix[:3, 3] -= centred_matrix[:3, :3] @ nav_centre
    conversion_errors = np.linalg.norm(transform_points(conversion_matrix, nav[:, 2:5])
                                       - (enh[:, 2:5] - origin), axis=1)
    if conversion_errors.max() > .02 or not .999 < conversion.scale < 1.001:
        raise ValueError('Navigation coordinate conversion does not fit; supply an explicit initial transform')
    poses = load_poses(pose_path)
    centres, positions, exposure_offsets, names = [], [], [], []
    for image in sorted(reconstruction.images.values(), key=lambda image: image.name):
        name = Path(image.name).name
        if name not in poses:
            continue
        pose = poses[name]
        week, seconds, _, _ = parse_name(name)
        if pose['week'] != week or week != nav[0, 0]:
            raise ValueError(f'Camera/navigation GPS week mismatch: {name}')
        centres.append(image.projection_center())
        positions.append(pose['center_raw'])
        exposure_offsets.append(pose['seconds'] - seconds)
        names.append(image.name)
    positions = transform_points(conversion_matrix, positions)
    initial, pose_fit = robust_similarity(centres, positions)
    audit = {'kind': 'vendor_navigation_frame_hypothesis',
             'hypothesis': 'LAS XYZ approximately equals ENH minus first ENH navigation position',
             'vendor_coordinate_convention_verified': False, 'las_origin_verified': False,
             'row_association_scope': 'coordinate conversion only; not temporal synchronization',
             'timestamps_modified': False, 'quaternions_used': False,
             'navigation_row_time_difference_seconds': {
                 'min': float(timestamp_difference.min()), 'median': float(np.median(timestamp_difference)),
                 'max': float(timestamp_difference.max())},
             'camera_pose_minus_filename_seconds': {
                 'min': float(min(exposure_offsets)), 'median': float(np.median(exposure_offsets)),
                 'max': float(max(exposure_offsets))},
             'enh_origin': origin.tolist(), 'navigation_to_candidate_las_matrix': conversion_matrix.tolist(),
             'coordinate_conversion_error': quantiles(conversion_errors),
             'coordinate_conversion_scale': float(conversion.scale), 'camera_position_fit': pose_fit,
             'matched_image_names': names, 'unmatched_images': reconstruction.num_reg_images() - len(names),
             'input_sha256': {str(path): file_digest(path) for path in (pose_path, nav_path, enh_path)}}
    return initial, audit


def projection_invariance(before, after):
    """Verify native projections as well as camera parameters and track identity."""
    if (set(before.points3D) != set(after.points3D) or set(before.images) != set(after.images)
            or set(before.frames) != set(after.frames)
            or before.compute_num_observations() != after.compute_num_observations()):
        raise ValueError('Similarity transform changed model topology')
    maximum, count = 0., 0
    for cid, camera in before.cameras.items():
        other = after.cameras[cid]
        if camera.model != other.model or not np.array_equal(camera.params, other.params):
            raise ValueError('Similarity transform changed camera intrinsics')
    for pid, point in before.points3D.items():
        if [(e.image_id, e.point2D_idx) for e in point.track.elements] != [
                (e.image_id, e.point2D_idx) for e in after.points3D[pid].track.elements]:
            raise ValueError('Similarity transform changed observation tracks')
    for image_id, image in before.images.items():
        other = after.images[image_id]
        if (image.name != other.name or image.camera_id != other.camera_id
                or image.frame_id != other.frame_id or image.num_points2D() != other.num_points2D()):
            raise ValueError('Similarity transform changed image identity')
        if any(a.point3D_id != b.point3D_id or not np.array_equal(a.xy, b.xy)
               for a, b in zip(image.points2D, other.points2D)):
            raise ValueError('Similarity transform changed 2D observations')
        point_ids = [point.point3D_id for point in image.points2D if point.has_point3D()]
        selected = np.unique(np.linspace(0, len(point_ids) - 1, min(20, len(point_ids))).astype(int))
        if not len(selected):
            continue
        ids = [point_ids[index] for index in selected]
        old_xyz = np.array([before.points3D[pid].xyz for pid in ids])
        new_xyz = np.array([after.points3D[pid].xyz for pid in ids])
        old_pose, new_pose = image.cam_from_world().matrix(), other.cam_from_world().matrix()
        old_local = old_xyz @ old_pose[:, :3].T + old_pose[:, 3]
        new_local = new_xyz @ new_pose[:, :3].T + new_pose[:, 3]
        old_pixels = before.cameras[image.camera_id].img_from_cam(old_local)
        new_pixels = after.cameras[other.camera_id].img_from_cam(new_local)
        if not np.isfinite(old_pixels).all() or not np.isfinite(new_pixels).all():
            raise ValueError('Nonfinite projection after similarity transform')
        maximum = max(maximum, float(np.max(np.linalg.norm(new_pixels - old_pixels, axis=1))))
        count += len(ids)
    if maximum > 1e-7 or not after.is_valid():
        raise ValueError('Similarity transform did not preserve visual projections')
    return {'sampled_observations': count, 'maximum_pixel_difference': maximum,
            'intrinsics_and_all_observation_tracks_unchanged': True}


def _common_comparison(initial_points, final_points, surface, tiles, *, workers):
    old_distance, old_plane, old_nearest = residuals(initial_points, surface, workers=workers)
    new_distance, new_plane, new_nearest = residuals(final_points, surface, workers=workers)
    common = ((old_distance <= .3) & surface['valid'][old_nearest]
              & (new_distance <= .3) & surface['valid'][new_nearest])
    result = {'points': int(common.sum()), 'initial_absolute_plane_error': quantiles(np.abs(old_plane[common])),
              'refined_absolute_plane_error': quantiles(np.abs(new_plane[common]))}
    groups, inverse = np.unique(tiles[common], axis=0, return_inverse=True)
    if len(groups) >= 5:
        indices = [np.flatnonzero(inverse == index) for index in range(len(groups))]
        old, new = np.abs(old_plane[common]), np.abs(new_plane[common])
        rng, differences = np.random.default_rng(0), []
        for _ in range(500):
            sample = np.concatenate([indices[index] for index in rng.integers(0, len(groups), len(groups))])
            differences.append(float(np.median(new[sample]) - np.median(old[sample])))
        result['median_change_spatial_bootstrap'] = {
            'tiles': len(groups), 'replicates': 500, 'seed': 0,
            'estimate': float(np.median(new) - np.median(old)),
            'ci95': np.quantile(differences, [.025, .975]).tolist(),
            'sign': 'negative means refinement reduces the supported point-to-plane median'}
    return result, common, old_plane, new_plane, new_distance, new_nearest


def _regions(path, initial_points, final_points, tiles, holdout, surface, *, workers):
    unique, inverse = np.unique(tiles, axis=0, return_inverse=True)
    rows = []
    for index, tile in enumerate(unique):
        mask = inverse == index
        old = error_summary(initial_points[mask], surface, workers=workers)
        new = error_summary(final_points[mask], surface, workers=workers)
        rows.append({'tile_x': int(tile[0]), 'tile_y': int(tile[1]), 'tile_z': int(tile[2]),
                     'partition': 'holdout' if holdout[mask][0] else 'fit', 'points': int(mask.sum()),
                     'initial_support_fraction': old['surface_support_fraction'],
                     'refined_support_fraction': new['surface_support_fraction'],
                     'initial_plane_median': old['point_to_plane_on_supported']['median'],
                     'refined_plane_median': new['point_to_plane_on_supported']['median'],
                     'refined_plane_p90': new['point_to_plane_on_supported']['p90'],
                     'initial_nearest_median': old['nearest_distance']['median'],
                     'refined_nearest_median': new['nearest_distance']['median'],
                     'refined_nearest_p90': new['nearest_distance']['p90']})
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def run_alignment(dataset, las, output, *, input_root=None, initial_transform=None,
                  max_las_points=1_500_000, workers=4, reference_alignment=None,
                  context_fingerprint=None):
    """Read the visual dataset, then publish only a separate diagnostic package."""
    if sum(value is not None for value in (input_root, initial_transform, reference_alignment)) != 1:
        raise ValueError('Choose raw input pose initialization, an explicit initial transform, or a reference alignment')
    if not 1 <= workers <= 16:
        raise ValueError('Expected 1--16 workers')
    dataset, las, output = Path(dataset).resolve(), Path(las).resolve(), Path(output).resolve()
    if output.is_relative_to(dataset) or dataset.is_relative_to(output) or las.is_relative_to(output):
        raise ValueError('Choose an alignment output separate from all source data')
    if input_root is not None and (output.is_relative_to(Path(input_root).resolve())
                                   or Path(input_root).resolve().is_relative_to(output)):
        raise ValueError('Choose an alignment output separate from the raw input')
    if reference_alignment is not None:
        reference_alignment = Path(reference_alignment).resolve()
        if output.is_relative_to(reference_alignment) or reference_alignment.is_relative_to(output):
            raise ValueError('Choose a refinement output separate from the reference alignment')
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError('Alignment output already exists; choose a fresh directory')
    started = time.monotonic()
    with digest_cache():
        source_verification = verify_colmap_export(dataset)
        source_identity = json.loads((dataset / 'colmap_export.json').read_text())
        source_provenance = source_identity.get('provenance', {})
        source_hashes = {name: file_digest(dataset / 'sparse/0' / name) for name in MODEL_FILES}
        las_hash = file_digest(las)
        original = pycolmap.Reconstruction(dataset / 'sparse/0')
        reference = None
        fit_diagnostic = None
        if reference_alignment is not None:
            hashes = {str(reference_alignment / name): file_digest(reference_alignment / name)
                      for name in ('report.json', 'transform.json', 'registration_points.npz')}
            reference = json.loads((reference_alignment / 'report.json').read_text())
            reference_transform = json.loads((reference_alignment / 'transform.json').read_text())
            if (reference.get('kind') != 'visual_to_las_registration_diagnostic'
                    or reference_transform['report_sha256'] != hashes[str(reference_alignment / 'report.json')]
                    or reference['inputs']['visual_model_sha256'] != source_hashes
                    or reference['inputs']['colmap_export_sha256'] != file_digest(dataset / 'colmap_export.json')
                    or reference['inputs']['las_sha256'] != las_hash):
                raise ValueError('Reference alignment does not match the verified visual/LAS sources')
            with np.load(reference_alignment / 'registration_points.npz', allow_pickle=False) as archive:
                reference_points = {name: archive[name] for name in archive.files}
            initial = validate_similarity(reference_transform['matrix'])
            partition_matrix = validate_similarity(reference_transform.get('partition_matrix',
                                                                           reference_transform['initial_matrix']))
            max_las_points = reference['recipe']['max_las_points']
            initialization = {'kind': 'reviewed_alignment_candidate', 'reference': str(reference_alignment),
                              'upstream_initialization': reference['initialization'], 'input_sha256': hashes}
        elif input_root is not None:
            initial, initialization = navigation_initialization(input_root, original)
        else:
            initial_path = Path(initial_transform).resolve()
            value = json.loads(initial_path.read_text())
            initial = validate_similarity(value['matrix'])
            initialization = {'kind': 'explicit_initial_transform', 'metadata': value,
                              'input_sha256': {str(initial_path): file_digest(initial_path)}}
        if reference is None:
            partition_matrix = initial
        target, las_metadata = sample_las(las, max_points=max_las_points)
        point_ids = np.array(sorted(original.points3D), dtype=np.uint64)
        xyz = np.array([original.points3D[int(pid)].xyz for pid in point_ids])
        quality = np.array([original.points3D[int(pid)].error <= 2.
                            and original.points3D[int(pid)].track.length() >= 4 for pid in point_ids])
        partition_all = transform_points(partition_matrix, xyz)
        lower, upper = np.array(las_metadata['bounds_min']) - .8, np.array(las_metadata['bounds_max']) + .8
        overlap = np.all((partition_all >= lower) & (partition_all <= upper), axis=1)
        eligible = quality & overlap
        source, selected_ids = xyz[eligible], point_ids[eligible]
        partition_points = partition_all[eligible]
        initial_points = transform_points(initial, source)
        split_options = ({} if reference is None else {
            'tile_size': reference['split']['tile_size'], 'fraction': reference['split']['fraction_requested'],
            'seed': reference['split']['seed']})
        holdout, tiles, split = spatial_split(partition_points, **split_options)
        if reference is not None and (
                not np.array_equal(selected_ids, reference_points['point_ids'])
                or not np.array_equal(holdout, reference_points['holdout'])
                or not np.array_equal(tiles, reference_points['tiles'])
                or not np.allclose(initial_points, reference_points['final_xyz'], atol=1e-10, rtol=0)
                or las_metadata != reference['las']):
            raise ValueError('Reference points, frozen split, transform, or LAS sample cannot be reproduced')
        if holdout.sum() < 100 or (~holdout).sum() < 100:
            raise ValueError('Insufficient fit/holdout points in the candidate LAS overlap')
        print(f'Alignment: {len(source)} eligible visual points, {len(target)} LAS samples; '
              f'{split["fit_points"]} fit / {split["holdout_points"]} spatial holdout', flush=True)
        if reference is None:
            final, optimization, surface = refine_similarity(source[~holdout], target, initial,
                                                             tiles_fit=tiles[~holdout], workers=workers)
        else:
            from geosunpre.alignment.structural import point_precision, refine_structural
            covariance, valid, observations = point_precision(original, selected_ids[~holdout])
            final, optimization, surface, fit_diagnostic = refine_structural(
                source[~holdout], target, initial, covariance, valid, tiles_fit=tiles[~holdout], workers=workers)
            fit_diagnostic.update(point_ids=selected_ids[~holdout], precision_observations=observations)
        final_points = transform_points(final, source)
        errors = {label: {stage: error_summary(points[mask], surface, workers=workers)
                          for stage, points in (('initial', initial_points), ('refined', final_points))}
                  for label, mask in (('fit', ~holdout), ('holdout', holdout))}
        common, *_ = _common_comparison(
            initial_points[holdout], final_points[holdout], surface, tiles[holdout], workers=workers)
        all_distances, all_plane, all_nearest = residuals(final_points, surface, workers=workers)
        supported = (all_distances <= .3) & surface['valid'][all_nearest]
        initial_distances, initial_plane, initial_nearest = residuals(initial_points, surface, workers=workers)
        initial_supported = (initial_distances <= .3) & surface['valid'][initial_nearest]
        normals = surface['normals'][all_nearest]
        orientations = {}
        for label, mask in [('horizontal', np.abs(normals[:, 2]) >= np.cos(np.deg2rad(25))),
                            ('vertical', np.abs(normals[:, 2]) <= np.sin(np.deg2rad(25))),
                            ('inclined', (np.abs(normals[:, 2]) > np.sin(np.deg2rad(25)))
                             & (np.abs(normals[:, 2]) < np.cos(np.deg2rad(25))))]:
            orientations[label] = {'supported_points': int((mask & supported).sum()),
                                   'holdout': quantiles(np.abs(all_plane[mask & supported & holdout]))}
        reference_comparison = None
        if reference is not None:
            fixed_normals = surface['normals'][initial_nearest]
            fixed_groups = {
                'horizontal': np.abs(fixed_normals[:, 2]) >= np.cos(np.deg2rad(25)),
                'vertical': np.abs(fixed_normals[:, 2]) <= np.sin(np.deg2rad(25)),
                'inclined': (np.abs(fixed_normals[:, 2]) > np.sin(np.deg2rad(25)))
                            & (np.abs(fixed_normals[:, 2]) < np.cos(np.deg2rad(25)))}
            reference_comparison = {'evaluation_scope': 'exploratory_reuses_previously_reviewed_data',
                                    'reference': str(reference_alignment),
                                    'orientation_groups_frozen_from_reference_correspondence': True,
                                    'groups': {}}
            for label, mask in fixed_groups.items():
                values, *_ = _common_comparison(initial_points[holdout & mask], final_points[holdout & mask],
                                                surface, tiles[holdout & mask], workers=workers)
                reference_comparison['groups'][label] = values
        transformed = pycolmap.Reconstruction(dataset / 'sparse/0')
        transformed.transform(pycolmap.Sim3d(final[:3]))
        invariance = projection_invariance(original, transformed)
        report = {
            'schema_version': 1, 'kind': 'visual_to_las_registration_diagnostic',
            'status': 'candidate_requires_review', 'absolute_accuracy_validated': False,
            'created_utc': datetime.now(timezone.utc).isoformat(),
            'inputs': {'dataset': str(dataset), 'las': str(las), 'las_sha256': las_hash,
                       'visual_model_sha256': source_hashes,
                       'colmap_export_sha256': file_digest(dataset / 'colmap_export.json'),
                       'initialization': initialization['input_sha256']},
            'implementation_sha256': {name: file_digest(Path(__file__).parent / name)
                                      for name in ('runner.py', 'geometry.py', 'las.py', 'report.py', 'publication.py')},
            'versions': {'numpy': np.__version__, 'pycolmap': pycolmap.__version__},
            'source_verification': source_verification, 'source_sfm_provenance': source_provenance,
            'initialization': initialization,
            'las': las_metadata,
            'recipe': {'max_las_points': max_las_points, 'workers': workers,
                       'voxel_sizes': [.4, .2, .1], 'correspondence_distances': [.8, .4, .2],
                       'maximum_iterations_per_scale': 12, 'surface_support_distance': .3,
                       'pixel_error_max': 2., 'track_length_min': 4,
                       'sparse_selection': 'visual quality and initial LAS header bounding box; no residual gate'},
            'visual_points': {'total': len(xyz), 'quality_selected': int(quality.sum()),
                              'quality_outside_initial_las_bounds': int((quality & ~overlap).sum()),
                              'eligible': len(source), 'model_points_removed': 0},
            'split': {**split, 'scope': 'LAS surface refinement only; initialization uses all matching camera positions'},
            'optimization': optimization, 'errors': errors,
            'holdout_common_surface_support': common, 'surface_orientations': orientations,
            'reprojection_invariance': invariance,
            'limitations': [
                '坐标单位按米假设；导航到 LAS 的坐标关系仍为待验证候选。',
                '空间留出只隔离表面细化；pose 初值使用全部对应相机位置。',
                '点到面指标仅覆盖稳定表面支持点；覆盖之外的点没有被证明配准成功。',
                'LAS 与视觉几何的一致性不代表绝对精度，也不能单独定位哪套数据存在局部误差。',
                '仅优化一个全局相似变换；未修正扫描轨迹、局部形变或内参。',
                '导出的 COLMAP 是诊断候选；图像使用链接，依赖原始视觉交付目录。'],
        }
        report['recipe']['method'] = 'surface' if reference is None else 'structural'
        report['split']['partition_matrix'] = partition_matrix.tolist()
        if reference is not None:
            report['reference_comparison'] = reference_comparison
            report['validation_scope'] = 'exploratory_reuses_previously_reviewed_data'
            report['recipe']['optimization'] = optimization['recipe']
            report['recipe']['voxel_sizes'] = [.1]
            report['recipe']['correspondence_distances'] = optimization['recipe']['correspondence_distances']
            report['recipe']['maximum_iterations_per_scale'] = optimization['recipe']['iterations_per_distance']
            report['implementation_sha256']['structural.py'] = file_digest(Path(__file__).parent / 'structural.py')
            report['limitations'].extend([
                '本轮重用已检查的旧留出区域；优化器仍只收到 fit 点，但这些对照不能作为独立精度验收。',
                '视觉点精度代理假定相机与内参固定，仅用于相对筛选和权重，不是完整 BA 协方差或真实深度精度。',
                '筛选使用几何一致性，未识别语义类别，也不能保证平行薄板来自同一侧面。'])
            if optimization['mode'] == 'ground_tilt_z_only':
                report['limitations'].append('竖直结构覆盖不足；本轮仅细化绕 LAS X/Y 轴的倾斜和局部枢轴的 Z 位移，'
                    '统一尺度及其他配准自由度由参考候选继承，没有重新验收。')
            elif optimization['mode'] == 'retain_reference':
                report['limitations'].append('可靠结构覆盖不足，未更新参考变换。')
        change = common.get('median_change_spatial_bootstrap')
        if change and change['ci95'][0] > 0:
            report['limitations'].append('留出共同支持点的 median 在细化后增加；当前细化未证明优于 pose 初值。')
        elif change and change['ci95'][1] >= 0:
            report['limitations'].append('留出共同支持点的 median 改变量区间包含零；未检出明确细化收益。')
        if not optimization['full_rank_at_all_solved_iterations']:
            dofs = optimization.get('estimated_dofs', 7)
            report['limitations'].append(f'部分尺度的表面约束未充分约束本轮释放的 {dofs} 个自由度。')
        output.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=f'.{output.name}-', dir=output.parent) as temporary:
            stage = Path(temporary)
            transformed_model = stage / 'transformed_model'
            transformed_model.mkdir()
            transformed.write(transformed_model)
            invariance = projection_invariance(original, pycolmap.Reconstruction(transformed_model))
            report['reprojection_invariance'] = invariance
            regional = _regions(stage / 'regions.csv', initial_points, final_points, tiles, holdout, surface,
                                 workers=workers)
            report['regional_summary'] = {'tiles': len(regional), 'csv': 'regions.csv'}
            report['elapsed_seconds_before_export'] = time.monotonic() - started
            write_json(stage / 'report.json', report)
            write_json(stage / 'transform.json', {
                'schema_version': 1, 'equation': 'X_las = s R X_visual + t',
                'source_frame': 'pure_visual_colmap', 'target_frame': 'input_las',
                'coordinate_units_verified': False, 'status': 'candidate_requires_review',
                'matrix': final.tolist(), 'initial_matrix': initial.tolist(),
                'partition_matrix': partition_matrix.tolist(),
                'scale': float(np.cbrt(np.linalg.det(final[:3, :3]))),
                'initial_scale': float(np.cbrt(np.linalg.det(initial[:3, :3]))),
                'report_sha256': file_digest(stage / 'report.json')})
            np.savez_compressed(stage / 'registration_points.npz', point_ids=selected_ids, tiles=tiles,
                                holdout=holdout, partition_xyz=partition_points,
                                initial_xyz=initial_points, final_xyz=final_points,
                                nearest_distance=all_distances, signed_plane_error=all_plane,
                                stable_surface_supported=supported,
                                surface_normals=surface['normals'][residuals(final_points, surface, workers=workers)[2]])
            if fit_diagnostic is not None:
                np.savez_compressed(stage / 'structural_fit_points.npz', **fit_diagnostic)
            write_ply(stage / 'las_sample.ply', surface['points'])
            write_ply(stage / 'visual_initial.ply', initial_points, (255, 200, 60))
            before = pycolmap.Reconstruction(dataset / 'sparse/0')
            before.transform(pycolmap.Sim3d(initial[:3]))
            overlays = projection_overlays(dataset, before, transformed, surface['points'], stage / 'overlays',
                initial_label='Pose initialization + LAS' if reference is None else 'Previous candidate + LAS',
                refined_label='Surface refinement + LAS' if reference is None else 'Structural refinement + LAS')
            viewer_errors = np.where(supported, np.abs(all_plane), 1.)
            viewer_errors_initial = np.where(initial_supported, np.abs(initial_plane), 1.)
            write_inspection(stage / 'inspection.html', report, target=surface['points'],
                             source_initial=initial_points, source_final=final_points, holdout=holdout,
                             errors=viewer_errors, errors_initial=viewer_errors_initial,
                             cameras=np.array([i.projection_center() for i in transformed.images.values()]),
                             cameras_initial=np.array([i.projection_center() for i in before.images.values()]),
                             overlays=overlays)
            scene = {'images': [{'name': image.name, 'source': str(dataset / 'images' / image.name)}
                                for image in original.images.values()]}
            candidate_hashes = {name: file_digest(transformed_model / name) for name in MODEL_FILES}
            export_colmap(transformed_model, scene, stage / 'aligned_colmap', image_mode='link', include_ply=True,
                          provenance={'kind': 'las_alignment_diagnostic_candidate',
                                      'source_model_sha256': candidate_hashes,
                                      'pure_visual_model_sha256': source_hashes,
                                      'alignment_report_sha256': file_digest(stage / 'report.json'),
                                      'alignment_transform_sha256': file_digest(stage / 'transform.json'),
                                      'las_sha256': las_hash, 'sfm_rerun': False,
                                      'source_sfm_provenance': source_provenance,
                                      'las_used_for_alignment': True, 'absolute_accuracy_validated': False},
                          quality_report=stage / 'report.json')
            current_hashes = {name: file_digest(dataset / 'sparse/0' / name) for name in MODEL_FILES}
            if current_hashes != source_hashes or file_digest(las) != las_hash:
                raise ValueError('Source geometry changed during alignment')
            for path, expected in initialization['input_sha256'].items():
                if file_digest(path) != expected:
                    raise ValueError('Initialization inputs changed during alignment')
            result = {'output': str(output), 'report': str(output / 'inspection.html'),
                      'status': report['status'], 'registered_images': transformed.num_reg_images(),
                      'model_points': transformed.num_points3D(), 'eligible_visual_points': len(source),
                      'holdout': errors['holdout'], 'holdout_common_surface_support': common,
                      'source_model_unchanged': True, 'elapsed_seconds': time.monotonic() - started}
            write_json(stage / 'result.json', result)
            from geosunpre.alignment.publication import complete_publication
            complete_publication(stage, context_fingerprint)
            stage.rename(output)
    return result


def refine_alignment(alignment, output, *, workers=4):
    """Start from a reviewed candidate, retaining its visual points and split."""
    alignment = Path(alignment).resolve()
    reference = json.loads((alignment / 'report.json').read_text())
    return run_alignment(reference['inputs']['dataset'], reference['inputs']['las'], output,
                         reference_alignment=alignment, workers=workers)
