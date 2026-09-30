"""Independent, reviewable fixed-intrinsics image/position/LAS BA experiments."""

from datetime import datetime, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time

import numpy as np
import pyceres
import pycolmap

from geosunpre.alignment.geometry import (error_summary, quantiles, residuals, spatial_split,
    transform_points, validate_similarity)
from geosunpre.alignment.joint_solver import (RECIPE, IMAGE_GATES, ImageObservations, assert_identity,
    geometry_gate, image_gate, interpolate, las_correspondences, mixed_objective, native_costs, solve_fixed)
from geosunpre.alignment.las import sample_las
from geosunpre.alignment.report import projection_overlays
from geosunpre.alignment.runner import _common_comparison, navigation_initialization, projection_invariance
from geosunpre.alignment.structural import coherent_las
from geosunpre.artifacts import digest_cache, file_digest, write_json
from geosunpre.camera import Camera
from geosunpre.colmap import MODEL_FILES, export_colmap, verify_colmap_export
from geosunpre.scene import load_poses


def _fresh_output(output, protected):
    if any(output.is_relative_to(path.resolve()) or path.resolve().is_relative_to(output) for path in protected):
        raise ValueError('Choose a joint BA output separate from all source data')
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError('Joint BA output already exists; choose a fresh directory')


def _positions(rec, input_root, audit):
    paths = [Path(path) for path in audit['input_sha256'] if path.endswith('_CameraPos_C2E.txt')]
    if len(paths) != 1:
        raise ValueError('Expected exactly one audited camera-position file')
    poses = load_poses(paths[0])
    names = {image.image_id: Path(image.name).name for image in rec.images.values()}
    if any(name not in poses for name in names.values()):
        raise ValueError('Joint BA requires a named input position for every registered image')
    matrix = validate_similarity(audit['navigation_to_candidate_las_matrix'])
    return {iid: transform_points(matrix, np.array([poses[name]['center_raw']]))[0]
            for iid, name in sorted(names.items())}


def _run_stage(label, base, initial_matrix, positions, surface, fit_ids, tiles_fit,
               observations, input_errors, recipe, directory):
    current, matrix = pycolmap.Reconstruction(base), initial_matrix.copy()
    steps = []
    directory.mkdir()
    for iteration in range(recipe['outer_iterations']):
        matches = las_correspondences(current, matrix, fit_ids, tiles_fit, surface,
            maximum=.4 if iteration == 0 else .25, workers=recipe['workers']) if recipe['las_weight'] else None
        errors_before, depth_before = observations.errors(current)
        objective_before = mixed_objective(current, matrix, errors_before, positions, matches, recipe)
        proposal = pycolmap.Reconstruction(current)
        print(f'Joint BA {label}: round {iteration + 1}, '
              f'{len(matches["point_ids"]) if matches is not None else 0} LAS constraints', flush=True)
        proposal_matrix, solver = solve_fixed(proposal, matrix, base, initial_matrix, positions, matches, recipe)
        print(f'Joint BA {label}: {solver["solver"]}', flush=True)
        trials, accepted = [], False
        for fraction in recipe['backtrack_fractions']:
            candidate, candidate_matrix = (proposal, proposal_matrix) if fraction == 1. else interpolate(
                current, proposal, matrix, proposal_matrix, fraction)
            candidate_errors, depths = observations.errors(candidate)
            visual = image_gate(observations, input_errors, candidate_errors)
            physical = geometry_gate(base, candidate, initial_matrix, candidate_matrix, recipe)
            depth_guard = bool(np.isfinite(depths).all() and np.all(depths >= .05 * depth_before))
            cost = mixed_objective(candidate, candidate_matrix, candidate_errors, positions, matches, recipe)
            descent = bool(np.isfinite(cost) and cost <= objective_before + 1e-6)
            passed = bool(solver['solution_usable'] and visual['passed'] and physical['passed'] and depth_guard and descent)
            trials.append({'fraction': fraction, 'passed': passed, 'image_gate': visual,
                'geometry_gate': physical, 'depth_guard_passed': depth_guard,
                'mixed_objective': cost if np.isfinite(cost) else None, 'descent': descent})
            if passed:
                current, matrix, accepted = candidate, candidate_matrix, True
                break
        step = {'round': iteration + 1, 'accepted_update': accepted, 'solver': solver,
                'correspondences': matches['summary'] if matches is not None else None,
                'mixed_objective_before': objective_before, 'backtracking': trials}
        if not accepted:
            rejected = directory / f'rejected_round_{iteration + 1}'
            rejected.mkdir(); proposal.write(rejected)
            write_json(rejected / 'global_transform.json', {'matrix': proposal_matrix.tolist()})
            step['rejected_proposal'] = rejected.name
        steps.append(step)
        write_json(directory / f'round_{iteration + 1}.json', step)
        if not accepted:
            break
    errors, depths = observations.errors(current)
    assert_identity(base, current)
    return current, matrix, errors, {'steps': steps, 'accepted_updates': sum(step['accepted_update'] for step in steps),
        'image_statistics': observations.summary(errors),
        'input_image_gate': image_gate(observations, input_errors, errors),
        'geometry_gate': geometry_gate(base, current, initial_matrix, matrix, recipe),
        'invalid_depth_observations': int((~np.isfinite(depths) | (depths <= 1e-9)).sum())}


def _laser_summary(source_points, candidate_points, surface, holdout, tiles, workers):
    initial_distances, _, nearest = residuals(source_points, surface, workers=workers)
    normals = surface['normals'][nearest]
    groups = {'horizontal': np.abs(normals[:, 2]) >= np.cos(np.deg2rad(25)),
              'vertical': np.abs(normals[:, 2]) <= np.sin(np.deg2rad(25)),
              'inclined': (np.abs(normals[:, 2]) > np.sin(np.deg2rad(25)))
                          & (np.abs(normals[:, 2]) < np.cos(np.deg2rad(25)))}
    result = {'fit': error_summary(candidate_points[~holdout], surface, workers=workers),
              'las_holdout': error_summary(candidate_points[holdout], surface, workers=workers),
              'fixed_reference_orientation_groups': {},
              'scope': 'Exploratory old LAS holdout; its image observations participate in all BA stages'}
    result['common_support'], *_ = _common_comparison(source_points[holdout], candidate_points[holdout],
                                                    surface, tiles[holdout], workers=workers)
    for name, mask in groups.items():
        comparison, *_ = _common_comparison(source_points[holdout & mask], candidate_points[holdout & mask],
                                            surface, tiles[holdout & mask], workers=workers)
        result['fixed_reference_orientation_groups'][name] = comparison
    return result


def _scene(dataset, rec, identity):
    cameras = {cid: Camera('L' if cid == 1 else 'R', cam.width, cam.height, tuple(cam.params), (0, 0))
               for cid, cam in rec.cameras.items()}
    records = []
    for image in rec.images.values():
        record = {'name': image.name, 'source': str(dataset / 'images' / image.name), 'camera': str(image.camera_id)}
        if identity['include_masks']:
            record['combined_mask'] = str(dataset / 'masks' / (image.name + '.png'))
        records.append(record)
    return {'images': records, 'cameras': {str(cid): cam.to_dict() for cid, cam in cameras.items()}, 'fov_degrees': 160}


def run_joint_alignment(alignment, input_root, output, *, workers=4, outer_iterations=2,
                        iterations=20, pose_std=.5, las_std=.05, pose_weight=.001, las_weight=.005,
                        mode='comparison', context_fingerprint=None):
    """Build pure BA / position-prior BA / position+LAS BA from the same initial model."""
    alignment, input_root, output = Path(alignment).resolve(), Path(input_root).resolve(), Path(output).resolve()
    _fresh_output(output, [alignment, input_root])
    if mode not in ('comparison', 'production'):
        raise ValueError('Joint BA mode must be comparison or production')
    if (not 1 <= workers <= 16 or not 1 <= outer_iterations <= 5 or not 1 <= iterations <= 100
            or not .1 <= pose_std <= 10 or not .01 <= las_std <= 1
            or not 0 < pose_weight <= .1 or not 0 < las_weight <= .1):
        raise ValueError('Invalid joint BA iteration, worker, noise or weight limits')
    native = native_costs()
    started = time.monotonic()
    recipe = {**RECIPE, 'workers': workers, 'outer_iterations': outer_iterations, 'iterations_per_solve': iterations,
              'pose_std_m': pose_std, 'las_std_m': las_std, 'pose_weight': pose_weight, 'las_weight': las_weight}
    with digest_cache():
        reference_hashes = {name: file_digest(alignment / name)
                            for name in ('report.json', 'transform.json', 'registration_points.npz')}
        reference = json.loads((alignment / 'report.json').read_text())
        transform = json.loads((alignment / 'transform.json').read_text())
        if (reference.get('kind') != 'visual_to_las_registration_diagnostic'
                or transform['report_sha256'] != reference_hashes['report.json']):
            raise ValueError('Reference alignment report/transform identity mismatch')
        dataset, las = Path(reference['inputs']['dataset']), Path(reference['inputs']['las'])
        _fresh_output(output, [alignment, input_root, dataset, las])
        source_verification = verify_colmap_export(dataset)
        identity = json.loads((dataset / 'colmap_export.json').read_text())
        source_hashes = {name: file_digest(dataset / 'sparse/0' / name) for name in MODEL_FILES}
        if (source_hashes != reference['inputs']['visual_model_sha256']
                or file_digest(dataset / 'colmap_export.json') != reference['inputs']['colmap_export_sha256']
                or file_digest(las) != reference['inputs']['las_sha256']):
            raise ValueError('Source data changed since reference alignment')
        initial_matrix = validate_similarity(transform['matrix'])
        base = pycolmap.Reconstruction(dataset / 'sparse/0')
        _, audit = navigation_initialization(input_root, base)
        positions = _positions(base, input_root, audit)
        position_centres = np.array(list(positions.values()))
        singular = np.linalg.svd(position_centres - position_centres.mean(0), compute_uv=False)
        if len(positions) < 3 or singular[1] < singular[0] * 1e-4:
            raise ValueError('Position priors do not constrain the global Sim3; a noncollinear trajectory is required')
        with np.load(alignment / 'registration_points.npz', allow_pickle=False) as archive:
            points = {name: archive[name] for name in archive.files}
        ids, holdout, tiles = points['point_ids'], points['holdout'], points['tiles']
        split = reference['split']
        partition_xyz = points.get('partition_xyz', points['initial_xyz'])
        partition_matrix = validate_similarity(transform.get('partition_matrix', transform['initial_matrix']))
        repeated_holdout, repeated_tiles, _ = spatial_split(partition_xyz, tile_size=split['tile_size'],
            fraction=split['fraction_requested'], seed=split['seed'])
        source = np.array([base.points3D[int(pid)].xyz for pid in ids])
        initial_points = transform_points(initial_matrix, source)
        if (not np.array_equal(holdout, repeated_holdout) or not np.array_equal(tiles, repeated_tiles)
                or len(np.unique(ids)) != reference['visual_points']['eligible']
                or not np.allclose(transform_points(partition_matrix, source), partition_xyz, atol=1e-10, rtol=0)
                or not np.allclose(initial_points, points['final_xyz'], atol=1e-10, rtol=0)):
            raise ValueError('Reference source points, transform or frozen split do not replay')
        target, metadata = sample_las(las, max_points=reference['recipe']['max_las_points'], seed=reference['las']['sample_seed'])
        if metadata != reference['las']:
            raise ValueError('LAS sample changed since reference alignment')
        surface = coherent_las(target, workers=workers)
        distances, plane, nearest = residuals(initial_points, surface, workers=workers)
        if (not np.allclose(distances, points['nearest_distance'], atol=1e-10, rtol=0)
                or not np.allclose(plane, points['signed_plane_error'], atol=1e-10, rtol=0)
                or not np.array_equal((distances <= .3) & surface['valid'][nearest], points['stable_surface_supported'])):
            raise ValueError('Reference surface support/residuals no longer replay')
        observations = ImageObservations(base, initial_matrix)
        input_errors, depths = observations.errors(base)
        if not np.isfinite(input_errors).all() or np.any(depths <= 1e-9):
            raise ValueError('Source visual model has nonphysical or nonfinite projections')
        package = Path(__file__).parent
        implementation = {name: file_digest(package / name) for name in
            ('joint.py', 'joint_solver.py', 'joint_report.py', 'structural.py', 'geometry.py', 'runner.py',
             'las.py', 'report.py', 'publication.py')}
        implementation['native_module'] = file_digest(native.__file__)
        implementation['colmap.py'] = file_digest(package.parent / 'colmap.py')
        output.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=f'.{output.name}-', dir=output.parent) as temporary:
            stage = Path(temporary)
            (stage / 'stages').mkdir()
            write_json(stage / 'protocol.json', {'recipe': recipe, 'image_gates': IMAGE_GATES,
                'reference_sha256': reference_hashes, 'source_hashes': source_hashes,
                'initial_global_matrix': initial_matrix.tolist(), 'implementation_sha256': implementation,
                'scope': 'Fixed-intrinsics exploratory post-SfM BA; no new independent image or LAS validation'})
            stages = {}
            states = {}
            pure_errors = None
            scene = _scene(dataset, base, identity)
            branches = [('pure_image_ba', False, False), ('pose_las', True, True)]
            if mode == 'comparison':
                branches.insert(1, ('pose_only', True, False))
            for label, use_pose, use_las in branches:
                selected_recipe = {**recipe, 'pose_weight': pose_weight if use_pose else 0., 'las_weight': las_weight if use_las else 0.}
                directory = stage / 'stages' / label
                rec, matrix, errors, diagnostic = _run_stage(label, base, initial_matrix, positions, surface,
                    ids[~holdout], tiles[~holdout], observations, input_errors, selected_recipe, directory)
                if pure_errors is None:
                    pure_errors = errors.copy()
                diagnostic['image_gate_vs_pure_ba'] = image_gate(observations, pure_errors, errors)
                diagnostic['accepted_for_candidate_review'] = bool(diagnostic['input_image_gate']['passed']
                    and diagnostic['image_gate_vs_pure_ba']['passed'] and diagnostic['geometry_gate']['passed']
                    and diagnostic['accepted_updates'] > 0 and diagnostic['invalid_depth_observations'] == 0)
                diagnostic['recipe'] = selected_recipe
                candidate_points = transform_points(matrix, np.array([rec.points3D[int(pid)].xyz for pid in ids]))
                diagnostic['las'] = _laser_summary(initial_points, candidate_points, surface, holdout, tiles, workers)
                centres = np.array([rec.images[iid].projection_center() for iid in positions])
                diagnostic['input_position_residual_m'] = quantiles(np.linalg.norm(
                    transform_points(matrix, centres) - np.array(list(positions.values())), axis=1))
                diagnostic['global_matrix'] = matrix.tolist()
                visual = directory / 'visual_model'; visual.mkdir(); rec.write(visual)
                aligned = pycolmap.Reconstruction(rec); aligned.transform(pycolmap.Sim3d(matrix[:3]))
                aligned_model = directory / 'aligned_model'; aligned_model.mkdir(); aligned.write(aligned_model)
                diagnostic['global_transform_projection_invariance'] = projection_invariance(rec, aligned)
                write_json(directory / 'diagnostic.json', diagnostic)
                np.savez_compressed(directory / 'image_observations.npz', point_ids=observations.point_ids,
                    image_ids=observations.image_ids, point2D_indices=observations.feature_ids,
                    camera_ids=observations.camera_ids, zone=observations.zones, tiles=observations.tiles,
                    observed_pixels=observations.pixels, input_error=input_errors, candidate_error=errors,
                    pure_ba_error=pure_errors)
                distances, plane, nearest = residuals(candidate_points, surface, workers=workers)
                supported = surface['valid'][nearest] & (distances <= .3)
                np.savez_compressed(directory / 'las_points.npz', point_ids=ids, holdout=holdout, tiles=tiles,
                                    initial_xyz=initial_points, final_xyz=candidate_points,
                                    nearest_distance=distances, signed_plane_error=plane,
                                    stable_surface_supported=supported)
                candidate_hashes = {name: file_digest(aligned_model / name) for name in MODEL_FILES}
                export_colmap(aligned_model, scene, directory / 'colmap', image_mode='link',
                    include_masks=identity['include_masks'], include_ply=True, quality_report=directory / 'diagnostic.json',
                    provenance={'kind': 'fixed_intrinsics_joint_ba_exploratory_candidate', 'stage': label,
                        'source_model_sha256': candidate_hashes, 'pure_visual_model_sha256': source_hashes,
                        'reference_alignment_sha256': reference_hashes,
                        'joint_ba_diagnostic_sha256': file_digest(directory / 'diagnostic.json'),
                        'las_sha256': reference['inputs']['las_sha256'], 'source_sfm_provenance': identity.get('provenance', {}),
                        'pose_values_used_for_ba': use_pose, 'las_used_for_ba': use_las,
                        'intrinsics_optimized': False, 'accepted_image_gate': diagnostic['accepted_for_candidate_review'],
                        'absolute_accuracy_validated': False, 'independent_image_validation': False})
                exported_identity = json.loads((directory / 'colmap' / 'colmap_export.json').read_text())
                if ({row['name']: row for row in exported_identity['images']} !=
                        {row['name']: row for row in identity['images']}):
                    raise ValueError('Candidate export changed original image or combined-mask content')
                stages[label], states[label] = diagnostic, (rec, matrix, aligned)
            reference_aligned = pycolmap.Reconstruction(base); reference_aligned.transform(pycolmap.Sim3d(initial_matrix[:3]))
            overlays = projection_overlays(dataset, reference_aligned, states['pose_las'][2], surface['points'], stage / 'overlays',
                initial_label='Reference visual model + LAS', refined_label='Fixed-intrinsics joint BA + LAS')
            report = {'schema_version': 1, 'kind': 'fixed_intrinsics_joint_ba_exploration',
                'status': 'candidate_image_gate_passed' if stages['pose_las']['accepted_for_candidate_review'] else 'candidate_rejected',
                'created_utc': datetime.now(timezone.utc).isoformat(), 'absolute_accuracy_validated': False,
                'independent_image_validation': False, 'training_run': False,
                'mode': mode, 'reference_alignment': str(alignment), 'reference_sha256': reference_hashes,
                'inputs': reference['inputs'], 'source_verification': source_verification,
                'pose_audit': audit, 'position_prior_trajectory_singular_values': singular.tolist(),
                'recipe': recipe, 'image_gates': IMAGE_GATES, 'implementation_sha256': implementation,
                'versions': {'pycolmap': pycolmap.__version__, 'numpy': np.__version__,
                    'native_implementation': native.implementation_version, 'ceres': pyceres.__ceres_version__},
                'reference_image_statistics': observations.summary(input_errors),
                'reference_las_holdout': error_summary(initial_points[holdout], surface, workers=workers),
                'reference_position_residual_m': quantiles(np.linalg.norm(
                    transform_points(initial_matrix, np.array([base.images[iid].projection_center() for iid in positions]))
                    - np.array(list(positions.values())), axis=1)),
                'image_names': {str(iid): image.name for iid, image in base.images.items()},
                'stages': stages, 'overlays': overlays,
                'visual_points': len(base.points3D), 'registered_images': len(base.images),
                'visual_observations': len(observations.point_ids), 'las_fit_points': int((~holdout).sum()),
                'las_holdout_points': int(holdout.sum()), 'source_model_unchanged': True,
                'limitations': [
                    '图像接受检查来自源模型已参与拟合的特征观测，不是独立图像验证，也不保证 3DGS 渲染质量。',
                    'LAS 原有留出区域已经用于方案分析；其特征观测仍参与 BA，仅排除直接 LAS 残差，属于探索性对照。',
                    '坐标单位、LAS 原点与厂家坐标约定继承参考候选，未验收绝对精度。',
                    '位置先验仅使用图像名对应的相机中心；没有使用未经核实的姿态四元数、修改时间戳或强制 L/R 同步。',
                    'pose/LAS 的噪声尺度为保守代理，不能把二者当作独立的高精度真值。',
                    '可靠平面与精度代理只用于筛选和相对权重，不能保证薄板来自同一侧，也不是完整 BA 协方差。',
                    '固定 LAS 点和全部相机内参；本轮没有修正扫描运动或 LAS 局部形变。',
                    '导出仅为独立候选，没有替换默认纯图像模型；图像链接依赖原始交付目录。'],
            }
            write_json(stage / 'report.json', report)
            from geosunpre.alignment.joint_report import write_joint_report
            write_joint_report(stage / 'inspection.html', report)
            if ({name: file_digest(dataset / 'sparse/0' / name) for name in MODEL_FILES} != source_hashes
                    or file_digest(las) != reference['inputs']['las_sha256']
                    or file_digest(dataset / 'colmap_export.json') != reference['inputs']['colmap_export_sha256']):
                raise ValueError('Source geometry changed during joint BA')
            for row in identity['images']:
                if (file_digest(dataset / 'images' / row['name']) != row['sha256']
                        or identity['include_masks'] and file_digest(dataset / 'masks' / (row['name'] + '.png')) != row['mask_sha256']):
                    raise ValueError('Source image or combined mask changed during joint BA')
            for name, expected in reference_hashes.items():
                if file_digest(alignment / name) != expected:
                    raise ValueError('Reference alignment changed during joint BA')
            for path, expected in audit['input_sha256'].items():
                if file_digest(path) != expected:
                    raise ValueError('Input position data changed during joint BA')
            for name, expected in implementation.items():
                path = (Path(native.__file__) if name == 'native_module' else
                        package.parent / 'colmap.py' if name == 'colmap.py' else package / name)
                if file_digest(path) != expected:
                    raise ValueError('Implementation changed during joint BA')
            result = {'output': str(output), 'report': str(output / 'inspection.html'), 'status': report['status'],
                'registered_images': len(base.images), 'model_points': len(base.points3D),
                'source_model_unchanged': True, 'intrinsics_optimized': False, 'training_run': False,
                'stages': {name: {'accepted_image_gate': value['accepted_for_candidate_review'],
                    'image': value['image_statistics']['all'], 'las_holdout': value['las']['las_holdout'],
                    'position_residual': value['input_position_residual_m']} for name, value in stages.items()},
                'elapsed_seconds': time.monotonic() - started}
            write_json(stage / 'result.json', result)
            from geosunpre.alignment.publication import complete_publication
            complete_publication(stage, context_fingerprint)
            stage.rename(output)
    return result
