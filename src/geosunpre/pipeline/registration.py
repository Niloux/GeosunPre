"""Mandatory LAS-frame delivery with guarded joint BA and a registered fallback."""

import html
import json
from pathlib import Path

import numpy as np
import pycolmap

from geosunpre.alignment.geometry import quantiles, validate_similarity
from geosunpre.alignment.joint import _scene, run_joint_alignment
from geosunpre.alignment.joint_solver import IMAGE_GATES, RECIPE, JointConstraintsUnavailable, assert_identity
from geosunpre.alignment.las import read_header
from geosunpre.alignment.publication import complete_publication, verify_publication
from geosunpre.alignment.runner import run_alignment
from geosunpre.artifacts import (archive_interrupted, digest, file_digest, staged_directory,
                                write_frozen_json, write_json)
from geosunpre.colmap import MODEL_FILES, export_colmap, verify_colmap_export
from geosunpre.config import load_config
from geosunpre.scene import load_scene


ROOT = Path('registration')
POLICY = {
    'minimum_supported_fit_points': 100,
    'minimum_supported_fit_tiles': 5,
    'minimum_regional_points': 30,
    'maximum_regional_p90_m': .5,
    'maximum_support_fraction_drop': .02,
    'maximum_paired_median_increase_m': .005,
    'maximum_paired_p90_increase_m': .02,
    'minimum_common_support_fraction': .9,
    'scope': 'Operational fit-region checks; LAS holdout is descriptive only, not used to select output.',
}


def _read(path):
    return json.loads(Path(path).read_text())


def _hashes(model):
    return {name: file_digest(Path(model) / name) for name in MODEL_FILES}


def freeze_inputs(work):
    """Record LAS and the vendor pose-frame assumption before expensive pending work."""
    work = Path(work).resolve()
    config = load_config(work)
    if config.registration_mode != 'joint':
        raise ValueError('LAS registration is disabled for this frozen task')
    scene = _read(work / 'manifest.json')
    input_root = Path(scene['input_root']).resolve()
    if config.registration_las:
        requested = Path(config.registration_las).expanduser()
        las = (requested if requested.is_absolute() else input_root / requested).resolve()
    else:
        files = sorted((input_root / 'ResultOut').glob('*.las'))
        if len(files) != 1:
            raise ValueError('Joint pipeline requires exactly one ResultOut/*.las; set registration_las explicitly')
        las = files[0].resolve()
    if not las.is_file() or las.is_relative_to(work):
        raise ValueError('LAS input must be an existing file outside the work directory')
    poses = sorted(input_root.glob('*_CameraPos_C2E.txt'))
    if len(poses) != 1:
        raise ValueError('Joint pipeline requires exactly one CameraPos_C2E file')
    stem = poses[0].name.removesuffix('_CameraPos_C2E.txt')
    inputs = [las, poses[0], input_root / f'{stem}_pos.txt', input_root / f'{stem}_pos_ENH.txt']
    if any(not path.is_file() for path in inputs):
        raise ValueError('Joint pipeline requires CameraPos_C2E, pos and pos_ENH navigation files')
    value = {'version': 1, 'input_root': str(input_root), 'las': str(las),
             'input_sha256': {str(path): file_digest(path) for path in inputs},
             'las_header': read_header(las),
             'settings': {key: val for key, val in vars(config).items() if key.startswith('registration_')},
             'pose_coordinate_convention': {
                 'name': config.registration_pose_frame,
                 'hypothesis': 'LAS XYZ approximately equals ENH minus first ENH navigation position',
                 'units_assumed': 'metres', 'absolute_crs_verified': False,
                 'timestamps_modified': False, 'raw_quaternions_used': False},
             'image_gates': IMAGE_GATES, 'geometry_policy': POLICY}
    value['fingerprint'] = digest(value)
    write_frozen_json(work / ROOT / 'inputs.json', value)
    return value


def _points(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def surface_acceptance(points, config):
    """Check supported surfaces and fixed spatial regions without trimming residuals."""
    fit = ~points['holdout']
    supported = points['stable_surface_supported'] & fit
    plane = np.abs(points['signed_plane_error'])
    errors = []
    count, support = int(fit.sum()), int(supported.sum())
    fraction = support / count if count else 0.
    values = quantiles(plane[supported])
    tile_count = len(np.unique(points['tiles'][supported], axis=0))
    if not np.isfinite(plane).all() or not np.isfinite(points['nearest_distance']).all():
        errors.append('Nonfinite registration residuals')
    if support < POLICY['minimum_supported_fit_points'] or tile_count < POLICY['minimum_supported_fit_tiles']:
        errors.append('Insufficient stable surface support or spatial coverage')
    if fraction < config.registration_min_support_fraction:
        errors.append('Stable surface support fraction below configured minimum')
    for metric, limit in [('median', config.registration_max_median_m), ('p90', config.registration_max_p90_m)]:
        if values[metric] is None or not np.isfinite(values[metric]) or values[metric] > limit:
            errors.append(f'Supported plane {metric} exceeds configured limit')
    regions = []
    for tile in np.unique(points['tiles'][fit], axis=0):
        selected = supported & np.all(points['tiles'] == tile, axis=1)
        if selected.sum() < POLICY['minimum_regional_points']:
            continue
        summary = quantiles(plane[selected])
        regions.append({'tile': tile.tolist(), **summary})
        if summary['p90'] is None or not np.isfinite(summary['p90']) or summary['p90'] > POLICY['maximum_regional_p90_m']:
            errors.append(f'Regional plane P90 exceeds limit: {tile.tolist()}')
    return {'passed': not errors, 'failures': errors, 'fit_points': count,
            'supported_fit_points': support, 'supported_fit_fraction': fraction,
            'supported_fit_tiles': tile_count, 'point_to_plane_m': values, 'regions': regions,
            'limits': {'support_fraction': config.registration_min_support_fraction,
                       'median_m': config.registration_max_median_m, 'p90_m': config.registration_max_p90_m,
                       **POLICY}, 'scope': POLICY['scope']}


def geometric_regression(reference, candidate):
    """Compare fixed fit-point identities on common supported surfaces and regions."""
    for key in ('point_ids', 'holdout', 'tiles'):
        if not np.array_equal(reference[key], candidate[key]):
            raise ValueError('Joint candidate changed frozen LAS point identities or partition')
    fit = ~reference['holdout']
    before = reference['stable_surface_supported'] & fit
    after = candidate['stable_surface_supported'] & fit
    common = before & after
    old, new = np.abs(reference['signed_plane_error']), np.abs(candidate['signed_plane_error'])
    failures, comparisons = [], []
    denominator = int(fit.sum())
    if int(after.sum()) / denominator < int(before.sum()) / denominator - POLICY['maximum_support_fraction_drop']:
        failures.append('Joint update reduces stable surface coverage')
    if common.sum() < before.sum() * POLICY['minimum_common_support_fraction']:
        failures.append('Joint update loses too much reference surface support')
    groups = [('all', fit)]
    normals = reference['surface_normals']
    groups += [('horizontal', np.abs(normals[:, 2]) >= np.cos(np.deg2rad(25))),
               ('vertical', np.abs(normals[:, 2]) <= np.sin(np.deg2rad(25)))]
    groups += [('tile/' + ','.join(map(str, tile)), np.all(reference['tiles'] == tile, axis=1))
               for tile in np.unique(reference['tiles'][fit], axis=0)]
    for name, mask in groups:
        selected = common & mask
        if selected.sum() < POLICY['minimum_regional_points']:
            continue
        a, b = quantiles(old[selected]), quantiles(new[selected])
        changes = {metric: b[metric] - a[metric] for metric in ('median', 'p90')}
        comparisons.append({'group': name, 'points': int(selected.sum()), 'before_m': a, 'after_m': b,
                            'changes_m': changes})
        if (changes['median'] > POLICY['maximum_paired_median_increase_m']
                or changes['p90'] > POLICY['maximum_paired_p90_increase_m']):
            failures.append(f'Joint update increases supported plane errors: {name}')
    return {'passed': not failures, 'failures': failures, 'common_fit_support_points': int(common.sum()),
            'comparisons': comparisons, 'scope': POLICY['scope']}


def select_candidate(reference_points, candidate_points, diagnostic, config):
    base = surface_acceptance(reference_points, config)
    if not base['passed']:
        raise ValueError('LAS registration acceptance rejected: ' + '; '.join(base['failures']))
    candidate = surface_acceptance(candidate_points, config) if candidate_points is not None else None
    regression = geometric_regression(reference_points, candidate_points) if candidate and candidate['passed'] else None
    accepted = bool(diagnostic and diagnostic['accepted_for_candidate_review'] and candidate['passed']
                    and regression['passed'])
    reasons = []
    if not diagnostic or not diagnostic.get('accepted_for_candidate_review'):
        reasons.append('Joint BA did not pass all image, geometry and update checks')
    if candidate and not candidate['passed']:
        reasons.extend(candidate['failures'])
    if regression and not regression['passed']:
        reasons.extend(regression['failures'])
    return {'selected': 'pose_las' if accepted else 'global_registration',
            'joint_accepted': accepted, 'fallback_reasons': reasons if not accepted else [],
            'reference_acceptance': base, 'joint_acceptance': candidate, 'joint_regression': regression,
            'target_frame': 'input_las_xyz', 'coordinate_units_assumed': 'metres',
            'absolute_accuracy_validated': False, 'training_run': False}


def _stage(work, name, context, compute):
    root = Path(work) / ROOT
    fingerprint = digest({'name': name, 'protocol_fingerprint': context})
    path = root / name
    if path.exists() and (path / 'completion.json').is_file():
        verify_publication(path, fingerprint)
        return path
    archive_interrupted(path)
    compute(path, fingerprint)
    verify_publication(path, fingerprint)
    return path


def register(work):
    """Publish exactly one accepted LAS-frame model, or reject the pipeline."""
    from geosunpre.pipeline.runner import FINAL, _check_model, _export_visual_result
    from geosunpre.pipeline.quality import production_acceptance, production_quality
    from geosunpre.pipeline.observations import is_validation, partitions
    from geosunpre.sfm.reconstruction import assert_holdout_isolated

    work = Path(work).resolve()
    config = load_config(work)
    inputs = freeze_inputs(work)
    root = work / ROOT
    source = root / 'visual_colmap'
    _export_visual_result(work, output=source, image_mode='link', include_masks=True, include_ply=True)
    protocol = {'version': 1, 'inputs_fingerprint': inputs['fingerprint'],
                'source_model_sha256': _hashes(source / 'sparse/0'),
                'source_export_sha256': file_digest(source / 'colmap_export.json'),
                'solver_recipe': {**RECIPE, 'workers': config.registration_workers,
                    'outer_iterations': config.registration_outer_iterations,
                    'iterations_per_solve': config.registration_ba_iterations,
                    'pose_std_m': config.registration_pose_std_m, 'las_std_m': config.registration_las_std_m,
                    'pose_weight': config.registration_pose_weight, 'las_weight': config.registration_las_weight},
                'image_gates': IMAGE_GATES, 'geometry_policy': POLICY,
                'mode': 'production_joint_with_pure_ba_control', 'intrinsics_optimized': False}
    protocol['fingerprint'] = digest(protocol)
    write_frozen_json(root / 'protocol.json', protocol)
    final = root / 'final'
    final_context = digest({'name': 'final', 'protocol_fingerprint': protocol['fingerprint']})
    if (final / 'completion.json').is_file():
        verify_registered(work)
        return _read(final / 'result.json')
    global_stage = _stage(work, 'global', protocol['fingerprint'], lambda path, fingerprint:
        run_alignment(source, inputs['las'], path, input_root=inputs['input_root'],
                      max_las_points=config.registration_max_las_points, workers=config.registration_workers,
                      context_fingerprint=fingerprint))
    reference = _points(global_stage / 'registration_points.npz')
    base_acceptance = surface_acceptance(reference, config)
    if not base_acceptance['passed']:
        write_json(root / 'rejection.json', {'stage': 'global', 'acceptance': base_acceptance})
        raise ValueError('LAS registration acceptance rejected: ' + '; '.join(base_acceptance['failures']))

    def compute_joint(path, fingerprint):
        try:
            run_joint_alignment(global_stage, inputs['input_root'], path, mode='production',
                workers=config.registration_workers, outer_iterations=config.registration_outer_iterations,
                iterations=config.registration_ba_iterations, pose_std=config.registration_pose_std_m,
                las_std=config.registration_las_std_m, pose_weight=config.registration_pose_weight,
                las_weight=config.registration_las_weight, context_fingerprint=fingerprint)
        except JointConstraintsUnavailable as exc:
            with staged_directory(path) as stage:
                write_json(stage / 'unavailable.json', {'error': str(exc), 'registered_fallback_available': True})
                complete_publication(stage, fingerprint)

    joint = _stage(work, 'joint', protocol['fingerprint'], compute_joint)
    unavailable = (joint / 'unavailable.json').is_file()
    candidate = joint / 'stages/pose_las'
    diagnostic = None if unavailable else _read(candidate / 'diagnostic.json')
    decision = select_candidate(reference, None if unavailable else _points(candidate / 'las_points.npz'),
                                diagnostic, config)
    if unavailable:
        decision['fallback_reasons'].append(_read(joint / 'unavailable.json')['error'])
    selected = candidate / 'aligned_model' if decision['joint_accepted'] else global_stage / 'transformed_model'
    rec = pycolmap.Reconstruction(selected)
    base = pycolmap.Reconstruction(work / FINAL / 'sparse/0')
    assert_identity(base, rec)
    scene = load_scene(work / 'sfm')
    _check_model(rec, scene)
    split, triangle = partitions(work)
    assert_holdout_isolated(rec, split)
    quality = {'model_sha256': _hashes(selected), **production_quality(rec), 'registration': decision,
               'camera_parameter_bounds': _read(work / FINAL / 'result.json')['ba'].get('cameras', {})}
    quality['acceptance'] = production_acceptance(quality, config)
    if is_validation(work):
        quality.update(mode='validation', observations_excluded_for_validation=None)
    with staged_directory(final) as stage:
        model = stage / 'sparse/0'
        model.mkdir(parents=True)
        rec.write(model)
        hashes = _hashes(model)
        quality['model_sha256'] = hashes
        if is_validation(work):
            from geosunpre.pipeline.reconstruction import freeze_geometry
            from geosunpre.sfm.evaluation import evaluate_reconstruction
            _, _, _, features, _, _ = freeze_geometry(work, require_existing=True)
            validation = {'model_sha256': hashes,
                'raw': evaluate_reconstruction(rec, scene, features, split, stage / 'validation'),
                'triangle': evaluate_reconstruction(rec, scene, features, triangle, stage / 'triangle_validation')}
            write_json(stage / 'validation_summary.json', validation)
        result = {'version': 1, 'status': 'registered', 'model': str(final / 'sparse/0'),
                  'selected': decision['selected'], 'joint_accepted': decision['joint_accepted'],
                  'fallback_reasons': decision['fallback_reasons'], 'target_frame': decision['target_frame'],
                  'model_sha256': hashes, 'protocol_fingerprint': protocol['fingerprint'],
                  'inputs_fingerprint': inputs['fingerprint'], 'registered_images': rec.num_reg_images(),
                  'points': rec.num_points3D(), 'observations': rec.compute_num_observations(),
                  'global_completion_sha256': file_digest(global_stage / 'completion.json'),
                  'joint_completion_sha256': file_digest(joint / 'completion.json'),
                  'absolute_accuracy_validated': False, 'intrinsics_optimized': False, 'training_run': False}
        write_json(stage / 'quality_summary.json', quality)
        write_json(stage / 'result.json', result)
        _write_report(stage / 'inspection.html', result, decision, unavailable)
        complete_publication(stage, final_context)
    freeze_inputs(work)
    return verify_registered(work)


def verify_registered(work):
    """Recheck completed stages and current sources before export or a resumed run."""
    work = Path(work).resolve()
    inputs = freeze_inputs(work)
    root = work / ROOT
    protocol = _read(root / 'protocol.json')
    if protocol.get('fingerprint') != digest({k: v for k, v in protocol.items() if k != 'fingerprint'}):
        raise ValueError('Registration protocol changed')
    if (protocol['inputs_fingerprint'] != inputs['fingerprint']
            or protocol['source_export_sha256'] != file_digest(root / 'visual_colmap/colmap_export.json')
            or protocol['source_model_sha256'] != _hashes(root / 'visual_colmap/sparse/0')):
        raise ValueError('Registration inputs or source visual model changed')
    from geosunpre.pipeline.runner import FINAL
    if protocol['source_model_sha256'] != _hashes(work / FINAL / 'sparse/0'):
        raise ValueError('Registered model belongs to another visual final model')
    verify_colmap_export(root / 'visual_colmap')
    for name in ('global', 'joint', 'final'):
        context = digest({'name': name, 'protocol_fingerprint': protocol['fingerprint']})
        verify_publication(root / name, context)
    result = _read(root / 'final/result.json')
    if (result['protocol_fingerprint'] != protocol['fingerprint']
            or result['inputs_fingerprint'] != inputs['fingerprint']
            or result['model_sha256'] != _hashes(root / 'final/sparse/0')
            or result['global_completion_sha256'] != file_digest(root / 'global/completion.json')
            or result['joint_completion_sha256'] != file_digest(root / 'joint/completion.json')
            or result['target_frame'] != 'input_las_xyz'):
        raise ValueError('Final registration identity changed')
    return result


def export_registered(work, *, output=None, image_mode='copy', include_masks=False, include_ply=False):
    from geosunpre.pipeline.runner import _check_model
    from geosunpre.pipeline.reconstruction import freeze_geometry
    from geosunpre.pipeline.observations import is_validation, partitions
    from geosunpre.pipeline.quality import production_acceptance
    from geosunpre.sfm.reconstruction import assert_holdout_isolated

    work = Path(work).resolve()
    result = verify_registered(work)
    root = work / ROOT
    final = root / 'final'
    _, visual_scene, *_ = freeze_geometry(work, require_existing=True)
    rec = pycolmap.Reconstruction(final / 'sparse/0')
    _check_model(rec, visual_scene)
    assert_holdout_isolated(rec, partitions(work)[0])
    quality = _read(final / 'quality_summary.json')
    production_acceptance(quality, load_config(work))
    source = root / 'visual_colmap'
    identity = _read(source / 'colmap_export.json')
    scene = _scene(source, rec, identity)
    output = Path(output).resolve() if output else work / 'colmap'
    if output == work or output.is_relative_to(root) or output.is_relative_to(work / 'sfm'):
        raise ValueError('Choose a COLMAP export separate from pipeline intermediate models')
    provenance = {**identity['provenance'], 'kind': 'registered_sfm_pipeline',
        'source_model_sha256': result['model_sha256'], 'source_visual_model_sha256': identity['model_sha256'],
        'registration_inputs_sha256': file_digest(root / 'inputs.json'),
        'registration_protocol_sha256': file_digest(root / 'protocol.json'),
        'registration_result_sha256': file_digest(final / 'result.json'),
        'registration_quality_sha256': file_digest(final / 'quality_summary.json'),
        'las_sha256': _read(root / 'inputs.json')['input_sha256'][_read(root / 'inputs.json')['las']],
        'target_frame': 'input_las_xyz', 'registration_selected': result['selected'],
        'pose_used_for_registration': True, 'las_used_for_registration': True,
        'pose_values_used_for_joint_ba': result['joint_accepted'], 'las_used_for_joint_ba': result['joint_accepted'],
        'navigation_pose_values_or_las_used': True, 'intrinsics_optimized_during_registration': False,
        'absolute_accuracy_validated': False, 'independent_image_validation': False,
        'registration_scope': POLICY['scope']}
    if is_validation(work):
        provenance['registered_validation_sha256'] = file_digest(final / 'validation_summary.json')
    return export_colmap(final / 'sparse/0', scene, output, image_mode=image_mode,
                         include_masks=include_masks, include_ply=include_ply, provenance=provenance,
                         quality_report=final / 'quality_summary.json')


def _write_report(path, result, decision, unavailable):
    selected = '图像 + pose + LAS 联合结果' if result['joint_accepted'] else '通过验收的全局配准回退结果'
    reasons = ''.join(f'<li>{html.escape(reason)}</li>' for reason in decision['fallback_reasons'])
    joint_link = '<a href="../joint/inspection.html">联合 BA 对照</a> · ' if not unavailable else ''
    Path(path).write_text(f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>LAS 坐标交付验收</title>
<style>body{{font:16px system-ui;max-width:1100px;margin:32px auto;padding:0 20px}}p,li{{line-height:1.7}}</style>
<h1>LAS 坐标交付验收</h1><p>最终采用：{selected}。图像 {result['registered_images']}，稀疏点 {result['points']}。
三维点和相机世界坐标均为输入 LAS 的 XYZ 框架；内参固定，LAS 未移动。</p>
<p><a href="result.json">交付决策</a> · <a href="quality_summary.json">图像及配准验收</a> ·
<a href="../global/inspection.html">全局配准</a> · {joint_link}<a href="../inputs.json">输入与坐标约定</a></p>
<ul>{reasons}</ul><p>配准门槛使用拟合区域。LAS 留出只作描述，不参与交付选择。
坐标单位按米假设；绝对地理精度及 3DGS 几何、渲染质量尚未验收。
最终 COLMAP 目录由主流程的 export 阶段交付，默认 WORK/colmap。</p></html>''', encoding='utf-8')
