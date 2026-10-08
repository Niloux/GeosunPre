"""Publish one accepted LAS-frame model and its portable COLMAP delivery."""

import html
import json
from pathlib import Path

import numpy as np
import pycolmap

from geosunpre.artifacts import archive_interrupted, digest, file_digest, staged_directory, write_frozen_json, write_json
from geosunpre.colmap import export_colmap, model_hashes
from geosunpre.initialization import DEFAULT_VOXEL_SIZE
from geosunpre.config import CONFIG_FILE, load_config
from geosunpre.pipeline.finalization import FINAL, _check_model, verify_final
from geosunpre.pipeline.observations import protocol_identity
from geosunpre.pipeline.quality import ProductionAcceptanceRejected, production_acceptance, production_quality
from geosunpre.pipeline.task import TASK_FILE
from geosunpre.registration.geometry import quantiles
from geosunpre.registration.global_alignment import MAX_LAS_POINTS, register_global
from geosunpre.registration.joint import refine_joint
from geosunpre.registration.joint_solver import IMAGE_GATES, RECIPE, JointConstraintsUnavailable, assert_identity
from geosunpre.registration.las import read_header
from geosunpre.registration.publication import complete_publication, verify_publication
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


def freeze_inputs(work):
    """Record LAS and the vendor pose-frame assumption before expensive pending work."""
    work = Path(work).resolve()
    config = load_config(work)
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
             'settings': {key: val for key, val in vars(config).items() if key.startswith('registration_') or key == 'workers'},
             'max_las_points': MAX_LAS_POINTS,
             'pose_coordinate_convention': {
                 'name': 'enh_first',
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


def _registration_acceptance(reference_points, candidate_points, diagnostic, config):
    """Evaluate joint eligibility; this alone does not establish delivery quality."""
    base = surface_acceptance(reference_points, config)
    if not base['passed']:
        raise ValueError('LAS registration acceptance rejected: ' + '; '.join(base['failures']))
    candidate = surface_acceptance(candidate_points, config) if candidate_points is not None else None
    regression = geometric_regression(reference_points, candidate_points) if candidate and candidate['passed'] else None
    accepted = bool(diagnostic and diagnostic['accepted_for_delivery'] and candidate['passed']
                    and regression['passed'])
    reasons = []
    if not diagnostic or not diagnostic.get('accepted_for_delivery'):
        reasons.append('Joint BA did not pass all image, geometry and update checks')
    if candidate and not candidate['passed']:
        reasons.extend(candidate['failures'])
    if regression and not regression['passed']:
        reasons.extend(regression['failures'])
    return {'joint_eligible': accepted, 'fallback_reasons': reasons if not accepted else [],
            'reference_acceptance': base, 'joint_acceptance': candidate, 'joint_regression': regression,
            'target_frame': 'input_las_xyz', 'coordinate_units_assumed': 'metres',
            'absolute_accuracy_validated': False, 'training_run': False}


def select_delivery(source, global_stage, joint, scene, config, *, camera_parameter_bounds):
    """Return a model, decision and quality only after full delivery acceptance.

    Stage identities must already be verified. Explicit fitted-data rejection of
    an eligible joint candidate falls back to the global model, which must pass
    the same delivery limits. Identity, model and computation errors propagate.
    """
    unavailable = (joint / 'unavailable.json').is_file()
    diagnostic = None if unavailable else _read(joint / 'diagnostic.json')
    decision = _registration_acceptance(_points(global_stage / 'registration_points.npz'),
        None if unavailable else _points(joint / 'las_points.npz'), diagnostic, config)
    eligible = decision.pop('joint_eligible')
    if unavailable:
        decision['fallback_reasons'].append(_read(joint / 'unavailable.json')['error'])
    original = pycolmap.Reconstruction(source)

    def inspect(stage):
        rec = pycolmap.Reconstruction(stage / 'sparse/0')
        assert_identity(original, rec)
        _check_model(rec, scene)
        report = {**production_quality(rec), 'camera_parameter_bounds': camera_parameter_bounds}
        return rec, report

    accepted = False
    joint_acceptance = {'status': 'not_evaluated'}
    if eligible:
        rec, quality = inspect(joint)
        try:
            quality['acceptance'] = production_acceptance(quality, config)
        except ProductionAcceptanceRejected as exc:
            joint_acceptance = {'status': 'rejected', 'reason': str(exc)}
            decision['fallback_reasons'].append('Joint delivery quality rejected: ' + str(exc))
        else:
            accepted = True
            joint_acceptance = quality['acceptance']
    if not accepted:
        rec, quality = inspect(global_stage)
        quality['acceptance'] = production_acceptance(quality, config)
    decision.update(selected='pose_las' if accepted else 'global_registration',
                    joint_accepted=accepted, joint_quality_acceptance=joint_acceptance)
    quality['registration'] = decision
    return rec, decision, quality


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
    """Keep the verified global model if joint refinement fails its quality gates."""
    work = Path(work).resolve()
    config = load_config(work)
    inputs = freeze_inputs(work)
    visual_scene, visual_result, _ = verify_final(work)
    source = work / FINAL / 'sparse/0'
    root = work / ROOT
    protocol = {'version': 2, 'inputs_fingerprint': inputs['fingerprint'],
                'source_model_sha256': model_hashes(source),
                'visual_result_sha256': file_digest(work / FINAL / 'result.json'),
                'scene_manifest_sha256': file_digest(work / 'sfm/manifest.json'),
                'solver_recipe': {**RECIPE, 'workers': config.workers},
                'image_gates': IMAGE_GATES, 'geometry_policy': POLICY, 'intrinsics_optimized': False}
    protocol['fingerprint'] = digest(protocol)
    write_frozen_json(root / 'protocol.json', protocol)
    final = root / 'final'
    if (final / 'completion.json').is_file():
        return verify_registered(work)
    global_stage = _stage(work, 'global', protocol['fingerprint'], lambda path, fingerprint:
        register_global(source, inputs, path, workers=config.workers, context_fingerprint=fingerprint))
    reference = _points(global_stage / 'registration_points.npz')
    base_acceptance = surface_acceptance(reference, config)
    if not base_acceptance['passed']:
        write_json(root / 'rejection.json', {'stage': 'global', 'acceptance': base_acceptance})
        raise ValueError('LAS registration acceptance rejected: ' + '; '.join(base_acceptance['failures']))

    def compute_joint(path, fingerprint):
        try:
            refine_joint(global_stage, inputs, path, workers=config.workers, context_fingerprint=fingerprint)
        except JointConstraintsUnavailable as exc:
            with staged_directory(path) as stage:
                write_json(stage / 'unavailable.json', {'error': str(exc)})
                complete_publication(stage, fingerprint)

    joint = _stage(work, 'joint', protocol['fingerprint'], compute_joint)
    unavailable = (joint / 'unavailable.json').is_file()
    rec, decision, quality = select_delivery(source, global_stage, joint, visual_scene, config,
        camera_parameter_bounds=visual_result['ba'].get('cameras', {}))
    with staged_directory(final) as stage:
        model = stage / 'sparse/0'
        model.mkdir(parents=True)
        rec.write(model)
        _check_model(pycolmap.Reconstruction(model), visual_scene)
        hashes = model_hashes(model)
        quality['model_sha256'] = hashes
        result = {'version': 2, 'status': 'registered', 'model': str(final / 'sparse/0'),
                  'selected': decision['selected'], 'joint_accepted': decision['joint_accepted'],
                  'fallback_reasons': decision['fallback_reasons'], 'target_frame': decision['target_frame'],
                  'model_sha256': hashes, 'protocol_fingerprint': protocol['fingerprint'],
                  'inputs_fingerprint': inputs['fingerprint'], 'registered_images': rec.num_reg_images(),
                  'points': rec.num_points3D(), 'observations': rec.compute_num_observations(),
                  'global_completion_sha256': file_digest(global_stage / 'completion.json'),
                  'joint_completion_sha256': file_digest(joint / 'completion.json'),
                  'absolute_accuracy_validated': False, 'intrinsics_optimized': False}
        write_json(stage / 'quality_summary.json', quality)
        write_json(stage / 'result.json', result)
        _write_report(stage / 'inspection.html', result, quality, unavailable)
        complete_publication(stage, digest({'name': 'final', 'protocol_fingerprint': protocol['fingerprint']}))
    return verify_registered(work)


def verify_registered(work):
    """Verify all completed registration stages and their current upstream model."""
    work = Path(work).resolve()
    inputs = freeze_inputs(work)
    verify_final(work)
    root = work / ROOT
    protocol = _read(root / 'protocol.json')
    if (protocol.get('version') != 2
            or protocol.get('fingerprint') != digest({k: v for k, v in protocol.items() if k != 'fingerprint'})
            or protocol['inputs_fingerprint'] != inputs['fingerprint']
            or protocol['source_model_sha256'] != model_hashes(work / FINAL / 'sparse/0')
            or protocol['visual_result_sha256'] != file_digest(work / FINAL / 'result.json')
            or protocol['scene_manifest_sha256'] != file_digest(work / 'sfm/manifest.json')):
        raise ValueError('Registration inputs or visual source identity changed')
    for name in ('global', 'joint', 'final'):
        verify_publication(root / name, digest({'name': name, 'protocol_fingerprint': protocol['fingerprint']}))
    result = _read(root / 'final/result.json')
    if (result['protocol_fingerprint'] != protocol['fingerprint']
            or result['inputs_fingerprint'] != inputs['fingerprint']
            or result['model_sha256'] != model_hashes(root / 'final/sparse/0')
            or result['global_completion_sha256'] != file_digest(root / 'global/completion.json')
            or result['joint_completion_sha256'] != file_digest(root / 'joint/completion.json')
            or result['target_frame'] != 'input_las_xyz'):
        raise ValueError('Final registration identity changed')
    return result


def export_registered(work, *, output=None, image_mode='copy', include_masks=True, include_ply=True,
                      ply_voxel_size=DEFAULT_VOXEL_SIZE):
    work = Path(work).resolve()
    result = verify_registered(work)
    scene = load_scene(work / 'sfm')
    final = work / ROOT / 'final'
    quality = _read(final / 'quality_summary.json')
    production_acceptance(quality, load_config(work))
    _check_model(pycolmap.Reconstruction(final / 'sparse/0'), scene)
    output = Path(output).resolve() if output else work / 'colmap'
    inputs = _read(work / ROOT / 'inputs.json')
    if (output == work or work.is_relative_to(output)
            or any(output.is_relative_to(work / directory) for directory in
                   ('registration', 'sfm', 'calibration', 'features', 'matches'))
            or output.is_relative_to(Path(inputs['input_root']))):
        raise ValueError('Choose a COLMAP export separate from pipeline models and raw input')
    provenance = {'kind': 'geosunpre_production', 'source_model_sha256': result['model_sha256'],
        'pipeline_config_sha256': file_digest(work / CONFIG_FILE),
        'pipeline_task_sha256': file_digest(work / TASK_FILE),
        'model_weights_sha256': {name: file_digest(work / name) for name in
            ('frontend_weights.json', 'segmentation_weights.json') if (work / name).is_file()},
        'scene_manifest_sha256': file_digest(work / 'sfm/manifest.json'), **protocol_identity(work),
        'registration_inputs_sha256': file_digest(work / ROOT / 'inputs.json'),
        'registration_protocol_sha256': file_digest(work / ROOT / 'protocol.json'),
        'registration_result_sha256': file_digest(final / 'result.json'),
        'las_sha256': inputs['input_sha256'][inputs['las']],
        'target_frame': 'input_las_xyz', 'registration_selected': result['selected'],
        'joint_refinement_accepted': result['joint_accepted'],
        'independent_exposure_poses': True, 'manual_annotations_used': False,
        'intrinsics_optimized_during_registration': False, 'absolute_accuracy_validated': False}
    return export_colmap(final / 'sparse/0', scene, output, image_mode=image_mode,
                         include_masks=include_masks, include_ply=include_ply,
                         initialization_las=inputs['las'], ply_voxel_size=ply_voxel_size,
                         provenance=provenance, quality_report=final / 'quality_summary.json')


def _write_report(path, result, quality, unavailable):
    selected = 'Pose / LAS 联合优化' if result['joint_accepted'] else '已验收全局配准（联合更新被拒绝）'
    reasons = ''.join(f'<li>{html.escape(reason)}</li>' for reason in result['fallback_reasons'])
    residual = quality['training_reprojection']
    joint_link = '<a href="../joint/diagnostic.json">联合求解记录</a>' if not unavailable else '<a href="../joint/unavailable.json">约束不足记录</a>'
    Path(path).write_text(f"""<!doctype html><html lang="zh"><meta charset="utf-8"><title>COLMAP 交付验收</title>
<style>body{{font:16px system-ui;max-width:1000px;margin:32px auto;padding:0 20px}}p,li{{line-height:1.7}}</style>
<h1>COLMAP 交付验收</h1><p>采用结果：{selected}。</p>
<p>图像 {result['registered_images']}，稀疏点 {result['points']}，观测 {result['observations']}；
拟合残差 median / P90：{residual['median']:.3f} / {residual['p90']:.3f} px。</p>
<p>相机和三维点位于输入 LAS XYZ 框架，内参固定。下游直接读取 COLMAP，不再套用配准变换。</p>
<ul>{reasons}</ul><p><a href="result.json">交付决策</a> · <a href="quality_summary.json">完整质量记录</a> ·
<a href="../global/report.json">全局配准记录</a> · {joint_link} · <a href="../inputs.json">输入约定</a></p>
<p>验收使用拟合观测和获支持的 LAS 表面；绝对地理精度与 3DGS 渲染质量需另行验证。</p></html>""", encoding='utf-8')
