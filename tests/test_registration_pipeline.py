import json
from pathlib import Path

import numpy as np
import pycolmap
import pytest

from geosunpre.artifacts import digest, file_digest, write_frozen_json, write_json
from geosunpre.config import PipelineConfig, save_config
from geosunpre.registration.publication import complete_publication
from geosunpre.pipeline import registration, runner, quality, reconstruction, finalization
from geosunpre.pipeline.observations import freeze_observations
from geosunpre.pipeline.task import bind_task
from geosunpre.colmap import verify_colmap_export
from helpers import make_las
from helpers import raw_input
from helpers import global_frontend, complete_calibration


@pytest.mark.parametrize('modified', ['las', 'pose'])
def test_registration_inputs_are_required_and_content_frozen(tmp_path, modified):
    raw = raw_input(tmp_path)
    work = tmp_path / 'work'
    save_config(work, PipelineConfig())
    write_json(work / 'manifest.json', {'input_root': str(raw)})
    with pytest.raises(ValueError, match='ResultOut'):
        registration.freeze_inputs(work)
    (raw / 'ResultOut').mkdir()
    las = raw / 'ResultOut/scene.las'
    make_las(las, np.random.default_rng(0).normal(size=(100, 3)))
    with pytest.raises(ValueError, match='pos_ENH'):
        registration.freeze_inputs(work)
    for name in ['scene_pos.txt', 'scene_pos_ENH.txt']:
        (raw / name).write_text('2435 1 0 0 0\n2435 2 1 0 0\n2435 3 1 1 0\n')
    frozen = registration.freeze_inputs(work)
    assert registration.freeze_inputs(work) == frozen
    target = las if modified == 'las' else raw / 'scene_CameraPos_C2E.txt'
    with target.open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='Frozen inputs changed'):
        registration.freeze_inputs(work)


def supported_points():
    count = 300
    tiles = np.column_stack((np.repeat(np.arange(6), 50), np.zeros(count), np.zeros(count))).astype(int)
    return {'point_ids': np.arange(count, dtype=np.uint64), 'holdout': np.zeros(count, dtype=bool),
            'tiles': tiles, 'stable_surface_supported': np.ones(count, dtype=bool),
            'signed_plane_error': np.full(count, .01), 'nearest_distance': np.full(count, .02),
            'surface_normals': np.tile([0., 0., 1.], (count, 1))}


def test_local_las_regression_blocks_joint_delivery_despite_good_overall_quantiles():
    base = supported_points()
    candidate = {k: v.copy() for k, v in base.items()}
    candidate['signed_plane_error'][:50] = .04
    candidate['signed_plane_error'][50:] = .005
    result = registration._registration_acceptance(base, candidate, {'accepted_for_delivery': True}, PipelineConfig())
    assert not result['joint_eligible']
    assert any('tile/0,0,0' in failure for failure in result['joint_regression']['failures'])
    assert result['target_frame'] == 'input_las_xyz'


def test_las_coverage_loss_and_global_failure_cannot_be_reported_as_joint_success():
    base = supported_points()
    candidate = {k: v.copy() for k, v in base.items()}
    candidate['stable_surface_supported'][:50] = False
    result = registration._registration_acceptance(base, candidate, {'accepted_for_delivery': True}, PipelineConfig())
    assert not result['joint_eligible']
    base['stable_surface_supported'][:] = False
    with pytest.raises(ValueError, match='LAS registration acceptance rejected'):
        registration._registration_acceptance(base, None, None, PipelineConfig())


def test_geometry_gates_never_select_using_las_holdout():
    base = supported_points()
    base['holdout'][-50:] = True
    candidate = {k: v.copy() for k, v in base.items()}
    candidate['signed_plane_error'][-50:] = .9
    candidate['stable_surface_supported'][-50:] = False
    result = registration._registration_acceptance(base, candidate, {'accepted_for_delivery': True}, PipelineConfig())
    assert result['joint_eligible']
    assert result['joint_acceptance']['fit_points'] == 250


def test_nonfinite_joint_geometry_returns_the_registered_fallback():
    base = supported_points()
    candidate = {k: v.copy() for k, v in base.items()}
    candidate['signed_plane_error'][:] = np.nan
    result = registration._registration_acceptance(base, candidate, {'accepted_for_delivery': True}, PipelineConfig())
    assert not result['joint_eligible'] and result['joint_regression'] is None
    assert 'Nonfinite registration residuals' in result['fallback_reasons']


def test_committed_stage_rejects_changed_metadata_and_interrupted_stage_is_rebuilt(tmp_path):
    calls = []

    def compute(path, fingerprint):
        calls.append(1)
        path.mkdir()
        write_json(path / 'result.json', {'status': 'complete'})
        complete_publication(path, fingerprint)

    root = tmp_path / registration.ROOT
    write_json(root / 'global/partial.json', {'status': 'interrupted'})
    path = registration._stage(tmp_path, 'global', 'frozen', compute)
    assert len(calls) == 1 and list(root.glob('global.interrupted-*/partial.json'))
    assert registration._stage(tmp_path, 'global', 'frozen', compute) == path and len(calls) == 1
    write_json(path / 'result.json', {'status': 'edited'})
    with pytest.raises(ValueError, match='Completed registration stage changed'):
        registration._stage(tmp_path, 'global', 'frozen', compute)


def registration_fixture(tmp_path, monkeypatch, *, accept=True, interrupt=False,
                         reprojection_offset=None, joint_shift=0.):
    work = global_frontend(tmp_path, frames=(70, 72))
    freeze_observations(work)
    complete_calibration(work, monkeypatch)
    reconstruction.freeze_geometry(work)
    rec = pycolmap.Reconstruction(work / 'calibration/stages/principal/sparse/0')
    if reprojection_offset is not None:
        for image in rec.images.values():
            for index, point in enumerate(image.points2D):
                if point.has_point3D():
                    point.xy = point.xy + [reprojection_offset(image, index), 0.]
    source = work / finalization.SOURCE
    source.mkdir(parents=True)
    rec.write(source)
    monkeypatch.setattr(finalization, 'bounded_bundle_adjustment', lambda *a, **kw: {'termination_type': 'test'})
    monkeypatch.setattr(quality, 'CONFIG', {**quality.CONFIG,
        'minimum_training_observations_per_image': 1, 'minimum_occupied_8x8_cells_per_image': 1})
    monkeypatch.setattr(registration, 'POLICY', {**registration.POLICY,
        'minimum_supported_fit_points': 1, 'minimum_supported_fit_tiles': 1})
    finalization.finalize(work)
    bind_task(work, include_masks=True, include_ply=True)
    las = tmp_path / 'fixture.las'
    make_las(las, np.random.default_rng(0).normal(size=(100, 3)))
    inputs = {'version': 1, 'input_root': str(tmp_path / 'raw'), 'las': str(las),
              'las_header': registration.read_header(las),
              'input_sha256': {str(las): file_digest(las)}}
    inputs['fingerprint'] = digest(inputs)

    def freeze(work):
        write_frozen_json(Path(work) / registration.ROOT / 'inputs.json', inputs)
        return inputs

    monkeypatch.setattr(registration, 'freeze_inputs', freeze)
    calls = {'global': 0, 'joint': 0}
    matrix = np.eye(4)
    matrix[:3, :3] *= 2.
    matrix[:3, 3] = [10., 20., 30.]

    def align(model_path, inputs, output, **kw):
        calls['global'] += 1
        output.mkdir()
        transformed = pycolmap.Reconstruction(model_path)
        transformed.transform(pycolmap.Sim3d(matrix[:3]))
        model = output / 'sparse/0'
        model.mkdir(parents=True); transformed.write(model)
        ids = sorted(transformed.points3D)
        points = supported_points()
        points = {k: v[:len(ids)] for k, v in points.items()}
        points['point_ids'] = np.array(ids, dtype=np.uint64)
        np.savez(output / 'registration_points.npz', **points)
        complete_publication(output, kw['context_fingerprint'])

    def joint(global_stage, raw, output, **kw):
        calls['joint'] += 1
        if interrupt and calls['joint'] == 1:
            write_json(output / 'partial.json', {'interrupted': True})
            raise KeyboardInterrupt('interrupted joint stage')
        candidate = output
        model = candidate / 'sparse/0'
        model.mkdir(parents=True)
        proposed = pycolmap.Reconstruction(global_stage / 'sparse/0')
        if joint_shift:
            for frame in proposed.frames.values():
                pose = frame.rig_from_world
                frame.rig_from_world = pycolmap.Rigid3d(pose.rotation, pose.translation + [joint_shift, 0., 0.])
        proposed.write(model)
        write_json(candidate / 'diagnostic.json', {'accepted_for_delivery': accept})
        with np.load(global_stage / 'registration_points.npz') as archive:
            np.savez(candidate / 'las_points.npz', **{k: archive[k] for k in archive.files})
        complete_publication(output, kw['context_fingerprint'])

    monkeypatch.setattr(registration, 'register_global', align)
    monkeypatch.setattr(registration, 'refine_joint', joint)
    return work, calls, matrix


@pytest.mark.parametrize('accept', [True, False])
def test_main_export_delivers_las_frame_for_joint_and_registered_fallback(tmp_path, monkeypatch, accept):
    work, calls, matrix = registration_fixture(tmp_path, monkeypatch, accept=accept)
    with pytest.raises(FileNotFoundError):
        registration.export_registered(work)
    first = registration.register(work)
    assert first['joint_accepted'] == accept
    assert first['selected'] == ('pose_las' if accept else 'global_registration')
    assert registration.register(work) == first and calls == {'global': 1, 'joint': 1}
    result = runner.run_pipeline(work, stage='export')
    assert result['delivery_complete']
    assert result['export']['initialization']['source'] == 'las'
    assert result['export']['initialization']['points'] == 100
    assert result['export']['initialization']['voxel_size_m'] == .05
    assert verify_colmap_export(work / 'colmap')['full_content_check']
    exported = pycolmap.Reconstruction(work / 'colmap/sparse/0')
    source = pycolmap.Reconstruction(work / finalization.FINAL / 'sparse/0')
    for pid, point in source.points3D.items():
        np.testing.assert_allclose(exported.points3D[pid].xyz, matrix[:3, :3] @ point.xyz + matrix[:3, 3])
    for iid, image in source.images.items():
        np.testing.assert_allclose(exported.images[iid].projection_center(),
            matrix[:3, :3] @ image.projection_center() + matrix[:3, 3])
    identity = json.loads((work / 'colmap/colmap_export.json').read_text())
    assert identity['provenance']['target_frame'] == 'input_las_xyz'
    assert identity['provenance']['joint_refinement_accepted'] == accept
    assert identity['image_mode'] == 'copy' and identity['include_masks']
    assert not (work / 'colmap/images/L/70.jpg').is_symlink()
    write_json(work / registration.ROOT / 'final/result.json', {'edited': True})
    with pytest.raises(ValueError, match='Completed registration stage changed'):
        registration.export_registered(work, output=tmp_path / 'second')
    assert not (tmp_path / 'second').exists()


def test_interrupted_joint_run_resumes_from_committed_global_registration(tmp_path, monkeypatch):
    work, calls, _ = registration_fixture(tmp_path, monkeypatch, interrupt=True)
    with pytest.raises(KeyboardInterrupt):
        registration.register(work)
    assert not (work / registration.ROOT / 'final/result.json').exists()
    registration.register(work)
    assert calls == {'global': 1, 'joint': 2}
    assert list((work / registration.ROOT).glob('joint.interrupted-*/partial.json'))


@pytest.mark.parametrize('metric,limit', [('median', 2.), ('p90', 4.)])
@pytest.mark.parametrize('lens', [None, 'L', 'R'])
def test_absolute_reprojection_rejection_delivers_the_registered_fallback(tmp_path, monkeypatch, metric, limit, lens):
    from geosunpre.registration.joint_solver import ImageObservations, image_gate

    def offset(image, index):
        if (lens is not None and image.name.split('/')[0] != lens) or (metric == 'p90' and index % 4):
            return 1.
        return limit - 1e-4

    work, calls, _ = registration_fixture(tmp_path, monkeypatch,
        reprojection_offset=offset, joint_shift=-1e-3)
    result = registration.register(work)
    root = work / registration.ROOT
    reference = pycolmap.Reconstruction(root / 'global/sparse/0')
    candidate = pycolmap.Reconstruction(root / 'joint/sparse/0')
    observations = ImageObservations(reference, np.eye(4))
    baseline_errors, _ = observations.errors(reference)
    candidate_errors, _ = observations.errors(candidate)
    assert image_gate(observations, baseline_errors, candidate_errors)['passed']
    label = lens or 'all'
    with pytest.raises(ValueError, match=f'{label}: fitted {metric}'):
        quality.production_acceptance(quality.production_quality(candidate), PipelineConfig())
    assert result['selected'] == 'global_registration' and not result['joint_accepted']
    assert any(f'{label}: fitted {metric}' in reason for reason in result['fallback_reasons'])
    summary = json.loads((root / 'final/quality_summary.json').read_text())
    assert summary['acceptance']['status'] == 'passed'
    assert summary['registration']['joint_quality_acceptance']['status'] == 'rejected'
    assert registration.register(work) == result and calls == {'global': 1, 'joint': 1}
    registration.export_registered(work)
    assert verify_colmap_export(work / 'colmap')['full_content_check']
    exported = pycolmap.Reconstruction(work / 'colmap/sparse/0')
    for iid, image in reference.images.items():
        np.testing.assert_allclose(exported.images[iid].projection_center(), image.projection_center())


@pytest.mark.parametrize('error_type', [ValueError, RuntimeError])
def test_unexpected_quality_error_stops_delivery(tmp_path, monkeypatch, error_type):
    work, _, _ = registration_fixture(tmp_path, monkeypatch)

    def unexpected(rec):
        raise error_type('Unexpected quality computation error')

    monkeypatch.setattr(registration, 'production_quality', unexpected)
    with pytest.raises(error_type, match='Unexpected quality computation error'):
        registration.register(work)
    assert not (work / registration.ROOT / 'final/completion.json').exists()


def test_failed_reference_quality_cannot_be_used_as_fallback(tmp_path, monkeypatch):
    work, _, _ = registration_fixture(tmp_path, monkeypatch)

    def rejected(report, config):
        raise quality.ProductionAcceptanceRejected('No model meets the delivery quality limits')

    monkeypatch.setattr(registration, 'production_acceptance', rejected)
    with pytest.raises(quality.ProductionAcceptanceRejected, match='No model meets'):
        registration.register(work)
    assert not (work / registration.ROOT / 'final/completion.json').exists()


def test_joint_model_identity_error_stops_delivery(tmp_path, monkeypatch):
    work, _, _ = registration_fixture(tmp_path, monkeypatch)
    original = registration.refine_joint

    def changed_intrinsics(global_stage, inputs, output, **kwargs):
        original(global_stage, inputs, output, **kwargs)
        candidate = pycolmap.Reconstruction(output / 'sparse/0')
        candidate.cameras[1].params[0] += .01
        candidate.write(output / 'sparse/0')
        complete_publication(output, kwargs['context_fingerprint'])

    monkeypatch.setattr(registration, 'refine_joint', changed_intrinsics)
    with pytest.raises(ValueError, match='Fixed-intrinsics joint BA changed the camera model'):
        registration.register(work)
    assert not (work / registration.ROOT / 'final/completion.json').exists()


def test_unavailable_joint_constraints_deliver_and_reuse_the_global_result(tmp_path, monkeypatch):
    from geosunpre.registration.joint_solver import JointConstraintsUnavailable

    work, calls, _ = registration_fixture(tmp_path, monkeypatch)

    def unavailable(*args, **kwargs):
        calls['joint'] += 1
        raise JointConstraintsUnavailable('Insufficient reliable LAS constraints')

    monkeypatch.setattr(registration, 'refine_joint', unavailable)
    result = registration.register(work)
    assert result['selected'] == 'global_registration' and not result['joint_accepted']
    assert 'Insufficient reliable LAS constraints' in result['fallback_reasons']
    assert registration.register(work) == result and calls == {'global': 1, 'joint': 1}
    summary = json.loads((work / registration.ROOT / 'final/quality_summary.json').read_text())
    assert summary['registration']['joint_quality_acceptance']['status'] == 'not_evaluated'
    assert summary['acceptance']['status'] == 'passed'


def test_completed_registration_and_export_do_not_load_feature_arrays(tmp_path, monkeypatch):
    work, calls, _ = registration_fixture(tmp_path, monkeypatch)
    expected = registration.register(work)
    catalog = json.loads((work / 'sfm/frontend.json').read_text())
    paths = {str(Path(row['path']).resolve()) for row in catalog['features'].values()}
    original = np.lib.npyio.NpzFile.__getitem__

    def checked(archive, key):
        if str(Path(archive.zip.filename).resolve()) in paths and key in ('keypoints', 'scores'):
            raise AssertionError('Completed model checks must not materialize feature arrays')
        return original(archive, key)

    monkeypatch.setattr(np.lib.npyio.NpzFile, '__getitem__', checked)
    assert registration.register(work) == expected and calls == {'global': 1, 'joint': 1}
    registration.export_registered(work)
    assert verify_colmap_export(work / 'colmap')['full_content_check']
