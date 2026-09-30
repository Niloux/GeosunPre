import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pycolmap
import pytest

from geosunpre.artifacts import digest, file_digest, write_frozen_json, write_json
from geosunpre.config import CONFIG_FILE, PipelineConfig, load_config, save_config
from geosunpre.alignment.publication import complete_publication, verify_publication
from geosunpre.pipeline import registration, runner, quality, reconstruction
from geosunpre.pipeline.observations import freeze_observations
from geosunpre.pipeline.task import bind_task
from geosunpre.colmap import verify_colmap_export
from test_alignment import make_las
from test_pipeline import raw_input
from test_unified_holdout import global_frontend, complete_calibration


@pytest.mark.parametrize('version', [1, 2, 3, 4])
def test_legacy_tasks_keep_visual_delivery_without_rewriting(tmp_path, version):
    settings = {k: v for k, v in asdict(PipelineConfig()).items() if not k.startswith('registration_')}
    if version < 4:
        settings.pop('pipeline_mode')
    if version < 3:
        settings.pop('calibration_mode')
    value = {'version': version, 'settings': settings}
    write_json(tmp_path / CONFIG_FILE, {**value, 'fingerprint': digest(value)})
    before = (tmp_path / CONFIG_FILE).read_bytes()
    loaded = load_config(tmp_path)
    assert loaded.registration_mode == 'visual'
    save_config(tmp_path, loaded)
    assert (tmp_path / CONFIG_FILE).read_bytes() == before
    with pytest.raises(ValueError, match='Frozen inputs changed'):
        save_config(tmp_path, PipelineConfig())


def test_joint_default_requires_complete_frozen_settings(tmp_path):
    assert PipelineConfig().registration_mode == 'joint'
    save_config(tmp_path, PipelineConfig())
    value = json.loads((tmp_path / CONFIG_FILE).read_text())
    value['settings'].pop('registration_pose_weight')
    value['fingerprint'] = digest({k: v for k, v in value.items() if k != 'fingerprint'})
    write_json(tmp_path / CONFIG_FILE, value)
    with pytest.raises(ValueError, match='Missing frozen registration'):
        load_config(tmp_path)


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
    result = registration.select_candidate(base, candidate, {'accepted_for_candidate_review': True}, PipelineConfig())
    assert result['selected'] == 'global_registration'
    assert any('tile/0,0,0' in failure for failure in result['joint_regression']['failures'])
    assert result['target_frame'] == 'input_las_xyz'


def test_las_coverage_loss_and_global_failure_cannot_be_reported_as_joint_success():
    base = supported_points()
    candidate = {k: v.copy() for k, v in base.items()}
    candidate['stable_surface_supported'][:50] = False
    result = registration.select_candidate(base, candidate, {'accepted_for_candidate_review': True}, PipelineConfig())
    assert not result['joint_accepted']
    base['stable_surface_supported'][:] = False
    with pytest.raises(ValueError, match='LAS registration acceptance rejected'):
        registration.select_candidate(base, None, None, PipelineConfig())


def test_geometry_gates_never_select_using_las_holdout():
    base = supported_points()
    base['holdout'][-50:] = True
    candidate = {k: v.copy() for k, v in base.items()}
    candidate['signed_plane_error'][-50:] = .9
    candidate['stable_surface_supported'][-50:] = False
    result = registration.select_candidate(base, candidate, {'accepted_for_candidate_review': True}, PipelineConfig())
    assert result['joint_accepted']
    assert result['joint_acceptance']['fit_points'] == 250


def test_nonfinite_joint_geometry_returns_the_registered_fallback():
    base = supported_points()
    candidate = {k: v.copy() for k, v in base.items()}
    candidate['signed_plane_error'][:] = np.nan
    result = registration.select_candidate(base, candidate, {'accepted_for_candidate_review': True}, PipelineConfig())
    assert not result['joint_accepted'] and result['joint_regression'] is None
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


def registration_fixture(tmp_path, monkeypatch, *, accept=True, interrupt=False):
    work = global_frontend(tmp_path, frames=(70, 72), pipeline_mode='production', registration_mode='joint')
    freeze_observations(work)
    complete_calibration(work, monkeypatch)
    reconstruction.freeze_geometry(work)
    rec = pycolmap.Reconstruction(work / 'calibration/stages/principal/sparse/0')
    source = work / runner.SOURCE
    source.mkdir(parents=True)
    rec.write(source)
    monkeypatch.setattr(runner, 'bounded_bundle_adjustment', lambda *a, **kw: {'termination_type': 'test'})
    monkeypatch.setattr(quality, 'CONFIG', {**quality.CONFIG,
        'minimum_training_observations_per_image': 1, 'minimum_occupied_8x8_cells_per_image': 1})
    monkeypatch.setattr(registration, 'POLICY', {**registration.POLICY,
        'minimum_supported_fit_points': 1, 'minimum_supported_fit_tiles': 1})
    runner.finalize(work)
    bind_task(work, include_masks=True, include_ply=True)
    las = tmp_path / 'fixture.las'
    make_las(las, np.random.default_rng(0).normal(size=(100, 3)))
    inputs = {'version': 1, 'input_root': str(tmp_path), 'las': str(las),
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

    def align(dataset, las, output, **kw):
        calls['global'] += 1
        output.mkdir()
        transformed = pycolmap.Reconstruction(Path(dataset) / 'sparse/0')
        transformed.transform(pycolmap.Sim3d(matrix[:3]))
        model = output / 'transformed_model'
        model.mkdir(); transformed.write(model)
        ids = sorted(transformed.points3D)
        points = supported_points()
        points = {k: v[:len(ids)] for k, v in points.items()}
        points['point_ids'] = np.array(ids, dtype=np.uint64)
        np.savez(output / 'registration_points.npz', **points)
        complete_publication(output, kw['context_fingerprint'])

    def joint(global_stage, raw, output, **kw):
        calls['joint'] += 1
        assert kw['mode'] == 'production'
        if interrupt and calls['joint'] == 1:
            write_json(output / 'partial.json', {'interrupted': True})
            raise KeyboardInterrupt('interrupted joint stage')
        candidate = output / 'stages/pose_las'
        model = candidate / 'aligned_model'
        model.mkdir(parents=True)
        pycolmap.Reconstruction(global_stage / 'transformed_model').write(model)
        write_json(candidate / 'diagnostic.json', {'accepted_for_candidate_review': accept})
        with np.load(global_stage / 'registration_points.npz') as archive:
            np.savez(candidate / 'las_points.npz', **{k: archive[k] for k in archive.files})
        complete_publication(output, kw['context_fingerprint'])

    monkeypatch.setattr(registration, 'run_alignment', align)
    monkeypatch.setattr(registration, 'run_joint_alignment', joint)
    return work, calls, matrix


@pytest.mark.parametrize('accept', [True, False])
def test_main_export_delivers_las_frame_for_joint_and_registered_fallback(tmp_path, monkeypatch, accept):
    work, calls, matrix = registration_fixture(tmp_path, monkeypatch, accept=accept)
    with pytest.raises(ValueError, match='Complete the register stage'):
        runner.export_result(work)
    first = registration.register(work)
    assert first['joint_accepted'] == accept
    assert first['selected'] == ('pose_las' if accept else 'global_registration')
    assert registration.register(work) == first and calls == {'global': 1, 'joint': 1}
    result = runner.run_pipeline(work, stage='export')
    assert result['delivery_complete']
    assert verify_colmap_export(work / 'colmap')['full_content_check']
    exported = pycolmap.Reconstruction(work / 'colmap/sparse/0')
    source = pycolmap.Reconstruction(work / runner.FINAL / 'sparse/0')
    for pid, point in source.points3D.items():
        np.testing.assert_allclose(exported.points3D[pid].xyz, matrix[:3, :3] @ point.xyz + matrix[:3, 3])
    for iid, image in source.images.items():
        np.testing.assert_allclose(exported.images[iid].projection_center(),
            matrix[:3, :3] @ image.projection_center() + matrix[:3, 3])
    identity = json.loads((work / 'colmap/colmap_export.json').read_text())
    assert identity['provenance']['target_frame'] == 'input_las_xyz'
    assert identity['provenance']['las_used_for_registration']
    assert identity['provenance']['las_used_for_joint_ba'] == accept
    assert identity['image_mode'] == 'copy' and identity['include_masks']
    assert not (work / 'colmap/images/L/70.jpg').is_symlink()
    write_json(work / registration.ROOT / 'final/result.json', {'edited': True})
    with pytest.raises(ValueError, match='Completed registration stage changed'):
        runner.export_result(work, output=tmp_path / 'second')
    assert not (tmp_path / 'second').exists()


def test_interrupted_joint_run_resumes_from_committed_global_registration(tmp_path, monkeypatch):
    work, calls, _ = registration_fixture(tmp_path, monkeypatch, interrupt=True)
    with pytest.raises(KeyboardInterrupt):
        registration.register(work)
    assert not (work / registration.ROOT / 'final/result.json').exists()
    registration.register(work)
    assert calls == {'global': 1, 'joint': 2}
    assert list((work / registration.ROOT).glob('joint.interrupted-*/partial.json'))
