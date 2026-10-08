from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pycolmap
import pytest

from geosunpre.artifacts import file_digest, staged_directory, write_json
from geosunpre.config import PipelineConfig, save_config
from geosunpre.features import FrontendConfig, LoMaFrontend
from geosunpre.pipeline import calibration, reconstruction, finalization as runner
from geosunpre.pipeline.observations import freeze_observations
from geosunpre.pipeline.quality import production_acceptance
from geosunpre.pipeline.runtime import StageRecorder
from geosunpre.pipeline.task import bind_task
from helpers import raw_input, tiny_visual_work, complete_calibration, global_frontend, make_las


def test_task_inherits_delivery_and_rejects_recipe_changes(tmp_path, monkeypatch):
    from geosunpre.pipeline import task as module
    save_config(tmp_path, PipelineConfig())
    first, delivery, cache = bind_task(tmp_path, output=tmp_path / 'delivery', image_mode='link',
                                     include_masks=True, include_ply=True, model_cache=tmp_path / 'weights')
    second, resumed, resumed_cache = bind_task(tmp_path)
    assert second == first and resumed == delivery and resumed_cache == cache
    with pytest.raises(ValueError, match='Frozen delivery option'):
        bind_task(tmp_path, include_masks=False)
    with pytest.raises(ValueError, match='Frozen delivery option'):
        bind_task(tmp_path, ply_voxel_size=.1)
    _, alternate, _ = bind_task(tmp_path, output=tmp_path / 'other', include_masks=False)
    assert not alternate['include_masks'] and bind_task(tmp_path)[1] == delivery
    monkeypatch.setattr(module, 'execution_identity', lambda: {'changed': True})
    with pytest.raises(ValueError, match='Frozen task execution changed'):
        bind_task(tmp_path)


def test_timing_remains_consistent_when_utc_clock_moves_backwards(tmp_path, monkeypatch):
    from geosunpre.pipeline import runtime
    monkeypatch.setattr(runtime, 'utc_now', lambda: '2000-01-01T00:00:00+00:00')
    recorder = StageRecorder(tmp_path, ['work'])
    with recorder.stage('work'):
        monkeypatch.setattr(runtime, 'utc_now', lambda: '1999-01-01T00:00:00+00:00')
    recorder.complete()
    result = recorder.value
    assert result['elapsed_seconds'] >= result['stage_elapsed_seconds'] >= 0
    assert result['overhead_seconds'] == result['elapsed_seconds'] - result['stage_elapsed_seconds']
    assert result['clock'] == 'perf_counter'


def test_cached_features_need_neither_model_nor_image_decode(tmp_path, monkeypatch):
    from geosunpre.scene import prepare
    scene = prepare(raw_input(tmp_path), tmp_path / 'prepared')
    frontend = LoMaFrontend(tmp_path / 'cache', scene, FrontendConfig())
    for row in scene['images']:
        _, _, token, path = frontend._feature_input(row)
        np.savez(path, token=token, keypoints=np.zeros((4, 2)), descriptors=np.zeros((4, 8)), scores=np.ones(4))
    def forbidden(*a, **k):
        raise AssertionError('Fully cached extraction must not load GPU models or decode images')
    monkeypatch.setattr(frontend, 'load_model', forbidden)
    monkeypatch.setattr(frontend, 'prepare_image', forbidden)
    values = list(frontend.extract_all(scene['images']))
    assert len(values) == 4 and frontend.cache_stats['feature_hits'] == 4


def test_failed_publication_keeps_original_and_archives_only_on_success(tmp_path):
    destination = tmp_path / 'stage'
    write_json(destination / 'partial.json', {'preserve': True})
    with pytest.raises(KeyboardInterrupt):
        with staged_directory(destination) as staged:
            (staged / 'result').write_text('partial')
            raise KeyboardInterrupt
    assert (destination / 'partial.json').is_file()
    assert not (destination / 'result').exists()
    with staged_directory(destination) as staged:
        (staged / 'result').write_text('complete')
    assert (destination / 'result').read_text() == 'complete'
    assert list(tmp_path.glob('stage.interrupted-*/partial.json'))


def test_calibration_resumes_after_interrupted_stage_publication(tmp_path, monkeypatch):
    work = global_frontend(tmp_path, frames=(70, 72))
    freeze_observations(work)
    original = calibration.staged_directory
    @contextmanager
    def interrupt(destination):
        with original(destination) as staged:
            yield staged
            if destination.name == 'focal_low_radial':
                raise KeyboardInterrupt
    monkeypatch.setattr(calibration, 'staged_directory', interrupt)
    with pytest.raises(KeyboardInterrupt):
        complete_calibration(work, monkeypatch)
    fixed = work / 'calibration/stages/fixed/ba.json'
    before = file_digest(fixed)
    partial = work / 'calibration/stages/focal_low_radial/sparse/0/cameras.bin'
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b'interrupted model')
    monkeypatch.setattr(calibration, 'staged_directory', original)
    solve = calibration.bounded_bundle_adjustment
    calls = []
    def tracked(rec, stage, *a, **k):
        calls.append(stage)
        return solve(rec, stage, *a, **k)
    monkeypatch.setattr(calibration, 'bounded_bundle_adjustment', tracked)
    calibration.calibrate(work)
    assert calls == ['focal_low_radial', 'all_radial', 'principal']
    assert file_digest(fixed) == before
    assert list((work / 'calibration/stages').glob('focal_low_radial.interrupted-*/sparse/0/cameras.bin'))


def test_final_ba_recovers_partial_output_and_then_reuses_it(tmp_path, monkeypatch):
    work = global_frontend(tmp_path, frames=(70, 72))
    freeze_observations(work)
    complete_calibration(work, monkeypatch)
    reconstruction.freeze_geometry(work)
    source = work / runner.SOURCE
    source.mkdir(parents=True)
    pycolmap.Reconstruction(work / 'calibration/stages/principal/sparse/0').write(source)
    before = file_digest(source / 'points3D.bin')
    partial = work / runner.FINAL / 'sparse/0/cameras.bin'
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b'interrupted')
    calls = []
    monkeypatch.setattr(runner, 'bounded_bundle_adjustment', lambda *a, **k: calls.append(True) or {})
    from geosunpre.pipeline import quality
    monkeypatch.setattr(quality, 'CONFIG', {**quality.CONFIG,
        'minimum_training_observations_per_image': 1, 'minimum_occupied_8x8_cells_per_image': 1})
    first = runner.finalize(work)
    assert runner.finalize(work) == first and calls == [True]
    assert file_digest(source / 'points3D.bin') == before
    assert list((work / 'sfm').glob('final.interrupted-*/sparse/0/cameras.bin'))


def test_global_ba_rebuilds_partial_stage_and_detects_changed_completed_model(tmp_path, monkeypatch):
    work = global_frontend(tmp_path, frames=(70, 72))
    freeze_observations(work)
    complete_calibration(work, monkeypatch)
    reconstruction.freeze_geometry(work)
    mapping = work / 'sfm/mapping'
    mapping.mkdir()
    pycolmap.Reconstruction(work / 'calibration/stages/principal/sparse/0').write(mapping)
    from geosunpre.artifacts import digest
    from geosunpre.colmap import model_hashes
    protocol = reconstruction.freeze_geometry(work)[4]
    write_json(mapping / 'identity.json', {'protocol_fingerprint': digest(protocol),
                                         'model_sha256': model_hashes(mapping)})
    monkeypatch.setattr(reconstruction, 'write_report', lambda *a: None)
    monkeypatch.setattr(reconstruction, 'filter_mapped_tracks', lambda *a: {'coverage_admissible': True})
    reconstruction.build_mapping(work)
    path = work / 'sfm/stages/fixed_calibration'
    write_json(path / 'ba.json', {'partial': True})
    calls = []
    monkeypatch.setattr(reconstruction, 'bounded_bundle_adjustment', lambda *a, **k: calls.append(True) or {})
    reconstruction.optimize(work)
    reconstruction.optimize(work)
    assert calls == [True, True, True]
    model = pycolmap.Reconstruction(path / 'sparse/0')
    next(iter(model.points3D.values())).xyz[0] += .1
    model.write(path / 'sparse/0')
    with pytest.raises(ValueError, match='stage identity changed'):
        reconstruction.optimize(work)
    (path / 'sparse/0/cameras.bin').unlink()
    with pytest.raises(ValueError, match='committed model is incomplete'):
        reconstruction.optimize(work)
    assert calls == [True, True, True]
    model = pycolmap.Reconstruction(mapping)
    next(iter(model.points3D.values())).xyz[0] += .1
    model.write(mapping)
    with pytest.raises(ValueError, match='Global mapping identity changed'):
        reconstruction.build_mapping(work)
    with pytest.raises(ValueError, match='Global mapping identity changed'):
        reconstruction.optimize(work)


def test_export_reuses_identical_request_without_copying_or_rendering_masks(tmp_path, monkeypatch):
    from geosunpre import colmap
    work, _, scene, _, _ = tiny_visual_work(tmp_path)
    output = tmp_path / 'export'
    las = tmp_path / 'input.las'
    make_las(las, np.random.default_rng(1).normal(size=(40, 3)))
    first = colmap.export_colmap(work / runner.SOURCE, scene, output, include_masks=True,
                                include_ply=True, initialization_las=las)
    def forbidden(*a, **k):
        raise AssertionError('Completed export must not be rebuilt')
    monkeypatch.setattr(colmap.shutil, 'copyfile', forbidden)
    monkeypatch.setattr(colmap, '_export_mask', forbidden)
    monkeypatch.setattr(colmap, 'export_las_ply', forbidden)
    assert colmap.export_colmap(work / runner.SOURCE, scene, output,
                               include_masks=True, include_ply=True, initialization_las=las) == first
    with pytest.raises(ValueError, match='Existing COLMAP export differs'):
        colmap.export_colmap(work / runner.SOURCE, scene, output, include_masks=True,
                             include_ply=True, initialization_las=las, ply_voxel_size=.1)


def test_acceptance_checks_each_lens_and_bound_policy():
    healthy = {'count': 10, 'finite_count': 10, 'median': 1., 'p90': 2.}
    report = {'training_reprojection': healthy, 'training_reprojection_by_camera': {'L': healthy, 'R': healthy},
              'camera_parameter_bounds': {'1': {'parameters_at_bounds': [2]}}}
    config = PipelineConfig(quality_max_median_px=2., quality_max_p90_px=4.)
    with pytest.raises(ValueError, match='optimization bounds'):
        production_acceptance(report, config)
    report['camera_parameter_bounds'] = {}
    report['training_reprojection_by_camera']['R'] = {**healthy, 'p90': 5.}
    with pytest.raises(ValueError, match='R: fitted p90'):
        production_acceptance(report, config)


def test_auto_calibration_skips_one_lens_prefix():
    scene = {'images': [{'camera': 'L', 'frame_index': i} for i in range(4)] +
                       [{'camera': 'R', 'frame_index': 4}]}
    selected = PipelineConfig(calibration_frame_count=2).calibration_images(scene)
    assert {r['frame_index'] for r in selected} == {3, 4}


def test_preflight_skips_gpu_for_all_cached_frontend(tmp_path, monkeypatch):
    import torch
    from geosunpre.pipeline.preflight import preflight
    from geosunpre.scene import prepare
    from geosunpre.features import FRONTEND_CONFIG as CONFIG
    work = tmp_path / 'work'
    scene = prepare(raw_input(tmp_path), work)
    save_config(work, PipelineConfig())
    write_json(work / 'sfm/manifest.json', scene)
    frontend = LoMaFrontend(work, scene, CONFIG)
    for row in scene['images']:
        frontend._feature_input(row)[3].write_bytes(b'cached')
    pair = work / 'matches/pair.npz'
    pair.parent.mkdir()
    pair.write_bytes(b'cached')
    write_json(work / 'sfm/frontend.json', {'pairs': [{'path': str(pair)}]})
    def forbidden():
        raise AssertionError('GPU availability must not be queried for fully cached work')
    monkeypatch.setattr(torch.cuda, 'is_available', forbidden)
    result = preflight(work, ('features', 'matches'), {'output': str(work / 'colmap'), 'image_mode': 'copy'})
    assert not result['gpu_needed']


def test_interrupted_calibration_database_is_archived_and_rebuilt(tmp_path, monkeypatch):
    work = global_frontend(tmp_path, frames=(70, 72))
    freeze_observations(work)
    original = pycolmap.Reconstruction.write
    def fail_seed(model, path):
        if Path(path).name == 'seed':
            (Path(path) / 'cameras.bin').write_bytes(b'partial seed')
            raise KeyboardInterrupt
        return original(model, path)
    with monkeypatch.context() as patch:
        patch.setattr(pycolmap.Reconstruction, 'write', fail_seed)
        with pytest.raises(KeyboardInterrupt):
            complete_calibration(work, monkeypatch)
    complete_calibration(work, monkeypatch)
    assert list((work / 'calibration').glob('database.db.interrupted-*'))
    assert list((work / 'calibration').glob('seed.interrupted-*/cameras.bin'))
    assert (work / 'calibration/calibration_identity.json').is_file()
