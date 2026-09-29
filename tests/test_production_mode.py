import json
from dataclasses import asdict

import numpy as np
import pycolmap
import pytest

from geosunpre.artifacts import digest, file_digest, write_json
from geosunpre.config import CONFIG_FILE, PipelineConfig, load_config, save_config
from geosunpre.pipeline import calibration, reconstruction, runner, quality
from geosunpre.pipeline.holdout import freeze_holdout
from geosunpre.pipeline.observations import freeze_observations, load_observations, partitions
from geosunpre.colmap import verify_colmap_export
from test_unified_holdout import global_frontend, complete_calibration


@pytest.mark.parametrize('version', [1, 2, 3])
def test_legacy_runs_remain_validation_without_rewriting(tmp_path, version):
    settings = asdict(PipelineConfig())
    del settings['pipeline_mode']
    if version < 3:
        del settings['calibration_mode']
    value = {'version': version, 'settings': settings}
    path = tmp_path / CONFIG_FILE
    write_json(path, {**value, 'fingerprint': digest(value)})
    before = path.read_bytes()
    loaded = load_config(tmp_path)
    assert loaded.pipeline_mode == 'validation'
    save_config(tmp_path, loaded)
    assert path.read_bytes() == before
    with pytest.raises(ValueError, match='Frozen inputs changed'):
        save_config(tmp_path, PipelineConfig())


def test_production_default_and_explicit_mode_required(tmp_path):
    assert PipelineConfig().pipeline_mode == 'production'
    with pytest.raises(ValueError, match='pipeline_mode'):
        PipelineConfig(pipeline_mode='unknown')
    save_config(tmp_path, PipelineConfig())
    path = tmp_path / CONFIG_FILE
    value = json.loads(path.read_text())
    del value['settings']['pipeline_mode']
    value['fingerprint'] = digest({k: v for k, v in value.items() if k != 'fingerprint'})
    write_json(path, value)
    with pytest.raises(ValueError, match='Missing frozen pipeline_mode'):
        load_config(tmp_path)


@pytest.mark.parametrize('change', ['match', 'protocol', 'mixed_split'])
def test_production_observations_reject_changes(tmp_path, change):
    work = global_frontend(tmp_path, pipeline_mode='production')
    protocol = freeze_observations(work)
    assert freeze_observations(work) == protocol
    _, _, data, _ = reconstruction.load_frontend(work)
    if change == 'match':
        path = data['pairs'][0]['path']
        with np.load(path) as raw:
            arrays = {k: raw[k].copy() for k in raw.files}
        arrays['matches'] = arrays['matches'][::-1]
        np.savez(path, **arrays)
    elif change == 'protocol':
        path = work / 'sfm/observation_protocol.json'
        protocol['excluded_for_validation'] = 1
        write_json(path, protocol)
    else:
        write_json(work / 'sfm/split.json', {'tracks': []})
    with pytest.raises(ValueError, match='changed|mode differs'):
        load_observations(work, data)


def test_production_uses_all_matches_before_geometry(tmp_path, monkeypatch):
    from geosunpre.sfm import reconstruction as geometry

    work = global_frontend(tmp_path, frames=(70, 72), pipeline_mode='production')
    with pytest.raises(ValueError, match='Missing production observation'):
        calibration.calibrate(work)
    freeze_observations(work)
    with pytest.raises(ValueError, match='validation mode'):
        freeze_holdout(work)
    out, scene, data, features = calibration.prepare_calibration(work)
    split, triangle = partitions(work)
    assert split['tracks'] == [] and triangle is None

    class GeometryReached(Exception):
        pass

    def pair(a, b, pa, pb, **kwargs):
        assert len(pa) == len(pb) == 20
        raise GeometryReached

    monkeypatch.setattr(geometry, 'fit_pair', pair)
    with pytest.raises(GeometryReached):
        geometry.database_and_seed(out / 'probe', scene, scene['images'], data['pairs'], features, split)
    assert not (out / 'split.json').exists()


def test_production_calibration_finalize_export_never_evaluate_holdouts(tmp_path, monkeypatch):
    work = global_frontend(tmp_path, frames=(70, 72), pipeline_mode='production')
    freeze_observations(work)

    def forbidden(*args, **kwargs):
        raise AssertionError('Production must not evaluate or compare held-out queries')

    for module in (calibration, reconstruction, runner):
        monkeypatch.setattr(module, 'evaluate_reconstruction', forbidden)
    monkeypatch.setattr(reconstruction, 'paired_common_support', forbidden)
    result, calls = complete_calibration(work, monkeypatch)
    assert len(calls) == 4
    assert all('validation' not in v for v in result['stages'].values())
    _, _, _, _, protocol, _ = reconstruction.freeze_geometry(work)
    assert protocol['pipeline_mode'] == 'production' and 'holdout' not in protocol
    model = pycolmap.Reconstruction(work / 'calibration/stages/principal/sparse/0')
    assert model.num_points3D() == 20
    target = work / runner.SOURCE
    target.mkdir(parents=True)
    model.write(target)
    monkeypatch.setattr(runner, 'bounded_bundle_adjustment', lambda *a, **kw: {'termination_type': 'test'})
    # This two-image fixture tests wiring and diagnostics, not production coverage acceptance.
    monkeypatch.setattr(quality, 'CONFIG', {**quality.CONFIG,
        'minimum_training_observations_per_image': 1, 'minimum_occupied_8x8_cells_per_image': 1})
    first = runner.finalize(work)
    assert first == runner.finalize(work)
    assert 'triangle_holdout' not in first
    report = json.loads((work / runner.FINAL / 'quality_summary.json').read_text())
    assert report['heldout_evaluation_performed'] is False
    assert report['training_reprojection']['count'] == 40
    runner.export_result(work)
    assert verify_colmap_export(work / 'colmap')['full_content_check']
    provenance = json.loads((work / 'colmap/colmap_export.json').read_text())['provenance']
    assert provenance['pipeline_mode'] == 'production'
    assert 'holdout_isolation' not in provenance
    for pattern in ('split.json', '*validation*', 'comparison.json', '*bootstrap*'):
        assert not list(work.rglob(pattern))
    with pytest.raises(ValueError, match='unavailable in production'):
        runner.run_pipeline(work, stage='holdout')


def test_quality_rejects_insufficient_coverage(tmp_path):
    from test_pipeline import tiny_visual_work
    work, _, _, _, _ = tiny_visual_work(tmp_path)
    model = pycolmap.Reconstruction(work / runner.SOURCE)
    with pytest.raises(ValueError, match='coverage'):
        quality.production_quality(model)


def test_production_input_calibration_uses_observation_identity(tmp_path):
    work = global_frontend(tmp_path, pipeline_mode='production', calibration_mode='input')
    freeze_observations(work)
    calibration.calibrate(work)
    _, scene, _, _, protocol, params = reconstruction.freeze_geometry(work)
    assert protocol['pipeline_mode'] == 'production'
    assert protocol['calibration_mode'] == 'input'
    for cid, lens in [(1, 'L'), (2, 'R')]:
        np.testing.assert_array_equal(params[cid], scene['cameras'][lens]['params'])
    assert not (work / 'calibration/stages').exists()


def test_default_pipeline_orders_observations_and_skips_mapping_evaluation(tmp_path, monkeypatch):
    from geosunpre import scene
    from geosunpre.pipeline import data, observations
    from geosunpre.pipeline import PRODUCTION_STAGES

    events = []

    def record(name, result=None):
        def call(*args, **kwargs):
            events.append(name)
            return result
        return call

    monkeypatch.setattr(scene, 'prepare', record('prepare', {'images': [
        {'camera': 'L', 'frame_index': 0}, {'camera': 'R', 'frame_index': 1}]}))
    for function, stage in [('prepare_masks', 'masks'), ('extract_features', 'features'), ('match_features', 'matches')]:
        monkeypatch.setattr(data, function, record(stage))
    monkeypatch.setattr(observations, 'freeze_observations', record('observations', {
        'fingerprint': 'test', 'scope': 'all_eligible_observations', 'excluded_for_validation': 0}))
    monkeypatch.setattr(calibration, 'calibrate', record('calibration'))
    monkeypatch.setattr(reconstruction, 'build_mapping', record('sfm'))
    monkeypatch.setattr(reconstruction, 'optimize', lambda *a: None)
    def forbidden(*args):
        raise AssertionError('Production must not evaluate mapping holdouts')
    monkeypatch.setattr(reconstruction, 'evaluate_mapping', forbidden)
    monkeypatch.setattr(runner, 'finalize', record('finalize'))
    monkeypatch.setattr(runner, 'export_result', record('export'))
    runner.run_pipeline(tmp_path, input_root='unused')
    assert events == list(PRODUCTION_STAGES)
