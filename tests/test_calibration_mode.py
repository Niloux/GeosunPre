import json
from dataclasses import asdict

import numpy as np
import pytest

from geosunpre.artifacts import digest, file_digest, write_json
from geosunpre.config import CONFIG_FILE, PipelineConfig, load_config, save_config
from geosunpre.pipeline import calibration, reconstruction
from geosunpre.pipeline.holdout import freeze_holdout
from test_unified_holdout import global_frontend


@pytest.mark.parametrize('version', [1, 2])
def test_legacy_config_remains_local_and_byte_identical(tmp_path, version):
    settings = asdict(PipelineConfig(matcher_attention='math', pipeline_mode='validation'))
    settings = {k: v for k, v in settings.items() if not k.startswith('registration_')}
    del settings['calibration_mode']
    value = {'version': version, 'settings': settings}
    write_json(tmp_path / CONFIG_FILE, {**value, 'fingerprint': digest(value)})
    before = (tmp_path / CONFIG_FILE).read_bytes()
    config = load_config(tmp_path)
    assert config.calibration_mode == 'local'
    save_config(tmp_path, config)
    assert (tmp_path / CONFIG_FILE).read_bytes() == before
    with pytest.raises(ValueError, match='Frozen inputs changed'):
        save_config(tmp_path, PipelineConfig(matcher_attention='math', calibration_mode='input'))


def test_input_mode_needs_no_local_segment_but_requires_both_lenses():
    config = PipelineConfig(calibration_mode='input', calibration_frames=(70, 94))
    scene = {'images': [{'camera': lens, 'frame_index': 1} for lens in ('L', 'R')]}
    assert config.calibration_images(scene) == []
    with pytest.raises(ValueError, match='both L/R'):
        config.calibration_images({'images': scene['images'][:1]})
    with pytest.raises(ValueError, match='calibration_mode'):
        PipelineConfig(calibration_mode='unknown')


def test_input_calibration_preserves_intrinsics_holdout_and_resume(tmp_path, monkeypatch):
    work = global_frontend(tmp_path, calibration_mode='input')

    def forbidden(*args, **kwargs):
        raise AssertionError('Input calibration must not build or optimize local geometry')

    monkeypatch.setattr(calibration, 'database_and_seed', forbidden)
    monkeypatch.setattr(calibration, 'bounded_bundle_adjustment', forbidden)
    with pytest.raises(ValueError, match='Missing pre-calibration holdout'):
        calibration.calibrate(work)
    freeze_holdout(work)
    split_path = work / 'sfm/split.json'
    before = file_digest(split_path)
    first = calibration.calibrate(work)
    assert first == calibration.calibrate(work)
    assert first['local_geometry_used'] is False
    assert not (work / 'calibration/stages').exists()
    _, scene, _, _, protocol, params = reconstruction.freeze_geometry(work)
    assert protocol['calibration_mode'] == 'input'
    for cid, lens in [(1, 'L'), (2, 'R')]:
        np.testing.assert_array_equal(params[cid], scene['cameras'][lens]['params'])
    assert file_digest(split_path) == before
    reconstruction.freeze_geometry(work, require_existing=True)
    identity_path = work / 'calibration/calibration_identity.json'
    identity = json.loads(identity_path.read_text())
    identity['cameras']['R']['params'][0] += .1
    write_json(identity_path, identity)
    with pytest.raises(ValueError, match='Input calibration identity changed'):
        reconstruction.freeze_geometry(work)


def test_input_mode_rejects_changed_holdout_and_local_artifacts(tmp_path):
    work = global_frontend(tmp_path, calibration_mode='input')
    freeze_holdout(work)
    (work / 'calibration/mapping').mkdir(parents=True)
    with pytest.raises(ValueError, match='cannot reuse local geometry'):
        calibration.calibrate(work)
    path = work / 'sfm/split.json'
    value = json.loads(path.read_text())
    value['tracks'] = []
    write_json(path, value)
    with pytest.raises(ValueError, match='Frozen holdout changed'):
        calibration.calibrate(work)


def test_new_config_requires_explicit_mode(tmp_path):
    save_config(tmp_path, PipelineConfig())
    path = tmp_path / CONFIG_FILE
    value = json.loads(path.read_text())
    del value['settings']['calibration_mode']
    value['fingerprint'] = digest({k: v for k, v in value.items() if k != 'fingerprint'})
    write_json(path, value)
    with pytest.raises(ValueError, match='Missing frozen calibration_mode'):
        load_config(tmp_path)
