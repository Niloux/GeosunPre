from dataclasses import asdict

import numpy as np
import pytest

from geosunpre.artifacts import digest, write_json
from geosunpre.config import PipelineConfig, load_config, save_config
from geosunpre.features import FrontendConfig, LoMaFrontend


def test_attention_is_frozen_in_pipeline_config(tmp_path):
    assert PipelineConfig().matcher_attention == 'flash'
    save_config(tmp_path, PipelineConfig())
    assert load_config(tmp_path).matcher_attention == 'flash'
    with pytest.raises(ValueError):
        save_config(tmp_path, PipelineConfig(matcher_attention='math'))
    with pytest.raises(ValueError):
        PipelineConfig(matcher_attention='unknown')


def test_legacy_math_configuration_keeps_its_frozen_bytes(tmp_path):
    settings = asdict(PipelineConfig())
    del settings['matcher_attention']
    value = {'version': 1, 'settings': settings}
    path = tmp_path / 'pipeline_config.json'
    write_json(path, {**value, 'fingerprint': digest(value)})
    before = path.read_bytes()
    save_config(tmp_path, PipelineConfig(matcher_attention='math'))
    assert path.read_bytes() == before
    assert load_config(tmp_path).matcher_attention == 'math'
    with pytest.raises(ValueError):
        save_config(tmp_path, PipelineConfig(matcher_attention='flash'))


def test_version_one_explicit_flash_is_preserved(tmp_path):
    value = {'version': 1, 'settings': asdict(PipelineConfig(matcher_attention='flash'))}
    write_json(tmp_path / 'pipeline_config.json', {**value, 'fingerprint': digest(value)})
    assert load_config(tmp_path).matcher_attention == 'flash'


def test_version_two_requires_explicit_frozen_backend(tmp_path):
    settings = asdict(PipelineConfig())
    del settings['matcher_attention']
    value = {'version': 2, 'settings': settings}
    write_json(tmp_path / 'pipeline_config.json', {**value, 'fingerprint': digest(value)})
    with pytest.raises(ValueError, match='Missing frozen matcher_attention'):
        load_config(tmp_path)


def test_native_default_does_not_silently_fall_back(monkeypatch):
    import sys
    from geosunpre.sfm.depth_guard import PositiveDepthCost, depth_cost_class
    monkeypatch.delenv('GEOSUNPRE_DEPTH_BACKEND', raising=False)
    monkeypatch.setitem(sys.modules, 'geosunpre_native', None)
    with pytest.raises(ImportError):
        depth_cost_class()
    monkeypatch.setenv('GEOSUNPRE_DEPTH_BACKEND', 'python')
    assert depth_cost_class() is PositiveDepthCost


def test_flash_does_not_reuse_math_match_cache(tmp_path):
    config = FrontendConfig()
    base = {'a': 'feature-a', 'b': 'feature-b', 'config': asdict(config)}
    (tmp_path / 'matches').mkdir()
    for backend, expected in [('math', 11), ('flash', 22)]:
        identity = dict(base)
        if backend == 'flash':
            identity['attention_backend'] = 'flash_v1'
        token = digest(identity)
        np.savez(tmp_path / 'matches' / f'{token}.npz', token=token, matches=[[expected, 0]])
    for backend, expected in [('math', 11), ('flash', 22)]:
        frontend = LoMaFrontend(tmp_path, {}, config, matching_attention=backend)
        actual = frontend.match({'token': base['a']}, {'token': base['b']})
        assert actual['matches'][0, 0] == expected
        assert frontend.cache_stats['match_hits'] == 1
