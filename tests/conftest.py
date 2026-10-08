"""Keep contract tests independent of installed user weights and cache state."""

import pytest


@pytest.fixture(autouse=True)
def isolated_model_cache(tmp_path, monkeypatch):
    monkeypatch.setenv('TORCH_HOME', str(tmp_path / 'torch-cache'))
    monkeypatch.setenv('HF_HUB_CACHE', str(tmp_path / 'hf-cache'))
