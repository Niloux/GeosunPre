"""Frozen observation ownership: all eligible data in production, holdouts in validation."""

import json
from pathlib import Path

import numpy as np

from geosunpre.artifacts import digest, file_digest, write_frozen_json
from geosunpre.config import CONFIG_FILE, load_config
from geosunpre.pipeline.holdout import _input_identity, _reject_existing_geometry
from geosunpre.pipeline.holdout import ISOLATION_SCOPE, load_frozen_holdout


def is_validation(work):
    return load_config(work).pipeline_mode == 'validation'


def observation_scope(work):
    return ISOLATION_SCOPE if is_validation(work) else 'all_eligible_observations'


def protocol_path(work):
    name = 'holdout_protocol.json' if is_validation(work) else 'observation_protocol.json'
    return Path(work) / 'sfm' / name


def protocol_identity(work, legacy_key='global_holdout_sha256'):
    key = legacy_key if is_validation(work) else 'observation_protocol_sha256'
    return {key: file_digest(protocol_path(work))}


def geometry_partition_identity(work):
    if is_validation(work):
        return {'split_sha256': file_digest(Path(work) / 'sfm/split.json')}
    return protocol_identity(work)


def partitions(work):
    if not is_validation(work):
        # The geometry algorithms accept an exclusion list; production has none.
        # No synthetic test split is written to disk.
        return {'tracks': [], 'isolation_scope': 'all_eligible_observations'}, None
    root = Path(work) / 'sfm'
    return (json.loads((root / 'split.json').read_text()),
            json.loads((root / 'triangle_validation_split.json').read_text()))


def _reject_mixed_protocols(work):
    root = Path(work) / 'sfm'
    forbidden = (('observation_protocol.json',) if is_validation(work) else
                 ('holdout_protocol.json', 'split.json', 'triangle_validation_split.json'))
    if any((root / name).exists() for name in forbidden):
        raise ValueError('Observation mode differs from existing artifacts; use a fresh work directory')


def _production_identity(work, data):
    return {'version': 1, 'mode': 'production', 'scope': 'all_eligible_observations',
            'excluded_for_validation': 0,
            'pipeline_config_sha256': file_digest(Path(work) / CONFIG_FILE),
            'inputs': _input_identity(Path(work) / 'sfm', data)}


def load_observations(work, data):
    _reject_mixed_protocols(work)
    if is_validation(work):
        return load_frozen_holdout(work, data)
    path = protocol_path(work)
    if not path.is_file():
        raise ValueError('Missing production observation protocol; run the observations stage first')
    expected = _production_identity(work, data)
    expected['fingerprint'] = digest(expected)
    if json.loads(path.read_text()) != expected:
        raise ValueError('Frozen production observations changed; use a fresh work directory')
    split, triangle = partitions(work)
    return expected, split, triangle


def freeze_observations(work):
    from geosunpre.pipeline.data import load_frontend

    work = Path(work).resolve()
    if is_validation(work):
        raise ValueError('Use the holdout stage in validation mode')
    _reject_mixed_protocols(work)
    if not protocol_path(work).exists():
        _reject_existing_geometry(work)
    _, _, data, features = load_frontend(work)
    if protocol_path(work).exists():
        return load_observations(work, data)[0]
    value = _production_identity(work, data)
    for entry in data['pairs']:
        with np.load(entry['path'], allow_pickle=False) as raw:
            matches, scores = raw['matches'], raw['scores']
            if (str(raw['token']) != entry['token'] or matches.ndim != 2
                    or matches.shape[1] != 2 or not np.issubdtype(matches.dtype, np.integer)
                    or scores.shape != (len(matches),) or not np.isfinite(scores).all()
                    or (matches < 0).any()
                    or (matches[:, 0] >= len(features[entry['a']]['keypoints'])).any()
                    or (matches[:, 1] >= len(features[entry['b']]['keypoints'])).any()):
                raise ValueError(f'Invalid production match cache: {entry["a"]}, {entry["b"]}')
    if value != _production_identity(work, data):
        raise ValueError('Frontend changed while freezing production observations')
    value['fingerprint'] = digest(value)
    write_frozen_json(protocol_path(work), value)
    return value
