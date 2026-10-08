"""Freeze all eligible feature observations before fitting any geometry."""

import json
from pathlib import Path

import numpy as np

from geosunpre.artifacts import digest, file_digest, write_frozen_json
from geosunpre.config import CONFIG_FILE

SCOPE = 'all_eligible_observations'


def protocol_path(work):
    return Path(work) / 'sfm/observation_protocol.json'


def protocol_identity(work):
    return {'observation_protocol_sha256': file_digest(protocol_path(work))}


def _input_identity(root, data):
    return {
        'scene_manifest_sha256': file_digest(root / 'manifest.json'),
        'frontend_identity': digest({k: v for k, v in data.items()
                                     if k not in ('fingerprint', 'cache', 'matching_cache')}),
        'pair_protocol_sha256': file_digest(root / 'pair_protocol.json'),
        # Tokens describe extraction inputs. Hash cache bytes as well, so reordered
        # keypoints or changed matches cannot retain their old observation IDs.
        'feature_cache_sha256': {name: file_digest(row['path'])
                                 for name, row in sorted(data['features'].items())},
        'match_cache_sha256': {row['path']: file_digest(row['path']) for row in data['pairs']},
    }


def _identity(work, data):
    return {'version': 2, 'scope': SCOPE,
            'pipeline_config_sha256': file_digest(Path(work) / CONFIG_FILE),
            'inputs': _input_identity(Path(work) / 'sfm', data)}


def load_observations(work, data):
    path = protocol_path(work)
    if not path.is_file():
        raise ValueError('Missing observation protocol; run the observations stage first')
    expected = _identity(work, data)
    expected['fingerprint'] = digest(expected)
    if json.loads(path.read_text()) != expected:
        raise ValueError('Frozen observations changed; use a fresh work directory')
    return expected


def freeze_observations(work):
    from geosunpre.pipeline.data import load_frontend

    work = Path(work).resolve()
    if not protocol_path(work).exists() and any((work / relative).exists() for relative in (
            'sfm/geometry_protocol.json', 'sfm/database.db', 'sfm/mapping',
            'sfm/stages', 'sfm/final', 'calibration/run_contract.json')):
        raise ValueError('Geometry predates observation ownership; use a fresh work directory')
    _, _, data, features = load_frontend(work, require_observations=False)
    if protocol_path(work).exists():
        return load_observations(work, data)
    value = _identity(work, data)
    # First publication validates every feature array, including images without
    # a retained match. Completed protocols need only their content identities.
    features = dict(features)
    for entry in data['pairs']:
        with np.load(entry['path'], allow_pickle=False) as raw:
            matches, scores = raw['matches'], raw['scores']
            if (str(raw['token']) != entry['token'] or matches.ndim != 2
                    or matches.shape[1] != 2 or not np.issubdtype(matches.dtype, np.integer)
                    or scores.shape != (len(matches),) or not np.isfinite(scores).all()
                    or (matches < 0).any()
                    or (matches[:, 0] >= len(features[entry['a']]['keypoints'])).any()
                    or (matches[:, 1] >= len(features[entry['b']]['keypoints'])).any()):
                raise ValueError(f'Invalid match cache: {entry["a"]}, {entry["b"]}')
    if value != _identity(work, data):
        raise ValueError('Frontend changed while freezing observations')
    value['fingerprint'] = digest(value)
    write_frozen_json(protocol_path(work), value)
    return value
