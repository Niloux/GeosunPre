"""Freeze all eligible feature observations before fitting any geometry."""

import json
from pathlib import Path

import numpy as np

from geosunpre.artifacts import file_metadata, write_frozen_json
from geosunpre.config import CONFIG_FILE

SCOPE = 'all_eligible_observations'


def protocol_path(work):
    return Path(work) / 'sfm/observation_protocol.json'


def protocol_identity(work):
    return {'observation_protocol_file': file_metadata(protocol_path(work))}


def _input_identity(root, data):
    return {
        'scene_manifest_file': file_metadata(root / 'manifest.json'),
        'frontend_file': file_metadata(root / 'frontend.json'),
        'pair_protocol_file': file_metadata(root / 'pair_protocol.json'),
        # Record the caches supplying the feature and match indices used by SfM.
        'feature_cache_files': {name: file_metadata(row['path'])
                                 for name, row in sorted(data['features'].items())},
        'match_cache_files': {row['path']: file_metadata(row['path']) for row in data['pairs']},
    }


def _identity(work, data):
    return {'version': 2, 'scope': SCOPE,
            'pipeline_config_file': file_metadata(Path(work) / CONFIG_FILE),
            'inputs': _input_identity(Path(work) / 'sfm', data)}


def load_observations(work, data):
    path = protocol_path(work)
    if not path.is_file():
        raise ValueError('Missing observation protocol; run the observations stage first')
    expected = _identity(work, data)
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
    # a retained match. Completed protocols check their recorded file metadata.
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
    write_frozen_json(protocol_path(work), value)
    return value
