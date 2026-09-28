"""Freeze one observation holdout before either local or global geometry."""

import json
from pathlib import Path

import numpy as np

from ..local_data import split_tracks
from ..local_evaluation import triangle_validation_split
from ..scene import digest, file_digest, write_json


POLICY = {'fraction': .2, 'seed': 314159, 'triangle_direct_match_confidence': .9}
ISOLATION_SCOPE = 'before_local_and_global_geometry'


def write_frozen_json(path, value):
    """Keep completed identities immutable; publish individual JSON files atomically."""
    path = Path(path)
    if path.exists():
        if digest(json.loads(path.read_text())) != digest(value):
            raise ValueError(f'Frozen inputs changed; use a fresh work directory: {path}')
    else:
        write_json(path, value)


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


def load_frozen_holdout(work, data):
    """Verify the frozen split and its feature/match identities without rewriting it."""
    root = Path(work).resolve() / 'global_sfm'
    path = root / 'holdout_protocol.json'
    if not path.is_file():
        raise ValueError('Missing pre-calibration holdout; run the holdout stage in a fresh work directory')
    protocol = json.loads(path.read_text())
    if (protocol.get('version') != 1 or protocol.get('policy') != POLICY
            or protocol.get('isolation_scope') != ISOLATION_SCOPE
            or protocol.get('fingerprint') != digest({k: v for k, v in protocol.items()
                                                      if k != 'fingerprint'})
            or protocol.get('inputs') != _input_identity(root, data)):
        raise ValueError('Frozen holdout inputs changed; use a fresh work directory')
    for name in ('split.json', 'triangle_validation_split.json'):
        if file_digest(root / name) != protocol['split_sha256'][name]:
            raise ValueError(f'Frozen holdout changed: {name}')
    return (protocol, json.loads((root / 'split.json').read_text()),
            json.loads((root / 'triangle_validation_split.json').read_text()))


def _reject_existing_geometry(work):
    root = work / 'global_sfm'
    if any((root / name).exists() for name in (
            'geometry_protocol.json', 'database.db', 'seed', 'mapping', 'mapper_output',
            'stages', 'baseline_equal', 'ground_joint_ba', 'lfs_ab')):
        raise ValueError('Geometry predates the unified holdout; use a fresh work directory')
    local = work / 'local_sfm'
    if any(local.glob('*/run_contract.json')) or any(local.rglob('database.db')) or any(local.rglob('cameras.bin')):
        raise ValueError('Local calibration predates the unified holdout; use a fresh work directory')


def freeze_global_holdout(work):
    """Select raw tracks and triangle queries using matches only, before calibration."""
    from .global_data import load_global

    work = Path(work).resolve()
    root = work / 'global_sfm'
    if not (root / 'holdout_protocol.json').exists():
        _reject_existing_geometry(work)
    root, scene, data, features = load_global(work)
    if (root / 'holdout_protocol.json').exists():
        protocol, _, _ = load_frozen_holdout(work, data)
        return protocol
    inputs = _input_identity(root, data)
    pairs = []
    for entry in data['pairs']:
        a, b = entry['a'], entry['b']
        with np.load(entry['path'], allow_pickle=False) as raw:
            matches, scores = raw['matches'], raw['scores']
            if str(raw['token']) != entry['token']:
                raise ValueError(f'Match cache identity changed: {a}, {b}')
            if (matches.ndim != 2 or matches.shape[1] != 2
                    or not np.issubdtype(matches.dtype, np.integer)
                    or scores.shape != (len(matches),) or not np.isfinite(scores).all()
                    or (matches < 0).any()
                    or (matches[:, 0] >= len(features[a]['keypoints'])).any()
                    or (matches[:, 1] >= len(features[b]['keypoints'])).any()):
                raise ValueError(f'Invalid match observation IDs: {a}, {b}')
            pairs.append((a, b, matches, scores))
    split, held = split_tracks(pairs, scene['images'], POLICY['fraction'], POLICY['seed'])
    split['isolation_scope'] = ISOLATION_SCOPE
    records = {row['name']: row for row in scene['images']}
    triangle = triangle_validation_split(
        split, data['pairs'], records, threshold=POLICY['triangle_direct_match_confidence'])
    triangle['selection_frozen_before_geometry'] = True
    if inputs != _input_identity(root, data):
        raise ValueError('Frontend changed while freezing the holdout')
    write_frozen_json(root / 'split.json', split)
    write_frozen_json(root / 'triangle_validation_split.json', triangle)
    summary = {'raw_heldout_tracks': len(split['tracks']), 'heldout_observations': len(held),
               'raw_queries': sum(len(t['queries']) for t in split['tracks']),
               'direct_triangle_tracks': len(triangle['tracks']),
               'triangle_queries': sum(len(t['queries']) for t in triangle['tracks']),
               'frontend_pairs': len(data['pairs']), 'isolation_scope': ISOLATION_SCOPE,
               'split_sha256': file_digest(root / 'split.json'),
               'triangle_split_sha256': file_digest(root / 'triangle_validation_split.json')}
    write_frozen_json(root / 'split_summary.json', summary)
    protocol = {'version': 1, 'policy': POLICY, 'isolation_scope': ISOLATION_SCOPE,
                'inputs': inputs, 'split_sha256': {
                    name: file_digest(root / name)
                    for name in ('split.json', 'triangle_validation_split.json')}}
    protocol['fingerprint'] = digest(protocol)
    # The protocol is the completion marker; an interrupted split write can be
    # retried only while no geometry exists and its content is identical.
    write_frozen_json(root / 'holdout_protocol.json', protocol)
    print('Frozen pre-calibration holdout:', summary, flush=True)
    return protocol


def restrict_split(split, images):
    """Project the global exclusion set onto local images without losing short tracks."""
    records = {row['name']: row for row in images}
    tracks = []
    for track in split['tracks']:
        nodes = [node for node in track['observations'] if node[0] in records]
        if not nodes:
            continue
        ordered = sorted(nodes, key=lambda n: (records[n[0]]['frame_index'], n[0], n[1]))
        support, queries = [], []
        if len(nodes) >= 3 and (records[ordered[-1][0]]['frame_index']
                                - records[ordered[0][0]]['frame_index'] >= 2):
            support = [ordered[0], ordered[-1]]
            queries = ordered[1:-1]
        # Even a single local observation of a global holdout track is excluded
        # from fitting. It simply cannot form a local validation query.
        tracks.append({'id': track['id'], 'observations': nodes,
                       'support': support, 'queries': queries})
    return {**split, 'tracks': tracks, 'parent_split_digest': digest(split),
            'heldout_observations': sum(len(t['observations']) for t in tracks),
            'scope': 'Local projection of global heldout observations; supports chosen by frame order.'}
