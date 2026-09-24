"""Bounded-memory preparation of all pose-bound L/R native images."""
from dataclasses import asdict
from datetime import datetime, timezone
from functools import lru_cache
import json
import os
from pathlib import Path
import shutil

import numpy as np

from ..frontend import LoMaFrontend
from ..local_data import CONFIG, local_pairs
from ..scene import digest, file_digest, load_scene, write_json

OUTPUT = 'global_sfm'


def prepare_global(work):
    from ..static_masks import generate_masks, mask_review, mask_contact_sheet
    work = Path(work).resolve(); out = work / OUTPUT; out.mkdir(exist_ok=True)
    base = load_scene(work)
    protocol = {'version': 1, 'source_manifest_sha256': file_digest(work / 'manifest.json'),
                'images': [r['name'] for r in base['images']], 'frontend': asdict(CONFIG),
                'mask': 'Reuse v2 cached sky/person/device masks; same initial 160-degree domain for feature extraction.',
                'geometry': 'Only independent exposure poses; input pose values and LAS excluded.',
                'scope': 'Full selected image preparation. Calibration and final geometry acceptance remain separate.'}
    fingerprint = digest(protocol)
    if (out / 'preparation_protocol.json').exists():
        if json.loads((out / 'preparation_protocol.json').read_text())['fingerprint'] != fingerprint:
            raise ValueError('Global input protocol changed; preserve old outputs')
    else:
        write_json(out / 'preparation_protocol.json', {**protocol, 'fingerprint': fingerprint,
                   'frozen_at_utc': datetime.now(timezone.utc).isoformat()})
    reused = []
    for name in ['local_sfm', 'transfer_sfm']:
        previous = work / name
        for record in load_scene(previous)['images']:
            for kind, suffix in [('valid', '.png'), ('labels', '.png'), ('stats', '.json')]:
                src = previous / 'masks_v2' / kind / (record['name'] + suffix)
                dest = out / 'masks_v2' / kind / (record['name'] + suffix)
                dest.parent.mkdir(parents=True, exist_ok=True)
                if not dest.exists(): shutil.copy2(src, dest)
                if file_digest(src) != file_digest(dest): raise ValueError('Conflicting reused mask')
            reused.append(record['name'])
    images, metadata = generate_masks(base, out / 'masks_v2', sorted({r['frame_index'] for r in base['images']}),
                                      model_cache=work / 'local_sfm/masks/model_cache')
    for record in images:
        stats = json.loads((out / 'masks_v2/stats' / (record['name'] + '.json')).read_text())
        if stats['source_sha256'] != file_digest(record['source']) or stats['mask_sha256'] != file_digest(record['semantic_mask']):
            raise ValueError(f'Stale global mask for {record["name"]}')
    scene = {**base, 'images': images, 'semantic_masks_available': True, 'segmentation': metadata,
             'intrinsics_are_ground_truth': False, 'scope': 'Full selected asynchronous L/R sequence.'}
    scene.pop('fingerprint'); scene['fingerprint'] = digest(scene)
    write_json(out / 'manifest.json', scene)
    write_json(out / 'reused_masks.json', {'count': len(reused), 'image_names': reused})
    frames = [r['frame_index'] for r in images]
    mask_review(scene, out, segments={'all': (min(frames), max(frames))})
    mask_contact_sheet(scene, images, out / 'masks_v2/contact_sheet.jpg', [33, 70, 120, 170, 220, 260, 300])
    return out


def extract_global_features(work):
    work = Path(work).resolve(); out = work / OUTPUT; scene = load_scene(out)
    frontend = LoMaFrontend(work, scene, CONFIG)
    catalog = {}; retrieval = []
    for i, record in enumerate(scene['images'], 1):
        f = frontend.extract(record); token = str(f['token'])
        catalog[record['name']] = {'token': token, 'path': str(frontend.cache / f'{token}.npz'),
                                  'keypoints': len(f['keypoints'])}
        vector = f['descriptors'].astype(np.float64).mean(axis=0)
        vector /= max(np.linalg.norm(vector), 1e-12)
        retrieval.append(vector.astype(np.float32))
        print(f'global features {i}/{len(scene["images"])} {record["name"]}', flush=True)
        del f
    path = out / 'retrieval_descriptors.npz'
    names, vectors = np.array(list(catalog)), np.array(retrieval)
    if path.exists():
        with np.load(path, allow_pickle=False) as old:
            if not np.array_equal(old['names'], names) or not np.array_equal(old['descriptors'], vectors):
                raise ValueError('Global retrieval vectors changed; preserve the existing run')
    else:
        np.savez_compressed(path, names=names, descriptors=vectors)
    result = {'version': 1, 'scene_fingerprint': scene['fingerprint'], 'config': asdict(CONFIG),
              'features': catalog, 'cache': frontend.cache_stats,
              'retrieval': 'Mean of retained LoMa descriptors, L2 normalized; a candidate-pair heuristic only.'}
    result['fingerprint'] = digest({k: v for k, v in result.items() if k != 'cache'})
    path = out / 'feature_catalog.json'
    if path.exists() and json.loads(path.read_text())['fingerprint'] != result['fingerprint']:
        raise ValueError('Global feature catalog changed; preserve the existing run')
    write_json(path, result)
    return result


def global_pairs(images, names, descriptors):
    by_name = {r['name']: r for r in images}
    rank = {r['name']: i for i, r in enumerate(images)}
    pairs = {}
    def add(a, b, reason):
        a, b = sorted((a, b), key=lambda n: rank[n])
        if a != b: pairs.setdefault((a, b), set()).add(reason)
    for a, b in local_pairs(images): add(a['name'], b['name'], 'temporal')
    similarity = descriptors @ descriptors.T
    for i, name in enumerate(names):
        record = by_name[str(name)]
        for camera in ['L', 'R']:
            eligible = [j for j, n in enumerate(names) if by_name[str(n)]['camera'] == camera
                        and abs(by_name[str(n)]['frame_index'] - record['frame_index']) >= 25]
            top = sorted(eligible, key=lambda j: (-float(similarity[i, j]), str(names[j])))[:4]
            for j in top: add(str(name), str(names[j]), 'appearance_retrieval')
    first = min(r['frame_index'] for r in images); last = max(r['frame_index'] for r in images)
    for a in images:
        if a['frame_index'] > first + 3: continue
        for b in images:
            if b['frame_index'] >= last - 3: add(a['name'], b['name'], 'endpoint_candidate')
    return [{'a': a, 'b': b, 'reasons': sorted(reasons)} for (a, b), reasons in sorted(pairs.items(), key=lambda e: (rank[e[0][0]], rank[e[0][1]]))]


def match_global_features(work):
    work = Path(work).resolve(); out = work / OUTPUT; scene = load_scene(out)
    catalog = json.loads((out / 'feature_catalog.json').read_text())
    if catalog['scene_fingerprint'] != scene['fingerprint']: raise ValueError('Global scene changed')
    with np.load(out / 'retrieval_descriptors.npz') as retrieval:
        pairs = global_pairs(scene['images'], retrieval['names'], retrieval['descriptors'])
    pair_protocol = {'pairs': pairs, 'feature_catalog_fingerprint': catalog['fingerprint'],
                     'retrieval_descriptors_sha256': file_digest(out / 'retrieval_descriptors.npz'),
                     'minimum_retrieval_frame_gap': 25, 'retrieval_neighbors_per_camera': 4,
                     'selection_uses_input_pose_or_las': False}
    fingerprint = digest(pair_protocol)
    if (out / 'pair_protocol.json').exists():
        if json.loads((out / 'pair_protocol.json').read_text())['fingerprint'] != fingerprint:
            raise ValueError('Global pair protocol changed')
    else: write_json(out / 'pair_protocol.json', {**pair_protocol, 'fingerprint': fingerprint})
    frontend = LoMaFrontend(work, scene, CONFIG)
    @lru_cache(maxsize=16)
    def get_features(name):
        with np.load(catalog['features'][name]['path'], allow_pickle=False) as f:
            return {k: f[k] for k in f.files}
    entries = []
    for i, pair in enumerate(pairs, 1):
        raw = frontend.match(get_features(pair['a']), get_features(pair['b']))
        entries.append({**pair, 'token': str(raw['token']), 'path': str(work / 'matches' / f'{str(raw["token"])}.npz')})
        if i % 40 == 0 or i == len(pairs): print(f'global matches {i}/{len(pairs)}', flush=True)
    result = {**catalog, 'pairs': entries, 'pair_protocol_fingerprint': fingerprint,
              'matching_cache': frontend.cache_stats}
    result['fingerprint'] = digest({k: v for k, v in result.items() if k not in ['fingerprint', 'cache', 'matching_cache']})
    path = out / 'frontend.json'
    if path.exists() and json.loads(path.read_text())['fingerprint'] != result['fingerprint']:
        raise ValueError('Global match catalog changed; preserve the existing run')
    write_json(path, result)
    return result
