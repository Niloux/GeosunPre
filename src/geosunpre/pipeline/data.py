"""Bounded-memory preparation of all pose-bound L/R native images."""
from dataclasses import asdict
from functools import lru_cache
from collections.abc import Mapping
import json
from pathlib import Path
from geosunpre.profiling import profiled

import numpy as np

from geosunpre.features import LoMaFrontend, feature_input, read_feature_cache
from geosunpre.features import FRONTEND_CONFIG as CONFIG
from geosunpre.artifacts import file_metadata, write_frozen_json, write_json
from geosunpre.scene import load_scene

OUTPUT = 'sfm'


class _FeatureArrays(Mapping):
    """A checked catalog whose geometry arrays are read once, when requested."""
    def __init__(self, sources, files):
        self._sources = sources
        self._files = files
        self._loaded = {}

    def __iter__(self):
        return iter(self._sources)

    def __len__(self):
        return len(self._sources)

    def __getitem__(self, name):
        if name not in self._loaded:
            row = self._sources[name]
            self._loaded[name] = read_feature_cache(row['path'], row['token'], name=name,
                fields=('keypoints', 'scores'),
                identity=row['identity'],
                expected_file=self._files[name] if self._files is not None else None)
        return self._loaded[name]


@profiled
def load_frontend(work, *, require_observations=True):
    """Verify frozen sources and expose geometry arrays on first named access.

    Only the observation-freezing stage may bootstrap without a frozen protocol.
    Metadata is always checked; the completed protocol binds later reads to the
    exact cache bytes it verified.
    """
    work = Path(work).resolve()
    out = work / OUTPUT
    scene = load_scene(out)
    data = json.loads((out / 'frontend.json').read_text(encoding='utf-8'))
    if data['scene_manifest_file'] != file_metadata(out / 'manifest.json') or data['config'] != asdict(CONFIG):
        raise ValueError('Global frontend does not match current scene/configuration')
    sources = {}
    for record in scene['images']:
        name = record['name']
        row = data['features'][name]
        _, identity, token, expected_path = feature_input(work, scene, CONFIG, record)
        if token != row['token'] or str(expected_path) != row['path']:
            raise ValueError(f'Global feature identity changed: {name}')
        read_feature_cache(row['path'], token, name=name, fields=(), identity=identity)
        sources[name] = {'path': row['path'], 'token': token, 'identity': identity}
    files = None
    if require_observations:
        from geosunpre.pipeline.observations import load_observations
        files = load_observations(work, data)['inputs']['feature_cache_files']
    return out, scene, data, _FeatureArrays(sources, files)


def prepare_masks(work, *, model_cache=None):
    from geosunpre.masks import generate_masks
    from geosunpre.masks import mask_review
    from geosunpre.masks import mask_contact_sheet
    work = Path(work).resolve(); out = work / OUTPUT; out.mkdir(exist_ok=True)
    base = load_scene(work)
    protocol = {'version': 1, 'source_manifest_file': file_metadata(work / 'manifest.json'),
                'images': [r['name'] for r in base['images']], 'frontend': asdict(CONFIG),
                'mask': 'Automatic sky/person segmentation; initial 160-degree feature domain.',
                'geometry': 'Only independent exposure poses; input pose values and LAS excluded.',
                'scope': 'Full selected image preparation. Calibration and final geometry acceptance remain separate.'}
    write_frozen_json(out / 'preparation_protocol.json', protocol)
    frames = sorted({record['frame_index'] for record in base['images']})
    images, metadata = generate_masks(base, out / 'masks', frames, model_cache=model_cache)
    scene = {**base, 'images': images, 'semantic_masks_available': True, 'segmentation': metadata,
             'intrinsics_are_ground_truth': False, 'scope': 'Full selected asynchronous L/R sequence.'}
    write_frozen_json(out / 'manifest.json', scene)
    frames = [r['frame_index'] for r in images]
    review_complete = ((out / 'mask_review.html').is_file() and (out / 'masks/summary.json').is_file()
                       and all((out / 'mask_review' / f'{r["camera"]}{r["frame_index"]:05d}.jpg').is_file()
                               for r in images))
    if not review_complete:
        mask_review(scene, out, segments={'all': (min(frames), max(frames))})
    available = sorted(set(frames))
    selected = [available[i] for i in np.linspace(0, len(available) - 1, min(7, len(available)), dtype=int)]
    if not (out / 'masks/contact_sheet.jpg').is_file():
        mask_contact_sheet(scene, images, out / 'masks/contact_sheet.jpg', selected)
    return out


def extract_features(work):
    work = Path(work).resolve(); out = work / OUTPUT; scene = load_scene(out)
    frontend = LoMaFrontend(work, scene, CONFIG)
    catalog = {}; retrieval = []
    for i, (record, f) in enumerate(zip(scene['images'], frontend.extract_all(scene['images'])), 1):
        token = str(f['token'])
        catalog[record['name']] = {'token': token, 'path': str(frontend.cache / f'{token}.npz'),
                                  'keypoints': len(f['keypoints'])}
        vector = f['descriptors'].astype(np.float64).mean(axis=0)
        vector /= max(np.linalg.norm(vector), 1e-12)
        retrieval.append(vector.astype(np.float32))
        if i == 1 or i % 40 == 0 or i == len(scene['images']):
            print(f'global features {i}/{len(scene["images"])}', flush=True)
        del f
    path = out / 'retrieval_descriptors.npz'
    names, vectors = np.array(list(catalog)), np.array(retrieval)
    if path.exists():
        with np.load(path, allow_pickle=False) as old:
            if not np.array_equal(old['names'], names) or not np.array_equal(old['descriptors'], vectors):
                raise ValueError('Global retrieval vectors changed; preserve the existing run')
    else:
        np.savez_compressed(path, names=names, descriptors=vectors)
    result = {'version': 1, 'scene_manifest_file': file_metadata(out / 'manifest.json'), 'config': asdict(CONFIG),
              'features': catalog, 'cache': frontend.cache_stats,
              'retrieval': 'Mean of retained LoMa descriptors, L2 normalized; a candidate-pair heuristic only.'}
    path = out / 'feature_catalog.json'
    _save_catalog(path, result)
    return result


def match_features(work):
    from geosunpre.sfm.pair_selection import PairSelection
    work = Path(work).resolve(); out = work / OUTPUT; scene = load_scene(out)
    catalog = json.loads((out / 'feature_catalog.json').read_text(encoding='utf-8'))
    if catalog['scene_manifest_file'] != file_metadata(out / 'manifest.json'): raise ValueError('Global scene changed')
    with np.load(out / 'retrieval_descriptors.npz') as retrieval:
        selection = PairSelection(scene['images'], retrieval['names'], retrieval['descriptors'])
    frontend = LoMaFrontend(work, scene, CONFIG)
    by_token = {row['token']: row['path'] for row in catalog['features'].values()}
    identities = {}
    for record in scene['images']:
        _, identity, token, expected_path = feature_input(work, scene, CONFIG, record)
        if by_token.get(token) != str(expected_path):
            raise ValueError(f'Global feature catalog changed: {record["name"]}')
        read_feature_cache(expected_path, token, name=token, fields=(), identity=identity)
        identities[token] = identity

    @lru_cache(maxsize=32)
    def get_features(token):
        return read_feature_cache(by_token[token], token, name=token, identity=identities[token],
            fields=('normalized_keypoints', 'descriptors', 'token'))

    matched = 0

    def match(pair):
        nonlocal matched
        raw = frontend.match({'token': catalog['features'][pair['a']]['token']},
                             {'token': catalog['features'][pair['b']]['token']},
                             feature_loader=get_features)
        entry = {**pair, 'token': str(raw['token']), 'path': str(work / 'matches' / f'{str(raw["token"])}.npz')}
        matched += 1
        if matched % 40 == 0:
            print(f'global matches {matched}', flush=True)
        return entry, int((raw['scores'] >= .5).sum())

    entries, report = selection.match(match)
    print(f'global matches complete: {len(entries)} pairs', flush=True)
    pair_protocol = {**report, 'pairs': [{k: row[k] for k in ('a', 'b', 'reasons')} for row in entries],
                     'feature_catalog_file': file_metadata(out / 'feature_catalog.json'),
                     'retrieval_descriptors_file': file_metadata(out / 'retrieval_descriptors.npz')}
    write_frozen_json(out / 'pair_protocol.json', pair_protocol)
    result = {**catalog, 'pairs': entries, 'pair_protocol_file': file_metadata(out / 'pair_protocol.json'),
              'matching_cache': frontend.cache_stats}
    result['matcher_attention'] = 'flash'
    path = out / 'frontend.json'
    _save_catalog(path, result)
    return result


def _save_catalog(path, result):
    """Keep completed catalog metadata stable when only cache counters change."""
    if path.is_file():
        saved = json.loads(path.read_text(encoding='utf-8'))
        inputs = lambda value: {k: v for k, v in value.items() if k not in ('cache', 'matching_cache')}
        if inputs(saved) != inputs(result):
            raise ValueError(f'Frontend catalog changed; use a fresh work directory: {path}')
    else:
        write_json(path, result)
