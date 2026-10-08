"""Frozen source integrity and demand-driven reads through pipeline interfaces."""

import json
from pathlib import Path

import numpy as np
import pytest

from geosunpre.artifacts import digest, file_digest, write_json
from geosunpre.pipeline import calibration, data, reconstruction
from geosunpre.pipeline.observations import freeze_observations
from helpers import complete_calibration, global_frontend


def feature_reads(monkeypatch, catalog, allowed):
    names = {str(Path(row['path']).resolve()): name for name, row in catalog['features'].items()}
    original = np.lib.npyio.NpzFile.__getitem__
    reads = []

    def checked(archive, key):
        name = names.get(str(Path(archive.zip.filename).resolve()))
        if name is not None and key in ('keypoints', 'scores', 'descriptors', 'normalized_keypoints'):
            assert name in allowed, f'Unexpected feature array read: {name}/{key}'
            assert key in ('keypoints', 'scores'), f'Geometry must not load descriptors: {name}/{key}'
            reads.append((name, key))
        return original(archive, key)

    monkeypatch.setattr(np.lib.npyio.NpzFile, '__getitem__', checked)
    return reads


def rewrite_cache(path, update):
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    update(arrays)
    temporary = path.with_suffix('.changed.npz')
    np.savez(temporary, **arrays)
    temporary.replace(path)


def test_verified_frontend_reads_only_requested_arrays(tmp_path, monkeypatch):
    work = global_frontend(tmp_path, frames=(70, 72, 100, 102))
    protocol = freeze_observations(work)
    catalog = json.loads((work / 'sfm/frontend.json').read_text())
    selected = list(catalog['features'])[:2]
    expected = {}
    for name in selected:
        with np.load(catalog['features'][name]['path'], allow_pickle=False) as archive:
            expected[name] = {key: archive[key] for key in ('keypoints', 'scores')}
    reads = feature_reads(monkeypatch, catalog, selected)
    _, _, _, features = data.load_frontend(work)
    assert not reads and list(features) == list(catalog['features'])
    for name in selected:
        for key in ('keypoints', 'scores'):
            np.testing.assert_array_equal(features[name][key], expected[name][key])
            assert features[name][key].dtype == expected[name][key].dtype
    assert reads == [(name, key) for name in selected for key in ('keypoints', 'scores')]
    assert json.loads((work / 'sfm/observation_protocol.json').read_text()) == protocol


def test_local_calibration_reads_only_its_segment(tmp_path, monkeypatch):
    work = global_frontend(tmp_path, frames=(70, 72, 100, 102))
    freeze_observations(work)
    catalog = json.loads((work / 'sfm/frontend.json').read_text())
    selected = set(list(catalog['features'])[:2])
    reads = feature_reads(monkeypatch, catalog, selected)
    _, scene, local, features = calibration.prepare_calibration(work)
    assert set(features) == selected == {row['name'] for row in scene['images']}
    assert {name for name, _ in reads} == selected
    assert all(row['a'] in selected and row['b'] in selected for row in local['pairs'])
    for name in selected:
        assert local['features'][name] == catalog['features'][name]


def test_geometry_verification_does_not_read_feature_arrays(tmp_path, monkeypatch):
    work = global_frontend(tmp_path)
    freeze_observations(work)
    complete_calibration(work, monkeypatch)
    catalog = json.loads((work / 'sfm/frontend.json').read_text())
    reads = feature_reads(monkeypatch, catalog, set())
    original = reconstruction.freeze_geometry(work)[4]
    assert reconstruction.freeze_geometry(work, require_existing=True)[4] == original
    assert not reads


def test_delayed_feature_read_rejects_changed_bytes_with_unchanged_token(tmp_path):
    work = global_frontend(tmp_path)
    freeze_observations(work)
    _, _, catalog, features = data.load_frontend(work)
    name = next(iter(catalog['features']))
    path = Path(catalog['features'][name]['path'])
    rewrite_cache(path, lambda arrays: arrays['keypoints'].__setitem__((0, 0), 99.))
    with pytest.raises(ValueError, match='Feature cache content changed'):
        features[name]


def test_feature_read_rejects_file_replacement_during_array_read(tmp_path, monkeypatch):
    work = global_frontend(tmp_path)
    freeze_observations(work)
    _, _, catalog, features = data.load_frontend(work)
    name = next(iter(catalog['features']))
    path = Path(catalog['features'][name]['path'])
    with np.load(path) as archive:
        replacement = {key: archive[key] for key in archive.files}
    replacement['scores'][0] = .2
    original = np.lib.npyio.NpzFile.__getitem__
    replaced = False

    def changed(archive, key):
        nonlocal replaced
        value = original(archive, key)
        if key == 'keypoints' and not replaced:
            temporary = path.with_suffix('.changed.npz')
            np.savez(temporary, **replacement)
            temporary.replace(path)
            replaced = True
        return value

    monkeypatch.setattr(np.lib.npyio.NpzFile, '__getitem__', changed)
    with pytest.raises(ValueError, match='Feature cache content changed'):
        features[name]


@pytest.mark.parametrize('missing', [False, True])
def test_frozen_identity_is_checked_before_arrays_are_exposed(tmp_path, monkeypatch, missing):
    work = global_frontend(tmp_path)
    freeze_observations(work)
    catalog = json.loads((work / 'sfm/frontend.json').read_text())
    if missing:
        (work / 'sfm/observation_protocol.json').unlink()
    else:
        path = Path(next(iter(catalog['features'].values()))['path'])
        rewrite_cache(path, lambda arrays: arrays['scores'].__setitem__(0, .2))
    reads = feature_reads(monkeypatch, catalog, set())
    with pytest.raises(ValueError, match='Missing observation protocol|Frozen observations changed'):
        data.load_frontend(work)
    assert not reads


@pytest.mark.parametrize('changed', ['catalog_token', 'cache_token', 'metadata_name', 'image_stat'])
def test_source_identity_errors_stop_before_array_reads(tmp_path, monkeypatch, changed):
    import os

    work = global_frontend(tmp_path)
    protocol = freeze_observations(work)
    catalog_path = work / 'sfm/frontend.json'
    catalog = json.loads(catalog_path.read_text())
    entry = next(iter(catalog['features'].values()))
    if changed == 'catalog_token':
        entry['token'] = 'changed'
        write_json(catalog_path, catalog)
    elif changed == 'cache_token':
        rewrite_cache(Path(entry['path']), lambda arrays: arrays.update(token=np.array('changed')))
    elif changed == 'metadata_name':
        rewrite_cache(Path(entry['path']), lambda arrays: arrays.update(metadata=json.dumps({'name': 'changed'})))
    else:
        scene = json.loads((work / 'sfm/manifest.json').read_text())
        path = Path(scene['images'][0]['source'])
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    reads = feature_reads(monkeypatch, catalog, set())
    with pytest.raises(ValueError, match='identity changed|image changed since prepare'):
        data.load_frontend(work)
    assert not reads
    assert json.loads((work / 'sfm/observation_protocol.json').read_text()) == protocol


def test_first_freeze_checks_actual_keypoint_count(tmp_path):
    work = global_frontend(tmp_path)
    catalog_path = work / 'sfm/frontend.json'
    catalog = json.loads(catalog_path.read_text())
    for row in catalog['features'].values():
        row['keypoints'] = 1000
    write_json(catalog_path, catalog)
    path = Path(catalog['pairs'][0]['path'])
    rewrite_cache(path, lambda arrays: arrays['matches'].__setitem__((0, 0), 100))
    with pytest.raises(ValueError, match='Invalid match cache'):
        freeze_observations(work)
    assert not (work / 'sfm/observation_protocol.json').exists()


def test_first_freeze_validates_images_without_retained_matches(tmp_path):
    work = global_frontend(tmp_path)
    catalog_path = work / 'sfm/frontend.json'
    catalog = json.loads(catalog_path.read_text())
    catalog['pairs'] = []
    write_json(catalog_path, catalog)
    path = Path(next(iter(catalog['features'].values()))['path'])
    rewrite_cache(path, lambda arrays: arrays.pop('keypoints'))
    with pytest.raises(KeyError, match='keypoints'):
        freeze_observations(work)
    assert not (work / 'sfm/observation_protocol.json').exists()


def test_frontend_identity_supports_an_absent_semantic_mask(tmp_path):
    work = global_frontend(tmp_path)
    root = work / 'sfm'
    scene = json.loads((root / 'manifest.json').read_text())
    catalog = json.loads((root / 'frontend.json').read_text())
    row = scene['images'][0]
    row['semantic_mask'] = None
    identity = {'image_sha256': file_digest(row['source']), 'mask_sha256': None,
                'camera': scene['cameras'][row['camera']], 'fov': scene['fov_degrees'],
                'config': catalog['config']}
    token = digest(identity)
    entry = catalog['features'][row['name']]
    entry['token'] = token
    rewrite_cache(Path(entry['path']), lambda arrays: arrays.update(token=np.array(token)))
    scene.pop('fingerprint')
    scene['fingerprint'] = digest(scene)
    catalog['scene_fingerprint'] = scene['fingerprint']
    catalog.pop('fingerprint')
    catalog['fingerprint'] = digest(catalog)
    write_json(root / 'manifest.json', scene)
    write_json(root / 'frontend.json', catalog)
    freeze_observations(work)
    _, _, _, features = data.load_frontend(work)
    assert len(features[row['name']]['keypoints']) == 20


@pytest.mark.parametrize('corrupt', [False, True])
def test_matching_reads_the_requested_feature_identity(tmp_path, monkeypatch, corrupt):
    work = global_frontend(tmp_path)
    root = work / 'sfm'
    catalog = json.loads((root / 'frontend.json').read_text())
    write_json(root / 'feature_catalog.json', catalog)
    names = list(catalog['features'])
    np.savez(root / 'retrieval_descriptors.npz', names=names, descriptors=np.ones((len(names), 2)))
    expected = {}
    for name, entry in catalog['features'].items():
        arrays = {'normalized_keypoints': np.full((20, 2), .1, dtype=np.float32),
                  'descriptors': np.full((20, 8), .2, dtype=np.float32)}
        rewrite_cache(Path(entry['path']), lambda cached: cached.update(arrays))
        expected[entry['token']] = arrays
    (root / 'pair_protocol.json').unlink()
    (root / 'frontend.json').unlink()
    if corrupt:
        rewrite_cache(Path(catalog['features'][names[0]]['path']),
                      lambda arrays: arrays.update(token=np.array('changed')))

    def match(frontend, a, b, *, feature_loader):
        for entry in (a, b):
            loaded = feature_loader(entry['token'])
            for key, values in expected[entry['token']].items():
                np.testing.assert_array_equal(loaded[key], values)
                assert loaded[key].dtype == values.dtype
        token = digest([a['token'], b['token']])
        result = {'token': np.array(token), 'matches': np.array([[0, 0]], dtype=np.uint32), 'scores': np.ones(1)}
        path = work / 'matches' / f'{token}.npz'
        path.parent.mkdir(exist_ok=True)
        np.savez(path, **result)
        return result

    monkeypatch.setattr(data.LoMaFrontend, 'match', match)
    if corrupt:
        with pytest.raises(ValueError, match='Cached feature identity changed'):
            data.match_features(work)
        assert not (root / 'frontend.json').exists()
    else:
        assert len(data.match_features(work)['pairs']) == 1
