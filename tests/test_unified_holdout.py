import json
from pathlib import Path

import numpy as np
from PIL import Image
import pycolmap
import pytest

from geosunpre.camera import Camera
from geosunpre.config import PipelineConfig, save_config
from geosunpre.sfm.tracks import CONFIG
from geosunpre.sfm.tracks import train_edge_mask
from geosunpre.sfm.evaluation import evaluate_reconstruction
from geosunpre.sfm.reconstruction import assert_holdout_isolated
from geosunpre.sfm.reconstruction import create_seed
from geosunpre.pipeline import runner as runner
from geosunpre.pipeline import reconstruction as global_reconstruction
from geosunpre.pipeline import calibration as local_calibration
from geosunpre.pipeline.holdout import freeze_holdout
from geosunpre.pipeline.holdout import load_frozen_holdout
from geosunpre.pipeline.holdout import restrict_split
from geosunpre.artifacts import digest
from geosunpre.artifacts import file_digest
from geosunpre.artifacts import write_json


def global_frontend(tmp_path, *, frames=(70, 72, 110, 112), calibration_mode='local',
                    pipeline_mode='validation', registration_mode='visual'):
    """Small real image/cache fixture; no LoMa, GPU or reconstruction is run."""
    work = tmp_path / 'run'
    root = work / 'sfm'
    root.mkdir(parents=True)
    save_config(work, PipelineConfig(calibration_frames=(70, 94), calibration_mode=calibration_mode,
                                     pipeline_mode=pipeline_mode, registration_mode=registration_mode))
    cameras = {lens: Camera(lens, 8, 8, (4., 4., 4., 4., 0., 0., 0., 0.), (0., 0.))
               for lens in ('L', 'R')}
    xyz = np.random.default_rng(4).uniform([-.8, -.8, 3.], [.8, .8, 5.], (20, 3))
    images, catalog = [], {}
    for index, frame in enumerate(frames):
        lens = 'L' if index % 2 == 0 else 'R'
        name = f'{lens}/{frame}.jpg'
        source = tmp_path / name
        source.parent.mkdir(exist_ok=True)
        Image.new('RGB', (8, 8), (40, 80, 120)).save(source)
        mask = source.with_suffix('.png')
        Image.new('L', (8, 8), 255).save(mask)
        images.append({'name': name, 'camera': lens, 'frame_index': frame,
                       'image_id': index + 1, 'source': str(source), 'semantic_mask': str(mask)})
        token = digest({'image_sha256': file_digest(source), 'mask_sha256': file_digest(mask),
                        'camera': cameras[lens].to_dict(), 'fov': 160, 'config': CONFIG.__dict__})
        # Identical images with the same lens can share extraction inputs, but
        # feature metadata still needs the actual source name in this fixture.
        cache = root / f'features-{index}.npz'
        keypoints = cameras[lens].project(xyz + [index * .3, 0., 0.])
        np.savez(cache, token=token, metadata=json.dumps({'name': name}),
                 keypoints=keypoints, scores=np.ones(20))
        catalog[name] = {'token': token, 'path': str(cache), 'keypoints': 20}
    scene = {'contract_version': 1, 'pose_model': 'independent_exposure',
             'input_root': str(tmp_path), 'fov_degrees': 160,
             'cameras': {k: v.to_dict() for k, v in cameras.items()}, 'images': images}
    scene['fingerprint'] = digest(scene)
    write_json(root / 'manifest.json', scene)
    pairs = []
    for i, a in enumerate(images):
        for b in images[i + 1:]:
            token = digest([a['name'], b['name']])
            path = root / f'{token}.npz'
            np.savez(path, token=token, matches=np.column_stack((np.arange(20), np.arange(20))),
                     scores=np.full(20, .95))
            pairs.append({'a': a['name'], 'b': b['name'], 'token': token, 'path': str(path)})
    data = {'scene_fingerprint': scene['fingerprint'], 'config': CONFIG.__dict__,
            'features': catalog, 'pairs': pairs}
    data['fingerprint'] = digest(data)
    write_json(root / 'frontend.json', data)
    write_json(root / 'pair_protocol.json', {'pairs': pairs})
    return work


def complete_calibration(work, monkeypatch):
    """Use real pipeline contracts/models/evaluation with only mapping and BA stubbed."""
    calls = []

    def database(directory, scene, images, entries, features, split):
        held = {(n, int(i)) for track in split['tracks'] for n, i in track['observations']}
        a, b = images[:2]
        matches = np.column_stack((np.arange(20), np.arange(20)))
        matches = matches[train_edge_mask(a['name'], b['name'], matches, held)]
        seed = {'a': a['name'], 'b': b['name'],
                'pose': np.column_stack((np.eye(3), [.3, 0., 0.])),
                'inliers': matches.astype(np.uint32), 'angles': np.full(len(matches), 5.)}
        model = create_seed({k: Camera.from_dict(v) for k, v in scene['cameras'].items()},
                            {row['name']: row for row in images}, features, seed)
        db, path = directory / 'database.db', directory / 'seed'
        path.mkdir()
        db.write_bytes(b'test database')
        model.write(path)
        return db, path

    def mapping(db, image_path, output, *, options, input_path):
        return {0: pycolmap.Reconstruction(input_path)}

    def solve(rec, stage, initial, **kwargs):
        calls.append(stage)
        return {'termination_type': 'test-solver'}

    monkeypatch.setattr(local_calibration, 'database_and_seed', database)
    monkeypatch.setattr(local_calibration.pycolmap, 'incremental_mapping', mapping)
    monkeypatch.setattr(local_calibration, 'bounded_bundle_adjustment', solve)
    result = local_calibration.calibrate(work)
    return result, calls


def test_global_holdout_precedes_calibration_and_survives_resume(tmp_path, monkeypatch):
    work = global_frontend(tmp_path)
    with pytest.raises(ValueError, match='Missing pre-calibration holdout'):
        local_calibration.calibrate(work)
    protocol = freeze_holdout(work)
    assert freeze_holdout(work) == protocol
    root, scene, data, features = global_reconstruction.load_frontend(work)
    _, split, triangle = load_frozen_holdout(work, data)
    assert split['tracks'] and triangle['tracks']
    out, local_scene, local_data, _ = local_calibration.prepare_calibration(work)
    projected = json.loads((out / 'split.json').read_text())
    held = {(n, i) for t in split['tracks'] for n, i in t['observations']}
    local_names = {r['name'] for r in local_scene['images']}
    assert {(n, i) for t in projected['tracks'] for n, i in t['observations']} == {
        (n, i) for n, i in held if n in local_names}
    assert all(local_data['features'][n] == data['features'][n] for n in local_names)
    result, calls = complete_calibration(work, monkeypatch)
    assert calls == list(local_calibration.STAGES)
    assert result['stages']['principal']['validation']['status'] == 'no_queries'
    model = local_calibration.verify_calibration(work, split)
    assert_holdout_isolated(model, split)
    assert model.num_points3D() == 20 - len(split['tracks'])
    local_calibration.calibrate(work)
    assert calls == list(local_calibration.STAGES)
    global_reconstruction.freeze_geometry(work)
    global_reconstruction.freeze_geometry(work, require_existing=True)


@pytest.mark.parametrize('change', ['feature', 'match', 'split', 'triangle'])
def test_changed_observation_identities_are_rejected(tmp_path, change):
    work = global_frontend(tmp_path)
    freeze_holdout(work)
    root, _, data, _ = global_reconstruction.load_frontend(work)
    if change in ('feature', 'match'):
        entry = next(iter(data['features'].values())) if change == 'feature' else data['pairs'][0]
        path = Path(entry['path'])
        with np.load(path) as cache:
            arrays = {key: cache[key] for key in cache.files}
        key = 'keypoints' if change == 'feature' else 'matches'
        arrays[key] = arrays[key][::-1]
        np.savez(path, **arrays)  # Retain the token to exercise content identity checks.
    else:
        path = root / ('split.json' if change == 'split' else 'triangle_validation_split.json')
        value = json.loads(path.read_text())
        value['tracks'] = []
        write_json(path, value)
    with pytest.raises(ValueError, match='Frozen holdout'):
        local_calibration.calibrate(work)
    assert not (work / 'calibration/calibration_identity.json').exists()


def test_old_geometry_cannot_be_relabelled_as_isolated(tmp_path):
    work = global_frontend(tmp_path)
    path = work / 'calibration/mapping'
    path.mkdir(parents=True)
    (path / 'cameras.bin').write_bytes(b'old model')
    with pytest.raises(ValueError, match='predates the unified holdout'):
        freeze_holdout(work)
    assert not (work / 'sfm/holdout_protocol.json').exists()


def test_changed_calibration_is_rejected_before_global_fitting(tmp_path, monkeypatch):
    work = global_frontend(tmp_path)
    freeze_holdout(work)
    complete_calibration(work, monkeypatch)
    path = work / 'calibration/stages/principal/sparse/0'
    model = pycolmap.Reconstruction(path)
    model.cameras[1].params[0] += .1
    model.write(path)
    with pytest.raises(ValueError, match='Calibration artifact changed'):
        global_reconstruction.freeze_geometry(work)
    with pytest.raises(ValueError, match='Calibration artifact changed'):
        local_calibration.calibrate(work)
    assert not (work / 'sfm/geometry_protocol.json').exists()


def test_short_local_tracks_are_excluded_before_pair_geometry(tmp_path, monkeypatch):
    from geosunpre.sfm import reconstruction as local_reconstruction

    work = global_frontend(tmp_path)
    freeze_holdout(work)
    out, scene, data, features = local_calibration.prepare_calibration(work)
    split = json.loads((out / 'split.json').read_text())
    # A one-image projection cannot be scored, but must still be excluded.
    single = restrict_split(split, scene['images'][:1])
    assert single['tracks'] and all(len(t['observations']) == 1 and not t['queries'] for t in single['tracks'])
    a, b = [r['name'] for r in scene['images']]
    held_indices = {t['observations'][0][1] for t in single['tracks']}

    class PairReached(Exception):
        pass

    def pair(cam_a, cam_b, pa, pb, **kwargs):
        expected = np.array([i for i in range(20) if i not in held_indices])
        np.testing.assert_array_equal(pa, features[a]['keypoints'][expected])
        np.testing.assert_array_equal(pb, features[b]['keypoints'][expected])
        raise PairReached

    monkeypatch.setattr(local_reconstruction, 'fit_pair', pair)
    with pytest.raises(PairReached):
        local_reconstruction.database_and_seed(
            out / 'probe', scene, scene['images'], data['pairs'],
            features, single)


def test_local_query_selection_preserves_global_exclusions(tmp_path):
    work = global_frontend(tmp_path, frames=(70, 72, 74, 110))
    freeze_holdout(work)
    out, scene, data, features = local_calibration.prepare_calibration(work)
    split = json.loads((out / 'split.json').read_text())
    assert split['tracks'] and all(len(t['observations']) == 3 and len(t['queries']) == 1
                                   for t in split['tracks'])
    summary = evaluate_reconstruction(pycolmap.Reconstruction(), scene, features, split, out / 'no_model')
    assert summary['all_candidate_queries']['count'] == len(split['tracks'])
    assert summary['all_candidate_queries']['finite_count'] == 0


def test_build_orders_holdout_before_any_calibration(tmp_path, monkeypatch):
    from geosunpre import scene
    from geosunpre.pipeline import data as global_data
    from geosunpre.pipeline import holdout
    monkeypatch.setattr(runner, 'preflight', lambda *a, **k: {})

    events = []

    def record(name, result=None):
        def stage(*args, **kwargs):
            events.append(name)
            return result
        return stage

    monkeypatch.setattr(scene, 'prepare', record('prepare', {'images': [
        {'camera': 'L', 'frame_index': 0}, {'camera': 'R', 'frame_index': 1}]} ))
    for name in ('prepare_masks', 'extract_features', 'match_features'):
        monkeypatch.setattr(global_data, name, record(name))
    monkeypatch.setattr(holdout, 'freeze_holdout', record(
        'holdout', {'fingerprint': 'test', 'isolation_scope': holdout.ISOLATION_SCOPE}))
    monkeypatch.setattr(local_calibration, 'calibrate', record('calibration'))
    for name in ('build_mapping', 'evaluate_mapping', 'optimize'):
        monkeypatch.setattr(global_reconstruction, name, record(name))
    monkeypatch.setattr(runner, 'finalize', record('equal'))
    monkeypatch.setattr(runner, 'export_result', record('export'))
    runner.run_pipeline(tmp_path, input_root='unused',
                        config=PipelineConfig(pipeline_mode='validation', registration_mode='visual'))
    assert events.index('match_features') < events.index('holdout')
    assert events.index('holdout') < events.index('calibration') < events.index('build_mapping')


def test_isolation_guard_runs_with_python_optimization():
    import os
    import subprocess
    import sys

    script = """
from types import SimpleNamespace as S
from geosunpre.sfm.reconstruction import assert_holdout_isolated
model = S(images={1: S(name='image', points2D=[S(has_point3D=lambda: True)])})
try:
    assert_holdout_isolated(model, {'tracks': [{'observations': [('image', 0)]}]})
except AssertionError:
    raise SystemExit(0)
raise SystemExit(1)
"""
    result = subprocess.run([sys.executable, '-O', '-c', script], env=os.environ.copy(),
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
