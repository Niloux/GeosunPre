from collections.abc import Mapping
import json
from pathlib import Path

import numpy as np
import pycolmap
import pytest
from PIL import Image

from geosunpre.artifacts import file_digest
from geosunpre.camera import Camera
from geosunpre.colmap import model_hashes
from geosunpre.sfm.reconstruction import create_seed, database_and_seed, map_from_inputs


def mapping_inputs(tmp_path):
    camera = Camera('L', 320, 400, (100.,100.,160.,200.,0.,0.,0.,0.), (0.,0.))
    xyz = np.random.default_rng(14).uniform([-2,-2,4], [2,2,8], (180,3))
    images, features, entries = [], {}, []
    for i in range(3):
        name = f'L/{i}.jpg'
        source = tmp_path / name
        source.parent.mkdir(exist_ok=True)
        Image.new('RGB', (320,400), (40,80,120)).save(source)
        images.append({'name': name, 'image_id': i + 1, 'frame_index': i,
                       'camera': 'L', 'source': str(source)})
        features[name] = {'keypoints': camera.project(xyz + [i, i * .1, 0.]).astype(np.float32)}
    matches = np.column_stack((np.arange(180), np.arange(180))).astype(np.uint32)
    for i, a in enumerate(images):
        for b in images[i+1:]:
            path = tmp_path / f'{a["image_id"]}-{b["image_id"]}.npz'
            np.savez(path, matches=matches)
            entries.append({'a': a['name'], 'b': b['name'], 'path': str(path)})
    scene = {'cameras': {'L': camera.to_dict()}}
    seed = {'a': images[0]['name'], 'b': images[1]['name'], 'inliers': matches,
            'pose': np.column_stack((np.eye(3), [1.,.1,0.])), 'angles': np.full(180, 5.)}
    rec = create_seed({'L': camera}, {r['name']: r for r in images}, features, seed)
    row = images[2]
    im = pycolmap.Image(name=row['name'], image_id=3, camera_id=1, keypoints=features[row['name']]['keypoints'].astype(float))
    im.frame_id = 3
    frame = pycolmap.Frame(frame_id=3, rig_id=1, rig_from_world=pycolmap.Rigid3d(np.column_stack((np.eye(3), [2.,.2,0.]))))
    frame.add_data_id(im.data_id)
    rec.add_frame(frame); rec.add_image(im); rec.register_frame(3)
    for point_id, point in rec.points3D.items():
        index = point.track.elements[0].point2D_idx
        rec.add_observation(point_id, pycolmap.TrackElement(3, index))
    model = tmp_path / 'local'; model.mkdir()
    rec.write(model)
    return scene, images, entries, features, model


def test_database_and_seed_publish_together_and_resume_without_array_reads(tmp_path, monkeypatch):
    scene, images, entries, features, _ = mapping_inputs(tmp_path)
    root = tmp_path / 'global'
    original = pycolmap.Reconstruction.write

    def interrupted(rec, path):
        if Path(path).name == 'seed':
            (Path(path) / 'cameras.bin').write_bytes(b'partial')
            raise KeyboardInterrupt
        return original(rec, path)

    with monkeypatch.context() as patch:
        patch.setattr(pycolmap.Reconstruction, 'write', interrupted)
        with pytest.raises(KeyboardInterrupt):
            database_and_seed(root, scene, images, entries, features, protocol_fingerprint='frozen')
    assert not (root / 'mapping_inputs').exists()
    assert not list(root.glob('.mapping_inputs-*'))
    db, seed = database_and_seed(root, scene, images, entries, features, protocol_fingerprint='frozen')
    assert db.is_file() and pycolmap.Reconstruction(seed).num_reg_images() == 2
    receipt = json.loads((root / 'mapping_inputs/completion.json').read_text())
    assert all(file_digest(root / 'mapping_inputs' / name) == value for name, value in receipt['files'].items())

    class Unreadable(Mapping):
        def __len__(self): return len(features)
        def __iter__(self): return iter(features)
        def __getitem__(self, key): raise AssertionError('Completed inputs must not load feature arrays')

    assert database_and_seed(root, scene, images, entries, Unreadable(), protocol_fingerprint='frozen') == (db, seed)
    with pytest.raises(ValueError, match='another geometry protocol'):
        database_and_seed(root, scene, images, entries, features, protocol_fingerprint='changed')
    db.write_bytes(b'changed database')
    with pytest.raises(ValueError, match='Committed mapping input changed'):
        database_and_seed(root, scene, images, entries, features, protocol_fingerprint='frozen')


def test_local_seed_retains_all_exposure_poses_points_and_feature_ids(tmp_path):
    scene, images, entries, features, model = mapping_inputs(tmp_path)
    original_hashes = model_hashes(model)
    _, seed = database_and_seed(tmp_path / 'global', scene, images, entries, features,
                                protocol_fingerprint='local-seed', initial_model=model)
    assert model_hashes(seed) == original_hashes == model_hashes(model)
    assert pycolmap.Reconstruction(seed).num_reg_images() == 3
    summary = json.loads((seed.parent / 'seed.json').read_text())
    assert summary['registered'] == 3 and not summary['pose_prior_used'] and summary['pair'] is None


def test_mapper_writes_only_to_its_scratch_copy(tmp_path, monkeypatch):
    scene, images, entries, features, _ = mapping_inputs(tmp_path)
    scene['input_root'] = str(tmp_path)
    root = tmp_path / 'global'
    db, seed = database_and_seed(root, scene, images, entries, features, protocol_fingerprint='frozen')
    before = file_digest(db)

    def mapper(working, image_root, output, *, options, input_path):
        assert working != db and file_digest(working) == before
        working.write_bytes(b'backend write')
        return {0: pycolmap.Reconstruction(input_path)}

    monkeypatch.setattr(pycolmap, 'incremental_mapping', mapper)
    assert map_from_inputs(root, scene, db, seed)[0].num_reg_images() == 2
    assert file_digest(db) == before and not list(root.glob('.mapper-*'))
    assert database_and_seed(root, scene, images, entries, features, protocol_fingerprint='frozen') == (db, seed)


@pytest.mark.parametrize('changed', ['features', 'calibration', 'exposure'])
def test_local_seed_rejects_different_input_identities_before_building_database(tmp_path, changed):
    scene, images, entries, features, model = mapping_inputs(tmp_path)
    if changed == 'features':
        features[images[0]['name']]['keypoints'] = features[images[0]['name']]['keypoints'][::-1].copy()
    elif changed == 'calibration':
        params = list(scene['cameras']['L']['params'])
        params[0] += 1.
        scene['cameras']['L']['params'] = params
    else:
        images[0]['image_id'] += 100
    with pytest.raises(ValueError, match='Initial reconstruction .* changed'):
        database_and_seed(tmp_path / 'global', scene, images, entries, features,
                          protocol_fingerprint='rejected', initial_model=model)
    assert not (tmp_path / 'global/mapping_inputs').exists()
