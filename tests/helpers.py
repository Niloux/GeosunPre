"""Small synthetic models and inputs shared by production contract tests."""

import json
import struct

import numpy as np
import pycolmap
from PIL import Image

from geosunpre.artifacts import digest, file_digest, write_json
from geosunpre.camera import Camera
from geosunpre.config import PipelineConfig, save_config
from geosunpre.features import FRONTEND_CONFIG as CONFIG
from geosunpre.pipeline import calibration as local_calibration
from geosunpre.pipeline import finalization as build
from geosunpre.sfm.reconstruction import create_seed


def tiny_visual_work(tmp_path):
    work = tmp_path / 'run'
    root = work / 'sfm'
    cameras = {lens: Camera(lens, 8, 8, (4., 4., 4., 4., 0., 0., 0., 0.), (0., 0.))
               for lens in ('L', 'R')}
    rng = np.random.default_rng(4)
    xyz = rng.uniform([-.8, -.8, 3.], [.8, .8, 5.], (20, 3))
    move = np.column_stack((np.eye(3), np.array([.3, 0., 0.])))
    names = ('L/a.jpg', 'R/b.jpg')
    records = {}
    features = {}
    for image_id, (lens, name) in enumerate(zip(('L', 'R'), names), 1):
        image_path = tmp_path / name
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new('RGB', (8, 8), (40, 80, 120)).save(image_path)
        mask_path = tmp_path / (name + '.png')
        Image.new('L', (8, 8), 255).save(mask_path)
        records[name] = {'name': name, 'camera': lens, 'image_id': image_id,
                         'source': str(image_path), 'semantic_mask': str(mask_path)}
        shift = np.array([.3, 0., 0.]) if lens == 'R' else np.zeros(3)
        features[name] = {'keypoints': cameras[lens].project(xyz + shift)}
    seed = {'a': names[0], 'b': names[1], 'pose': move,
            'inliers': np.column_stack((np.arange(20), np.arange(20))).astype(np.uint32),
            'angles': np.full(20, 5.)}
    rec = create_seed(cameras, records, features, seed)
    source = work / build.SOURCE
    source.mkdir(parents=True)
    rec.write(source)
    scene = {'contract_version': 1, 'pose_model': 'independent_exposure',
             'fov_degrees': 160, 'cameras': {k: v.to_dict() for k, v in cameras.items()},
             'images': list(records.values()), 'fingerprint': 'tiny-visual-scene'}
    write_json(root / 'manifest.json', scene)
    write_json(root / 'geometry_protocol.json',
               {'calibration': {str(cid): cam.params.tolist()
                                for cid, cam in rec.cameras.items()}})
    return work, root, scene, features, records


def raw_input(tmp_path):
    root = tmp_path / 'input'
    root.mkdir()
    devices, poses = [], []
    for lens in ('L', 'R'):
        devices.append({
            'deviceId': 'camera_' + lens,
            'dataBasicParams': {'pixelSize': .01, 'focalLength': .04,
                                'photoWidth': 8, 'photoHeight': 8},
            'dataSensorParams': {'k1': 0, 'k2': 0, 'k3': 0, 'p3': 0,
                                 'principalX': 0, 'principalY': 0},
        })
        folder = root / 'Image' / lens / '0000'
        folder.mkdir(parents=True)
        for frame in (10, 11):
            name = f'2435-{352990 + frame}-001_{frame:05d}-{lens}.JPG'
            Image.new('RGB', (8, 8), (frame, 2, 3)).save(folder / name)
            poses.append(f'{name} 1 2435 {352990 + frame}.002 100 200 300 0 0 0 1')
    write_json(root / 'scene.pro', {'cameraDevices': {'devices': devices}})
    (root / 'scene_CameraPos_C2E.txt').write_text('\n'.join(poses))
    return root


def make_las(path, points, point_format=3):
    header = bytearray(227)
    header[:4] = b'LASF'
    header[24:26] = bytes([1, 2])
    struct.pack_into('<H', header, 94, 227)
    struct.pack_into('<I', header, 96, 227)
    header[104] = point_format
    length = (20, 28, 26, 34)[point_format]
    struct.pack_into('<H', header, 105, length)
    struct.pack_into('<I', header, 107, len(points))
    struct.pack_into('<3d', header, 131, .001, .001, .001)
    struct.pack_into('<3d', header, 155, 100., 200., 300.)
    values = np.rint(np.asarray(points) / .001).astype(np.int32)
    actual = values * .001 + [100., 200., 300.]
    struct.pack_into('<6d', header, 179, *np.column_stack((actual.max(0), actual.min(0))).ravel())
    fields = {'names': ['xyz', 'class'], 'formats': [('<i4', 3), 'u1'],
              'offsets': [0, 15], 'itemsize': length}
    if point_format in (1, 3):
        fields['names'].append('time')
        fields['formats'].append('<f8')
        fields['offsets'].append(20)
    dtype = np.dtype(fields)
    records = np.zeros(len(points), dtype=dtype)
    records['xyz'] = values
    records['class'] = 1
    if point_format in (1, 3):
        records['time'] = np.arange(len(points)) * .005 + 352995.
    with path.open('wb') as stream:
        stream.write(header)
        records.tofile(stream)
    return actual


def global_frontend(tmp_path, frames=(70, 72)):
    """Small real image/cache fixture; no LoMa, GPU or reconstruction is run."""
    work = tmp_path / 'run'
    root = work / 'sfm'
    root.mkdir(parents=True)
    save_config(work, PipelineConfig(calibration_frames=(70, 94)))
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
    """Use real pipeline contracts and models with only mapping and BA stubbed."""
    calls = []

    def database(directory, scene, images, entries, features):
        a, b = images[:2]
        matches = np.column_stack((np.arange(20), np.arange(20)))
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
