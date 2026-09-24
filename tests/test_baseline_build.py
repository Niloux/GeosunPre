import json
from pathlib import Path

import numpy as np
from PIL import Image

from geosunpre.camera import Camera
from geosunpre.local_reconstruction import create_seed
from geosunpre.pipeline import baseline_build as build
from geosunpre.pipeline import global_data
from geosunpre.pipeline import global_reconstruction
from geosunpre.scene import file_digest, write_json


def tiny_visual_work(tmp_path):
    work = tmp_path / 'run'
    root = work / 'global_sfm'
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
    write_json(root / 'split.json', {'tracks': []})
    write_json(root / 'triangle_validation_split.json', {'tracks': []})
    write_json(root / 'geometry_protocol.json',
               {'calibration': {str(cid): cam.params.tolist()
                                for cid, cam in rec.cameras.items()}})
    return work, root, scene, features, records


def test_equal_baseline_build_and_export_need_no_manual_labels(tmp_path, monkeypatch):
    work, root, scene, features, records = tiny_visual_work(tmp_path)
    monkeypatch.setattr(build, 'EXPECTED_IMAGES', 2)
    calls = []

    def solve(rec, stage, initial, **options):
        calls.append((stage, options))
        return {'termination_type': 'test-solver'}

    monkeypatch.setattr(build, 'bounded_bundle_adjustment', solve)
    monkeypatch.setattr(global_reconstruction, 'load_global',
                        lambda work: (root, scene, {}, features))
    monkeypatch.setattr(build, 'evaluate_reconstruction',
                        lambda *args: {'all_candidate_queries': {'count': 0}})
    source_hash = file_digest(work / build.SOURCE / 'cameras.bin')
    first = build.build_equal_control(work)
    second = build.build_equal_control(work)
    assert first == second
    assert len(calls) == 1
    assert calls[0][0] == 'principal'
    assert calls[0][1]['max_iterations'] == build.BA_ITERATIONS
    assert calls[0][1]['protect_depth'] is True
    assert file_digest(work / build.SOURCE / 'cameras.bin') == source_hash
    assert json.loads((work / build.EQUAL / 'protocol.json').read_text())['manual_annotations_used'] is False
    assert not (root / 'static_validation').exists()

    monkeypatch.setattr(build, 'filtered_points',
                        lambda rec, **options: (b'ply\nend_header\n', {'exported_points': 20}))
    config = tmp_path / 'lfs_config.json'
    write_json(config, {'dataset': {'images': 'images', 'use_test_split': False},
                        'optimization': {'undistort': False}})
    exported = build.export_computed_baseline(work, lfs_config=config)
    assert exported['images'] == 2
    output = work / 'baseline_colmap'
    assert (output / 'images/L/a.jpg').resolve() == Path(records['L/a.jpg']['source'])
    assert (output / 'images/R/b.jpg').resolve() == Path(records['R/b.jpg']['source'])
    assert np.asarray(Image.open(output / 'masks/L/a.jpg.png')).shape == (8, 8)
    identity = json.loads((output / 'computed_baseline.json').read_text())
    assert identity['manual_annotations_used'] is False
    assert identity['test_split'] is False
    assert build.verify_computed_baseline(work)['full_content_check'] is True
    assert build.export_computed_baseline(work, lfs_config=config) == exported

    Image.new('L', (8, 8), 0).save(records['L/a.jpg']['semantic_mask'])
    try:
        build.verify_computed_baseline(work)
    except ValueError as error:
        assert 'image or mask content changed' in str(error)
    else:
        raise AssertionError('Changed semantic mask passed content verification')
    try:
        build.export_computed_baseline(work, lfs_config=config)
    except ValueError as error:
        assert 'Existing combined mask changed' in str(error)
    else:
        raise AssertionError('Changed mask was silently accepted')


def test_global_data_can_start_without_transfer_experiment(tmp_path, monkeypatch):
    from geosunpre import static_masks

    work = tmp_path / 'fresh'
    source = tmp_path / 'L.jpg'
    Image.new('RGB', (8, 8), (1, 2, 3)).save(source)
    camera = Camera('L', 8, 8, (4., 4., 4., 4., 0., 0., 0., 0.), (0., 0.))
    scene = {'contract_version': 1, 'pose_model': 'independent_exposure',
             'cameras': {'L': camera.to_dict()}, 'fov_degrees': 160,
             'images': [{'name': 'L/a.jpg', 'source': str(source), 'camera': 'L',
                         'frame_index': 70, 'semantic_mask': None}],
             'fingerprint': 'fresh-scene'}
    write_json(work / 'manifest.json', scene)

    def generate_masks(base, output, frames, **kwargs):
        assert frames == [70]
        mask = output / 'valid/L/a.jpg.png'
        mask.parent.mkdir(parents=True)
        Image.new('L', (8, 8), 255).save(mask)
        write_json(output / 'stats/L/a.jpg.json',
                   {'source_sha256': file_digest(source), 'mask_sha256': file_digest(mask)})
        return [{**base['images'][0], 'semantic_mask': str(mask)}], {'revision': 'test'}

    monkeypatch.setattr(static_masks, 'generate_masks', generate_masks)
    monkeypatch.setattr(static_masks, 'mask_review', lambda *args, **kwargs: None)
    monkeypatch.setattr(static_masks, 'mask_contact_sheet', lambda *args, **kwargs: None)
    output = global_data.prepare_global(work)
    assert json.loads((output / 'reused_masks.json').read_text())['count'] == 0
    assert json.loads((output / 'manifest.json').read_text())['images'][0]['semantic_mask'].endswith('.png')
