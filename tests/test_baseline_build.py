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
    from geosunpre.pipeline.holdout import freeze_global_holdout
    from test_unified_holdout import complete_calibration, global_frontend
    import pycolmap

    work = global_frontend(tmp_path, frames=(70, 72))
    freeze_global_holdout(work)
    complete_calibration(work, monkeypatch)
    global_reconstruction.freeze_global_geometry(work)
    root, scene, _, _ = global_reconstruction.load_global(work)
    records = {row['name']: row for row in scene['images']}
    model = pycolmap.Reconstruction(work / 'local_sfm/normal_070_094/stages/principal/sparse/0')
    (work / build.SOURCE).mkdir(parents=True)
    model.write(work / build.SOURCE)
    monkeypatch.setattr(build, 'EXPECTED_IMAGES', 2)
    calls = []

    def solve(rec, stage, initial, **options):
        calls.append((stage, options))
        return {'termination_type': 'test-solver'}

    monkeypatch.setattr(build, 'bounded_bundle_adjustment', solve)
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

    validation = json.loads((work / build.EQUAL / 'validation_summary.json').read_text())
    assert validation['raw']['status'] == 'no_queries'
    assert validation['triangle']['all_candidate_queries']['count'] == 0
    with np.load(work / build.EQUAL / 'validation/validation_observations.npz') as observations:
        assert observations['support_ok'].dtype == np.bool_
        assert observations['predicted'].shape == (0, 2)

    exported = build.run_baseline(work, stage='export')['export']
    assert exported['images'] == 2
    assert exported['points'] == 20
    output = work / 'colmap'
    for name, record in records.items():
        assert (output / 'images' / name).resolve() == Path(record['source'])
    assert not (output / 'lfs_config.json').exists()
    assert not (output / 'masks').exists()
    assert not (output / 'sparse/0/points3D.ply').exists()
    identity = json.loads((output / 'colmap_export.json').read_text())
    assert identity['provenance']['manual_annotations_used'] is False
    assert identity['provenance']['holdout_isolation'] == 'before_local_and_global_geometry'
    assert 'test_split' not in identity
    assert 'all_selected_images_train' not in identity
    assert build.verify_computed_baseline(output=output)['full_content_check'] is True
    assert build.export_computed_baseline(work) == exported


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
        mask.parent.mkdir(parents=True, exist_ok=True)
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
    # The new calibration manifest references global masks, with no local mask
    # directory to copy from when global-data is resumed.
    write_json(work / 'local_sfm/manifest.json', json.loads((output / 'manifest.json').read_text()))
    assert global_data.prepare_global(work) == output
