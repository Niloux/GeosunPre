import json
from pathlib import Path

import numpy as np
from PIL import Image

from geosunpre.camera import Camera
from geosunpre.colmap import verify_colmap_export
from geosunpre.sfm.reconstruction import create_seed
from geosunpre.pipeline import runner as build
from geosunpre.pipeline import data as global_data
from geosunpre.pipeline import reconstruction as global_reconstruction
from geosunpre.artifacts import file_digest
from geosunpre.artifacts import write_json


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
    write_json(root / 'split.json', {'tracks': []})
    write_json(root / 'triangle_validation_split.json', {'tracks': []})
    write_json(root / 'geometry_protocol.json',
               {'calibration': {str(cid): cam.params.tolist()
                                for cid, cam in rec.cameras.items()}})
    return work, root, scene, features, records


def test_equal_runner_and_export_need_no_manual_labels(tmp_path, monkeypatch):
    from geosunpre.pipeline.holdout import freeze_holdout
    from test_unified_holdout import complete_calibration, global_frontend
    import pycolmap

    work = global_frontend(tmp_path, frames=(70, 72))
    freeze_holdout(work)
    complete_calibration(work, monkeypatch)
    global_reconstruction.freeze_geometry(work)
    root, scene, _, _ = global_reconstruction.load_frontend(work)
    records = {row['name']: row for row in scene['images']}
    model = pycolmap.Reconstruction(work / 'calibration/stages/principal/sparse/0')
    (work / build.SOURCE).mkdir(parents=True)
    model.write(work / build.SOURCE)
    calls = []

    def solve(rec, stage, initial, **options):
        calls.append((stage, options))
        return {'termination_type': 'test-solver'}

    monkeypatch.setattr(build, 'bounded_bundle_adjustment', solve)
    source_hash = file_digest(work / build.SOURCE / 'cameras.bin')
    first = build.finalize(work)
    second = build.finalize(work)
    assert first == second
    assert len(calls) == 1
    assert calls[0][0] == 'principal'
    assert calls[0][1]['max_iterations'] == build.BA_ITERATIONS
    assert calls[0][1]['protect_depth'] is True
    assert file_digest(work / build.SOURCE / 'cameras.bin') == source_hash
    assert json.loads((work / build.FINAL / 'protocol.json').read_text())['manual_annotations_used'] is False
    assert not (root / 'static_validation').exists()

    validation = json.loads((work / build.FINAL / 'validation_summary.json').read_text())
    assert validation['raw']['status'] == 'no_queries'
    assert validation['triangle']['all_candidate_queries']['count'] == 0
    with np.load(work / build.FINAL / 'validation/validation_observations.npz') as observations:
        assert observations['support_ok'].dtype == np.bool_
        assert observations['predicted'].shape == (0, 2)

    exported = build.run_pipeline(work, stage='export')['export']
    assert exported['images'] == 2
    assert exported['points'] == 20
    output = work / 'colmap'
    for name, record in records.items():
        assert file_digest(output / 'images' / name) == file_digest(record['source'])
        assert not (output / 'images' / name).is_symlink()
    assert not (output / 'lfs_config.json').exists()
    assert not (output / 'masks').exists()
    assert not (output / 'sparse/0/points3D.ply').exists()
    identity = json.loads((output / 'colmap_export.json').read_text())
    assert identity['provenance']['manual_annotations_used'] is False
    assert identity['provenance']['holdout_isolation'] == 'before_local_and_global_geometry'
    assert 'test_split' not in identity
    assert 'all_selected_images_train' not in identity
    assert verify_colmap_export(output)['full_content_check'] is True
    assert build.export_result(work) == exported


def test_masks_start_fresh_and_resume_using_their_own_cache(tmp_path, monkeypatch):
    from geosunpre import masks as static_masks

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
    output = global_data.prepare_masks(work)
    assert json.loads((output / 'manifest.json').read_text())['images'][0]['semantic_mask'].endswith('.png')
    # A later calibration manifest must not change mask generation on resume.
    write_json(work / 'calibration/manifest.json', json.loads((output / 'manifest.json').read_text()))
    assert global_data.prepare_masks(work) == output


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


def test_preparation_freezes_config_and_can_resume_without_input(tmp_path):
    import pytest
    from geosunpre.config import PipelineConfig, load_config

    root = raw_input(tmp_path)
    work = tmp_path / 'prepared'
    config = PipelineConfig(expected_images=4, calibration_frames=(10, 11))
    first = build.run_pipeline(work, input_root=root, config=config, stage='prepare')
    assert first['input'] == {'images': 4, 'calibration_images': 4}
    before = file_digest(work / 'manifest.json')
    second = build.run_pipeline(work, stage='prepare')
    assert first == second
    assert file_digest(work / 'manifest.json') == before
    assert load_config(work) == config
    status = json.loads((work / 'pipeline_status.json').read_text())
    assert status['status'] == 'completed'
    assert status['stages'][0]['elapsed_seconds'] >= 0
    with pytest.raises(ValueError, match='Frozen inputs changed'):
        build.run_pipeline(work, config=PipelineConfig(calibration_frames=(10, 12)), stage='prepare')
    assert file_digest(work / 'manifest.json') == before
    with pytest.raises(ValueError, match='Input directory changed'):
        build.run_pipeline(work, input_root=tmp_path / 'another', stage='prepare')


def test_pipeline_records_failure_and_releases_work_lock(tmp_path, monkeypatch):
    import pytest

    work = tmp_path / 'prepared'
    build.run_pipeline(work, input_root=raw_input(tmp_path), stage='prepare')

    def fail(work):
        raise RuntimeError('feature extraction failed')

    monkeypatch.setattr(global_data, 'extract_features', fail)
    with pytest.raises(RuntimeError, match='feature extraction failed'):
        build.run_pipeline(work, stage='features')
    status = json.loads((work / 'pipeline_status.json').read_text())
    assert status['status'] == 'failed'
    assert status['stages'][0]['name'] == 'features'
    assert 'feature extraction failed' in status['stages'][0]['error']
    # Starting again proves the exception did not retain the lock.
    monkeypatch.setattr(global_data, 'extract_features', lambda work: None)
    build.run_pipeline(work, stage='features')
    assert json.loads((work / 'pipeline_status.json').read_text())['status'] == 'completed'


def test_work_lock_excludes_concurrent_runs_and_protects_unmanaged_directories(tmp_path):
    import pytest
    from geosunpre.pipeline.runtime import work_lock

    work = tmp_path / 'locked'
    with work_lock(work):
        with pytest.raises(RuntimeError, match='Another pipeline process'):
            build.run_pipeline(work, input_root='unused')
    foreign = tmp_path / 'foreign'
    write_json(foreign / 'other.json', {'preserve': True})
    with pytest.raises(ValueError, match='Unmanaged work directory'):
        build.run_pipeline(foreign, input_root='unused')
    assert json.loads((foreign / 'other.json').read_text()) == {'preserve': True}
    assert not (foreign / 'pipeline_config.json').exists()


def test_scene_config_selects_frames_and_rejects_incompatible_inputs(tmp_path):
    import pytest
    from geosunpre.config import PipelineConfig

    scene = {'images': [{'camera': lens, 'frame_index': frame}
                        for frame, lens in [(4, 'L'), (5, 'R'), (8, 'L'), (9, 'R'), (10, 'R')]]}
    assert len(PipelineConfig(calibration_frame_count=2).calibration_images(scene)) == 2
    selected = PipelineConfig(calibration_frames=(8, 10)).calibration_images(scene)
    assert [row['frame_index'] for row in selected] == [8, 9, 10]
    with pytest.raises(ValueError, match='Expected 6'):
        PipelineConfig(expected_images=6).check_scene(scene)
    with pytest.raises(ValueError, match='both L/R'):
        PipelineConfig(calibration_frames=(9, 10)).calibration_images(scene)
    path = tmp_path / 'config.toml'
    path.write_text('calibration_frame_count = 2\nexpected_images = 5\n')
    assert PipelineConfig.from_file(path).expected_images == 5
    path.write_text('unknown = 1\n')
    with pytest.raises(ValueError, match='Unknown pipeline settings'):
        PipelineConfig.from_file(path)


def test_cli_exposes_only_pipeline_commands(capsys):
    import pytest
    from geosunpre.cli import main

    with pytest.raises(SystemExit) as error:
        main(['--help'])
    assert error.value.code == 0
    assert '{run,export,check}' in capsys.readouterr().out
    with pytest.raises(SystemExit) as error:
        main(['build-baseline'])
    assert error.value.code == 2
    assert 'invalid choice' in capsys.readouterr().err
