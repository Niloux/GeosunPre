import json

from PIL import Image

from geosunpre.camera import Camera
from geosunpre.pipeline import runner as build
from geosunpre.pipeline import data as global_data
from geosunpre.artifacts import file_digest
from helpers import raw_input
from geosunpre.artifacts import write_json


def test_masks_start_fresh_and_resume_using_their_own_cache(tmp_path, monkeypatch):
    from geosunpre import masks as static_masks

    work = tmp_path / 'fresh'
    source = tmp_path / 'L.jpg'
    Image.new('RGB', (8, 8), (1, 2, 3)).save(source)
    camera = Camera('L', 8, 8, (4., 4., 4., 4., 0., 0., 0., 0.), (0., 0.))
    scene = {'contract_version': 2, 'pose_model': 'independent_exposure',
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


def test_preparation_freezes_config_and_can_resume_without_input(tmp_path):
    import pytest
    from geosunpre.config import PipelineConfig, load_config

    root = raw_input(tmp_path)
    work = tmp_path / 'prepared'
    config = PipelineConfig(expected_images=4, calibration_frames=(10, 11))
    first = build.run_pipeline(work, input_root=root, config=config, stage='prepare')
    assert first['input'] == {'images': 4, 'calibration_images': 4}
    assert json.loads((work / 'pipeline_task.json').read_text())['delivery']['include_ply'] is True
    assert json.loads((work / 'pipeline_task.json').read_text())['delivery']['ply_voxel_size'] == .05
    before = file_digest(work / 'manifest.json')
    second = build.run_pipeline(work, stage='prepare')
    assert first['input'] == second['input']
    assert first['task_fingerprint'] == second['task_fingerprint']
    assert first['invocation_id'] != second['invocation_id']
    assert not second['delivery_complete']
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
    monkeypatch.setattr(build, 'preflight', lambda *a, **k: {})

    work = tmp_path / 'prepared'
    build.run_pipeline(work, input_root=raw_input(tmp_path), stage='prepare')

    def fail(work):
        raise RuntimeError('feature extraction failed')

    monkeypatch.setattr(global_data, 'extract_features', fail)
    with pytest.raises(RuntimeError, match='feature extraction failed'):
        build.run_pipeline(work, stage='features')
    status = json.loads((work / 'pipeline_status.json').read_text())
    assert status['status'] == 'failed'
    assert status['stages'][-1]['name'] == 'features'
    assert 'feature extraction failed' in status['stages'][-1]['error']
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


def test_cli_exposes_only_production_commands(capsys):
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
