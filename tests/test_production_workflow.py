"""Contracts and real numerical handoffs for the sole production workflow."""

import json
import os
from pathlib import Path

import numpy as np
import pycolmap
import pytest
from PIL import Image

from geosunpre.artifacts import digest, file_digest
from geosunpre.camera import Camera
from geosunpre.config import PipelineConfig, load_config, save_config
from geosunpre.pipeline.runner import run_pipeline
from geosunpre.pipeline.observations import freeze_observations
from geosunpre.pipeline.registration import _registration_acceptance, surface_acceptance
from geosunpre.registration import global_alignment, joint
from geosunpre.registration.joint_solver import JointConstraintsUnavailable, assert_identity
from geosunpre.registration.publication import verify_publication
from geosunpre.sfm.reconstruction import create_seed
from helpers import complete_calibration, global_frontend, make_las, raw_input


@pytest.mark.parametrize('family', ['features', 'pairs'])
def test_frozen_observations_reject_changed_cache_bytes(tmp_path, family):
    work = global_frontend(tmp_path)
    original = freeze_observations(work)
    data = json.loads((work / 'sfm/frontend.json').read_text())
    entry = next(iter(data['features'].values())) if family == 'features' else data['pairs'][0]
    path = Path(entry['path'])
    with np.load(path, allow_pickle=False) as cached:
        arrays = {key: cached[key] for key in cached.files}
    if family == 'features':
        arrays['keypoints'][0, 0] += .1
    else:
        arrays['scores'][0] -= .1
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match='Frozen observations changed'):
        freeze_observations(work)
    assert json.loads((work / 'sfm/observation_protocol.json').read_text()) == original


@pytest.mark.parametrize('relative', ['mapping', 'stages/fixed/sparse/0'])
def test_interrupted_calibration_rejects_modified_completed_models(tmp_path, monkeypatch, relative):
    from geosunpre.pipeline.calibration import calibrate

    work = global_frontend(tmp_path)
    freeze_observations(work)
    complete_calibration(work, monkeypatch)
    # Simulate termination before the final calibration completion marker.
    (work / 'calibration/calibration_identity.json').unlink()
    path = work / 'calibration' / relative
    model = pycolmap.Reconstruction(path)
    next(iter(model.points3D.values())).xyz[0] += .1
    model.write(path)
    with pytest.raises(ValueError, match='identity changed'):
        calibrate(work)


@pytest.mark.parametrize('field,value', [
    ('pipeline_mode', 'validation'), ('calibration_mode', 'input'),
    ('registration_mode', 'visual'), ('matcher_attention', 'math'),
    ('registration_pose_frame', 'enh_first'),
])
def test_config_rejects_removed_route_selectors(tmp_path, field, value):
    path = tmp_path / 'scene.toml'
    path.write_text(f'{field} = "{value}"\n')
    with pytest.raises(ValueError, match='Unknown'):
        PipelineConfig.from_file(path)


def test_old_frozen_config_requires_a_fresh_run(tmp_path):
    save_config(tmp_path, PipelineConfig())
    path = tmp_path / 'pipeline_config.json'
    value = json.loads(path.read_text())
    value['version'] = 5
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='fresh'):
        load_config(tmp_path)


def test_prepare_and_missing_weights_can_resume_but_used_weights_cannot_change(tmp_path, monkeypatch):
    import torch
    from geosunpre.pipeline import data

    work = tmp_path / 'work'
    run_pipeline(work, input_root=raw_input(tmp_path), stage='prepare')
    assert not (work / 'frontend_weights.json').exists()
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda: 'fixture GPU')
    monkeypatch.setattr(data, 'extract_features', lambda work: None)
    with pytest.raises(RuntimeError, match='Missing offline LoMa weights'):
        run_pipeline(work, stage='features')
    assert not (work / 'frontend_weights.json').exists()
    checkpoints = Path(os.environ['TORCH_HOME']) / 'hub/checkpoints'
    checkpoints.mkdir(parents=True)
    for name in ('dad.pth', 'dinov2_vitl14_pretrain.pth', 'loma_B.pt'):
        (checkpoints / name).write_bytes(b'fixture weights')
    # Installing cached weights after preparation is a normal supported action.
    run_pipeline(work, stage='features')
    assert (work / 'frontend_weights.json').is_file()
    (checkpoints / 'loma_B.pt').write_bytes(b'changed weights')
    with pytest.raises(ValueError, match='Frozen model weights changed'):
        run_pipeline(work, stage='prepare')


def registration_inputs(tmp_path):
    rng = np.random.default_rng(29)
    uv = rng.uniform(-5, 5, (6000, 2))
    target = np.vstack((np.column_stack((uv, np.zeros(len(uv)))),
                        np.column_stack((np.full(len(uv), -5.), uv)),
                        np.column_stack((uv[:, 0], np.full(len(uv), 5.), uv[:, 1])),
                        np.column_stack((np.full(len(uv), 5.), uv))))
    xyz = target[::15] + [0., 0., 8.]
    cameras = {lens: Camera(lens, 800, 800, (400., 400., 400., 400., 0., 0., 0., 0.), (0., 0.))
               for lens in ('L', 'R')}
    centres = np.array([[0., 0., 0.], [1.5, 0., 0.], [0., 1.5, 0.], [1.5, 1.5, .05]])
    visible = np.ones(len(xyz), dtype=bool)
    for centre in centres:
        pixels = cameras['L'].project(xyz - centre)
        visible &= np.all((pixels >= 0) & (pixels < 800), axis=1)
    xyz = xyz[visible]
    names = [f'{lens}/2435-{352995 + index}-001_{index:05d}-{lens}.JPG'
             for index, lens in enumerate(('L', 'R', 'L', 'R'))]
    records, features = {}, {}
    for index in range(2):
        source = tmp_path / ('source_' + str(index) + '.jpg')
        Image.new('RGB', (800, 800)).save(source)
        records[names[index]] = {'name': names[index], 'camera': ('L', 'R')[index],
                                'image_id': index + 1, 'source': str(source)}
        features[names[index]] = {'keypoints': cameras[('L', 'R')[index]].project(xyz - centres[index])}
    indices = np.arange(len(xyz), dtype=np.uint32)
    seed = {'a': names[0], 'b': names[1],
            'pose': np.column_stack((np.eye(3), -centres[1])),
            'inliers': np.column_stack((indices, indices)), 'angles': np.full(len(xyz), 5.)}
    rec = create_seed(cameras, records, features, seed)
    ids = sorted(rec.points3D)
    reconstructed = np.array([rec.points3D[pid].xyz for pid in ids])
    for index in (2, 3):
        camera_id, image_id = index % 2 + 1, index + 1
        image = pycolmap.Image(name=names[index], camera_id=camera_id, image_id=image_id,
                              keypoints=rec.cameras[camera_id].img_from_cam(reconstructed - centres[index]))
        image.frame_id = image_id
        frame = pycolmap.Frame(frame_id=image_id, rig_id=camera_id,
            rig_from_world=pycolmap.Rigid3d(pycolmap.Rotation3d(), -centres[index]))
        frame.add_data_id(image.data_id)
        rec.add_frame(frame)
        rec.add_image(image)
        rec.register_frame(image_id)
        for point_index, pid in enumerate(ids):
            rec.add_observation(pid, pycolmap.TrackElement(image_id, point_index))
    rec.update_point_3d_errors()
    model = tmp_path / 'visual'
    model.mkdir()
    rec.write(model)
    raw = tmp_path / 'input'
    raw.mkdir()
    las = raw / 'scene.las'
    make_las(las, target)
    ecef, enh = np.array([6_400_000., 200_000., 300_000.]), np.array([1000., 2000., 3000.])
    offset = np.array([100., 200., 292.])
    pose = raw / 'scene_CameraPos_C2E.txt'
    pose.write_text('\n'.join(
        f'{name.split("/")[-1]} 1 2435 {352995 + i}.002 '
        + ' '.join(map(str, ecef + offset + centre)) + ' 0 0 0 1'
        for i, (name, centre) in enumerate(zip(names, centres))))
    times = np.column_stack((np.full(4, 2435), np.arange(4) + 352995.))
    nav, converted = raw / 'scene_pos.txt', raw / 'scene_pos_ENH.txt'
    np.savetxt(nav, np.column_stack((times, centres + ecef)))
    np.savetxt(converted, np.column_stack((times + [0., -.003], centres + enh)))
    inputs = {'input_root': str(raw), 'las': str(las),
              'input_sha256': {str(path): file_digest(path) for path in (las, pose, nav, converted)}}
    inputs['fingerprint'] = digest(inputs)
    return model, inputs


def test_real_global_and_native_joint_registration_handoff(tmp_path, monkeypatch):
    model, inputs = registration_inputs(tmp_path)
    output = tmp_path / 'global'
    report = global_alignment.register_global(model, inputs, output, workers=1,
                                               context_fingerprint='global test')
    verify_publication(output, 'global test')
    assert report['initialization']['camera_pose_minus_filename_seconds']['median'] == pytest.approx(.001)
    assert report['initialization']['navigation_row_time_difference_seconds']['median'] == pytest.approx(.003)
    assert report['split']['fit_points'] > 100 and report['split']['holdout_points'] > 100
    with np.load(output / 'registration_points.npz', allow_pickle=False) as archive:
        reference = {key: archive[key] for key in archive.files}
    assert surface_acceptance(reference, PipelineConfig())['passed']
    assert not (output / 'images').exists()
    monkeypatch.setattr(joint, 'RECIPE', {**joint.RECIPE, 'outer_iterations': 1, 'iterations_per_solve': 3})
    candidate = tmp_path / 'joint'
    result = joint.refine_joint(output, inputs, candidate, workers=1, context_fingerprint='joint test')
    verify_publication(candidate, 'joint test')
    assert result['input_image_gate']['passed']
    assert result['invalid_depth_observations'] == 0
    assert result['projection_invariance']['maximum_pixel_difference'] < 1e-7
    assert_identity(pycolmap.Reconstruction(model), pycolmap.Reconstruction(candidate / 'sparse/0'))
    assert not (candidate / 'images').exists()
    with np.load(candidate / 'las_points.npz', allow_pickle=False) as archive:
        candidate_points = {key: archive[key] for key in archive.files}
    decision = _registration_acceptance(reference, candidate_points, result, PipelineConfig())
    assert decision['target_frame'] == 'input_las_xyz'
    # Unobservable navigation is the same pipeline's guarded fallback condition.
    with monkeypatch.context() as positions_mock:
        positions_mock.setattr(joint, '_positions', lambda *a: {i: np.array([float(i), 0., 0.]) for i in range(1, 5)})
        with pytest.raises(JointConstraintsUnavailable, match='Noncollinear'):
            joint.refine_joint(output, inputs, tmp_path / 'unobservable', workers=1)
    path = output / 'registration_points.npz'
    with np.load(path, allow_pickle=False) as archive:
        points = {key: archive[key] for key in archive.files}
    points['tiles'][0, 0] += 1
    np.savez_compressed(path, **points)
    with pytest.raises(ValueError, match='partition do not replay'):
        joint.refine_joint(output, inputs, tmp_path / 'tampered', workers=1)
