from types import SimpleNamespace

import numpy as np
import pycolmap
import pytest
from scipy.spatial.transform import Rotation

from geosunpre.registration import joint
from geosunpre.registration.joint_solver import (RECIPE, ImageObservations, assert_identity, geometry_gate,
    image_gate, interpolate, mixed_objective, native_costs, similarity_matrix, similarity_parameters, solve_fixed)
from geosunpre.pipeline.finalization import SOURCE
from helpers import tiny_visual_work


@pytest.mark.parametrize('kind', ['point_plane', 'position'])
def test_native_joint_residual_jacobians_and_geometry(kind):
    native = native_costs()
    q = Rotation.from_rotvec([.3, -.2, .1]).as_quat()
    translation_scale = np.array([2., -1., 3., np.log(2.3)])
    if kind == 'point_plane':
        cost = native.Sim3PointPlaneCost([1., 2., 3.], [0., .6, .8], 3.2)
        variable = np.array([.7, -1.3, 4.2])
        expected = 3.2 * np.dot([0., .6, .8],
            2.3 * Rotation.from_quat(q).apply(variable) + translation_scale[:3] - [1., 2., 3.])
    else:
        cost = native.Sim3PositionPriorCost([1., 2., 3.], 1.7)
        variable = np.r_[Rotation.from_rotvec([-.1, .2, .3]).as_quat(), [-2., 1., .5]]
        centre = -Rotation.from_quat(variable[:4]).inv().apply(variable[4:])
        expected = 1.7 * (2.3 * Rotation.from_quat(q).apply(centre) + translation_scale[:3] - [1., 2., 3.])
    parameters = [variable, q, translation_scale]
    actual = cost.evaluate_arrays(*parameters)
    np.testing.assert_allclose(actual[0], np.atleast_1d(expected), atol=1e-12)
    for block_index, block in enumerate(parameters):
        columns = []
        for index in range(len(block)):
            left, right = [v.copy() for v in parameters], [v.copy() for v in parameters]
            left[block_index][index] -= 1e-6
            right[block_index][index] += 1e-6
            columns.append((cost.evaluate_arrays(*right)[0] - cost.evaluate_arrays(*left)[0]) / 2e-6)
        np.testing.assert_allclose(actual[block_index + 1], np.array(columns).T, rtol=1e-6, atol=1e-7)


def test_image_gate_detects_local_regression_hidden_by_global_improvement():
    observations = SimpleNamespace(point_ids=np.arange(20000), groups=[
        ('all', 'all', np.arange(20000)), ('image', 'image/1', np.arange(19900)),
        ('image', 'image/2', np.arange(19900, 20000))])
    baseline = np.ones(20000)
    candidate = np.r_[np.full(19900, .5), np.full(100, 1.13)]
    gate = image_gate(observations, baseline, candidate)
    assert not gate['passed']
    assert all(row['group'] != 'all' for row in gate['failures'])
    assert any(row['group'] == 'image/2' for row in gate['failures'])
    candidate[-1] = np.inf
    assert any(row.get('reason') == 'nonfinite_or_nonpositive_depth'
               for row in image_gate(observations, baseline, candidate)['failures'])


def third_exposure(tmp_path):
    work, _, _, _, _ = tiny_visual_work(tmp_path)
    rec = pycolmap.Reconstruction(work / SOURCE)
    ids = sorted(rec.points3D)
    centre = np.array([.15, .25, 0.])
    points = np.array([rec.points3D[pid].xyz for pid in ids])
    image = pycolmap.Image(name='L/third.jpg', camera_id=1, image_id=3,
                           keypoints=rec.cameras[1].img_from_cam(points - centre))
    image.frame_id = 3
    frame = pycolmap.Frame(frame_id=3, rig_id=1,
                           rig_from_world=pycolmap.Rigid3d(pycolmap.Rotation3d(), -centre))
    frame.add_data_id(image.data_id)
    rec.add_frame(frame); rec.add_image(image); rec.register_frame(3)
    for index, pid in enumerate(ids):
        rec.add_observation(pid, pycolmap.TrackElement(3, index))
    return rec


def test_joint_ceres_solve_improves_correspondence_without_changing_intrinsics(tmp_path):
    truth = third_exposure(tmp_path)
    base = pycolmap.Reconstruction(truth)
    ids = sorted(base.points3D)
    positions = {iid: im.projection_center().copy() for iid, im in truth.images.items()}
    centres = np.array([truth.points3D[pid].xyz for pid in ids])
    for point in base.points3D.values():
        point.xyz += [0., 0., .05]
    matrix = np.eye(4)
    matrix[:3, :3] = 1.002 * Rotation.from_rotvec([.002, -.002, .001]).as_matrix()
    matrix[:3, 3] = [.015, -.01, .02]
    matches = {'point_ids': np.array(ids), 'centres': centres,
               'normals': np.broadcast_to([0., 0., 1.], centres.shape), 'weights': np.full(len(ids), 1 / len(ids))}
    recipe = {**RECIPE, 'iterations_per_solve': 60, 'workers': 1}
    observations = ImageObservations(base, matrix)
    before, _ = observations.errors(base)
    initial_cost = mixed_objective(base, matrix, before, positions, matches, recipe)
    candidate = pycolmap.Reconstruction(base)
    final, summary = solve_fixed(candidate, matrix, base, matrix, positions, matches, recipe)
    after, depths = observations.errors(candidate)
    assert summary['solution_usable'] and summary['global_sim3_variable']
    assert summary['fixed_visual_frame_ids']
    assert summary['depth_barriers'] == len(before)
    assert np.all(depths > 0)
    assert mixed_objective(candidate, final, after, positions, matches, recipe) < initial_cost * .1
    assert image_gate(observations, before, after)['passed']
    assert geometry_gate(base, candidate, matrix, final, recipe)['passed']
    assert_identity(base, candidate)
    np.testing.assert_allclose(similarity_matrix(*similarity_parameters(final)), final, atol=1e-12)


def test_backtracking_interpolates_optical_centres_on_the_rotation_manifold(tmp_path):
    before = third_exposure(tmp_path)
    after = pycolmap.Reconstruction(before)
    centre = before.images[3].projection_center() + [.04, .02, 0.]
    rotation = Rotation.from_rotvec([.004, -.003, .002]).as_matrix()
    after.frames[3].rig_from_world = pycolmap.Rigid3d(pycolmap.Rotation3d(rotation), -rotation @ centre)
    halfway, matrix = interpolate(before, after, np.eye(4), np.eye(4), .5)
    np.testing.assert_allclose(halfway.images[3].projection_center(),
        .5 * (before.images[3].projection_center() + after.images[3].projection_center()), atol=1e-12)
    np.testing.assert_allclose(Rotation.from_matrix(halfway.images[3].cam_from_world().matrix()[:, :3]).as_rotvec(),
                              [.002, -.0015, .001], atol=1e-12)
    assert_identity(before, halfway)
    np.testing.assert_array_equal(matrix, np.eye(4))


def test_rejected_joint_step_keeps_the_previous_model_and_saves_the_proposal(tmp_path, monkeypatch):
    base = third_exposure(tmp_path)
    observations = ImageObservations(base, np.eye(4))
    baseline, _ = observations.errors(base)

    def force_bad_pose(proposal, matrix, *args):
        centre = proposal.images[3].projection_center()
        rotation = Rotation.from_rotvec([0., 1., 0.]).as_matrix()
        proposal.frames[3].rig_from_world = pycolmap.Rigid3d(pycolmap.Rotation3d(rotation), -rotation @ centre)
        return matrix, {'solution_usable': True, 'solver': 'forced bad camera proposal'}

    monkeypatch.setattr(joint, 'solve_fixed', force_bad_pose)
    monkeypatch.setattr(joint, 'las_correspondences', lambda *a, **k: {'point_ids': np.array([], dtype=int), 'summary': {}})
    recipe = {**RECIPE, 'pose_weight': 0., 'las_weight': 0., 'outer_iterations': 1}
    rec, matrix, errors, result = joint._run_stage('rejection', base, np.eye(4), {}, None, [], [],
        observations, baseline, recipe, tmp_path / 'rejection')
    assert result['accepted_updates'] == 0
    assert not result['steps'][0]['accepted_update']
    assert all(not trial['passed'] for trial in result['steps'][0]['backtracking'])
    assert (tmp_path / 'rejection/rejected_round_1/frames.bin').is_file()
    for iid in base.images:
        np.testing.assert_allclose(rec.images[iid].cam_from_world().matrix(), base.images[iid].cam_from_world().matrix())
    np.testing.assert_array_equal(errors, baseline)
    np.testing.assert_array_equal(matrix, np.eye(4))
