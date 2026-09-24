import numpy as np

from geosunpre.ground_pose_probe import pose_projection


def test_fisheye_pose_jacobian_matches_finite_difference():
    camera = {'K': np.array([[1100., 0., 1500.], [0., 1100., 2000.], [0., 0., 1.]]),
              'D': np.array([-.015, .003, -.0003, .00002])}
    xyz = np.array([[1., 2., 3.], [2., -1., 4.], [-2., 1., 5.]], dtype=float)
    pose = np.array([.1, -.2, .15, .05, .02, -.03])
    _, analytic = pose_projection(pose, xyz, camera)
    numeric = np.column_stack([
        (pose_projection(pose + np.eye(6)[i] * 1e-6, xyz, camera)[0].ravel()
         - pose_projection(pose - np.eye(6)[i] * 1e-6, xyz, camera)[0].ravel()) / 2e-6
        for i in range(6)])
    np.testing.assert_allclose(analytic, numeric, rtol=1e-7, atol=2e-5)
