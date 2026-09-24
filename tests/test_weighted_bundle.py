import numpy as np
import pyceres

from geosunpre.camera import Camera
from geosunpre.weighted_bundle import (ExtraFisheyeObservation,
                                       fisheye_factor, quaternion_matrix_and_jacobian)


def test_weighted_fisheye_factor_jacobians_and_ceres_callback():
    point = np.array([.28, -.44, 2.7], dtype=float)
    quaternion = np.array([.1, -.2, .04, .97], dtype=float)
    quaternion /= np.linalg.norm(quaternion)
    pose = np.r_[quaternion, [.04, -.03, .1]]
    intrinsics = np.array([965., 966., 1515., 2000., -.028, -.005, .002, -.0007])
    observed = np.array([1540., 1850.])
    blocks = [point, pose, intrinsics]
    residual, *analytic = fisheye_factor(*blocks, observed, .5)
    rotation, _ = quaternion_matrix_and_jacobian(quaternion, point)
    colmap_camera = Camera('L', 3000, 4000, tuple(intrinsics), (0, 0)).pycolmap()
    np.testing.assert_allclose(residual / np.sqrt(.5) + observed,
                               colmap_camera.img_from_cam(rotation @ point + pose[4:]),
                               atol=1e-8)
    for block_index, block in enumerate(blocks):
        numeric = []
        for j in range(len(block)):
            delta = np.zeros_like(block)
            delta[j] = 1e-6
            plus, minus = [x.copy() for x in blocks], [x.copy() for x in blocks]
            plus[block_index] += delta
            minus[block_index] -= delta
            a = fisheye_factor(*plus, observed, .5)[0]
            b = fisheye_factor(*minus, observed, .5)[0]
            numeric.append((a - b) / 2e-6)
        np.testing.assert_allclose(analytic[block_index],
                                   np.column_stack(numeric), rtol=2e-6, atol=5e-5)

    cost = ExtraFisheyeObservation(observed, .5)
    returned = np.empty(2)
    jacobians = [np.empty(6), np.empty(14), np.empty(16)]
    assert cost.Evaluate(blocks, returned, jacobians)
    np.testing.assert_allclose(returned, residual)
    for returned_jac, target in zip(jacobians, analytic):
        np.testing.assert_allclose(returned_jac, target.ravel())


def test_incremental_huber_factor_matches_target_observation_weight():
    base_loss = pyceres.HuberLoss(2.)
    for weight in (1.25, 1.5, 2.):
        increment = weight - 1.
        extra_loss = pyceres.HuberLoss(2. * np.sqrt(increment))
        for norm in (.1, 1., 3., 8.):
            native = base_loss.evaluate(norm * norm)[0]
            extra = extra_loss.evaluate(increment * norm * norm)[0]
            np.testing.assert_allclose(extra, increment * native, rtol=1e-12, atol=1e-12)
