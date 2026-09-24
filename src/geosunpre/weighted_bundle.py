"""Add finite ground-observation weights to the native Ceres fisheye BA.

The existing COLMAP reprojection factor is kept. An additional factor with
weight (w - 1) has the same Huber transition as the original factor, so the
sum is exactly w times the original robust reprojection objective. The
reconstruction still has one observation and one 3D point per track entry.
"""
import cv2
import numpy as np
import pyceres


HUBER_PX = 2.0


def quaternion_matrix_and_jacobian(q, point):
    """Eigen xyzw quaternion rotation and ambient 3x4 derivative at unit q."""
    v = np.asarray(q[:3], dtype=float)
    w = float(q[3])
    point = np.asarray(point, dtype=float)
    skew = np.array([[0., -v[2], v[1]],
                     [v[2], 0., -v[0]],
                     [-v[1], v[0], 0.]], dtype=float)
    rotation = (w*w - v @ v) * np.eye(3) + 2 * np.outer(v, v) + 2 * w * skew
    jacobian = np.empty((3, 4), dtype=float)
    cross_v_point = np.cross(v, point)
    for j in range(3):
        axis = np.eye(3)[j]
        jacobian[:, j] = (-2*v[j]*point + 2*axis*(v @ point)
                          + 2*v*point[j] + 2*w*np.cross(axis, point))
    jacobian[:, 3] = 2*w*point + 2*cross_v_point
    return rotation, jacobian


def fisheye_factor(point, pose, intrinsics, observed, extra_weight):
    """Return residual plus Jacobians for XYZ, xyzw+translation, and K/D."""
    rotation, rotation_jac = quaternion_matrix_and_jacobian(pose[:4], point)
    camera_point = rotation @ point + pose[4:]
    fx, fy, cx, cy, *distortion = intrinsics
    matrix = np.array([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]], dtype=float)
    projected, jac = cv2.fisheye.projectPoints(
        np.asarray(camera_point, dtype=np.float64).reshape(1, 1, 3),
        np.zeros(3), np.zeros(3), matrix,
        np.asarray(distortion, dtype=np.float64))
    scale = np.sqrt(extra_weight)
    pixel_jac = jac[:, 11:14]
    pose_jac = np.empty((2, 7), dtype=float)
    pose_jac[:, :4] = pixel_jac @ rotation_jac
    pose_jac[:, 4:] = pixel_jac
    return (scale * (projected.reshape(2) - observed),
            scale * pixel_jac @ rotation,
            scale * pose_jac,
            scale * jac[:, :8])


class ExtraFisheyeObservation(pyceres.CostFunction):
    def __init__(self, observed, extra_weight):
        super().__init__()
        if extra_weight <= 0:
            raise ValueError('Extra observation weight must be positive')
        self.set_num_residuals(2)
        self.set_parameter_block_sizes([3, 7, 8])
        self.observed = np.asarray(observed, dtype=float).copy()
        self.extra_weight = float(extra_weight)

    def Evaluate(self, parameters, residuals, jacobians):
        values = fisheye_factor(*parameters, self.observed, self.extra_weight)
        residuals[:] = values[0]
        if jacobians is not None:
            for output, jac in zip(jacobians, values[1:]):
                if output is not None:
                    output[:] = jac.reshape(output.shape)
        return True


def add_weighted_ground_factors(problem, rec, floor_ids, near_cutoff, weights,
                                image_ids=None, near_membership=None):
    """Add exact incremental robust weight to selected training observations."""
    costs = []
    losses = {key: pyceres.HuberLoss(HUBER_PX * np.sqrt(value - 1.))
              for key, value in weights.items() if value > 1.}
    counts = {'near': 0, 'far': 0, 'skipped_constant': 0}
    for image_id in sorted(rec.reg_image_ids() if image_ids is None else image_ids):
        image = rec.images[image_id]
        frame = rec.frames[image.frame_id]
        if len(frame.image_ids) != 1:
            raise ValueError('Weighted BA requires one independent exposure per frame')
        pose = frame.rig_from_world.params
        camera = rec.cameras[image.camera_id].params
        center = image.projection_center()
        for observation_index, observation in enumerate(image.points2D):
            if not observation.has_point3D() or int(observation.point3D_id) not in floor_ids:
                continue
            point = rec.points3D[observation.point3D_id]
            if near_membership is None:
                is_near = np.linalg.norm(point.xyz - center) <= near_cutoff
            else:
                is_near = (image_id, observation_index) in near_membership
            key = 'near' if is_near else 'far'
            weight = weights[key]
            counts[key] += 1
            if weight <= 1:
                continue
            blocks = [point.xyz, pose, camera]
            if not all(problem.has_parameter_block(block) for block in blocks):
                raise RuntimeError('Selected training observation is absent from native BA')
            if all(problem.is_parameter_block_constant(block) for block in blocks):
                counts['skipped_constant'] += 1
                continue
            cost = ExtraFisheyeObservation(observation.xy, weight - 1.)
            costs.append(cost)
            problem.add_residual_block(cost, losses[key], blocks)
    return costs, losses, counts
