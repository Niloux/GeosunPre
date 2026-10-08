"""Native positive-depth barriers for fisheye bundle adjustment."""

import numpy as np


def depth_cost_class():
    from geosunpre_native import PositiveDepthCost
    return PositiveDepthCost


def add_positive_depth_guards(problem,rec, *, cost_class=None):
    cost_class = cost_class or depth_cost_class()
    costs=[]
    for image in rec.images.values():
        if not image.has_pose:continue
        frame=rec.frames[image.frame_id]
        if len(frame.image_ids)!=1:
            raise ValueError('This depth guard expects one independent exposure per frame')
        pose=frame.rig_from_world
        matrix=pose.matrix()
        was_variable=problem.has_parameter_block(pose.params)
        for obs in image.points2D:
            if not obs.has_point3D():continue
            point=rec.points3D[obs.point3D_id]
            depth=float((matrix[:,:3]@point.xyz+matrix[:,3])[2])
            if not np.isfinite(depth) or depth<=1e-9:raise ValueError('Initial depth must be positive')
            cost=cost_class(depth)
            costs.append(cost)
            problem.add_residual_block(cost,None,[point.xyz,pose.params])
        if not was_variable:
            problem.set_parameter_block_constant(pose.params)
    return costs
