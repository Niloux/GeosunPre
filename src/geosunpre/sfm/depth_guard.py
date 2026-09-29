"""Positive-depth barriers for front-hemisphere fisheye reconstruction."""
import numpy as np
import pyceres
import os
import warnings


def depth_cost_class():
    """Require native BA by default; slower paths must be selected explicitly."""
    backend = os.environ.get('GEOSUNPRE_DEPTH_BACKEND', 'native')
    if backend not in ('auto', 'native', 'python'):
        raise ValueError('GEOSUNPRE_DEPTH_BACKEND must be auto, native, or python')
    if backend != 'python':
        try:
            from geosunpre_native import PositiveDepthCost as NativeCost
            return NativeCost
        except ImportError:
            if backend == 'native':
                raise
            warnings.warn('Native depth guard unavailable; using slower single-thread Python BA. '
                          'Install ./native for accelerated BA.', RuntimeWarning, stacklevel=2)
    return PositiveDepthCost


class PositiveDepthCost(pyceres.CostFunction):
    """Zero in the valid region; protects against zero-cost invalid projections."""
    def __init__(self,initial_depth):
        super().__init__()
        self.set_num_residuals(1)
        self.set_parameter_block_sizes([3,7])
        self.minimum=.05*initial_depth
        self.scale=1000/initial_depth

    def Evaluate(self,parameters,residuals,jacobians):
        X,Y,Z=parameters[0]
        x,y,z,w,tx,ty,tz=parameters[1]
        row=np.array([2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)])
        depth=row[0]*X+row[1]*Y+row[2]*Z+tz
        active=depth<self.minimum
        residuals[0]=(self.minimum-depth)*self.scale if active else 0.
        if jacobians is not None:
            if jacobians[0] is not None:
                jacobians[0][:]=-self.scale*row if active else 0.
            if jacobians[1] is not None:
                if active:
                    jacobians[1][:]=-self.scale*np.array([
                        2*z*X+2*w*Y-4*x*Z,-2*w*X+2*z*Y-4*y*Z,
                        2*x*X+2*y*Y,-2*y*X+2*x*Y,0,0,1])
                else:
                    jacobians[1][:]=0.
        return True


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
