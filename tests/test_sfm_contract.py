import numpy as np
from PIL import Image
import pytest

from geosunpre.camera import Camera
from geosunpre.sfm.statistics import error_summary
from geosunpre.sfm.reconstruction import create_seed
from geosunpre.sfm.reconstruction import bounded_bundle_adjustment
from geosunpre.sfm.reconstruction import mapper_options
from geosunpre.sfm.reconstruction import cheirality_health


def test_evaluation_keeps_failed_predictions_in_denominator():
    s=error_summary([1.,2.,np.inf,np.nan])
    assert s['count']==4 and s['finite_count']==2
    assert s['fraction_le_2px']==.5 and s['p90'] is None


def test_fisheye_seed_and_bounded_ba_preserve_locked_intrinsics(tmp_path):
    cam=Camera('L',3000,4000,(965,965,1500,2000,-.03,-.004,.001,-.0003),(0,0))
    rng=np.random.default_rng(2)
    xyz=rng.uniform([-3,-2,5],[3,2,10],(80,3))
    p0=np.column_stack((np.eye(3),np.zeros(3)))
    p1=np.column_stack((np.eye(3),np.array([1.,0,0])))
    features={'a':{'keypoints':cam.project(xyz)},'b':{'keypoints':cam.project(xyz+p1[:,3])}}
    source=tmp_path/'image.jpg';Image.new('RGB',(3000,4000)).save(source)
    records={name:{'name':name,'camera':'L','image_id':i+1,'source':str(source)} for i,name in enumerate(['a','b'])}
    seed={'a':'a','b':'b','pose':p1,'inliers':np.column_stack((np.arange(80),np.arange(80))).astype(np.uint32),
          'angles':np.full(80,5.)}
    rec=create_seed({'L':cam},records,features,seed)
    assert rec.num_reg_frames()==rec.num_reg_images()==2
    initial={cid:c.params.copy() for cid,c in rec.cameras.items()}
    report=bounded_bundle_adjustment(rec,'focal_low_radial',initial)
    np.testing.assert_allclose(rec.cameras[1].params[[2,3,6,7]],initial[1][[2,3,6,7]],atol=1e-12)
    assert report['cameras']['1']['monotonic_in_160deg']
    guarded=bounded_bundle_adjustment(rec,'principal',initial)
    assert guarded['positive_depth_barriers']==160
    assert guarded['cheirality']['invalid_depth']==0
    opts=mapper_options()
    assert not opts.use_prior_position and not opts.ba_refine_sensor_from_rig
    assert not opts.mapper.abs_pose_refine_focal_length
    for point in rec.points3D.values():
        point.xyz[2]*=-1
    assert cheirality_health(rec)=={'observations':160,'positive_depth':0,'invalid_depth':160}


def test_native_depth_barrier_residual_and_finite_differences():
    from geosunpre_native import PositiveDepthCost

    rng = np.random.default_rng(193)
    for depth in (.01, 1., 100.):
        cost = PositiveDepthCost(depth)
        for active in (False, True):
            for _ in range(8):
                point = rng.normal(size=3)
                pose = rng.normal(size=7)
                pose[:4] /= np.linalg.norm(pose[:4])
                pose[-1] = -20. if active else 20.
                x, y, z, w = pose[:4]
                camera_z = np.dot([2 * (x * z - w * y), 2 * (y * z + w * x),
                                  1 - 2 * (x * x + y * y)], point) + pose[-1]
                expected = max(0., .05 * depth - camera_z) * 1000 / depth
                actual = cost.evaluate_arrays(point, pose)
                np.testing.assert_allclose(actual[0][0], expected, atol=1e-9)
                for block, values in enumerate((point, pose)):
                    for k in range(len(values)):
                        plus, minus = [point.copy(), pose.copy()], [point.copy(), pose.copy()]
                        plus[block][k] += 1e-6
                        minus[block][k] -= 1e-6
                        numeric = (cost.evaluate_arrays(*plus)[0] - cost.evaluate_arrays(*minus)[0]) / 2e-6
                        np.testing.assert_allclose(actual[block + 1][k], numeric[0], rtol=1e-5, atol=1e-3)
    for bad in (0., -1., float('nan'), float('inf')):
        with pytest.raises(ValueError):
            PositiveDepthCost(bad)
