import numpy as np
from PIL import Image
import pycolmap

from geosunpre.camera import Camera
from geosunpre.local_data import split_tracks,train_edge_mask
from geosunpre.local_evaluation import error_summary,triangulate_support
from geosunpre.local_reconstruction import create_seed,bounded_bundle_adjustment,mapper_options,cheirality_health
from geosunpre.depth_guard import PositiveDepthCost


def test_depth_barrier_penalizes_invalid_projection_and_has_correct_jacobians():
    cost=PositiveDepthCost(5.)
    point=np.array([1.,2.,-.4])
    pose=np.array([.1,-.2,.05,.97,.2,.3,.1]);pose[:4]/=np.linalg.norm(pose[:4])
    # Force the barrier active; quaternion parameter order is Eigen xyzw.
    pose[-1]=-3
    r=np.zeros(1);jac=[np.zeros(3),np.zeros(7)]
    cost.Evaluate([point,pose],r,jac)
    assert r[0]>0
    for block,values in enumerate([point,pose]):
        for index in range(len(values)):
            plus=[point.copy(),pose.copy()];minus=[point.copy(),pose.copy()]
            plus[block][index]+=1e-6;minus[block][index]-=1e-6
            rp=np.zeros(1);rm=np.zeros(1)
            cost.Evaluate(plus,rp,None);cost.Evaluate(minus,rm,None)
            np.testing.assert_allclose(jac[block][index],(rp[0]-rm[0])/2e-6,rtol=1e-6,atol=1e-6)


def test_whole_track_holdout_removes_all_incident_edges_before_geometry():
    images=[{'name':f'i{i}','camera':'L','frame_index':i} for i in range(5)]
    pairs=[(f'i{i}',f'i{i+1}',np.array([[0,0],[1,1]]),np.array([.9,.8])) for i in range(4)]
    split,held=split_tracks(pairs,images,fraction=1)
    assert len(split['tracks'])==2 and len(held)==10
    # A conflicting edge outside the raw-track union must also be removed.
    assert not train_edge_mask('i0','i4',np.array([[0,2],[2,1]]),held).any()
    assert train_edge_mask('i0','i4',np.array([[2,2]]),held).all()
    assert all(len(t['queries'])==3 for t in split['tracks'])


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
    point,error,angle=triangulate_support([cam,cam],[p0,p1],[features['a']['keypoints'][10],features['b']['keypoints'][10]])
    np.testing.assert_allclose(point,xyz[10],atol=1e-7)
    assert error<1e-7 and angle>1
    opts=mapper_options()
    assert not opts.use_prior_position and not opts.ba_refine_sensor_from_rig
    assert not opts.mapper.abs_pose_refine_focal_length
    for point in rec.points3D.values():
        point.xyz[2]*=-1
    assert cheirality_health(rec)=={'observations':160,'positive_depth':0,'invalid_depth':160}
