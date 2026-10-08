import json

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from geosunpre.camera import Camera
from geosunpre.sfm.geometry import epipolar_pixel_error
from geosunpre.sfm.geometry import fit_pair
from geosunpre.sfm.geometry import rotation_graph_diagnostics
from geosunpre.sfm.geometry import skew
from geosunpre.scene import prepare


def camera():
    return Camera('L',3000,4000,(964.75,964.75,1516.291875,2001.706875,
                                -.02911172,-.0039987,.00086782,-.00035317),
                  (16.291875,-1.706875))


def test_native_fisheye_matches_colmap_at_periphery():
    cam=camera()
    angles=np.deg2rad(np.linspace(0,79.9,120))
    azimuth=np.linspace(0,7*np.pi,120)
    rays=np.column_stack((np.sin(angles)*np.cos(azimuth),np.sin(angles)*np.sin(azimuth),np.cos(angles)))
    pixels=cam.project(rays)
    np.testing.assert_allclose(pixels,cam.pycolmap().img_from_cam(rays),atol=1e-7)
    np.testing.assert_allclose(cam.rays(pixels),rays,atol=1e-9)
    assert cam.valid(pixels,160).all()
    assert not cam.valid(cam.project([[np.sin(np.deg2rad(81)),0,np.cos(np.deg2rad(81))]]),160)[0]


def test_epipolar_distance_in_native_pixels_and_independent_holdout():
    cam=camera()
    rng=np.random.default_rng(10)
    angles=rng.uniform(0,np.deg2rad(70),180)
    azimuth=rng.uniform(0,2*np.pi,180)
    xyz=np.column_stack((np.sin(angles)*np.cos(azimuth),np.sin(angles)*np.sin(azimuth),np.cos(angles)))
    xyz*=rng.uniform(5,15,(len(xyz),1))
    R=Rotation.from_euler('xyz',[2,4,-3],degrees=True).as_matrix()
    t=np.array([.7,.1,.05])
    a,b=cam.project(xyz),cam.project(xyz@R.T+t)
    exact=epipolar_pixel_error(cam,cam,a,b,skew(t)@R)
    assert exact.max()<1e-7
    a+=rng.normal(0,.1,a.shape);b+=rng.normal(0,.1,b.shape)
    b[-30:]=b[-30:][rng.permutation(30)]
    result,arrays=fit_pair(cam,cam,a,b,max_error=2,seed=42)
    assert result['success'] and result['verified']>=140
    assert result['train_count']==144 and result['holdout_count']==36
    assert result['holdout_epipolar_error_px_all_raw']['count']==36
    assert result['holdout_epipolar_error_px_all_raw']['p90']>10
    error=Rotation.from_matrix(arrays['pose_cam_b_from_a'][:,:3]@R.T).magnitude()
    assert np.rad2deg(error)<.15


def test_large_rotation_cross_fisheye_never_uses_raw_pixel_f():
    cam=camera()
    rng=np.random.default_rng(12)
    angle=np.deg2rad(rng.uniform(53,76,220))
    xyz=np.column_stack((np.sin(angle),rng.uniform(-.25,.25,len(angle)),np.cos(angle)))
    xyz*=rng.uniform(3,15,(len(angle),1))
    R=Rotation.from_euler('y',-128,degrees=True).as_matrix()
    t=-R@np.array([.12,0,.01])
    a=cam.project(xyz)+rng.normal(0,.1,(len(xyz),2))
    b=cam.project(xyz@R.T+t)+rng.normal(0,.1,(len(xyz),2))
    result,arrays=fit_pair(cam,cam,a,b,max_error=2)
    assert result['success'] and result['verified']>=210
    error=Rotation.from_matrix(arrays['pose_cam_b_from_a'][:,:3]@R.T).magnitude()
    assert np.rad2deg(error)<1
    assert 'E-only' in result['estimator']


def test_rotation_cycles_handle_reversed_edges_and_isolated_images():
    ra=Rotation.from_euler('xyz',[10,30,-20],degrees=True).as_matrix()
    rb=Rotation.from_euler('xyz',[-5,70,15],degrees=True).as_matrix()
    rc=Rotation.from_euler('xyz',[0,-40,90],degrees=True).as_matrix()
    edges=[('a','b',rb@ra.T,'ab'),('c','b',rb@rc.T,'cb'),('a','c',rc@ra.T,'ac')]
    report=rotation_graph_diagnostics(edges,['a','b','c','isolated'])
    assert report['summary']['component_sizes']==[3,1]
    assert report['summary']['triangles']==1
    assert report['summary']['rotation_closure_degrees']['median']<1e-10
    edges[-1]=('a','c',Rotation.from_euler('z',12,degrees=True).as_matrix()@rc@ra.T,'ac')
    report=rotation_graph_diagnostics(edges,['a','b','c','isolated'])
    np.testing.assert_allclose(report['summary']['rotation_closure_degrees']['median'],12,atol=1e-10)
    assert report['summary']['triangles_above_review_threshold']==1


def test_prepare_excludes_missing_pose_and_m_without_synchronizing(tmp_path):
    root=tmp_path/'input';root.mkdir()
    devices=[]
    for label in ['L','R']:
        devices.append({'deviceId':'camera_'+label,
                        'dataBasicParams':{'pixelSize':.01,'focalLength':.1,'photoWidth':30,'photoHeight':40},
                        'dataSensorParams':{'k1':-.01,'k2':0,'k3':0,'p3':-.0001,'principalX':.001,'principalY':.002}})
    (root/'scene.pro').write_text(json.dumps({'cameraDevices':{'devices':devices}}))
    pose_lines=[]
    for cam in ['L','R','M']:
        path=root/'Image'/cam/'0000';path.mkdir(parents=True)
        for frame in [0,1]:
            name=f'2435-35299{frame}-001_{frame:05d}-{cam}.JPG'
            Image.new('RGB',(30,40)).save(path/name)
            if cam!='M' and frame==1:
                pose_lines.append(name+' 1 2435 352991.002 100 200 300 0 0 0 1')
    (root/'scene_CameraPos_C2E.txt').write_text('\n'.join(pose_lines))
    manifest=prepare(root,tmp_path/'work')
    assert len(manifest['images'])==2
    assert manifest['excluded']=={'no_camera_pose':2,'M_or_other_camera':2}
    assert len({im['exposure_id'] for im in manifest['images']})==2
    assert {im['frame_index'] for im in manifest['images']}=={1}
    assert not manifest['pose_values_used_for_geometry']
    assert manifest['cameras']['L']['params'][-1]==-.0001
    assert not manifest['principal_convention_verified']
