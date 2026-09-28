"""Image-only local SfM, explicit fisheye seeds and bounded shared calibration."""
import time

import cv2
import numpy as np
import pyceres
import pycolmap
from PIL import Image

from geosunpre.camera import Camera
from geosunpre.sfm.geometry import fit_pair
from geosunpre.sfm.geometry import rotation_graph_diagnostics
from geosunpre.sfm.tracks import train_edge_mask
from geosunpre.artifacts import write_json

STAGES = ('fixed','focal_low_radial','all_radial','principal')


def cheirality_health(rec):
    total=positive=0
    for im in rec.images.values():
        if not im.has_pose:continue
        xyz=np.array([rec.points3D[p.point3D_id].xyz for p in im.points2D if p.has_point3D()])
        if not len(xyz):continue
        pose=im.cam_from_world().matrix()
        local=xyz@pose[:,:3].T+pose[:,3]
        total+=len(local)
        positive+=int((np.isfinite(local).all(axis=1)&(local[:,2]>1e-9)).sum())
    return {'observations':total,'positive_depth':positive,'invalid_depth':total-positive}


def database_and_seed(directory, scene, images, entries, features, split):
    cameras={k:Camera.from_dict(v) for k,v in scene['cameras'].items()}
    records={im['name']:im for im in images}
    held={(name,int(idx)) for t in split['tracks'] for name,idx in t['observations']}
    candidates=[]
    directory.mkdir(parents=True,exist_ok=True)
    (directory/'geometry').mkdir(exist_ok=True)
    for i,entry in enumerate(entries,1):
        a,b=records[entry['a']],records[entry['b']]
        with np.load(entry['path']) as raw:
            matches=raw['matches']
            keep=train_edge_mask(a['name'],b['name'],matches,held)
            train=matches[keep]
        pa=features[a['name']]['keypoints'][train[:,0]].astype(float)
        pb=features[b['name']]['keypoints'][train[:,1]].astype(float)
        metrics,arrays=fit_pair(cameras[a['camera']],cameras[b['camera']],pa,pb,max_error=3,seed=271828)
        name=f'{a["camera"]}{a["frame_index"]:05d}__{b["camera"]}{b["frame_index"]:05d}'
        row={'label':name,'a':a['name'],'b':b['name'],'raw_count':len(matches),
             'after_holdout_removal':len(train),**metrics}
        if metrics['success']:
            inliers=train[arrays['verified_mask']]
            candidate={**row,'inliers':inliers,'pose':arrays['pose_cam_b_from_a'],
                       'essential':arrays['essential'],
                       'angles':arrays['triangulation_angle_degrees'][arrays['verified_mask']]}
            candidates.append(candidate)
        write_json(directory/'geometry'/f'{name}.json',row)
        if i%40==0 or i==len(entries):
            print(f'{directory.name}: train E verification {i}/{len(entries)}',flush=True)
    graph=rotation_graph_diagnostics([(r['a'],r['b'],r['pose'][:,:3],r['label']) for r in candidates],records)
    write_json(directory/'rotation_cycles.json',graph)
    closure={r['label']:r for r in graph['pairs']}
    accepted=[]
    for row in candidates:
        c=closure[row['label']]
        median=c['rotation_closure_degrees']['median']
        row['excluded_by_rotation_consensus']=bool(c['triangle_count']>=3 and median is not None and median>5)
        if len(row['inliers'])>=30 and not row['excluded_by_rotation_consensus']:
            accepted.append(row)
    write_json(directory/'geometry_summary.json',{
        'pairs':len(entries),'successful_E_pairs':len(candidates),'database_pairs':len(accepted),
        'rotation_consensus_excluded':[r['label'] for r in candidates if r['excluded_by_rotation_consensus']],
        'heldout_observations':len(held),'holdout_isolated_before_geometry':True,
        'note':'Rotation-median screening is a robust heuristic, not a ground-truth edge label.'})
    # Prefer a same-camera baseline with large support. No navigation pose enters.
    seeds=[r for r in accepted if records[r['a']]['camera']==records[r['b']]['camera']
           and len(r['inliers'])>=100 and np.median(r['angles'])>=2
           and (closure[r['label']]['rotation_closure_degrees']['median'] or 0)<2]
    if not seeds:
        raise RuntimeError('No well-supported image-only fisheye seed')
    seeds.sort(key=lambda r:len(r['inliers'])*min(float(np.median(r['angles'])),10),reverse=True)
    seed=seeds[0]
    path=directory/'database.db'
    if path.exists():
        raise FileExistsError(f'Database already exists without completed mapping: {path}')
    with pycolmap.Database.open(path) as db:
        for label,cam in cameras.items():
            camera=cam.pycolmap();camera.camera_id=1 if label=='L' else 2
            db.write_camera(camera,use_camera_id=True)
            rig=pycolmap.Rig(rig_id=camera.camera_id)
            rig.add_ref_sensor(pycolmap.sensor_t(pycolmap.SensorType.CAMERA,camera.camera_id))
            db.write_rig(rig,use_rig_id=True)
        for im in images:
            cid=1 if im['camera']=='L' else 2
            image=pycolmap.Image(name=im['name'],camera_id=cid,image_id=im['image_id'])
            image.frame_id=im['image_id']
            frame=pycolmap.Frame(frame_id=image.frame_id,rig_id=cid)
            frame.add_data_id(image.data_id)
            db.write_frame(frame,use_frame_id=True)
            db.write_image(image,use_image_id=True)
            db.write_keypoints(image.image_id,features[im['name']]['keypoints'])
        for row in accepted:
            a,b=records[row['a']],records[row['b']]
            assert train_edge_mask(row['a'],row['b'],row['inliers'],held).all()
            geometry=pycolmap.TwoViewGeometry(config=pycolmap.TwoViewGeometryConfiguration.CALIBRATED,
                E=row['essential'],cam2_from_cam1=pycolmap.Rigid3d(row['pose']),
                inlier_matches=row['inliers'],tri_angle=float(np.deg2rad(np.median(row['angles']))))
            db.write_matches(a['image_id'],b['image_id'],row['inliers'])
            db.write_two_view_geometry(a['image_id'],b['image_id'],geometry)
        assert db.num_pose_priors()==0 and db.num_frames()==len(images)
    reconstruction=create_seed(cameras,records,features,seed)
    seed_path=directory/'seed';seed_path.mkdir(exist_ok=True)
    reconstruction.write(seed_path)
    write_json(directory/'seed.json',{'pair':seed['label'],'points':reconstruction.num_points3D(),
        'initialization':'Explicit camera-aware E-only relative pose. Bypasses generic mapper two-view model selection.',
        'pose_prior_used':False})
    return path,seed_path


def create_seed(cameras,records,features,seed):
    rec=pycolmap.Reconstruction()
    for label,cam in cameras.items():
        camera=cam.pycolmap();camera.camera_id=1 if label=='L' else 2
        rec.add_camera(camera)
        rig=pycolmap.Rig(rig_id=camera.camera_id)
        rig.add_ref_sensor(pycolmap.sensor_t(pycolmap.SensorType.CAMERA,camera.camera_id))
        rec.add_rig(rig)
    a,b=records[seed['a']],records[seed['b']]
    for im,pose in [(a,np.column_stack((np.eye(3),np.zeros(3)))),(b,seed['pose'])]:
        cid=1 if im['camera']=='L' else 2
        image=pycolmap.Image(name=im['name'],keypoints=features[im['name']]['keypoints'].astype(float),
                            camera_id=cid,image_id=im['image_id'])
        image.frame_id=image.image_id
        frame=pycolmap.Frame(frame_id=image.frame_id,rig_id=cid,rig_from_world=pycolmap.Rigid3d(pose))
        frame.add_data_id(image.data_id);rec.add_frame(frame);rec.add_image(image);rec.register_frame(frame.frame_id)
    matches=seed['inliers']
    pa=features[a['name']]['keypoints'][matches[:,0]]
    pb=features[b['name']]['keypoints'][matches[:,1]]
    ra,rb=cameras[a['camera']].rays(pa),cameras[b['camera']].rays(pb)
    h=cv2.triangulatePoints(np.column_stack((np.eye(3),np.zeros(3))),seed['pose'],
        (ra[:,:2]/ra[:,2:]).T,(rb[:,:2]/rb[:,2:]).T).T
    xyz=h[:,:3]/h[:,3:]
    with Image.open(a['source']) as im:rgb=np.asarray(im.convert('RGB'))
    for (ia,ib),point,xy,angle in zip(matches,xyz,pa,seed['angles']):
        if not np.isfinite(point).all() or angle<1:
            continue
        track=pycolmap.Track([pycolmap.TrackElement(a['image_id'],int(ia)),pycolmap.TrackElement(b['image_id'],int(ib))])
        x,y=np.floor(xy).astype(int)
        rec.add_point3D(point,track,rgb[y,x])
    return rec


def mapper_options():
    opts=pycolmap.IncrementalPipelineOptions()
    opts.num_threads=4;opts.random_seed=271828;opts.multiple_models=False
    opts.min_model_size=10;opts.min_num_matches=30
    opts.structure_less_registration_fallback=False
    opts.ba_refine_focal_length=False;opts.ba_refine_principal_point=False
    opts.ba_refine_extra_params=False;opts.ba_refine_sensor_from_rig=False
    opts.use_prior_position=False
    opts.mapper.abs_pose_refine_focal_length=False;opts.mapper.abs_pose_refine_extra_params=False
    opts.mapper.abs_pose_max_error=6;opts.mapper.abs_pose_min_num_inliers=30
    opts.mapper.filter_max_reproj_error=4;opts.mapper.filter_min_tri_angle=1
    opts.triangulation.min_angle=1
    opts.ba_local_max_num_iterations=35;opts.ba_global_max_num_iterations=60
    return opts


def bounded_bundle_adjustment(rec,stage,initial_cameras, *, protect_depth=False, trust_region=None,
                              show_progress=False, max_iterations=None):
    options=pycolmap.BundleAdjustmentOptions()
    free=stage!='fixed'
    options.refine_focal_length=free;options.refine_extra_params=free
    options.refine_principal_point=stage=='principal';options.refine_sensor_from_rig=False
    options.ceres.loss_function_type=pycolmap.LossFunctionType.HUBER
    options.ceres.loss_function_scale=2
    options.ceres.solver_options.num_threads=4
    options.ceres.solver_options.max_num_iterations=100
    options.ceres.solver_options.minimizer_progress_to_stdout=show_progress
    guarded=stage=='principal' or protect_depth
    if guarded:
        # Native COLMAP residuals can become zero for non-projectable points.
        # Limit steps when releasing principal point and audit cheirality after
        # solving; Ceres "CONVERGENCE" is never sufficient for acceptance.
        options.ceres.solver_options.initial_trust_region_radius=10
        options.ceres.solver_options.max_trust_region_radius=10000
        options.ceres.solver_options.max_num_iterations=400
        # Python cost callbacks execute on one thread. The geometric factors
        # remain native Ceres; only the positivity guard has a Python callback.
        options.ceres.solver_options.num_threads=1
    if trust_region is not None:
        options.ceres.solver_options.initial_trust_region_radius=float(trust_region[0])
        options.ceres.solver_options.max_trust_region_radius=float(trust_region[1])
    if max_iterations is not None:
        options.ceres.solver_options.max_num_iterations=int(max_iterations)
    options.ceres.solver_options.function_tolerance=1e-7
    options.print_summary=show_progress
    config=pycolmap.BundleAdjustmentConfig()
    for image_id in rec.reg_image_ids():config.add_image(image_id)
    config.fix_gauge(pycolmap.BundleAdjustmentGauge.TWO_CAMS_FROM_WORLD)
    adjuster=pycolmap.create_default_ceres_bundle_adjuster(options,config,rec)
    problem=adjuster.problem
    depth_costs=[]
    if guarded:
        from geosunpre.sfm.depth_guard import add_positive_depth_guards
        depth_costs=add_positive_depth_guards(problem,rec)
    manifolds=[]
    bounds={}
    if free:
        for cid,cam in rec.cameras.items():
            base=np.asarray(initial_cameras[cid],dtype=float)
            delta=np.array([base[0]*.1,base[1]*.1,50,50,.05,.02,.01,.005])
            lower,upper=base-delta,base+delta
            for j in range(8):
                problem.set_parameter_lower_bound(cam.params,j,float(lower[j]))
                problem.set_parameter_upper_bound(cam.params,j,float(upper[j]))
            constant={'focal_low_radial':[2,3,6,7],'all_radial':[2,3],'principal':[]}[stage]
            manifold=pyceres.SubsetManifold(8,constant)
            manifolds.append(manifold);problem.set_manifold(cam.params,manifold)
            bounds[cid]={'lower':lower.tolist(),'upper':upper.tolist()}
    started=time.monotonic();summary=adjuster.solve();rec.update_point_3d_errors()
    if not summary.is_solution_usable():
        raise RuntimeError(f'BA failed: {summary.brief_report()}')
    depth=cheirality_health(rec)
    if depth['invalid_depth']:
        raise RuntimeError(f'BA produced nonphysical depths: {depth}; result must not be accepted')
    barrier_health={'active':0,'cost':0.}
    if depth_costs:
        depths=[]
        for image in rec.images.values():
            if not image.has_pose:continue
            xyz=np.array([rec.points3D[p.point3D_id].xyz for p in image.points2D if p.has_point3D()])
            if not len(xyz):continue
            pose=image.cam_from_world().matrix()
            depths.extend((xyz@pose[2,:3]+pose[2,3]).tolist())
        assert len(depths)==len(depth_costs)
        residuals=np.array([max(c.minimum-z,0)*c.scale for c,z in zip(depth_costs,depths)])
        barrier_health={'active':int((residuals>0).sum()),'cost':float(.5*(residuals**2).sum())}
    health={}
    for cid,cam in rec.cameras.items():
        candidate=Camera('L' if cid==1 else 'R',cam.width,cam.height,tuple(cam.params),(0,0))
        try:candidate.distorted_radius(160);monotonic=True
        except ValueError:monotonic=False
        if not monotonic:
            raise RuntimeError(f'BA produced nonmonotonic fisheye calibration for camera {cid}')
        hit=[]
        if cid in bounds:
            lo=np.array(bounds[cid]['lower']);hi=np.array(bounds[cid]['upper'])
            hit=np.flatnonzero(np.minimum(cam.params-lo,hi-cam.params)<1e-4*(hi-lo)).tolist()
        health[str(cid)]={'params':cam.params.tolist(),'delta':(cam.params-initial_cameras[cid]).tolist(),
                          'monotonic_in_160deg':monotonic,'parameters_at_bounds':hit}
    return {'stage':stage,'elapsed_seconds':time.monotonic()-started,'solver':summary.brief_report(),
            'solver_details':{'message':summary.ceres_summary.message,
                'initial_cost':summary.ceres_summary.initial_cost,
                'final_cost':summary.ceres_summary.final_cost,
                'successful_steps':summary.ceres_summary.num_successful_steps,
                'unsuccessful_steps':summary.ceres_summary.num_unsuccessful_steps,
                'full_report':summary.ceres_summary.FullReport()},
            'cheirality':depth,
            'positive_depth_barriers':len(depth_costs),
            'positive_depth_barrier_health':barrier_health,
            'termination_type':str(summary.termination_type),
            'max_iterations':options.ceres.solver_options.max_num_iterations,
            'initial_trust_region_radius':options.ceres.solver_options.initial_trust_region_radius,
            'max_trust_region_radius':options.ceres.solver_options.max_trust_region_radius,
            'cameras':health,'bounds':bounds,'observations':rec.compute_num_observations(),
            'bound_note':'Exploratory numerical limits around uncertain calibration, not measured uncertainty or truth.'}


def assert_holdout_isolated(rec,split):
    names={im.name:im for im in rec.images.values()}
    for track in split['tracks']:
        for name,idx in track['observations']:
            if name in names:
                if not 0 <= idx < len(names[name].points2D):
                    raise ValueError(f'Validation feature index differs from model: {name}:{idx}')
                if names[name].points2D[idx].has_point3D():
                    raise AssertionError(f'Validation observation leaked: {name}:{idx}')
