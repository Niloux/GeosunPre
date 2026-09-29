"""Predict withheld tracks from two support views; no query enters fitting."""
import csv
from pathlib import Path
from geosunpre.profiling import profiled

import cv2
import numpy as np
from scipy.optimize import least_squares

from geosunpre.camera import Camera
from geosunpre.artifacts import write_json


def triangle_validation_split(split,entries,records,threshold=.9):
    """A correspondence-consistent validation subset, without any pose/error gate.

    This supplements, rather than replaces, evaluation of all raw heldout tracks.
    Every chosen query has direct high-confidence matches to both support points,
    and the supports also match each other. All remain held out from SfM/BA.
    """
    from collections import defaultdict
    held={tuple(n) for t in split['tracks'] for n in t['observations']}
    adjacency=defaultdict(dict)
    for entry in entries:
        with np.load(entry['path']) as data:
            for (i,j),score in zip(data['matches'],data['scores']):
                if score<threshold:continue
                a,b=(entry['a'],int(i)),(entry['b'],int(j))
                if a in held and b in held:
                    adjacency[a][b]=float(score);adjacency[b][a]=float(score)
    selected=[]
    for track in split['tracks']:
        nodes={tuple(n) for n in track['observations']}
        candidates=[]
        for a in sorted(nodes):
            for b,score in sorted(adjacency[a].items()):
                if b not in nodes or b<=a:continue
                gap=abs(records[a[0]]['frame_index']-records[b[0]]['frame_index'])
                if gap<2:continue
                queries=sorted(nodes & adjacency[a].keys() & adjacency[b].keys())
                if queries:
                    candidates.append((len(queries),gap,score,a,b,queries))
        if not candidates:continue
        _,_,score,a,b,queries=max(candidates)
        selected.append({**track,'support':[a,b],'queries':queries,'support_match_confidence':score})
    return {**split,'tracks':selected,'protocol':'direct_triangle_confidence',
            'match_confidence_threshold':threshold,'parent_heldout_tracks':len(split['tracks']),
            'selection_uses_pose_or_reprojection_error':False}


def error_summary(errors):
    errors=np.asarray(errors,dtype=float)
    if not len(errors):
        return {'count':0,'finite_count':0,'median':None,'p90':None,'p99':None,
                'fraction_le_1px':None,'fraction_le_2px':None,'fraction_le_3px':None}
    values=np.where(np.isfinite(errors),errors,np.inf)
    q=np.quantile(values,[.5,.9,.99],method='nearest')
    return {'count':len(values),'finite_count':int(np.isfinite(values).sum()),
            **{k:float(v) if np.isfinite(v) else None for k,v in zip(['median','p90','p99'],q)},
            **{f'fraction_le_{t}px':float(np.mean(values<=t)) for t in [1,2,3]}}


def triangulate_support(cams,poses,pixels):
    rays=[c.rays(np.asarray(xy)[None])[0] for c,xy in zip(cams,pixels)]
    h=cv2.triangulatePoints(poses[0],poses[1],
        (rays[0][:2]/rays[0][2]).reshape(2,1),(rays[1][:2]/rays[1][2]).reshape(2,1)).ravel()
    if abs(h[3])<1e-12:
        return None,np.inf,0.
    point=h[:3]/h[3]
    projection=[(cam.K,cam.D,pose,xy) for cam,pose,xy in zip(cams,poses,pixels)]
    zero=np.zeros(3)
    def residual(xyz):
        return np.concatenate([cv2.fisheye.projectPoints((pose[:,:3]@xyz+pose[:,3]).reshape(1,1,3),zero,zero,K,D)[0].reshape(2)-xy
                               for K,D,pose,xy in projection])
    def jacobian(xyz):
        # Specialize SciPy float64 unbounded 2-point differences to 3 unknowns.
        # Keep the signed step and actual representable dx used by _numdiff;
        # LM method, evaluation budget and all convergence tolerances are unchanged.
        f0=residual(xyz)
        h=np.finfo(np.float64).eps**.5*((xyz>=0).astype(float)*2-1)*np.maximum(1.,np.abs(xyz))
        result=np.empty((3,4))
        for i in range(3):
            x1=xyz.copy();x1[i]=xyz[i]+h[i]
            result[i]=(residual(x1)-f0)/(x1[i]-xyz[i])
        return result.T
    if not np.isfinite(point).all():
        return None,np.inf,0.
    fitted=least_squares(residual,point,jac=jacobian,method='lm',max_nfev=20,ftol=1e-6,xtol=1e-6,gtol=1e-6)
    point=fitted.x
    positive=all((pose[:,:3]@point+pose[:,3])[2]>1e-9 for pose in poses)
    world_rays=[pose[:,:3].T@ray for pose,ray in zip(poses,rays)]
    angle=float(np.rad2deg(np.arccos(np.clip(world_rays[0]@world_rays[1],-1,1))))
    support_error=float(np.linalg.norm(residual(point).reshape(2,2),axis=1).max())
    return (point if positive else None),support_error,angle


@profiled
def evaluate_reconstruction(rec,scene,features,split,directory):
    directory=Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    cameras={cid:Camera('L' if cid==1 else 'R',c.width,c.height,tuple(c.params),(0,0))
             for cid,c in rec.cameras.items()}
    original={k:Camera.from_dict(v) for k,v in scene['cameras'].items()}
    radii={k:cam.distorted_radius(160) for k,cam in original.items()}
    images={im.name:im for im in rec.images.values() if im.has_pose}
    records={im['name']:im for im in scene['images']}
    poses={name:im.cam_from_world().matrix() for name,im in images.items()}
    tracks=[track for track in split['tracks'] if track['queries']]
    count=sum(len(track['queries']) for track in tracks)
    name_width=max((len(name) for track in tracks for name,idx in track['queries']),default=1)
    track_width=max((len(str(track['id'])) for track in tracks),default=1)
    arrays={
        'track_id':np.empty(count,dtype=f'U{track_width}'),
        'name':np.fromiter((name for track in tracks for name,idx in track['queries']),
                           dtype=f'U{name_width}',count=count),
        'feature_index':np.fromiter((idx for track in tracks for name,idx in track['queries']),
                                    dtype=np.int64,count=count),
        'camera':np.empty(count,dtype=f'U{max(map(len,original),default=1)}'),
        'zone':np.empty(count,dtype='U6'), 'rho':np.empty(count),
        'observed':np.empty((count,2)), 'predicted':np.full((count,2),np.nan),
        'error':np.full(count,np.inf), 'support_error':np.empty(count),
        'support_angle':np.empty(count), 'support_ok':np.zeros(count,dtype=bool)}
    points=np.full((count,3),np.nan)
    offset=0
    for track in tracks:
        support=track['support']
        point=None;support_error=np.inf;angle=0.
        if all(name in images for name,idx in support):
            cs=[cameras[images[name].camera_id] for name,idx in support]
            ps=[poses[name] for name,idx in support]
            xy=[features[name]['keypoints'][idx].astype(float) for name,idx in support]
            point,support_error,angle=triangulate_support(cs,ps,xy)
        stop=offset+len(track['queries']);sl=slice(offset,stop)
        arrays['track_id'][sl]=track['id']
        arrays['support_error'][sl]=support_error
        arrays['support_angle'][sl]=angle
        arrays['support_ok'][sl]=bool(point is not None and support_error<=3 and angle>=1)
        if point is not None:points[sl]=point
        offset=stop
    # Group only projection work by image; output order and failure denominators
    # remain in the original frozen track/query order.
    grouped={}
    for index,name in enumerate(arrays['name']):grouped.setdefault(str(name),[]).append(index)
    for name,indices in grouped.items():
        indices=np.asarray(indices,dtype=np.int64)
        observed=features[name]['keypoints'][arrays['feature_index'][indices]].astype(float)
        arrays['observed'][indices]=observed
        label=records[name]['camera'];base=original[label]
        arrays['camera'][indices]=label
        rho=np.linalg.norm((observed-base.params[2:4])/base.params[:2],axis=1)/radii[label]
        arrays['rho'][indices]=rho
        arrays['zone'][indices]=np.where(rho<.5,'center',np.where(rho<.8,'middle','edge'))
        if name not in images:continue
        pose=poses[name]
        local=points[indices]@pose[:,:3].T+pose[:,3]
        valid=local[:,2]>1e-9
        selected=indices[valid]
        if len(selected):
            predicted=cameras[images[name].camera_id].project(local[valid])
            arrays['predicted'][selected]=predicted
            arrays['error'][selected]=np.linalg.norm(predicted-observed[valid],axis=1)
    del points
    np.savez_compressed(directory/'validation_observations.npz',**arrays)
    errors=arrays['error']
    groups={}
    for label in ['L','R']:
        groups[label]=error_summary(errors[arrays['camera']==label])
        for zone in ['center','middle','edge']:
            groups[f'{label}/{zone}']=error_summary(errors[(arrays['camera']==label)&(arrays['zone']==zone)])
    per_image={name:error_summary(errors[grouped[name]]) for name in sorted(grouped)}
    del grouped
    training=[];coverage={}
    for name,im in images.items():
        obs=[];xyz=[]
        for p in im.points2D:
            if p.has_point3D():
                obs.append(p.xy);xyz.append(rec.points3D[p.point3D_id].xyz)
        if not obs:
            continue
        pose=im.cam_from_world().matrix();xyz=np.asarray(xyz)@pose[:,:3].T+pose[:,3]
        error=np.linalg.norm(cameras[im.camera_id].project(xyz)-np.asarray(obs),axis=1)
        error[xyz[:,2]<=0]=np.inf;training.extend(error)
        xy=np.asarray(obs)
        occupied=len(np.unique(np.floor(xy/[cameras[im.camera_id].width,cameras[im.camera_id].height]*8).astype(int),axis=0))
        coverage[name]={'training_observations':len(obs),'occupied_8x8_cells':occupied,
                        'top_half_fraction':float(np.mean(xy[:,1]<cameras[im.camera_id].height/2))}
    summary={'status': 'evaluated' if count else 'no_queries',
        'holdout_isolation': split.get('isolation_scope', 'stage_only'),
        'all_candidate_queries':error_summary(errors),'by_camera_and_radius':groups,'per_image':per_image,
        'support_qualified_queries':error_summary(errors[arrays['support_ok']]),
        'training_reprojection':error_summary(training),'coverage':coverage,
        'note':'The holdout_isolation field records the split protocol; pipeline entrypoints verify its input identities. Legacy splits only certify stage-level exclusion. Only two support observations triangulate each validation point. Images themselves participate through other features. Raw tracks are not ground-truth correspondences. Failed predictions stay in denominators; null quantiles may mean nonfinite failures.'}
    write_json(directory/'validation.json',summary)
    with (directory/'validation.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['track','image','feature','camera','zone','error_px','support_error_px','support_angle_deg','support_ok'])
        keys=('track_id','name','feature_index','camera','zone','error','support_error','support_angle','support_ok')
        writer.writerows(zip(*(arrays[key] for key in keys)))
    return summary


def paired_common_support(directory,stages,subdirectory='', *, baseline_stage='fixed'):
    values={}
    for stage in stages:
        with np.load(directory/'stages'/stage/subdirectory/'validation_observations.npz') as data:
            values[stage]={k:data[k] for k in data.files}
    baseline=values[baseline_stage];common=np.ones(len(baseline['error']),dtype=bool)
    for stage,a in values.items():
        assert np.array_equal(a['name'],baseline['name']) and np.array_equal(a['feature_index'],baseline['feature_index'])
        common &= a['support_ok']
    result={'definition':'Identical queries whose two support observations satisfy support-only checks in all stages; never filtered on query error.',
            'baseline_stage':baseline_stage,
            'queries':int(common.sum()),'stages':{}}
    for stage,a in values.items():
        errors=a['error']
        groups={'all':error_summary(errors[common])}
        for label in ['L','R']:
            groups[label]=error_summary(errors[common&(a['camera']==label)])
            for zone in ['center','middle','edge']:
                groups[f'{label}/{zone}']=error_summary(errors[common&(a['camera']==label)&(a['zone']==zone)])
        finite=common&np.isfinite(errors)&np.isfinite(baseline['error'])
        delta_key='paired_delta_vs_fixed' if baseline_stage=='fixed' else 'paired_delta_vs_baseline'
        groups[delta_key]={'count':int(finite.sum()),
            'median_px':float(np.median(errors[finite]-baseline['error'][finite])) if finite.any() else None,
            'fraction_improved':float(np.mean(errors[finite]<baseline['error'][finite])) if finite.any() else None}
        result['stages'][stage]=groups
    filename='triangle_common_support_comparison.json' if subdirectory else 'common_support_comparison.json'
    write_json(directory/filename,result)
    return result,values,common
