"""Bounded image-only training-keypoint refinement with frozen validation."""
from datetime import datetime, timezone
import json
from pathlib import Path

import cv2
import numpy as np
import pycolmap
from PIL import Image

from .camera import Camera
from .local_data import load_local
from .local_evaluation import evaluate_reconstruction, paired_common_support
from .local_reconstruction import assert_holdout_isolated, bounded_bundle_adjustment
from .scene import digest, file_digest, write_json
from .transfer import topology_signature, paired_track_bootstrap

REFINEMENT_CONFIG={'half_window':7,'iterations':50,'epsilon_px':.01,
                   'jitter_px':.75,'jitter_agreement_px':.25,
                   'max_shift_px':3.,'min_corner_eigenvalue':1e-4}


def refine_native_corners(gray,pixels,config=REFINEMENT_CONFIG):
    """Use image gradients only; no camera pose, point3D, or target prediction."""
    pixels=np.asarray(pixels,dtype=np.float32)
    # OpenCV pixel-center coordinates start at 0; COLMAP corner coordinates at .5.
    opencv=pixels-.5
    window=(config['half_window'],)*2
    criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,config['iterations'],config['epsilon_px'])
    solutions=[]
    for jitter in [-config['jitter_px'],config['jitter_px']]:
        initial=(opencv+jitter).reshape(-1,1,2).copy()
        solutions.append(cv2.cornerSubPix(gray,initial,window,(-1,-1),criteria).reshape(-1,2))
    refined=(solutions[0]+solutions[1])*.5+.5
    stable=np.linalg.norm(solutions[0]-solutions[1],axis=1)<=config['jitter_agreement_px']
    shift=np.linalg.norm(refined-pixels,axis=1)
    finite=np.isfinite(refined).all(axis=1)
    eigen=cv2.cornerMinEigenVal(gray,blockSize=7,ksize=3)
    ij=np.floor(np.where(np.isfinite(refined),refined,0)).astype(int)
    ij[:,0]=ij[:,0].clip(0,gray.shape[1]-1);ij[:,1]=ij[:,1].clip(0,gray.shape[0]-1)
    response=eigen[ij[:,1],ij[:,0]]
    accepted=finite&stable&(shift<=config['max_shift_px'])&(response>=config['min_corner_eigenvalue'])
    return refined,accepted,{'shift':shift,'jitter_disagreement':np.linalg.norm(solutions[0]-solutions[1],axis=1),
                             'min_eigenvalue':response}


def refine_training_observations(rec,scene,split,directory):
    records={r['name']:r for r in scene['images']}
    held={(name,int(i)) for t in split['tracks'] for name,i in t['observations']}
    rows=[];per_image={}
    for i,image in enumerate(rec.images.values(),1):
        if not image.has_pose:continue
        indices=np.array([j for j,p in enumerate(image.points2D) if p.has_point3D()],dtype=int)
        if not len(indices):continue
        assert not any((image.name,int(j)) in held for j in indices)
        before=np.array([image.points2D[j].xy.copy() for j in indices])
        record=records[image.name]
        with Image.open(record['source']) as picture:gray=np.asarray(picture.convert('L'))
        refined,accepted,metrics=refine_native_corners(gray,before)
        with Image.open(record['semantic_mask']) as picture:mask=np.asarray(picture)>0
        finite=np.isfinite(refined).all(axis=1)
        ij=np.floor(np.where(np.isfinite(refined),refined,0)).astype(int)
        ij[:,0]=ij[:,0].clip(0,mask.shape[1]-1);ij[:,1]=ij[:,1].clip(0,mask.shape[0]-1)
        camera=Camera.from_dict(scene['cameras'][record['camera']])
        accepted&=finite&camera.valid(refined,scene['fov_degrees'])&mask[ij[:,1],ij[:,0]]
        for j,old,new,keep in zip(indices,before,refined,accepted):
            if keep:image.points2D[int(j)].xy=new.astype(float)
            rows.append((image.name,int(j),old,new,bool(keep)))
        per_image[image.name]={'training_observations':len(indices),'refined':int(accepted.sum())}
        print(f'training corner refinement {i}/{rec.num_reg_images()} {image.name}: {accepted.sum()}/{len(indices)}',flush=True)
    np.savez_compressed(directory/'refinement_observations.npz',
        name=np.array([r[0] for r in rows]),index=np.array([r[1] for r in rows]),
        before=np.array([r[2] for r in rows]),proposed=np.array([r[3] for r in rows]),accepted=np.array([r[4] for r in rows]))
    shifts=np.array([np.linalg.norm(r[3]-r[2]) for r in rows if r[4]])
    summary={'training_observations':len(rows),'refined':len(shifts),
             'accepted_shift_px':{'median':float(np.median(shifts)) if len(shifts) else None,
                                  'p90':float(np.quantile(shifts,.9)) if len(shifts) else None},
             'per_image':per_image,'validation_features_refined':0,'config':REFINEMENT_CONFIG,
             'note':'A local corner estimator is not a correspondence verifier. Original features and all validation observations remain unchanged.'}
    write_json(directory/'refinement.json',summary)
    return summary


def run_refinement_probe(work):
    local,scene,data,features=load_local(work)
    label='normal_070_094';root=local/label
    out=Path(work).resolve()/'feature_refinement'/label;out.mkdir(parents=True,exist_ok=True)
    source=root/'stages/principal/sparse/0'
    split=json.loads((root/'split.json').read_text())
    triangle=json.loads((root/'triangle_validation_split.json').read_text())
    protocol={'config':REFINEMENT_CONFIG,'segment':label,
              'source_model_sha256':{p.name:file_digest(p) for p in source.glob('*.bin')},
              'split_sha256':file_digest(root/'split.json'),
              'triangle_split_sha256':file_digest(root/'triangle_validation_split.json'),
              'calibration_fixed':True,'validation_features_refined':False,
              'same_track_topology':True,'hypothesis':'Native-image training-point localization may limit pose estimation. Supplemental diagnostic; not a predeclared final test.'}
    fingerprint=digest(protocol)
    if (out/'protocol.json').exists():
        if json.loads((out/'protocol.json').read_text())['fingerprint']!=fingerprint:
            raise ValueError('Refinement protocol changed; preserve this run')
    else:write_json(out/'protocol.json',{**protocol,'fingerprint':fingerprint,'frozen_at_utc':datetime.now(timezone.utc).isoformat()})
    base=pycolmap.Reconstruction(source);topology=topology_signature(base)
    params={cid:c.params.copy() for cid,c in base.cameras.items()}
    names=['loma_fixed','corner_refined'];reports={}
    for name in names:
        stage=out/'stages'/name;stage.mkdir(parents=True,exist_ok=True)
        if (stage/'ba.json').exists():
            rec=pycolmap.Reconstruction(stage/'sparse/0');ba=json.loads((stage/'ba.json').read_text())
        else:
            rec=pycolmap.Reconstruction(source)
            if name=='corner_refined':refine_training_observations(rec,scene,split,out)
            assert_holdout_isolated(rec,split)
            for t in split['tracks']:
                for image_name,index in t['observations']:
                    im=next(im for im in rec.images.values() if im.name==image_name)
                    np.testing.assert_array_equal(im.points2D[index].xy,features[image_name]['keypoints'][index])
            ba=bounded_bundle_adjustment(rec,'fixed',params,protect_depth=True)
            model=stage/'sparse/0';model.mkdir(parents=True,exist_ok=True)
            rec.write(model);rec.write_text(model);write_json(stage/'ba.json',ba)
        assert_holdout_isolated(rec,split)
        assert topology_signature(rec)==topology
        for cid,p in params.items():np.testing.assert_array_equal(rec.cameras[cid].params,p)
        raw=evaluate_reconstruction(rec,scene,features,split,stage)
        tri_dir=stage/'triangle_validation';tri_dir.mkdir(exist_ok=True)
        tri=evaluate_reconstruction(rec,scene,features,triangle,tri_dir)
        reports[name]={'ba':ba,'raw':raw,'triangle':tri}
    common,_,_=paired_common_support(out,names,baseline_stage='loma_fixed')
    tri_common,values,take=paired_common_support(out,names,'triangle_validation',baseline_stage='loma_fixed')
    result={'stages':reports,'common_support':common,'triangle_common_support':tri_common,
            'paired_track_bootstrap':paired_track_bootstrap(values,take,baseline='loma_fixed'),
            'topology_signature':topology,'validation_unchanged':True}
    write_json(out/'comparison.json',result)
    rows=[]
    for name,s in reports.items():
        a=s['triangle']['all_candidate_queries'];b=tri_common['stages'][name]['all']
        rows.append(f'<tr><td>{name}</td><td>{a["median"]:.3f} / {a["p90"]:.3f}</td>'
                    f'<td>{b["median"]:.3f} / {b["p90"]:.3f}</td><td>{b["fraction_le_2px"]:.1%}</td></tr>')
    (out/'index.html').write_text('''<!doctype html><html lang="zh"><meta charset="utf-8"><title>原图训练点精化对照</title>
<style>body{font:16px system-ui;background:#171b23;color:#ddd;margin:24px}p{line-height:1.7}a{color:#8bc5ff}td,th{padding:10px;border:1px solid #576070}table{border-collapse:collapse}</style>
<h1>70–94：原图训练点精化对照</h1><p>内参固定为本段上一轮候选。只对已进入训练的观测做局部灰度角点精化，要求两个扰动初值收敛一致、角点响应足够、位移 ≤3px、仍在有效域内。
训练点数量和拓扑保持相同，未通过条件的点保持原坐标；两组均重新优化姿态和三维点。所有留出支持点及查询点保留原始 LoMa 坐标。</p>
<p>这是看到局部定位歧义后追加的小范围诊断。局部角点算法不保证找到了相同物理点，也不能处理所有重复纹理或遮挡。全部误差以原图像素计。</p>
<p><strong>该试验未启用为默认流程。默认 SfM 仍使用原始 LoMa 观测。</strong></p>
<table><tr><th>训练观测</th><th>全部高置信查询 median / P90</th><th>共同支持 median / P90</th><th>共同支持 ≤2px</th></tr>'''+''.join(rows)+'''
</table><p><a href="comparison.json">全部结果与 L/R 分区</a> · <a href="refinement.json">实际精化数量和幅度</a> · <a href="protocol.json">实验约定</a></p></html>''')
    return result
