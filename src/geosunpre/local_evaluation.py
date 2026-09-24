"""Predict withheld tracks from two support views; no query enters fitting."""
import csv
import html
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import least_squares

from .camera import Camera
from .scene import write_json


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
    def residual(xyz):
        return np.concatenate([cam.project((pose[:,:3]@xyz+pose[:,3])[None])[0]-xy
                               for cam,pose,xy in zip(cams,poses,pixels)])
    if not np.isfinite(point).all():
        return None,np.inf,0.
    fitted=least_squares(residual,point,method='lm',max_nfev=20,ftol=1e-6,xtol=1e-6,gtol=1e-6)
    point=fitted.x
    positive=all((pose[:,:3]@point+pose[:,3])[2]>1e-9 for pose in poses)
    world_rays=[pose[:,:3].T@ray for pose,ray in zip(poses,rays)]
    angle=float(np.rad2deg(np.arccos(np.clip(world_rays[0]@world_rays[1],-1,1))))
    support_error=float(np.linalg.norm(residual(point).reshape(2,2),axis=1).max())
    return (point if positive else None),support_error,angle


def evaluate_reconstruction(rec,scene,features,split,directory):
    directory=Path(directory)
    cameras={cid:Camera('L' if cid==1 else 'R',c.width,c.height,tuple(c.params),(0,0))
             for cid,c in rec.cameras.items()}
    original={k:Camera.from_dict(v) for k,v in scene['cameras'].items()}
    images={im.name:im for im in rec.images.values() if im.has_pose}
    records={im['name']:im for im in scene['images']}
    rows=[]
    for track in split['tracks']:
        support=track['support']
        point=None;support_error=np.inf;angle=0.
        if all(name in images for name,idx in support):
            ims=[images[name] for name,idx in support]
            cs=[cameras[im.camera_id] for im in ims]
            ps=[im.cam_from_world().matrix() for im in ims]
            xy=[features[name]['keypoints'][idx].astype(float) for name,idx in support]
            point,support_error,angle=triangulate_support(cs,ps,xy)
        for name,idx in track['queries']:
            observed=features[name]['keypoints'][idx].astype(float)
            predicted=np.full(2,np.nan);error=np.inf
            if point is not None and name in images:
                im=images[name];pose=im.cam_from_world().matrix()
                local=pose[:,:3]@point+pose[:,3]
                if local[2]>1e-9:
                    predicted=cameras[im.camera_id].project(local[None])[0]
                    error=float(np.linalg.norm(predicted-observed))
            label=records[name]['camera'];base=original[label]
            rho=float(np.linalg.norm((observed-base.params[2:4])/base.params[:2])/base.distorted_radius(160))
            zone='center' if rho<.5 else 'middle' if rho<.8 else 'edge'
            rows.append({'track_id':track['id'],'name':name,'feature_index':idx,'camera':label,
                'zone':zone,'rho':rho,'observed':observed,'predicted':predicted,'error':error,
                'support_error':support_error,'support_angle':angle,
                'support_ok':bool(point is not None and support_error<=3 and angle>=1)})
    arrays={k:np.asarray([r[k] for r in rows]) for k in ['track_id','name','feature_index','camera','zone','rho',
            'observed','predicted','error','support_error','support_angle','support_ok']}
    np.savez_compressed(directory/'validation_observations.npz',**arrays)
    errors=arrays['error']
    groups={}
    for label in ['L','R']:
        groups[label]=error_summary(errors[arrays['camera']==label])
        for zone in ['center','middle','edge']:
            groups[f'{label}/{zone}']=error_summary(errors[(arrays['camera']==label)&(arrays['zone']==zone)])
    per_image={name:error_summary(errors[arrays['name']==name]) for name in sorted(set(arrays['name']))}
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
    summary={'all_candidate_queries':error_summary(errors),'by_camera_and_radius':groups,'per_image':per_image,
        'support_qualified_queries':error_summary(errors[arrays['support_ok']]),
        'training_reprojection':error_summary(training),'coverage':coverage,
        'note':'Queries and support features are entirely withheld from SfM/BA. Only two support observations triangulate each validation point. Images themselves participate through other features. Raw tracks are not ground-truth correspondences. Failed predictions stay in denominators; null quantiles may mean nonfinite failures.'}
    write_json(directory/'validation.json',summary)
    with (directory/'validation.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['track','image','feature','camera','zone','error_px','support_error_px','support_angle_deg','support_ok'])
        for r in rows:writer.writerow([r['track_id'],r['name'],r['feature_index'],r['camera'],r['zone'],r['error'],r['support_error'],r['support_angle'],r['support_ok']])
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


def evaluate_triangle_protocol(out,scene,data,features):
    import pycolmap
    records={r['name']:r for r in scene['images']}
    for label,segment in data['segments'].items():
        directory=out/label
        split=json.loads(Path(segment['split_path']).read_text())
        triangle=triangle_validation_split(split,segment['pairs'],records)
        write_json(directory/'triangle_validation_split.json',triangle)
        for stage in ['fixed','focal_low_radial','all_radial','principal']:
            stage_dir=directory/'stages'/stage
            dest=stage_dir/'triangle_validation';dest.mkdir(exist_ok=True)
            rec=pycolmap.Reconstruction(stage_dir/'sparse/0')
            evaluate_reconstruction(rec,scene,features,triangle,dest)
        print(f'{label}: independent triangle validation {len(triangle["tracks"])} tracks',flush=True)


def prediction_previews(directory,scene,values,common):
    baseline,final=values['fixed'],values['principal']
    records={r['name']:r for r in scene['images']}
    out=directory/'previews';out.mkdir(exist_ok=True)
    selected=[]
    for name in sorted(set(baseline['name'])):
        ids=np.flatnonzero((baseline['name']==name)&common)
        if not len(ids):continue
        record=records[name]
        with Image.open(record['source']) as im: picture=im.convert('RGB').resize((600,800))
        canvas=Image.new('RGB',(1200,840),(24,24,24));draw=ImageDraw.Draw(canvas)
        for col,(title,arr) in enumerate([('Fixed intrinsics',baseline),('Refined principal stage',final)]):
            canvas.paste(picture,(col*600,40));draw.text((col*600+8,8),title,fill='white')
            for i in ids:
                if not np.isfinite(arr['error'][i]):continue
                xy=arr['observed'][i]*.2+[col*600,40]
                pred=arr['predicted'][i]*.2+[col*600,40]
                color=(60,225,90) if arr['error'][i]<=2 else (245,185,40) if arr['error'][i]<=5 else (250,65,70)
                draw.ellipse((xy[0]-2,xy[1]-2,xy[0]+2,xy[1]+2),outline=color,width=1)
                if arr['error'][i]<100:draw.line((tuple(xy),tuple(pred)),fill=color,width=1)
        label=f'{record["camera"]}{record["frame_index"]:05d}'
        canvas.save(out/f'{label}.jpg',quality=90)
        selected.append({'label':label,'path':f'previews/{label}.jpg','queries':len(ids),
                         'fixed':error_summary(baseline['error'][ids]),'refined':error_summary(final['error'][ids])})
    return selected


def write_experiment_report(out,scene,reports):
    sections=[]
    fmt=lambda v:f'{v:.3f}' if v is not None else '—'
    for label,report in reports.items():
        directory=out/label
        common,values,selection=paired_common_support(directory,report['stages'])
        triangle_text=''
        if (directory/'stages/fixed/triangle_validation/validation.json').exists():
            tri_common,values,selection=paired_common_support(directory,report['stages'],'triangle_validation')
            report['triangle_common_support']=tri_common
            triangle_rows=[]
            for stage,rr in tri_common['stages'].items():
                all_=rr['all'];fraction=all_['fraction_le_2px']
                full=json.loads((directory/'stages'/stage/'triangle_validation/validation.json').read_text())['all_candidate_queries']
                fraction_text=f'{fraction:.1%}' if fraction is not None else '—'
                triangle_rows.append(f'<tr><td>{stage}</td><td>{fmt(full["median"])} / {fmt(full["p90"])}</td>'
                    f'<td>{full["count"]-full["finite_count"]} / {full["count"]}</td>'
                    f'<td>{fmt(all_["median"])} / {fmt(all_["p90"])}</td><td>{fraction_text}</td></tr>')
            triangle_text=(f'<h3>高置信闭合三图子集：共同支持 {tri_common["queries"]} 个观测</h3>'
                '<p>三个图像对均有直接 LoMa 匹配，置信度均 ≥0.9；选择不使用相机姿态或重投影误差。'
                '此子集是本轮看到原始长轨迹问题后追加的诊断，不是预先锁定的验收集，也不是人工真值。'
                '共同支持要求两个支持观测在所有阶段均满足误差 ≤3px、视差 ≥1°，不检查查询误差。</p>'
                '<table><tr><th>阶段</th><th>全部子集 median / P90</th><th>预测失败 / 查询数</th>'
                '<th>共同支持 median / P90</th><th>共同支持 ≤2px</th></tr>'
                +''.join(triangle_rows)+'</table><p><a href="'+label+'/triangle_common_support_comparison.json">子集分区统计</a></p>')
        previews=prediction_previews(directory,scene,values,selection)
        report['common_support']=common
        lines=[]
        for stage,rr in report['stages'].items():
            raw=rr['validation']['all_candidate_queries'];qualified=common['stages'][stage]['all']
            fraction=qualified['fraction_le_2px']
            fraction_text=f'{fraction:.1%}' if fraction is not None else '—'
            lines.append(f'<tr><td>{stage}</td><td>{fmt(rr["validation"]["training_reprojection"]["median"])}</td>'
                         f'<td>{fmt(raw["median"])} / {fmt(raw["p90"])}</td>'
                         f'<td>{fmt(qualified["median"])} / {fmt(qualified["p90"])}</td>'
                         f'<td>{fraction_text}</td></tr>')
        cards=[]
        for card in previews:
            cards.append(f'<details><summary>{card["label"]} · {card["queries"]} 留出观测 · '
                         f'median {fmt(card["fixed"]["median"])} → {fmt(card["refined"]["median"])} px</summary>'
                         f'<img loading="lazy" src="{label}/{card["path"]}"></details>')
        statuses={stage:rr['ba'].get('termination_type',rr['ba']['solver'].split('Termination: ')[-1]).split('.')[-1]
                  for stage,rr in report['stages'].items()}
        solver_status=' · '.join(f'{stage}: {status}' for stage,status in statuses.items())
        if report['stages']['principal']['ba'].get('continuation'):
            solver_status+='；主点阶段达到迭代上限后有一次继续求解，详见 ba.json。'
        sections.append(f'<section><h2>{label}</h2><p>注册 {report["registered_images"]}/{report["selected_images"]}；'
                        f'{report["points"]} 个稀疏点；{report["split_tracks"]} 条完整留出轨迹；'
                        f'共同支持条件下 {common["queries"]} 个预测观测。</p>'
                        f'<p>数值停止状态：{html.escape(solver_status)}。收敛状态不等于正确标定。</p>'
                        '<table><tr><th>阶段</th><th>训练 median</th><th>全部候选留出 median / P90</th>'
                        '<th>共同支持留出 median / P90</th><th>共同支持 ≤2px</th></tr>'
                        +''.join(lines)+'</table><p><a href="'+label+'/common_support_comparison.json">L/R 与径向分区统计</a></p>'+triangle_text
                        +''.join(cards)+'</section>')
    write_json(out/'comparison.json',reports)
    body='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>篮球场局部 SfM 与标定对照</title>
<style>body{font:16px system-ui;margin:24px;background:#171b23;color:#ddd}a{color:#8bc5ff}table{border-collapse:collapse}td,th{padding:10px;border:1px solid #576070}section{margin:30px 0}img{max-width:100%}summary{padding:8px;cursor:pointer}p{line-height:1.6}</style>
<h1>篮球场局部 SfM 与标定对照</h1><p>全部误差单位均为 3000×4000 原图像素。每张图独立曝光；输入 pose 和 LAS 没有参与拟合。
完整留出轨迹在几何验证前隔离；两个支持观测用于三角化，其余观测用于预测。保留无法预测的失败，不以查询误差筛选验收集。</p>
<p>相机仍通过同一批图像中的其他特征估计。这是观测留出，不是新采集图像或人工真值；当前结果未通过最终 3DGS 一致性验收。</p>
<p>预览来自高置信闭合三图子集，左为固定内参，右为主点也参与优化的阶段。圆点是 LoMa 观测，短线指向预测；绿 ≤2px，黄 2–5px，红 >5px。
这是稀疏观测的对齐检查，不是密集图像配准或 3DGS 渲染。主点阶段不自动认定为最佳结果。</p>
<p><a href="mask_review.html">掩码原图对照</a> · <a href="comparison.json">完整对照数据</a> · <a href="../../../reports/basketball_local_sfm.md">实验说明与局限</a></p>'''+''.join(sections)+'</html>'
    (out/'index.html').write_text(body)
