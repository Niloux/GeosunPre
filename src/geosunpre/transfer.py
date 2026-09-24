"""Predeclared calibration transfer check on a third, disjoint image segment."""
from dataclasses import asdict
from datetime import datetime, timezone
import html
import json
import os
from pathlib import Path

import numpy as np
import pycolmap

from .local_data import CONFIG, build_frontend, load_local
from .local_evaluation import evaluate_reconstruction, paired_common_support, triangle_validation_split
from .local_reconstruction import (
    assert_holdout_isolated, bounded_bundle_adjustment, cheirality_health,
    database_and_seed, mapper_options,
)
from .scene import digest, file_digest, load_scene, write_json

OUTPUT = 'transfer_sfm'
SEGMENTS = {'transfer_145_169': (145, 169)}
CANDIDATES = ('pro_initial', 'from_normal', 'from_turn')


def prepare_transfer(work):
    from .static_masks import generate_masks, mask_contact_sheet, mask_review, MODEL_REVISION
    work=Path(work).resolve();out=work/OUTPUT
    out.mkdir(parents=True,exist_ok=True)
    base=load_scene(work)
    candidates={'pro_initial':{'params':{label:cam['params'] for label,cam in base['cameras'].items()},
                              'source':'Prepared .pro calibration; unverified initial convention'}}
    for name,segment in [('from_normal','normal_070_094'),('from_turn','turn_190_214')]:
        path=work/'local_sfm'/segment/'stages/principal/sparse/0'
        rec=pycolmap.Reconstruction(path)
        candidates[name]={'params':{'L' if cid==1 else 'R':c.params.tolist() for cid,c in rec.cameras.items()},
                          'source':str(path),'cameras_sha256':file_digest(path/'cameras.bin')}
    frames=[f for low,high in SEGMENTS.values() for f in range(low,high+1)]
    source_names=[r['name'] for r in base['images'] if r['frame_index'] in frames]
    previous=load_scene(work/'local_sfm')
    assert not set(source_names)&{r['name'] for r in previous['images']}
    protocol={'version':1,'segments':SEGMENTS,'image_names':source_names,'candidates':candidates,
              'frontend_config':asdict(CONFIG),'segmentation_revision':MODEL_REVISION,
              'mask_policy':'Same v2 masks and initial 160-degree FOV for all candidates',
              'holdout':{'fraction':.2,'seed':314159,'whole_raw_track_before_geometry':True,
                         'triangle_confidence':.9,'support_max_error_px':3.,'support_min_angle_deg':1.},
              'comparison':'Build one training reconstruction with initial calibration, clone its identical point/observation topology, fix each candidate calibration and optimize only independent exposure poses and 3D points.',
              'solver':{'huber_px':2.,'positive_depth_guard':True,'max_iterations':400,
                        'function_tolerance':1e-7,'initial_trust_region_radius':10.,'max_trust_region_radius':10000.},
              'pose_values_used':False,'las_used':False,
              'limitations':'Image-observation holdout, not ground truth or independent acquisition. Training topology is conditioned on initial-calibration mapping. This transfer check is not final 3DGS acceptance.'}
    fingerprint=digest(protocol)
    path=out/'protocol.json'
    if path.exists():
        if json.loads(path.read_text())['fingerprint']!=fingerprint:
            raise ValueError('Transfer protocol changed; preserve this experiment and use a fresh directory')
    else:
        write_json(path,{**protocol,'fingerprint':fingerprint,'frozen_at_utc':datetime.now(timezone.utc).isoformat()})
    os.environ.setdefault('HF_HOME',str(out/'hf_cache'))
    images,metadata=generate_masks(base,out/'masks_v2',frames,
        model_cache=work/'local_sfm/masks/model_cache')
    scene={**base,'images':images,'semantic_masks_available':True,'segmentation':metadata,
           'intrinsics_are_ground_truth':False,'scope':'Calibration transfer on a third continuous segment; all exposures independent.'}
    scene.pop('fingerprint');scene['fingerprint']=digest(scene)
    write_json(out/'manifest.json',scene)
    mask_contact_sheet(scene,images,out/'masks_v2/contact_sheet.jpg',[145,151,157,163,169])
    mask_review(scene,out,segments=SEGMENTS)
    return out


def transfer_frontend(work):
    data=build_frontend(work,output_name=OUTPUT,segments=SEGMENTS)
    out=Path(work).resolve()/OUTPUT;scene=load_scene(out)
    records={r['name']:r for r in scene['images']}
    for label,segment in data['segments'].items():
        split=json.loads(Path(segment['split_path']).read_text())
        triangle=triangle_validation_split(split,segment['pairs'],records,threshold=.9)
        triangle['selection_frozen_before_geometry']=True
        write_json(out/label/'triangle_validation_split.json',triangle)
        print(f'{label}: frozen {len(split["tracks"])} raw heldout tracks, {len(triangle["tracks"])} triangle tracks',flush=True)
    return data


def topology_signature(rec):
    return digest(sorted((int(pid),sorted((e.image_id,e.point2D_idx) for e in p.track.elements))
                         for pid,p in rec.points3D.items()))


def solve_transfer(work, *, continue_iteration_limit=False):
    out,scene,data,features=load_local(work,output_name=OUTPUT)
    protocol=json.loads((out/'protocol.json').read_text())
    reports={}
    records={im['name']:im for im in scene['images']}
    for label,segment in data['segments'].items():
        directory=out/label
        split=json.loads((directory/'split.json').read_text())
        triangle=json.loads((directory/'triangle_validation_split.json').read_text())
        frontend_identity={'scene_fingerprint':data['scene_fingerprint'],'config':data['config'],
                           'features':{n:data['features'][n]['token'] for n in segment['image_names']},
                           'pairs':[{k:e[k] for k in ('a','b','token')} for e in segment['pairs']]}
        contract={'version':2,'protocol':protocol['fingerprint'],'frontend':digest(frontend_identity),
                  'split_sha256':file_digest(directory/'split.json'),
                  'triangle_split_sha256':file_digest(directory/'triangle_validation_split.json')}
        contract['fingerprint']=digest(contract)
        path=directory/'run_contract.json'
        if path.exists():
            old=json.loads(path.read_text())
            # Version 1 included cache hit/miss counters in the frontend digest.
            # Migrate only while the original exact frontend snapshot is present.
            if old['version']==1 and old['frontend']==data['fingerprint'] and all(
                    old[k]==contract[k] for k in ['protocol','split_sha256','triangle_split_sha256']):
                write_json(directory/'run_contract_v1.json',old)
            elif old['fingerprint']!=contract['fingerprint']:
                raise ValueError('Transfer inputs changed; do not reuse existing models')
        write_json(path,contract)
        mapping=directory/'mapping'
        if not (mapping/'cameras.bin').exists():
            db,seed=directory/'database.db',directory/'seed'
            if not (db.exists() and (seed/'cameras.bin').exists()):
                db,seed=database_and_seed(directory,scene,[records[n] for n in segment['image_names']],
                                           segment['pairs'],features,split)
            opts=mapper_options()
            maps=pycolmap.incremental_mapping(db,Path(scene['input_root'])/'Image',directory/'mapper_output',
                                              options=opts,input_path=seed)
            if not maps:raise RuntimeError('Third segment did not reconstruct')
            rec=max(maps.values(),key=lambda r:r.num_reg_images())
            assert_holdout_isolated(rec,split)
            mapping.mkdir(exist_ok=True);rec.write(mapping)
            write_json(directory/'mapping_summary.json',{'selected':len(segment['image_names']),
                'registered':rec.num_reg_images(),'points':rec.num_points3D(),'observations':rec.compute_num_observations(),
                'options':opts.summary(),'pose_priors':False})
        base=pycolmap.Reconstruction(mapping);topology=topology_signature(base)
        stages={}
        for name in CANDIDATES:
            stage=directory/'stages'/name
            params={1 if label=='L' else 2:np.asarray(values,dtype=float)
                    for label,values in protocol['candidates'][name]['params'].items()}
            if (stage/'ba.json').exists():
                rec=pycolmap.Reconstruction(stage/'sparse/0');ba=json.loads((stage/'ba.json').read_text())
            else:
                rec=pycolmap.Reconstruction(mapping)
                for cid,values in params.items():rec.cameras[cid].params=values.copy()
                print(f'{label}: fixed calibration BA {name}',flush=True)
                ba=bounded_bundle_adjustment(rec,'fixed',params,protect_depth=True)
                model=stage/'sparse/0';model.mkdir(parents=True,exist_ok=True)
                rec.write(model);rec.write_text(model);rec.export_PLY(model/'points3D.ply')
                ba['candidate']=name;ba['calibration_fixed']=True;ba['topology_signature']=topology_signature(rec)
                write_json(stage/'ba.json',ba)
            if continue_iteration_limit and ba['termination_type'].endswith('NO_CONVERGENCE') and not ba.get('continuation'):
                archive=stage.with_name(name+'_iteration_limit')
                if archive.exists():raise FileExistsError(archive)
                amendment={'candidate':name,'trigger':ba['solver'],
                           'previous_ba_sha256':file_digest(stage/'ba.json'),
                           'recorded_at_utc':datetime.now(timezone.utc).isoformat(),
                           'reason':'Train-objective numerical follow-up after reaching the iteration limit. One continuation with identical fixed calibration and observations; not selected using validation errors.'}
                write_json(out/'numerical_followup.json',amendment)
                stage.rename(archive)
                previous=ba
                ba=bounded_bundle_adjustment(rec,'fixed',params,protect_depth=True)
                ba.update(candidate=name,calibration_fixed=True,topology_signature=topology_signature(rec),
                          continuation={'source':str(archive),'previous_solver':previous['solver'],
                                        'previous_elapsed_seconds':previous['elapsed_seconds'],
                                        'reason':amendment['reason']})
                model=stage/'sparse/0';model.mkdir(parents=True,exist_ok=True)
                rec.write(model);rec.write_text(model);rec.export_PLY(model/'points3D.ply')
                write_json(stage/'ba.json',ba)
            assert_holdout_isolated(rec,split)
            assert topology_signature(rec)==topology
            assert cheirality_health(rec)['invalid_depth']==0
            for cid,values in params.items():np.testing.assert_array_equal(rec.cameras[cid].params,values)
            raw=evaluate_reconstruction(rec,scene,features,split,stage)
            tri_dir=stage/'triangle_validation';tri_dir.mkdir(exist_ok=True)
            tri=evaluate_reconstruction(rec,scene,features,triangle,tri_dir)
            stages[name]={'ba':ba,'raw':raw,'triangle':tri}
        reports[label]={'selected':len(segment['image_names']),'registered':base.num_reg_images(),
                        'points':base.num_points3D(),'observations':base.compute_num_observations(),
                        'topology_signature':topology,'stages':stages}
    write_transfer_report(out,reports)
    return reports


def paired_track_bootstrap(values,common,baseline='pro_initial',replicates=1000):
    """Uncertainty of paired error changes, resampling complete heldout tracks."""
    base=values[baseline];result={}
    for name,v in values.items():
        selected=common&np.isfinite(v['error'])&np.isfinite(base['error'])
        track_ids=np.unique(base['track_id'][selected])
        deltas=v['error'][selected]-base['error'][selected]
        selected_tracks=base['track_id'][selected]
        chunks=[deltas[selected_tracks==t] for t in track_ids]
        if not chunks:
            result[name]={'queries':0,'tracks':0,'median_delta_px':None,'interval95_px':None};continue
        rng=np.random.default_rng(271829)
        medians=[float(np.median(np.concatenate([chunks[i] for i in rng.integers(len(chunks),size=len(chunks))])))
                 for _ in range(replicates)]
        result[name]={'queries':int(selected.sum()),'tracks':len(chunks),
                      'median_delta_px':float(np.median(np.concatenate(chunks))),
                      'interval95_px':np.quantile(medians,[.025,.975]).tolist(),
                      'replicates':replicates,'seed':271829,
                      'note':'Conditional paired median error change on finite common-support queries; not calibration uncertainty or a ground-truth accuracy interval.'}
    return result


def write_transfer_report(out,reports):
    fmt=lambda x:f'{x:.3f}' if x is not None else '—'
    sections=[]
    for label,report in reports.items():
        directory=out/label
        raw_common,_,_=paired_common_support(directory,CANDIDATES,baseline_stage='pro_initial')
        tri_common,values,common=paired_common_support(directory,CANDIDATES,'triangle_validation',baseline_stage='pro_initial')
        report['common_support']=raw_common;report['triangle_common_support']=tri_common
        report['paired_track_bootstrap']=paired_track_bootstrap(values,common)
        rows=[]
        for name,s in report['stages'].items():
            raw=s['raw']['all_candidate_queries'];tri=s['triangle']['all_candidate_queries']
            paired=tri_common['stages'][name]['all']
            fraction=paired['fraction_le_2px'];percent=f'{fraction:.1%}' if fraction is not None else '—'
            status=s['ba']['termination_type'].split('.')[-1]
            if s['ba'].get('continuation'):status+=' (一次继续求解)'
            rows.append(f'<tr><td>{name}</td><td>{fmt(s["raw"]["training_reprojection"]["median"])}</td>'
                f'<td>{fmt(raw["median"])} / {fmt(raw["p90"])}</td><td>{fmt(tri["median"])} / {fmt(tri["p90"])}</td>'
                f'<td>{tri["count"]-tri["finite_count"]} / {tri["count"]}</td>'
                f'<td>{fmt(paired["median"])} / {fmt(paired["p90"])}</td><td>{percent}</td><td>{html.escape(status)}</td></tr>')
        radial_rows=[]
        for zone in ['L/center','L/middle','L/edge','R/center','R/middle','R/edge']:
            entries=[tri_common['stages'][name][zone] for name in CANDIDATES]
            radial_rows.append(f'<tr><td>{zone}</td><td>{entries[0]["count"]}</td>'+''.join(
                f'<td>{fmt(entry["median"])} / {fmt(entry["p90"])}</td>' for entry in entries)+'</tr>')
        sections.append(f'<h2>{html.escape(label)}</h2><p>注册 {report["registered"]}/{report["selected"]}；'
            f'{report["points"]} 个相同三维轨迹、{report["observations"]} 个相同训练观测。'
            f'三组内参保持固定，共同支持查询 {tri_common["queries"]} 个。</p>'
            '<table><tr><th>候选内参</th><th>训练 median</th><th>原始留出 median / P90</th>'
            '<th>全部高置信子集 median / P90</th><th>预测失败 / 子集查询数</th>'
            '<th>共同支持 median / P90</th><th>共同支持 ≤2px</th><th>数值停止状态</th></tr>'+''.join(rows)+'</table>'
            '<h3>共同支持查询的 L/R 径向分区</h3><p>半径按初始内参下 160° FOV 归一化；center &lt;0.5，middle 0.5–0.8，edge ≥0.8。下面每格为 median / P90。</p>'
            '<table><tr><th>区域</th><th>查询数</th><th>.pro 初值</th><th>常规片段候选</th><th>转动片段候选</th></tr>'
            +''.join(radial_rows)+'</table>'
            f'<p><a href="{label}/triangle_common_support_comparison.json">L/R 及径向分区</a> · '
            f'<a href="{label}/comparison.json">逐阶段完整数据与按轨迹重采样的误差变化区间</a></p>')
        write_json(directory/'comparison.json',report)
    write_json(out/'comparison.json',reports)
    (out/'index.html').write_text('''<!doctype html><html lang="zh"><meta charset="utf-8"><title>第三片段内参迁移检查</title>
<style>body{font:16px system-ui;background:#171b23;color:#ddd;margin:24px}a{color:#8bc5ff}p{line-height:1.7}table{border-collapse:collapse}td,th{border:1px solid #576070;padding:10px}</style>
<h1>第三片段：固定候选内参的迁移检查</h1>
<p>第 145–169 帧，与前两个标定片段不重叠。全部误差均为 3000×4000 原图像素。
三组实验复用相同特征、匹配、训练点和观测；锁定候选内参，仅优化每张独立曝光的姿态与三维点。
初始内参建图决定训练拓扑，这是一项在该拓扑条件下的比较。输入 pose 和 LAS 不参与拟合。</p>
<p>完整轨迹在几何拟合前留出。闭合三图置信度 ≥0.9 的规则在本段拟合前固定；共同支持要求所有模型的两个支持观测误差 ≤3px、视差 ≥1°。
查询误差不参与筛选，失败保留在分母中。这是观测留出，仍非人工真值或最终 3DGS 验收。</p>
<p>表中保留各原始试验的停止状态；重启后的收敛标签不能单独证明结果稳定。后续如执行数值继续求解，应单独检查更大步长下的模型和误差变化。</p>
<p><a href="protocol.json">拟合前冻结的实验约定</a> · <a href="mask_review.html">50 张原图与掩码</a> · <a href="comparison.json">完整结果</a> · <a href="../residual_audit/index.html">前两段原图误差复核</a></p>
'''+''.join(sections)+'</html>')
