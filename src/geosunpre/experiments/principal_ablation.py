"""Secondary diagnostic isolating the transferred right-camera principal y."""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pycolmap

from .local_data import load_local
from ..local_evaluation import evaluate_reconstruction, paired_common_support
from ..local_reconstruction import assert_holdout_isolated, bounded_bundle_adjustment, cheirality_health
from ..scene import digest, file_digest, write_json
from .transfer import OUTPUT, topology_signature, paired_track_bootstrap

NAMES=('pro_initial','only_R_cy','from_normal','normal_without_R_cy')


def run_principal_ablation(work):
    parent,scene,data,features=load_local(work,output_name=OUTPUT)
    root=parent/'transfer_145_169';out=root/'principal_ablation';out.mkdir(exist_ok=True)
    primary=json.loads((parent/'protocol.json').read_text())
    original=primary['candidates']['pro_initial']['params']
    normal=primary['candidates']['from_normal']['params']
    only=deepcopy(original);only['R'][3]=normal['R'][3]
    without=deepcopy(normal);without['R'][3]=original['R'][3]
    variants={'only_R_cy':only,'normal_without_R_cy':without}
    protocol={'kind':'secondary_after_primary_transfer_results','candidates':variants,
              'controls':{'pro_initial':str(root/'stages/pro_initial'),
                          'from_normal':str(root/'stages/from_normal')},
              'primary_protocol':primary['fingerprint'],
              'mapping_sha256':{p.name:file_digest(p) for p in (root/'mapping').glob('*.bin')},
              'split_sha256':file_digest(root/'split.json'),
              'triangle_split_sha256':file_digest(root/'triangle_validation_split.json'),
              'calibration_fixed':True,'topology_fixed':True,
              'note':'Tests one parameter contribution conditional on existing training topology. No validation measurements or candidate parameter values are optimized. This is a follow-up diagnostic, not an independent final test.'}
    fingerprint=digest(protocol)
    if (out/'protocol.json').exists():
        if json.loads((out/'protocol.json').read_text())['fingerprint']!=fingerprint:
            raise ValueError('Ablation inputs changed')
    else:write_json(out/'protocol.json',{**protocol,'fingerprint':fingerprint,'frozen_at_utc':datetime.now(timezone.utc).isoformat()})
    split=json.loads((root/'split.json').read_text());triangle=json.loads((root/'triangle_validation_split.json').read_text())
    topology=topology_signature(pycolmap.Reconstruction(root/'mapping'))
    (out/'stages').mkdir(exist_ok=True)
    for name in ['pro_initial','from_normal']:
        target=out/'stages'/name
        if not target.exists():target.symlink_to(root/'stages'/name,target_is_directory=True)
    for name,values in variants.items():
        stage=out/'stages'/name;params={1 if label=='L' else 2:np.array(p) for label,p in values.items()}
        if (stage/'ba.json').exists():
            rec=pycolmap.Reconstruction(stage/'sparse/0')
        else:
            rec=pycolmap.Reconstruction(root/'mapping')
            for cid,p in params.items():rec.cameras[cid].params=p.copy()
            print(f'R cy ablation: {name}',flush=True)
            ba=bounded_bundle_adjustment(rec,'fixed',params,protect_depth=True)
            model=stage/'sparse/0';model.mkdir(parents=True,exist_ok=True)
            rec.write(model);rec.write_text(model);write_json(stage/'ba.json',ba)
        assert topology_signature(rec)==topology
        assert_holdout_isolated(rec,split)
        assert cheirality_health(rec)['invalid_depth']==0
        for cid,p in params.items():np.testing.assert_array_equal(rec.cameras[cid].params,p)
        evaluate_reconstruction(rec,scene,features,split,stage)
        tri=stage/'triangle_validation';tri.mkdir(exist_ok=True)
        evaluate_reconstruction(rec,scene,features,triangle,tri)
    common,_,_=paired_common_support(out,NAMES,baseline_stage='pro_initial')
    tri_common,values,take=paired_common_support(out,NAMES,'triangle_validation',baseline_stage='pro_initial')
    stages={name:{'ba':json.loads((out/'stages'/name/'ba.json').read_text()),
                  'raw':json.loads((out/'stages'/name/'validation.json').read_text()),
                  'triangle':json.loads((out/'stages'/name/'triangle_validation/validation.json').read_text())}
            for name in NAMES}
    result={'stages':stages,'common_support':common,'triangle_common_support':tri_common,
            'paired_track_bootstrap':paired_track_bootstrap(values,take),
            'topology_signature':topology,'kind':protocol['kind']}
    write_json(out/'comparison.json',result)
    rows=[]
    for name,s in stages.items():
        a=s['triangle']['all_candidate_queries'];b=tri_common['stages'][name]['all']
        rows.append(f'<tr><td>{name}</td><td>{a["median"]:.3f} / {a["p90"]:.3f}</td>'
                    f'<td>{b["median"]:.3f} / {b["p90"]:.3f}</td><td>{b["fraction_le_2px"]:.1%}</td>'
                    f'<td>{s["ba"]["termination_type"].split(".")[-1]}</td></tr>')
    (out/'index.html').write_text('''<!doctype html><html lang="zh"><meta charset="utf-8"><title>R 主点纵坐标消融</title>
<style>body{font:16px system-ui;background:#171b23;color:#ddd;margin:24px}p{line-height:1.7}a{color:#8bc5ff}td,th{padding:10px;border:1px solid #576070}table{border-collapse:collapse}</style>
<h1>第三片段：R 的 cy 作用</h1><p>在三组候选的迁移比较之后追加，属于解释性诊断。only_R_cy 仅将 .pro 初值中的 R cy 替换为常规片段候选值；normal_without_R_cy 保留常规候选的其余参数，将 R cy 恢复 .pro 初值。
所有内参固定，复用同一组训练点/观测和冻结的验证测量，只优化独立曝光位姿与三维点。</p>
<p>全部误差为原图像素。该消融只能说明此参数在现有数据与模型下的贡献，不能认证其真值，也不是新增的独立验收集。</p>
<table><tr><th>内参组合</th><th>全部高置信查询 median / P90</th><th>共同支持 median / P90</th><th>共同支持 ≤2px</th><th>数值状态</th></tr>'''+''.join(rows)+'''
</table><p><a href="comparison.json">全部结果、L/R 与径向分区</a> · <a href="protocol.json">消融约定</a> · <a href="../../index.html">主要迁移实验</a></p></html>''')
    return result
