"""Local experiment evaluation pages and report generation."""
import html
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ..local_evaluation import (triangle_validation_split, evaluate_reconstruction,
                                paired_common_support, error_summary)
from ..scene import write_json

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
