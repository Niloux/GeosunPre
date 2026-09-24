"""Native-image evidence for selected heldout errors; no fitting or relabeling."""
from functools import lru_cache
import html
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .local_data import load_local
from .local_evaluation import paired_common_support
from .local_reconstruction import STAGES
from .scene import digest, file_digest, write_json


@lru_cache(maxsize=6)
def source_image(path):
    with Image.open(path) as im:return im.convert('RGB')


def select_diagnostic_queries(values,common,used,*,raw=False):
    """Fixed strata and representative tails, explicitly not an accuracy sample."""
    final=values['principal'];selected=[]
    for camera in ['L','R']:
        for zone in (['middle','edge'] if raw else ['center','middle','edge']):
            take=(final['camera']==camera)&(final['zone']==zone)&np.isfinite(final['error'])
            if not raw:take&=common
            for band in (['raw_tail'] if raw else ['near_5px','large_error','control_le_2px']):
                ids=np.flatnonzero(take)
                ids=[i for i in ids if str(final['track_id'][i]) not in used]
                if band=='raw_tail':ids=[i for i in ids if final['error'][i]>=30]
                elif band=='control_le_2px':ids=[i for i in ids if final['error'][i]<=2]
                else:ids=[i for i in ids if final['error'][i]>5]
                if not ids:continue
                if band=='near_5px':key=lambda i:(abs(final['error'][i]-6),str(final['track_id'][i]))
                elif band in ('large_error','raw_tail'):key=lambda i:(-min(final['error'][i],300),str(final['track_id'][i]))
                else:key=lambda i:digest([str(final['track_id'][i]),str(final['name'][i]),int(final['feature_index'][i])])
                i=min(ids,key=key);used.add(str(final['track_id'][i]));selected.append((i,band))
    return selected


def draw_marker(draw,xy,color):
    x,y=map(float,xy)
    draw.ellipse((x-5,y-5,x+5,y+5),outline=color,width=2)
    for dx,dy in [(-1,0),(1,0),(0,-1),(0,1)]:
        draw.line((x+dx*8,y+dy*8,x+dx*13,y+dy*13),fill=color,width=2)


def case_images(case,records,features,folder):
    columns=3;panel=408;patch=384;half=96
    full=Image.new('RGB',(columns*panel,850),(23,27,35));draw=ImageDraw.Draw(full)
    compact=Image.new('RGB',(columns*panel,446),(23,27,35));small=ImageDraw.Draw(compact)
    title=(f'{case["id"]} | {case["segment"]} | {case["protocol"]} {case["band"]} | '
           f'{case["camera"]}/{case["zone"]} | fixed {case["fixed_error_px"]:.2f} -> candidate {case["error_px"]:.2f} px')
    draw.text((10,6),title,fill='white');small.text((10,6),title,fill='white')
    observations=case['support']+[[case['name'],case['feature_index']]]
    original_patches=[]
    for col,(name,idx) in enumerate(observations):
        rec=records[name];xy=features[name]['keypoints'][idx].astype(float)
        picture=source_image(rec['source'])
        origin=np.floor(xy-half).astype(int)
        crop=picture.crop((int(origin[0]),int(origin[1]),int(origin[0]+half*2),int(origin[1]+half*2)))
        zoom=crop.resize((patch,patch),Image.Resampling.NEAREST)
        original_patches.append(zoom.copy())
        patch_draw=ImageDraw.Draw(zoom)
        draw_marker(patch_draw,(xy-origin)*2-.5,(70,245,120))
        if col==2:
            for key,color in [('fixed_predicted',(250,190,50)),('predicted',(250,75,95))]:
                pred=np.array(case[key])
                if np.isfinite(pred).all():
                    q=(pred-origin)*2-.5
                    if ((q>=0)&(q<patch)).all():draw_marker(patch_draw,q,color)
        context=picture.resize((288,384))
        context_draw=ImageDraw.Draw(context)
        cx,cy=xy*[288/picture.width,384/picture.height]
        context_draw.ellipse((cx-4,cy-4,cx+4,cy+4),outline=(70,245,120),width=2)
        context_draw.rectangle(tuple((origin*[288/picture.width,384/picture.height]).tolist()+
            ((origin+half*2)*[288/picture.width,384/picture.height]).tolist()),outline=(70,245,120),width=1)
        label=f'{"support" if col<2 else "query"} {rec["camera"]}{rec["frame_index"]:05d}  feature {idx}'
        draw.text((col*panel+12,31),label,fill='white')
        small.text((col*panel+12,31),label,fill='white')
        full.paste(context,(col*panel+60,52));full.paste(zoom,(col*panel+12,454))
        compact.paste(zoom,(col*panel+12,56))
    original=Image.new('RGB',(columns*panel,446),(23,27,35));plain=ImageDraw.Draw(original)
    plain.text((10,6),title,fill='white')
    for col,picture in enumerate(original_patches):original.paste(picture,(col*panel+12,56))
    full.save(folder/f'{case["id"]}.jpg',quality=94)
    compact.save(folder/f'{case["id"]}_zoom.jpg',quality=96)
    original.save(folder/f'{case["id"]}_plain.jpg',quality=96)
    return compact


def build_residual_review(work):
    local,scene,data,features=load_local(work)
    out=Path(work).resolve()/'residual_audit';folder=out/'cases';folder.mkdir(parents=True,exist_ok=True)
    records={r['name']:r for r in scene['images']}
    cases=[];provenance={}
    for label in data['segments']:
        root=local/label;used=set()
        for raw in [False,True]:
            sub='' if raw else 'triangle_validation'
            split_name='split.json' if raw else 'triangle_validation_split.json'
            tracks={r['id']:r for r in json.loads((root/split_name).read_text())['tracks']}
            _,values,common=paired_common_support(root,STAGES,sub)
            final=values['principal'];fixed=values['fixed']
            for i,band in select_diagnostic_queries(values,common,used,raw=raw):
                track=tracks[str(final['track_id'][i])]
                cases.append({'id':f'case_{len(cases)+1:03d}','segment':label,
                    'protocol':'raw_track' if raw else 'direct_triangle','band':band,
                    'track_id':str(final['track_id'][i]),'name':str(final['name'][i]),
                    'feature_index':int(final['feature_index'][i]),'camera':str(final['camera'][i]),
                    'zone':str(final['zone'][i]),'error_px':float(final['error'][i]),
                    'fixed_error_px':float(fixed['error'][i]),'support_error_px':float(final['support_error'][i]),
                    'support_angle_deg':float(final['support_angle'][i]),'support':track['support'],
                    'observed':final['observed'][i].tolist(),'predicted':final['predicted'][i].tolist(),
                    'fixed_predicted':fixed['predicted'][i].tolist()})
            provenance[f'{label}/{split_name}']=file_digest(root/split_name)
            for stage in ['fixed','principal']:
                p=root/'stages'/stage/sub/'validation_observations.npz'
                provenance[str(p.relative_to(local))]=file_digest(p)
    if any(not np.isfinite(c['fixed_error_px']) for c in cases):
        raise ValueError('Diagnostic sample includes a nonfinite baseline; handle explicitly before rendering')
    write_json(out/'cases.json',{'selection':'Diagnostic strata by segment/camera/radius with near-6px, largest-error and <=2px controls; raw-track tails added. Not a representative error-rate sample. One query per selected raw track.',
        'source_sha256':provenance,'cases':cases,'units':'native 3000x4000 pixels; zoom uses nearest-neighbor 2x display'})
    tiles=[]
    for case in cases:tiles.append(case_images(case,records,features,folder))
    sheets=[]
    for start in range(0,len(tiles),4):
        chunk=tiles[start:start+4];sheet=Image.new('RGB',(chunk[0].width,chunk[0].height*len(chunk)))
        for i,im in enumerate(chunk):sheet.paste(im,(0,i*im.height))
        name=f'sheet_{start//4+1:02d}.jpg';sheet.save(out/name,quality=94);sheets.append(name)
    render_residual_review(out)
    print(f'Residual review: {len(cases)} cases, {len(sheets)} crop sheets in {out}',flush=True)
    return out


def render_residual_review(out):
    out=Path(out);cases=json.loads((out/'cases.json').read_text())['cases']
    annotation_path=out/'annotations.json'
    annotations=json.loads(annotation_path.read_text()) if annotation_path.exists() else {}
    cards=[]
    for c in cases:
        note=annotations.get(c['id'],{})
        comment=html.escape(note.get('note','尚未标记；图片和几何误差都不能单独当作真值。'))
        label=html.escape(note.get('category','unreviewed'))
        cards.append(f'<details id="{c["id"]}" data-camera="{c["camera"]}"><summary>{c["id"]} · {c["segment"]} · '
            f'{c["protocol"]} · {c["camera"]}/{c["zone"]} · {c["fixed_error_px"]:.2f} → {c["error_px"]:.2f} px · {label}</summary>'
            f'<p>{comment}</p><p>支持点最大误差 {c["support_error_px"]:.3f} px，视差 {c["support_angle_deg"]:.2f}°。'
            f'<a href="cases/{c["id"]}_plain.jpg">无标记的原图局部</a> · '
            f'<a href="cases/{c["id"]}.jpg">全图位置与局部放大</a></p>'
            f'<img loading="lazy" src="cases/{c["id"]}.jpg"></details>')
    (out/'index.html').write_text('''<!doctype html><html lang="zh"><meta charset="utf-8"><title>原图残差复核</title>
<style>body{font:16px system-ui;background:#171b23;color:#ddd;margin:24px}a{color:#8bc5ff}p{line-height:1.7}img{max-width:100%}details{border:1px solid #576070;margin:12px 0;padding:10px}summary{cursor:pointer}</style>
<h1>前两段留出观测：原图残差复核</h1><p>每例左、中为两个留出支持观测，右为留出查询。绿色圆环：LoMa 观测；黄色：固定内参预测；红色：优化候选预测。
下方是 192×192 原始像素区域放大两倍；圆环中心保持可见。可打开无标记局部检查纹理。</p>
<p>样本按片段、L/R、径向分区选择大误差、接近 6px 的误差和 ≤2px 对照，并补充原始长轨迹大误差。
这是一组用于定位问题的样例，不用于估计整体错误率。目视说明属于图像证据判读，不是人工亚像素真值；未修改匹配、相机或验证点。</p>
<p><a href="cases.json">观测 ID 与来源</a> · <a href="../transfer_sfm/index.html">第三片段内参迁移检查</a></p>
'''+''.join(cards)+'''<script>function showCase(){const d=document.getElementById(location.hash.slice(1));if(d&&d.tagName==='DETAILS'){d.open=true;d.scrollIntoView()}}window.addEventListener('hashchange',showCase);showCase()</script></html>''')
