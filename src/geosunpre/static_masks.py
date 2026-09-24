"""Conservative local semantic masks, kept distinct from hard fisheye validity."""
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .camera import Camera
from .scene import file_digest, write_json

MODEL = 'nvidia/segformer-b2-finetuned-ade-512-512'
MODEL_REVISION = 'de01bae28967510f9ddd496c60a969357195400c'
EXCLUDED = {'sky', 'person'}
MASK_VERSION = 2


def hardware_guard(width, height, camera):
    """Manually inspected overhang outlines in 3000x4000 reference coordinates."""
    polygons = {
        'L':[(1500,0),(3000,0),(3000,1500),(2700,1150),(2300,1030),
             (2000,1020),(1780,860),(1500,690)],
        'R':[(0,0),(1700,0),(1670,700),(1470,930),(1200,1030),
             (900,1100),(600,1260),(350,1340),(0,1750)]}
    mask=np.zeros((height,width),dtype=np.uint8)
    polygon=(np.asarray(polygons[camera])*[width/3000,height/4000]).astype(np.int32)
    cv2.fillPoly(mask,[polygon],1)
    mask[int(.8*height):]=1
    return mask.astype(bool)


def virtual_views(cam, size=512, fov=100):
    """Pure image resampling; the five views never become SfM cameras."""
    from scipy.spatial.transform import Rotation
    f=size/(2*np.tan(np.deg2rad(fov/2)))
    yy,xx=np.mgrid[:size,:size]
    local=np.stack(((xx+.5-size/2)/f,(yy+.5-size/2)/f,np.ones_like(xx)),axis=-1)
    for axis,angle in [('y',0),('y',45),('y',-45),('x',45),('x',-45)]:
        R=Rotation.from_euler(axis,angle,degrees=True).as_matrix()
        rays=local.reshape(-1,3)@R.T
        pixels=cam.project(rays).reshape(size,size,2)-.5
        valid=rays[:,2]/np.linalg.norm(rays,axis=1)>=np.cos(np.deg2rad(80))
        pixels[~valid.reshape(size,size)]=-10
        yield R,f,pixels.astype(np.float32)


def predict_sky_person(original,cam,model,processor):
    import torch
    # Fuse probabilities on a reduced native-image grid, keeping continuous
    # probabilities so uncertainty near poles/sky boundaries is not hard-cut.
    width,height=900,1200
    yy,xx=np.mgrid[:height,:width]
    pixels=np.stack(((xx+.5)*cam.width/width,(yy+.5)*cam.height/height),axis=-1)
    valid=cam.valid(pixels)
    rays=cam.rays(pixels.reshape(-1,2)).reshape(height,width,3)
    sky_id=next(int(k) for k,v in model.config.id2label.items() if v=='sky')
    person_id=next(int(k) for k,v in model.config.id2label.items() if v=='person')
    sums=np.zeros((height,width,2),dtype=np.float32)
    weights=np.zeros((height,width),dtype=np.float32)
    for R,f,source in virtual_views(cam):
        view=cv2.remap(original,source[...,0],source[...,1],cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT)
        with torch.inference_mode():
            inp={k:v.cuda() for k,v in processor(images=Image.fromarray(view),return_tensors='pt').items()}
            probability=model(**inp).logits.softmax(1)[:,[sky_id,person_id]]
            probability=torch.nn.functional.interpolate(probability,size=(512,512),mode='bilinear',
                                                        align_corners=False)[0].permute(1,2,0).cpu().numpy()
        local=rays@R
        z=local[...,2]
        uv=local[...,:2]/np.maximum(z[...,None],1e-6)*f+256-.5
        inside=valid&(z>.1)&(uv[...,0]>=0)&(uv[...,0]<511)&(uv[...,1]>=0)&(uv[...,1]<511)
        w=np.where(inside,np.maximum(z,0)**8,0).astype(np.float32)
        sampled=cv2.remap(probability,uv[...,0].astype(np.float32),uv[...,1].astype(np.float32),
                          cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
        sums+=sampled*w[...,None]
        weights+=w
    if np.any(valid&(weights==0)):
        raise ValueError('Virtual views do not cover the native FOV')
    return sums/np.maximum(weights[...,None],1e-10)


def generate_masks(scene, output, frames, *, model_cache=None):
    import torch
    from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    metadata_path = output/'model.json'
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        revision = metadata['revision']
    else:
        previous=output.parent/'masks/model.json'
        revision = json.loads(previous.read_text())['revision'] if previous.exists() else MODEL_REVISION
        metadata = {'model': MODEL, 'revision': revision, 'mask_version': MASK_VERSION,
                    'excluded_labels': sorted(EXCLUDED),
                    'inference_domain':'Five overlapping 100-degree virtual pinhole views, probability fusion in native coordinates',
                    'thresholds':{'sky':.90,'person':.55},
                    'boundary_policy':'Erode excluded sky by 3 grid pixels; dilate person by 3 grid pixels on 900x1200 grid',
                    'hardware_guard': 'Inspected polygon outlines and bottom y>0.8H; see hardware_guard source',
                    'limitation': 'Predicted sky/person and manual device outlines, not ground truth. Vegetation and potential moving vehicles remain for geometric filtering.'}
        write_json(metadata_path, metadata)
    if metadata['mask_version'] != MASK_VERSION:
        raise ValueError('Mask policy changed; use a new output directory')
    images = [dict(im) for im in scene['images'] if im['frame_index'] in frames]
    processor = model = None
    records = []
    for index, record in enumerate(images, 1):
        path = output/'valid'/(record['name']+'.png')
        label_path = output/'labels'/(record['name']+'.png')
        info_path = output/'stats'/(record['name']+'.json')
        cam = Camera.from_dict(scene['cameras'][record['camera']])
        if not all(p.exists() for p in (path,label_path,info_path)):
            if model is None:
                if not torch.cuda.is_available():
                    raise RuntimeError('Mask generation requires GPU access')
                cache = Path(model_cache) if model_cache else output.parent/'masks/model_cache'
                if not cache.exists():
                    cache = output/'model_cache'
                processor = AutoImageProcessor.from_pretrained(MODEL, revision=revision, use_fast=False,
                    cache_dir=str(cache))
                model = SegformerForSemanticSegmentation.from_pretrained(MODEL, revision=revision,
                    cache_dir=str(cache)).eval().cuda()
                write_json(output/'id2label.json', model.config.id2label)
            with Image.open(record['source']) as original:
                probability=predict_sky_person(np.asarray(original.convert('RGB')),cam,model,processor)
            kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
            sky=cv2.erode((probability[...,0]>.90).astype(np.uint8),kernel)
            person=cv2.dilate((probability[...,1]>.55).astype(np.uint8),kernel)
            sky=cv2.resize(sky,(cam.width,cam.height),interpolation=cv2.INTER_NEAREST)>0
            person=cv2.resize(person,(cam.width,cam.height),interpolation=cv2.INTER_NEAREST)>0
            device=hardware_guard(cam.width,cam.height,cam.label)
            # Reason map: 0 retained, 1 sky, 2 person, 3 device. FOV is separate.
            full=np.zeros((cam.height,cam.width),dtype=np.uint8)
            full[sky]=1;full[person]=2;full[device]=3
            valid=full==0
            for p in (path,label_path,info_path):
                p.parent.mkdir(parents=True,exist_ok=True)
            Image.fromarray(valid.astype(np.uint8)*255).save(path)
            Image.fromarray(full).save(label_path)
            fov = cam.mask(scene['fov_degrees'])>0
            write_json(info_path, {'name':record['name'],'source_sha256':file_digest(record['source']),
                'fov_pixels':int(fov.sum()),'static_valid_pixels_in_fov':int((fov&valid).sum()),
                'static_fraction_in_fov':float(valid[fov].mean()),'mask_sha256':file_digest(path)})
        record['semantic_mask'] = str(path.resolve())
        records.append(record)
        print(f'masks {index}/{len(images)} {record["camera"]}{record["frame_index"]:05d}',flush=True)
    return records, metadata


def mask_contact_sheet(scene, images, path, selected_frames):
    cameras = {k:Camera.from_dict(v) for k,v in scene['cameras'].items()}
    canvas = Image.new('RGB',(320*len(selected_frames),440*2),(24,24,24))
    draw = ImageDraw.Draw(canvas)
    for record in images:
        if record['frame_index'] not in selected_frames:
            continue
        im = Image.open(record['source']).convert('RGB').resize((320,426))
        sem = Image.open(record['semantic_mask']).resize(im.size,Image.Resampling.NEAREST)
        fov = Image.fromarray(cameras[record['camera']].mask(scene['fov_degrees'])).resize(im.size,Image.Resampling.NEAREST)
        arr = np.asarray(im).copy()
        invalid = np.asarray(fov)==0
        ignored = (np.asarray(sem)==0)&~invalid
        arr[invalid] = (arr[invalid]*.2).astype(np.uint8)
        arr[ignored] = (arr[ignored]*.45+np.array([150,20,20])*.55).astype(np.uint8)
        x=selected_frames.index(record['frame_index'])*320
        y=(0 if record['camera']=='L' else 1)*440
        canvas.paste(Image.fromarray(arr),(x,y+14))
        draw.text((x+5,y+1),f'{record["camera"]}{record["frame_index"]:05d}',fill='white')
    canvas.save(path,quality=90)


def mask_review(scene,output, *, segments=None):
    """Save side-by-side originals and explicitly colored exclusion reasons."""
    from collections import defaultdict
    segments=segments or {'normal':(70,94),'turn':(190,214)}
    output=Path(output)
    folder=output/'mask_review';folder.mkdir(exist_ok=True)
    frames=defaultdict(list)
    summaries=[]
    fovs={label:np.asarray(Image.fromarray(Camera.from_dict(cam).mask(scene['fov_degrees']))
                          .resize((600,800),Image.Resampling.NEAREST))>0
          for label,cam in scene['cameras'].items()}
    for record in scene['images']:
        stats_path=output/'masks_v2/stats'/(record['name']+'.json')
        stats=json.loads(stats_path.read_text());summaries.append(stats)
        with Image.open(record['source']) as im:original=im.convert('RGB').resize((600,800))
        labels=np.asarray(Image.open(output/'masks_v2/labels'/(record['name']+'.png')).resize((600,800),Image.Resampling.NEAREST))
        fov=fovs[record['camera']]
        arr=np.asarray(original).copy()
        arr[~fov]=(arr[~fov]*.18).astype(np.uint8)
        colors={1:(60,150,255),2:(255,65,65),3:(210,80,245)}
        for reason,color in colors.items():
            take=fov&(labels==reason)
            arr[take]=(arr[take]*.45+np.array(color)*.55).astype(np.uint8)
        canvas=Image.new('RGB',(1200,840),(24,24,24));draw=ImageDraw.Draw(canvas)
        canvas.paste(original,(0,40));canvas.paste(Image.fromarray(arr),(600,40))
        draw.text((8,8),'Original',fill='white')
        draw.text((608,8),'Blue: sky | Red: person | Purple: device | Dark: outside FOV',fill='white')
        label=f'{record["camera"]}{record["frame_index"]:05d}'
        canvas.save(folder/f'{label}.jpg',quality=92)
        frames[record['frame_index']].append((record['camera'],stats['static_fraction_in_fov'],
            f'<article><h3>{label} · FOV 内保留 {stats["static_fraction_in_fov"]:.1%}</h3>'
            f'<a href="mask_review/{label}.jpg" target="_blank"><img loading="lazy" '
            f'src="mask_review/{label}.jpg" alt="{label} 原图与掩码"></a></article>'))
    cards=[]
    for frame,items in sorted(frames.items()):
        segment=next(label for label,(low,high) in segments.items() if low<=frame<=high)
        opened=' open' if frame in [low for low,high in segments.values()] else ''
        fractions=' · '.join(f'{camera} {fraction:.1%}' for camera,fraction,_ in sorted(items))
        cards.append(f'<details data-segment="{segment}"{opened}><summary>第 {frame} 帧 · {fractions}</summary>'
                     '<div class="pair">'+''.join(card for _,_,card in sorted(items))+'</div></details>')
    values=[r['static_fraction_in_fov'] for r in summaries]
    report={'images':len(values),'retained_fraction_in_fov':{'min':float(min(values)),
             'median':float(np.median(values)),'max':float(max(values))},
             'reason':'Version 1 removed court paving misclassified as grass or swimming pool. Version 2 uses virtual views and excludes sky/person/device only.',
             'limitation':'Vegetation retained; semantic and device boundaries remain approximate, not ground truth.'}
    write_json(output/'masks_v2/summary.json',report)
    previous='<a href="masks/contact_sheet.jpg">已废弃的第一版</a> · ' if (output/'masks/contact_sheet.jpg').exists() else ''
    options=''.join(f'<option value="{label}">{low}–{high}</option>' for label,(low,high) in segments.items())
    body='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>篮球场掩码检查</title>
<style>body{font:16px system-ui;background:#171b23;color:#ddd;margin:24px}img{max-width:100%}article{min-width:0}p{line-height:1.7}a{color:#8bc5ff}summary{padding:12px;cursor:pointer}details{margin:12px 0;border:1px solid #576070}.pair{display:grid;grid-template-columns:repeat(auto-fit,minmax(520px,1fr));gap:16px;padding:12px}select{font:inherit;padding:5px}@media(max-width:600px){.pair{grid-template-columns:1fr}}</style>
<h1>掩码修正版：原图与排除区域</h1><p>蓝色：天空；红色：人物；紫色：设备；暗色：硬 FOV 外。其余区域保留。
草地、水体等类别不再直接剔除，避免将球场铺装误删；植被保留，仍需由几何筛选处理潜在运动。
语义边界和设备轮廓是近似筛选，不是真值。这里只展示进入 SfM 的原生图像域掩码，虚拟视图没有新增相机或曝光。</p>
<p>展示本轮全部 '''+str(len(scene['images']))+''' 张图像，按帧号并排排列 L/R；此排列不表示同步曝光。每幅预览左边是原图，右边是掩码，点击可放大。
圆形暗边来自初始内参下的 160° FOV 候选，并非语义识别边界；160° 尚不是实测真值。</p>
<p>'''+previous+'''<a href="masks_v2/summary.json">修正版统计</a></p>
<label>片段 <select id="segment"><option value="all">全部</option>'''+options+'''</select></label>'''+''.join(cards)+'''
<script>document.getElementById('segment').onchange=function(){document.querySelectorAll('details[data-segment]').forEach(d=>{d.hidden=this.value!=='all'&&d.dataset.segment!==this.value})}</script></html>'''
    (output/'mask_review.html').write_text(body)
    return report
