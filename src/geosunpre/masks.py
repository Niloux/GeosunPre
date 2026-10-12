"""Conservative local semantic masks, kept distinct from hard fisheye validity."""
import json
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from geosunpre.camera import Camera
from geosunpre.scene import prepared_image_metadata
from geosunpre.artifacts import file_metadata
from geosunpre.artifacts import write_json, write_frozen_json
from geosunpre.parallel import ordered_map

MODEL = 'nvidia/segformer-b2-finetuned-ade-512-512'
MODEL_REVISION = 'de01bae28967510f9ddd496c60a969357195400c'
EXCLUDED = {'sky', 'person'}


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


def fusion_geometry(cam):
    """Image-independent sampling maps, owned by the current mask invocation."""
    width,height=900,1200
    yy,xx=np.mgrid[:height,:width]
    pixels=np.stack(((xx+.5)*cam.width/width,(yy+.5)*cam.height/height),axis=-1)
    valid=cam.valid(pixels)
    rays=cam.rays(pixels.reshape(-1,2)).reshape(height,width,3)
    weights=np.zeros((height,width),dtype=np.float32)
    views=[]
    for R,f,source in virtual_views(cam):
        local=rays@R
        z=local[...,2]
        uv=local[...,:2]/np.maximum(z[...,None],1e-6)*f+256-.5
        inside=valid&(z>.1)&(uv[...,0]>=0)&(uv[...,0]<511)&(uv[...,1]>=0)&(uv[...,1]<511)
        w=np.where(inside,np.maximum(z,0)**8,0).astype(np.float32)
        views.append((source,uv[...,0].astype(np.float32),uv[...,1].astype(np.float32),w))
        weights+=w
    if np.any(valid&(weights==0)):
        raise ValueError('Virtual views do not cover the native FOV')
    return views,np.maximum(weights[...,None],1e-10)


def predict_sky_person(prepared_views,geometry,model,processor):
    import torch
    views,weights=geometry
    sky_id=next(int(k) for k,v in model.config.id2label.items() if v=='sky')
    person_id=next(int(k) for k,v in model.config.id2label.items() if v=='person')
    sums=np.zeros((*weights.shape[:2],2),dtype=np.float32)
    # Keep single-view inference order and arithmetic identical to the reference.
    for view,(_,map_x,map_y,w) in zip(prepared_views,views,strict=True):
        with torch.inference_mode():
            inp={k:v.cuda() for k,v in processor(images=Image.fromarray(view),return_tensors='pt').items()}
            probability=model(**inp).logits.softmax(1)[:,[sky_id,person_id]]
            probability=torch.nn.functional.interpolate(probability,size=(512,512),mode='bilinear',
                                                        align_corners=False)[0].permute(1,2,0).cpu().numpy()
        sampled=cv2.remap(probability,map_x,map_y,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
        sums+=sampled*w[...,None]
    return sums/weights


def mask_cached(record, output):
    """Reuse a completed mask when its source and output metadata match."""
    source_file = prepared_image_metadata(record)
    output = Path(output)
    stats = output / 'stats' / (record['name'] + '.json')
    if not stats.is_file():
        return False
    info = json.loads(stats.read_text(encoding='utf-8'))
    if info.get('source_file') != source_file:
        raise ValueError(f'Stale mask source: {record["name"]}')
    for folder, key in (('valid', 'mask_file'), ('labels', 'labels_file')):
        path = output / folder / (record['name'] + '.png')
        if not path.is_file() or file_metadata(path) != info.get(key):
            raise ValueError(f'Committed mask artifact changed: {record["name"]}: {folder}')
    return True


def _load_segmentation(model_cache):
    import torch
    from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

    if not torch.cuda.is_available():
        raise RuntimeError('Mask generation requires GPU access')
    cache = Path(model_cache or os.environ.get('HF_HUB_CACHE', '.cache/huggingface')).expanduser().resolve()
    processor = AutoImageProcessor.from_pretrained(MODEL, revision=MODEL_REVISION, use_fast=False, cache_dir=str(cache))
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL, revision=MODEL_REVISION, cache_dir=str(cache)).eval().cuda()
    return processor, model


def _review_path(output, record):
    return Path(output) / 'mask_review' / f'{record["camera"]}{record["frame_index"]:05d}.jpg'


def _write_review(path, original, labels, fov):
    """Render from the already decoded thumbnail, outside the numeric mask path."""
    arr = np.asarray(original).copy()
    arr[~fov] = (arr[~fov] * .18).astype(np.uint8)
    for reason, color in {1: (60,150,255), 2: (255,65,65)}.items():
        take = fov & (labels == reason)
        arr[take] = (arr[take] * .45 + np.array(color) * .55).astype(np.uint8)
    canvas = Image.new('RGB', (1200,840), (24,24,24))
    draw = ImageDraw.Draw(canvas)
    canvas.paste(original, (0,40)); canvas.paste(Image.fromarray(arr), (600,40))
    draw.text((8,8), 'Original', fill='white')
    draw.text((608,8), 'Blue: sky | Red: person | Dark: outside FOV', fill='white')
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=92)


def generate_masks(scene, output, frames, *, model_cache=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    revision = MODEL_REVISION
    metadata = {
        'model': MODEL, 'revision': revision,
        'excluded_labels': sorted(EXCLUDED),
        'mask_values': {'static': 255, 'sky': 128, 'person': 0},
        'inference_domain': 'Five overlapping 100-degree virtual pinhole views, probability fusion in native coordinates',
        'thresholds': {'sky': .90, 'person': .55},
        'boundary_policy': 'Erode excluded sky by 3 grid pixels; dilate person by 3 grid pixels on 900x1200 grid',
        'cameras': scene['cameras'], 'fov_degrees': scene['fov_degrees'],
        'limitation': 'Approximate semantic boundaries; vegetation and moving vehicles require geometric filtering.',
    }
    write_frozen_json(output / 'model.json', metadata)
    images = [dict(im) for im in scene['images'] if im['frame_index'] in frames]
    processor = model = None
    records = []
    # Check every committed packet before starting workers or loading a model.
    jobs = [(record, Camera.from_dict(scene['cameras'][record['camera']]), mask_cached(record, output))
            for record in images]
    pending_cameras = {cam for _, cam, cached in jobs if not cached}
    camera_masks = {cam: cam.mask(scene['fov_degrees']) > 0
                    for cam in pending_cameras}
    review_domains = {cam: np.asarray(Image.fromarray(fov.astype(np.uint8)*255)
                         .resize((600,800), Image.Resampling.NEAREST)) > 0
                      for cam, fov in camera_masks.items()}
    # Share two camera maps within this invocation; none survive into SfM.
    geometry = {cam: fusion_geometry(cam) for cam in pending_cameras}

    def prepare_image(job):
        record, cam, cached = job
        if cached:
            return record, cam, None, None
        with Image.open(record['source']) as source:
            original = source.convert('RGB')
        rgb = np.asarray(original)
        views = [cv2.remap(rgb, source[...,0], source[...,1], cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT) for source, _, _, _ in geometry[cam][0]]
        thumbnail = original.resize((600,800))
        prepared_image_metadata(record)
        return record, cam, views, thumbnail

    def infer():
        nonlocal processor,model
        for record, cam, views, thumbnail in ordered_map(prepare_image, jobs, workers=2):
            if views is None:
                yield record,cam,None,None,None
                continue
            if model is None:
                processor, model = _load_segmentation(model_cache)
                write_json(output/'id2label.json',model.config.id2label)
            probability = predict_sky_person(views, geometry[cam], model, processor)
            yield record,cam,probability,camera_masks[cam],thumbnail

    def finish_mask(item):
        record,cam,probability,masks,thumbnail=item
        path=output/'valid'/(record['name']+'.png')
        label_path=output/'labels'/(record['name']+'.png')
        info_path=output/'stats'/(record['name']+'.json')
        if probability is not None:
            kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
            sky=cv2.erode((probability[...,0]>.90).astype(np.uint8),kernel)
            person=cv2.dilate((probability[...,1]>.55).astype(np.uint8),kernel)
            sky=cv2.resize(sky,(cam.width,cam.height),interpolation=cv2.INTER_NEAREST)>0
            person=cv2.resize(person,(cam.width,cam.height),interpolation=cv2.INTER_NEAREST)>0
            fov=masks
            # Reason map: 0 retained, 1 sky, 2 person. FOV is separate.
            full=np.zeros((cam.height,cam.width),dtype=np.uint8)
            full[sky]=1;full[person]=2
            valid=full==0
            semantic=np.array([255,128,0],dtype=np.uint8)[full]
            for p in (path,label_path,info_path):
                p.parent.mkdir(parents=True,exist_ok=True)
            for target, array in ((path, semantic), (label_path, full)):
                temporary = target.with_suffix('.tmp.png')
                Image.fromarray(array).save(temporary)
                temporary.replace(target)
            write_json(info_path, {'name':record['name'],'source_file':prepared_image_metadata(record),
                'fov_pixels':int(fov.sum()),'static_valid_pixels_in_fov':int((fov&valid).sum()),
                'static_fraction_in_fov':float(valid[fov].mean()),'mask_file':file_metadata(path),
                'labels_file':file_metadata(label_path)})
            labels = np.asarray(Image.fromarray(full).resize((600,800), Image.Resampling.NEAREST))
            _write_review(_review_path(output.parent, record), thumbnail, labels, review_domains[cam])
        record['semantic_mask']=str(path.resolve())
        return record

    # Decode/remap and postprocess/write have separate bounded CPU queues.
    # GPU inference stays single-view, ordered, and on the calling thread.
    for index,record in enumerate(ordered_map(finish_mask,infer(),workers=2),1):
        records.append(record)
        if index == 1 or index % 40 == 0 or index == len(images):
            print(f'masks {index}/{len(images)}',flush=True)
    return records, metadata


def mask_contact_sheet(scene, images, path, selected_frames):
    cameras = {k:Camera.from_dict(v) for k,v in scene['cameras'].items()}
    fovs = {label: np.asarray(Image.fromarray(cam.mask(scene['fov_degrees']))
                  .resize((320,426), Image.Resampling.NEAREST)) == 0
            for label, cam in cameras.items()}
    canvas = Image.new('RGB',(320*len(selected_frames),440*2),(24,24,24))
    draw = ImageDraw.Draw(canvas)
    for record in images:
        if record['frame_index'] not in selected_frames:
            continue
        im = Image.open(record['source']).convert('RGB').resize((320,426))
        sem = np.asarray(Image.open(record['semantic_mask']).resize(im.size,Image.Resampling.NEAREST))
        arr = np.asarray(im).copy()
        invalid = fovs[record['camera']]
        arr[invalid] = (arr[invalid]*.2).astype(np.uint8)
        for value,color in {128:(60,150,255),0:(150,20,20)}.items():
            ignored = (sem==value)&~invalid
            arr[ignored] = (arr[ignored]*.45+np.array(color)*.55).astype(np.uint8)
        x=selected_frames.index(record['frame_index'])*320
        y=(0 if record['camera']=='L' else 1)*440
        canvas.paste(Image.fromarray(arr),(x,y+14))
        draw.text((x+5,y+1),f'{record["camera"]}{record["frame_index"]:05d}',fill='white')
    canvas.save(path,quality=90)


def mask_review(scene,output, *, segments=None):
    """Save side-by-side originals and explicitly colored exclusion reasons."""
    from collections import defaultdict
    indices = [row['frame_index'] for row in scene['images']]
    segments = segments or {'all': (min(indices), max(indices))}
    output=Path(output)
    folder=output/'mask_review';folder.mkdir(exist_ok=True)
    frames=defaultdict(list)
    summaries=[]
    pending = {r['camera'] for r in scene['images'] if not _review_path(output, r).is_file()}
    fovs={label:np.asarray(Image.fromarray(Camera.from_dict(cam).mask(scene['fov_degrees']))
                          .resize((600,800),Image.Resampling.NEAREST))>0
          for label,cam in scene['cameras'].items() if label in pending}
    def render_review(record):
        stats_path=output/'masks/stats'/(record['name']+'.json')
        stats=json.loads(stats_path.read_text(encoding='utf-8'))
        path = _review_path(output, record)
        if not path.is_file():
            with Image.open(record['source']) as im:
                original = im.convert('RGB').resize((600,800))
            with Image.open(output/'masks/labels'/(record['name']+'.png')) as im:
                labels = np.asarray(im.resize((600,800),Image.Resampling.NEAREST))
            _write_review(path, original, labels, fovs[record['camera']])
        label=f'{record["camera"]}{record["frame_index"]:05d}'
        card=(record['camera'],stats['static_fraction_in_fov'],
            f'<article><h3>{label} · FOV 内保留 {stats["static_fraction_in_fov"]:.1%}</h3>'
            f'<a href="mask_review/{label}.jpg" target="_blank"><img loading="lazy" '
            f'src="mask_review/{label}.jpg" alt="{label} 原图与掩码"></a></article>')
        return record['frame_index'],stats,card
    for frame,stats,card in ordered_map(render_review,scene['images']):
        summaries.append(stats)
        frames[frame].append(card)
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
             'policy':'Virtual-view semantic inference; exclude sky and person.',
             'limitation':'Vegetation retained; semantic boundaries remain approximate, not ground truth.'}
    write_json(output/'masks/summary.json',report)
    options=''.join(f'<option value="{label}">{low}–{high}</option>' for label,(low,high) in segments.items())
    body='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>掩码检查</title>
<style>body{font:16px system-ui;background:#171b23;color:#ddd;margin:24px}img{max-width:100%}article{min-width:0}p{line-height:1.7}a{color:#8bc5ff}summary{padding:12px;cursor:pointer}details{margin:12px 0;border:1px solid #576070}.pair{display:grid;grid-template-columns:repeat(auto-fit,minmax(520px,1fr));gap:16px;padding:12px}select{font:inherit;padding:5px}@media(max-width:600px){.pair{grid-template-columns:1fr}}</style>
<h1>原图与排除区域</h1><p>蓝色：天空；红色：人物；暗色：硬 FOV 外。其余区域保留。
植被等类别保留，潜在运动区域继续由几何筛选处理。
语义边界是近似筛选，不是真值。这里只展示进入 SfM 的原生图像域掩码，虚拟视图没有新增相机或曝光。</p>
<p>展示本轮全部 '''+str(len(scene['images']))+''' 张图像，按帧号并排排列 L/R；此排列不表示同步曝光。每幅预览左边是原图，右边是掩码，点击可放大。
圆形暗边来自初始内参下的 160° FOV 候选，并非语义识别边界；160° 尚不是实测真值。</p>
<p><a href="masks/summary.json">掩码统计</a>。masks/valid 中静态区域为 255、天空为 128、人物为 0；最终导出的 masks 还会将 FOV 外设为 0。</p>
<label>片段 <select id="segment"><option value="all">全部</option>'''+options+'''</select></label>'''+''.join(cards)+'''
<script>document.getElementById('segment').onchange=function(){document.querySelectorAll('details[data-segment]').forEach(d=>{d.hidden=this.value!=='all'&&d.dataset.segment!==this.value})}</script></html>'''
    (output/'mask_review.html').write_text(body, encoding='utf-8')
    return report
