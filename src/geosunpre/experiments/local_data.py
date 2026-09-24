"""Local and transfer experiment data preparation."""
from dataclasses import asdict
import json
import os
from pathlib import Path

import numpy as np

from ..frontend import LoMaFrontend
from ..local_data import CONFIG, local_pairs, split_tracks, train_edge_mask
from ..scene import digest, file_digest, load_scene, write_json

SEGMENTS = {'normal_070_094': (70, 94), 'turn_190_214': (190, 214)}
LOCAL_VERSION = 1

def prepare_masks(work):
    from ..static_masks import generate_masks, mask_contact_sheet, mask_review
    work = Path(work).resolve()
    out = work/'local_sfm'
    os.environ.setdefault('HF_HOME',str(out/'hf_cache'))
    base = load_scene(work)
    frames = [f for low,high in SEGMENTS.values() for f in range(low,high+1)]
    records, meta = generate_masks(base,out/'masks_v2',frames)
    scene = {**base,'images':records,'semantic_masks_available':True,'segmentation':meta,
             'intrinsics_are_ground_truth':False,
             'scope':'Two continuous local SfM experiments; no vendor-pose constraints.'}
    scene.pop('fingerprint')
    scene['fingerprint'] = digest(scene)
    write_json(out/'manifest.json',scene)
    mask_contact_sheet(scene,records,out/'masks_v2/contact_sheet.jpg',[70,80,94,190,199,200,214])
    mask_review(scene,out)
    return out

def build_frontend(work, *, output_name='local_sfm', segments=None, config=CONFIG):
    work=Path(work).resolve()
    out=work/output_name
    scene=load_scene(out)
    frontend=LoMaFrontend(work,scene,config)
    features={}
    catalog={}
    for i,im in enumerate(scene['images'],1):
        f=frontend.extract(im)
        features[im['name']]=f
        catalog[im['name']]={'token':str(f['token']),'path':str(frontend.cache/f'{str(f["token"])}.npz'),
                             'keypoints':len(f['keypoints'])}
        print(f'local features {i}/{len(scene["images"])} {im["name"]}',flush=True)
    selected_segments=SEGMENTS if segments is None else segments
    segments={}
    for label,(low,high) in selected_segments.items():
        images=[im for im in scene['images'] if low<=im['frame_index']<=high]
        pairs=local_pairs(images)
        entries=[]
        pair_data=[]
        for i,(a,b) in enumerate(pairs,1):
            raw=frontend.match(features[a['name']],features[b['name']])
            entries.append({'a':a['name'],'b':b['name'],'token':str(raw['token']),
                            'path':str(work/'matches'/f'{str(raw["token"])}.npz')})
            pair_data.append((a['name'],b['name'],raw['matches'],raw['scores']))
            if i%20==0 or i==len(pairs):
                print(f'{label} raw pairs {i}/{len(pairs)}',flush=True)
        split,_=split_tracks(pair_data,images)
        directory=out/label
        write_json(directory/'split.json',split)
        segments[label]={'image_names':[im['name'] for im in images],'pairs':entries,
                         'split_path':str(directory/'split.json')}
    manifest={'version':LOCAL_VERSION,'scene_fingerprint':scene['fingerprint'],
              'config':asdict(config),'features':catalog,'segments':segments,
              'cache':frontend.cache_stats}
    manifest['fingerprint']=digest(manifest)
    write_json(out/'frontend.json',manifest)
    return manifest

def load_local(work, *, output_name='local_sfm'):
    out=Path(work).resolve()/output_name
    scene=load_scene(out)
    data=json.loads((out/'frontend.json').read_text())
    if data['scene_fingerprint']!=scene['fingerprint']:
        raise ValueError('Local scene changed; regenerate frontend')
    features={}
    records={im['name']:im for im in scene['images']}
    for name,entry in data['features'].items():
        record=records[name]
        with np.load(entry['path'],allow_pickle=False) as f:
            meta=json.loads(str(f['metadata']))
            identity={'image_sha256':file_digest(record['source']),
                      'mask_sha256':file_digest(record['semantic_mask']) if record['semantic_mask'] else None,
                      'camera':scene['cameras'][record['camera']],
                      'fov':scene['fov_degrees'],'config':data['config']}
            expected=digest(identity)
            if expected!=entry['token'] or str(f['token'])!=expected or meta['name']!=name:
                raise ValueError(f'{name}: source, mask, or frontend identity changed; regenerate frontend before solving')
            features[name]={'keypoints':f['keypoints'],'scores':f['scores']}
    return out,scene,data,features
