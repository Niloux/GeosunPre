"""Image-only local calibration used by the baseline and historical experiments."""
from dataclasses import asdict
import json
import os
from pathlib import Path

import numpy as np
import pycolmap

from ..frontend import LoMaFrontend
from ..local_data import CONFIG, local_pairs, split_tracks, train_edge_mask
from ..local_evaluation import evaluate_reconstruction
from ..local_reconstruction import (STAGES, assert_holdout_isolated,
                                    bounded_bundle_adjustment, cheirality_health,
                                    database_and_seed, mapper_options)
from ..scene import digest, file_digest, load_scene, write_json

SEGMENTS = {'normal_070_094': (70, 94), 'turn_190_214': (190, 214)}
BASELINE_SEGMENTS = {'normal_070_094': SEGMENTS['normal_070_094']}
LOCAL_VERSION = 1

def prepare_masks(work, *, segments=None):
    from ..static_masks import generate_masks, mask_contact_sheet, mask_review
    work = Path(work).resolve()
    out = work/'local_sfm'
    os.environ.setdefault('HF_HOME',str(out/'hf_cache'))
    base = load_scene(work)
    selected = SEGMENTS if segments is None else segments
    frames = [f for low,high in selected.values() for f in range(low,high+1)]
    records, meta = generate_masks(base,out/'masks_v2',frames)
    scene = {**base,'images':records,'semantic_masks_available':True,'segmentation':meta,
             'intrinsics_are_ground_truth':False,
             'scope':'Two continuous local SfM experiments; no vendor-pose constraints.'}
    scene.pop('fingerprint')
    scene['fingerprint'] = digest(scene)
    write_json(out/'manifest.json',scene)
    mask_contact_sheet(scene,records,out/'masks_v2/contact_sheet.jpg',
                       [f for f in (70,80,94,190,199,200,214) if f in frames])
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


def solve_baseline_calibration(work):
    """Fit the 70-94 visual calibration without transfer or manual labels."""
    out, scene, data, features = load_local(work)
    if set(data['segments']) != set(BASELINE_SEGMENTS):
        raise ValueError('Baseline calibration requires only the 70-94 segment')
    records = {row['name']: row for row in scene['images']}
    label = 'normal_070_094'
    segment = data['segments'][label]
    directory = out / label
    images = [records[name] for name in segment['image_names']]
    split = json.loads(Path(segment['split_path']).read_text())
    contract = {
        'geometry_pipeline_version': 1, 'scene': scene['fingerprint'],
        'frontend_config': data['config'],
        'features': {name: data['features'][name]['token'] for name in segment['image_names']},
        'pairs': [{key: entry[key] for key in ('a', 'b', 'token')}
                  for entry in segment['pairs']],
        'split_sha256': file_digest(segment['split_path']),
    }
    contract['fingerprint'] = digest(contract)
    contract_path = directory / 'run_contract.json'
    if contract_path.exists() and json.loads(contract_path.read_text())['fingerprint'] != contract['fingerprint']:
        raise ValueError('Local calibration inputs changed; use a fresh work directory')
    write_json(contract_path, contract)

    mapping = directory / 'mapping'
    if (mapping / 'cameras.bin').exists():
        rec = pycolmap.Reconstruction(mapping)
    else:
        db, seed = directory / 'database.db', directory / 'seed'
        if not (db.exists() and (seed / 'cameras.bin').exists()):
            db, seed = database_and_seed(directory, scene, images, segment['pairs'], features, split)
        opts = mapper_options()
        maps = pycolmap.incremental_mapping(db, Path(scene['input_root']) / 'Image',
                                            directory / 'mapper_output', options=opts,
                                            input_path=seed)
        if not maps:
            raise RuntimeError('No local calibration reconstruction')
        rec = max(maps.values(), key=lambda item: item.num_reg_images())
        if rec.num_reg_images() != len(images):
            raise RuntimeError('Local calibration did not register all selected images')
        mapping.mkdir(parents=True, exist_ok=True)
        rec.write(mapping)
        write_json(directory / 'mapping_summary.json', {
            'registered': rec.num_reg_images(), 'selected': len(images),
            'points': rec.num_points3D(), 'observations': rec.compute_num_observations(),
            'pose_priors': False, 'intrinsics_fixed_during_mapping': True,
            'options': opts.summary(),
        })
    assert_holdout_isolated(rec, split)
    if rec.num_reg_images() != len(images):
        raise RuntimeError('Cached local calibration is missing images')
    initial = {cid: camera.params.copy() for cid, camera in rec.cameras.items()}
    stages = {}
    for stage in STAGES:
        stage_dir = directory / 'stages' / stage
        model = stage_dir / 'sparse/0'
        report_path = stage_dir / 'ba.json'
        if report_path.exists() and (model / 'cameras.bin').exists():
            rec = pycolmap.Reconstruction(model)
            ba = json.loads(report_path.read_text())
            if cheirality_health(rec)['invalid_depth']:
                raise RuntimeError(f'Cached calibration has invalid depths: {stage}')
        else:
            if report_path.exists() or (model / 'cameras.bin').exists():
                raise RuntimeError(f'Incomplete calibration stage: {stage_dir}')
            print(f'Baseline local BA {stage}', flush=True)
            ba = bounded_bundle_adjustment(rec, stage, initial)
            if stage == 'principal' and ba['termination_type'].endswith('NO_CONVERGENCE'):
                archive = stage_dir.with_name('principal_guarded_400_iterations')
                archived_model = archive / 'sparse/0'
                archived_model.mkdir(parents=True, exist_ok=True)
                rec.write(archived_model)
                write_json(archive / 'ba.json', ba)
                previous = ba
                ba = bounded_bundle_adjustment(rec, stage, initial)
                ba['continuation'] = {
                    'from': str(archive), 'previous_solver': previous['solver'],
                    'previous_elapsed_seconds': previous['elapsed_seconds'],
                    'reason': 'Continue from the iteration limit without using validation errors.',
                }
            model.mkdir(parents=True, exist_ok=True)
            rec.write(model)
            rec.write_text(model)
            write_json(report_path, ba)
        assert_holdout_isolated(rec, split)
        validation = evaluate_reconstruction(rec, scene, features, split, stage_dir)
        stages[stage] = {'ba': ba, 'validation': validation,
                         'registered': rec.num_reg_images(), 'points': rec.num_points3D()}
    result = {'segment': label, 'selected_images': len(images),
              'registered_images': rec.num_reg_images(), 'stages': stages,
              'manual_annotations_used': False, 'navigation_pose_or_las_used': False}
    write_json(out / 'baseline_calibration_summary.json', result)
    return result
