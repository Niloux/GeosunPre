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
from .colmap_export import MODEL_FILES
from .holdout import ISOLATION_SCOPE, load_frozen_holdout, restrict_split, write_frozen_json

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


def prepare_baseline_calibration(work):
    """Reuse global feature IDs and holdout exclusions for the calibration segment."""
    from .global_data import load_global

    root, global_scene, global_data, features = load_global(work)
    _, split, _ = load_frozen_holdout(work, global_data)
    out = Path(work).resolve() / 'local_sfm'
    label = 'normal_070_094'
    low, high = BASELINE_SEGMENTS[label]
    images = [row for row in global_scene['images'] if low <= row['frame_index'] <= high]
    if not images:
        raise ValueError('No images in the baseline calibration segment')
    names = {row['name'] for row in images}
    scene = {**global_scene, 'images': images,
             'scope': 'Local calibration from frozen global features and holdout.'}
    scene.pop('fingerprint')
    scene['fingerprint'] = digest(scene)
    local_split = restrict_split(split, images)
    data = {'version': 2, 'scene_fingerprint': scene['fingerprint'],
            'config': global_data['config'],
            'features': {name: entry for name, entry in global_data['features'].items() if name in names},
            'global_holdout_sha256': file_digest(root / 'holdout_protocol.json'),
            'segments': {label: {'image_names': [row['name'] for row in images],
                                'pairs': [entry for entry in global_data['pairs']
                                          if entry['a'] in names and entry['b'] in names],
                                'split_path': str(out / label / 'split.json')}}}
    data['fingerprint'] = digest(data)
    directory = out / label
    if not (directory / 'run_contract.json').exists() and any(
            (directory / name).exists() for name in ('database.db', 'seed', 'mapping', 'mapper_output', 'stages')):
        raise ValueError('Uncontracted local geometry exists; use a fresh work directory')
    write_frozen_json(out / 'manifest.json', scene)
    write_frozen_json(directory / 'split.json', local_split)
    write_frozen_json(out / 'frontend.json', data)
    return out, scene, data, {name: features[name] for name in names}


def _calibration_artifacts():
    label = 'normal_070_094'
    files = ['manifest.json', 'frontend.json', f'{label}/split.json', f'{label}/run_contract.json']
    files += [f'{label}/mapping/{name}' for name in MODEL_FILES]
    for stage in STAGES:
        files += [f'{label}/stages/{stage}/sparse/0/{name}' for name in MODEL_FILES]
        files.append(f'{label}/stages/{stage}/ba.json')
    return files


def verify_baseline_calibration(work, global_split):
    """Require a completed calibration tied to this pre-calibration holdout."""
    work = Path(work).resolve()
    out = work / 'local_sfm'
    path = out / 'calibration_identity.json'
    if not path.is_file():
        raise ValueError('Missing isolated calibration identity; run local-calibration in a fresh work directory')
    identity = json.loads(path.read_text())
    holdout_hash = file_digest(work / 'global_sfm/holdout_protocol.json')
    if (identity.get('version') != 1 or identity.get('global_holdout_sha256') != holdout_hash
            or set(identity.get('files', {})) != set(_calibration_artifacts())):
        raise ValueError('Calibration belongs to another holdout protocol')
    for relative, expected in identity['files'].items():
        if file_digest(out / relative) != expected:
            raise ValueError(f'Calibration artifact changed: {relative}')
    contract = json.loads((out / 'normal_070_094/run_contract.json').read_text())
    if contract.get('geometry_pipeline_version') != 2 or contract.get('global_holdout_sha256') != holdout_hash:
        raise ValueError('Calibration has no pre-geometry holdout contract')
    model = pycolmap.Reconstruction(out / 'normal_070_094/stages/principal/sparse/0')
    assert_holdout_isolated(model, global_split)
    return model


def solve_baseline_calibration(work):
    """Fit the 70-94 visual calibration with the global holdout already excluded."""
    out, scene, data, features = prepare_baseline_calibration(work)
    if (out / 'calibration_identity.json').exists():
        verify_baseline_calibration(work, json.loads((Path(work) / 'global_sfm/split.json').read_text()))
    if set(data['segments']) != set(BASELINE_SEGMENTS):
        raise ValueError('Baseline calibration requires only the 70-94 segment')
    records = {row['name']: row for row in scene['images']}
    label = 'normal_070_094'
    segment = data['segments'][label]
    directory = out / label
    images = [records[name] for name in segment['image_names']]
    split = json.loads(Path(segment['split_path']).read_text())
    contract = {
        'geometry_pipeline_version': 2, 'scene': scene['fingerprint'],
        'global_holdout_sha256': data['global_holdout_sha256'],
        'frontend_config': data['config'],
        'features': {name: data['features'][name]['token'] for name in segment['image_names']},
        'pairs': [{key: entry[key] for key in ('a', 'b', 'token')}
                  for entry in segment['pairs']],
        'split_sha256': file_digest(segment['split_path']),
    }
    contract['fingerprint'] = digest(contract)
    contract_path = directory / 'run_contract.json'
    write_frozen_json(contract_path, contract)

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
              'holdout_isolation': ISOLATION_SCOPE,
              'manual_annotations_used': False, 'navigation_pose_or_las_used': False}
    write_json(out / 'baseline_calibration_summary.json', result)
    write_frozen_json(out / 'calibration_identity.json', {
        'version': 1, 'global_holdout_sha256': data['global_holdout_sha256'],
        'files': {name: file_digest(out / name) for name in _calibration_artifacts()},
    })
    return result
