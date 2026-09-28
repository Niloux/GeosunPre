"""Visual calibration using a subset of the globally frozen observations."""
import json
from pathlib import Path

import pycolmap

from geosunpre.config import CONFIG_FILE, load_config
from geosunpre.sfm.evaluation import evaluate_reconstruction
from geosunpre.sfm.reconstruction import STAGES
from geosunpre.sfm.reconstruction import assert_holdout_isolated
from geosunpre.sfm.reconstruction import bounded_bundle_adjustment
from geosunpre.sfm.reconstruction import cheirality_health
from geosunpre.sfm.reconstruction import database_and_seed
from geosunpre.sfm.reconstruction import mapper_options
from geosunpre.artifacts import digest
from geosunpre.artifacts import file_digest
from geosunpre.artifacts import write_json
from geosunpre.colmap import MODEL_FILES
from geosunpre.pipeline.holdout import ISOLATION_SCOPE
from geosunpre.pipeline.holdout import load_frozen_holdout
from geosunpre.pipeline.holdout import restrict_split
from geosunpre.artifacts import write_frozen_json


def prepare_calibration(work):
    """Reuse global feature IDs and holdout exclusions for the calibration segment."""
    from geosunpre.pipeline.data import load_frontend

    root, global_scene, global_data, features = load_frontend(work)
    _, split, _ = load_frozen_holdout(work, global_data)
    out = Path(work).resolve() / 'calibration'
    images = load_config(work).calibration_images(global_scene)
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
            'pipeline_config_sha256': file_digest(Path(work) / CONFIG_FILE),
            'image_names': [row['name'] for row in images],
            'pairs': [entry for entry in global_data['pairs']
                      if entry['a'] in names and entry['b'] in names],
            'split_path': str(out / 'split.json')}
    data['fingerprint'] = digest(data)
    directory = out
    if not (directory / 'run_contract.json').exists() and any(
            (directory / name).exists() for name in ('database.db', 'seed', 'mapping', 'mapper_output', 'stages')):
        raise ValueError('Uncontracted local geometry exists; use a fresh work directory')
    write_frozen_json(out / 'manifest.json', scene)
    write_frozen_json(directory / 'split.json', local_split)
    write_frozen_json(out / 'frontend.json', data)
    return out, scene, data, {name: features[name] for name in names}


def _calibration_artifacts():
    files = ['manifest.json', 'frontend.json', 'split.json', 'run_contract.json']
    files += [f'mapping/{name}' for name in MODEL_FILES]
    for stage in STAGES:
        files += [f'stages/{stage}/sparse/0/{name}' for name in MODEL_FILES]
        files.append(f'stages/{stage}/ba.json')
    return files


def verify_calibration(work, global_split):
    """Require a completed calibration tied to this pre-calibration holdout."""
    work = Path(work).resolve()
    out = work / 'calibration'
    path = out / 'calibration_identity.json'
    if not path.is_file():
        raise ValueError('Missing calibration identity; run the calibration stage first')
    identity = json.loads(path.read_text())
    holdout_hash = file_digest(work / 'sfm/holdout_protocol.json')
    if (identity.get('version') != 1 or identity.get('global_holdout_sha256') != holdout_hash
            or identity.get('pipeline_config_sha256') != file_digest(work / CONFIG_FILE)
            or set(identity.get('files', {})) != set(_calibration_artifacts())):
        raise ValueError('Calibration belongs to another holdout protocol')
    for relative, expected in identity['files'].items():
        if file_digest(out / relative) != expected:
            raise ValueError(f'Calibration artifact changed: {relative}')
    contract = json.loads((out / 'run_contract.json').read_text())
    if contract.get('geometry_pipeline_version') != 2 or contract.get('global_holdout_sha256') != holdout_hash:
        raise ValueError('Calibration has no pre-geometry holdout contract')
    model = pycolmap.Reconstruction(out / 'stages/principal/sparse/0')
    assert_holdout_isolated(model, global_split)
    return model


def calibrate(work):
    """Fit the configured calibration subset with global holdout observations excluded."""
    out, scene, data, features = prepare_calibration(work)
    if (out / 'calibration_identity.json').exists():
        verify_calibration(work, json.loads((Path(work) / 'sfm/split.json').read_text()))
    records = {row['name']: row for row in scene['images']}
    directory = out
    images = [records[name] for name in data['image_names']]
    split = json.loads(Path(data['split_path']).read_text())
    contract = {
        'geometry_pipeline_version': 2, 'scene': scene['fingerprint'],
        'global_holdout_sha256': data['global_holdout_sha256'],
        'pipeline_config_sha256': data['pipeline_config_sha256'],
        'frontend_config': data['config'],
        'features': {name: data['features'][name]['token'] for name in data['image_names']},
        'pairs': [{key: entry[key] for key in ('a', 'b', 'token')}
                  for entry in data['pairs']],
        'split_sha256': file_digest(data['split_path']),
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
            if db.exists():
                raise RuntimeError(f'Incomplete calibration database; use a fresh work directory: {db}')
            db, seed = database_and_seed(directory, scene, images, data['pairs'], features, split)
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
            print(f'Calibration BA {stage}', flush=True)
            ba = bounded_bundle_adjustment(rec, stage, initial, show_progress=True)
            if stage == 'principal' and ba['termination_type'].endswith('NO_CONVERGENCE'):
                archive = stage_dir.with_name('principal_guarded_400_iterations')
                archived_model = archive / 'sparse/0'
                archived_model.mkdir(parents=True, exist_ok=True)
                rec.write(archived_model)
                write_json(archive / 'ba.json', ba)
                previous = ba
                ba = bounded_bundle_adjustment(rec, stage, initial, show_progress=True)
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
    result = {'selected_images': len(images),
              'frame_indices': sorted({row['frame_index'] for row in images}),
              'registered_images': rec.num_reg_images(), 'stages': stages,
              'holdout_isolation': ISOLATION_SCOPE,
              'manual_annotations_used': False, 'navigation_pose_or_las_used': False}
    write_json(out / 'summary.json', result)
    write_frozen_json(out / 'calibration_identity.json', {
        'version': 1, 'global_holdout_sha256': data['global_holdout_sha256'],
        'pipeline_config_sha256': data['pipeline_config_sha256'],
        'files': {name: file_digest(out / name) for name in _calibration_artifacts()},
    })
    return result
