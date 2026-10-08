"""Visual calibration using a subset of the globally frozen observations."""
import json
from pathlib import Path

import pycolmap

from geosunpre.config import CONFIG_FILE, load_config
from geosunpre.sfm.reconstruction import STAGES
from geosunpre.sfm.reconstruction import bounded_bundle_adjustment
from geosunpre.sfm.reconstruction import cheirality_health
from geosunpre.sfm.reconstruction import database_and_seed
from geosunpre.sfm.reconstruction import mapper_options
from geosunpre.artifacts import digest
from geosunpre.artifacts import file_digest
from geosunpre.artifacts import write_json, archive_interrupted, staged_directory
from geosunpre.colmap import MODEL_FILES, model_hashes
from geosunpre.pipeline.observations import SCOPE, load_observations, protocol_identity
from geosunpre.artifacts import write_frozen_json


def prepare_calibration(work):
    """Reuse global feature IDs and source identities for the calibration segment."""
    from geosunpre.pipeline.data import load_frontend

    _, global_scene, global_data, features = load_frontend(work)
    load_observations(work, global_data)
    out = Path(work).resolve() / 'calibration'
    images = load_config(work).calibration_images(global_scene)
    names = {row['name'] for row in images}
    scene = {**global_scene, 'images': images,
             'scope': 'Local calibration from all eligible frozen global observations.'}
    scene.pop('fingerprint')
    scene['fingerprint'] = digest(scene)
    data = {'version': 2, 'scene_fingerprint': scene['fingerprint'],
            'config': global_data['config'],
            'features': {name: entry for name, entry in global_data['features'].items() if name in names},
            **protocol_identity(work),
            'pipeline_config_sha256': file_digest(Path(work) / CONFIG_FILE),
            'image_names': [row['name'] for row in images],
            'pairs': [entry for entry in global_data['pairs']
                      if entry['a'] in names and entry['b'] in names],
            'observation_scope': SCOPE}
    data['fingerprint'] = digest(data)
    if not (out / 'run_contract.json').exists() and any(
            (out / name).exists() for name in ('database.db', 'seed', 'mapping', 'mapper_output', 'stages')):
        raise ValueError('Uncontracted local geometry exists; use a fresh work directory')
    write_frozen_json(out / 'manifest.json', scene)
    write_frozen_json(out / 'frontend.json', data)
    return out, scene, data, {name: features[name] for name in names}


def _calibration_artifacts():
    files = ['manifest.json', 'frontend.json', 'run_contract.json']
    files += [f'mapping/{name}' for name in MODEL_FILES]
    files.append('mapping/identity.json')
    for stage in STAGES:
        files += [f'stages/{stage}/sparse/0/{name}' for name in MODEL_FILES]
        files.append(f'stages/{stage}/ba.json')
    return files


def verify_calibration(work):
    """Require a completed calibration tied to the frozen observation sources."""
    work = Path(work).resolve()
    out = work / 'calibration'
    path = out / 'calibration_identity.json'
    if not path.is_file():
        raise ValueError('Missing calibration identity; run the calibration stage first')
    identity = json.loads(path.read_text())
    observation_id = protocol_identity(work)
    if (identity.get('version') != 1 or any(identity.get(k) != v for k, v in observation_id.items())
            or identity.get('pipeline_config_sha256') != file_digest(work / CONFIG_FILE)
            or set(identity.get('files', {})) != set(_calibration_artifacts())):
        raise ValueError('Calibration belongs to another observation protocol')
    for relative, expected in identity['files'].items():
        if file_digest(out / relative) != expected:
            raise ValueError(f'Calibration artifact changed: {relative}')
    contract = json.loads((out / 'run_contract.json').read_text())
    if contract.get('geometry_pipeline_version') != 2 or any(contract.get(k) != v for k, v in observation_id.items()):
        raise ValueError('Calibration has no pre-geometry observation contract')
    model = pycolmap.Reconstruction(out / 'stages/principal/sparse/0')
    return model


def calibrate(work):
    """Fit the configured calibration subset from frozen global observations."""
    out, scene, data, features = prepare_calibration(work)
    if (out / 'calibration_identity.json').exists():
        verify_calibration(work)
    records = {row['name']: row for row in scene['images']}
    directory = out
    images = [records[name] for name in data['image_names']]
    contract = {
        'geometry_pipeline_version': 2, 'scene': scene['fingerprint'],
        **protocol_identity(work),
        'pipeline_config_sha256': data['pipeline_config_sha256'],
        'frontend_config': data['config'],
        'features': {name: data['features'][name]['token'] for name in data['image_names']},
        'pairs': [{key: entry[key] for key in ('a', 'b', 'token')}
                  for entry in data['pairs']],
        'observation_scope': SCOPE,
    }
    contract['fingerprint'] = digest(contract)
    contract_path = directory / 'run_contract.json'
    write_frozen_json(contract_path, contract)

    mapping = directory / 'mapping'
    if (mapping / 'identity.json').is_file():
        identity = json.loads((mapping / 'identity.json').read_text())
        if (identity.get('protocol_fingerprint') != contract['fingerprint']
                or identity.get('model_sha256') != model_hashes(mapping)):
            raise ValueError('Calibration mapping identity changed')
        rec = pycolmap.Reconstruction(mapping)
    else:
        db, seed = directory / 'database.db', directory / 'seed'
        if not (db.exists() and all((seed / name).is_file() for name in MODEL_FILES)):
            archive_interrupted(db)
            archive_interrupted(seed)
            db, seed = database_and_seed(directory, scene, images, data['pairs'], features)
        opts = mapper_options()
        maps = pycolmap.incremental_mapping(db, Path(scene['input_root']) / 'Image',
                                            directory / 'mapper_output', options=opts,
                                            input_path=seed)
        if not maps:
            raise RuntimeError('No local calibration reconstruction')
        rec = max(maps.values(), key=lambda item: item.num_reg_images())
        if rec.num_reg_images() != len(images):
            raise RuntimeError('Local calibration did not register all selected images')
        with staged_directory(mapping) as staged:
            rec.write(staged)
            if not pycolmap.Reconstruction(staged).is_valid():
                raise RuntimeError('Incomplete calibration mapping output')
            write_json(staged / 'identity.json', {'protocol_fingerprint': contract['fingerprint'],
                                                 'model_sha256': model_hashes(staged)})
        write_json(directory / 'mapping_summary.json', {
            'registered': rec.num_reg_images(), 'selected': len(images),
            'points': rec.num_points3D(), 'observations': rec.compute_num_observations(),
            'pose_priors': False, 'intrinsics_fixed_during_mapping': True,
            'options': opts.summary(),
        })
    if rec.num_reg_images() != len(images):
        raise RuntimeError('Cached local calibration is missing images')
    initial = {cid: camera.params.copy() for cid, camera in rec.cameras.items()}
    stages = {}
    for index, stage in enumerate(STAGES):
        stage_dir = directory / 'stages' / stage
        model = stage_dir / 'sparse/0'
        report_path = stage_dir / 'ba.json'
        previous_model = mapping if index == 0 else directory / 'stages' / STAGES[index - 1] / 'sparse/0'
        input_hashes = model_hashes(previous_model)
        if report_path.is_file() and not all((model / name).is_file() for name in MODEL_FILES):
            if 'model_sha256' in json.loads(report_path.read_text()):
                raise ValueError(f'Committed calibration model is incomplete: {stage}')
        if report_path.is_file() and all((model / name).is_file() for name in MODEL_FILES):
            rec = pycolmap.Reconstruction(model)
            ba = json.loads(report_path.read_text())
            if (ba.get('input_model_sha256') != input_hashes
                    or ba.get('model_sha256') != model_hashes(model)):
                raise ValueError(f'Calibration stage identity changed: {stage}')
            if cheirality_health(rec)['invalid_depth']:
                raise RuntimeError(f'Cached calibration has invalid depths: {stage}')
        else:
            print(f'Calibration BA {stage}', flush=True)
            ba = bounded_bundle_adjustment(rec, stage, initial, show_progress=False)
            if stage == 'principal' and ba['termination_type'].endswith('NO_CONVERGENCE'):
                archive = stage_dir.with_name('principal_guarded_400_iterations')
                archived_model = archive / 'sparse/0'
                archived_model.mkdir(parents=True, exist_ok=True)
                rec.write(archived_model)
                write_json(archive / 'ba.json', ba)
                previous = ba
                ba = bounded_bundle_adjustment(rec, stage, initial, show_progress=False)
                ba['continuation'] = {
                    'from': str(archive), 'previous_solver': previous['solver'],
                    'previous_elapsed_seconds': previous['elapsed_seconds'],
                    'reason': 'Continue from the solver iteration limit.',
                }
            with staged_directory(stage_dir) as staged:
                staged_model = staged / 'sparse/0'
                staged_model.mkdir(parents=True)
                rec.write(staged_model)
                rec.write_text(staged_model)
                if not pycolmap.Reconstruction(staged_model).is_valid():
                    raise RuntimeError(f'Incomplete calibration BA output: {stage}')
                ba['input_model_sha256'] = input_hashes
                ba['model_sha256'] = model_hashes(staged_model)
                write_json(staged / 'ba.json', ba)
        stages[stage] = {'ba': ba,
                         'registered': rec.num_reg_images(), 'points': rec.num_points3D()}
    result = {'selected_images': len(images),
              'frame_indices': sorted({row['frame_index'] for row in images}),
              'registered_images': rec.num_reg_images(), 'stages': stages,
              'observation_scope': SCOPE,
              'manual_annotations_used': False, 'navigation_pose_or_las_used': False}
    write_json(out / 'summary.json', result)
    write_frozen_json(out / 'calibration_identity.json', {
        'version': 1, **protocol_identity(work),
        'pipeline_config_sha256': data['pipeline_config_sha256'],
        'files': {name: file_digest(out / name) for name in _calibration_artifacts()},
    })
    return result
