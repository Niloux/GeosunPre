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
from geosunpre.artifacts import write_json, archive_interrupted, staged_directory
from geosunpre.colmap import MODEL_FILES
from geosunpre.pipeline.holdout import ISOLATION_SCOPE
from geosunpre.pipeline.observations import (is_validation, load_observations, observation_scope,
                                           partitions, protocol_identity)
from geosunpre.pipeline.holdout import restrict_split
from geosunpre.artifacts import write_frozen_json
from geosunpre.camera import Camera
from geosunpre.scene import load_scene


def _input_identity(work, global_split):
    """Bind supplied intrinsics to the same pre-geometry observation protocol."""
    scene = load_scene(work / 'sfm')
    load_config(work).check_scene(scene)
    return {
        'version': 2, 'mode': 'input',
        'scene_manifest_sha256': file_digest(work / 'sfm/manifest.json'),
        'pipeline_config_sha256': file_digest(work / CONFIG_FILE),
        **protocol_identity(work),
        'split_digest': digest(global_split),
        'cameras': scene['cameras'],
        'local_geometry_used': False,
    }


def _verify_input_calibration(work, global_split):
    expected = _input_identity(work, global_split)
    path = work / 'calibration/calibration_identity.json'
    if not path.is_file() or json.loads(path.read_text()) != expected:
        raise ValueError('Input calibration identity changed or missing; run calibration in a fresh work directory')
    model = pycolmap.Reconstruction()
    for cid, label in ((1, 'L'), (2, 'R')):
        camera = Camera.from_dict(expected['cameras'][label])
        camera.distorted_radius(160)
        native = camera.pycolmap()
        native.camera_id = cid
        model.add_camera(native)
    return model


def _freeze_input_calibration(work):
    from geosunpre.pipeline.data import load_frontend

    work = Path(work).resolve()
    _, _, data, _ = load_frontend(work)
    _, split, _ = load_observations(work, data)
    out = work / 'calibration'
    if any((out / name).exists() for name in ('run_contract.json', 'database.db', 'mapping', 'stages', 'seed', 'mapper_output')):
        raise ValueError('Input calibration cannot reuse local geometry; use a fresh work directory')
    identity = _input_identity(work, split)
    write_frozen_json(out / 'calibration_identity.json', identity)
    _verify_input_calibration(work, split)
    result = {'mode': 'input', 'selected_images': 0, 'local_geometry_used': False,
              'observation_scope': observation_scope(work),
              'manual_annotations_used': False, 'navigation_pose_or_las_used': False}
    write_json(out / 'summary.json', result)
    return result


def prepare_calibration(work):
    """Reuse global feature IDs and holdout exclusions for the calibration segment."""
    from geosunpre.pipeline.data import load_frontend

    root, global_scene, global_data, features = load_frontend(work)
    _, split, _ = load_observations(work, global_data)
    out = Path(work).resolve() / 'calibration'
    images = load_config(work).calibration_images(global_scene)
    names = {row['name'] for row in images}
    validation = is_validation(work)
    scene = {**global_scene, 'images': images,
             'scope': ('Local calibration from frozen global features and holdout.' if validation else
                       'Local calibration from all eligible frozen global observations.')}
    scene.pop('fingerprint')
    scene['fingerprint'] = digest(scene)
    local_split = restrict_split(split, images)
    data = {'version': 2, 'scene_fingerprint': scene['fingerprint'],
            'config': global_data['config'],
            'features': {name: entry for name, entry in global_data['features'].items() if name in names},
            **protocol_identity(work),
            'pipeline_config_sha256': file_digest(Path(work) / CONFIG_FILE),
            'image_names': [row['name'] for row in images],
            'pairs': [entry for entry in global_data['pairs']
                      if entry['a'] in names and entry['b'] in names],
            **({'split_path': str(out / 'split.json')} if validation else
               {'observation_scope': observation_scope(work)})}
    data['fingerprint'] = digest(data)
    directory = out
    if not (directory / 'run_contract.json').exists() and any(
            (directory / name).exists() for name in ('database.db', 'seed', 'mapping', 'mapper_output', 'stages')):
        raise ValueError('Uncontracted local geometry exists; use a fresh work directory')
    write_frozen_json(out / 'manifest.json', scene)
    if validation:
        write_frozen_json(directory / 'split.json', local_split)
    write_frozen_json(out / 'frontend.json', data)
    return out, scene, data, {name: features[name] for name in names}


def _calibration_artifacts(work):
    files = ['manifest.json', 'frontend.json', 'run_contract.json']
    if is_validation(work):
        files.append('split.json')
    files += [f'mapping/{name}' for name in MODEL_FILES]
    for stage in STAGES:
        files += [f'stages/{stage}/sparse/0/{name}' for name in MODEL_FILES]
        files.append(f'stages/{stage}/ba.json')
    return files


def verify_calibration(work, global_split):
    """Require a completed calibration tied to this pre-calibration holdout."""
    work = Path(work).resolve()
    if load_config(work).calibration_mode == 'input':
        return _verify_input_calibration(work, global_split)
    out = work / 'calibration'
    path = out / 'calibration_identity.json'
    if not path.is_file():
        raise ValueError('Missing calibration identity; run the calibration stage first')
    identity = json.loads(path.read_text())
    observation_id = protocol_identity(work)
    if (identity.get('version') != 1 or any(identity.get(k) != v for k, v in observation_id.items())
            or identity.get('pipeline_config_sha256') != file_digest(work / CONFIG_FILE)
            or set(identity.get('files', {})) != set(_calibration_artifacts(work))):
        raise ValueError('Calibration belongs to another holdout protocol')
    for relative, expected in identity['files'].items():
        if file_digest(out / relative) != expected:
            raise ValueError(f'Calibration artifact changed: {relative}')
    contract = json.loads((out / 'run_contract.json').read_text())
    if contract.get('geometry_pipeline_version') != 2 or any(contract.get(k) != v for k, v in observation_id.items()):
        raise ValueError('Calibration has no pre-geometry holdout contract')
    model = pycolmap.Reconstruction(out / 'stages/principal/sparse/0')
    assert_holdout_isolated(model, global_split)
    return model


def calibrate(work):
    """Fit the configured calibration subset with global holdout observations excluded."""
    if load_config(work).calibration_mode == 'input':
        return _freeze_input_calibration(work)
    out, scene, data, features = prepare_calibration(work)
    if (out / 'calibration_identity.json').exists():
        verify_calibration(work, partitions(work)[0])
    records = {row['name']: row for row in scene['images']}
    directory = out
    images = [records[name] for name in data['image_names']]
    validation = is_validation(work)
    split = json.loads(Path(data['split_path']).read_text()) if validation else partitions(work)[0]
    contract = {
        'geometry_pipeline_version': 2, 'scene': scene['fingerprint'],
        **protocol_identity(work),
        'pipeline_config_sha256': data['pipeline_config_sha256'],
        'frontend_config': data['config'],
        'features': {name: data['features'][name]['token'] for name in data['image_names']},
        'pairs': [{key: entry[key] for key in ('a', 'b', 'token')}
                  for entry in data['pairs']],
        **({'split_sha256': file_digest(data['split_path'])} if validation else
           {'observation_scope': observation_scope(work)}),
    }
    contract['fingerprint'] = digest(contract)
    contract_path = directory / 'run_contract.json'
    write_frozen_json(contract_path, contract)

    mapping = directory / 'mapping'
    if all((mapping / name).is_file() for name in MODEL_FILES):
        rec = pycolmap.Reconstruction(mapping)
    else:
        db, seed = directory / 'database.db', directory / 'seed'
        if not (db.exists() and all((seed / name).is_file() for name in MODEL_FILES)):
            archive_interrupted(db)
            archive_interrupted(seed)
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
        with staged_directory(mapping) as staged:
            rec.write(staged)
            if not pycolmap.Reconstruction(staged).is_valid():
                raise RuntimeError('Incomplete calibration mapping output')
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
        if report_path.is_file() and all((model / name).is_file() for name in MODEL_FILES):
            rec = pycolmap.Reconstruction(model)
            ba = json.loads(report_path.read_text())
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
                    'reason': 'Continue from the iteration limit without using validation errors.',
                }
            with staged_directory(stage_dir) as staged:
                staged_model = staged / 'sparse/0'
                staged_model.mkdir(parents=True)
                rec.write(staged_model)
                rec.write_text(staged_model)
                if not pycolmap.Reconstruction(staged_model).is_valid():
                    raise RuntimeError(f'Incomplete calibration BA output: {stage}')
                write_json(staged / 'ba.json', ba)
        assert_holdout_isolated(rec, split)
        evaluation = ({'validation': evaluate_reconstruction(rec, scene, features, split, stage_dir)}
                      if validation else {})
        stages[stage] = {'ba': ba, **evaluation,
                         'registered': rec.num_reg_images(), 'points': rec.num_points3D()}
    result = {'selected_images': len(images),
              'frame_indices': sorted({row['frame_index'] for row in images}),
              'registered_images': rec.num_reg_images(), 'stages': stages,
              **({'holdout_isolation': ISOLATION_SCOPE} if validation else
                 {'observation_scope': observation_scope(work)}),
              'manual_annotations_used': False, 'navigation_pose_or_las_used': False}
    write_json(out / 'summary.json', result)
    write_frozen_json(out / 'calibration_identity.json', {
        'version': 1, **protocol_identity(work),
        'pipeline_config_sha256': data['pipeline_config_sha256'],
        'files': {name: file_digest(out / name) for name in _calibration_artifacts(work)},
    })
    return result
