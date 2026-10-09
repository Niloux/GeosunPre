"""Visual calibration using a subset of the globally frozen observations."""
import json
from pathlib import Path

import pycolmap

from geosunpre.config import CONFIG_FILE
from geosunpre.sfm.calibration_selection import select_calibration
from geosunpre.sfm.reconstruction import STAGES
from geosunpre.sfm.reconstruction import bounded_bundle_adjustment
from geosunpre.sfm.reconstruction import cheirality_health
from geosunpre.sfm.reconstruction import database_and_seed
from geosunpre.sfm.reconstruction import mapper_options, map_from_inputs
from geosunpre.artifacts import file_metadata, staged_directory, write_frozen_json, write_json
from geosunpre.colmap import MODEL_FILES, model_files
from geosunpre.pipeline.observations import SCOPE, protocol_identity


def prepare_calibration(work):
    """Reuse global feature IDs and source identities for the calibration subset."""
    from geosunpre.pipeline.data import load_frontend

    _, global_scene, global_data, features = load_frontend(work)
    out = Path(work).resolve() / 'calibration'
    selection_path = out / 'selection.json'
    if selection_path.is_file():
        selection = json.loads(selection_path.read_text())
        selected = set(selection['image_names'])
        images = [row for row in global_scene['images'] if row['name'] in selected]
        if len(images) != len(selected):
            raise ValueError('Calibration selection differs from the frozen scene')
    else:
        images, selection = select_calibration(global_scene, global_data['pairs'], features)
        write_frozen_json(selection_path, selection)
    names = {row['name'] for row in images}
    if not (out / 'run_contract.json').exists() and any(
            (out / name).exists() for name in ('mapping_inputs', 'mapping', 'mapper_output', 'stages')):
        raise ValueError('Uncontracted local geometry exists; use a fresh work directory')
    scene = {**global_scene, 'images': images,
             'scope': 'Automatically selected calibration support from frozen global observations.'}
    write_frozen_json(out / 'manifest.json', scene)
    data = {'version': 2, 'scene_manifest_file': file_metadata(out / 'manifest.json'),
            'config': global_data['config'],
            'features': {name: entry for name, entry in global_data['features'].items() if name in names},
            **protocol_identity(work),
            'pipeline_config_file': file_metadata(Path(work) / CONFIG_FILE),
            'image_names': [row['name'] for row in images],
            'pairs': [entry for entry in global_data['pairs']
                      if entry['a'] in names and entry['b'] in names],
            'observation_scope': SCOPE}
    write_frozen_json(out / 'frontend.json', data)
    return out, scene, data, {name: features[name] for name in names}


def _calibration_artifacts():
    files = ['manifest.json', 'frontend.json', 'run_contract.json', 'selection.json']
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
            or identity.get('pipeline_config_file') != file_metadata(work / CONFIG_FILE)
            or set(identity.get('files', {})) != set(_calibration_artifacts())):
        raise ValueError('Calibration belongs to another observation protocol')
    for relative, expected in identity['files'].items():
        if file_metadata(out / relative) != expected:
            raise ValueError(f'Calibration artifact changed: {relative}')
    contract = json.loads((out / 'run_contract.json').read_text())
    if contract.get('geometry_pipeline_version') != 2 or any(contract.get(k) != v for k, v in observation_id.items()):
        raise ValueError('Calibration has no pre-geometry observation contract')
    model = pycolmap.Reconstruction(out / 'stages/principal/sparse/0')
    return model


def calibrate(work):
    """Fit automatically selected support from frozen global observations."""
    out, scene, data, features = prepare_calibration(work)
    if (out / 'calibration_identity.json').exists():
        verify_calibration(work)
    records = {row['name']: row for row in scene['images']}
    directory = out
    images = [records[name] for name in data['image_names']]
    contract = {
        'geometry_pipeline_version': 2, 'scene_manifest_file': file_metadata(out / 'manifest.json'),
        'selection_file': file_metadata(out / 'selection.json'),
        **protocol_identity(work),
        'pipeline_config_file': data['pipeline_config_file'],
        'frontend_config': data['config'],
        'features': {name: data['features'][name]['token'] for name in data['image_names']},
        'pairs': [{key: entry[key] for key in ('a', 'b', 'token')}
                  for entry in data['pairs']],
        'observation_scope': SCOPE,
    }
    contract_path = directory / 'run_contract.json'
    write_frozen_json(contract_path, contract)
    contract_file = file_metadata(contract_path)

    mapping = directory / 'mapping'
    if (mapping / 'identity.json').is_file():
        identity = json.loads((mapping / 'identity.json').read_text())
        if (identity.get('protocol_file') != contract_file
                or identity.get('model_files') != model_files(mapping)):
            raise ValueError('Calibration mapping identity changed')
        rec = pycolmap.Reconstruction(mapping)
    else:
        db, seed = database_and_seed(directory, scene, images, data['pairs'], features,
                                     protocol_file=contract_file)
        opts = mapper_options()
        maps = map_from_inputs(directory, scene, db, seed)
        if not maps:
            raise RuntimeError('No local calibration reconstruction')
        rec = max(maps.values(), key=lambda item: item.num_reg_images())
        if rec.num_reg_images() != len(images):
            raise RuntimeError('Local calibration did not register all selected images')
        with staged_directory(mapping) as staged:
            rec.write(staged)
            if not pycolmap.Reconstruction(staged).is_valid():
                raise RuntimeError('Incomplete calibration mapping output')
            write_json(staged / 'identity.json', {'protocol_file': contract_file,
                                                 'model_files': model_files(staged)})
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
        input_files = model_files(previous_model)
        if report_path.is_file() and not all((model / name).is_file() for name in MODEL_FILES):
            if 'model_files' in json.loads(report_path.read_text()):
                raise ValueError(f'Committed calibration model is incomplete: {stage}')
        if report_path.is_file() and all((model / name).is_file() for name in MODEL_FILES):
            rec = pycolmap.Reconstruction(model)
            ba = json.loads(report_path.read_text())
            if (ba.get('input_model_files') != input_files
                    or ba.get('model_files') != model_files(model)):
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
                ba['input_model_files'] = input_files
                ba['model_files'] = model_files(staged_model)
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
        'pipeline_config_file': data['pipeline_config_file'],
        'files': {name: file_metadata(out / name) for name in _calibration_artifacts()},
    })
    return result
