"""Image-only, asynchronous full-sequence SfM with frozen observation sources."""
import json
from pathlib import Path
from geosunpre.profiling import profiled, operation

import numpy as np
import pycolmap

from geosunpre.sfm.reconstruction import bounded_bundle_adjustment
from geosunpre.sfm.reconstruction import cheirality_health
from geosunpre.sfm.reconstruction import database_and_seed
from geosunpre.sfm.reconstruction import mapper_options
from geosunpre.artifacts import digest
from geosunpre.artifacts import file_digest
from geosunpre.artifacts import write_json, archive_interrupted, staged_directory
from geosunpre.colmap import MODEL_FILES, model_hashes
from geosunpre.sfm.track_filter import CONFIG as TRACK_CONFIG
from geosunpre.sfm.track_filter import filter_mapped_tracks
from geosunpre.sfm.track_filter import training_adjacency
from geosunpre.pipeline.observations import load_observations, protocol_identity
from geosunpre.artifacts import write_frozen_json
from geosunpre.pipeline.data import load_frontend

OUTPUT = 'sfm'


@profiled
def freeze_geometry(work, *, require_existing=False):
    out, scene, data, features = load_frontend(work)
    work = Path(work).resolve()
    from geosunpre.pipeline.calibration import verify_calibration

    load_observations(work, data)
    local = work / 'calibration/stages/principal/sparse/0'
    calibration = verify_calibration(work)
    params = {i: camera.params.tolist() for i, camera in calibration.cameras.items()}
    protocol = {'version': 2, 'scene_fingerprint': scene['fingerprint'],
                'frontend_fingerprint': data['fingerprint'],
                **protocol_identity(work),
                'calibration_identity_sha256': file_digest(work / 'calibration/calibration_identity.json'),
                'pair_protocol_sha256': file_digest(out / 'pair_protocol.json'),
                'calibration_source_model_sha256': {p.name: file_digest(p) for p in local.glob('*.bin')},
                'calibration': params,
                'track_filter': TRACK_CONFIG,
                'bundle_adjustment_stages': ['fixed_calibration', 'triangle_supported', 'shared_calibration'],
                'shared_calibration_limits': 'Same bounded visual-only principal-stage limits as local SfM; final parameters must pass 160-degree monotonicity and positive-depth checks.',
                'independent_exposure_poses': True, 'input_pose_values_used': False, 'las_used': False,
                'scope': 'Full selected L/R sequence, candidate shared calibration from local visual fit. No navigation/LAS alignment or photometric acceptance.'}
    path = out / 'geometry_protocol.json'
    if not path.exists():
        if require_existing:
            raise ValueError('Missing global geometry protocol; run the mapping stage first')
        if any((out / name).exists() for name in ('database.db', 'seed', 'mapping', 'mapper_output', 'stages')):
            raise ValueError('Uncontracted global geometry exists; use a fresh work directory')
    write_frozen_json(path, {**protocol, 'fingerprint': digest(protocol)})
    return out, scene, data, features, protocol, params


def geometric_scene(scene, params):
    result = {**scene, 'cameras': {label: {**camera, 'params': params[1 if label == 'L' else 2]}
                                    for label, camera in scene['cameras'].items()}}
    result['geometry_calibration_is_candidate'] = True
    return result


@profiled
def build_mapping(work):
    out, scene, data, features, protocol, params = freeze_geometry(work)
    mapping = out / 'mapping'
    if (mapping / 'identity.json').is_file():
        identity = json.loads((mapping / 'identity.json').read_text())
        if (identity.get('protocol_fingerprint') != digest(protocol)
                or identity.get('model_sha256') != model_hashes(mapping)):
            raise ValueError('Global mapping identity changed')
        rec = pycolmap.Reconstruction(mapping)
    else:
        db, seed = out / 'database.db', out / 'seed'
        if not (db.exists() and all((seed / name).is_file() for name in MODEL_FILES)):
            archive_interrupted(db)
            archive_interrupted(seed)
            db, seed = database_and_seed(out, geometric_scene(scene, params), scene['images'],
                                         data['pairs'], features)
        opts = mapper_options()
        with operation('incremental_mapping'):
            maps = pycolmap.incremental_mapping(db, Path(scene['input_root']) / 'Image',
                                                out / 'mapper_output', options=opts, input_path=seed)
        if not maps: raise RuntimeError('Full mapping did not reconstruct a component')
        rec = max(maps.values(), key=lambda r: r.num_reg_images())
        if cheirality_health(rec)['invalid_depth']: raise RuntimeError('Global mapper model has invalid depths')
        for cid, values in params.items():
            np.testing.assert_array_equal(rec.cameras[cid].params, np.asarray(values))
        with staged_directory(mapping) as staged:
            rec.write(staged)
            rec.write_text(staged)
            if not pycolmap.Reconstruction(staged).is_valid():
                raise RuntimeError('Incomplete global mapping output')
            write_json(staged / 'identity.json', {'protocol_fingerprint': digest(protocol),
                                                 'model_sha256': model_hashes(staged)})
    names = {i.name for i in rec.images.values() if i.has_pose}
    summary = {'selected': len(scene['images']), 'registered': rec.num_reg_images(),
               'registered_names': sorted(names), 'unregistered_names': sorted(set(features) - names),
               'points': rec.num_points3D(), 'observations': rec.compute_num_observations(),
               'cheirality': cheirality_health(rec), 'pose_prior_used': False, 'LAS_used': False,
               'independent_frame_for_each_exposure': True}
    write_json(out / 'mapping_summary.json', summary)
    write_report(out, scene)
    print('Full mapping:', {k: v for k, v in summary.items() if k not in ['registered_names', 'unregistered_names']}, flush=True)
    return rec, summary


@profiled
def optimize(work):
    out, scene, data, features, protocol, _ = freeze_geometry(work, require_existing=True)
    identity = json.loads((out / 'mapping/identity.json').read_text())
    if (identity.get('protocol_fingerprint') != digest(protocol)
            or identity.get('model_sha256') != model_hashes(out / 'mapping')):
        raise ValueError('Global mapping identity changed')
    params = {int(i): np.asarray(p, dtype=float) for i, p in protocol['calibration'].items()}
    summary_path = out / 'optimization_summary.json'
    existing = json.loads(summary_path.read_text()) if summary_path.exists() else {'stages': {}}
    stages = {}
    for stage in ['fixed_calibration', 'triangle_supported', 'shared_calibration']:
        path = out / 'stages' / stage
        previous = {'fixed_calibration': out / 'mapping',
                    'triangle_supported': out / 'stages/fixed_calibration/sparse/0',
                    'shared_calibration': out / 'stages/triangle_supported/sparse/0'}[stage]
        complete = (path / 'ba.json').is_file() and all(
            (path / 'sparse/0' / name).is_file() for name in MODEL_FILES)
        if not complete and (path / 'ba.json').is_file():
            saved = json.loads((path / 'ba.json').read_text())
            if 'model_sha256' in saved:
                raise ValueError(f'Global BA stage identity changed: {stage}: committed model is incomplete')
        if complete:
            ba = json.loads((path / 'ba.json').read_text())
            for key, model_path in (('model_sha256', path / 'sparse/0'),
                                    ('input_model_sha256', previous)):
                if ba.get(key) != model_hashes(model_path):
                    raise ValueError(f'Global BA stage identity changed: {stage}: {key}')
        if stage in existing['stages'] and complete:
            rec = pycolmap.Reconstruction(path / 'sparse/0')
            if stage != 'shared_calibration':
                for cid, p in params.items(): np.testing.assert_array_equal(rec.cameras[cid].params, p)
            stages[stage] = existing['stages'][stage]
            print('Global saved stage', stage, rec.num_reg_images(), rec.num_points3D(), flush=True)
            continue
        if complete:
            rec = pycolmap.Reconstruction(path / 'sparse/0')
            ba = json.loads((path / 'ba.json').read_text())
        else:
            rec = pycolmap.Reconstruction(previous)
            with staged_directory(path) as staged:
                if stage == 'triangle_supported':
                    mapped_points = {(im.name, idx): int(p.point3D_id)
                                     for im in rec.images.values() if im.has_pose
                                     for idx, p in enumerate(im.points2D) if p.has_point3D()}
                    adjacency, count = training_adjacency(data['pairs'],
                        TRACK_CONFIG['match_confidence'], mapped_points=mapped_points)
                    filtration = filter_mapped_tracks(rec, adjacency, staged)
                    del adjacency, mapped_points
                    filtration['input_graph_edges'] = count
                    write_json(staged / 'track_filter.json', filtration)
                    if not filtration['coverage_admissible']:
                        write_json(path / 'REJECTED.json', {'reason': 'Training coverage fell below frozen threshold', 'coverage': filtration['after']['coverage']})
                        raise RuntimeError('Full-scene track filter rejected by training coverage')
                print('Global BA', stage, rec.num_reg_images(), rec.num_points3D(), rec.compute_num_observations(), flush=True)
                ba = bounded_bundle_adjustment(rec, 'principal' if stage == 'shared_calibration' else 'fixed',
                                               params, protect_depth=True, show_progress=False)
                model = staged / 'sparse/0'; model.mkdir(parents=True)
                rec.write(model); rec.write_text(model)
                if not pycolmap.Reconstruction(model).is_valid():
                    raise RuntimeError(f'Incomplete global BA output: {stage}')
                ba['input_model_sha256'] = model_hashes(previous)
                ba['model_sha256'] = model_hashes(model)
                write_json(staged / 'ba.json', ba)
        if stage != 'shared_calibration':
            for cid, p in params.items(): np.testing.assert_array_equal(rec.cameras[cid].params, p)
        stages[stage] = {'ba': ba, 'registered': rec.num_reg_images(), 'points': rec.num_points3D(),
                         'observations': rec.compute_num_observations(),
                         'camera_params': {cid: c.params.tolist() for cid, c in rec.cameras.items()}}
        write_json(summary_path, {'stages': stages,
            'scope': 'Optimization on all eligible observations; fitted-data diagnostics.'})
        write_report(out, scene)
    return stages


def write_report(out, scene):
    from geosunpre.pipeline.quality import write_production_report
    return write_production_report(out, scene)
