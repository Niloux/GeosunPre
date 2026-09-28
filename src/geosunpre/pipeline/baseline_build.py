"""Unattended visual baseline reconstruction and native COLMAP export."""

from collections import Counter
import json
from pathlib import Path

import numpy as np
import pycolmap

from ..local_evaluation import evaluate_reconstruction
from ..local_reconstruction import assert_holdout_isolated, bounded_bundle_adjustment
from ..scene import digest, file_digest, load_scene, write_json
from .colmap_export import MODEL_FILES, export_colmap, verify_colmap_export
from .holdout import ISOLATION_SCOPE


EXPECTED_IMAGES = 536
BA_ITERATIONS = 40
SOURCE = Path('global_sfm/stages/shared_calibration/sparse/0')
EQUAL = Path('global_sfm/baseline_equal')
DEFAULT_OUTPUT = 'colmap'


def _model_hashes(path):
    return {name: file_digest(path / name) for name in MODEL_FILES}


def _check_model(rec, scene):
    records = {row['name']: row for row in scene['images']}
    names = {rec.images[image_id].name for image_id in rec.reg_image_ids()}
    if (len(records) != EXPECTED_IMAGES or names != set(records)
            or rec.num_reg_images() != len(records)):
        raise ValueError('Expected all 536 selected L/R exposures to be registered')
    if Counter(row['camera'] for row in records.values()) != {
            'L': EXPECTED_IMAGES // 2, 'R': EXPECTED_IMAGES // 2}:
        raise ValueError('Expected equal L/R exposure counts')
    if len(rec.cameras) != 2 or any(camera.model.name != 'OPENCV_FISHEYE'
                                    for camera in rec.cameras.values()):
        raise ValueError('Expected two native OPENCV_FISHEYE cameras')
    for image_id in rec.reg_image_ids():
        image = rec.images[image_id]
        if len(rec.frames[image.frame_id].image_ids) != 1:
            raise ValueError('Each asynchronous exposure must have its own frame')
        if image.name.split('/')[0] != records[image.name]['camera']:
            raise ValueError(f'Image/camera identity differs: {image.name}')
        if image.camera_id != (1 if records[image.name]['camera'] == 'L' else 2):
            raise ValueError(f'Image uses the wrong fisheye camera: {image.name}')
        expected = scene['cameras'][records[image.name]['camera']]
        camera = rec.cameras[image.camera_id]
        if (camera.width, camera.height) != (expected['width'], expected['height']):
            raise ValueError(f'Image dimensions differ: {image.name}')
    return records


def build_equal_control(work):
    """Continue native fisheye BA at weight one, without floor or manual inputs."""
    from .global_reconstruction import freeze_global_geometry

    work = Path(work).resolve()
    root, scene, _, features, _, _ = freeze_global_geometry(work, require_existing=True)
    source_path = work / SOURCE
    source = pycolmap.Reconstruction(source_path)
    _check_model(source, scene)
    split_path = root / 'split.json'
    split = json.loads(split_path.read_text())
    assert_holdout_isolated(source, split)
    geometry_path = root / 'geometry_protocol.json'
    geometry = json.loads(geometry_path.read_text())
    initial_cameras = {int(cid): np.asarray(values, dtype=float)
                       for cid, values in geometry['calibration'].items()}
    if set(initial_cameras) != set(source.cameras):
        raise ValueError('Frozen initial calibration differs from the model cameras')
    output = work / EQUAL
    output.mkdir(parents=True, exist_ok=True)
    protocol = {
        'schema_version': 1,
        'source_model_sha256': _model_hashes(source_path),
        'scene_manifest_sha256': file_digest(root / 'manifest.json'),
        'scene_fingerprint': scene['fingerprint'],
        'geometry_protocol_sha256': file_digest(geometry_path),
        'split_sha256': file_digest(split_path),
        'solver': {'stage': 'principal', 'protect_depth': True,
                   'max_iterations': BA_ITERATIONS, 'factor_hook': None},
        'manual_annotations_used': False,
        'floor_classification_used': False,
        'navigation_pose_or_las_used': False,
        'all_exposures_independent': True,
    }
    protocol['fingerprint'] = digest(protocol)
    protocol_path = output / 'protocol.json'
    if protocol_path.exists():
        if json.loads(protocol_path.read_text())['fingerprint'] != protocol['fingerprint']:
            raise ValueError('Equal BA inputs changed; use a fresh work directory')
    else:
        write_json(protocol_path, protocol)

    model_path = output / 'sparse/0'
    result_path = output / 'result.json'
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if (result['protocol_fingerprint'] != protocol['fingerprint']
                or result['model_sha256'] != _model_hashes(model_path)):
            raise ValueError('Saved equal BA model changed')
        candidate = pycolmap.Reconstruction(model_path)
    else:
        if any((model_path / name).exists() for name in MODEL_FILES):
            raise RuntimeError(f'Interrupted equal BA output; inspect {model_path}')
        candidate = pycolmap.Reconstruction(source_path)
        before = (candidate.num_reg_images(), candidate.num_points3D(),
                  candidate.compute_num_observations())
        ba = bounded_bundle_adjustment(candidate, 'principal', initial_cameras,
                                       protect_depth=True, show_progress=True,
                                       max_iterations=BA_ITERATIONS)
        assert_holdout_isolated(candidate, split)
        after = (candidate.num_reg_images(), candidate.num_points3D(),
                 candidate.compute_num_observations())
        if before != after:
            raise RuntimeError('Equal BA changed reconstruction topology')
        _check_model(candidate, scene)
        model_path.mkdir(parents=True, exist_ok=True)
        candidate.write(model_path)
        result = {'protocol_fingerprint': protocol['fingerprint'],
                  'model_sha256': _model_hashes(model_path), 'ba': ba,
                  'registered': after[0], 'points': after[1],
                  'observations': after[2],
                  'manual_annotations_used': False}
        write_json(result_path, result)
    _check_model(candidate, scene)
    assert_holdout_isolated(candidate, split)
    validation_path = output / 'validation_summary.json'
    if validation_path.exists():
        validation = json.loads(validation_path.read_text())
        if validation['model_sha256'] != result['model_sha256']:
            raise ValueError('Equal BA validation belongs to another model')
    else:
        triangle = json.loads((root / 'triangle_validation_split.json').read_text())
        raw = evaluate_reconstruction(candidate, scene, features, split,
                                      output / 'validation')
        high = evaluate_reconstruction(candidate, scene, features, triangle,
                                       output / 'triangle_validation')
        validation = {'model_sha256': result['model_sha256'],
                      'raw': raw, 'triangle': high,
                      'manual_annotations_used': False}
        write_json(validation_path, validation)
    return {'model': str(model_path), 'registered': candidate.num_reg_images(),
            'points': candidate.num_points3D(),
            'protocol_fingerprint': protocol['fingerprint'],
            'triangle_holdout': validation['triangle']['all_candidate_queries']}


def export_computed_baseline(work, *, output=None, image_mode='link',
                             include_masks=False, include_ply=False):
    """Validate the computed candidate and export a consumer-independent dataset."""
    work = Path(work).resolve()
    root = work / 'global_sfm'
    scene = load_scene(root)
    equal = work / EQUAL
    model_path = equal / 'sparse/0'
    protocol_path = equal / 'protocol.json'
    result_path = equal / 'result.json'
    validation_path = equal / 'validation_summary.json'
    protocol = json.loads(protocol_path.read_text())
    result = json.loads(result_path.read_text())
    validation = json.loads(validation_path.read_text())
    if (result['protocol_fingerprint'] != protocol['fingerprint']
            or result['model_sha256'] != _model_hashes(model_path)
            or validation['model_sha256'] != result['model_sha256']
            or protocol['scene_manifest_sha256'] != file_digest(root / 'manifest.json')
            or protocol['source_model_sha256'] != _model_hashes(work / SOURCE)
            or protocol['geometry_protocol_sha256'] != file_digest(root / 'geometry_protocol.json')
            or protocol['split_sha256'] != file_digest(root / 'split.json')
            or protocol['manual_annotations_used'] is not False):
        raise ValueError('Computed equal BA identity differs')
    from .global_reconstruction import freeze_global_geometry
    freeze_global_geometry(work, require_existing=True)
    rec = pycolmap.Reconstruction(model_path)
    _check_model(rec, scene)
    assert_holdout_isolated(rec, json.loads((root / 'split.json').read_text()))
    output = Path(output).resolve() if output else work / DEFAULT_OUTPUT
    if output == work or output.is_relative_to(root):
        raise ValueError('Choose a separate COLMAP export directory')
    provenance = {
        'kind': 'computed_visual_baseline', 'status': 'new_run_candidate',
        'source_model_sha256': result['model_sha256'],
        'scene_manifest_sha256': file_digest(root / 'manifest.json'),
        'equal_ba_protocol_sha256': file_digest(protocol_path),
        'equal_ba_result_sha256': file_digest(result_path),
        'equal_ba_validation_sha256': file_digest(validation_path),
        'holdout_protocol_sha256': file_digest(root / 'holdout_protocol.json'),
        'calibration_identity_sha256': file_digest(work / 'local_sfm/calibration_identity.json'),
        'holdout_isolation': ISOLATION_SCOPE,
        'independent_exposure_poses': True,
        'manual_annotations_used': False, 'navigation_pose_values_or_las_used': False,
    }
    return export_colmap(model_path, scene, output, image_mode=image_mode,
                         include_masks=include_masks, include_ply=include_ply,
                         provenance=provenance, quality_report=validation_path)


def verify_computed_baseline(work=None, *, output=None):
    """Check the export independently; work is only a default path convenience."""
    if output is None:
        if work is None:
            raise ValueError('Provide --output or --work')
        output = Path(work).resolve() / DEFAULT_OUTPUT
    return verify_colmap_export(output)


def run_baseline(work, *, input_root=None, stage='all', output=None,
                 image_mode='link', include_masks=False, include_ply=False):
    """Run or resume the basketball baseline stages in dependency order."""
    stages = ('prepare', 'global-data', 'holdout', 'local-calibration', 'global-sfm',
              'equal-ba', 'export')
    if stage not in (*stages, 'all'):
        raise ValueError(f'Unknown baseline stage: {stage}')
    work = Path(work).resolve()
    if ((work / 'baseline_colmap/baseline_export.json').exists()
            or (work / 'global_sfm/lfs_ab/protocol.json').exists()):
        raise ValueError('This work directory contains the frozen historical baseline; use a fresh work directory')
    if stage in ('all', 'prepare') and input_root is None:
        raise ValueError('--input is required for prepare/all')
    if stage in ('all', 'export'):
        destination = Path(output).resolve() if output else work / DEFAULT_OUTPUT
        if (destination / 'baseline_export.json').exists():
            raise ValueError('Destination contains the frozen historical baseline export')
    selected = stages if stage == 'all' else (stage,)
    summary = {'work': str(work), 'stages': []}
    for current in selected:
        print(f'Baseline stage: {current}', flush=True)
        if current == 'prepare':
            from ..scene import prepare
            scene = prepare(input_root, work, fov=160, principal_y='up')
            if len(scene['images']) != EXPECTED_IMAGES:
                raise ValueError(f'Expected {EXPECTED_IMAGES} pose-selected L/R images')
        elif current == 'local-calibration':
            from .local_calibration import solve_baseline_calibration
            solve_baseline_calibration(work)
        elif current == 'global-data':
            from .global_data import (extract_global_features,
                                      match_global_features, prepare_global)
            prepare_global(work)
            extract_global_features(work)
            match_global_features(work)
        elif current == 'holdout':
            from .holdout import freeze_global_holdout
            protocol = freeze_global_holdout(work)
            summary['holdout'] = {key: protocol[key] for key in ('fingerprint', 'isolation_scope')}
        elif current == 'global-sfm':
            from .global_reconstruction import (build_global_mapping,
                                                evaluate_global_mapping, solve_global)
            build_global_mapping(work)
            evaluate_global_mapping(work)
            solve_global(work)
        elif current == 'equal-ba':
            summary['equal'] = build_equal_control(work)
        elif current == 'export':
            summary['export'] = export_computed_baseline(
                work, output=output, image_mode=image_mode,
                include_masks=include_masks, include_ply=include_ply)
        summary['stages'].append(current)
    return summary
