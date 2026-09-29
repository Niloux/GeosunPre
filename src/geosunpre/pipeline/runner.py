"""Unattended reconstruction and native COLMAP delivery."""

import json
from pathlib import Path

import numpy as np
import pycolmap

from geosunpre.sfm.evaluation import evaluate_reconstruction
from geosunpre.sfm.reconstruction import assert_holdout_isolated
from geosunpre.sfm.reconstruction import bounded_bundle_adjustment
from geosunpre.artifacts import digest
from geosunpre.artifacts import digest_cache
from geosunpre.artifacts import file_digest
from geosunpre.scene import load_scene
from geosunpre.artifacts import write_json
from geosunpre.colmap import MODEL_FILES
from geosunpre.colmap import export_colmap
from geosunpre.pipeline.holdout import ISOLATION_SCOPE
from geosunpre.pipeline import STAGES
from geosunpre.config import CONFIG_FILE, PipelineConfig, load_config, save_config
from geosunpre.pipeline.runtime import StageRecorder, work_lock


BA_ITERATIONS = 40
SOURCE = Path('sfm/stages/shared_calibration/sparse/0')
FINAL = Path('sfm/final')
DEFAULT_OUTPUT = 'colmap'


def _model_hashes(path):
    return {name: file_digest(path / name) for name in MODEL_FILES}


def _check_model(rec, scene):
    records = {row['name']: row for row in scene['images']}
    names = {rec.images[image_id].name for image_id in rec.reg_image_ids()}
    if not records or names != set(records) or rec.num_reg_images() != len(scene['images']):
        raise ValueError('Expected every selected L/R exposure to be registered exactly once')
    if {row['camera'] for row in records.values()} != {'L', 'R'}:
        raise ValueError('Expected both L/R cameras')
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


def finalize(work):
    """Finalize the native fisheye model with guarded bundle adjustment."""
    from geosunpre.pipeline.reconstruction import freeze_geometry

    work = Path(work).resolve()
    root, scene, _, features, _, _ = freeze_geometry(work, require_existing=True)
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
    output = work / FINAL
    output.mkdir(parents=True, exist_ok=True)
    protocol = {
        'schema_version': 1,
        'source_model_sha256': _model_hashes(source_path),
        'scene_manifest_sha256': file_digest(root / 'manifest.json'),
        'scene_fingerprint': scene['fingerprint'],
        'geometry_protocol_sha256': file_digest(geometry_path),
        'split_sha256': file_digest(split_path),
        'solver': {'stage': 'principal', 'protect_depth': True,
                   'max_iterations': BA_ITERATIONS},
        'manual_annotations_used': False,
        'navigation_pose_or_las_used': False,
        'all_exposures_independent': True,
    }
    protocol['fingerprint'] = digest(protocol)
    protocol_path = output / 'protocol.json'
    if protocol_path.exists():
        if json.loads(protocol_path.read_text())['fingerprint'] != protocol['fingerprint']:
            raise ValueError('Final BA inputs changed; use a fresh work directory')
    else:
        write_json(protocol_path, protocol)

    model_path = output / 'sparse/0'
    result_path = output / 'result.json'
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if (result['protocol_fingerprint'] != protocol['fingerprint']
                or result['model_sha256'] != _model_hashes(model_path)):
            raise ValueError('Saved final BA model changed')
        candidate = pycolmap.Reconstruction(model_path)
    else:
        if any((model_path / name).exists() for name in MODEL_FILES):
            raise RuntimeError(f'Interrupted final BA output; inspect {model_path}')
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
            raise RuntimeError('Final BA changed reconstruction topology')
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
            raise ValueError('Final BA validation belongs to another model')
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


def export_result(work, *, output=None, image_mode='copy',
                  include_masks=False, include_ply=False):
    """Validate the computed candidate and export a consumer-independent dataset."""
    work = Path(work).resolve()
    root = work / 'sfm'
    scene = load_scene(root)
    final = work / FINAL
    model_path = final / 'sparse/0'
    protocol_path = final / 'protocol.json'
    result_path = final / 'result.json'
    validation_path = final / 'validation_summary.json'
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
        raise ValueError('Computed final BA identity differs')
    from geosunpre.pipeline.reconstruction import freeze_geometry
    freeze_geometry(work, require_existing=True)
    rec = pycolmap.Reconstruction(model_path)
    _check_model(rec, scene)
    assert_holdout_isolated(rec, json.loads((root / 'split.json').read_text()))
    output = Path(output).resolve() if output else work / DEFAULT_OUTPUT
    if output == work or output.is_relative_to(root):
        raise ValueError('Choose a separate COLMAP export directory')
    provenance = {
        'kind': 'visual_sfm_pipeline',
        'source_model_sha256': result['model_sha256'],
        'scene_manifest_sha256': file_digest(root / 'manifest.json'),
        'final_ba_protocol_sha256': file_digest(protocol_path),
        'final_ba_result_sha256': file_digest(result_path),
        'final_ba_validation_sha256': file_digest(validation_path),
        'holdout_protocol_sha256': file_digest(root / 'holdout_protocol.json'),
        'calibration_identity_sha256': file_digest(work / 'calibration/calibration_identity.json'),
        'holdout_isolation': ISOLATION_SCOPE,
        'independent_exposure_poses': True,
        'manual_annotations_used': False, 'navigation_pose_values_or_las_used': False,
    }
    return export_colmap(model_path, scene, output, image_mode=image_mode,
                         include_masks=include_masks, include_ply=include_ply,
                         provenance=provenance, quality_report=validation_path)


def run_pipeline(work, *, input_root=None, config=None, stage='all', output=None,
                 image_mode='copy', include_masks=False, include_ply=False,
                 model_cache=None):
    """Run all stages, or resume one stage using the work directory's frozen config."""
    if stage not in (*STAGES, 'all'):
        raise ValueError(f'Unknown pipeline stage: {stage}')
    work = Path(work).resolve()
    selected = STAGES if stage == 'all' else (stage,)
    if config is not None and not isinstance(config, PipelineConfig):
        config = PipelineConfig.from_file(config)
    with work_lock(work), digest_cache():
        if (work / CONFIG_FILE).exists():
            saved = load_config(work)
            if config is not None:
                save_config(work, config)
            config = saved
        else:
            if any(path.name != '.pipeline.lock' for path in work.iterdir()):
                raise ValueError('Unmanaged work directory; choose a fresh --work directory')
            if 'prepare' not in selected or input_root is None:
                raise ValueError('A new run requires --input and the prepare/all stage')
            config = config or PipelineConfig()
            save_config(work, config)
        if 'prepare' in selected and input_root is None:
            if not (work / 'manifest.json').exists():
                raise ValueError('--input is required until the prepare stage completes')
            input_root = load_scene(work)['input_root']
        if input_root is not None and (work / 'manifest.json').exists():
            if Path(input_root).resolve() != Path(load_scene(work)['input_root']).resolve():
                raise ValueError('Input directory changed; use a fresh work directory')

        recorder = StageRecorder(work, selected)
        summary = {'work': str(work), 'stages': [],
                   'config_sha256': file_digest(work / CONFIG_FILE)}
        for current in selected:
            with recorder.stage(current):
                if current == 'prepare':
                    from geosunpre.scene import prepare
                    scene = prepare(input_root, work)
                    calibration = config.calibration_images(scene)
                    summary['input'] = {'images': len(scene['images']),
                                        'calibration_images': len(calibration)}
                elif current == 'masks':
                    from geosunpre.pipeline.data import prepare_masks
                    prepare_masks(work, model_cache=model_cache)
                elif current == 'features':
                    from geosunpre.pipeline.data import extract_features
                    extract_features(work)
                elif current == 'matches':
                    from geosunpre.pipeline.data import match_features
                    match_features(work)
                elif current == 'holdout':
                    from geosunpre.pipeline.holdout import freeze_holdout
                    protocol = freeze_holdout(work)
                    summary['holdout'] = {key: protocol[key]
                                          for key in ('fingerprint', 'isolation_scope')}
                elif current == 'calibration':
                    from geosunpre.pipeline.calibration import calibrate
                    calibrate(work)
                elif current == 'sfm':
                    from geosunpre.pipeline.reconstruction import build_mapping, evaluate_mapping, optimize
                    build_mapping(work)
                    evaluate_mapping(work)
                    optimize(work)
                elif current == 'finalize':
                    summary['model'] = finalize(work)
                elif current == 'export':
                    summary['export'] = export_result(
                        work, output=output, image_mode=image_mode,
                        include_masks=include_masks, include_ply=include_ply)
                summary['stages'].append(current)
        write_json(work / 'run_result.json', summary)
        recorder.complete()
        return summary
