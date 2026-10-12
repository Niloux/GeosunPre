"""Finalize and validate the visual model before LAS registration."""

import json
from pathlib import Path

import pycolmap

from geosunpre.artifacts import file_metadata, staged_directory, write_json
from geosunpre.colmap import model_files
from geosunpre.config import load_config
from geosunpre.pipeline.observations import protocol_identity
from geosunpre.pipeline.quality import production_acceptance

SOURCE = Path('sfm/stages/shared_calibration/sparse/0')
FINAL = Path('sfm/final')


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
    """Accept and publish the completed visual solve without repeating BA."""
    from geosunpre.pipeline.reconstruction import freeze_geometry

    work = Path(work).resolve()
    root, scene, _, _, _, _ = freeze_geometry(work, require_existing=True)
    source_path = work / SOURCE
    source = pycolmap.Reconstruction(source_path)
    _check_model(source, scene)
    geometry_path = root / 'geometry_protocol.json'
    ba_path = source_path.parent.parent / 'ba.json'
    ba = json.loads(ba_path.read_text(encoding='utf-8'))
    if ba['model_files'] != model_files(source_path):
        raise ValueError('Final optimization report differs from the source model')
    output = work / FINAL
    output.mkdir(parents=True, exist_ok=True)
    protocol = {
        'schema_version': 2,
        'source_model_files': model_files(source_path),
        'scene_manifest_file': file_metadata(root / 'manifest.json'),
        'geometry_protocol_file': file_metadata(geometry_path),
        **protocol_identity(work),
        'source_optimization_file': file_metadata(ba_path),
        'numerical_fit_performed': False,
        'manual_annotations_used': False,
        'navigation_pose_or_las_used': False,
        'all_exposures_independent': True,
    }
    protocol_path = output / 'protocol.json'
    if protocol_path.exists():
        if json.loads(protocol_path.read_text(encoding='utf-8')) != protocol:
            raise ValueError('Final BA inputs changed; use a fresh work directory')
    model_path = output / 'sparse/0'
    result_path = output / 'result.json'
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding='utf-8'))
        if (result['protocol_file'] != file_metadata(protocol_path)
                or result['model_files'] != model_files(model_path)):
            raise ValueError('Saved final visual model changed')
        candidate = pycolmap.Reconstruction(model_path)
    else:
        candidate = pycolmap.Reconstruction(source_path)
        after = (candidate.num_reg_images(), candidate.num_points3D(),
                 candidate.compute_num_observations())
        _check_model(candidate, scene)
        with staged_directory(output) as staged:
            staged_model = staged / 'sparse/0'
            staged_model.mkdir(parents=True)
            candidate.write(staged_model)
            _check_model(pycolmap.Reconstruction(staged_model), scene)
            write_json(staged / 'protocol.json', protocol)
            result = {'protocol_file': file_metadata(staged / 'protocol.json'),
                      'model_files': model_files(staged_model), 'ba': ba,
                      'numerical_fit_performed': False,
                      'registered': after[0], 'points': after[1],
                      'observations': after[2],
                      'manual_annotations_used': False}
            write_json(staged / 'result.json', result)
    _check_model(candidate, scene)
    validation_path = output / 'quality_summary.json'
    if validation_path.exists():
        validation = json.loads(validation_path.read_text(encoding='utf-8'))
        if validation['model_files'] != result['model_files']:
            raise ValueError('Final BA quality report belongs to another model')
    else:
        from geosunpre.pipeline.quality import production_quality
        validation = {'model_files': result['model_files'],
                      **production_quality(candidate), 'manual_annotations_used': False}
        validation['camera_parameter_bounds'] = result['ba'].get('cameras', {})
        validation['acceptance'] = production_acceptance(validation, load_config(work))
        write_json(validation_path, validation)
    acceptance = production_acceptance(validation, load_config(work))
    return {'model': str(model_path), 'registered': candidate.num_reg_images(),
            'points': candidate.num_points3D(),
            'protocol_file': file_metadata(protocol_path),
            'training_reprojection': validation['training_reprojection'], 'acceptance': acceptance}


def verify_final(work):
    """Check the visual model without fitting or publishing anything."""
    from geosunpre.pipeline.reconstruction import freeze_geometry

    work = Path(work).resolve()
    root, scene, *_ = freeze_geometry(work, require_existing=True)
    final = work / FINAL
    protocol = json.loads((final / 'protocol.json').read_text(encoding='utf-8'))
    result = json.loads((final / 'result.json').read_text(encoding='utf-8'))
    quality = json.loads((final / 'quality_summary.json').read_text(encoding='utf-8'))
    if (result['protocol_file'] != file_metadata(final / 'protocol.json')
            or result['model_files'] != model_files(final / 'sparse/0')
            or quality['model_files'] != result['model_files']
            or protocol['scene_manifest_file'] != file_metadata(root / 'manifest.json')
            or protocol['source_model_files'] != model_files(work / SOURCE)
            or protocol['source_optimization_file'] != file_metadata(work / SOURCE.parent.parent / 'ba.json')
            or protocol['geometry_protocol_file'] != file_metadata(root / 'geometry_protocol.json')
            or any(protocol.get(k) != v for k, v in protocol_identity(work).items())):
        raise ValueError('Final visual model identity changed')
    _check_model(pycolmap.Reconstruction(final / 'sparse/0'), scene)
    production_acceptance(quality, load_config(work))
    return scene, result, quality
