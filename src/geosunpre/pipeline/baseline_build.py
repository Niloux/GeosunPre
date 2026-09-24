"""Unattended visual baseline reconstruction and single-branch LFS export."""

from collections import Counter
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pycolmap

from ..camera import Camera
from ..local_evaluation import evaluate_reconstruction
from ..local_reconstruction import assert_holdout_isolated, bounded_bundle_adjustment
from ..scene import digest, file_digest, load_scene, write_json
from .baseline import MODEL_FILES, _copy_immutable, _link_immutable
from .lfs_assets import filtered_points


EXPECTED_IMAGES = 536
BA_ITERATIONS = 40
SOURCE = Path('global_sfm/stages/shared_calibration/sparse/0')
EQUAL = Path('global_sfm/baseline_equal')
DEFAULT_CONFIG = Path(__file__).resolve().parents[3] / 'baselines/basketball/lfs_config.json'


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
    work = Path(work).resolve()
    root = work / 'global_sfm'
    scene = load_scene(root)
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
        from .global_reconstruction import load_global
        _, _, _, features = load_global(work)
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


def _write_immutable(path, data):
    path = Path(path)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f'Existing baseline export differs: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    if temporary.exists():
        raise FileExistsError(temporary)
    temporary.write_bytes(data)
    temporary.replace(path)


def export_computed_baseline(work, *, output=None, lfs_config=DEFAULT_CONFIG):
    """Package one computed model; never consult the old frozen manifest."""
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
    from .global_reconstruction import load_global
    load_global(work)  # Check every original/mask against the frozen LoMa cache.
    rec = pycolmap.Reconstruction(model_path)
    records = _check_model(rec, scene)
    split = json.loads((root / 'split.json').read_text())
    assert_holdout_isolated(rec, split)
    output = Path(output).resolve() if output else work / 'baseline_colmap'
    if output == work or output.is_relative_to(root):
        raise ValueError('Choose a separate baseline export directory')
    if (output / 'baseline_export.json').exists():
        raise ValueError('Destination contains the historical frozen baseline export')
    config_path = Path(lfs_config).resolve()
    config = json.loads(config_path.read_text())
    if (config.get('dataset', {}).get('images') != 'images'
            or config['dataset'].get('use_test_split') is not False
            or config.get('optimization', {}).get('undistort') is not False
            or 'init_path' in config):
        raise ValueError('LFS config must use original fisheye images and all-view training')

    for name in MODEL_FILES:
        _copy_immutable(model_path / name, output / 'sparse/0' / name)
    centers = np.asarray([rec.images[i].projection_center()
                          for i in sorted(rec.reg_image_ids())], dtype=float)
    trajectory_radius = float(np.quantile(
        np.linalg.norm(centers - np.median(centers, axis=0), axis=1), .9))
    if not np.isfinite(trajectory_radius) or trajectory_radius <= 0:
        raise ValueError('Invalid visual trajectory scale for point filtering')
    ply, point_info = filtered_points(rec, max_distance=10 * trajectory_radius)
    _write_immutable(output / 'sparse/0/points3D.ply', ply)
    _copy_immutable(config_path, output / 'lfs_config.json')
    fov = {lens: Camera.from_dict(scene['cameras'][lens]).mask(scene['fov_degrees'])
           for lens in ('L', 'R')}
    images_and_masks = []
    for name, record in sorted(records.items()):
        source = Path(record['source'])
        semantic = Path(record['semantic_mask'])
        if not source.is_file() or not semantic.is_file():
            raise FileNotFoundError(f'Missing source image or semantic mask: {name}')
        with Image.open(semantic) as image:
            valid = np.asarray(image.convert('L')) >= 128
        combined = (valid & (fov[record['camera']] > 0)).astype(np.uint8) * 255
        mask_path = output / 'masks' / (name + '.png')
        if mask_path.exists():
            with Image.open(mask_path) as image:
                if image.size != (scene['cameras'][record['camera']]['width'],
                                  scene['cameras'][record['camera']]['height']) or not np.array_equal(
                                      np.asarray(image), combined):
                    raise ValueError(f'Existing combined mask changed: {name}')
        else:
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = mask_path.with_suffix('.png.tmp')
            if temporary.exists():
                raise FileExistsError(temporary)
            Image.fromarray(combined).save(temporary, format='PNG', compress_level=1)
            temporary.replace(mask_path)
        _link_immutable(output / 'images' / name, source)
        images_and_masks.append({'name': name, 'image_sha256': file_digest(source),
                                 'source_semantic_mask_sha256': file_digest(semantic),
                                 'combined_mask_sha256': file_digest(mask_path)})

    identity = {
        'schema_version': 1, 'name': 'basketball_equal_control_computed',
        'status': 'new_run_candidate',
        'scene_manifest_sha256': file_digest(root / 'manifest.json'),
        'equal_ba_protocol_sha256': file_digest(protocol_path),
        'equal_ba_result_sha256': file_digest(result_path),
        'equal_ba_validation_sha256': file_digest(validation_path),
        'model_sha256': _model_hashes(model_path),
        'point_cloud_sha256': file_digest(output / 'sparse/0/points3D.ply'),
        'lfs_config_sha256': file_digest(output / 'lfs_config.json'),
        'originals_and_masks_sha256': digest(images_and_masks),
        'point_filter': point_info, 'image_count': len(records),
        'lenses': dict(Counter(row['camera'] for row in records.values())),
        'camera_model': 'OPENCV_FISHEYE',
        'independent_exposure_poses': True,
        'all_selected_images_train': True, 'test_split': False,
        'manual_annotations_used': False,
        'navigation_pose_values_or_las_used': False,
    }
    write_path = output / 'computed_baseline.json'
    _write_immutable(write_path, (json.dumps(identity, ensure_ascii=False, indent=2) + '\n').encode())
    verify_computed_baseline(work, output=output)
    return {'output': str(output), 'identity': str(write_path),
            'images': len(records), 'points': point_info['exported_points'],
            'manual_annotations_used': False}


def verify_computed_baseline(work, *, output=None):
    """Read-only content and camera-contract check of an automated export."""
    work = Path(work).resolve()
    output = Path(output).resolve() if output else work / 'baseline_colmap'
    identity_path = output / 'computed_baseline.json'
    identity = json.loads(identity_path.read_text())
    if (identity.get('schema_version') != 1
            or identity.get('status') != 'new_run_candidate'
            or identity.get('camera_model') != 'OPENCV_FISHEYE'
            or identity.get('manual_annotations_used') is not False
            or identity.get('navigation_pose_values_or_las_used') is not False
            or identity.get('independent_exposure_poses') is not True
            or identity.get('all_selected_images_train') is not True
            or identity.get('test_split') is not False):
        raise ValueError('Computed baseline contract differs')
    root = work / 'global_sfm'
    scene = load_scene(root)
    source_model = work / EQUAL / 'sparse/0'
    equal_protocol = json.loads((work / EQUAL / 'protocol.json').read_text())
    equal_result = json.loads((work / EQUAL / 'result.json').read_text())
    equal_validation = json.loads((work / EQUAL / 'validation_summary.json').read_text())
    if (equal_protocol.get('manual_annotations_used') is not False
            or equal_protocol.get('floor_classification_used') is not False
            or equal_result.get('manual_annotations_used') is not False
            or equal_validation.get('manual_annotations_used') is not False):
        raise ValueError('Equal BA provenance requires manual inputs')
    for name, actual in (
            ('scene_manifest_sha256', root / 'manifest.json'),
            ('equal_ba_protocol_sha256', work / EQUAL / 'protocol.json'),
            ('equal_ba_result_sha256', work / EQUAL / 'result.json'),
            ('equal_ba_validation_sha256', work / EQUAL / 'validation_summary.json'),
            ('point_cloud_sha256', output / 'sparse/0/points3D.ply'),
            ('lfs_config_sha256', output / 'lfs_config.json')):
        if file_digest(actual) != identity[name]:
            raise ValueError(f'Computed baseline file changed: {actual}')
    if _model_hashes(source_model) != identity['model_sha256']:
        raise ValueError('Computed source model changed')
    if _model_hashes(output / 'sparse/0') != identity['model_sha256']:
        raise ValueError('Exported COLMAP model changed')
    rec = pycolmap.Reconstruction(output / 'sparse/0')
    records = _check_model(rec, scene)
    if (len(records) != identity['image_count']
            or dict(Counter(row['camera'] for row in records.values())) != identity['lenses']):
        raise ValueError('Exported image membership changed')
    images_and_masks = []
    for name, record in sorted(records.items()):
        original = Path(record['source']).resolve()
        linked = output / 'images' / name
        semantic = Path(record['semantic_mask'])
        mask = output / 'masks' / (name + '.png')
        if (not linked.is_file() or linked.resolve() != original
                or not semantic.is_file() or not mask.is_file()):
            raise FileNotFoundError(f'Computed baseline image or mask missing: {name}')
        images_and_masks.append({'name': name, 'image_sha256': file_digest(original),
                                 'source_semantic_mask_sha256': file_digest(semantic),
                                 'combined_mask_sha256': file_digest(mask)})
    if digest(images_and_masks) != identity['originals_and_masks_sha256']:
        raise ValueError('Computed baseline image or mask content changed')
    return {'output': str(output), 'identity': str(identity_path),
            'images': len(records), 'model': str(output / 'sparse/0'),
            'manual_annotations_used': False, 'full_content_check': True}


def run_baseline(work, *, input_root=None, stage='all', output=None,
                 lfs_config=DEFAULT_CONFIG):
    """Run or resume the basketball baseline stages in dependency order."""
    stages = ('prepare', 'local-calibration', 'global-data', 'global-sfm',
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
        config_path = Path(lfs_config).resolve()
        if not config_path.is_file():
            raise FileNotFoundError(config_path)
        destination = Path(output).resolve() if output else work / 'baseline_colmap'
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
            from .local_calibration import (BASELINE_SEGMENTS, build_frontend,
                                            prepare_masks, solve_baseline_calibration)
            prepare_masks(work, segments=BASELINE_SEGMENTS)
            build_frontend(work, segments=BASELINE_SEGMENTS)
            solve_baseline_calibration(work)
        elif current == 'global-data':
            from .global_data import (extract_global_features,
                                      match_global_features, prepare_global)
            prepare_global(work)
            extract_global_features(work)
            match_global_features(work)
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
                work, output=output, lfs_config=lfs_config)
        summary['stages'].append(current)
    return summary
