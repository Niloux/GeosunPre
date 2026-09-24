"""Export a controlled, all-training-view COLMAP A/B for LichtFeld-Studio."""

from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
import pycolmap

from ..camera import Camera
from ..scene import digest, file_digest, load_scene, write_json


BRANCHES = ('equal_control', 'near_floor_2p0')
MAX_POINT_ERROR_PX = 2.0
MIN_POINT_TRACK = 4
MAX_POINT_CAMERA_DISTANCE = 50.0  # Visual-SfM units, not metres.


def _link(path, target):
    path = Path(path)
    target = Path(target).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        if path.resolve() != target:
            raise ValueError(f'Existing link points elsewhere: {path}')
    elif path.exists():
        raise ValueError(f'Expected a link, found another file: {path}')
    else:
        path.symlink_to(os.path.relpath(target, path.parent),
                        target_is_directory=target.is_dir())


def _copy_immutable(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if file_digest(source) != file_digest(destination):
            raise ValueError(f'Existing export differs from source: {destination}')
        return
    temporary = destination.with_suffix(destination.suffix + '.tmp')
    shutil.copyfile(source, temporary)
    temporary.replace(destination)


def _write_immutable(path, content):
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f'Existing export differs: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_bytes(content)
    temporary.replace(path)


def _filtered_points(rec):
    centers = np.asarray([rec.images[i].projection_center()
                          for i in sorted(rec.reg_image_ids())], dtype=np.float64)
    center = np.median(centers, axis=0)
    accepted = []
    rejected = {'error': 0, 'track': 0, 'distance': 0, 'nonfinite': 0}
    for point_id in sorted(rec.points3D):
        point = rec.points3D[point_id]
        xyz = np.asarray(point.xyz, dtype=np.float64)
        if not np.isfinite(xyz).all() or not np.isfinite(point.error):
            rejected['nonfinite'] += 1
        elif point.error > MAX_POINT_ERROR_PX:
            rejected['error'] += 1
        elif point.track.length() < MIN_POINT_TRACK:
            rejected['track'] += 1
        elif np.linalg.norm(xyz - center) > MAX_POINT_CAMERA_DISTANCE:
            rejected['distance'] += 1
        else:
            accepted.append((point_id, point))
    if len(accepted) < 10000:
        raise ValueError('Too few training-SfM points for common initialization')
    vertices = np.empty(len(accepted), dtype=[('x', '<f4'), ('y', '<f4'),
                                              ('z', '<f4'), ('red', 'u1'),
                                              ('green', 'u1'), ('blue', 'u1')])
    for index, (_, point) in enumerate(accepted):
        vertices[index]['x'], vertices[index]['y'], vertices[index]['z'] = point.xyz
        vertices[index]['red'], vertices[index]['green'], vertices[index]['blue'] = point.color
    header = (f'ply\nformat binary_little_endian 1.0\n'
              f'element vertex {len(vertices)}\n'
              'property float x\nproperty float y\nproperty float z\n'
              'property uchar red\nproperty uchar green\nproperty uchar blue\n'
              'end_header\n').encode('ascii')
    return header + vertices.tobytes(), {
        'source_points': rec.num_points3D(), 'exported_points': len(vertices),
        'rejected': rejected, 'camera_center_median': center.tolist(),
        'source': 'equal_control visual SfM only; no navigation pose or LAS',
        'filter': {'max_reprojection_error_px': MAX_POINT_ERROR_PX,
                   'min_track_length': MIN_POINT_TRACK,
                   'max_distance_from_median_camera_center_visual_units': MAX_POINT_CAMERA_DISTANCE}}


def _training_config(template):
    # LFS's optimization JSON parser requires its complete field set. Reuse
    # the installed, scene-tuned template, but remove its unregistered LiDAR
    # seed and prior rig dataset paths before exporting a self-contained copy.
    config = json.loads(Path(template).read_text())
    if not isinstance(config.get('dataset'), dict) or not isinstance(config.get('optimization'), dict):
        raise ValueError('Expected a complete LFS dataset/optimization JSON template')
    config.pop('init_path', None)
    config['export_formats'] = []
    dataset = config['dataset']
    dataset.pop('data_path', None)
    dataset.pop('output_folder', None)
    dataset.pop('output_path', None)
    dataset.update(images='images', resize_factor=2, max_width=0,
                   min_track_length=0, use_test_split=False, test_every=8)
    opt = config['optimization']
    opt.update(strategy='mrnf', gut=True, iterations=30000, steps_scaler=1.0,
               sh_degree=3, max_cap=5000000, undistort=False,
               mask_mode='ignore', invert_masks=False, mask_threshold=0.5,
               use_exposure_correction=True, use_bilateral_grid=False,
               use_ppisp=False, enable_eval=True,
               eval_steps=[7000, 15000, 30000],
               save_steps=[7000, 15000, 30000],
               enable_save_eval_images=True)
    return config


def prepare_lfs_ab(work, lfs_template):
    root = Path(work).resolve() / 'global_sfm'
    scene = load_scene(root)
    out = root / 'lfs_ab'
    out.mkdir(parents=True, exist_ok=True)
    models = {branch: root / 'ground_joint_ba' / branch / 'sparse' / '0'
              for branch in BRANCHES}
    reconstructions = {branch: pycolmap.Reconstruction(path)
                       for branch, path in models.items()}
    records = {record['name']: record for record in scene['images']}
    if len(records) != 536 or any(record['camera'] not in ('L', 'R')
                                  for record in records.values()):
        raise ValueError('Expected exactly 536 pose-name selected L/R images')
    for branch, rec in reconstructions.items():
        names = {rec.images[i].name for i in rec.reg_image_ids()}
        if names != set(records) or rec.num_reg_images() != len(records):
            raise ValueError(f'{branch}: COLMAP image membership differs from manifest')
        if len(rec.cameras) != 2 or any(cam.model.name != 'OPENCV_FISHEYE'
                                        or (cam.width, cam.height) != (3000, 4000)
                                        for cam in rec.cameras.values()):
            raise ValueError(f'{branch}: expected two native 3000x4000 fisheye cameras')
        for image_id in rec.reg_image_ids():
            image = rec.images[image_id]
            if records[image.name]['camera'] != Path(image.name).parts[0]:
                raise ValueError(f'{branch}: image lens identity mismatch')
    baseline = reconstructions[BRANCHES[0]]
    if any(reconstructions[BRANCHES[1]].images[i].name != baseline.images[i].name
           for i in baseline.reg_image_ids()):
        raise ValueError('A/B image IDs do not refer to the same names')

    shared = out / 'shared'
    points_bytes, point_info = _filtered_points(baseline)
    _write_immutable(shared / 'points3D.ply', points_bytes)
    fov = {label: Camera.from_dict(scene['cameras'][label]).mask(scene['fov_degrees'])
           for label in ('L', 'R')}
    images_and_masks = []
    for name, record in sorted(records.items()):
        source = Path(record['source'])
        semantic = Path(record['semantic_mask'])
        if not source.is_file() or not semantic.is_file():
            raise FileNotFoundError(f'Missing original image or semantic mask: {name}')
        with Image.open(source) as im:
            if im.size != (3000, 4000):
                raise ValueError(f'Unexpected original image dimensions: {name}')
        with Image.open(semantic) as im:
            if im.size != (3000, 4000):
                raise ValueError(f'Unexpected semantic mask dimensions: {name}')
            semantic_valid = np.asarray(im.convert('L')) >= 128
        combined_path = shared / 'masks' / (name + '.png')
        combined = (semantic_valid & (fov[record['camera']] > 0)).astype(np.uint8) * 255
        if combined_path.exists():
            with Image.open(combined_path) as existing:
                if existing.size != (3000, 4000) or not np.array_equal(np.asarray(existing), combined):
                    raise ValueError(f'Existing combined mask changed: {name}')
        else:
            combined_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = combined_path.with_suffix('.png.tmp')
            Image.fromarray(combined).save(temporary, format='PNG', compress_level=1)
            temporary.replace(combined_path)
        _link(shared / 'images' / name, source)
        images_and_masks.append({'name': name, 'image_sha256': file_digest(source),
                                 'source_semantic_mask_sha256': file_digest(semantic),
                                 'combined_mask_sha256': file_digest(combined_path)})
    for branch, source_model in models.items():
        destination = out / branch
        _link(destination / 'images', shared / 'images')
        _link(destination / 'masks', shared / 'masks')
        sparse = destination / 'sparse' / '0'
        for filename in ('cameras.bin', 'images.bin', 'rigs.bin', 'frames.bin'):
            if (source_model / filename).exists():
                _copy_immutable(source_model / filename, sparse / filename)
        # The weighted BA also moved its SfM points. LFS loads the common PLY,
        # but make the COLMAP binary fallback common too so a loader option
        # cannot silently change the initialization between branches.
        _copy_immutable(models[BRANCHES[0]] / 'points3D.bin', sparse / 'points3D.bin')
        _link(sparse / 'points3D.ply', shared / 'points3D.ply')
    config_bytes = (json.dumps(_training_config(lfs_template), ensure_ascii=False, indent=2) + '\n').encode()
    _write_immutable(out / 'lfs_config.json', config_bytes)
    protocol = {'version': 1, 'scene_fingerprint': scene['fingerprint'],
                'images': len(records),
                'lenses': dict(Counter(record['camera'] for record in records.values())),
                'originals_and_masks_sha256': digest(images_and_masks),
                'same_originals_masks_and_init_points_for_both_branches': True,
                'common_point_cloud_sha256': file_digest(shared / 'points3D.ply'),
                'common_colmap_points3D_bin_sha256': file_digest(models[BRANCHES[0]] / 'points3D.bin'),
                'common_points': point_info,
                'models': {branch: {name: file_digest(path / name)
                                    for name in ('cameras.bin', 'images.bin', 'points3D.bin')}
                           for branch, path in models.items()},
                'lfs_config_sha256': file_digest(out / 'lfs_config.json'),
                'lfs_template_sha256': file_digest(lfs_template),
                'all_selected_images_train': True, 'test_split': False,
                'evaluation': 'Training-view diagnostics only; not independent holdout',
                'mask': 'Original semantic sky/person/device validity intersected with the same 160-degree candidate FOV for both A/B; not hardware ground truth',
                'camera_model': 'Native COLMAP OPENCV_FISHEYE, independent exposure poses',
                'input_navigation_pose_values_or_las_used': False}
    protocol['fingerprint'] = digest(protocol)
    path = out / 'protocol.json'
    if path.exists():
        old = json.loads(path.read_text())
        if {k: v for k, v in old.items() if k != 'created_at_utc'} != protocol:
            raise ValueError('LFS A/B input protocol changed; use a new output directory')
    else:
        write_json(path, {**protocol, 'created_at_utc': datetime.now(timezone.utc).isoformat()})
    return {'output': str(out), 'images': len(records),
            'initialization_points': point_info['exported_points'],
            'protocol_fingerprint': protocol['fingerprint']}
