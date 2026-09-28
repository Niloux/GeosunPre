"""Validate a PCC project without promoting vendor poses to reconstruction poses."""
from collections import Counter
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image

from geosunpre.camera import Camera
from geosunpre.artifacts import digest, file_digest, write_frozen_json

NAME = re.compile(r'(\d+)-(\d+)-(\d+)_(\d+)-([LRM])\.JPG', re.I)
CONTRACT_VERSION = 1


def parse_name(name):
    m = NAME.fullmatch(name)
    if not m:
        raise ValueError(f'Unrecognized image filename: {name}')
    week, seconds, ms, frame, camera = m.groups()
    return int(week), int(seconds)+int(ms)/1000, int(frame), camera.upper()


def parse_cameras(pro, principal_y='up'):
    cameras = {}
    for device in pro['cameraDevices']['devices']:
        label = device['deviceId'].removeprefix('camera_')
        if label not in ('L', 'R'):
            continue
        b, s = device['dataBasicParams'], device['dataSensorParams']
        pixel = float(b['pixelSize'])
        if pixel <= 0 or b['focalLength'] <= 0:
            raise ValueError(f'{label}: invalid focal length or pixel size')
        width, height = int(b['photoWidth']), int(b['photoHeight'])
        f = float(b['focalLength'])/pixel
        dx, dy = float(s['principalX'])/pixel, float(s['principalY'])/pixel
        D = tuple(float(s[k]) for k in ['k1', 'k2', 'k3', 'p3'])
        camera = Camera(label, width, height, (f, f, width/2, height/2, *D), (dx, dy))
        camera = camera.with_principal_signs(1, -1 if principal_y == 'up' else 1)
        if not np.isfinite(camera.params).all():
            raise ValueError(f'{label}: nonfinite calibration')
        cameras[label] = camera
    if set(cameras) != {'L', 'R'}:
        raise ValueError('Both camera_L and camera_R calibration are required')
    return cameras


def load_poses(path):
    poses = {}
    for line_number, line in enumerate(Path(path).read_text(encoding='utf-8-sig').splitlines(), 1):
        if not line.strip():
            continue
        v = line.split()
        if len(v) != 11:
            raise ValueError(f'Pose line {line_number}: expected 11 fields')
        if v[0] in poses:
            raise ValueError(f'Duplicate pose filename: {v[0]}')
        values = np.array(v[3:], dtype=np.float64)
        if not np.isfinite(values).all() or not .99 < np.linalg.norm(values[4:8]) < 1.01:
            raise ValueError(f'Invalid pose values: {v[0]}')
        poses[v[0]] = {'camera_id_raw': int(v[1]), 'week': int(v[2]), 'seconds': float(v[3]),
                       'center_raw': values[1:4].tolist(), 'quaternion_raw': values[4:8].tolist(),
                       'convention': 'unverified vendor C2E', 'is_ground_truth': False}
    return poses


def prepare(input_root, work):
    fov, principal_y = 160, 'up'
    root, work = Path(input_root).resolve(), Path(work).resolve()
    projects = sorted(root.glob('*.pro'))
    if len(projects) != 1:
        raise ValueError('Expected exactly one .pro project')
    pro_path = projects[0]
    pro = json.loads(pro_path.read_text(encoding='utf-8-sig'))
    cameras = parse_cameras(pro, principal_y)
    pose_path = root/f'{pro_path.stem}_CameraPos_C2E.txt'
    poses = load_poses(pose_path)
    images, names, excluded, frame_keys = [], set(), Counter(), set()
    for path in sorted((root/'Image').rglob('*')):
        if path.suffix.lower() not in ('.jpg', '.jpeg'):
            continue
        week, seconds, frame, camera = parse_name(path.name)
        if camera not in cameras:
            excluded['M_or_other_camera'] += 1
            continue
        if path.name not in poses:
            excluded['no_camera_pose'] += 1
            continue
        if path.name in names or (camera, frame) in frame_keys:
            raise ValueError(f'Ambiguous image identity: {path.name}')
        names.add(path.name)
        frame_keys.add((camera, frame))
        pose = poses[path.name]
        if pose['week'] != week:
            raise ValueError(f'Pose/image GPS week mismatch: {path.name}')
        with Image.open(path) as im:
            cam = cameras[camera]
            if im.size != (cam.width, cam.height) or im.getexif().get(274, 1) != 1:
                raise ValueError(f'Unexpected size or EXIF orientation: {path}')
            exposure = im.getexif().get(33434)
        relative = path.relative_to(root/'Image').as_posix()
        semantic_mask = None
        stat = path.stat()
        images.append({'name': relative, 'source': str(path), 'camera': camera, 'frame_index': frame,
                       'week': week, 'seconds': seconds, 'exposure_seconds': float(exposure) if exposure else None,
                       'size_bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
                       'semantic_mask': semantic_mask, 'prior': pose})
    missing = [name for name in poses if parse_name(name)[3] in cameras and name not in names]
    if missing or not images:
        raise ValueError(f'Pose-bound images missing: {missing[:8]}; selected count={len(images)}')
    images.sort(key=lambda im: (im['seconds'], im['camera'], im['name']))
    for i, im in enumerate(images):
        # Every exposure has its own frame; the frame_index in filenames is metadata only.
        im.update(image_id=i+1, exposure_id=i+1)
    manifest = {
        'contract_version': CONTRACT_VERSION, 'input_root': str(root), 'pose_model': 'independent_exposure',
        'selection': 'L/R with exact CameraPos filename only; M excluded', 'excluded': dict(excluded),
        'pixel_coordinates': 'corner coordinates; top-left pixel centre (0.5, 0.5)',
        'fov_degrees': fov, 'principal_y_axis': principal_y, 'principal_convention_verified': False,
        'distortion_mapping': ['k1', 'k2', 'k3', 'p3'], 'pose_values_used_for_geometry': False,
        'semantic_masks_available': False,
        'source_hashes': {'project': file_digest(pro_path), 'camera_pose': file_digest(pose_path)},
        'cameras': {k: c.to_dict() for k,c in cameras.items()}, 'images': images,
    }
    manifest['fingerprint'] = digest(manifest)
    write_frozen_json(work/'manifest.json', manifest)
    for label, cam in cameras.items():
        mask_path = work/'fov_masks'/f'{label}.png'
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(cam.mask(fov)).save(mask_path)
    (work/'images.txt').write_text('\n'.join(im['name'] for im in images)+'\n')
    return manifest


def load_scene(work):
    scene = json.loads((Path(work)/'manifest.json').read_text())
    if scene['contract_version'] != CONTRACT_VERSION or scene['pose_model'] != 'independent_exposure':
        raise ValueError('Incompatible prepared scene contract')
    return scene
