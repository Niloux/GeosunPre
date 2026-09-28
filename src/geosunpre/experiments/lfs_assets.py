"""Historical LichtFeld-Studio training inputs for downstream A/B experiments."""

import json
from pathlib import Path

import numpy as np


MAX_POINT_ERROR_PX = 2.0
MIN_POINT_TRACK = 4
MAX_POINT_CAMERA_DISTANCE = 50.0  # Visual-SfM units, not metres.


def filtered_points(rec, *, max_distance=MAX_POINT_CAMERA_DISTANCE):
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
        elif np.linalg.norm(xyz - center) > max_distance:
            rejected['distance'] += 1
        else:
            accepted.append((point_id, point))
    if len(accepted) < 10000:
        raise ValueError('Too few training-SfM points for LFS initialization')
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
                   'max_distance_from_median_camera_center_visual_units': max_distance},
    }


def training_config(template):
    """Turn a complete LFS template into the fixed all-training fisheye setup."""
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
