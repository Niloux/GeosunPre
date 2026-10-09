"""Pose-initialized multiscale global similarity registration."""

from pathlib import Path

import numpy as np
import pycolmap

from geosunpre.artifacts import file_metadata, staged_directory, write_json
from geosunpre.colmap import model_files
from geosunpre.registration.geometry import (error_summary, matrix4, quantiles,
    refine_similarity, residuals, robust_similarity, spatial_split, transform_points)
from geosunpre.registration.las import sample_las
from geosunpre.registration.publication import complete_publication
from geosunpre.scene import load_poses, parse_name

MAX_LAS_POINTS = 1_500_000


def validate_output(output, *sources):
    output = Path(output).resolve()
    if any(output.is_relative_to(Path(source).resolve())
           or Path(source).resolve().is_relative_to(output) for source in sources):
        raise ValueError('Choose a registration output separate from all source data')
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError('Registration output already exists; choose a fresh directory')


def navigation_initialization(input_root, reconstruction):
    """Test an explicit vendor-frame hypothesis without shifting timestamps.

    ECEF/ENH rows are used ONLY to fit a coordinate conversion. Their differing
    timestamps are recorded, not reconciled or used to pair camera exposures.
    ENH minus its first position is a candidate LAS frame, never a verified CRS.
    """
    root = Path(input_root).resolve()
    files = sorted(root.glob('*_CameraPos_C2E.txt'))
    if len(files) != 1:
        raise ValueError('Expected exactly one CameraPos_C2E file for pose initialization')
    pose_path = files[0]
    stem = pose_path.name.removesuffix('_CameraPos_C2E.txt')
    nav_path, enh_path = root / f'{stem}_pos.txt', root / f'{stem}_pos_ENH.txt'
    nav, enh = np.loadtxt(nav_path), np.loadtxt(enh_path)
    if (nav.shape != enh.shape or nav.ndim != 2 or nav.shape[1] < 5 or len(nav) < 3
            or not np.isfinite(nav).all() or not np.isfinite(enh).all()
            or not np.array_equal(nav[:, 0], enh[:, 0])
            or len(np.unique(nav[:, 0])) != 1 or np.any(np.diff(nav[:, 1]) <= 0)
            or np.any(np.diff(enh[:, 1]) <= 0)):
        raise ValueError('Navigation rows cannot support the coordinate-conversion hypothesis')
    timestamp_difference = nav[:, 1] - enh[:, 1]
    if np.ptp(timestamp_difference) > .02:
        raise ValueError('Navigation row time differences vary too much')
    origin = enh[0, 2:5]
    nav_centre = nav[:, 2:5].mean(0)
    spread = np.linalg.svd(nav[:, 2:5] - nav_centre, compute_uv=False)
    if spread[1] < spread[0] * 1e-4:
        raise ValueError('Navigation coordinates are collinear; frame conversion is unobservable')
    conversion = pycolmap.estimate_sim3d(nav[:, 2:5] - nav_centre, enh[:, 2:5] - origin)
    if conversion is None:
        raise ValueError('Failed to estimate navigation coordinate conversion')
    centred_matrix = matrix4(conversion)
    conversion_matrix = centred_matrix.copy()
    conversion_matrix[:3, 3] -= centred_matrix[:3, :3] @ nav_centre
    conversion_errors = np.linalg.norm(transform_points(conversion_matrix, nav[:, 2:5])
                                       - (enh[:, 2:5] - origin), axis=1)
    if conversion_errors.max() > .02 or not .999 < conversion.scale < 1.001:
        raise ValueError('Navigation coordinate conversion does not fit')
    poses = load_poses(pose_path)
    centres, positions, exposure_offsets, names = [], [], [], []
    for image in sorted(reconstruction.images.values(), key=lambda image: image.name):
        name = Path(image.name).name
        if name not in poses:
            continue
        pose = poses[name]
        week, seconds, _, _ = parse_name(name)
        if pose['week'] != week or week != nav[0, 0]:
            raise ValueError(f'Camera/navigation GPS week mismatch: {name}')
        centres.append(image.projection_center())
        positions.append(pose['center_raw'])
        exposure_offsets.append(pose['seconds'] - seconds)
        names.append(image.name)
    positions = transform_points(conversion_matrix, positions)
    initial, pose_fit = robust_similarity(centres, positions)
    audit = {'kind': 'vendor_navigation_frame_hypothesis',
             'hypothesis': 'LAS XYZ approximately equals ENH minus first ENH navigation position',
             'vendor_coordinate_convention_verified': False, 'las_origin_verified': False,
             'row_association_scope': 'coordinate conversion only; not temporal synchronization',
             'timestamps_modified': False, 'quaternions_used': False,
             'navigation_row_time_difference_seconds': {
                 'min': float(timestamp_difference.min()), 'median': float(np.median(timestamp_difference)),
                 'max': float(timestamp_difference.max())},
             'camera_pose_minus_filename_seconds': {
                 'min': float(min(exposure_offsets)), 'median': float(np.median(exposure_offsets)),
                 'max': float(max(exposure_offsets))},
             'enh_origin': origin.tolist(), 'navigation_to_candidate_las_matrix': conversion_matrix.tolist(),
             'coordinate_conversion_error': quantiles(conversion_errors),
             'coordinate_conversion_scale': float(conversion.scale), 'camera_position_fit': pose_fit,
             'matched_image_names': names, 'unmatched_images': reconstruction.num_reg_images() - len(names),
             'input_files': {str(path): file_metadata(path) for path in (pose_path, nav_path, enh_path)}}
    return initial, audit


def projection_invariance(before, after):
    """Verify native projections as well as camera parameters and track identity."""
    if (set(before.points3D) != set(after.points3D) or set(before.images) != set(after.images)
            or set(before.frames) != set(after.frames)
            or before.compute_num_observations() != after.compute_num_observations()):
        raise ValueError('Similarity transform changed model topology')
    maximum, count = 0., 0
    for cid, camera in before.cameras.items():
        other = after.cameras[cid]
        if camera.model != other.model or not np.array_equal(camera.params, other.params):
            raise ValueError('Similarity transform changed camera intrinsics')
    for pid, point in before.points3D.items():
        if [(e.image_id, e.point2D_idx) for e in point.track.elements] != [
                (e.image_id, e.point2D_idx) for e in after.points3D[pid].track.elements]:
            raise ValueError('Similarity transform changed observation tracks')
    for image_id, image in before.images.items():
        other = after.images[image_id]
        if (image.name != other.name or image.camera_id != other.camera_id
                or image.frame_id != other.frame_id or image.num_points2D() != other.num_points2D()):
            raise ValueError('Similarity transform changed image identity')
        if any(a.point3D_id != b.point3D_id or not np.array_equal(a.xy, b.xy)
               for a, b in zip(image.points2D, other.points2D)):
            raise ValueError('Similarity transform changed 2D observations')
        point_ids = [point.point3D_id for point in image.points2D if point.has_point3D()]
        selected = np.unique(np.linspace(0, len(point_ids) - 1, min(20, len(point_ids))).astype(int))
        if not len(selected):
            continue
        ids = [point_ids[index] for index in selected]
        old_xyz = np.array([before.points3D[pid].xyz for pid in ids])
        new_xyz = np.array([after.points3D[pid].xyz for pid in ids])
        old_pose, new_pose = image.cam_from_world().matrix(), other.cam_from_world().matrix()
        old_local = old_xyz @ old_pose[:, :3].T + old_pose[:, 3]
        new_local = new_xyz @ new_pose[:, :3].T + new_pose[:, 3]
        old_pixels = before.cameras[image.camera_id].img_from_cam(old_local)
        new_pixels = after.cameras[other.camera_id].img_from_cam(new_local)
        if not np.isfinite(old_pixels).all() or not np.isfinite(new_pixels).all():
            raise ValueError('Nonfinite projection after similarity transform')
        maximum = max(maximum, float(np.max(np.linalg.norm(new_pixels - old_pixels, axis=1))))
        count += len(ids)
    if maximum > 1e-7 or not after.is_valid():
        raise ValueError('Similarity transform did not preserve visual projections')
    return {'sampled_observations': count, 'maximum_pixel_difference': maximum,
            'intrinsics_and_all_observation_tracks_unchanged': True}


def register_global(model_path, inputs, output, *, workers=4, context=None):
    """Register a completed visual model without exporting intermediate datasets."""
    model_path, output = Path(model_path).resolve(), Path(output).resolve()
    las = Path(inputs['las'])
    validate_output(output, model_path, inputs['input_root'], las)
    source_files = model_files(model_path)
    original = pycolmap.Reconstruction(model_path)
    initial, initialization = navigation_initialization(inputs['input_root'], original)
    target, las_metadata = sample_las(las, max_points=MAX_LAS_POINTS)
    point_ids = np.array(sorted(original.points3D), dtype=np.uint64)
    xyz = np.array([original.points3D[int(pid)].xyz for pid in point_ids])
    quality = np.array([original.points3D[int(pid)].error <= 2.
                        and original.points3D[int(pid)].track.length() >= 4 for pid in point_ids])
    partition_all = transform_points(initial, xyz)
    lower, upper = np.array(las_metadata['bounds_min']) - .8, np.array(las_metadata['bounds_max']) + .8
    eligible = quality & np.all((partition_all >= lower) & (partition_all <= upper), axis=1)
    source, selected_ids = xyz[eligible], point_ids[eligible]
    initial_points = partition_all[eligible]
    holdout, tiles, split = spatial_split(initial_points)
    if holdout.sum() < 100 or (~holdout).sum() < 100:
        raise ValueError('Insufficient fit/diagnostic points in the LAS overlap')
    print(f'Registration: {len(source)} visual points, {len(target)} LAS samples', flush=True)
    final, optimization, surface = refine_similarity(source[~holdout], target, initial,
                                                     tiles_fit=tiles[~holdout], workers=workers)
    final_points = transform_points(final, source)
    distances, plane, nearest = residuals(final_points, surface, workers=workers)
    supported = (distances <= .3) & surface['valid'][nearest]
    transformed = pycolmap.Reconstruction(original)
    transformed.transform(pycolmap.Sim3d(final[:3]))
    report = {'version': 1, 'inputs_file': inputs['inputs_file'],
              'inputs': {'model': str(model_path), 'las': str(las),
                         'visual_model_files': source_files, 'input_files': inputs['input_files']},
              'initialization': initialization, 'las': las_metadata,
              'split': split, 'optimization': optimization,
              'errors': {label: error_summary(final_points[mask], surface, workers=workers)
                         for label, mask in (('fit', ~holdout), ('diagnostic', holdout))},
              'scope': 'Fit-region delivery checks; no independent absolute accuracy validation.'}
    with staged_directory(output) as stage:
        model = stage / 'sparse/0'
        model.mkdir(parents=True)
        transformed.write(model)
        report['projection_invariance'] = projection_invariance(original, pycolmap.Reconstruction(model))
        write_json(stage / 'report.json', report)
        write_json(stage / 'transform.json', {'matrix': final.tolist(), 'initial_matrix': initial.tolist(),
                                            'report_file': file_metadata(stage / 'report.json')})
        np.savez_compressed(stage / 'registration_points.npz', point_ids=selected_ids, tiles=tiles,
                            holdout=holdout, partition_xyz=initial_points,
                            initial_xyz=initial_points, final_xyz=final_points,
                            nearest_distance=distances, signed_plane_error=plane,
                            stable_surface_supported=supported, surface_normals=surface['normals'][nearest])
        if model_files(model_path) != source_files or any(
                file_metadata(path) != expected for path, expected in inputs['input_files'].items()):
            raise ValueError('Registration sources changed during fitting')
        complete_publication(stage, context)
    return report
