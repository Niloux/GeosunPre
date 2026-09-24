"""Isolated dense court-image homography refinement; never writes SfM poses."""

from datetime import datetime, timezone
import json
from pathlib import Path

import cv2
import numpy as np
import pycolmap

from .geometry import stats
from .scene import digest, file_digest, load_scene, write_json
from .sfm_court_plane_probe import _pose_plane_fit, _virtual_pose
from .sfm_court_trajectory_probe import (OUTPUT as TRAJECTORY_OUTPUT,
                                         VIEW_SIZE, VIEW_FOV_DEG,
                                         _class_iou, _cycle_error,
                                         _view_classes,
                                         run_court_trajectory_graph_cycles)


OUTPUT = 'sfm_court_dense_probe'
ECC_MAX_ITERATIONS = 80
ECC_EPSILON = 1e-5
ECC_GAUSS_KERNEL = 5


def _court_grid(labels):
    yy, xx = np.mgrid[40:VIEW_SIZE-40:40, 40:VIEW_SIZE-40:40]
    xy = np.column_stack((xx.ravel(), yy.ravel())).astype(np.float32)
    return xy[labels[xy[:, 1].astype(int), xy[:, 0].astype(int)] > 0]


def _matrix_shift(a, b, labels):
    grid = _court_grid(labels)
    if not len(grid):
        return stats([])
    first = cv2.perspectiveTransform(grid[None], a)[0]
    second = cv2.perspectiveTransform(grid[None], b)[0]
    valid = (np.isfinite(first).all(axis=1) & np.isfinite(second).all(axis=1)
             & (first >= 0).all(axis=1) & (first < VIEW_SIZE).all(axis=1)
             & (second >= 0).all(axis=1) & (second < VIEW_SIZE).all(axis=1))
    return stats(np.linalg.norm(first[valid] - second[valid], axis=1))


def run_court_dense_probe(work):
    cv2.setNumThreads(4)
    work = Path(work).resolve()
    root, trajectory = work / 'global_sfm', work / 'global_sfm' / TRAJECTORY_OUTPUT
    graph = run_court_trajectory_graph_cycles(work)
    pair_result = json.loads((trajectory / 'pairs.json').read_text())
    selected = [row for row in pair_result['pairs']
                if row['status'] == 'fitted' and row['homography_inliers'] >= 20]
    scene = load_scene(root)
    records = {row['name']: row for row in scene['images']}
    view_paths = {name: trajectory / 'views'
                  / f'{records[name]["camera"]}{records[name]["frame_index"]:05d}.jpg'
                  for row in selected for name in row['images']}
    model_path = root / 'ground_joint_ba/equal_control/sparse/0'
    protocol = {'version': 1,
                'trajectory_pair_sha256': file_digest(trajectory / 'pairs.json'),
                'trajectory_graph_cycles_sha256': file_digest(trajectory / 'graph_cycles.json'),
                'view_sha256': {name: file_digest(path)
                                for name, path in sorted(view_paths.items())},
                'visual_model_hashes': {p.name: file_digest(p)
                                        for p in sorted(model_path.glob('*.bin'))},
                'input_edge': 'All fitted trajectory pairs with >=20 image-only MAGSAC inliers',
                'template': 'source 640-view grayscale JPEG',
                'input': 'target 640-view grayscale JPEG',
                'target_mask': 'HSV court-color connected components; rough, not segmentation truth',
                'ecc_motion': 'homography', 'ecc_iterations': ECC_MAX_ITERATIONS,
                'ecc_epsilon': ECC_EPSILON, 'ecc_gauss_kernel': ECC_GAUSS_KERNEL,
                'uses_pose_or_las_for_dense_alignment': False,
                'pose_used_only_for_post_fit_compatibility': True,
                'scope': 'Image-only dense registration and graph-cycle diagnostic. No feature, pose, calibration, point, or LFS training input changes.'}
    fingerprint = digest(protocol)
    output = root / OUTPUT
    output.mkdir(exist_ok=True)
    protocol_path = output / 'protocol.json'
    if protocol_path.exists():
        if json.loads(protocol_path.read_text())['fingerprint'] != fingerprint:
            raise ValueError('Existing court dense protocol belongs to different inputs')
    else:
        write_json(protocol_path, {**protocol, 'fingerprint': fingerprint,
                                   'frozen_at_utc': datetime.now(timezone.utc).isoformat()})
    result_path = output / 'results.json'
    if result_path.exists():
        old = json.loads(result_path.read_text())
        if old['protocol_fingerprint'] != fingerprint:
            raise ValueError('Changed court dense result inputs')
        return old
    reconstruction = pycolmap.Reconstruction(model_path)
    images = {image.name: image for image in reconstruction.images.values()}
    views = {}
    for name, path in view_paths.items():
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None or bgr.shape[:2] != (VIEW_SIZE, VIEW_SIZE):
            raise ValueError(f'Missing trajectory view: {path}')
        labels, _ = _view_classes(bgr, bgr.any(axis=2))
        views[name] = {'bgr': bgr, 'gray': cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY),
                       'labels': labels}
    rows = []
    by_pair = {}
    for index, input_row in enumerate(selected, 1):
        a, b = input_row['images']
        first, second = views[a], views[b]
        initial = np.asarray(input_row['homography'], dtype=np.float32)
        mask = (second['labels'] > 0).astype(np.uint8) * 255
        row = {'images': [a, b], 'cameras': input_row['cameras'],
               'initial_homography_inliers': input_row['homography_inliers'],
               'target_court_mask_pixels': int((mask > 0).sum()),
               'initial_saved_pose_plane_view_px':
               input_row.get('saved_pose_plane_audit', {})
               .get('fitted_plane_warp_disagreement_view_px')}
        try:
            coefficient, refined = cv2.findTransformECC(
                first['gray'], second['gray'], initial.copy(),
                cv2.MOTION_HOMOGRAPHY,
                (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS,
                 ECC_MAX_ITERATIONS, ECC_EPSILON), mask, ECC_GAUSS_KERNEL)
            refined = np.asarray(refined, dtype=np.float64)
            refined /= refined[2, 2]
            if not np.isfinite(refined).all():
                raise ValueError('Nonfinite ECC homography')
            shift = _matrix_shift(np.asarray(input_row['homography']), refined,
                                  first['labels'])
            if shift['count'] < 12:
                raise ValueError('Insufficient common court grid after ECC')
            row.update(status='converged', ecc_coefficient=float(coefficient),
                       refined_homography=refined.tolist(),
                       update_shift_view_px=shift,
                       initial_color_iou=_class_iou(first['labels'],
                                                    second['labels'],
                                                    np.asarray(input_row['homography'])),
                       refined_color_iou=_class_iou(first['labels'],
                                                    second['labels'], refined))
            try:
                _, pose = _pose_plane_fit(refined, first['labels'],
                                          _virtual_pose(images[a]),
                                          _virtual_pose(images[b]),
                                          view_size=VIEW_SIZE,
                                          view_fov_deg=VIEW_FOV_DEG)
                row['refined_saved_pose_plane_view_px'] = pose[
                    'fitted_plane_warp_disagreement_view_px']
            except (ValueError, RuntimeError) as exc:
                row['pose_audit_error'] = str(exc)
        except (cv2.error, ValueError) as exc:
            row.update(status='failed', error=str(exc))
        rows.append(row)
        by_pair[(a, b)] = row
        if index % 25 == 0:
            print(f'court dense registration {index}/{len(selected)}', flush=True)
    cycles = []
    for old_cycle in graph['triangles']:
        a, b, c = old_cycle['images']
        edges = [by_pair[(x, y)] for x, y in ((a, b), (a, c), (b, c))]
        row = {'images': [a, b, c], 'cameras': old_cycle['cameras'],
               'frames': old_cycle['frames'],
               'initial_cycle_view_px': old_cycle['cycle_error_view_px']}
        if all(edge['status'] == 'converged' for edge in edges):
            H = [np.asarray(edge['refined_homography']) for edge in edges]
            row['refined_cycle_view_px'] = _cycle_error(*H, views[a]['labels'])
        else:
            row['refined_cycle_error'] = 'At least one edge did not converge'
        cycles.append(row)
    result = {'protocol_fingerprint': fingerprint, 'pairs': rows,
              'graph_cycles': cycles,
              'guard': 'ECC optimizes grayscale appearance on the same court images. Better correlation, color IoU, or cycle closure does not prove physical point identity or a better 3DGS pose; repeated tiles and lighting can fool dense alignment. No observation or pose has been promoted.'}
    write_json(result_path, result)
    write_json(output / 'DIAGNOSTIC_ONLY.json', {
        'result_sha256': file_digest(result_path), 'status': 'DIAGNOSTIC_ONLY',
        'reason': 'Dense court registration is only an image-plane proposal and has no independent native-pixel point identity validation.'})
    return result


def run_court_dense_review(work):
    work = Path(work).resolve()
    root = work / 'global_sfm'
    output, trajectory = root / OUTPUT, root / TRAJECTORY_OUTPUT
    dense = run_court_dense_probe(work)
    baseline = json.loads((trajectory / 'pairs.json').read_text())
    records = {(r['camera'], r['frame_index']): r['name']
               for r in load_scene(root)['images']}
    initial = {tuple(row['images']): row for row in baseline['pairs']
               if row['status'] == 'fitted'}
    refined = {tuple(row['images']): row for row in dense['pairs']
               if row['status'] == 'converged'}
    examples = ((('L', 156), ('L', 158)),
                (('L', 230), ('L', 232)),
                (('L', 169), ('R', 249)),
                (('R', 188), ('R', 189)),
                (('L', 50), ('L', 52)))
    chosen = [(records[a], records[b]) for a, b in examples]
    width, height = 360, 395
    canvas = np.zeros((height * len(chosen), width * 4, 3), np.uint8)
    rows = []
    for index, (a, b) in enumerate(chosen):
        before, after = initial[(a, b)], refined[(a, b)]
        paths = [trajectory / 'views' / f'{name[0]}{int(name[-11:-6]):05d}.jpg'
                 for name in (a, b)]
        source, target = [cv2.imread(str(path)) for path in paths]
        if source is None or target is None:
            raise ValueError(f'Missing court dense review view: {paths}')
        overlays = []
        for H in (np.asarray(before['homography']),
                  np.asarray(after['refined_homography'])):
            warped = cv2.warpPerspective(source, H, (VIEW_SIZE, VIEW_SIZE))
            overlays.append(cv2.addWeighted(warped, .5, target, .5, 0))
        for column, image in enumerate((source, target, *overlays)):
            tile = cv2.resize(image, (width, width),
                              interpolation=cv2.INTER_AREA)
            canvas[index * height:index * height + width,
                   column * width:(column + 1) * width] = tile
        label = (f'{a[0]}{int(a[-11:-6])} to {b[0]}{int(b[-11:-6])}  '
                 f'ECC {after["ecc_coefficient"]:.2f}  '
                 f'overlap shift {after["update_shift_view_px"]["median"]:.1f}px')
        cv2.putText(canvas, label, (8, index * height + width + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, .6, (245, 245, 245), 1,
                    cv2.LINE_AA)
        rows.append({'images': [a, b], 'ecc_coefficient': after['ecc_coefficient'],
                     'overlap_grid_shift_view_px': after['update_shift_view_px']})
    review_dir = output / 'review'
    review_dir.mkdir(exist_ok=True)
    path = review_dir / 'selected_overlays.jpg'
    cv2.imwrite(str(path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
    result = {'dense_result_sha256': file_digest(output / 'results.json'),
              'image': str(path), 'image_sha256': file_digest(path),
              'columns': ['source', 'target', 'LoMa homography overlay',
                          'dense ECC overlay'], 'rows': rows}
    write_json(output / 'review.json', result)
    return result
