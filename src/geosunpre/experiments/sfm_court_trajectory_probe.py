"""Court-visible trajectory audit using cached native LoMa matches.

The temporary pinhole view is only a coordinate chart for a planar diagnostic.
All feature observations remain native fisheye pixels. Navigation poses and LAS
are never used to select pairs or estimate homographies.
"""

from datetime import datetime, timezone
from functools import lru_cache
import json
from pathlib import Path

import cv2
import numpy as np
import pycolmap
from scipy.spatial.transform import Rotation

from ..camera import Camera
from ..geometry import stats
from ..scene import digest, file_digest, load_scene, write_json
from .sfm_court_plane_probe import _pose_plane_fit


OUTPUT = 'sfm_court_trajectory_probe'
VIEW_SIZE = 640
VIEW_FOV_DEG = 80.
PITCH_DEG = -45.
SCAN_STRIDE = 10
SCAN_MIN_COMPONENT = 3000  # quarter-resolution native pixels
SELECT_MIN_COURT_PIXELS = 10000
VIEW_MIN_COMPONENT = 2000
MIN_MATCH_SCORE = .9
MIN_PAIR_POINTS = 20
RANSAC_THRESHOLD_VIEW_PX = 2.


def _hue_masks(bgr, open_size):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    chromatic = (s >= 80) & (v >= 60)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (open_size, open_size))
    return [cv2.morphologyEx((chromatic & hue).astype(np.uint8),
                             cv2.MORPH_OPEN, kernel)
            for hue in (((h <= 10) | (h >= 170)),
                        ((h >= 74) & (h <= 95)),
                        ((h >= 98) & (h <= 115)))]


def _components(mask, minimum):
    count, labels, regions, _ = cv2.connectedComponentsWithStats(mask, 8)
    accepted = [i for i in range(1, count)
                if regions[i, cv2.CC_STAT_AREA] >= minimum]
    kept = np.isin(labels, accepted).astype(np.uint8)
    return kept, int(kept.sum()), int(max((regions[i, cv2.CC_STAT_AREA]
                                          for i in range(1, count)), default=0))


def _native_coverage(record):
    bgr = cv2.imread(record['source'], cv2.IMREAD_REDUCED_COLOR_4)
    if bgr is None or bgr.shape[:2] != (1000, 750):
        raise ValueError(f'Missing quarter-resolution native image: {record["name"]}')
    components = {}
    for name, mask in zip(('red', 'green', 'blue'), _hue_masks(bgr, 7)):
        mask[:350] = 0
        _, pixels, largest = _components(mask, SCAN_MIN_COMPONENT)
        components[name] = {'pixels': pixels, 'largest_component': largest}
    return {'name': record['name'], 'camera': record['camera'],
            'frame_index': record['frame_index'], 'colors': components,
            'court_color_pixels': sum(c['pixels'] for c in components.values())}


def _selection(rows):
    first = min(row['frame_index'] for row in rows)
    groups = []
    for camera in ('L', 'R'):
        by_frame = {r['frame_index']: r for r in rows if r['camera'] == camera}
        for bin_start in range(first, max(by_frame) + 1, SCAN_STRIDE):
            candidates = []
            for frame in range(bin_start, bin_start + SCAN_STRIDE - 2):
                triple = [by_frame.get(frame + offset) for offset in range(3)]
                if all(triple):
                    # Favor three court-visible exposures, not a single colorful frame.
                    score = min(r['court_color_pixels'] for r in triple)
                    candidates.append((score, frame, triple))
            if not candidates:
                continue
            score, frame, triple = max(candidates,
                                       key=lambda entry: (entry[0], -entry[1]))
            if score >= SELECT_MIN_COURT_PIXELS:
                groups.append({'camera': camera, 'bin_start': bin_start,
                               'first_frame': frame, 'minimum_court_color_pixels': score,
                               'images': [r['name'] for r in triple]})
    return groups


def _write_protocol(path, value):
    fingerprint = digest(value)
    if path.exists():
        if json.loads(path.read_text())['fingerprint'] != fingerprint:
            raise ValueError(f'Existing court trajectory protocol changed: {path}')
    else:
        write_json(path, {**value, 'fingerprint': fingerprint,
                          'frozen_at_utc': datetime.now(timezone.utc).isoformat()})
    return fingerprint


def run_court_trajectory_scan(work):
    cv2.setNumThreads(4)
    root = Path(work).resolve() / 'global_sfm'
    scene = load_scene(root)
    output = root / OUTPUT
    output.mkdir(exist_ok=True)
    protocol = {'version': 1, 'scene_manifest_sha256': file_digest(root / 'manifest.json'),
                'source_size_and_mtime': {r['name']: [r['size_bytes'], r['mtime_ns']]
                                          for r in scene['images']},
                'scan_read_scale': 'JPEG reduced 4; 750x1000',
                'scan_y_start': 350, 'hsv_s_min': 80, 'hsv_v_min': 60,
                'hues_cv8': {'red': '0..10 or 170..179',
                             'green': '74..95', 'blue': '98..115'},
                'open_kernel': 7, 'minimum_component_pixels': SCAN_MIN_COMPONENT,
                'bin_stride_frames': SCAN_STRIDE,
                'minimum_triplet_court_color_pixels': SELECT_MIN_COURT_PIXELS,
                'selection': 'Maximum of the minimum HSV court-color area over each consecutive three-exposure camera-local triplet in a ten-frame bin. Color is a proposal, not segmentation truth.',
                'uses_navigation_pose_values_or_las': False}
    fingerprint = _write_protocol(output / 'scan_protocol.json', protocol)
    result_path = output / 'scan.json'
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if result['protocol_fingerprint'] != fingerprint:
            raise ValueError('Changed trajectory scan inputs')
        return result
    rows = []
    for index, record in enumerate(scene['images'], 1):
        rows.append(_native_coverage(record))
        if index % 100 == 0:
            print(f'court visibility {index}/{len(scene["images"])}', flush=True)
    groups = _selection(rows)
    result = {'protocol_fingerprint': fingerprint, 'images_scanned': len(rows),
              'rows': rows, 'groups': groups,
              'guard': 'HSV court colors can include similarly colored objects. Group selection is a visual proposal only; pair fitting requires actual LoMa overlap.'}
    write_json(result_path, result)
    return result


def _mapping(camera):
    focal = VIEW_SIZE / (2 * np.tan(np.deg2rad(VIEW_FOV_DEG / 2)))
    yy, xx = np.mgrid[:VIEW_SIZE, :VIEW_SIZE]
    local = np.stack(((xx + .5 - VIEW_SIZE / 2) / focal,
                      (yy + .5 - VIEW_SIZE / 2) / focal,
                      np.ones_like(xx)), axis=-1)
    rotation = Rotation.from_euler('x', PITCH_DEG, degrees=True).as_matrix()
    rays = local.reshape(-1, 3) @ rotation.T
    native = camera.project(rays).reshape(VIEW_SIZE, VIEW_SIZE, 2)
    valid = camera.valid(native.reshape(-1, 2), 179).reshape(VIEW_SIZE, VIEW_SIZE)
    valid &= rays[:, 2].reshape(VIEW_SIZE, VIEW_SIZE) > 0
    return (native[:, :, 0] - .5).astype(np.float32), (native[:, :, 1] - .5).astype(np.float32), valid


def _view_classes(bgr, valid):
    labels = np.zeros(bgr.shape[:2], np.uint8)
    selected = np.zeros_like(labels)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    for label, mask in enumerate(_hue_masks(bgr, 5), 1):
        mask[~valid] = 0
        kept, _, _ = _components(mask, VIEW_MIN_COMPONENT)
        labels[kept > 0] = label
        dilated = cv2.dilate(kept, kernel)
        selected[dilated > 0] = label
    return labels, selected


def _view_keypoints(camera, xy):
    rays = camera.rays(xy)
    rotation = Rotation.from_euler('x', PITCH_DEG, degrees=True).as_matrix()
    virtual = rays @ rotation
    focal = VIEW_SIZE / (2 * np.tan(np.deg2rad(VIEW_FOV_DEG / 2)))
    out = np.column_stack((focal * virtual[:, 0] / virtual[:, 2] + VIEW_SIZE / 2,
                           focal * virtual[:, 1] / virtual[:, 2] + VIEW_SIZE / 2))
    inside = ((virtual[:, 2] > 0) & np.isfinite(out).all(axis=1)
              & (out[:, 0] >= 0) & (out[:, 0] < VIEW_SIZE)
              & (out[:, 1] >= 0) & (out[:, 1] < VIEW_SIZE)
              & camera.valid(xy, 179))
    return out.astype(np.float32), inside


def _class_iou(a, b, H):
    shape = (VIEW_SIZE, VIEW_SIZE)
    valid = cv2.warpPerspective(np.ones_like(a), H, shape,
                                flags=cv2.INTER_NEAREST) > 0
    warped = cv2.warpPerspective(a, H, shape, flags=cv2.INTER_NEAREST)
    groups = {}
    for label, name in ((1, 'red'), (2, 'green'), (3, 'blue')):
        source = (warped == label) & valid
        target = (b == label) & valid
        union = source | target
        direct_source = a == label
        direct_target = b == label
        direct_union = direct_source | direct_target
        groups[name] = {'source_pixels': int(source.sum()),
                        'target_pixels': int(target.sum()),
                        'iou': float((source & target).sum() / union.sum()) if union.any() else None,
                        'identity_iou': float((direct_source & direct_target).sum()
                                              / direct_union.sum()) if direct_union.any() else None}
    return {'overlap_view_pixels': int(valid.sum()), 'classes': groups}


def _cycle_error(H_ab, H_ac, H_bc, source_class):
    yy, xx = np.mgrid[40:VIEW_SIZE-40:40, 40:VIEW_SIZE-40:40]
    xy = np.column_stack((xx.ravel(), yy.ravel())).astype(np.float32)
    xy = xy[source_class[xy[:, 1].astype(int), xy[:, 0].astype(int)] > 0]
    if not len(xy):
        return stats([])
    direct = cv2.perspectiveTransform(xy[None], H_ac)[0]
    via = cv2.perspectiveTransform(xy[None], H_bc @ H_ab)[0]
    valid = (np.isfinite(direct).all(axis=1) & np.isfinite(via).all(axis=1)
             & (direct >= 0).all(axis=1) & (direct < VIEW_SIZE).all(axis=1)
             & (via >= 0).all(axis=1) & (via < VIEW_SIZE).all(axis=1))
    return stats(np.linalg.norm(direct[valid] - via[valid], axis=1))


def run_court_trajectory_pairs(work):
    cv2.setNumThreads(4)
    work = Path(work).resolve()
    root, output = work / 'global_sfm', work / 'global_sfm' / OUTPUT
    scan = run_court_trajectory_scan(work)
    scene = load_scene(root)
    records = {r['name']: r for r in scene['images']}
    names = {name for group in scan['groups'] for name in group['images']}
    frontend = json.loads((root / 'frontend.json').read_text())
    model_path = root / 'ground_joint_ba/equal_control/sparse/0'
    protocol = {'version': 1, 'scan_protocol_fingerprint': scan['protocol_fingerprint'],
                'scan_sha256': file_digest(output / 'scan.json'),
                'frontend_sha256': file_digest(root / 'frontend.json'),
                'visual_model_hashes': {p.name: file_digest(p)
                                        for p in sorted(model_path.glob('*.bin'))},
                'selected_names': sorted(names), 'view_size': VIEW_SIZE,
                'view_fov_degrees': VIEW_FOV_DEG, 'view_pitch_degrees': PITCH_DEG,
                'view_min_component': VIEW_MIN_COMPONENT,
                'minimum_match_score': MIN_MATCH_SCORE,
                'minimum_pair_points': MIN_PAIR_POINTS,
                'homography_method': 'USAC_MAGSAC',
                'ransac_threshold_view_px': RANSAC_THRESHOLD_VIEW_PX,
                'pair_candidates': 'Existing LoMa retrieval/temporal graph, restricted to image-content-selected triplets. Every edge still needs actual court-local matched features.',
                'uses_navigation_pose_values_or_las': False,
                'saved_visual_poses_used_only_after_image_homography_fit': True}
    fingerprint = _write_protocol(output / 'pair_protocol.json', protocol)
    result_path = output / 'pairs.json'
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if result['protocol_fingerprint'] != fingerprint:
            raise ValueError('Changed court trajectory pair inputs')
        return result
    reconstruction = pycolmap.Reconstruction(model_path)
    cameras = {}
    for label, camera_id in (('L', 1), ('R', 2)):
        fitted = reconstruction.cameras[camera_id]
        cameras[label] = Camera(label, fitted.width, fitted.height,
                                tuple(fitted.params), (0, 0))
    mappings = {label: _mapping(camera) for label, camera in cameras.items()}
    views = {}
    for index, name in enumerate(sorted(names), 1):
        record = records[name]
        mx, my, valid = mappings[record['camera']]
        source = cv2.imread(record['source'], cv2.IMREAD_COLOR)
        if source is None:
            raise ValueError(f'Missing image: {record["source"]}')
        bgr = cv2.remap(source, mx, my, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT)
        bgr[~valid] = 0
        labels, selected = _view_classes(bgr, valid)
        path = output / 'views' / f'{record["camera"]}{record["frame_index"]:05d}.jpg'
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        views[name] = {'path': str(path), 'labels': labels, 'selected': selected}
        if index % 30 == 0:
            print(f'court views {index}/{len(names)}', flush=True)

    @lru_cache(maxsize=24)
    def feature(name):
        with np.load(frontend['features'][name]['path'], allow_pickle=False) as data:
            xy_native = data['keypoints']
        xy, inside = _view_keypoints(cameras[records[name]['camera']], xy_native)
        ij = np.floor(np.where(np.isfinite(xy), xy, 0)).astype(int)
        ij = np.clip(ij, 0, VIEW_SIZE - 1)
        court = views[name]['selected'][ij[:, 1], ij[:, 0]]
        court[~inside] = 0
        return xy, court

    poses = {image.name: image for image in reconstruction.images.values()}
    # The source and target exposures are independent, including L/R pairs.
    from .sfm_court_plane_probe import _virtual_pose
    virtual_poses = {name: _virtual_pose(poses[name]) for name in names}
    pairs = []
    candidates = [row for row in frontend['pairs']
                  if row['a'] in names and row['b'] in names]
    for index, entry in enumerate(candidates, 1):
        a, b = entry['a'], entry['b']
        xa, ca = feature(a)
        xb, cb = feature(b)
        with np.load(entry['path'], allow_pickle=False) as match:
            indices, scores = match['matches'], match['scores']
            if str(match['token']) != entry['token']:
                raise ValueError(f'Changed cached match: {a}, {b}')
        good = ((scores >= MIN_MATCH_SCORE)
                & (ca[indices[:, 0]] > 0)
                & (ca[indices[:, 0]] == cb[indices[:, 1]]))
        indices = indices[good]
        row = {'images': [a, b], 'cameras': records[a]['camera'] + records[b]['camera'],
               'frame_gap': records[b]['frame_index'] - records[a]['frame_index'],
               'reasons': entry['reasons'],
               'court_high_score_matches': int(len(indices))}
        if len(indices) < MIN_PAIR_POINTS:
            row['status'] = 'insufficient_court_matches'
            pairs.append(row)
            continue
        source, target = xa[indices[:, 0]], xb[indices[:, 1]]
        H, inliers = cv2.findHomography(source, target, cv2.USAC_MAGSAC,
                                        RANSAC_THRESHOLD_VIEW_PX,
                                        maxIters=10000, confidence=.999)
        if H is None or inliers is None:
            row['status'] = 'homography_failed'
            pairs.append(row)
            continue
        inliers = inliers.ravel().astype(bool)
        predicted = cv2.perspectiveTransform(source[None], H)[0]
        error = np.linalg.norm(predicted - target, axis=1)
        hull = (float(cv2.contourArea(cv2.convexHull(source[inliers])))
                if int(inliers.sum()) >= 3 else 0.)
        row.update(status='fitted', homography=H.tolist(),
                   homography_inliers=int(inliers.sum()),
                   inlier_fraction=float(inliers.mean()),
                   source_inlier_hull_view_px2=hull,
                   inlier_error_view_px=stats(error[inliers]))
        if int(inliers.sum()) >= 20 and hull >= 2000:
            row['color_alignment'] = _class_iou(views[a]['labels'],
                                                 views[b]['labels'], H)
            try:
                _, audit = _pose_plane_fit(H, views[a]['labels'],
                                           virtual_poses[a], virtual_poses[b],
                                           view_size=VIEW_SIZE,
                                           view_fov_deg=VIEW_FOV_DEG)
                row['saved_pose_plane_audit'] = audit
            except (ValueError, RuntimeError) as exc:
                row['saved_pose_plane_audit_error'] = str(exc)
        pairs.append(row)
        if index % 100 == 0:
            print(f'court pair audit {index}/{len(candidates)}', flush=True)
    by_pair = {(a, b): row for row in pairs
               for a, b in [row['images']] if row['status'] == 'fitted'}
    cycles = []
    for group in scan['groups']:
        a, b, c = group['images']
        edges = [by_pair.get((x, y)) for x, y in ((a, b), (a, c), (b, c))]
        if not all(edges):
            continue
        if min(e['homography_inliers'] for e in edges) < 20:
            continue
        H_ab, H_ac, H_bc = [np.asarray(e['homography']) for e in edges]
        cycles.append({'images': [a, b, c], 'camera': group['camera'],
                       'frame_start': group['first_frame'],
                       'cycle_error_view_px': _cycle_error(H_ab, H_ac, H_bc,
                                                            views[a]['labels'])})
    result = {'protocol_fingerprint': fingerprint,
              'selected_image_count': len(names), 'candidate_pair_count': len(candidates),
              'pairs': pairs, 'triplet_cycles': cycles,
              'guard': 'Only a diagnostic. Color masks are rough and repeated court tiles can support false point identities. Per-pair best-plane fit checks pose compatibility under current fitted intrinsics, not pose or calibration truth. No SfM/LFS input changed.'}
    write_json(result_path, result)
    return result


def _epipolar_angle_degrees(xy_a, xy_b, camera_a, camera_b,
                            source_pose, target_pose):
    ra, rb = source_pose[:3, :3], target_pose[:3, :3]
    relative_rotation = rb @ ra.T
    translation = target_pose[:3, 3] - relative_rotation @ source_pose[:3, 3]
    translation /= np.linalg.norm(translation)
    source_rays, target_rays = camera_a.rays(xy_a), camera_b.rays(xy_b)
    epipolar_normals = np.cross(translation, source_rays @ relative_rotation.T)
    normal_length = np.linalg.norm(epipolar_normals, axis=1)
    sine = np.abs(np.sum(target_rays * epipolar_normals, axis=1)) / np.maximum(normal_length, 1e-12)
    return np.rad2deg(np.arcsin(np.clip(sine, 0, 1)))


def _native_color_labels(record, xy):
    bgr = cv2.imread(record['source'], cv2.IMREAD_REDUCED_COLOR_4)
    if bgr is None:
        raise ValueError(f'Missing native image: {record["source"]}')
    h, s, v = cv2.split(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV))
    q = np.floor(xy / 4).astype(int)
    q[:, 0] = np.clip(q[:, 0], 0, bgr.shape[1] - 1)
    q[:, 1] = np.clip(q[:, 1], 0, bgr.shape[0] - 1)
    hue, saturation, brightness = h[q[:, 1], q[:, 0]], s[q[:, 1], q[:, 0]], v[q[:, 1], q[:, 0]]
    labels = np.zeros(len(xy), np.uint8)
    chromatic = (saturation >= 80) & (brightness >= 60)
    labels[chromatic & ((hue <= 10) | (hue >= 170))] = 1
    labels[chromatic & (hue >= 74) & (hue <= 95)] = 2
    labels[chromatic & (hue >= 98) & (hue <= 115)] = 3
    return labels


def run_court_trajectory_corroboration(work):
    """Check fitted pair candidates against saved tracks and epipolar geometry."""
    cv2.setNumThreads(4)
    work = Path(work).resolve()
    root, output = work / 'global_sfm', work / 'global_sfm' / OUTPUT
    result = run_court_trajectory_pairs(work)
    frontend = json.loads((root / 'frontend.json').read_text())
    scene = load_scene(root)
    model_path = root / 'ground_joint_ba/equal_control/sparse/0'
    protocol = {'version': 1, 'pair_result_sha256': file_digest(output / 'pairs.json'),
                'frontend_sha256': file_digest(root / 'frontend.json'),
                'scene_manifest_sha256': file_digest(root / 'manifest.json'),
                'visual_model_hashes': {p.name: file_digest(p)
                                        for p in sorted(model_path.glob('*.bin'))},
                'minimum_loma_score': MIN_MATCH_SCORE,
                'epipolar_unit': 'angle between matched target ray and saved-pose epipolar plane, degrees',
                'track_support': 'intersection of saved sparse-model point3D IDs in both images',
                'court_color_check': 'quarter-resolution native HSV same-class pixels; rough and separate from virtual-view fitting',
                'scope': 'Independent support audit of already fitted image-only homographies. Saved visual poses are candidate values, not input navigation truth.'}
    fingerprint = _write_protocol(output / 'corroboration_protocol.json', protocol)
    path = output / 'corroboration.json'
    if path.exists():
        previous = json.loads(path.read_text())
        if previous['protocol_fingerprint'] != fingerprint:
            raise ValueError('Changed court trajectory corroboration inputs')
        return previous
    reconstruction = pycolmap.Reconstruction(model_path)
    images = {image.name: image for image in reconstruction.images.values()}
    records = {r['name']: r for r in scene['images']}
    cameras = {label: Camera(label, reconstruction.cameras[camera_id].width,
                             reconstruction.cameras[camera_id].height,
                             tuple(reconstruction.cameras[camera_id].params), (0, 0))
               for label, camera_id in (('L', 1), ('R', 2))}
    catalog = {(entry['a'], entry['b']): entry for entry in frontend['pairs']}

    @lru_cache(maxsize=24)
    def feature(name):
        with np.load(frontend['features'][name]['path'], allow_pickle=False) as data:
            return data['keypoints']

    @lru_cache(maxsize=24)
    def point_ids(name):
        return {int(p.point3D_id) for p in images[name].points2D if p.has_point3D()}

    rows = []
    for fitted in result['pairs']:
        if fitted['status'] != 'fitted':
            continue
        a, b = fitted['images']
        entry = catalog[(a, b)]
        with np.load(entry['path'], allow_pickle=False) as match:
            if str(match['token']) != entry['token']:
                raise ValueError(f'Changed cached match: {a}, {b}')
            matched = match['matches'][match['scores'] >= MIN_MATCH_SCORE]
        xy_a, xy_b = feature(a)[matched[:, 0]], feature(b)[matched[:, 1]]
        angles = _epipolar_angle_degrees(xy_a, xy_b,
                                         cameras[records[a]['camera']],
                                         cameras[records[b]['camera']],
                                         images[a].cam_from_world().matrix(),
                                         images[b].cam_from_world().matrix())
        label_a = _native_color_labels(records[a], xy_a)
        label_b = _native_color_labels(records[b], xy_b)
        same_court_color = (label_a > 0) & (label_a == label_b)
        rows.append({'images': [a, b], 'cameras': fitted['cameras'],
                     'frame_gap': fitted['frame_gap'],
                     'fitted_court_matches': fitted['court_high_score_matches'],
                     'high_score_matches_all_native': int(len(matched)),
                     'same_court_color_matches_native': int(same_court_color.sum()),
                     'saved_model_shared_3d_tracks': len(point_ids(a) & point_ids(b)),
                     'saved_pose_epipolar_degrees_all': stats(angles),
                     'saved_pose_epipolar_degrees_same_court_color':
                     stats(angles[same_court_color]),
                     'homography_pose_best_plane_view_px':
                     fitted.get('saved_pose_plane_audit', {})
                     .get('fitted_plane_warp_disagreement_view_px')})
    output_result = {'protocol_fingerprint': fingerprint, 'rows': rows,
                     'guard': 'Sparse-model track overlap and epipolar compatibility corroborate a candidate but do not prove pixel identity. Missing shared tracks alone is inconclusive. A long-range match with only repeated-color court evidence, no shared track, and large epipolar error must not become a pose constraint.'}
    write_json(path, output_result)
    return output_result


def run_court_trajectory_graph_cycles(work):
    """Check all image-only homography triangles, including asynchronous L/R."""
    work = Path(work).resolve()
    root, output = work / 'global_sfm', work / 'global_sfm' / OUTPUT
    pairs = run_court_trajectory_pairs(work)
    scene = load_scene(root)
    rank = {record['name']: i for i, record in enumerate(scene['images'])}
    records = {record['name']: record for record in scene['images']}
    edges = {tuple(row['images']): row for row in pairs['pairs']
             if row['status'] == 'fitted' and row['homography_inliers'] >= 20}
    successors = {}
    for a, b in edges:
        successors.setdefault(a, set()).add(b)
    triangles = []
    for a in sorted(successors, key=rank.get):
        for b in sorted(successors[a], key=rank.get):
            for c in sorted(successors[a] & successors.get(b, set()),
                            key=rank.get):
                triangles.append((a, b, c))
    view_paths = {name: output / 'views'
                  / f'{records[name]["camera"]}{records[name]["frame_index"]:05d}.jpg'
                  for triple in triangles for name in triple}
    protocol = {'version': 1, 'pair_result_sha256': file_digest(output / 'pairs.json'),
                'scene_manifest_sha256': file_digest(root / 'manifest.json'),
                'view_sha256': {name: file_digest(path)
                                for name, path in sorted(view_paths.items())},
                'edge_threshold': 'fitted image-only homography with >=20 inliers',
                'grid': '40-pixel interior 640-view court-color grid; direct AC vs composed BC*AB',
                'uses_pose_or_las_for_cycle': False}
    fingerprint = _write_protocol(output / 'cycle_protocol.json', protocol)
    path = output / 'graph_cycles.json'
    if path.exists():
        result = json.loads(path.read_text())
        if result['protocol_fingerprint'] != fingerprint:
            raise ValueError('Changed court trajectory cycle inputs')
        return result

    @lru_cache(maxsize=24)
    def class_map(name):
        bgr = cv2.imread(str(view_paths[name]), cv2.IMREAD_COLOR)
        if bgr is None or bgr.shape[:2] != (VIEW_SIZE, VIEW_SIZE):
            raise ValueError(f'Missing review view: {view_paths[name]}')
        return _view_classes(bgr, bgr.any(axis=2))[0]

    rows = []
    for a, b, c in triangles:
        ab, ac, bc = [edges[edge] for edge in ((a, b), (a, c), (b, c))]
        error = _cycle_error(np.asarray(ab['homography']),
                             np.asarray(ac['homography']),
                             np.asarray(bc['homography']), class_map(a))
        rows.append({'images': [a, b, c],
                     'cameras': ''.join(records[name]['camera'] for name in (a, b, c)),
                     'frames': [records[name]['frame_index'] for name in (a, b, c)],
                     'edge_inliers': [ab['homography_inliers'], ac['homography_inliers'],
                                      bc['homography_inliers']],
                     'cycle_error_view_px': error})
    result = {'protocol_fingerprint': fingerprint, 'triangles': rows,
              'guard': 'Triangles check image-only plane-motion agreement, not truth. A sparse direct edge can deviate even when all three saved visual poses are acceptable. A coherent false repetitive-tile shift can still pass a triangle.'}
    write_json(path, result)
    return result


def run_court_trajectory_review(work):
    """Render the strongest anomaly and its well-supported comparisons."""
    work = Path(work).resolve()
    root, output = work / 'global_sfm', work / 'global_sfm' / OUTPUT
    pairs = run_court_trajectory_pairs(work)
    corroboration = run_court_trajectory_corroboration(work)
    cycles = run_court_trajectory_graph_cycles(work)
    support = {tuple(row['images']): row for row in corroboration['rows']}
    by_pair = {tuple(row['images']): row for row in pairs['pairs']
               if row['status'] == 'fitted'}
    scored = [row for row in by_pair.values() if 'saved_pose_plane_audit' in row]
    biggest = sorted(scored, key=lambda row:
                     -row['saved_pose_plane_audit']
                     ['fitted_plane_warp_disagreement_view_px']['median'])[:2]
    cycle_pairs = []
    for cycle in sorted(pairs['triplet_cycles'],
                        key=lambda row: -(row['cycle_error_view_px']['median'] or 0))[:2]:
        a, _, c = cycle['images']
        cycle_pairs.append(by_pair[(a, c)])
    cross = [row for row in scored if row['cameras'] in ('LR', 'RL')
             and support[tuple(row['images'])]['saved_model_shared_3d_tracks'] > 0
             and row['saved_pose_plane_audit']
             ['fitted_plane_warp_disagreement_view_px']['median'] < 1.5]
    comparison = max(cross, key=lambda row: row['homography_inliers']) if cross else None
    chosen = list({tuple(row['images']): row for row in
                   [*biggest, *cycle_pairs, *([comparison] if comparison else [])]}.values())
    review = output / 'review'
    review.mkdir(exist_ok=True)
    width, height = 360, 390
    montage = np.zeros((height * len(chosen), width * 3, 3), np.uint8)
    rows = []
    for index, row in enumerate(chosen):
        a, b = row['images']
        paths = [output / 'views' / f'{name[0]}{int(name[-11:-6]):05d}.jpg'
                 for name in (a, b)]
        source, target = (cv2.imread(str(path)) for path in paths)
        if source is None or target is None:
            raise ValueError(f'Missing court review view: {paths}')
        warped = cv2.warpPerspective(source, np.asarray(row['homography']),
                                     (VIEW_SIZE, VIEW_SIZE))
        overlay = cv2.addWeighted(warped, .5, target, .5, 0)
        for column, image in enumerate((source, target, overlay)):
            tile = cv2.resize(image, (width, width), interpolation=cv2.INTER_AREA)
            montage[index * height:index * height + width,
                    column * width:(column + 1) * width] = tile
        corroborated = support[(a, b)]
        label = (f'{a[0]}{int(a[-11:-6])} to {b[0]}{int(b[-11:-6])}  '
                 f'inliers {row["homography_inliers"]}  '
                 f'shared 3D {corroborated["saved_model_shared_3d_tracks"]}  '
                 f'plane {row["saved_pose_plane_audit"]["fitted_plane_warp_disagreement_view_px"]["median"]:.2f}px')
        cv2.putText(montage, label, (8, index * height + width + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, .55, (245, 245, 245), 1,
                    cv2.LINE_AA)
        rows.append({'images': [a, b], 'homography_inliers': row['homography_inliers'],
                     'saved_model_shared_3d_tracks': corroborated['saved_model_shared_3d_tracks'],
                     'pose_plane_median_view_px': row['saved_pose_plane_audit']
                     ['fitted_plane_warp_disagreement_view_px']['median']})
    montage_path = review / 'pair_overlays.jpg'
    cv2.imwrite(str(montage_path), montage, [cv2.IMWRITE_JPEG_QUALITY, 92])
    scene = load_scene(root)
    records = {(r['camera'], r['frame_index']): r for r in scene['images']}
    frames = (37, 75, 120, 150, 180, 210, 240, 270, 295)
    thumbnails = np.zeros((3 * 430, 6 * 300, 3), np.uint8)
    for index, (frame, camera) in enumerate((frame, camera)
                                            for frame in frames for camera in ('L', 'R')):
        record = records[(camera, frame)]
        bgr = cv2.imread(record['source'], cv2.IMREAD_REDUCED_COLOR_4)
        tile = cv2.resize(bgr, (300, 400), interpolation=cv2.INTER_AREA)
        row, column = divmod(index, 6)
        thumbnails[row * 430:row * 430 + 400,
                   column * 300:(column + 1) * 300] = tile
        cv2.putText(thumbnails, f'{camera}{frame}',
                    (column * 300 + 8, row * 430 + 422),
                    cv2.FONT_HERSHEY_SIMPLEX, .7, (245, 245, 245), 1,
                    cv2.LINE_AA)
    contact_path = review / 'court_visibility.jpg'
    cv2.imwrite(str(contact_path), thumbnails, [cv2.IMWRITE_JPEG_QUALITY, 90])
    result = {'pair_result_sha256': file_digest(output / 'pairs.json'),
              'corroboration_sha256': file_digest(output / 'corroboration.json'),
              'graph_cycles_sha256': file_digest(output / 'graph_cycles.json'),
              'graph_cycle_count': len(cycles['triangles']),
              'pair_overlay': str(montage_path),
              'pair_overlay_sha256': file_digest(montage_path),
              'visibility_contact': str(contact_path),
              'visibility_contact_sha256': file_digest(contact_path),
              'rows': rows}
    write_json(output / 'review.json', result)
    write_json(output / 'DIAGNOSTIC_ONLY.json', {
        'pair_result_sha256': result['pair_result_sha256'],
        'corroboration_sha256': result['corroboration_sha256'],
        'graph_cycles_sha256': result['graph_cycles_sha256'],
        'status': 'DIAGNOSTIC_ONLY',
        'reason': 'The sole extreme pose disagreement is an uncorroborated long-range court-texture match; local image homographies are mostly compatible with saved visual poses and do not isolate a pose/calibration error. No new pose or ground-track weights are promoted.'})
    return result
