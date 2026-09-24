"""Bounded image-only LoMa re-sampling around a distinctive court paint patch."""

from dataclasses import asdict
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path

import cv2
import numpy as np

from ..frontend import FrontendConfig, LoMaFrontend, balanced_selection
from ..scene import digest, file_digest, load_scene, write_json


OUTPUT = 'sfm_l150_crop_loma_probe'
FRAMES = (151, 152, 153, 154, 155)
CROP_CENTERS = {151: (2079, 2197), 152: (1915, 2289), 153: (1735, 2358),
                154: (1463, 2536), 155: (1055, 2700)}
CROP_SIDE = 768
REVIEW_RADIUS_NATIVE_PX = 160
CONFIG = FrontendConfig(detector_size=1024, num_keypoints=1536,
                        oversample=2, grid_size=6, match_threshold=.1)
REVIEWED = (
    (4, 'white_patch_edge_a', 'patch_candidate',
     'Track stays on the persistent white patch edge, but broad worn paint limits pixel precision.'),
    (5, 'white_patch_edge_b', 'patch_candidate',
     'Nearby white-patch track with a different center; correlated with edge A.'),
    (7, 'red_green_boundary', 'context_only',
     'Boundary point has no distinct local endpoint; repeated paint edge is ambiguous.'),
    (1, 'red_tile_a', 'rejected',
     'Ten-edge closure lands on repeated red tile texture without a unique pixel mark.'),
    (2, 'red_tile_b', 'rejected',
     'Second high-score red tile track also lacks a unique physical pixel mark.'),
)


def _protocol(path, payload):
    fingerprint = digest(payload)
    if path.exists():
        if json.loads(path.read_text())['fingerprint'] != fingerprint:
            raise ValueError(f'Frozen crop LoMa inputs changed: {path}')
    else:
        write_json(path, {**payload, 'fingerprint': fingerprint,
                          'frozen_at_utc': datetime.now(timezone.utc).isoformat()})
    return fingerprint


def _crop(output, record, frame):
    center = CROP_CENTERS[frame]
    half = CROP_SIDE // 2
    image = cv2.imread(record['source'], cv2.IMREAD_COLOR)
    x, y = center
    crop = image[y-half:y+half, x-half:x+half]
    if crop.shape[:2] != (CROP_SIDE, CROP_SIDE):
        raise ValueError(f'L{frame}: native crop meets image border')
    path = output / 'views' / f'L{frame:05d}.png'
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        cv2.imwrite(str(path), crop)
    elif not np.array_equal(cv2.imread(str(path)), crop):
        raise ValueError(f'Frozen LoMa crop differs from source: {path}')
    return {'frame': frame, 'source_name': record['name'], 'source_path': record['source'],
            'view_path': str(path), 'view_sha256': file_digest(path),
            'native_origin_xy': [x-half, y-half],
            'coarse_patch_center_native_xy': list(center)}


def _feature(output, view, frontend, fingerprint):
    frame = view['frame']
    token = digest({'protocol': fingerprint, 'frame': frame,
                    'view_sha256': view['view_sha256']})
    path = output / 'features' / f'L{frame:05d}.npz'
    if path.exists():
        with np.load(path, allow_pickle=False) as data:
            if str(data['token']) != token:
                raise ValueError(f'Changed crop feature cache: {path}')
            return {key: data[key] for key in data.files}
    import torch
    model = frontend.load_model()
    with torch.inference_mode():
        result = model._detector.detect_from_path(
            view['view_path'], num_keypoints=CONFIG.num_keypoints * CONFIG.oversample)
        normalized = result['keypoints'][0].float().cpu().numpy()
        scores = result['keypoint_probs'][0].float().cpu().numpy().reshape(-1)
        xy = (normalized + 1) * CROP_SIDE / 2
        inside = np.all((xy >= 8) & (xy < CROP_SIDE - 8), axis=1)
        candidates = np.flatnonzero(inside)
        chosen = balanced_selection(xy[candidates], scores[candidates],
                                    CONFIG.num_keypoints, CROP_SIDE, CROP_SIDE,
                                    CONFIG.grid_size)
        indices = candidates[chosen]
        selected = result['keypoints'][:, torch.as_tensor(indices, device=frontend.device)]
        descriptors = model._descriptor.describe_keypoints_from_path(
            view['view_path'], selected)['descriptions'][0].float().cpu().numpy()
    native = xy[indices] + np.asarray(view['native_origin_xy'])
    feature = {'keypoints': xy[indices].astype(np.float32),
               'native_xy': native.astype(np.float32),
               'normalized_keypoints': normalized[indices].astype(np.float32),
               'descriptors': descriptors.astype(np.float32),
               'scores': scores[indices].astype(np.float32),
               'token': np.array(token)}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp.npz')
    np.savez_compressed(tmp, **feature)
    tmp.replace(path)
    return feature


def _exact_cycles(frames, matches):
    a, b, c = frames
    ab, ac, bc = (matches[a, b], matches[a, c], matches[b, c])
    return [(i, j, ac[i]) for i, j in ab.items()
            if i in ac and bc.get(j) == ac[i]]


def _five_chains(matches):
    rows = []
    for first in matches[151, 152]:
        indices = [first]
        for a, b in zip(FRAMES, FRAMES[1:]):
            nxt = matches[a, b].get(indices[-1])
            if nxt is None:
                break
            indices.append(nxt)
        if len(indices) == 5:
            rows.append(tuple(indices))
    return rows


def _closure(indices, matches, max_gap):
    by_frame = dict(zip(FRAMES, indices))
    return all(matches[a, b].get(by_frame[a]) == by_frame[b]
               for a, b in itertools.combinations(FRAMES, 2)
               if b-a <= max_gap)


def _review(output, rows, features, names, records):
    if not rows:
        return None
    shown = rows[:min(20, len(rows))]
    crop_size, stride = 160, 190
    canvas = np.full((len(shown)*stride, len(FRAMES)*stride, 3), 24, np.uint8)
    images = {frame: cv2.imread(records[names[frame]]['source']) for frame in FRAMES}
    for row, indices in enumerate(shown):
        for col, (frame, index) in enumerate(zip(FRAMES, indices)):
            x, y = np.rint(features[frame]['native_xy'][index]).astype(int)
            patch = images[frame][y-80:y+80, x-80:x+80]
            if patch.shape[:2] != (crop_size, crop_size):
                raise ValueError(f'L{frame} review crop meets border')
            xx, yy = col*stride, row*stride
            canvas[yy:yy+crop_size, xx:xx+crop_size] = patch
            cv2.drawMarker(canvas, (xx+80, yy+80), (0, 0, 255), cv2.MARKER_CROSS, 16, 2)
            cv2.putText(canvas, f'L{frame} i{index}', (xx+2, yy+180),
                        cv2.FONT_HERSHEY_SIMPLEX, .45, (245, 245, 245), 1)
    path = output / 'review/five_view_candidates.jpg'
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 94])
    return {'path': str(path), 'sha256': file_digest(path), 'rows_shown': len(shown)}


def run_l150_crop_loma_probe(work):
    cv2.setNumThreads(4)
    work = Path(work).resolve()
    root = work / 'global_sfm'
    output = root / OUTPUT
    output.mkdir(exist_ok=True)
    scene = load_scene(root)
    records = {record['name']: record for record in scene['images']}
    names = {frame: next(name for name in records
                         if name.startswith('L/') and f'_{frame:05d}-L' in name)
             for frame in FRAMES}
    protocol = {
        'version': 1, 'frames': FRAMES, 'crop_centers_native_xy': CROP_CENTERS,
        'crop_side_native_px': CROP_SIDE, 'review_radius_native_px': REVIEW_RADIUS_NATIVE_PX,
        'source_sha256': {names[frame]: file_digest(records[names[frame]]['source'])
                          for frame in FRAMES},
        'config': asdict(CONFIG),
        'selection': 'Coarse centers of persistent white paint patch beside red/green court boundary; crop pixels remain native fisheye',
        'uses_pose_intrinsics_navigation_or_las': False,
        'scope': 'Bounded image-only crop LoMa proposal; no full frontend or SfM/LFS modification.'}
    fingerprint = _protocol(output / 'protocol.json', protocol)
    views = {frame: _crop(output, records[names[frame]], frame) for frame in FRAMES}
    write_json(output / 'views.json', {'protocol_fingerprint': fingerprint,
                                       'views': list(views.values())})
    frontend = LoMaFrontend(output / 'isolated_frontend_cache', scene, CONFIG)
    features = {}
    for frame in FRAMES:
        features[frame] = _feature(output, views[frame], frontend, fingerprint)
        print(f'Crop LoMa feature L{frame}: {len(features[frame]["native_xy"])}', flush=True)
    matches, scores, pair_summary = {}, {}, []
    for a, b in itertools.combinations(FRAMES, 2):
        result = frontend.match(features[a], features[b])
        index_pairs = result['matches']
        mapping = {int(i): int(j) for i, j in index_pairs}
        if len(mapping) != len(index_pairs) or len(set(mapping.values())) != len(mapping):
            raise ValueError(f'Non-unique crop LoMa pair: L{a}-L{b}')
        matches[a, b] = mapping
        scores[a, b] = {int(i): float(s) for (i, _), s in zip(index_pairs, result['scores'])}
        pair_summary.append({'frames': (a, b), 'matches': len(mapping),
                             'score_ge_0p9': sum(s >= .9 for s in scores[a, b].values()),
                             'match_token': str(result['token'])})
        print(f'Crop LoMa match L{a}-L{b}: {len(mapping)}', flush=True)
    triplets = []
    for first in (151, 152, 153):
        frames = (first, first+1, first+2)
        cycles = _exact_cycles(frames, matches)
        near = [row for row in cycles if np.linalg.norm(
            features[first]['native_xy'][row[0]]-CROP_CENTERS[first]) < REVIEW_RADIUS_NATIVE_PX]
        triplets.append({'frames': frames, 'exact_three_edge_cycles': len(cycles),
                         'near_white_patch_cycles': len(near)})
    chains = _five_chains(matches)
    near_chains = [row for row in chains if np.linalg.norm(
        features[151]['native_xy'][row[0]]-CROP_CENTERS[151]) < REVIEW_RADIUS_NATIVE_PX]
    two_hop = [row for row in near_chains if _closure(row, matches, 2)]
    all_edges = [row for row in near_chains if _closure(row, matches, 4)]
    ranked = sorted(two_hop, key=lambda row: min(
        scores[a, b][row[FRAMES.index(a)]] for a, b in itertools.combinations(FRAMES, 2)
        if b-a <= 2), reverse=True)
    candidate_rows = []
    for rank, indices in enumerate(ranked, 1):
        edge_scores = {f'L{a}-L{b}': scores[a, b][indices[FRAMES.index(a)]]
                       for a, b in itertools.combinations(FRAMES, 2)
                       if matches[a, b].get(indices[FRAMES.index(a)]) == indices[FRAMES.index(b)]}
        candidate_rows.append({
            'rank': rank, 'all_ten_edges_closed': _closure(indices, matches, 4),
            'min_two_hop_score': min(value for key, value in edge_scores.items()
                                     if int(key.split('-L')[1])-int(key.split('-L')[0][1:]) <= 2),
            'edge_scores': edge_scores,
            'observations': [{'frame': frame, 'image': names[frame],
                              'crop_feature_index': index,
                              'native_xy': features[frame]['native_xy'][index].tolist()}
                             for frame, index in zip(FRAMES, indices)]})
    result = {'protocol_fingerprint': fingerprint,
              'feature_count': {f'L{frame}': len(features[frame]['native_xy']) for frame in FRAMES},
              'pairs': pair_summary, 'triplets': triplets,
              'five_view': {'adjacent_chains': len(chains),
                            'near_white_patch_adjacent_chains': len(near_chains),
                            'near_white_patch_two_hop_closed': len(two_hop),
                            'near_white_patch_all_edges_closed': len(all_edges)},
              'candidate_rows': candidate_rows,
              'review': _review(output, ranked, features, names, records),
              'guard': 'Crop centers are coarse image-only regions, not precise points. High match scores and graph closure cannot prove physical identity or native-pixel accuracy; no candidate is promoted without original-image review.'}
    write_json(output / 'results.json', result)
    write_json(output / 'runtime_cache_stats.json', {'protocol_fingerprint': fingerprint,
                                                    'cache_stats': frontend.cache_stats})
    write_json(output / 'DIAGNOSTIC_ONLY.json', {
        'status': 'DIAGNOSTIC_ONLY', 'results_sha256': file_digest(output / 'results.json'),
        'reason': 'Image-only local feature proposal; no verified SfM training or acceptance tracks.'})
    return result


def _selected_review(output, selected, records):
    side, stride = 320, 342
    canvas = np.full((len(selected)*stride, len(FRAMES)*stride, 3), 24, np.uint8)
    images = {frame: cv2.imread(records[selected[0]['observations'][frame-FRAMES[0]]['image']]['source'])
              for frame in FRAMES}
    for row_index, row in enumerate(selected):
        for col, obs in enumerate(row['observations']):
            x, y = np.rint(obs['native_xy']).astype(int)
            crop = images[obs['frame']][y-160:y+160, x-160:x+160]
            if crop.shape[:2] != (side, side):
                raise ValueError(f'Review crop meets border: {row["id"]}')
            xx, yy = col*stride, row_index*stride
            canvas[yy:yy+side, xx:xx+side] = crop
            cv2.drawMarker(canvas, (xx+160, yy+160), (0, 0, 255), cv2.MARKER_CROSS, 24, 2)
            cv2.putText(canvas, f'{row["id"]} L{obs["frame"]}', (xx+3, yy+335),
                        cv2.FONT_HERSHEY_SIMPLEX, .48, (245, 245, 245), 1)
    path = output / 'review/frozen_selected_tracks.jpg'
    cv2.imwrite(str(path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return {'path': str(path), 'sha256': file_digest(path)}


def run_l150_crop_selection(work):
    root = Path(work).resolve() / 'global_sfm'
    output = root / OUTPUT
    candidate_path = output / 'results.json'
    if not candidate_path.exists():
        raise FileNotFoundError(f'Run crop LoMa proposal first: {candidate_path}')
    proposal = json.loads(candidate_path.read_text())
    protocol = {'version': 1, 'proposal_sha256': file_digest(candidate_path),
                'reviewed': REVIEWED,
                'selection': 'Native-image crop review after image-only LoMa closure, before reading any visual pose',
                'uses_pose_intrinsics_navigation_or_las': False,
                'scope': 'Freeze image observations only; no SfM or LFS changes.'}
    fingerprint = _protocol(output / 'selection_protocol.json', protocol)
    path = output / 'selection.json'
    if path.exists():
        previous = json.loads(path.read_text())
        if previous['protocol_fingerprint'] != fingerprint:
            raise ValueError('Existing crop selection belongs to changed inputs')
        return previous
    rows = {row['rank']: row for row in proposal['candidate_rows']}
    selected = []
    for rank, identity, role, note in REVIEWED:
        if rank not in rows:
            raise ValueError(f'Missing crop LoMa candidate rank {rank}')
        row = rows[rank]
        selected.append({'id': identity, 'role': role, 'candidate_rank': rank,
                         'image_review_note': note,
                         'all_ten_edges_closed': row['all_ten_edges_closed'],
                         'min_two_hop_score': row['min_two_hop_score'],
                         'observations': row['observations']})
    records = {row['name']: row for row in load_scene(root)['images']}
    result = {'protocol_fingerprint': fingerprint, 'selected': selected,
              'review': _selected_review(output, selected, records),
              'guard': 'The two white-patch tracks share a small scene region and worn paint. They are candidates, not precise independent 3D points. Repeated red tile tracks are rejected despite ten-edge closure.'}
    write_json(path, result)
    return result


def run_l150_crop_audit(work):
    import pycolmap

    from ..camera import Camera
    from .sfm_court_anchor_probe import _triangulate
    from .sfm_court_native_anchor_probe import _project, _world_ray

    root = Path(work).resolve() / 'global_sfm'
    output = root / OUTPUT
    selection_path = output / 'selection.json'
    if not selection_path.exists():
        raise FileNotFoundError(f'Freeze image selection first: {selection_path}')
    selection = json.loads(selection_path.read_text())
    model_path = root / 'ground_joint_ba/equal_control/sparse/0'
    protocol = {'version': 1, 'selection_sha256': file_digest(selection_path),
                'visual_model_hashes': {path.name: file_digest(path)
                                        for path in sorted(model_path.glob('*.bin'))},
                'support_pairs': ((151, 155), (152, 154)),
                'camera_model': 'Saved equal-control visual SfM and candidate L OPENCV_FISHEYE, not truth',
                'uses_navigation_pose_values_or_las': False,
                'scope': 'Fixed-pose diagnostic only; no BA, calibration or LFS change.'}
    fingerprint = _protocol(output / 'audit_protocol.json', protocol)
    path = output / 'audit.json'
    if path.exists():
        previous = json.loads(path.read_text())
        if previous['protocol_fingerprint'] != fingerprint:
            raise ValueError('Existing crop audit belongs to changed inputs')
        return previous
    model = pycolmap.Reconstruction(model_path)
    images = {image.name: image for image in model.images.values()}
    data = model.cameras[1]
    camera = Camera('L', data.width, data.height, tuple(data.params), (0, 0))
    rows = []
    for item in selection['selected']:
        observations = {obs['frame']: obs for obs in item['observations']}
        near_model = []
        for frame in FRAMES:
            obs = observations[frame]
            image = images[obs['image']]
            mapped = [(index, point) for index, point in enumerate(image.points2D)
                      if point.has_point3D()]
            indices = np.array([index for index, _ in mapped], dtype=int)
            positions = np.asarray([point.xy for _, point in mapped])
            distances = np.linalg.norm(positions-np.asarray(obs['native_xy']), axis=1)
            nearest = int(np.argmin(distances))
            near_model.append({'frame': frame,
                               'nearest_mapped_distance_native_px': float(distances[nearest]),
                               'nearest_mapped_feature_index': int(indices[nearest]),
                               'nearest_point3D_id': int(mapped[nearest][1].point3D_id)})
        supports = []
        for first, last in protocol['support_pairs']:
            first_obs, last_obs = observations[first], observations[last]
            rays = [_world_ray(images[obs['image']], camera, np.asarray(obs['native_xy']))
                    for obs in (first_obs, last_obs)]
            angle = float(np.rad2deg(np.arccos(np.clip(rays[0][1] @ rays[1][1], -1., 1.))))
            world = _triangulate(*rays)
            predictions = []
            for frame in FRAMES:
                obs = observations[frame]
                projected, depth = _project(images[obs['image']], camera, world)
                residual = projected-np.asarray(obs['native_xy']) if projected is not None else None
                predictions.append({'frame': frame, 'used_as_support': frame in (first, last),
                                    'camera_depth': depth,
                                    'predicted_native_xy': projected.tolist() if projected is not None else None,
                                    'residual_xy_native_px': residual.tolist() if residual is not None else None,
                                    'error_native_px': float(np.linalg.norm(residual))
                                    if residual is not None else None})
            supports.append({'support_frames': (first, last),
                             'support_ray_angle_degrees': angle,
                             'predictions': predictions})
        rows.append({'id': item['id'], 'role': item['role'],
                     'candidate_rank': item['candidate_rank'],
                     'nearest_saved_3D_features': near_model, 'support_checks': supports})
    result = {'protocol_fingerprint': fingerprint, 'rows': rows,
              'guard': 'Image-side LoMa closure and worn white paint do not establish subpixel identity. Saved visual pose and intrinsics are candidate models; near-model distances only show training proximity, not independent truth.'}
    write_json(path, result)
    write_json(output / 'DIAGNOSTIC_ONLY.json', {
        'status': 'DIAGNOSTIC_ONLY', 'audit_sha256': file_digest(path),
        'reason': 'A small number of correlated patch tracks cannot authorize BA or calibration update.'})
    return result
