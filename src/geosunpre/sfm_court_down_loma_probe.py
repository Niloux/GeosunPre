"""Ground-focused virtual-view LoMa proposal for six native L fisheye images.

The pinhole view is only a frontend sampling surface. Keypoints return to
native OPENCV_FISHEYE pixels, and every L exposure keeps its own pose.
"""

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path

import cv2
import numpy as np
import pycolmap
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .camera import Camera
from .frontend import FrontendConfig, LoMaFrontend, balanced_selection
from .geometry import stats
from .scene import digest, file_digest, load_scene, write_json
from .sfm_hybrid_court_probe import FRAMES
from .sfm_lk_probe import _flow
from .sfm_resolution_probe import _cached_feature
from .sfm_virtual_cycle_probe import _closed_cycles, _three_view_geometry


OUTPUT = 'sfm_court_down_loma_probe'
VIEW_SIZE = 1280
VIEW_FOV_DEG = 80.
PITCH_DEG = -45.
VIEW_KEYPOINTS = 3072
VIEW_DETECT_OVERSAMPLE = 3
NATIVE_NOVEL_MIN_PX = 5.
MIN_MATCH_SCORE = .9
CONFIG = FrontendConfig(detector_size=VIEW_SIZE, num_keypoints=VIEW_KEYPOINTS)
LK_FB_MAX_NATIVE_PX = 1.
LK_MATCH_MAX_NATIVE_PX = 2.
HALF_SIZE = (1500, 2000)


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
    return native, valid


def _view_path(output, record):
    return output / 'views' / f'L{record["frame_index"]:05d}.png'


def _prepare_views(output, records, names, masks, camera, native, valid):
    rows = []
    remap_x = (native[:, :, 0] - .5).astype(np.float32)
    remap_y = (native[:, :, 1] - .5).astype(np.float32)
    ij = np.floor(native).astype(int)
    ij[:, :, 0] = np.clip(ij[:, :, 0], 0, camera.width - 1)
    ij[:, :, 1] = np.clip(ij[:, :, 1], 0, camera.height - 1)
    for name in names:
        record = records[name]
        bgr = cv2.imread(record['source'], cv2.IMREAD_COLOR)
        candidate = cv2.imread(masks[name], cv2.IMREAD_GRAYSCALE)
        if bgr is None or bgr.shape[:2] != (camera.height, camera.width):
            raise ValueError(f'Missing native image: {name}')
        if candidate is None or candidate.shape != bgr.shape[:2]:
            raise ValueError(f'Missing court candidate mask: {name}')
        court = valid & (candidate[ij[:, :, 1], ij[:, :, 0]] >= 128)
        view = cv2.remap(bgr, remap_x, remap_y, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT)
        view[~valid] = 0
        path = _view_path(output, record)
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), view)
        if record['frame_index'] in (120, 150):
            preview = cv2.resize(view, (640, 640), interpolation=cv2.INTER_AREA)
            tint = preview.copy()
            qcourt = cv2.resize(court.astype(np.uint8), (640, 640),
                                interpolation=cv2.INTER_NEAREST) > 0
            tint[qcourt] = (.5 * tint[qcourt]
                            + .5 * np.array([210, 30, 210])).astype(np.uint8)
            canvas = np.concatenate((preview, tint), axis=1)
            preview_path = output / 'review' / f'L{record["frame_index"]:05d}_view.jpg'
            preview_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(preview_path), canvas)
        rows.append({'name': name, 'frame_index': record['frame_index'],
                     'view': str(path), 'view_sha256': file_digest(path),
                     'valid_view_pixels': int(valid.sum()),
                     'court_candidate_view_pixels': int(court.sum())})
    write_json(output / 'views.json', {'rows': rows})
    return rows


def _feature(output, record, mask_path, view_row, frontend, camera, valid):
    token = digest({'protocol': json.loads((output / 'protocol.json').read_text())['fingerprint'],
                    'name': record['name'], 'view_sha256': view_row['view_sha256']})
    cache = output / 'features' / f'{token}.npz'
    if cache.exists():
        frontend.cache_stats['feature_hits'] += 1
        with np.load(cache, allow_pickle=False) as data:
            if str(data['token']) != token:
                raise ValueError('Ground-view LoMa feature cache token mismatch')
            return {key: data[key] for key in data.files}
    frontend.cache_stats['feature_misses'] += 1
    import torch
    model = frontend.load_model()
    with torch.inference_mode():
        detected = model._detector.detect_from_path(
            view_row['view'], num_keypoints=VIEW_KEYPOINTS * VIEW_DETECT_OVERSAMPLE)
        normalized = detected['keypoints'][0].float().cpu().numpy()
        scores = detected['keypoint_probs'][0].float().cpu().numpy().reshape(-1)
        xy_view = (normalized + 1) * VIEW_SIZE / 2
        ij_view = np.floor(xy_view).astype(int)
        inside_view = ((ij_view[:, 0] >= 8) & (ij_view[:, 0] < VIEW_SIZE - 8)
                       & (ij_view[:, 1] >= 8) & (ij_view[:, 1] < VIEW_SIZE - 8))
        ij_view = np.clip(ij_view, 0, VIEW_SIZE - 1)
        # Map subpixel detector coordinates directly to the native fisheye.
        focal = VIEW_SIZE / (2 * np.tan(np.deg2rad(VIEW_FOV_DEG / 2)))
        local = np.column_stack(((xy_view[:, 0] - VIEW_SIZE / 2) / focal,
                                 (xy_view[:, 1] - VIEW_SIZE / 2) / focal,
                                 np.ones(len(xy_view))))
        rotation = Rotation.from_euler('x', PITCH_DEG, degrees=True).as_matrix()
        xy_native = camera.project(local @ rotation.T)
        candidate = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if candidate is None or candidate.shape != (camera.height, camera.width):
            raise ValueError(f'Missing court candidate mask: {mask_path}')
        ij_native = np.floor(np.where(np.isfinite(xy_native), xy_native, 0)).astype(int)
        ij_native[:, 0] = np.clip(ij_native[:, 0], 0, camera.width - 1)
        ij_native[:, 1] = np.clip(ij_native[:, 1], 0, camera.height - 1)
        keep = (inside_view & valid[ij_view[:, 1], ij_view[:, 0]]
                & np.isfinite(xy_native).all(axis=1)
                & camera.valid(np.where(np.isfinite(xy_native), xy_native, 0), 179)
                & (candidate[ij_native[:, 1], ij_native[:, 0]] >= 128))
        candidates = np.flatnonzero(keep)
        chosen = balanced_selection(xy_view[candidates], scores[candidates],
                                    VIEW_KEYPOINTS, VIEW_SIZE, VIEW_SIZE)
        indices = candidates[chosen]
        if len(indices) < 32:
            raise RuntimeError(f'Only {len(indices)} valid ground-view keypoints: {record["name"]}')
        selected = detected['keypoints'][:, torch.as_tensor(indices, device=frontend.device)]
        descriptors = model._descriptor.describe_keypoints_from_path(
            view_row['view'], selected)['descriptions'][0].float().cpu().numpy()
    result = {'keypoints': xy_native[indices].astype(np.float32),
              'normalized_keypoints': normalized[indices].astype(np.float32),
              'view_keypoints': xy_view[indices].astype(np.float32),
              'descriptors': descriptors.astype(np.float32),
              'scores': scores[indices].astype(np.float32),
              'detected': np.array(len(normalized)),
              'valid_before_quota': np.array(len(candidates)),
              'token': np.array(token)}
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix('.tmp.npz')
    np.savez_compressed(tmp, **result)
    tmp.replace(cache)
    return result


def _render_tracks(path, record, xy, geometry_good, near):
    with Image.open(record['source']) as im:
        small = im.convert('RGB').resize((750, 1000), Image.Resampling.LANCZOS)
    canvas = Image.new('RGB', (750, 1040), (24, 28, 35))
    canvas.paste(small, (0, 40))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 10), f'L{record["frame_index"]:05d}: closed {len(xy)}, '
              f'fixed 3-view {int(geometry_good.sum())}, y>=.6H {int(near.sum())}',
              fill=(245, 245, 245))
    for (x, y), good in zip(xy, geometry_good):
        cx, cy = x / 4, 40 + y / 4
        color = (35, 235, 205) if good else (255, 135, 35)
        draw.ellipse((cx - 2, cy - 2, cx + 2, cy + 2), fill=color)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=92)


def run_court_down_loma_probe(work, stage='all'):
    cv2.setNumThreads(4)
    work = Path(work).resolve()
    root = work / 'global_sfm'
    output = root / OUTPUT
    output.mkdir(exist_ok=True)
    scene = load_scene(root)
    records = {record['name']: record for record in scene['images']}
    audit = json.loads((root / 'sfm_fov179_court_candidate/mask_audit.json').read_text())
    mask_protocol = json.loads((root / 'sfm_fov179_court_candidate/protocol.json').read_text())
    if audit['protocol_fingerprint'] != mask_protocol['fingerprint']:
        raise ValueError('Court mask audit/protocol mismatch')
    names = [row['name'] for row in audit['rows']]
    if tuple(records[name]['frame_index'] for name in names) != FRAMES:
        raise ValueError('Court image order drifted')
    masks = {row['name']: row['mask'] for row in audit['rows']}
    triplets = [tuple(names[start:start + 3]) for start in (0, 3)]
    model_path = root / 'ground_joint_ba/equal_control/sparse/0'
    protocol = {'version': 1, 'triplets': triplets,
                'source_sha256': {name: file_digest(records[name]['source']) for name in names},
                'mask_protocol_fingerprint': mask_protocol['fingerprint'],
                'mask_sha256': {name: file_digest(masks[name]) for name in names},
                'baseline_frontend_sha256': file_digest(root / 'frontend.json'),
                'fixed_visual_model_hashes': {p.name: file_digest(p)
                                              for p in sorted(model_path.glob('*.bin'))},
                'view_size': VIEW_SIZE, 'view_fov_degrees': VIEW_FOV_DEG,
                'view_pitch_degrees': PITCH_DEG,
                'keypoints_per_view': VIEW_KEYPOINTS,
                'detect_oversample': VIEW_DETECT_OVERSAMPLE,
                'high_confidence_cycle_min_score': MIN_MATCH_SCORE,
                'novelty_min_distance_native_px': NATIVE_NOVEL_MIN_PX,
                'loma_config': asdict(CONFIG),
                'uses_navigation_las_or_manual_labels': False,
                'scope': 'Six ground-focused L virtual views as feature source; all observations map back to native fisheye. Fixed saved visual poses only for diagnosis. No full SfM/LFS change.'}
    fingerprint = digest(protocol)
    protocol_path = output / 'protocol.json'
    if protocol_path.exists():
        if json.loads(protocol_path.read_text())['fingerprint'] != fingerprint:
            raise ValueError('Existing ground-view LoMa probe belongs to changed inputs')
    else:
        write_json(protocol_path, {**protocol, 'fingerprint': fingerprint,
                                   'frozen_at_utc': datetime.now(timezone.utc).isoformat()})
    reconstruction = pycolmap.Reconstruction(model_path)
    fitted = reconstruction.cameras[1]
    camera = Camera('L', fitted.width, fitted.height, tuple(fitted.params), (0, 0))
    native, valid = _mapping(camera)
    if stage in ('all', 'prepare'):
        view_rows = _prepare_views(output, records, names, masks, camera, native, valid)
    else:
        view_rows = json.loads((output / 'views.json').read_text())['rows']
    if len(view_rows) != len(names):
        raise ValueError('Ground-view preparation incomplete')
    for row in view_rows:
        if file_digest(row['view']) != row['view_sha256']:
            raise ValueError(f'Ground view changed: {row["view"]}')
    if stage == 'prepare':
        return {'protocol_fingerprint': fingerprint, 'views': view_rows}
    rows_by_name = {row['name']: row for row in view_rows}
    frontend = LoMaFrontend(work, scene, CONFIG)
    features = {}
    for index, name in enumerate(names, 1):
        features[name] = _feature(output, records[name], masks[name],
                                  rows_by_name[name], frontend, camera, valid)
        print(f'Ground-view LoMa features {index}/{len(names)} '
              f'L{records[name]["frame_index"]}: '
              f'{len(features[name]["keypoints"])}', flush=True)
    baseline = json.loads((root / 'frontend.json').read_text())
    old = {name: _cached_feature(baseline['features'][name]) for name in names}
    images_by_name = {image.name: image for image in reconstruction.images.values()}
    rows = []
    for index, triple in enumerate(triplets, 1):
        a, b, c = triple
        pair_matches = [frontend.match(features[x], features[y])
                        for x, y in ((a, b), (a, c), (b, c))]
        cycles = _closed_cycles(pair_matches)
        geometry, detail = _three_view_geometry(triple, features, cycles,
                                                images_by_name, camera,
                                                return_masks=True)
        xy_a = features[a]['keypoints'][cycles[:, 0]]
        near = xy_a[:, 1] >= .6 * camera.height
        good = detail['three_view_ok']
        novel = cKDTree(old[a]['keypoints'].astype(float)).query(xy_a, k=1)[0] > NATIVE_NOVEL_MIN_PX
        review = output / 'review' / f'L{records[a]["frame_index"]:05d}_tracks.jpg'
        _render_tracks(review, records[a], xy_a, good, near)
        tracks_path = output / 'tracks' / f'L{records[a]["frame_index"]:05d}.npz'
        tracks_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(tracks_path, xy_a=xy_a,
                            xy_b=features[b]['keypoints'][cycles[:, 1]],
                            xy_c=features[c]['keypoints'][cycles[:, 2]],
                            fixed_geometry_ok=good, novel_vs_loma=novel)
        row = {'start_frame': records[a]['frame_index'], 'images': triple,
               'review': str(review), 'tracks': str(tracks_path),
               'feature_counts': [int(len(features[name]['keypoints'])) for name in triple],
               'pair_match_counts': [int(len(match['matches'])) for match in pair_matches],
               'pair_high_score_counts': [int((match['scores'] >= MIN_MATCH_SCORE).sum())
                                          for match in pair_matches],
               'high_score_three_way_closed': int(len(cycles)),
               'closed_y_ge_60pct': int(near.sum()),
               'fixed_visual_geometry': geometry,
               'fixed_geometry_ok_y_ge_60pct': int((good & near).sum()),
               'closed_gt_5px_from_native_loma': int(novel.sum()),
               'fixed_geometry_ok_gt_5px_from_native_loma': int((good & novel).sum()),
               'fixed_geometry_ok_novel_y_ge_60pct': int((good & novel & near).sum())}
        rows.append(row)
        print(f'Ground-view LoMa {index}/{len(triplets)} L{row["start_frame"]}: '
              f'{len(cycles)} high-score closed cycles', flush=True)
    # Cache hits change on every rerun but the scientific result should not.
    # Preserve the originally recorded runtime counters when replaying the
    # same frozen protocol, so downstream content hashes remain stable.
    previous_result_path = output / 'results.json'
    previous = (json.loads(previous_result_path.read_text())
                if previous_result_path.exists() else None)
    first_run_cache_stats = (previous['cache_stats']
                             if previous is not None
                             and previous['protocol_fingerprint'] == fingerprint
                             else frontend.cache_stats)
    result = {'protocol_fingerprint': fingerprint, 'rows': rows,
              'cache_stats': first_run_cache_stats,
              'guard': 'Ground virtual views sample features only. Native OPENCV_FISHEYE coordinates and independent poses remain the SfM domain. The fitted camera and saved visual poses are untrusted candidates; no candidate SfM or LFS input changed.'}
    write_json(output / 'results.json', result)
    return result


def _lk_edge(source, target, xy_source, xy_target):
    a = (xy_source / 2 - .5).astype(np.float32).reshape(-1, 1, 2)
    b = (xy_target / 2 - .5).astype(np.float32).reshape(-1, 1, 2)
    ab, ok_ab, _ = _flow(source, target, a)
    back_a, ok_back_a, _ = _flow(target, source, ab)
    ba, ok_ba, _ = _flow(target, source, b)
    back_b, ok_back_b, _ = _flow(source, target, ba)
    fb_a = np.linalg.norm((back_a - a).reshape(-1, 2), axis=1) * 2
    fb_b = np.linalg.norm((back_b - b).reshape(-1, 2), axis=1) * 2
    mismatch = np.maximum(np.linalg.norm((ab - b).reshape(-1, 2), axis=1),
                          np.linalg.norm((ba - a).reshape(-1, 2), axis=1)) * 2
    status = np.logical_and.reduce([array.ravel().astype(bool)
                                    for array in (ok_ab, ok_back_a, ok_ba, ok_back_b)])
    repeatable = (status & np.isfinite(fb_a) & np.isfinite(fb_b)
                  & (fb_a < LK_FB_MAX_NATIVE_PX) & (fb_b < LK_FB_MAX_NATIVE_PX))
    agrees = repeatable & np.isfinite(mismatch) & (mismatch <= LK_MATCH_MAX_NATIVE_PX)
    return {'repeatable': repeatable, 'agrees': agrees, 'mismatch': mismatch}


def _lk_group(take, edges, all_agree):
    if not take.any():
        return {'tracks': 0, 'all_three_edges_agree': 0,
                'all_three_edges_agree_fraction': None}
    result = {'tracks': int(take.sum()),
              'all_three_edges_agree': int((take & all_agree).sum()),
              'all_three_edges_agree_fraction': float(np.mean(all_agree[take]))}
    for label, edge in edges.items():
        result[label] = {'repeatable': int((take & edge['repeatable']).sum()),
                         'match_agreement_le_2px': int((take & edge['agrees']).sum()),
                         'mismatch_px_on_repeatable':
                         stats(edge['mismatch'][take & edge['repeatable']])}
    return result


def run_court_down_loma_audit(work):
    """Check frozen LoMa three-cycles against native-image bidirectional LK."""
    cv2.setNumThreads(4)
    work = Path(work).resolve()
    root = work / 'global_sfm'
    output = root / OUTPUT
    source_protocol = json.loads((output / 'protocol.json').read_text())
    source_result = json.loads((output / 'results.json').read_text())
    if source_result['protocol_fingerprint'] != source_protocol['fingerprint']:
        raise ValueError('Ground-view LoMa result/protocol mismatch')
    scene = load_scene(root)
    records = {record['name']: record for record in scene['images']}
    names = [name for row in source_result['rows'] for name in row['images']]
    protocol = {'version': 1,
                'ground_loma_protocol_fingerprint': source_protocol['fingerprint'],
                'ground_loma_result_sha256': file_digest(output / 'results.json'),
                'tracks_sha256': {row['tracks']: file_digest(row['tracks'])
                                  for row in source_result['rows']},
                'source_sha256': {name: file_digest(records[name]['source']) for name in names},
                'half_size': HALF_SIZE, 'lk_window': [21, 21],
                'lk_max_pyramid_level': 3,
                'symmetric_forward_backward_max_native_px': LK_FB_MAX_NATIVE_PX,
                'symmetric_match_disagreement_max_native_px': LK_MATCH_MAX_NATIVE_PX,
                'scope': 'Read-only image-only pixel repeatability audit of frozen ground-view LoMa three-cycles; no camera truth or SfM modification.'}
    fingerprint = digest(protocol)
    audit_path = output / 'lk_audit_protocol.json'
    if audit_path.exists():
        if json.loads(audit_path.read_text())['fingerprint'] != fingerprint:
            raise ValueError('Existing LK audit belongs to changed ground-view LoMa tracks')
    else:
        write_json(audit_path, {**protocol, 'fingerprint': fingerprint,
                                'frozen_at_utc': datetime.now(timezone.utc).isoformat()})
    gray = {}
    for name in names:
        original = cv2.imread(records[name]['source'], cv2.IMREAD_GRAYSCALE)
        if original is None or original.shape != (4000, 3000):
            raise ValueError(f'Missing native JPEG: {name}')
        gray[name] = cv2.resize(original, HALF_SIZE, interpolation=cv2.INTER_AREA)
    rows = []
    for row in source_result['rows']:
        a, b, c = row['images']
        with np.load(row['tracks'], allow_pickle=False) as data:
            xy_a, xy_b, xy_c = [data[key] for key in ('xy_a', 'xy_b', 'xy_c')]
            good = data['fixed_geometry_ok'].astype(bool)
            novel = data['novel_vs_loma'].astype(bool)
        edges = {'a_b': _lk_edge(gray[a], gray[b], xy_a, xy_b),
                 'a_c': _lk_edge(gray[a], gray[c], xy_a, xy_c),
                 'b_c': _lk_edge(gray[b], gray[c], xy_b, xy_c)}
        displacement = {'a_b': stats(np.linalg.norm(xy_a - xy_b, axis=1)),
                        'a_c': stats(np.linalg.norm(xy_a - xy_c, axis=1)),
                        'b_c': stats(np.linalg.norm(xy_b - xy_c, axis=1))}
        all_agree = np.logical_and.reduce([edge['agrees'] for edge in edges.values()])
        lower = xy_a[:, 1] >= 2400
        groups = {'all_closed': np.ones(len(xy_a), dtype=bool),
                  'fixed_geometry_ok': good,
                  'lower_image_y_ge_60pct': lower,
                  'fixed_geometry_ok_lower': good & lower,
                  'fixed_geometry_ok_novel_lower': good & novel & lower}
        rows.append({'start_frame': row['start_frame'],
                     'match_displacement_native_px': displacement,
                     'groups': {label: _lk_group(take, edges, all_agree)
                                for label, take in groups.items()}})
    result = {'protocol_fingerprint': fingerprint, 'rows': rows,
              'guard': 'The pair displacements are hundreds of native pixels, outside the useful capture range of this 21px-window, three-level LK setup. Its zero three-edge agreement is INCONCLUSIVE about LoMa correctness and must not reject the candidate. The lower image is not a ground semantic mask; fixed geometry uses untrusted saved visual poses.'}
    write_json(output / 'lk_audit.json', result)
    return result


def run_court_down_loma_review(work):
    """Render every lower-image, novel, fixed-model-valid cycle at native scale."""
    work = Path(work).resolve()
    output = work / 'global_sfm' / OUTPUT
    result = json.loads((output / 'results.json').read_text())
    protocol = json.loads((output / 'protocol.json').read_text())
    if result['protocol_fingerprint'] != protocol['fingerprint']:
        raise ValueError('Ground-view LoMa result/protocol mismatch')
    records = {record['name']: record for record in load_scene(work / 'global_sfm')['images']}
    rows = []
    for row in result['rows']:
        names = row['images']
        with np.load(row['tracks'], allow_pickle=False) as data:
            xy = [data[key] for key in ('xy_a', 'xy_b', 'xy_c')]
            choose = (data['fixed_geometry_ok'].astype(bool)
                      & data['novel_vs_loma'].astype(bool)
                      & (xy[0][:, 1] >= 2400))
        selected = np.flatnonzero(choose)
        label = f'L{row["start_frame"]:05d}'
        path = output / 'review' / f'{label}_near_novel_crops.jpg'
        width, height = 3 * 252 + 45, 42 + 255 * max(1, len(selected))
        canvas = Image.new('RGB', (width, height), (24, 28, 35))
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 10), f'{label}: {len(selected)} fixed-geometry, novel, y>=.6H',
                  fill=(245, 245, 245))
        images = [Image.open(records[name]['source']).convert('RGB') for name in names]
        try:
            for line, track_id in enumerate(selected):
                top = 42 + 255 * line
                draw.text((5, top + 100), str(track_id), fill=(245, 245, 245))
                for column, points in enumerate(xy):
                    x, y = points[track_id]
                    bounds = (int(round(x)) - 80, int(round(y)) - 80,
                              int(round(x)) + 80, int(round(y)) + 80)
                    crop = images[column].crop(bounds).resize((240, 240),
                                    Image.Resampling.NEAREST)
                    left = 38 + 252 * column
                    canvas.paste(crop, (left, top))
                    draw.line((left + 108, top + 120, left + 132, top + 120),
                              fill=(255, 245, 25), width=2)
                    draw.line((left + 120, top + 108, left + 120, top + 132),
                              fill=(255, 245, 25), width=2)
        finally:
            for image in images:
                image.close()
        path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(path, quality=95)
        rows.append({'start_frame': row['start_frame'], 'tracks': len(selected),
                     'review': str(path), 'review_sha256': file_digest(path)})
    review = {'ground_loma_protocol_fingerprint': protocol['fingerprint'],
              'ground_loma_result_sha256': file_digest(output / 'results.json'),
              'rows': rows,
              'guard': 'The fixed-geometry and novelty filters only choose a manageable review subset. Crops allow visual inspection, not pixel-ground-truth validation.'}
    write_json(output / 'near_review.json', review)
    return review
