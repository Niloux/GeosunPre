"""Isolated SIFT correspondence proposal around coarse court markings and seams.

This uses image evidence only for selection. Saved visual poses are read solely
to diagnose the resulting three-frame tracks, never to accept or refine them.
"""

from datetime import datetime, timezone
import json
from pathlib import Path

import cv2
import numpy as np
import pycolmap
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree

from ..camera import Camera
from ..scene import digest, file_digest, load_scene, write_json
from .sfm_hybrid_court_probe import FRAMES, court_color_proposal
from .sfm_resolution_probe import _cached_feature
from .sfm_virtual_cycle_probe import _three_view_geometry


OUTPUT = 'sfm_court_structure_probe_v2'
QUARTER_SIZE = (750, 1000)
HALF_SIZE = (1500, 2000)
LAB_BLUR_SIGMA_QUARTER_PX = 2.5
LAB_EDGE_MIN = 10.
EDGE_DILATE_QUARTER_PX = 7
SIFT_NFEATURES = 8000
SIFT_CONTRAST = .03
SIFT_MIN_SIZE_HALF_PX = 4.
WHITE_LINE_MAX_SATURATION = 115
WHITE_LINE_MIN_VALUE = 125
WHITE_LINE_DILATE_QUARTER_PX = 5
LOWE_RATIO = .75
NOVEL_MIN_DISTANCE_NATIVE_PX = 5.


def _image_and_structure(record, camera, candidate_mask):
    bgr = cv2.imread(record['source'], cv2.IMREAD_COLOR)
    if bgr is None or bgr.shape[:2] != (camera.height, camera.width):
        raise ValueError(f'Missing or changed native JPEG: {record["name"]}')
    region, _ = court_color_proposal(bgr, camera.mask(179) > 0)
    candidate = cv2.imread(candidate_mask, cv2.IMREAD_GRAYSCALE)
    if candidate is None or candidate.shape != bgr.shape[:2]:
        raise ValueError(f'Missing candidate mask: {candidate_mask}')
    candidate_q = cv2.resize(candidate, QUARTER_SIZE,
                             interpolation=cv2.INTER_NEAREST) >= 128
    region &= candidate_q
    quarter = cv2.resize(bgr, QUARTER_SIZE, interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(quarter, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab = cv2.GaussianBlur(lab, (0, 0), LAB_BLUR_SIGMA_QUARTER_PX)
    dx = cv2.Sobel(lab, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(lab, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(np.sum(dx * dx + dy * dy, axis=2)) / 8
    edges = magnitude >= LAB_EDGE_MIN
    edges = cv2.dilate(edges.astype(np.uint8),
                       cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                 (EDGE_DILATE_QUARTER_PX,
                                                  EDGE_DILATE_QUARTER_PX))) > 0
    hsv = cv2.cvtColor(quarter, cv2.COLOR_BGR2HSV)
    whitish = ((hsv[:, :, 1] <= WHITE_LINE_MAX_SATURATION)
               & (hsv[:, :, 2] >= WHITE_LINE_MIN_VALUE) & region)
    colored_nearby = cv2.dilate(((hsv[:, :, 1] > WHITE_LINE_MAX_SATURATION)
                                 & region).astype(np.uint8),
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                          (9, 9))) > 0
    line = whitish & colored_nearby
    line = cv2.dilate(line.astype(np.uint8),
                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                (WHITE_LINE_DILATE_QUARTER_PX,
                                                 WHITE_LINE_DILATE_QUARTER_PX))) > 0
    line &= region
    structural = (edges | line) & region
    structural_half = cv2.resize(structural.astype(np.uint8), HALF_SIZE,
                                  interpolation=cv2.INTER_NEAREST)
    gray_half = cv2.cvtColor(cv2.resize(bgr, HALF_SIZE,
                                        interpolation=cv2.INTER_AREA),
                             cv2.COLOR_BGR2GRAY)
    return gray_half, structural_half, region, structural, line


def _native(keypoints):
    xy = np.asarray([kp.pt for kp in keypoints], dtype=float).reshape(-1, 2)
    return (xy + .5) * 2


def _features(gray_half, structural_half, white_line, camera):
    sift = cv2.SIFT_create(nfeatures=SIFT_NFEATURES,
                           contrastThreshold=SIFT_CONTRAST)
    keypoints, descriptors = sift.detectAndCompute(gray_half, structural_half)
    if descriptors is None:
        return {'keypoints': np.empty((0, 2)),
                'descriptors': np.empty((0, 128), dtype=np.float32),
                'on_white_line': np.empty(0, dtype=bool),
                'detected': 0, 'size_rejected': 0}
    detected = len(keypoints)
    xy = _native(keypoints)
    sizes = np.asarray([kp.size for kp in keypoints])
    keep = (sizes >= SIFT_MIN_SIZE_HALF_PX) & camera.valid(xy, 179)
    ix = np.clip(np.floor(xy[:, 0] / 4).astype(int), 0, white_line.shape[1] - 1)
    iy = np.clip(np.floor(xy[:, 1] / 4).astype(int), 0, white_line.shape[0] - 1)
    return {'keypoints': xy[keep], 'descriptors': descriptors[keep],
            'on_white_line': white_line[iy[keep], ix[keep]],
            'detected': detected, 'size_rejected': int((sizes < SIFT_MIN_SIZE_HALF_PX).sum())}


def _ratio_mutual(a, b):
    if len(a) < 2 or len(b) < 2:
        return np.empty((0, 2), dtype=np.int64)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    forward = matcher.knnMatch(a, b, k=2)
    backward = matcher.knnMatch(b, a, k=2)
    back = {m.queryIdx: m.trainIdx for pair in backward if len(pair) == 2
            for m, n in [pair] if m.distance < LOWE_RATIO * n.distance}
    pairs = [(m.queryIdx, m.trainIdx) for pair in forward if len(pair) == 2
             for m, n in [pair]
             if m.distance < LOWE_RATIO * n.distance
             and back.get(m.trainIdx) == m.queryIdx]
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)


def _closed_cycles(ab, ac, bc):
    ab_map, ac_map, bc_map = [{int(i): int(j) for i, j in pairs}
                               for pairs in (ab, ac, bc)]
    cycles = [(ia, ib, ac_map[ia]) for ia, ib in ab_map.items()
              if ia in ac_map and bc_map.get(ib) == ac_map[ia]]
    return np.asarray(cycles, dtype=np.int64).reshape(-1, 3)


def _render(path, record, structural, line, all_points, cycle_points, good, camera):
    with Image.open(record['source']) as source:
        small = source.convert('RGB').resize(QUARTER_SIZE, Image.Resampling.LANCZOS)
    overlay = np.asarray(small).copy()
    overlay[structural] = (overlay[structural] * .45
                           + np.array([255, 202, 35]) * .55).astype(np.uint8)
    overlay[line] = (overlay[line] * .35
                     + np.array([255, 60, 220]) * .65).astype(np.uint8)
    canvas = Image.new('RGB', (1500, 1040), (24, 28, 35))
    canvas.paste(Image.fromarray(overlay), (0, 40))
    canvas.paste(small, (750, 40))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 10), f'L{record["frame_index"]:05d}: Lab edges (yellow), white lines (pink)',
              fill=(245, 245, 245))
    draw.text((760, 10), f'SIFT {len(all_points)}, closed {len(cycle_points)}, '
              f'fixed-geometry {int(good.sum())}', fill=(245, 245, 245))
    for x, y in all_points:
        cx, cy = 750 + x / 4, 40 + y / 4
        draw.ellipse((cx - 1, cy - 1, cx + 1, cy + 1), fill=(245, 208, 40))
    for (x, y), accepted in zip(cycle_points, good):
        cx, cy = 750 + x / 4, 40 + y / 4
        color = (35, 235, 205) if accepted else (255, 95, 75)
        draw.ellipse((cx - 3, cy - 3, cx + 3, cy + 3), fill=color)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=92)


def run_court_structure_probe(work):
    cv2.setNumThreads(4)
    work = Path(work).resolve()
    root = work / 'global_sfm'
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
    protocol = {'version': 2, 'triplets': triplets,
                'source_sha256': {name: file_digest(records[name]['source']) for name in names},
                'court_mask_protocol_fingerprint': mask_protocol['fingerprint'],
                'court_mask_sha256': {name: file_digest(masks[name]) for name in names},
                'baseline_frontend_sha256': file_digest(root / 'frontend.json'),
                'fixed_visual_model_hashes': {p.name: file_digest(p)
                                              for p in sorted(model_path.glob('*.bin'))},
                'quarter_size': QUARTER_SIZE, 'half_size': HALF_SIZE,
                'lab_blur_sigma_quarter_px': LAB_BLUR_SIGMA_QUARTER_PX,
                'lab_edge_min_sobel_div_8': LAB_EDGE_MIN,
                'edge_dilate_quarter_px': EDGE_DILATE_QUARTER_PX,
                'sift_nfeatures': SIFT_NFEATURES,
                'sift_contrast_threshold': SIFT_CONTRAST,
                'sift_min_size_half_px': SIFT_MIN_SIZE_HALF_PX,
                'white_line_max_saturation_cv8': WHITE_LINE_MAX_SATURATION,
                'white_line_min_value_cv8': WHITE_LINE_MIN_VALUE,
                'white_line_dilate_quarter_px': WHITE_LINE_DILATE_QUARTER_PX,
                'mutual_lowe_ratio': LOWE_RATIO,
                'novelty_distance_native_px': NOVEL_MIN_DISTANCE_NATIVE_PX,
                'uses_navigation_las_or_manual_labels': False,
                'scope': 'Six court-facing L images only; image-only selection, fixed saved visual model for downstream diagnosis, no full SfM or LFS modification.'}
    fingerprint = digest(protocol)
    output = root / OUTPUT
    output.mkdir(exist_ok=True)
    protocol_path = output / 'protocol.json'
    if protocol_path.exists():
        if json.loads(protocol_path.read_text())['fingerprint'] != fingerprint:
            raise ValueError('Existing structure probe belongs to changed inputs')
    else:
        write_json(protocol_path, {**protocol, 'fingerprint': fingerprint,
                                   'frozen_at_utc': datetime.now(timezone.utc).isoformat()})
    reconstruction = pycolmap.Reconstruction(model_path)
    images_by_name = {image.name: image for image in reconstruction.images.values()}
    fitted = reconstruction.cameras[1]
    camera = Camera('L', fitted.width, fitted.height, tuple(fitted.params), (0, 0))
    baseline = json.loads((root / 'frontend.json').read_text())
    baseline_features = {name: _cached_feature(baseline['features'][name])
                         for name in names}
    rows = []
    for index, triple in enumerate(triplets, 1):
        features, structures, lines = {}, {}, {}
        for name in triple:
            gray, structural_half, region, structural, line = _image_and_structure(
                records[name], camera, masks[name])
            features[name] = _features(gray, structural_half, line, camera)
            structures[name] = structural
            lines[name] = line
            features[name]['region'] = region
            features[name]['structural_pixels_quarter'] = int(structural.sum())
            features[name]['white_line_pixels_quarter'] = int(line.sum())
        a, b, c = triple
        ab = _ratio_mutual(features[a]['descriptors'], features[b]['descriptors'])
        ac = _ratio_mutual(features[a]['descriptors'], features[c]['descriptors'])
        bc = _ratio_mutual(features[b]['descriptors'], features[c]['descriptors'])
        cycles = _closed_cycles(ab, ac, bc)
        geometry, detail = _three_view_geometry(triple, features, cycles,
                                                images_by_name, camera,
                                                features[a]['region'],
                                                return_masks=True)
        xy_a = features[a]['keypoints'][cycles[:, 0]]
        near = xy_a[:, 1] >= .6 * camera.height
        good = detail['three_view_ok']
        white = features[a]['on_white_line'][cycles[:, 0]]
        old_xy = baseline_features[a]['keypoints'].astype(float)
        novel = cKDTree(old_xy).query(xy_a, k=1)[0] > NOVEL_MIN_DISTANCE_NATIVE_PX
        review = output / 'review' / f'L{records[a]["frame_index"]:05d}.jpg'
        _render(review, records[a], structures[a], lines[a], features[a]['keypoints'],
                xy_a, good, camera)
        tracks_path = output / 'tracks' / f'L{records[a]["frame_index"]:05d}.npz'
        tracks_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(tracks_path, xy_a=xy_a,
                            xy_b=features[b]['keypoints'][cycles[:, 1]],
                            xy_c=features[c]['keypoints'][cycles[:, 2]],
                            fixed_geometry_ok=good, novel_vs_loma=novel,
                            on_white_line=white)
        row = {'start_frame': records[a]['frame_index'], 'images': triple,
               'review': str(review), 'tracks': str(tracks_path),
               'per_image': {name: {'detected': features[name]['detected'],
                                    'size_rejected': features[name]['size_rejected'],
                                    'retained': int(len(features[name]['keypoints'])),
                                    'retained_on_white_line': int(
                                        features[name]['on_white_line'].sum()),
                                    'structural_pixels_quarter':
                                    features[name]['structural_pixels_quarter'],
                                    'white_line_pixels_quarter':
                                    features[name]['white_line_pixels_quarter']}
                             for name in triple},
               'mutual_ratio_matches': {'a_b': len(ab), 'a_c': len(ac), 'b_c': len(bc)},
               'three_way_descriptor_closed': int(len(cycles)),
               'closed_y_ge_60pct': int(near.sum()),
               'closed_on_white_line': int(white.sum()),
               'closed_on_white_line_y_ge_60pct': int((white & near).sum()),
               'fixed_visual_geometry': geometry,
               'fixed_geometry_ok_y_ge_60pct': int((good & near).sum()),
               'fixed_geometry_ok_on_white_line': int((good & white).sum()),
               'closed_gt_5px_from_loma': int(novel.sum()),
               'fixed_geometry_ok_gt_5px_from_loma': int((good & novel).sum()),
               'fixed_geometry_ok_novel_y_ge_60pct': int((good & novel & near).sum())}
        rows.append(row)
        print(f'Court structure {index}/{len(triplets)} L{row["start_frame"]}: '
              f'{len(cycles)} descriptor-closed', flush=True)
    result = {'protocol_fingerprint': fingerprint, 'rows': rows,
              'guard': 'Structural mask is an image-derived court-color/large-Lab-edge proposal, not semantic truth. Mutual descriptor cycles can still be wrong on repeated structures. Fixed visual geometry is biased toward old features; no candidate SfM or LFS change.'}
    write_json(output / 'results.json', result)
    return result
