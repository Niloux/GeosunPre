"""Independent Shi-Tomasi/LK track proposal on six court-facing fisheye JPEGs.

This only tests whether local image tracking supplies stable court points that
the current LoMa frontend missed. It does not alter SfM or accept the tracks.
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
from ..geometry import stats
from ..scene import digest, file_digest, load_scene, write_json
from .sfm_hybrid_court_probe import FRAMES, court_color_proposal
from .sfm_lk_probe import _flow
from .sfm_resolution_probe import _cached_feature
from .sfm_virtual_cycle_probe import _three_view_geometry


OUTPUT = 'sfm_court_flow_probe_v2'
STRUCTURAL_OUTPUT = 'sfm_court_structural_probe'
STRUCTURAL_BLUR_SIGMA_HALF_PX = 2.5
HALF_SIZE = (1500, 2000)
MAX_CORNERS = 5000
CORNER_QUALITY = .01
MIN_CORNER_DISTANCE_HALF_PX = 6
MAX_FB_ORIGINAL_PX = 1.
MAX_THREE_WAY_CYCLE_ORIGINAL_PX = 1.
NOVEL_MIN_DISTANCE_ORIGINAL_PX = 5.
PATCH_HALF_SIDE_HALF_PX = 20
PATCH_SEARCH_RADIUS_HALF_PX = 24
SECOND_PEAK_EXCLUSION_HALF_PX = 6


def _image_and_region(record, camera, candidate_mask):
    bgr = cv2.imread(record['source'], cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(cv2.resize(bgr, HALF_SIZE, interpolation=cv2.INTER_AREA),
                        cv2.COLOR_BGR2GRAY)
    region, _ = court_color_proposal(bgr, camera.mask(179) > 0)
    valid = cv2.imread(candidate_mask, cv2.IMREAD_GRAYSCALE) >= 128
    region_half = cv2.resize(region.astype(np.uint8), HALF_SIZE,
                             interpolation=cv2.INTER_NEAREST)
    valid_half = cv2.resize(valid.astype(np.uint8), HALF_SIZE,
                            interpolation=cv2.INTER_NEAREST)
    region_half &= valid_half
    return gray, region_half, region


def _inside_half(mask, points):
    ij = np.rint(points.reshape(-1, 2)).astype(int)
    inside = ((ij[:, 0] >= 0) & (ij[:, 0] < mask.shape[1])
              & (ij[:, 1] >= 0) & (ij[:, 1] < mask.shape[0]))
    ij[:, 0] = np.clip(ij[:, 0], 0, mask.shape[1] - 1)
    ij[:, 1] = np.clip(ij[:, 1], 0, mask.shape[0] - 1)
    return inside & (mask[ij[:, 1], ij[:, 0]] > 0)


def _native(points):
    return (points.reshape(-1, 2).astype(float) + .5) * 2


def _patch_competition(source, target, source_points, target_points):
    """Compare an LK location to competing 41x41 half-res patches nearby."""
    size = 2 * PATCH_HALF_SIDE_HALF_PX + 1
    radius = PATCH_SEARCH_RADIUS_HALF_PX
    best_offset, best_score, predicted_score, runner_up = [], [], [], []
    for src, dst in zip(source_points.reshape(-1, 2), target_points.reshape(-1, 2)):
        template = cv2.getRectSubPix(source, (size, size), tuple(map(float, src)))
        search = cv2.getRectSubPix(target, (size + 2 * radius, size + 2 * radius),
                                   tuple(map(float, dst)))
        response = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
        _, best, _, location = cv2.minMaxLoc(response)
        best_offset.append(2 * np.linalg.norm(np.asarray(location) - radius))
        best_score.append(best)
        predicted_score.append(float(response[radius, radius]))
        yy, xx = np.indices(response.shape)
        other = (xx - location[0]) ** 2 + (yy - location[1]) ** 2
        other = response[other >= SECOND_PEAK_EXCLUSION_HALF_PX ** 2]
        runner_up.append(float(np.max(other)) if len(other) else np.nan)
    return {key: np.asarray(value, dtype=float) for key, value in
            [('best_offset_original_px', best_offset), ('best_ncc', best_score),
             ('lk_ncc', predicted_score), ('runner_up_ncc', runner_up)]}


def _patch_summary(comp, selected):
    if not selected.any():
        return {'count': 0}
    delta = comp['best_ncc'] - comp['runner_up_ncc']
    return {'count': int(selected.sum()),
            'best_peak_offset_original_px': stats(comp['best_offset_original_px'][selected]),
            'best_minus_second_ncc_median': float(np.nanmedian(delta[selected])),
            'fraction_best_minus_second_ncc_lt_0p03': float(
                np.nanmean(delta[selected] < .03)),
            'lk_location_ncc_median': float(np.nanmedian(comp['lk_ncc'][selected]))}


def _render_review(path, record, xy, geometry_ok):
    with Image.open(record['source']) as image:
        original = image.convert('RGB').resize((750, 1000), Image.Resampling.LANCZOS)
    canvas = Image.new('RGB', (750, 1040), (24, 28, 35))
    canvas.paste(original, (0, 40))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 10), f'L{record["frame_index"]:05d}: image-only flow {len(xy)}, '
              f'fixed-model 3-view {int(geometry_ok.sum())}', fill=(245, 245, 245))
    for (x, y), good in zip(xy, geometry_ok):
        cx, cy = x / 4, 40 + y / 4
        color = (35, 235, 205) if good else (250, 185, 40)
        draw.ellipse((cx - 2, cy - 2, cx + 2, cy + 2), fill=color)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=91)


def _render_near_crops(path, triple, records, tracks, camera):
    """Side-by-side native-pixel patches for manual inspection of near-court flow."""
    xy_a, xy_b, xy_c = tracks
    near = ((xy_a[:, 1] >= .6 * camera.height)
            & (xy_a[:, 0] >= .45 * camera.width)
            & (xy_a[:, 0] <= .7 * camera.width))
    indices = np.flatnonzero(near)
    if not len(indices):
        return None
    indices = indices[np.argsort(xy_a[indices, 1])]
    indices = indices[np.unique(np.linspace(0, len(indices) - 1,
                                            min(8, len(indices))).round().astype(int))]
    canvas = Image.new('RGB', (820, 42 + 214 * len(indices)), (24, 28, 35))
    draw = ImageDraw.Draw(canvas)
    for col, name in enumerate(triple):
        draw.text((col * 205 + 80, 14), Path(name).stem, fill=(240, 240, 240))
    originals = [Image.open(records[name]['source']).convert('RGB') for name in triple]
    try:
        for row, index in enumerate(indices):
            y = 42 + 214 * row
            draw.text((5, y + 80), str(index), fill=(240, 240, 240))
            for col, xy in enumerate((xy_a, xy_b, xy_c)):
                x, yy = xy[index]
                patch = originals[col].crop((int(round(x)) - 48, int(round(yy)) - 48,
                                             int(round(x)) + 48, int(round(yy)) + 48))
                patch = patch.resize((192, 192), Image.Resampling.NEAREST)
                left = 36 + 205 * col
                canvas.paste(patch, (left, y))
                draw.line((left + 88, y + 96, left + 104, y + 96), fill=(255, 240, 25), width=2)
                draw.line((left + 96, y + 88, left + 96, y + 104), fill=(255, 240, 25), width=2)
    finally:
        for original in originals:
            original.close()
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=95)
    return str(path)


def _track_triplet(triple, records, candidate_masks, camera, images_by_name,
                   baseline_features, output, blur_sigma_half_px):
    gray, masks, regions = [], [], []
    for name in triple:
        g, m, r = _image_and_region(records[name], camera, candidate_masks[name])
        gray.append(g)
        masks.append(m)
        regions.append(r)
    flow_gray = ([cv2.GaussianBlur(g, (0, 0), blur_sigma_half_px) for g in gray]
                 if blur_sigma_half_px else gray)
    corners = cv2.goodFeaturesToTrack(
        flow_gray[0], maxCorners=MAX_CORNERS, qualityLevel=CORNER_QUALITY,
        minDistance=MIN_CORNER_DISTANCE_HALF_PX, mask=masks[0], blockSize=7,
        useHarrisDetector=False)
    if corners is None:
        raise RuntimeError(f'No court corners in first image: {triple[0]}')
    a = corners.astype(np.float32)
    b, ok_ab, _ = _flow(flow_gray[0], flow_gray[1], a)
    back_a_b, ok_ba, _ = _flow(flow_gray[1], flow_gray[0], b)
    c_direct, ok_ac, _ = _flow(flow_gray[0], flow_gray[2], a)
    back_a_c, ok_ca, _ = _flow(flow_gray[2], flow_gray[0], c_direct)
    c_via_b, ok_bc, _ = _flow(flow_gray[1], flow_gray[2], b)
    back_b_c, ok_cb, _ = _flow(flow_gray[2], flow_gray[1], c_via_b)
    status = np.logical_and.reduce([array.ravel().astype(bool) for array in
                                    (ok_ab, ok_ba, ok_ac, ok_ca, ok_bc, ok_cb)])
    finite = np.isfinite(np.column_stack([x.reshape(len(a), 2)
                                          for x in (a, b, back_a_b, c_direct,
                                                    back_a_c, c_via_b, back_b_c)])).all(axis=1)
    fb_ab = np.linalg.norm((back_a_b - a).reshape(-1, 2), axis=1) * 2
    fb_ac = np.linalg.norm((back_a_c - a).reshape(-1, 2), axis=1) * 2
    fb_bc = np.linalg.norm((back_b_c - b).reshape(-1, 2), axis=1) * 2
    edge_repeatable = {}
    for label, forward, reverse, error in (
            ('a_to_b', ok_ab, ok_ba, fb_ab),
            ('a_to_c', ok_ac, ok_ca, fb_ac),
            ('b_to_c', ok_bc, ok_cb, fb_bc)):
        edge_repeatable[label] = {
            'forward_status_ok': int(forward.sum()),
            'both_status_ok': int((forward.astype(bool) & reverse.astype(bool)).sum()),
            'forward_backward_lt_1_original_px': int((forward.ravel().astype(bool)
                & reverse.ravel().astype(bool) & np.isfinite(error)
                & (error < MAX_FB_ORIGINAL_PX)).sum())}
    fb_ok = (status & finite & (fb_ab < MAX_FB_ORIGINAL_PX)
             & (fb_ac < MAX_FB_ORIGINAL_PX) & (fb_bc < MAX_FB_ORIGINAL_PX))
    cycle = np.linalg.norm((c_via_b - c_direct).reshape(-1, 2), axis=1) * 2
    cycle_ok = fb_ok & (cycle < MAX_THREE_WAY_CYCLE_ORIGINAL_PX)
    court_ok = (cycle_ok & _inside_half(masks[1], b)
                & _inside_half(masks[2], c_direct))
    selected = np.flatnonzero(court_ok)
    xy_a, xy_b, xy_c = [_native(points[selected]) for points in (a, b, c_direct)]
    for xy in (xy_a, xy_b, xy_c):
        if not camera.valid(xy, 179).all():
            raise ValueError('Image-tracked court point outside fitted 179-degree domain')
    features = {name: {'keypoints': xy} for name, xy in zip(triple, (xy_a, xy_b, xy_c))}
    indices = np.arange(len(selected), dtype=np.int64)
    cycles = np.column_stack((indices, indices, indices))
    geometry, detail = _three_view_geometry(triple, features, cycles, images_by_name,
                                            camera, regions[0], return_masks=True)
    old_xy = baseline_features[triple[0]]['keypoints'].astype(float)
    novelty = cKDTree(old_xy).query(xy_a, k=1)[0] > NOVEL_MIN_DISTANCE_ORIGINAL_PX
    good = detail['three_view_ok']
    near = xy_a[:, 1] >= camera.height * .6
    very_near = xy_a[:, 1] >= camera.height * .7
    positive = detail['positive_depth']
    angle_ok = detail['angle_deg'] >= 1.
    support_error_ok = detail['support_error_px'] <= 3.
    geometry_rejection = {
        'nonpositive_or_nonfinite_depth': int((~positive).sum()),
        'positive_but_angle_lt_1deg': int((positive & ~angle_ok).sum()),
        'positive_angle_ok_but_endpoint_error_gt_3px': int(
            (positive & angle_ok & ~support_error_ok).sum()),
        'support_ok_but_middle_error_gt_3px': int(
            (detail['support_ok'] & ~good).sum())}
    patches_ab = _patch_competition(gray[0], gray[1], a[selected], b[selected])
    patches_ac = _patch_competition(gray[0], gray[2], a[selected], c_direct[selected])
    patch_audit = {edge: {group: _patch_summary(comp, mask)
                          for group, mask in [('geometry_ok', good),
                                              ('image_closed_but_geometry_bad', ~good),
                                              ('near_floor_y_ge_60pct', near)]}
                   for edge, comp in [('a_to_b', patches_ab), ('a_to_c', patches_ac)]}
    label = f'L{records[triple[0]]["frame_index"]:05d}'
    review = output / 'review' / f'{label}.jpg'
    _render_review(review, records[triple[0]], xy_a, good)
    tracks_path = output / 'tracks' / f'{label}.npz'
    tracks_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(tracks_path, xy_a=xy_a, xy_b=xy_b, xy_c=xy_c,
                        fixed_geometry_ok=good, fixed_positive_depth=positive,
                        nearest_loma_gt_5px=novelty)
    near_crops = _render_near_crops(output / 'review' / f'{label}_near_crops.jpg',
                                   triple, records, (xy_a, xy_b, xy_c), camera)
    return {'start_frame': records[triple[0]]['frame_index'], 'images': triple,
            'review': str(review), 'near_crops_review': near_crops,
            'tracks': str(tracks_path),
            'detected_court_corners': int(len(a)),
            'per_edge_repeatability': edge_repeatable,
            'bidirectional_three_edge_repeatable': int(fb_ok.sum()),
            'three_way_flow_cycle_ok': int(cycle_ok.sum()),
            'court_candidate_in_all_three': int(court_ok.sum()),
            'flow_cycle_disagreement_original_px': stats(cycle[court_ok]),
            'fixed_visual_geometry': geometry,
            'fixed_visual_geometry_rejections': geometry_rejection,
            'local_patch_competition': patch_audit,
            'image_closed_y_ge_60pct': int(near.sum()),
            'image_closed_y_ge_70pct': int(very_near.sum()),
            'geometry_ok_y_ge_60pct': int((good & near).sum()),
            'geometry_ok_y_ge_70pct': int((good & very_near).sum()),
            'geometry_ok_and_novel_y_ge_60pct': int((good & novelty & near).sum()),
            'distance_gt_5px_from_baseline_loma': int(novelty.sum()),
            'geometry_ok_and_gt_5px_from_baseline_loma': int((good & novelty).sum()),
            'geometry_ok_on_baseline_feature_neighborhood': int((good & ~novelty).sum())}


def run_court_flow_probe(work, structural=False):
    cv2.setNumThreads(4)
    work = Path(work).resolve()
    root = work / 'global_sfm'
    scene = load_scene(root)
    records = {record['name']: record for record in scene['images']}
    audit = json.loads((root / 'sfm_fov179_court_candidate/mask_audit.json').read_text())
    mask_protocol = json.loads((root / 'sfm_fov179_court_candidate/protocol.json').read_text())
    if audit['protocol_fingerprint'] != mask_protocol['fingerprint']:
        raise ValueError('Candidate mask audit/protocol mismatch')
    names = [row['name'] for row in audit['rows']]
    if tuple(records[name]['frame_index'] for name in names) != FRAMES:
        raise ValueError('Court image order drifted')
    candidate_masks = {row['name']: row['mask'] for row in audit['rows']}
    triplets = [tuple(names[start:start + 3]) for start in (0, 3)]
    model_path = root / 'ground_joint_ba/equal_control/sparse/0'
    model_hashes = {p.name: file_digest(p) for p in sorted(model_path.glob('*.bin'))}
    blur_sigma_half_px = STRUCTURAL_BLUR_SIGMA_HALF_PX if structural else 0.
    protocol = {'version': 3 if structural else 2, 'triplets': triplets,
                'source_sha256': {name: file_digest(records[name]['source']) for name in names},
                'candidate_mask_protocol_fingerprint': mask_protocol['fingerprint'],
                'candidate_mask_sha256': {name: file_digest(path)
                                          for name, path in candidate_masks.items()},
                'baseline_frontend_sha256': file_digest(root / 'frontend.json'),
                'fixed_visual_model_hashes': model_hashes,
                'half_image_size': HALF_SIZE, 'corners_max': MAX_CORNERS,
                'corners_quality': CORNER_QUALITY,
                'corners_min_distance_half_px': MIN_CORNER_DISTANCE_HALF_PX,
                'lk_window': [21, 21], 'lk_max_pyramid_level': 3,
                'patch_competition_template_half_px': 41,
                'patch_competition_search_radius_half_px': PATCH_SEARCH_RADIUS_HALF_PX,
                'patch_competition_second_peak_exclusion_half_px': SECOND_PEAK_EXCLUSION_HALF_PX,
                'max_each_forward_backward_original_px': MAX_FB_ORIGINAL_PX,
                'max_three_way_cycle_original_px': MAX_THREE_WAY_CYCLE_ORIGINAL_PX,
                'novelty_distance_to_baseline_original_px': NOVEL_MIN_DISTANCE_ORIGINAL_PX,
                'uses_navigation_las_or_manual_labels': False,
                'scope': 'Two independent court-facing L triplets, read-only image-tracking proposal. Fixed baseline visual poses used only for downstream diagnostic.'}
    if structural:
        protocol['blur_before_detection_and_flow_sigma_half_px'] = blur_sigma_half_px
    fingerprint = digest(protocol)
    output = root / (STRUCTURAL_OUTPUT if structural else OUTPUT)
    output.mkdir(exist_ok=True)
    protocol_path = output / 'protocol.json'
    if protocol_path.exists():
        if json.loads(protocol_path.read_text())['fingerprint'] != fingerprint:
            raise ValueError('Existing court flow probe belongs to changed inputs')
    else:
        write_json(protocol_path, {**protocol, 'fingerprint': fingerprint,
                                   'frozen_at_utc': datetime.now(timezone.utc).isoformat()})
    reconstruction = pycolmap.Reconstruction(model_path)
    images_by_name = {image.name: image for image in reconstruction.images.values()}
    fitted = reconstruction.cameras[1]
    camera = Camera('L', fitted.width, fitted.height, tuple(fitted.params), (0, 0))
    baseline = json.loads((root / 'frontend.json').read_text())
    features = {name: _cached_feature(baseline['features'][name]) for name in names}
    rows = []
    for index, triple in enumerate(triplets, 1):
        row = _track_triplet(triple, records, candidate_masks, camera,
                             images_by_name, features, output, blur_sigma_half_px)
        rows.append(row)
        print(f'Court image-only flow {index}/{len(triplets)} L{row["start_frame"]}: '
              f'{row["court_candidate_in_all_three"]} image-closed', flush=True)
    result = {'protocol_fingerprint': fingerprint, 'rows': rows,
              'guard': 'Bidirectional LK and three-frame flow closure are not physical-point truth; repeated paving can be coherently wrong. The fixed visual pose was trained with baseline LoMa and is biased toward it. No flow track has been added to SfM or LFS.'}
    write_json(output / 'results.json', result)
    return result
