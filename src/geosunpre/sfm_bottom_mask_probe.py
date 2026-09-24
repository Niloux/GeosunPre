"""Audit and isolate the legacy bottom-strip device mask on court images."""

from datetime import datetime, timezone
import json
from pathlib import Path

import cv2
import numpy as np
import pycolmap
from PIL import Image, ImageDraw

from .camera import Camera
from .scene import digest, file_digest, load_scene, write_json
from .sfm_hybrid_court_probe import FRAMES, court_color_proposal


OUTPUT = 'sfm_bottom_mask_probe'
FOV_CANDIDATE_OUTPUT = 'sfm_fov179_court_candidate'
RELEASE_START = .80
RELEASE_END = .90


def _render_review(path, bgr, fov, old, proposal, restored, label):
    size = (750, 1000)
    original = cv2.cvtColor(cv2.resize(bgr, size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    fov_small = cv2.resize(fov.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
    old_small = cv2.resize(old.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
    restored_small = cv2.resize(restored.astype(np.uint8), size,
                                interpolation=cv2.INTER_NEAREST) > 0
    left = original.copy()
    left[fov_small & ~old_small] = (left[fov_small & ~old_small] * .38
                                     + np.array([215, 65, 180]) * .62).astype(np.uint8)
    right = original.copy()
    right[proposal] = (right[proposal] * .6
                       + np.array([40, 210, 200]) * .4).astype(np.uint8)
    right[restored_small] = (right[restored_small] * .28
                             + np.array([255, 200, 25]) * .72).astype(np.uint8)
    canvas = Image.new('RGB', (1500, 1040), (24, 28, 35))
    canvas.paste(Image.fromarray(left), (0, 40))
    canvas.paste(Image.fromarray(right), (750, 40))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 10), f'{label}: old excluded in magenta', fill=(245, 245, 245))
    draw.text((760, 10), f'{label}: color proposal cyan, newly restored yellow',
              fill=(245, 245, 245))
    for x in (0, 750):
        for fraction in (RELEASE_START, RELEASE_END):
            y = 40 + 1000 * fraction
            draw.line((x, y, x + 750, y), fill=(255, 255, 255), width=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=90)


def run_bottom_mask_probe(work):
    work = Path(work).resolve()
    root = work / 'global_sfm'
    scene = load_scene(root)
    records = sorted((record for record in scene['images'] if record['camera'] == 'L'
                      and record['frame_index'] in FRAMES),
                     key=lambda record: record['frame_index'])
    if len(records) != len(FRAMES):
        raise ValueError('Expected all six court-facing L images')
    source_model = root / 'ground_joint_ba/equal_control/sparse/0'
    reconstruction = pycolmap.Reconstruction(source_model)
    source_camera = reconstruction.cameras[1]
    camera = Camera('L', source_camera.width, source_camera.height,
                    tuple(source_camera.params), (0, 0))
    protocol = {'version': 1, 'images': [record['name'] for record in records],
                'image_sha256': {record['name']: file_digest(record['source']) for record in records},
                'old_mask_sha256': {record['name']: file_digest(record['semantic_mask'])
                                    for record in records},
                'label_sha256': {record['name']: file_digest(
                    root / 'masks_v2/labels' / f'{record["name"]}.png') for record in records},
                'visual_model_hashes': {p.name: file_digest(p) for p in sorted(source_model.glob('*.bin'))},
                'fov_degrees': scene['fov_degrees'],
                'restoration': 'Only court-color candidate pixels inside current visual FOV, legacy reason-3 device label, and native y from 0.80H inclusive to 0.90H exclusive',
                'court_color_protocol_sha256': file_digest(root / 'sfm_hybrid_court_probe/court_protocol.json'),
                'uses_navigation_las_or_manual_labels': False,
                'scope': 'Isolated six-image alternative semantic masks. Existing feature cache, SfM, and LFS masks remain untouched.'}
    fingerprint = digest(protocol)
    out = root / OUTPUT
    out.mkdir(exist_ok=True)
    protocol_path = out / 'protocol.json'
    if protocol_path.exists():
        if json.loads(protocol_path.read_text())['fingerprint'] != fingerprint:
            raise ValueError('Bottom mask probe inputs changed')
    else:
        write_json(protocol_path, {**protocol, 'fingerprint': fingerprint,
                                   'frozen_at_utc': datetime.now(timezone.utc).isoformat()})
    fov = camera.mask(scene['fov_degrees']) > 0
    yy = np.arange(camera.height)[:, None]
    band = (yy >= int(camera.height * RELEASE_START)) & (yy < int(camera.height * RELEASE_END))
    rows = []
    for record in records:
        bgr = cv2.imread(record['source'], cv2.IMREAD_COLOR)
        old = cv2.imread(record['semantic_mask'], cv2.IMREAD_GRAYSCALE) >= 128
        label_path = root / 'masks_v2/labels' / f'{record["name"]}.png'
        reasons = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
        if bgr is None or old.shape != bgr.shape[:2] or reasons.shape != old.shape:
            raise ValueError(f'Missing or mismatched inputs: {record["name"]}')
        color_proposal, _ = court_color_proposal(bgr, fov)
        color_native = cv2.resize(color_proposal.astype(np.uint8),
                                  (camera.width, camera.height),
                                  interpolation=cv2.INTER_NEAREST) > 0
        restorable = (fov & band & (reasons == 3))
        restored = restorable & color_native
        candidate = old | restored
        name = f'L{record["frame_index"]:05d}'
        mask_path = out / 'candidate_masks' / f'{name}.png'
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(candidate.astype(np.uint8) * 255).save(mask_path)
        review_path = out / 'review' / f'{name}.jpg'
        _render_review(review_path, bgr, fov, old, color_proposal, restored, name)
        rows.append({'name': record['name'], 'frame': record['frame_index'],
                     'candidate_mask': str(mask_path), 'review': str(review_path),
                     'fov_pixels_in_band': int((fov & band).sum()),
                     'old_valid_pixels_in_band': int((old & fov & band).sum()),
                     'device_labeled_pixels_in_band': int(restorable.sum()),
                     'restored_court_color_pixels': int(restored.sum()),
                     'candidate_valid_pixels_in_band': int((candidate & fov & band).sum())})
        print(f'Bottom mask {name}: restored {int(restored.sum())} original pixels', flush=True)
    result = {'protocol_fingerprint': fingerprint, 'rows': rows,
              'guard': 'Color and position are reviewable proposals, not ground truth. The current all-bottom-strip device rule overwrote any person label in that region; only six visually reviewed L images are considered. No production masks are changed.'}
    write_json(out / 'audit.json', result)
    return result


def run_fov_audit(work):
    """Separate the 160-degree ray limit from the semantic bottom-strip rule."""
    work = Path(work).resolve()
    root = work / 'global_sfm'
    scene = load_scene(root)
    records = sorted((record for record in scene['images'] if record['camera'] == 'L'
                      and record['frame_index'] in FRAMES),
                     key=lambda record: record['frame_index'])
    source_model = root / 'ground_joint_ba/equal_control/sparse/0'
    reconstruction = pycolmap.Reconstruction(source_model)
    source_camera = reconstruction.cameras[1]
    camera = Camera('L', source_camera.width, source_camera.height,
                    tuple(source_camera.params), (0, 0))
    fovs = {fov: camera.mask(fov) > 0 for fov in (160, 175, 179)}
    out = root / OUTPUT
    rows = []
    for record in records:
        bgr = cv2.imread(record['source'], cv2.IMREAD_COLOR)
        old = cv2.imread(record['semantic_mask'], cv2.IMREAD_GRAYSCALE) >= 128
        candidate, _ = court_color_proposal(bgr, fovs[179])
        court = cv2.resize(candidate.astype(np.uint8), (camera.width, camera.height),
                           interpolation=cv2.INTER_NEAREST) > 0
        view = cv2.resize(bgr, (750, 1000), interpolation=cv2.INTER_AREA)
        colors = {160: (255, 110, 35), 175: (35, 210, 250), 179: (50, 50, 255)}
        for fov in (160, 175, 179):
            small_mask = cv2.resize(fovs[fov].astype(np.uint8), (750, 1000),
                                    interpolation=cv2.INTER_NEAREST)
            contours, _ = cv2.findContours(small_mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(view, contours, -1, colors[fov], 2)
        cv2.line(view, (0, 800), (749, 800), (235, 235, 235), 1)
        image = Image.fromarray(cv2.cvtColor(view, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image)
        draw.text((12, 12), '160 blue | 175 yellow | 179 red | semantic below 80% is all invalid',
                  fill=(255, 255, 255))
        label = f'L{record["frame_index"]:05d}'
        review_path = out / 'fov_review' / f'{label}.jpg'
        review_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(review_path, quality=92)
        band = np.arange(camera.height)[:, None] >= int(camera.height * .8)
        rows.append({'frame': record['frame_index'], 'name': record['name'],
                     'review': str(review_path),
                     'court_color_pixels_within_179': int(court.sum()),
                     'court_color_pixels_within_160': int((court & fovs[160]).sum()),
                     'court_color_pixels_between_160_and_179': int((court & ~fovs[160]).sum()),
                     'court_color_pixels_in_bottom_20pct_within_179': int((court & band).sum()),
                     'current_semantic_valid_in_bottom_20pct': int((old & band).sum()),
                     'fov_pixels_in_bottom_20pct': {str(fov): int((mask & band).sum())
                                                     for fov, mask in fovs.items()}})
    result = {'fixed_visual_camera_params': list(camera.params),
              'distorted_radii': {str(fov): camera.distorted_radius(fov) for fov in fovs},
              'rows': rows,
              'guard': 'Colors are only an exploratory court-region proxy. FOV contours come from a fitted visual camera, not lens calibration truth. OpenCV front-hemisphere ray inversion needs separate verification before any wider native fisheye reconstruction.'}
    write_json(out / 'fov_audit.json', result)
    return result


def run_fov_candidate_masks(work):
    """Prepare separate six-image masks for a 179-degree front hemisphere."""
    work = Path(work).resolve()
    root = work / 'global_sfm'
    scene = load_scene(root)
    records = sorted((record for record in scene['images'] if record['camera'] == 'L'
                      and record['frame_index'] in FRAMES),
                     key=lambda record: record['frame_index'])
    source_model = root / 'ground_joint_ba/equal_control/sparse/0'
    reconstruction = pycolmap.Reconstruction(source_model)
    source_camera = reconstruction.cameras[1]
    camera = Camera('L', source_camera.width, source_camera.height,
                    tuple(source_camera.params), (0, 0))
    fov160, fov179 = camera.mask(160) > 0, camera.mask(179) > 0
    protocol = {'version': 2, 'images': [record['name'] for record in records],
                'image_sha256': {record['name']: file_digest(record['source']) for record in records},
                'old_mask_sha256': {record['name']: file_digest(record['semantic_mask'])
                                    for record in records},
                'label_sha256': {record['name']: file_digest(
                    root / 'masks_v2/labels' / f'{record["name"]}.png') for record in records},
                'visual_model_hashes': {p.name: file_digest(p) for p in sorted(source_model.glob('*.bin'))},
                'old_fov_degrees': 160, 'candidate_fov_degrees': 179,
                'annulus_policy': 'Only court-color proposal outside old 160-degree FOV, not all sky/rim pixels',
                'bottom_restoration': 'Only court-color proposal in original device-labeled y=.8H..9H band',
                'camera_domain': 'Current fitted fisheye camera remains a candidate, not calibration truth',
                'uses_navigation_las_or_manual_labels': False,
                'scope': 'Six isolated L images. No production scene masks, feature caches, SfM, or LFS output overwritten.'}
    fingerprint = digest(protocol)
    out = root / FOV_CANDIDATE_OUTPUT
    out.mkdir(exist_ok=True)
    protocol_path = out / 'protocol.json'
    if protocol_path.exists():
        if json.loads(protocol_path.read_text())['fingerprint'] != fingerprint:
            raise ValueError('FOV candidate inputs changed')
    else:
        write_json(protocol_path, {**protocol, 'fingerprint': fingerprint,
                                   'frozen_at_utc': datetime.now(timezone.utc).isoformat()})
    band = ((np.arange(camera.height)[:, None] >= int(camera.height * RELEASE_START))
            & (np.arange(camera.height)[:, None] < int(camera.height * RELEASE_END)))
    rows = []
    for record in records:
        bgr = cv2.imread(record['source'], cv2.IMREAD_COLOR)
        old = cv2.imread(record['semantic_mask'], cv2.IMREAD_GRAYSCALE) >= 128
        reasons = cv2.imread(str(root / 'masks_v2/labels' / f'{record["name"]}.png'),
                             cv2.IMREAD_GRAYSCALE)
        proposal, _ = court_color_proposal(bgr, fov179)
        court = cv2.resize(proposal.astype(np.uint8), (camera.width, camera.height),
                           interpolation=cv2.INTER_NEAREST) > 0
        bottom_restore = (reasons == 3) & band & court & fov179
        baseline_valid = old & fov160
        candidate_valid = baseline_valid | ((old | bottom_restore) & court & fov179)
        new_pixels = candidate_valid & ~baseline_valid
        label = f'L{record["frame_index"]:05d}'
        path = out / 'masks' / f'{label}.png'
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(candidate_valid.astype(np.uint8) * 255).save(path)
        review = out / 'review' / f'{label}.jpg'
        _render_review(review, bgr, fov179, baseline_valid, proposal, new_pixels, label)
        rows.append({'name': record['name'], 'frame': record['frame_index'],
                     'mask': str(path), 'review': str(review),
                     'baseline_valid_pixels': int(baseline_valid.sum()),
                     'candidate_valid_pixels': int(candidate_valid.sum()),
                     'new_pixels_in_annulus': int((new_pixels & ~fov160).sum()),
                     'new_pixels_from_bottom_restore': int(bottom_restore.sum()),
                     'new_court_color_pixels': int((new_pixels & court).sum())})
        print(f'FOV179 candidate {label}: {int(new_pixels.sum())} additional pixels', flush=True)
    result = {'protocol_fingerprint': fingerprint, 'rows': rows,
              'guard': 'Candidate FOV179 is a numerical front-hemisphere limit, not verified lens coverage. Bottom restoration is court-color-only and six-image-only. Any added features still require round-trip ray checks and independent SfM comparison.'}
    write_json(out / 'mask_audit.json', result)
    return result
