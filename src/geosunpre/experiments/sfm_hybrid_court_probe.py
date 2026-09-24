"""Exploratory court-color proposal and native-plus-virtual LoMa front end.

The color proposal is deliberately a reviewable image-region prior, not a
semantic ground-truth label. Candidate features remain in native fisheye pixels.
"""

from datetime import datetime, timezone
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from ..scene import digest, file_digest, load_scene, write_json


OUTPUT = 'sfm_hybrid_court_probe'
FRAMES = (120, 121, 122, 150, 151, 152)
MASK_SIZE = (750, 1000)
HSV_S_MIN = 70
HSV_V_MIN = 55
NATIVE_Y_MIN = .47
MIN_COMPONENT_PIXELS = 500


def court_color_proposal(bgr, valid):
    """Broad color/position proposal for the blue, teal and red court paving."""
    small = cv2.resize(bgr, MASK_SIZE, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    court_hue = (((hue >= 72) & (hue <= 128)) | (hue <= 10) | (hue >= 170))
    chromatic = (court_hue & (saturation >= HSV_S_MIN) & (value >= HSV_V_MIN))
    yy = np.arange(MASK_SIZE[1])[:, None]
    chromatic &= yy >= int(MASK_SIZE[1] * NATIVE_Y_MIN)
    valid_small = cv2.resize(valid.astype(np.uint8), MASK_SIZE,
                             interpolation=cv2.INTER_NEAREST) > 0
    chromatic &= valid_small
    cleaned = cv2.morphologyEx(chromatic.astype(np.uint8), cv2.MORPH_CLOSE,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    components, labels, areas, _ = cv2.connectedComponentsWithStats(cleaned, 8)
    kept = np.zeros_like(cleaned, dtype=np.uint8)
    for index in range(1, components):
        if areas[index, cv2.CC_STAT_AREA] >= MIN_COMPONENT_PIXELS:
            kept[labels == index] = 1
    proposal = cv2.dilate(kept,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    proposal &= valid_small.astype(np.uint8)
    return proposal.astype(bool), {'raw_color_pixels': int(chromatic.sum()),
                                   'kept_component_pixels': int(kept.sum()),
                                   'proposal_pixels': int(proposal.sum()),
                                   'components_kept': int(np.sum(areas[1:, cv2.CC_STAT_AREA]
                                                             >= MIN_COMPONENT_PIXELS))}


def _render_review(path, bgr, proposal, label):
    original = Image.fromarray(cv2.cvtColor(cv2.resize(bgr, MASK_SIZE), cv2.COLOR_BGR2RGB))
    overlay = np.asarray(original).copy()
    overlay[proposal] = (overlay[proposal] * .45 + np.array([35, 235, 215]) * .55).astype(np.uint8)
    canvas = Image.new('RGB', (1500, 1040), (24, 28, 35))
    canvas.paste(original, (0, 40))
    canvas.paste(Image.fromarray(overlay), (750, 40))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 10), f'{label}: original', fill=(245, 245, 245))
    draw.text((760, 10), f'{label}: court-color proposal (cyan)', fill=(245, 245, 245))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=90)


def run_court_region_probe(work):
    work = Path(work).resolve()
    root = work / 'global_sfm'
    scene = load_scene(root)
    records = [r for r in scene['images'] if r['camera'] == 'L'
               and r['frame_index'] in FRAMES]
    if len(records) != len(FRAMES):
        raise ValueError('Expected all six court-facing L images')
    records.sort(key=lambda record: record['frame_index'])
    protocol = {'version': 1, 'images': [record['name'] for record in records],
                'image_sha256': {r['name']: file_digest(r['source']) for r in records},
                'semantic_mask_sha256': {r['name']: file_digest(r['semantic_mask']) for r in records},
                'size': MASK_SIZE, 'hue_degrees_cv8': '72..128 or 0..10 or 170..179',
                'saturation_min': HSV_S_MIN, 'value_min': HSV_V_MIN,
                'native_y_min_fraction': NATIVE_Y_MIN,
                'close_kernel': 17, 'open_kernel': 5, 'min_component_pixels': MIN_COMPONENT_PIXELS,
                'dilation_kernel': 9,
                'uses_pose_las_or_manual_labels': False,
                'scope': 'Reviewable color/position proposal on six L images only; no SfM or LFS change.'}
    fingerprint = digest(protocol)
    out = root / OUTPUT
    out.mkdir(exist_ok=True)
    protocol_path = out / 'court_protocol.json'
    if protocol_path.exists():
        if json.loads(protocol_path.read_text())['fingerprint'] != fingerprint:
            raise ValueError('Court-region probe inputs changed')
    else:
        write_json(protocol_path, {**protocol, 'fingerprint': fingerprint,
                                   'frozen_at_utc': datetime.now(timezone.utc).isoformat()})
    rows = []
    for record in records:
        bgr = cv2.imread(record['source'], cv2.IMREAD_COLOR)
        valid = cv2.imread(record['semantic_mask'], cv2.IMREAD_GRAYSCALE) >= 128
        if bgr is None or valid.shape != bgr.shape[:2]:
            raise ValueError(f'Missing native image/mask: {record["name"]}')
        proposal, counts = court_color_proposal(bgr, valid)
        label = f'L{record["frame_index"]:05d}'
        mask_path = out / 'court_masks' / f'{label}.png'
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(proposal.astype(np.uint8) * 255).save(mask_path)
        review_path = out / 'court_review' / f'{label}.jpg'
        _render_review(review_path, bgr, proposal, label)
        rows.append({'name': record['name'], 'frame': record['frame_index'],
                     'mask': str(mask_path), 'review': str(review_path), **counts})
        print(f'Court-region proposal {label}: {counts["proposal_pixels"]} pixels', flush=True)
    result = {'protocol_fingerprint': fingerprint, 'rows': rows,
              'guard': 'Color and lower-image location are only candidate court-region cues. They do not establish scene geometry, staticness, point accuracy, or full-scene generalization.'}
    write_json(out / 'court_review.json', result)
    return result
