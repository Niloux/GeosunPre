"""Image-only native-coordinate repeatability probe for cached LoMa pairs.

LK flow is an independent local correspondence proposal, not ground truth.
This never changes cached LoMa observations or a reconstruction.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import pycolmap

from .scene import digest, file_digest, write_json


OUTPUT = 'sfm_lk_probe'
PAIR_FRAMES = (('R', 65), ('R', 70), ('R', 145), ('R', 195), ('R', 245),
               ('L', 70), ('L', 120), ('L', 150), ('L', 195), ('L', 245))
MIN_LOMA_SCORE = .9
HALF_WIDTH, HALF_HEIGHT = 1500, 2000
MAX_FORWARD_BACKWARD_ERROR_ORIGINAL_PX = 1.


def _frame(name):
    return int(Path(name).stem.rsplit('_', 1)[-1].split('-')[0])


def _distribution(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {'count': 0, 'median': None, 'p90': None,
                'fraction_gt_2px': None, 'fraction_gt_5px': None}
    return {'count': int(len(values)),
            'median': float(np.median(values)),
            'p90': float(np.quantile(values, .9)),
            'fraction_gt_2px': float(np.mean(values > 2)),
            'fraction_gt_5px': float(np.mean(values > 5))}


def _flow(source, target, points):
    return cv2.calcOpticalFlowPyrLK(
        source, target, points, None, winSize=(21, 21), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, .01),
        minEigThreshold=1e-4)


def _pair_probe(entry, frontend, records, database):
    with np.load(entry['path'], allow_pickle=False) as raw:
        if str(raw['token']) != entry['token']:
            raise ValueError(f'Changed LoMa match cache: {entry["path"]}')
        matches = raw['matches'][raw['scores'] >= MIN_LOMA_SCORE]
    with np.load(frontend['features'][entry['a']]['path'], allow_pickle=False) as raw:
        if str(raw['token']) != frontend['features'][entry['a']]['token']:
            raise ValueError(f'Changed feature cache: {entry["a"]}')
        first = raw['keypoints'][matches[:, 0]]
    with np.load(frontend['features'][entry['b']]['path'], allow_pickle=False) as raw:
        if str(raw['token']) != frontend['features'][entry['b']]['token']:
            raise ValueError(f'Changed feature cache: {entry["b"]}')
        second = raw['keypoints'][matches[:, 1]]

    a_id, b_id = records[entry['a']]['image_id'], records[entry['b']]['image_id']
    if not database.exists_two_view_geometry(a_id, b_id):
        raise ValueError(f'No training two-view geometry for {entry["a"]}, {entry["b"]}')
    accepted = {tuple(map(int, pair)) for pair in
                database.read_two_view_geometry(a_id, b_id).inlier_matches}
    training = np.asarray([tuple(map(int, pair)) in accepted for pair in matches], dtype=bool)

    images = []
    for name in (entry['a'], entry['b']):
        original = cv2.imread(records[name]['source'], cv2.IMREAD_GRAYSCALE)
        if original is None or original.shape != (4000, 3000):
            raise ValueError(f'Missing or changed native JPEG: {name}')
        images.append(cv2.resize(original, (HALF_WIDTH, HALF_HEIGHT),
                                 interpolation=cv2.INTER_AREA))
    image_a, image_b = images
    # OpenCV centers are 0-based; cached COLMAP coordinates place the first
    # pixel center at .5. The resize maps x_native to x_half = x_native/2-.5.
    a = (first / 2 - .5).astype(np.float32).reshape(-1, 1, 2)
    b = (second / 2 - .5).astype(np.float32).reshape(-1, 1, 2)
    flow_b, ok_ab, _ = _flow(image_a, image_b, a)
    back_a, ok_ba, _ = _flow(image_b, image_a, flow_b)
    flow_a, ok_ba_direct, _ = _flow(image_b, image_a, b)
    back_b, ok_ab_direct, _ = _flow(image_a, image_b, flow_a)
    forward_backward_a = np.linalg.norm((back_a - a).reshape(-1, 2), axis=1) * 2
    forward_backward_b = np.linalg.norm((back_b - b).reshape(-1, 2), axis=1) * 2
    valid = (ok_ab.ravel().astype(bool) & ok_ba.ravel().astype(bool)
             & ok_ba_direct.ravel().astype(bool) & ok_ab_direct.ravel().astype(bool)
             & (forward_backward_a < MAX_FORWARD_BACKWARD_ERROR_ORIGINAL_PX)
             & (forward_backward_b < MAX_FORWARD_BACKWARD_ERROR_ORIGINAL_PX))
    disagreement_a = np.linalg.norm((flow_a - a).reshape(-1, 2), axis=1) * 2
    disagreement_b = np.linalg.norm((flow_b - b).reshape(-1, 2), axis=1) * 2
    disagreement = np.maximum(disagreement_a, disagreement_b)
    groups = {}
    for name, take in (('all_high_confidence', np.ones(len(matches), dtype=bool)),
                       ('accepted_training_geometry', training)):
        measured = take & valid
        groups[name] = {'matches': int(take.sum()),
                        'bidirectionally_repeatable': int(measured.sum()),
                        'repeatable_fraction': float(measured.sum() / take.sum()),
                        'symmetric_lk_vs_loma_px': _distribution(disagreement[measured])}
    return {'camera': entry['a'][0], 'frame': _frame(entry['a']),
            'a': entry['a'], 'b': entry['b'], 'groups': groups,
            'training_inlier_fingerprint': digest(sorted(accepted))}


def run_lk_probe(work):
    cv2.setNumThreads(4)
    root = Path(work).resolve() / 'global_sfm'
    frontend = json.loads((root / 'frontend.json').read_text())
    scene = json.loads((root / 'manifest.json').read_text())
    records = {record['name']: record for record in scene['images']}
    chosen = []
    for camera, frame in PAIR_FRAMES:
        found = [entry for entry in frontend['pairs']
                 if entry['a'][0] == camera and entry['b'][0] == camera
                 and _frame(entry['a']) == frame and _frame(entry['b']) == frame + 1]
        if len(found) != 1:
            raise ValueError(f'Expected one adjacent {camera} pair for frame {frame}')
        chosen.append(found[0])
    protocol = {'version': 1, 'frontend_sha256': file_digest(root / 'frontend.json'),
                'scene_fingerprint': scene['fingerprint'],
                'pairs': [(entry['a'], entry['b'], entry['token']) for entry in chosen],
                'image_sha256': {name: file_digest(records[name]['source'])
                                 for name in sorted({e[key] for e in chosen for key in ('a', 'b')})},
                'minimum_loma_match_score': MIN_LOMA_SCORE,
                'flow_image_size': [HALF_WIDTH, HALF_HEIGHT],
                'flow_window': [21, 21], 'flow_max_level': 3,
                'flow_max_forward_backward_original_px': MAX_FORWARD_BACKWARD_ERROR_ORIGINAL_PX,
                'uses_pose_intrinsics_las_or_manual_labels': False,
                'scope': 'Ten selected adjacent same-camera pairs. Bidirectional LK is an image-only consistency probe, not pixel ground truth or a reconstruction change.'}
    fingerprint = digest(protocol)
    output = root / OUTPUT
    output.mkdir(exist_ok=True)
    protocol_path = output / 'protocol.json'
    if protocol_path.exists():
        if json.loads(protocol_path.read_text())['fingerprint'] != fingerprint:
            raise ValueError('Existing LK probe belongs to different frozen images/features')
    else:
        write_json(protocol_path, {**protocol, 'fingerprint': fingerprint})
    with pycolmap.Database.open(root / 'database.db') as database:
        rows = [_pair_probe(entry, frontend, records, database) for entry in chosen]
    summary = {}
    for camera in ('L', 'R'):
        values = [row['groups']['accepted_training_geometry'] for row in rows
                  if row['camera'] == camera]
        repeatable = sum(v['bidirectionally_repeatable'] for v in values)
        disagreeing = sum(v['bidirectionally_repeatable']
                          * v['symmetric_lk_vs_loma_px']['fraction_gt_2px']
                          for v in values if v['bidirectionally_repeatable'])
        summary[camera] = {'selected_pairs': len(values),
                           'accepted_high_confidence_matches': sum(v['matches'] for v in values),
                           'bidirectionally_repeatable': repeatable,
                           'repeatable_pairs_with_at_least_50_matches': sum(
                               v['bidirectionally_repeatable'] >= 50 for v in values),
                           'fraction_gt_2px_among_repeatable': (float(disagreeing / repeatable)
                                                               if repeatable else None)}
    result = {'protocol_fingerprint': fingerprint, 'rows': rows, 'summary': summary,
              'interpretation_guard': 'LK may fail on parallax, reflections, repeated texture, vegetation or exposure changes. Failed LK is not a LoMa error; successful forward-backward agreement is not proof of ground truth. Only accepted geometry edges are training-side. No SfM model or feature cache was changed.'}
    write_json(output / 'results.json', result)
    return result
