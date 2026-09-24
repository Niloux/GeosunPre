"""Read-only audit of image-only evidence on frozen manual SfM queries.

The annotations are used solely to evaluate signals. This module does not
select a threshold, alter a feature, or feed an observation into SfM.
"""

from collections import Counter, defaultdict
from itertools import combinations
import json
from pathlib import Path

import cv2
import numpy as np

from .scene import digest, file_digest, write_json


OUTPUT = 'sfm_match_evidence'
PATCH_SIZES = (21, 41)


def _auc(positive, negative):
    """Probability that a randomly drawn positive scores above a negative."""
    positive = np.asarray(positive, dtype=float)
    negative = np.asarray(negative, dtype=float)
    if not len(positive) or not len(negative):
        return None
    delta = positive[:, None] - negative[None, :]
    return float((np.count_nonzero(delta > 0) + .5 * np.count_nonzero(delta == 0))
                 / delta.size)


def _quantiles(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return None
    return {'count': int(len(values)), 'median': float(np.median(values)),
            'p10': float(np.quantile(values, .1)),
            'p90': float(np.quantile(values, .9))}


def _patch(crop_path, xy, origin, side):
    image = cv2.imread(str(crop_path), cv2.IMREAD_GRAYSCALE)
    if image is None or image.shape != (384, 384):
        raise ValueError(f'Missing or changed static-review crop: {crop_path}')
    native = cv2.resize(image, (192, 192), interpolation=cv2.INTER_AREA)
    center = tuple((np.asarray(xy, dtype=float) - np.asarray(origin, dtype=float)
                    - .5).astype(np.float32))
    if any(c < side / 2 or c >= 192 - side / 2 for c in center):
        raise ValueError(f'Feature no longer centered in review crop: {crop_path}')
    return cv2.getRectSubPix(native, (side, side), center).astype(np.float32)


def _ncc(a, b):
    a = a - a.mean(); b = b - b.mean()
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a.ravel(), b.ravel()) / norm) if norm > 1e-8 else 0.


def _feature_subset(frontend, rows):
    needed = defaultdict(set)
    for row in rows:
        for obs in row['observations']:
            needed[obs['name']].add(int(obs['feature_index']))
    selected = {}
    for name, indices in sorted(needed.items()):
        entry = frontend['features'][name]
        with np.load(entry['path'], allow_pickle=False) as data:
            if str(data['token']) != entry['token']:
                raise ValueError(f'Changed feature cache: {name}')
            for index in indices:
                selected[(name, index)] = (float(data['scores'][index]),
                                           data['descriptors'][index].astype(float))
    return selected


def _pair_score(pair, left, right):
    with np.load(pair['path'], allow_pickle=False) as data:
        if str(data['token']) != pair['token']:
            raise ValueError(f'Changed LoMa match cache: {pair["path"]}')
        target = (left[1], right[1]) if pair['a'] == left[0] else (right[1], left[1])
        matches = data['matches']
        found = np.flatnonzero((matches[:, 0] == target[0]) & (matches[:, 1] == target[1]))
        if len(found) != 1:
            raise ValueError(f'Frozen closed-three-view edge missing: {left}, {right}')
        return float(data['scores'][found[0]])


def _score_row(row, features, pairs, crop_root):
    obs = row['observations']
    keys = [(o['name'], int(o['feature_index'])) for o in obs]
    detectors = [features[key][0] for key in keys]
    descriptors = [features[key][1] for key in keys]
    patches = {side: [_patch(crop_root / o['crop'], o['xy'], o['crop_origin'], side)
                      for o in obs] for side in PATCH_SIZES}
    matches = []; cosine = []; ncc = {side: [] for side in PATCH_SIZES}
    for a, b in combinations(range(3), 2):
        pair = pairs.get(frozenset((keys[a][0], keys[b][0])))
        if pair is None:
            raise ValueError(f'Missing frontend pair for {keys[a]}, {keys[b]}')
        matches.append(_pair_score(pair, keys[a], keys[b]))
        da, db = descriptors[a], descriptors[b]
        cosine.append(float(np.dot(da, db) / (np.linalg.norm(da) * np.linalg.norm(db))))
        for side in PATCH_SIZES:
            ncc[side].append(_ncc(patches[side][a], patches[side][b]))
    return {'id': row['id'], 'camera': row['camera'], 'time_bin': row['time_bin'],
            'gap_kind': row['gap_kind'], 'zone': row['zone'],
            'camera_mix': 'cross_LR' if len({key[0][0] for key in keys}) > 1 else 'single_camera',
            'min_detector_score': min(detectors),
            'min_match_score': min(matches),
            'min_descriptor_cosine': min(cosine),
            **{f'min_patch_ncc_{side}': min(ncc[side]) for side in PATCH_SIZES}}


def _summarize(rows):
    metrics = ('min_detector_score', 'min_match_score', 'min_descriptor_cosine',
               'min_patch_ncc_21', 'min_patch_ncc_41')
    classes = ('same_static', 'wrong_match', 'dynamic', 'ambiguous', 'occluded')
    summary = {'labels': dict(Counter(r['label'] for r in rows)), 'metrics': {}}
    for metric in metrics:
        by_label = {label: [r[metric] for r in rows if r['label'] == label]
                    for label in classes}
        result = {'by_label': {label: _quantiles(values)
                               for label, values in by_label.items()},
                  'auc_static_vs_wrong_or_dynamic': _auc(by_label['same_static'],
                                                         by_label['wrong_match'] + by_label['dynamic']),
                  'auc_static_vs_wrong': _auc(by_label['same_static'], by_label['wrong_match']),
                  'auc_static_vs_dynamic': _auc(by_label['same_static'], by_label['dynamic']),
                  'by_stratum': {}}
        for field in ('camera', 'gap_kind', 'camera_mix'):
            for value in sorted({r[field] for r in rows}):
                take = [r for r in rows if r[field] == value]
                positive = [r[metric] for r in take if r['label'] == 'same_static']
                negative = [r[metric] for r in take if r['label'] in ('wrong_match', 'dynamic')]
                result['by_stratum'][f'{field}/{value}'] = {
                    'static': len(positive), 'wrong_or_dynamic': len(negative),
                    'auc': _auc(positive, negative)}
        summary['metrics'][metric] = result
    return summary


def run_match_evidence_audit(work):
    root = Path(work).resolve() / 'global_sfm'
    review = root / 'static_validation'
    out = root / OUTPUT
    out.mkdir(exist_ok=True)
    frontend = json.loads((root / 'frontend.json').read_text())
    rendered_path = review / 'rendered_candidates.json'
    annotations_path = review / 'basketball-static-annotations.json'
    rendered = json.loads(rendered_path.read_text())
    annotations = json.loads(annotations_path.read_text())
    if (annotations['protocol_fingerprint'] != rendered['protocol_fingerprint']
            or annotations['candidate_fingerprint'] != rendered['candidate_fingerprint']):
        raise ValueError('Manual labels do not belong to these frozen candidates')
    protocol = {'version': 1,
                'frontend_sha256': file_digest(root / 'frontend.json'),
                'rendered_candidates_sha256': file_digest(rendered_path),
                'annotations_sha256': file_digest(annotations_path),
                'patch_crop_source': 'Unmarked 384x384 native-image review JPG, downsampled to 192x192',
                'patch_sizes_original_px': list(PATCH_SIZES),
                'measurements': ['Minimum LoMa detector score of three points',
                                 'Minimum direct LoMa match score of three pairs',
                                 'Minimum raw descriptor cosine of three pairs',
                                 'Minimum zero-mean, unwarped local patch NCC of three pairs'],
                'uses_pose_intrinsics_las_or_sfm_error': False,
                'manual_labels_used_for': 'Evaluation summaries only; no threshold fitting or SfM filtering',
                'scope': 'Exploratory evidence on one stratified capture; closed-three-view LoMa candidates already have high matching confidence.'}
    fingerprint = digest(protocol)
    protocol_path = out / 'protocol.json'
    if protocol_path.exists():
        if json.loads(protocol_path.read_text())['fingerprint'] != fingerprint:
            raise ValueError('Existing evidence audit belongs to different frozen inputs')
    else:
        write_json(protocol_path, {**protocol, 'fingerprint': fingerprint})
    pairs = {frozenset((pair['a'], pair['b'])): pair for pair in frontend['pairs']}
    features = _feature_subset(frontend, rendered['rows'])
    rows = []
    for row in rendered['rows']:
        evidence = _score_row(row, features, pairs, review)
        evidence['label'] = annotations['annotations'].get(row['id'], {}).get('label', 'unreviewed')
        rows.append(evidence)
    summary = _summarize(rows)
    result = {'protocol_fingerprint': fingerprint, 'rows': rows, 'summary': summary,
              'interpretation_guard': 'The scored manual set is for diagnosis only. No learned or tuned rule may be validated on these same labels; raw NCC is sensitive to viewpoint and exposure.'}
    write_json(out / 'results.json', result)
    return result
