"""Read-only spatial audit of frozen SfM and manually verified residuals."""

from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from ..scene import digest, file_digest, write_json


OUTPUT = 'sfm_residual_structure'
MAX_AUTOMATIC_ERROR_PX = 5.
MIN_OTHER_QUERIES_PER_IMAGE = 3


def _distribution(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {'count': 0, 'median': None, 'p90': None}
    return {'count': int(len(values)), 'median': float(np.median(values)),
            'p90': float(np.quantile(values, .9))}


def _group_manual(rows, field):
    return {value: _distribution([row['query_error_px'] for row in rows
                                  if row[field] == value and row['query_error_px'] is not None])
            for value in sorted({row[field] for row in rows})}


def run_residual_structure_audit(work):
    root = Path(work).resolve() / 'global_sfm'
    base = root / 'sfm_track_cleanup'
    manual_path = base / 'equal_control_manual_rows.json'
    automatic_path = base / 'validation/stages/equal_control/triangle_validation/validation_observations.npz'
    review_path = root / 'static_validation/rendered_candidates.json'
    model = root / 'ground_joint_ba/equal_control/sparse/0'
    protocol = {'version': 1, 'model_hashes': {p.name: file_digest(p)
                                               for p in sorted(model.glob('*.bin'))},
                'manual_rows_sha256': file_digest(manual_path),
                'automatic_queries_sha256': file_digest(automatic_path),
                'review_rows_sha256': file_digest(review_path),
                'automatic_support_qualified': True,
                'automatic_max_query_error_px': MAX_AUTOMATIC_ERROR_PX,
                'per_image_min_other_tracks': MIN_OTHER_QUERIES_PER_IMAGE,
                'scope': 'Post hoc diagnostic; automatic queries are not static truth. The vector subtraction is not a camera-pose update or SfM candidate.'}
    fingerprint = digest(protocol)
    out = root / OUTPUT
    out.mkdir(exist_ok=True)
    existing = out / 'protocol.json'
    if existing.exists():
        if json.loads(existing.read_text())['fingerprint'] != fingerprint:
            raise ValueError('Existing residual audit belongs to different frozen inputs')
    else:
        write_json(existing, {**protocol, 'fingerprint': fingerprint})

    manual = json.loads(manual_path.read_text())
    review = {r['id']: r for r in json.loads(review_path.read_text())['rows']}
    with np.load(automatic_path, allow_pickle=False) as raw:
        queries = {key: raw[key] for key in raw.files}
    good = (queries['support_ok'] & np.isfinite(queries['error'])
            & (queries['error'] < MAX_AUTOMATIC_ERROR_PX))
    indices = defaultdict(list)
    for index in np.flatnonzero(good):
        indices[str(queries['name'][index])].append(int(index))

    comparisons = []
    for row in manual:
        if row['query_error_px'] is None:
            continue
        source = review[row['id']]
        name, track = source['query'][0], source['track_id']
        others = [index for index in indices[name]
                  if str(queries['track_id'][index]) != track]
        if len(others) < MIN_OTHER_QUERIES_PER_IMAGE:
            continue
        residual = np.asarray(row['predicted_xy']) - np.asarray(row['observed_xy'])
        automatic_residual = queries['predicted'][others] - queries['observed'][others]
        image_offset = np.median(automatic_residual, axis=0)
        comparisons.append({'id': row['id'], 'image': name, 'other_tracks': len(others),
                            'manual_error_px': float(np.linalg.norm(residual)),
                            'after_subtracting_other_track_median_px': float(np.linalg.norm(residual - image_offset)),
                            'manual_residual_xy': residual.tolist(),
                            'other_track_median_xy': image_offset.tolist()})
    before = np.asarray([r['manual_error_px'] for r in comparisons])
    after = np.asarray([r['after_subtracting_other_track_median_px'] for r in comparisons])

    center = np.array([1500., 2000.])
    observed = queries['observed']
    displacement = queries['predicted'] - observed
    direction = observed - center
    direction /= np.maximum(np.linalg.norm(direction, axis=1)[:, None], 1e-12)
    radial = np.sum(displacement * direction, axis=1)
    tangential = -displacement[:, 0] * direction[:, 1] + displacement[:, 1] * direction[:, 0]
    zones = {}
    for camera in ('L', 'R'):
        for zone in ('center', 'middle', 'edge'):
            take = good & (queries['camera'] == camera) & (queries['zone'] == zone)
            zones[f'{camera}/{zone}'] = {
                'automatic_queries': int(take.sum()),
                'radial_median_px': float(np.median(radial[take])),
                'radial_mean_px': float(np.mean(radial[take])),
                'tangential_median_px': float(np.median(tangential[take])),
                'tangential_mean_px': float(np.mean(tangential[take]))}
    result = {'protocol_fingerprint': fingerprint,
              'manual_total': len(manual),
              'manual_finite': sum(r['query_error_px'] is not None for r in manual),
              'automatic_total': len(queries['name']),
              'automatic_support_and_error_qualified': int(good.sum()),
              'manual_error_groups': {field: _group_manual(manual, field)
                                      for field in ('camera', 'zone', 'gap_kind',
                                                    'camera_mix', 'time_bin')},
              'per_image_pseudo_correction': {
                  'manual_cases_with_other_tracks': len(comparisons),
                  'before': _distribution(before), 'after': _distribution(after),
                  'improved_cases': int(np.count_nonzero(after < before)),
                  'median_image_offset_norm_px': float(np.median([
                      np.linalg.norm(r['other_track_median_xy']) for r in comparisons]))},
              'automatic_radial_by_camera_zone': zones,
              'comparisons': comparisons,
              'guard': 'Automatic query matches may be dynamic or wrong. No manual query track contributes to its own image offset. A 2-D residual vector is not a camera correction, and this is not an independent acceptance test.'}
    write_json(out / 'results.json', result)
    return result
