"""Six-image native LoMa probe for conservative court-only FOV expansion."""

from datetime import datetime, timezone
import json
from pathlib import Path

import cv2
import numpy as np
import pycolmap

from ..camera import Camera
from ..frontend import FrontendConfig, LoMaFrontend
from ..geometry import epipolar_pixel_error, skew, stats
from ..scene import digest, file_digest, load_scene, write_json
from .sfm_hybrid_court_probe import FRAMES, court_color_proposal
from .sfm_resolution_probe import _cached_feature, _cached_match
from .sfm_virtual_cycle_probe import _closed_cycles, _three_view_geometry


OUTPUT = 'sfm_fov179_court_frontend_probe_v2'
CONFIG = FrontendConfig(detector_size=1536, num_keypoints=3072)
MATCH_CONFIDENCE = .9


def _region_take(points, region, camera):
    ij = np.floor(points / [camera.width, camera.height]
                  * [region.shape[1], region.shape[0]]).astype(int)
    ij[:, 0] = np.clip(ij[:, 0], 0, region.shape[1] - 1)
    ij[:, 1] = np.clip(ij[:, 1], 0, region.shape[0] - 1)
    return region[ij[:, 1], ij[:, 0]]


def _pair_summary(names, features, match, images_by_name, camera, region):
    a, b = names
    pairs = match['matches']
    pa, pb = [images_by_name[name].cam_from_world().matrix() for name in names]
    R = pb[:, :3] @ pa[:, :3].T
    t = pb[:, 3] - R @ pa[:, 3]
    xy_a = features[a]['keypoints'][pairs[:, 0]].astype(float)
    xy_b = features[b]['keypoints'][pairs[:, 1]].astype(float)
    error = epipolar_pixel_error(camera, camera, xy_a, xy_b, skew(t) @ R)
    high = match['scores'] >= MATCH_CONFIDENCE
    court = _region_take(xy_a, region[a], camera) & _region_take(xy_b, region[b], camera)
    result = {'raw_matches': len(pairs)}
    for label, keep in (('all_high_confidence', high),
                        ('court_proposal_both_images', high & court)):
        values = error[keep]
        result[label] = {'matches': int(keep.sum()),
                         'fixed_pose_epipolar_error_px': stats(values),
                         'fraction_le_3px': float(np.mean(values <= 3)) if len(values) else None}
    return result


def run_fov_frontend_probe(work):
    cv2.setNumThreads(4)
    work = Path(work).resolve()
    root = work / 'global_sfm'
    candidate_root = root / 'sfm_fov179_court_candidate'
    mask_audit = json.loads((candidate_root / 'mask_audit.json').read_text())
    mask_protocol = json.loads((candidate_root / 'protocol.json').read_text())
    if mask_audit['protocol_fingerprint'] != mask_protocol['fingerprint']:
        raise ValueError('Candidate mask audit/protocol mismatch')
    scene = load_scene(root)
    records = {record['name']: record for record in scene['images']}
    names = [row['name'] for row in mask_audit['rows']]
    if tuple(int(records[name]['frame_index']) for name in names) != FRAMES:
        raise ValueError('Candidate image order drifted')
    baseline = json.loads((root / 'frontend.json').read_text())
    source_model = root / 'ground_joint_ba/equal_control/sparse/0'
    model_hashes = {p.name: file_digest(p) for p in sorted(source_model.glob('*.bin'))}
    triplets = [tuple(names[index:index + 3]) for index in (0, 3)]
    pair_entries = {triple: tuple(next(e for e in baseline['pairs']
                                       if e['a'] == triple[i] and e['b'] == triple[j])
                                  for i, j in ((0, 1), (0, 2), (1, 2)))
                    for triple in triplets}
    candidate_mask_paths = {row['name']: row['mask'] for row in mask_audit['rows']}
    protocol = {'version': 2, 'candidate_mask_protocol_fingerprint': mask_protocol['fingerprint'],
                'candidate_mask_sha256': {name: file_digest(path)
                                          for name, path in candidate_mask_paths.items()},
                'baseline_frontend_sha256': file_digest(root / 'frontend.json'),
                'fixed_visual_model_hashes': model_hashes,
                'triplets': triplets,
                'baseline_match_tokens': [entry['token'] for triple in triplets
                                          for entry in pair_entries[triple]],
                'candidate_scene_fov_degrees': 179,
                'feature_filter': 'Fitted visual camera valid at 179 degrees and native ray roundtrip <=0.1 pixel; compensates subpixel mask-cell boundary',
                'frontend_config': CONFIG.__dict__,
                'match_confidence_min': MATCH_CONFIDENCE,
                'region': 'Same HSV color and lower-screen court proposal for both candidates; not a ground-truth segmentation',
                'uses_navigation_las_or_manual_labels': False,
                'scope': 'Six L images only; compare features/matches and fixed-pose three-view support. No candidate SfM or LFS training.'}
    fingerprint = digest(protocol)
    output = root / OUTPUT
    output.mkdir(exist_ok=True)
    protocol_path = output / 'protocol.json'
    if protocol_path.exists():
        if json.loads(protocol_path.read_text())['fingerprint'] != fingerprint:
            raise ValueError('Existing FOV frontend probe belongs to changed inputs')
    else:
        write_json(protocol_path, {**protocol, 'fingerprint': fingerprint,
                                   'frozen_at_utc': datetime.now(timezone.utc).isoformat()})
    reconstruction = pycolmap.Reconstruction(source_model)
    images_by_name = {image.name: image for image in reconstruction.images.values()}
    fitted = reconstruction.cameras[1]
    camera = Camera('L', fitted.width, fitted.height, tuple(fitted.params), (0, 0))
    candidate_scene = {**scene, 'fov_degrees': 179}
    candidate_records = {name: {**records[name], 'semantic_mask': candidate_mask_paths[name]}
                         for name in names}
    frontend = LoMaFrontend(work, candidate_scene, CONFIG)
    old_features = {name: _cached_feature(baseline['features'][name]) for name in names}
    new_features, regions, feature_rows = {}, {}, []
    fov160 = camera.mask(160) > 0
    for index, name in enumerate(names, 1):
        record = candidate_records[name]
        raw_feature = frontend.extract(record)
        bgr = cv2.imread(record['source'], cv2.IMREAD_COLOR)
        regions[name], _ = court_color_proposal(bgr, camera.mask(179) > 0)
        old_xy = old_features[name]['keypoints'].astype(float)
        raw_xy = raw_feature['keypoints'].astype(float)
        rays = camera.rays(raw_xy)
        raw_roundtrip = np.linalg.norm(camera.project(rays) - raw_xy, axis=1)
        safe = camera.valid(raw_xy, 179) & np.isfinite(raw_roundtrip) & (raw_roundtrip <= .1)
        if safe.sum() < 3000:
            raise ValueError(f'Too many candidate points fail fitted ray checks: {name}, {safe.sum()}')
        if safe.all():
            new_features[name] = raw_feature
        else:
            token = digest({'raw_feature_token': str(raw_feature['token']),
                            'filter': 'fitted_camera_valid179_roundtrip_le_0.1px',
                            'visual_model_hashes': model_hashes})
            new_features[name] = {key: raw_feature[key][safe] for key in
                                  ('keypoints', 'normalized_keypoints', 'descriptors', 'scores')}
            new_features[name]['token'] = np.array(token)
        new_xy = new_features[name]['keypoints'].astype(float)
        roundtrip = raw_roundtrip[safe]
        old_region = _region_take(old_xy, regions[name], camera)
        new_region = _region_take(new_xy, regions[name], camera)
        old_fov = _region_take(old_xy, fov160, camera)
        new_fov = _region_take(new_xy, fov160, camera)
        feature_rows.append({'name': name, 'frame': records[name]['frame_index'],
                             'baseline_features': len(old_xy), 'candidate_features': len(new_xy),
                             'baseline_court_proposal_features': int(old_region.sum()),
                             'candidate_court_proposal_features': int(new_region.sum()),
                             'baseline_outside_fov160': int((~old_fov).sum()),
                             'candidate_outside_fov160': int((~new_fov).sum()),
                             'candidate_rejected_at_fitted_ray_boundary': int((~safe).sum()),
                             'candidate_ray_roundtrip_px': stats(roundtrip),
                             'candidate_ray_roundtrip_max_px': float(roundtrip.max()),
                             'candidate_token': str(new_features[name]['token'])})
        print(f'FOV frontend features {index}/{len(names)} {name}', flush=True)
    rows = []
    for index, triple in enumerate(triplets, 1):
        old_matches = [_cached_match(entry) for entry in pair_entries[triple]]
        new_matches = [frontend.match(new_features[entry['a']], new_features[entry['b']])
                       for entry in pair_entries[triple]]
        old_cycles = _closed_cycles(old_matches)
        new_cycles = _closed_cycles(new_matches)
        edges = []
        for edge_index, entry in enumerate(pair_entries[triple]):
            pair = (entry['a'], entry['b'])
            edges.append({'a': pair[0], 'b': pair[1],
                          'baseline': _pair_summary(pair, old_features, old_matches[edge_index],
                                                    images_by_name, camera, regions),
                          'candidate': _pair_summary(pair, new_features, new_matches[edge_index],
                                                     images_by_name, camera, regions),
                          'candidate_match_token': str(new_matches[edge_index]['token'])})
        rows.append({'start_frame': records[triple[0]]['frame_index'], 'images': triple,
                     'edges': edges,
                     'baseline': _three_view_geometry(triple, old_features, old_cycles,
                                                      images_by_name, camera, regions[triple[0]]),
                     'candidate': _three_view_geometry(triple, new_features, new_cycles,
                                                       images_by_name, camera, regions[triple[0]])})
        print(f'FOV frontend triplet {index}/{len(triplets)}', flush=True)
    result = {'protocol_fingerprint': fingerprint, 'feature_rows': feature_rows, 'rows': rows,
              'cache_stats': frontend.cache_stats,
              'guard': 'Fixed poses were trained from baseline matches and may favor them. Court color is a rough region proposal. Three-view closed matches can still be wrong, and this probe has not reconstructed candidate poses or tested 3DGS.'}
    write_json(output / 'results.json', result)
    return result
