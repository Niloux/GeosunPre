"""Conservative image-only training-point refinement on one local SfM segment.

Each modified mapped observation needs two agreeing, forward/backward-stable
nearby-view flow proposals. Entire pre-geometry heldout tracks remain isolated.
"""

from collections import Counter, defaultdict
from functools import lru_cache
import json
from pathlib import Path

import cv2
import numpy as np
import pycolmap

from .local_data import load_local
from .local_evaluation import evaluate_reconstruction, paired_common_support
from .local_reconstruction import assert_holdout_isolated, bounded_bundle_adjustment
from .scene import digest, file_digest, write_json
from .transfer import topology_signature


OUTPUT = 'sfm_lk_refinement'
SEGMENT = 'normal_070_094'
SOURCE = f'local_sfm/{SEGMENT}/stages/principal/sparse/0'
CONFIG = {'flow_width': 1500, 'flow_height': 2000,
          'lk_window': [21, 21], 'lk_max_level': 3,
          'max_forward_backward_original_px': 1.,
          'max_symmetric_lk_loma_original_px': 4.,
          'minimum_independent_neighbor_proposals': 2,
          'max_proposal_disagreement_original_px': 1.,
          'max_consensus_shift_original_px': 2.,
          'applied_consensus_fraction': .5}


def _hashes(directory):
    return {path.name: file_digest(path) for path in sorted(Path(directory).glob('*.bin'))}


def _index_by_point(image):
    indices = defaultdict(list)
    for index, observation in enumerate(image.points2D):
        if observation.has_point3D():
            indices[int(observation.point3D_id)].append(index)
    return {point_id: positions[0] for point_id, positions in indices.items()
            if len(positions) == 1}


def _flow(source, target, points):
    return cv2.calcOpticalFlowPyrLK(
        source, target, points, None, winSize=tuple(CONFIG['lk_window']),
        maxLevel=CONFIG['lk_max_level'],
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, .01),
        minEigThreshold=1e-4)


def _pair_proposals(image_a, image_b, records, image_gray):
    a_index, b_index = _index_by_point(image_a), _index_by_point(image_b)
    shared = sorted(a_index.keys() & b_index.keys())
    if not shared:
        return [], {'shared_training_points': 0, 'bidirectional_flow_stable': 0,
                    'symmetric_lk_loma_within_4px': 0}
    a_ids = [a_index[point] for point in shared]
    b_ids = [b_index[point] for point in shared]
    original_a = np.asarray([image_a.points2D[index].xy for index in a_ids])
    original_b = np.asarray([image_b.points2D[index].xy for index in b_ids])
    a = (original_a / 2 - .5).astype(np.float32).reshape(-1, 1, 2)
    b = (original_b / 2 - .5).astype(np.float32).reshape(-1, 1, 2)
    gray_a, gray_b = image_gray(records[image_a.name]['source']), image_gray(records[image_b.name]['source'])
    moved_b, ok_ab, _ = _flow(gray_a, gray_b, a)
    returned_a, ok_ba, _ = _flow(gray_b, gray_a, moved_b)
    moved_a, ok_ba_direct, _ = _flow(gray_b, gray_a, b)
    returned_b, ok_ab_direct, _ = _flow(gray_a, gray_b, moved_a)
    fb_a = np.linalg.norm((returned_a - a).reshape(-1, 2), axis=1) * 2
    fb_b = np.linalg.norm((returned_b - b).reshape(-1, 2), axis=1) * 2
    stable = (ok_ab.ravel().astype(bool) & ok_ba.ravel().astype(bool)
              & ok_ba_direct.ravel().astype(bool) & ok_ab_direct.ravel().astype(bool)
              & (fb_a < CONFIG['max_forward_backward_original_px'])
              & (fb_b < CONFIG['max_forward_backward_original_px']))
    proposed_a = (moved_a.reshape(-1, 2) + .5) * 2
    proposed_b = (moved_b.reshape(-1, 2) + .5) * 2
    delta_a = np.linalg.norm(proposed_a - original_a, axis=1)
    delta_b = np.linalg.norm(proposed_b - original_b, axis=1)
    eligible = (stable & (delta_a <= CONFIG['max_symmetric_lk_loma_original_px'])
                & (delta_b <= CONFIG['max_symmetric_lk_loma_original_px']))
    proposals = []
    for j in np.flatnonzero(eligible):
        proposals.append((int(image_a.image_id), a_ids[j], image_b.name,
                          proposed_a[j].astype(float)))
        proposals.append((int(image_b.image_id), b_ids[j], image_a.name,
                          proposed_b[j].astype(float)))
    return proposals, {'shared_training_points': len(shared),
                       'bidirectional_flow_stable': int(stable.sum()),
                       'symmetric_lk_loma_within_4px': int(eligible.sum())}


def refine_training_points(rec, scene, split):
    records = {record['name']: record for record in scene['images']}
    available = {(record['camera'], int(record['frame_index'])): record['name']
                 for record in scene['images']}
    images = {image.name: image for image in rec.images.values() if image.has_pose}
    held = {(name, int(index)) for track in split['tracks']
            for name, index in track['observations']}

    @lru_cache(maxsize=4)
    def image_gray(path):
        original = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if original is None or original.shape != (4000, 3000):
            raise ValueError(f'Missing or changed original JPEG: {path}')
        return cv2.resize(original, (CONFIG['flow_width'], CONFIG['flow_height']),
                          interpolation=cv2.INTER_AREA)

    proposals = defaultdict(list)
    pair_audit = []
    for camera in ('L', 'R'):
        for frame in range(70, 94):
            name_a, name_b = available.get((camera, frame)), available.get((camera, frame + 1))
            if not name_a or not name_b or name_a not in images or name_b not in images:
                continue
            pair, audit = _pair_proposals(images[name_a], images[name_b], records, image_gray)
            for image_id, index, neighbor, xy in pair:
                name = rec.images[image_id].name
                if (name, index) in held:
                    raise RuntimeError('Heldout point entered LK proposal graph')
                proposals[(image_id, index)].append((neighbor, xy))
            pair_audit.append({'a': name_a, 'b': name_b, **audit})

    updates = []
    for (image_id, index), neighbors in sorted(proposals.items()):
        if len({name for name, _ in neighbors}) < CONFIG['minimum_independent_neighbor_proposals']:
            continue
        original = np.asarray(rec.images[image_id].points2D[index].xy, dtype=float)
        candidates = np.asarray([xy for _, xy in neighbors], dtype=float)
        median = np.median(candidates, axis=0)
        if np.max(np.linalg.norm(candidates - median, axis=1)) > CONFIG['max_proposal_disagreement_original_px']:
            continue
        shift = float(np.linalg.norm(median - original))
        if shift > CONFIG['max_consensus_shift_original_px']:
            continue
        adjusted = original + CONFIG['applied_consensus_fraction'] * (median - original)
        rec.images[image_id].points2D[index].xy = adjusted
        updates.append({'image': rec.images[image_id].name, 'feature_index': index,
                        'original_xy': original.tolist(), 'consensus_xy': median.tolist(),
                        'adjusted_xy': adjusted.tolist(), 'proposals': len(neighbors),
                        'full_consensus_shift_px': shift})
    return {'pairs': pair_audit, 'total_mapped_observations': rec.compute_num_observations(),
            'with_at_least_one_flow_proposal': len(proposals),
            'updated_observations': len(updates),
            'updated_images': len({row['image'] for row in updates}),
            'updated_by_camera': dict(Counter(row['image'][0] for row in updates)),
            'consensus_shift_px_median': float(np.median([
                row['full_consensus_shift_px'] for row in updates])) if updates else None,
            'updates': updates,
            'guard': 'Only mapped training observations, same-camera adjacent images, two agreeing neighbor proposals. No manual label, heldout feature, navigation pose or LAS used. LK is not truth.'}


def _protocol(root, scene, split, triangle):
    names = [record for record in scene['images']
             if 70 <= record['frame_index'] <= 94]
    return {'version': 1, 'source_model_hashes': _hashes(root / SOURCE),
            'split_sha256': file_digest(split),
            'triangle_sha256': file_digest(triangle),
            'source_image_sha256': {record['name']: file_digest(record['source'])
                                    for record in names},
            'config': CONFIG, 'camera_calibration_fixed_during_refit': True,
            'each_exposure_has_independent_pose': True,
            'manual_or_holdout_used_for_proposals_or_ba': False,
            'scope': 'One 50-image local development experiment. Existing full-scene and LFS inputs remain unchanged.'}


def run_lk_refinement(work, stage='all'):
    cv2.setNumThreads(4)
    work = Path(work).resolve()
    local, scene, data, features = load_local(work)
    segment = local / SEGMENT
    split_path = segment / 'split.json'
    triangle_path = segment / 'triangle_validation_split.json'
    split = json.loads(split_path.read_text())
    triangle = json.loads(triangle_path.read_text())
    output = work / OUTPUT / SEGMENT
    output.mkdir(parents=True, exist_ok=True)
    protocol = _protocol(work, scene, split_path, triangle_path)
    fingerprint = digest(protocol)
    protocol_path = output / 'protocol.json'
    if protocol_path.exists():
        if json.loads(protocol_path.read_text())['fingerprint'] != fingerprint:
            raise ValueError('Existing LK refinement belongs to another frozen source')
    else:
        write_json(protocol_path, {**protocol, 'fingerprint': fingerprint})
    source = pycolmap.Reconstruction(work / SOURCE)
    assert_holdout_isolated(source, split)
    topology = topology_signature(source)
    cameras = {camera_id: camera.params.copy() for camera_id, camera in source.cameras.items()}
    initial = output / 'lk_initial/sparse/0'
    audit_path = output / 'lk_audit.json'
    if initial.exists() or audit_path.exists():
        if not initial.exists() or not audit_path.exists():
            raise RuntimeError('Partial LK audit exists; inspect before continuing')
        audit = json.loads(audit_path.read_text())
        if _hashes(initial) != audit['initial_model_hashes']:
            raise ValueError('Saved LK initial model changed')
    else:
        candidate = pycolmap.Reconstruction(work / SOURCE)
        audit = refine_training_points(candidate, scene, split)
        assert_holdout_isolated(candidate, split)
        if topology_signature(candidate) != topology:
            raise RuntimeError('LK refinement changed point topology')
        if audit['updated_observations'] < 10:
            raise RuntimeError('Too few stable training proposals to justify a BA probe')
        initial.mkdir(parents=True)
        candidate.write(initial)
        audit['initial_model_hashes'] = _hashes(initial)
        write_json(audit_path, audit)
    if stage == 'audit':
        return {'output': str(output), 'audit': audit}

    stages = {}
    for label in ('baseline', 'lk_refined'):
        directory = output / 'stages' / label
        model = directory / 'sparse/0'
        ba_path = directory / 'ba.json'
        if model.exists() or ba_path.exists():
            if not model.exists() or not ba_path.exists():
                raise RuntimeError(f'Partial saved {label} stage')
            rec = pycolmap.Reconstruction(model)
            ba = json.loads(ba_path.read_text())
            if _hashes(model) != ba['model_hashes']:
                raise ValueError(f'Saved {label} model changed')
        else:
            rec = pycolmap.Reconstruction(work / SOURCE if label == 'baseline' else initial)
            assert_holdout_isolated(rec, split)
            ba = bounded_bundle_adjustment(rec, 'fixed', cameras, protect_depth=True)
            model.mkdir(parents=True)
            rec.write(model)
            ba['model_hashes'] = _hashes(model)
            write_json(ba_path, ba)
        assert_holdout_isolated(rec, split)
        if topology_signature(rec) != topology:
            raise RuntimeError(f'{label} changed training point topology')
        for camera_id, params in cameras.items():
            np.testing.assert_array_equal(rec.cameras[camera_id].params, params)
        for track in split['tracks']:
            for name, index in track['observations']:
                image = next(image for image in rec.images.values() if image.name == name)
                np.testing.assert_array_equal(image.points2D[index].xy,
                                              features[name]['keypoints'][index])
        raw = evaluate_reconstruction(rec, scene, features, split, directory)
        high_dir = directory / 'triangle_validation'
        high_dir.mkdir(exist_ok=True)
        high = evaluate_reconstruction(rec, scene, features, triangle, high_dir)
        stages[label] = {'ba': ba, 'raw': raw, 'triangle': high,
                         'points': rec.num_points3D(),
                         'observations': rec.compute_num_observations()}
    comparison, _, _ = paired_common_support(output, list(stages),
                                             'triangle_validation', baseline_stage='baseline')
    result = {'protocol_fingerprint': fingerprint, 'audit': {k: v for k, v in audit.items()
                                                            if k != 'updates'},
              'stages': stages, 'triangle_common_support': comparison,
              'topology_unchanged': True, 'validation_observations_unchanged': True,
              'scope': 'Exploratory local front-end refinement; frozen original LoMa holdout is not exact pixel ground truth.'}
    write_json(output / 'comparison.json', result)
    return result
