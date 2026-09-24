import copy

import pytest

from geosunpre.experiments.static_validation import (CAMERAS, TIME_BINS, ZONES, _stage_summary,
                                         checked_annotations, choose_candidates)


def test_geometry_blind_candidate_selection_is_order_and_residual_independent():
    pool = []
    for camera in CAMERAS:
        for lo, hi in TIME_BINS:
            for zone in ZONES:
                for i in range(4):
                    pool.append({'track_id': f'{camera}-{lo}-{zone}-{i}',
                                 'query': [f'{camera}-{lo}-{zone}', i],
                                 'camera': camera, 'time_bin': f'{lo:03d}-{hi:03d}',
                                 'zone': zone, 'gap_kind': 'near' if i % 2 else 'far',
                                 'pose_error_px': i * 100})
    first = choose_candidates(pool, per_cell=2)
    changed = copy.deepcopy(pool)
    for row in changed:
        row['pose_error_px'] *= -1000
        row['reprojection_error_px'] = 1e9
    second = choose_candidates(list(reversed(changed)), per_cell=2)
    assert [r['track_id'] for r in first] == [r['track_id'] for r in second]
    assert len(first) == 48 == len({r['track_id'] for r in first})
    assert {r['gap_kind'] for r in first} == {'near', 'far'}


def test_manual_static_requires_fingerprint_and_three_verified_image_points():
    row = {'id': 'one', 'camera': 'L', 'time_bin': '033-099', 'zone': 'center',
           'observations': [{'name': name} for name in ('a', 'b', 'c')]}
    scene = {'images': [{'name': name, 'camera': 'L'} for name in ('a', 'b', 'c')],
             'cameras': {'L': {'width': 3000, 'height': 4000}}}
    candidates = {'protocol_fingerprint': 'p', 'fingerprint': 'c', 'rows': [row]}
    value = {'label': 'same_static', 'scene_type': 'building',
             'coords': [[20, 30], [21, 31], [22, 32]],
             'verified': [True, True, False]}
    annotations = {'schema_version': 1, 'protocol_fingerprint': 'p',
                   'candidate_fingerprint': 'c', 'annotations': {'one': value}}
    accepted, audit = checked_annotations(annotations, candidates, scene)
    assert not accepted and audit['incomplete_same_static'] == 1
    value['verified'][2] = True
    accepted, audit = checked_annotations(annotations, candidates, scene)
    assert len(accepted) == 1 and audit['eligible_same_static'] == 1
    value['scene_type'] = None
    accepted, audit = checked_annotations(annotations, candidates, scene)
    assert accepted[0][1] == 'unspecified'
    assert audit['eligible_missing_scene_type'] == 1
    value['coords'][2] = [3000, 32]
    with pytest.raises(ValueError, match='outside image'):
        checked_annotations(annotations, candidates, scene)
    value['coords'][2] = [22, 32]
    annotations['candidate_fingerprint'] = 'wrong'
    with pytest.raises(ValueError, match='do not match'):
        checked_annotations(annotations, candidates, scene)


def test_failed_manual_query_remains_in_geometric_denominator():
    basis = {'camera': 'L', 'time_bin': '033-099', 'zone': 'edge',
             'scene_type': 'building', 'gap_kind': 'far',
             'camera_mix': 'cross_LR', 'camera_pattern': 'LRL'}
    rows = [{**basis, 'query_error_px': 1.5, 'failure': None},
            {**basis, 'query_error_px': None, 'failure': 'support_quality'},
            {**basis, 'query_error_px': None, 'failure': 'nonpositive_query_depth'}]
    summary = _stage_summary(rows)
    assert summary['all']['count'] == 3
    assert summary['all']['failures'] == 2
    assert summary['all']['fraction_le_2px'] == pytest.approx(1 / 3)
    assert summary['failure_reasons'] == {'support_quality': 1,
                                          'nonpositive_query_depth': 1}
