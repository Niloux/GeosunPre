import json
import struct

import numpy as np
import pycolmap
import pytest
from scipy.spatial.transform import Rotation

from geosunpre.alignment.geometry import (error_summary, refine_similarity, robust_similarity,
                                         spatial_split, surfaces, transform_points, validate_similarity)
from geosunpre.alignment.las import read_header, sample_las, sample_las_records
from geosunpre.alignment.diagnostics import compare_visits, support_diagnostics, temporal_consistency, viewing_side_proxy
from geosunpre.alignment.inspection import inspect_alignment
from geosunpre.alignment.runner import projection_invariance, run_alignment
from geosunpre.alignment.structural import (balanced_weights, coherent_las, point_precision,
                                           refine_structural, structural_matches)
from test_pipeline import tiny_visual_work
from geosunpre.pipeline.runner import SOURCE


def make_las(path, points, point_format=3):
    header = bytearray(227)
    header[:4] = b'LASF'
    header[24:26] = bytes([1, 2])
    struct.pack_into('<H', header, 94, 227)
    struct.pack_into('<I', header, 96, 227)
    header[104] = point_format
    length = (20, 28, 26, 34)[point_format]
    struct.pack_into('<H', header, 105, length)
    struct.pack_into('<I', header, 107, len(points))
    struct.pack_into('<3d', header, 131, .001, .001, .001)
    struct.pack_into('<3d', header, 155, 100., 200., 300.)
    values = np.rint(np.asarray(points) / .001).astype(np.int32)
    actual = values * .001 + [100., 200., 300.]
    struct.pack_into('<6d', header, 179, *np.column_stack((actual.max(0), actual.min(0))).ravel())
    fields = {'names': ['xyz', 'class'], 'formats': [('<i4', 3), 'u1'],
              'offsets': [0, 15], 'itemsize': length}
    if point_format in (1, 3):
        fields['names'].append('time')
        fields['formats'].append('<f8')
        fields['offsets'].append(20)
    dtype = np.dtype(fields)
    records = np.zeros(len(points), dtype=dtype)
    records['xyz'] = values
    records['class'] = 1
    if point_format in (1, 3):
        records['time'] = np.arange(len(points)) * .005 + 352995.
    with path.open('wb') as stream:
        stream.write(header)
        records.tofile(stream)
    return actual


@pytest.mark.parametrize('point_format', [0, 1, 2, 3])
def test_las_chunk_sampling_preserves_offsets_and_is_chunk_independent(tmp_path, point_format):
    points = np.random.default_rng(4).normal(size=(1000, 3))
    path = tmp_path / 'sample.las'
    actual = make_las(path, points, point_format)
    a, metadata = sample_las(path, max_points=101, chunk_points=17)
    b, other = sample_las(path, max_points=101, chunk_points=500)
    np.testing.assert_array_equal(a, b)
    assert len(a) <= 101
    if point_format in (1, 3):
        assert metadata['gps_time']['encoding'] == 'gps_week'
    else:
        assert metadata['gps_time'] is None
    assert metadata['crs'] == 'absent'
    assert metadata['point_count'] == 1000
    assert other['sampled_points'] == len(b)
    assert all(any(np.array_equal(p, q) for q in actual) for p in a)
    sampled, same, times = sample_las_records(path, max_points=101, chunk_points=17)
    np.testing.assert_array_equal(sampled, a)
    assert same == metadata
    if point_format in (1, 3):
        indices = np.array([next(i for i, q in enumerate(actual) if np.array_equal(p, q)) for p in a])
        np.testing.assert_array_equal(times, indices * .005 + 352995.)
    else:
        assert times is None


def test_las_withheld_and_noise_records_are_counted_and_excluded(tmp_path):
    path = tmp_path / 'sample.las'
    actual = make_las(path, np.random.default_rng(0).normal(size=(40, 3)))
    with path.open('r+b') as stream:
        stream.seek(227 + 15)
        stream.write(bytes([7]))
        stream.seek(227 + 34 + 15)
        stream.write(bytes([129]))
    points, metadata, times = sample_las_records(path, max_points=40, chunk_points=9)
    np.testing.assert_array_equal(points, actual[2:])
    np.testing.assert_array_equal(times, np.arange(2, 40) * .005 + 352995.)
    assert metadata['excluded_withheld_or_noise'] == 2


def test_las_rejects_truncation_and_unsupported_compression(tmp_path):
    path = tmp_path / 'sample.las'
    make_las(path, np.random.default_rng(0).normal(size=(40, 3)))
    with path.open('r+b') as stream:
        stream.truncate(path.stat().st_size - 1)
    with pytest.raises(ValueError, match='truncated'):
        read_header(path)
    make_las(path, np.random.default_rng(0).normal(size=(40, 3)))
    with path.open('r+b') as stream:
        stream.seek(104)
        stream.write(bytes([131]))
    with pytest.raises(ValueError, match='LAZ'):
        read_header(path)


def test_similarity_initialization_recovers_scale_despite_pose_outliers():
    rng = np.random.default_rng(0)
    source = rng.normal(size=(100, 3))
    truth = np.eye(4)
    truth[:3, :3] = 3.2 * Rotation.from_rotvec([.2, -.3, .1]).as_matrix()
    truth[:3, 3] = [10., -7., 2.]
    target = transform_points(truth, source)
    target[:20] += [20., 50., -40.]
    fitted, report = robust_similarity(source, target, threshold=.05)
    np.testing.assert_allclose(fitted, truth, atol=1e-8)
    assert report['inliers'] == 80
    line = np.column_stack((np.arange(10), np.zeros((10, 2))))
    with pytest.raises(ValueError, match='collinear'):
        robust_similarity(line, line)


def test_similarity_rejects_reflection_and_anisotropic_scaling():
    for linear in (np.diag([-1., 1., 1.]), np.diag([1., 2., 1.])):
        matrix = np.eye(4)
        matrix[:3, :3] = linear
        with pytest.raises(ValueError, match='proper rotation'):
            validate_similarity(matrix)


def test_spatial_holdout_keeps_entire_tiles_together():
    points = np.random.default_rng(0).uniform(-12, 12, (2000, 3))
    holdout, tiles, report = spatial_split(points)
    for tile in np.unique(tiles, axis=0):
        members = np.all(tiles == tile, axis=1)
        assert np.unique(holdout[members]).size == 1
    np.testing.assert_array_equal(holdout, spatial_split(points)[0])
    assert report['fit_points'] + report['holdout_points'] == len(points)


def plane_scene():
    rng = np.random.default_rng(20)
    uv = rng.uniform(-5, 5, (8000, 2))
    return np.vstack((np.column_stack((uv, np.zeros(len(uv)))),
                      np.column_stack((np.full(len(uv), -5.), uv)),
                      np.column_stack((uv[:, 0], np.full(len(uv), 5.), uv[:, 1])),
                      np.column_stack((np.full(len(uv), 5.), uv))))


def test_surface_fit_recovers_similarity_and_predicts_spatial_holdout():
    target = plane_scene()
    source = target[::5]
    initial = np.eye(4)
    initial[:3, :3] = 1.015 * Rotation.from_rotvec([.008, -.006, .01]).as_matrix()
    initial[:3, 3] = [.04, -.03, .02]
    holdout, tiles, _ = spatial_split(transform_points(initial, source), tile_size=2.)
    final, diagnostic, surface = refine_similarity(source[~holdout], target, initial,
                                                    tiles_fit=tiles[~holdout], workers=1)
    np.testing.assert_allclose(final, np.eye(4), atol=2e-3)
    before = error_summary(transform_points(initial, source[holdout]), surface, workers=1)
    after = error_summary(transform_points(final, source[holdout]), surface, workers=1)
    assert after['point_to_plane_on_supported']['median'] < before['point_to_plane_on_supported']['median'] / 10
    assert diagnostic['full_rank_at_all_solved_iterations']


def test_single_plane_does_not_claim_all_similarity_axes_are_observable():
    target = plane_scene()[:8000]
    initial = np.eye(4)
    initial[2, 3] = .03
    final, diagnostic, _ = refine_similarity(target[::5], target, initial, workers=1)
    assert not diagnostic['full_rank_at_all_solved_iterations']
    assert diagnostic['minimum_rank'] < 7
    np.testing.assert_array_equal(final, initial)


def test_collapsed_target_is_rejected_before_surface_covariance():
    with pytest.raises(ValueError, match='spatially distinct'):
        refine_similarity(np.zeros((100, 3)), np.zeros((100, 3)), np.eye(4))


def test_transform_preserves_native_fisheye_projections_and_all_tracks(tmp_path):
    work, _, _, _, _ = tiny_visual_work(tmp_path)
    before = pycolmap.Reconstruction(work / SOURCE)
    after = pycolmap.Reconstruction(work / SOURCE)
    matrix = np.eye(4)
    matrix[:3, :3] = 3.5 * Rotation.from_rotvec([.7, -.2, .4]).as_matrix()
    matrix[:3, 3] = [100., -30., 7.]
    after.transform(pycolmap.Sim3d(matrix[:3]))
    report = projection_invariance(before, after)
    assert report['maximum_pixel_difference'] < 1e-10
    after.cameras[1].params[0] *= 1.1
    with pytest.raises(ValueError, match='intrinsics'):
        projection_invariance(before, after)


def test_alignment_refuses_to_write_inside_source_data(tmp_path):
    dataset = tmp_path / 'source'
    dataset.mkdir()
    initial = tmp_path / 'initial.json'
    initial.write_text(json.dumps({'matrix': np.eye(4).tolist()}))
    with pytest.raises(ValueError, match='separate'):
        run_alignment(dataset, tmp_path / 'source.las', dataset / 'alignment', initial_transform=initial)
    output = tmp_path / 'alignment'
    output.mkdir()
    (output / 'foreign.txt').write_text('preserve me')
    with pytest.raises(ValueError, match='fresh'):
        run_alignment(dataset, tmp_path / 'source.las', output, initial_transform=initial)
    assert (output / 'foreign.txt').read_text() == 'preserve me'


def test_support_reason_replay_matches_original_surface_gate():
    target = plane_scene()
    a = surfaces(target, voxel_size=.1, workers=1)
    b = surfaces(target, voxel_size=.1, workers=1, diagnostics=True)
    for key in ('points', 'normals', 'centres', 'valid', 'noise'):
        np.testing.assert_array_equal(a[key], b[key])
    probes = np.vstack((target[::7], [[20, 20, 20]]))
    diagnostic = support_diagnostics(probes, b, workers=1)
    expected = error_summary(probes, a, workers=1)
    assert np.count_nonzero(diagnostic['reason'] == 0) == expected['surface_support_count']
    assert diagnostic['reason'][-1] == 1


def test_repeat_visit_offset_requires_shared_surface_footprint():
    uv = np.random.default_rng(7).uniform(-.25, .25, (200, 2))
    early = np.column_stack((np.zeros(len(uv)), uv))
    late = early + [.06, 0., 0.]
    comparison = compare_visits(early, late)
    assert comparison['status'] == 'compared'
    assert comparison['abs_median_separation'] == pytest.approx(.06)
    assert comparison['orientation'] == 'vertical'
    assert compare_visits(early, late + [0, 2, 0])['status'] == 'no_shared_tangential_footprint'
    horizontal = np.column_stack((uv, np.zeros(len(uv))))
    assert compare_visits(early, horizontal)['status'] == 'incompatible_visit_normals'


def test_temporal_checks_do_not_confuse_continuous_scan_with_repeat_visits():
    uv = np.random.default_rng(4).uniform(-.2, .2, (200, 2))
    plane = np.column_stack((np.zeros(len(uv)), uv))
    target = np.vstack((plane, plane + [.06, 0, 0]))
    centres = np.array([[.03, 0., 0.]])
    continuous = np.linspace(100., 110., len(target))
    assert temporal_consistency(target, continuous, centres, workers=1)['patches'] == []
    repeat = np.r_[np.linspace(100., 101., len(plane)), np.linspace(150., 151., len(plane))]
    result = temporal_consistency(target, repeat, centres, workers=1)
    assert result['abs_median_separation_by_orientation']['vertical']['median'] == pytest.approx(.06)
    assert temporal_consistency(target, None, centres)['status'] == 'gps_time_unavailable'


def test_structure_inspection_refuses_source_nesting_and_foreign_outputs(tmp_path):
    alignment = tmp_path / 'alignment'
    alignment.mkdir()
    with pytest.raises(ValueError, match='separate'):
        inspect_alignment(alignment, alignment / 'inspect')
    output = tmp_path / 'inspection'
    output.mkdir()
    (output / 'foreign.txt').write_text('keep')
    with pytest.raises(ValueError, match='fresh'):
        inspect_alignment(alignment, output)
    assert (output / 'foreign.txt').read_text() == 'keep'


def test_viewing_side_proxy_flags_opposite_faces_without_certifying_surfaces():
    temporal = {'patches': [{'centre': [0, 0, 0], 'normal': [1, 0, 0],
                            'early_time': 100., 'late_time': 150., 'orientation': 'vertical',
                            'abs_median_separation': .1}]}
    result = viewing_side_proxy(temporal, [100., 150.], [[-3., 0, 0], [3., 0, 0]])
    assert result['patches'][0]['viewing_side_proxy']['status'] == 'opposite_sides_proxy'
    assert result['viewing_side_proxy']['same_side_abs_median_separation_by_orientation']['vertical']['count'] == 0
    result = viewing_side_proxy(temporal, [100., 150.], [[3., 0, 0], [2., 0, 0]])
    assert result['patches'][0]['viewing_side_proxy']['status'] == 'same_side_proxy'
    result = viewing_side_proxy(temporal, [100., 150.], [[.1, 0, 0], [2., 0, 0]])
    assert result['patches'][0]['viewing_side_proxy']['status'] == 'near_plane_ambiguous'
    result = viewing_side_proxy(temporal, [110., 150.], [[3., 0, 0], [2., 0, 0]])
    assert result['patches'][0]['viewing_side_proxy']['status'] == 'time_proxy_unavailable'


def test_native_point_precision_transforms_with_uniform_scale_and_rotation(tmp_path):
    work, _, _, _, _ = tiny_visual_work(tmp_path)
    before = pycolmap.Reconstruction(work / SOURCE)
    ids = sorted(before.points3D)
    covariance, valid, observations = point_precision(before, ids)
    assert valid.all() and np.all(observations == 2)
    matrix = np.eye(4)
    matrix[:3, :3] = 3.5 * Rotation.from_rotvec([.2, -.3, .4]).as_matrix()
    matrix[:3, 3] = [20., -7., 8.]
    after = pycolmap.Reconstruction(work / SOURCE)
    after.transform(pycolmap.Sim3d(matrix[:3]))
    transformed, other_valid, other_observations = point_precision(after, ids)
    np.testing.assert_array_equal(valid, other_valid)
    np.testing.assert_array_equal(observations, other_observations)
    np.testing.assert_allclose(transformed, matrix[:3, :3] @ covariance @ matrix[:3, :3].T, rtol=1e-6, atol=1e-7)


def test_orientation_weights_do_not_increase_when_one_tile_is_duplicated():
    tiles = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    normals = np.array([[0., 0., 1.], [0., 0., 1.], [1., 0., 0.]])
    weight, orientation = balanced_weights(tiles, normals, np.ones(3))
    assert weight[orientation == 0].sum() == pytest.approx(.45)
    assert weight[orientation == 1].sum() == pytest.approx(.45)
    repeated = np.r_[np.zeros(100, dtype=int), [1, 2]]
    other, labels = balanced_weights(tiles[repeated], normals[repeated], np.ones(len(repeated)))
    assert other[labels == 0].sum() == pytest.approx(.45)
    assert other[labels == 1].sum() == pytest.approx(.45)
    assert other[:100].sum() == pytest.approx(weight[0])


def test_structural_refinement_recovers_sim3_without_using_holdout():
    target = plane_scene()
    source = target[::5]
    initial = np.eye(4)
    initial[:3, :3] = 1.002 * Rotation.from_rotvec([.004, -.003, .004]).as_matrix()
    initial[:3, 3] = [.04, -.03, .02]
    holdout, tiles, _ = spatial_split(transform_points(initial, source), tile_size=2.)
    covariance = np.broadcast_to(np.eye(3) * .0001, ((~holdout).sum(), 3, 3))
    final, report, surface, attributes = refine_structural(source[~holdout], target, initial,
        covariance, np.ones((~holdout).sum(), dtype=bool), tiles_fit=tiles[~holdout], workers=1)
    np.testing.assert_allclose(final, np.eye(4), atol=2e-3)
    assert not report['holdout_used_for_optimization']
    assert report['full_rank_at_all_solved_iterations']
    assert attributes['final_structural_matches'].sum() > 100
    assert error_summary(transform_points(final, source[holdout]), surface, workers=1)[
        'point_to_plane_on_supported']['median'] < .002


def test_single_plane_refinement_releases_only_tilt_and_height():
    target = plane_scene()[:8000]
    source = target[::5]
    initial = np.eye(4)
    initial[2, 3] = .03
    covariance = np.broadcast_to(np.eye(3) * .0001, (len(source), 3, 3))
    final, report, _, _ = refine_structural(source, target, initial,
        covariance, np.ones(len(source), dtype=bool), tiles_fit=np.floor(source / 2).astype(int), workers=1)
    assert report['mode'] == 'ground_tilt_z_only'
    assert report['estimated_dofs'] == 3
    assert not report['all_seven_similarity_dofs_reestimated']
    assert report['full_rank_at_all_solved_iterations']
    np.testing.assert_allclose(final, np.eye(4), atol=1e-8)


def test_structural_correspondence_rejects_incompatible_visual_normals():
    target = plane_scene()[:8000]
    surface = coherent_las(target, workers=1)
    source = target[::100]
    wrong_normals = np.broadcast_to([1., 0., 0.], source.shape)
    keep, _, _, _ = structural_matches(source, wrong_normals, np.ones(len(source), dtype=bool),
                                        surface, .3, workers=1)
    assert not keep.any()


def test_precise_visual_points_without_normals_can_constrain_a_las_plane():
    target = plane_scene()[:8000]
    x = np.tile(np.linspace(-4., 4., 100), 5)
    y = np.repeat([-4., -2., 0., 2., 4.], 100)
    source = np.column_stack((x, y, np.zeros(len(x))))
    initial = np.eye(4)
    initial[2, 3] = .04
    covariance = np.broadcast_to(np.eye(3) * .00001, (len(source), 3, 3))
    final, report, _, attributes = refine_structural(source, target, initial,
        covariance, np.ones(len(source), dtype=bool), tiles_fit=np.floor(source / 2).astype(int), workers=1)
    assert report['visual_planar_fit_points'] == 0
    assert attributes['precise_point_only'].all()
    assert report['mode'] == 'ground_tilt_z_only'
    assert report['final_fit_matches'] > 100
    np.testing.assert_allclose(final, np.eye(4), atol=1e-8)
