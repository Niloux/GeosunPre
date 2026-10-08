
import numpy as np
import pycolmap
import pytest
from scipy.spatial.transform import Rotation

from geosunpre.registration.geometry import error_summary, refine_similarity, robust_similarity, spatial_split, transform_points, validate_similarity
from geosunpre.registration.las import read_header, sample_las, sample_las_records
from geosunpre.registration.global_alignment import projection_invariance
from geosunpre.registration.structural import (balanced_weights, coherent_las, point_precision,
                                           structural_matches)
from helpers import tiny_visual_work, make_las
from geosunpre.pipeline.finalization import SOURCE


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


def test_structural_correspondence_rejects_incompatible_visual_normals():
    target = plane_scene()[:8000]
    surface = coherent_las(target, workers=1)
    source = target[::100]
    wrong_normals = np.broadcast_to([1., 0., 0.], source.shape)
    keep, _, _, _ = structural_matches(source, wrong_normals, np.ones(len(source), dtype=bool),
                                        surface, .3, workers=1)
    assert not keep.any()
