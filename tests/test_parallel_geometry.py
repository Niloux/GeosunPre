import numpy as np
import pytest

from geosunpre.camera import Camera
from geosunpre.parallel import ordered_map
from geosunpre.sfm.geometry import fit_pair


def test_pair_fitting_preserves_random_seeds_across_threads():
    camera = Camera('L', 3000, 4000, (965., 965., 1500., 2000., -.03, -.004, .001, -.0003), (0., 0.))
    rng = np.random.default_rng(213)
    points = rng.uniform([-3., -2., 5.], [3., 2., 10.], (100, 3))
    a = camera.project(points)
    b = camera.project(points + [1., .1, 0.]) + rng.normal(0., .1, (100, 2))
    b[:15] = b[15:30]
    def fit(seed):
        return fit_pair(camera, camera, a, b, seed=seed)
    seeds = list(range(20, 28))
    expected = [fit(seed) for seed in seeds]
    actual = list(ordered_map(fit, seeds))
    for (metrics_a, arrays_a), (metrics_b, arrays_b) in zip(expected, actual):
        assert metrics_a == metrics_b
        for key in arrays_a:
            np.testing.assert_array_equal(arrays_a[key], arrays_b[key])


def test_ordered_workers_bound_input_consumption_and_propagate_errors():
    consumed = []
    def items():
        for i in range(1000):
            consumed.append(i)
            yield i
    def work(i):
        if i == 5:
            raise ValueError('worker failed')
        return i
    output = ordered_map(work, items(), workers=2)
    assert next(output) == 0
    assert len(consumed) == 4
    with pytest.raises(ValueError, match='worker failed'):
        list(output)
    assert len(consumed) < 12
