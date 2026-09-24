import numpy as np

from geosunpre.experiments.ground_photo_proxy import _sample


def test_native_corner_pixels_sample_original_jpeg_centers():
    image = np.array([[10., 20.], [30., 40.]], dtype=np.float32)
    pixels = np.array([[.5, .5], [1.5, .5], [.5, 1.5], [1.5, 1.5]])
    np.testing.assert_array_equal(_sample(image, pixels), [10., 20., 30., 40.])
