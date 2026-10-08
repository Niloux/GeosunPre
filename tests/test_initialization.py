import struct

import numpy as np
import pycolmap
import pytest

from geosunpre.artifacts import file_digest
from geosunpre.initialization import export_las_ply, verify_initialization_ply
from helpers import make_las


def read_ply(path):
    # Consume through COLMAP's PLY reader, independently of our writer.
    model = pycolmap.Reconstruction()
    model.import_PLY(path)
    return sorted((tuple(point.xyz), tuple(point.color)) for point in model.points3D.values())


@pytest.mark.parametrize('point_format', [0, 1, 2, 3])
@pytest.mark.parametrize('color_scale', [1, 257])
def test_las_initialization_merges_across_chunks_preserving_frame_and_color(tmp_path, point_format, color_scale):
    las = tmp_path / 'scene.las'
    actual = make_las(las, [[.011, .012, .013], [.111, .112, .113], [.021, .022, .023],
                            [.211, .212, .213], [.221, .222, .223]], point_format)
    length = (20, 28, 26, 34)[point_format]
    if point_format in (2, 3):
        offset = 20 if point_format == 2 else 28
        colors = np.array([[100, 20, 40], [255, 30, 40], [200, 40, 60],
                           [10, 20, 30], [20, 40, 50]]) * color_scale
        with las.open('r+b') as stream:
            for index, color in enumerate(colors):
                stream.seek(227 + index * length + offset)
                stream.write(struct.pack('<3H', *color))
    a, b = tmp_path / 'a.ply', tmp_path / 'b.ply'
    first = export_las_ply(las, a, voxel_size=.05, chunk_points=2)
    second = export_las_ply(las, b, voxel_size=.05, chunk_points=100)
    assert first == second and file_digest(a) == file_digest(b)
    assert first['points'] == 3 and first['accepted_records'] == 5
    assert first['source_las_sha256'] == file_digest(las)
    assert first['coordinate_frame'] == 'input_las_xyz'
    vertices = read_ply(a)
    expected_xyz = [actual[[0, 2]].mean(0), actual[1], actual[[3, 4]].mean(0)]
    np.testing.assert_allclose([p[0] for p in vertices], expected_xyz, rtol=0, atol=2e-5)
    expected_rgb = [[150, 30, 50], [255, 30, 40], [15, 30, 40]] if point_format in (2, 3) else [[127]*3]*3
    np.testing.assert_array_equal([p[1] for p in vertices], expected_rgb)
    verify_initialization_ply(a, 3)


def test_initialization_scans_all_valid_records_and_excludes_las_noise_flags(tmp_path):
    las = tmp_path / 'scene.las'
    actual = make_las(las, np.column_stack((np.arange(40) * .1, np.zeros((40, 2)))))
    with las.open('r+b') as stream:
        stream.seek(227 + 15); stream.write(bytes([7]))
        stream.seek(227 + 34 + 15); stream.write(bytes([129]))
    output = tmp_path / 'points.ply'
    result = export_las_ply(las, output, chunk_points=3)
    assert result['source_point_count'] == 40
    assert result['accepted_records'] == result['points'] == 38
    assert result['excluded_withheld_or_noise'] == 2
    np.testing.assert_allclose([p[0] for p in read_ply(output)], actual[2:], rtol=0, atol=2e-5)


@pytest.mark.parametrize('size', [0, -1, float('nan'), float('inf')])
def test_initialization_rejects_invalid_voxel_sizes_without_publishing(tmp_path, size):
    las = tmp_path / 'scene.las'
    make_las(las, [[0, 0, 0]])
    output = tmp_path / 'points.ply'
    with pytest.raises(ValueError, match='finite and positive'):
        export_las_ply(las, output, voxel_size=size)
    assert not output.exists()


def test_initialization_rejects_all_noise_and_source_overwrite(tmp_path):
    las = tmp_path / 'scene.las'
    make_las(las, [[0, 0, 0]])
    before = file_digest(las)
    with pytest.raises(ValueError, match='separate from the source LAS'):
        export_las_ply(las, las)
    assert file_digest(las) == before
    with las.open('r+b') as stream:
        stream.seek(227 + 15); stream.write(bytes([7]))
    output = tmp_path / 'points.ply'
    with pytest.raises(ValueError, match='No usable LAS records'):
        export_las_ply(las, output)
    assert not output.exists()
    assert not list(tmp_path.glob('.las-initialization-*'))
