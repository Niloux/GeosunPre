"""Voxel centroids and RGB from the complete LAS, in the LAS coordinate frame."""

import math
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import numpy as np

from geosunpre.artifacts import _file_signature, file_digest
from geosunpre.registration.las import read_header

DEFAULT_VOXEL_SIZE = .05
PARTITIONS = 128
AGGREGATE = np.dtype([('key', '<u8'), ('count', '<u8'),
                      ('xyz_sum', '<i8', 3), ('rgb_sum', '<u8', 3)])
VERTEX = np.dtype([(name, '<f4') for name in ('x', 'y', 'z', 'nx', 'ny', 'nz')]
                  + [(name, 'u1') for name in ('red', 'green', 'blue')])


def initialization_recipe(voxel_size):
    if isinstance(voxel_size, bool) or not math.isfinite(voxel_size) or voxel_size <= 0:
        raise ValueError('PLY voxel size must be finite and positive')
    return {'version': 1, 'voxel_size_m': float(voxel_size),
            'sampling': 'all non-withheld, non-class-7 LAS records; one centroid per voxel',
            'color': 'voxel mean of LAS RGB; auto-detect 8/16-bit range; gray if absent',
            'normals': 'zero', 'coordinate_frame': 'input_las_xyz'}


def _reduce(records):
    ordered = records[np.argsort(records['key'], kind='stable')]
    starts = np.r_[0, np.flatnonzero(ordered['key'][1:] != ordered['key'][:-1]) + 1]
    result = ordered[starts].copy()
    for field in ('count', 'xyz_sum', 'rgb_sum'):
        result[field] = np.add.reduceat(ordered[field], starts, axis=0)
    return result


def _partition(records, directory):
    # Mix the spatial key so horizontal planes also spread over disk partitions.
    mixed = records['key'] ^ (records['key'] >> np.uint64(33))
    mixed *= np.uint64(0xff51afd7ed558ccd)
    mixed ^= mixed >> np.uint64(33)
    groups = (mixed & np.uint64(PARTITIONS - 1)).astype(np.int32)
    order = np.argsort(groups, kind='stable')
    groups, records = groups[order], records[order]
    starts = np.r_[0, np.flatnonzero(groups[1:] != groups[:-1]) + 1, len(groups)]
    for lo, hi in zip(starts[:-1], starts[1:]):
        with (directory / f'{groups[lo]:03d}.bin').open('ab') as stream:
            records[lo:hi].tofile(stream)


def _ply_header(count):
    return (f'ply\nformat binary_little_endian 1.0\nelement vertex {count}\n'
            + ''.join(f'property float {name}\n' for name in ('x', 'y', 'z', 'nx', 'ny', 'nz'))
            + ''.join(f'property uchar {name}\n' for name in ('red', 'green', 'blue'))
            + 'end_header\n').encode('ascii')


def verify_initialization_ply(path, points):
    path = Path(path)
    if type(points) is not int or points < 1:
        raise ValueError('Initialization PLY must contain points')
    header = _ply_header(points)
    with path.open('rb') as stream:
        if stream.read(len(header)) != header or path.stat().st_size != len(header) + points * VERTEX.itemsize:
            raise ValueError('Initialization PLY header or vertex count differs')
        for start in range(0, points, 500_000):
            vertices = np.fromfile(stream, dtype=VERTEX, count=min(500_000, points - start))
            if (any(not np.isfinite(vertices[name]).all() for name in ('x', 'y', 'z'))
                    or any(np.any(vertices[name] != 0) for name in ('nx', 'ny', 'nz'))):
                raise ValueError('Invalid initialization PLY coordinates or normals')


def export_las_ply(las, output, *, voxel_size=DEFAULT_VOXEL_SIZE, chunk_points=500_000):
    """Scan all records in chunks, merge voxel sums on disk, then publish XYZ/RGB."""
    recipe = initialization_recipe(voxel_size)
    if type(chunk_points) is not int or chunk_points < 1:
        raise ValueError('Expected a positive LAS chunk size')
    las, output = Path(las).resolve(), Path(output)
    if las == output.resolve():
        raise ValueError('Choose a PLY output separate from the source LAS')
    before = _file_signature(las.stat())
    source_sha256 = file_digest(las)
    header = read_header(las)
    scales, offsets = np.asarray(header['scales']), np.asarray(header['offsets'])
    lower = np.floor(np.asarray(header['bounds_min']) / voxel_size) - 1
    upper = np.floor(np.asarray(header['bounds_max']) / voxel_size) + 1
    if (np.max(np.abs(np.r_[lower, upper])) >= 2**63
            or math.prod(int(n) for n in upper - lower + 1) >= 2**63):
        raise ValueError('LAS extent / PLY voxel size exceeds the supported grid')
    lower, shape = lower.astype(np.int64), (upper - lower + 1).astype(np.int64)
    has_rgb = header['point_format'] in (2, 3)
    fields = {'names': ['xyz', 'classification'], 'formats': [('<i4', 3), 'u1'],
              'offsets': [0, 15], 'itemsize': header['record_size']}
    if has_rgb:
        fields['names'].append('rgb')
        fields['formats'].append(('<u2', 3))
        fields['offsets'].append(20 if header['point_format'] == 2 else 28)
    dtype = np.dtype(fields)
    accepted = rejected = rgb_max = points = 0
    bounds_min, bounds_max = np.full(3, np.inf), np.full(3, -np.inf)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.las-initialization-', dir=output.parent) as temporary:
        directory = Path(temporary)
        with las.open('rb') as stream:
            stream.seek(header['point_data_offset'])
            for start in range(0, header['point_count'], chunk_points):
                count = min(chunk_points, header['point_count'] - start)
                block = np.fromfile(stream, dtype=dtype, count=count)
                if len(block) != count:
                    raise ValueError('LAS changed or was truncated during initialization')
                flags = block['classification']
                keep = ((flags & 128) == 0) & ((flags & 31) != 7)
                rejected += int(np.count_nonzero(~keep))
                block = block[keep]
                if not len(block):
                    continue
                accepted += len(block)
                cells = np.floor((block['xyz'] * scales + offsets) / voxel_size).astype(np.int64) - lower
                if np.any(cells < 0) or np.any(cells >= shape):
                    raise ValueError('LAS coordinates exceed the declared bounds')
                values = np.empty(len(block), dtype=AGGREGATE)
                values['key'] = (cells[:, 0] * shape[1] + cells[:, 1]) * shape[2] + cells[:, 2]
                values['count'] = 1
                values['xyz_sum'] = block['xyz']
                values['rgb_sum'] = block['rgb'] if has_rgb else 127
                if has_rgb:
                    rgb_max = max(rgb_max, int(block['rgb'].max()))
                _partition(_reduce(values), directory)
                del block, cells, values
        if not accepted:
            raise ValueError('No usable LAS records for initialization')
        color_scale = 255. / 65535. if has_rgb and rgb_max > 255 else 1.
        payload = directory / 'vertices.bin'
        with payload.open('wb') as stream:
            for bucket in sorted(directory.glob('[0-9][0-9][0-9].bin')):
                voxels = _reduce(np.fromfile(bucket, dtype=AGGREGATE))
                counts = voxels['count'].astype(np.float64)
                xyz = voxels['xyz_sum'] / counts[:, None] * scales + offsets
                rgb = np.clip(np.rint(voxels['rgb_sum'] / counts[:, None] * color_scale), 0, 255)
                vertices = np.zeros(len(voxels), dtype=VERTEX)
                for axis, name in enumerate(('x', 'y', 'z')):
                    vertices[name] = xyz[:, axis]
                    bounds_min[axis] = min(bounds_min[axis], float(vertices[name].min()))
                    bounds_max[axis] = max(bounds_max[axis], float(vertices[name].max()))
                for axis, name in enumerate(('red', 'green', 'blue')):
                    vertices[name] = rgb[:, axis]
                vertices.tofile(stream)
                points += len(vertices)
                del voxels, counts, xyz, rgb, vertices
        if _file_signature(las.stat()) != before:
            raise ValueError('LAS changed during initialization')
        staged = directory / 'points3D.ply'
        with staged.open('wb') as destination, payload.open('rb') as source:
            destination.write(_ply_header(points))
            shutil.copyfileobj(source, destination)
        verify_initialization_ply(staged, points)
        staged.replace(output)
    return {'version': 1, 'source': 'las', 'source_las_sha256': source_sha256,
            'source_point_count': header['point_count'], 'accepted_records': accepted,
            'excluded_withheld_or_noise': rejected, 'points': points, **recipe,
            'rgb_encoding': ('uint16_0_65535' if rgb_max > 255 else 'uint16_0_255') if has_rgb else 'absent_gray_127',
            'bounds_min': bounds_min.tolist(), 'bounds_max': bounds_max.tolist(),
            'ply_sha256': file_digest(output)}
