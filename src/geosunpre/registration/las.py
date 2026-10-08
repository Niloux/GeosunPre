"""Bounded-memory sampling of uncompressed LAS 1.2 point formats 0--3.

The reader rejects other versions and LAZ instead of guessing
their layouts. Coordinate units and CRS remain caller-provided assumptions.
"""

from pathlib import Path
import struct

import numpy as np


def read_header(path):
    path = Path(path)
    with path.open('rb') as stream:
        header = stream.read(227)
    if len(header) < 227 or header[:4] != b'LASF':
        raise ValueError('Expected a complete LAS header')
    if tuple(header[24:26]) != (1, 2):
        raise ValueError('LAS registration supports only uncompressed LAS 1.2')
    point_format = header[104]
    if point_format not in (0, 1, 2, 3):
        raise ValueError('LAS registration supports point formats 0--3; LAZ is unsupported')
    record_size = struct.unpack_from('<H', header, 105)[0]
    count = struct.unpack_from('<I', header, 107)[0]
    header_size = struct.unpack_from('<H', header, 94)[0]
    offset = struct.unpack_from('<I', header, 96)[0]
    scales = np.array(struct.unpack_from('<3d', header, 131))
    offsets = np.array(struct.unpack_from('<3d', header, 155))
    bounds = np.array(struct.unpack_from('<6d', header, 179)).reshape(3, 2)
    if (header_size < 227 or offset < header_size or record_size < (20, 28, 26, 34)[point_format]
            or not count or offset + count * record_size > path.stat().st_size
            or not np.isfinite(scales).all() or np.any(scales <= 0)
            or not np.isfinite(offsets).all() or not np.isfinite(bounds).all()
            or np.any(bounds[:, 0] < bounds[:, 1])):
        raise ValueError('Invalid or truncated LAS point records/header')
    return {'version': '1.2', 'point_format': point_format, 'point_count': count,
            'record_size': record_size, 'point_data_offset': offset,
            'global_encoding': struct.unpack_from('<H', header, 6)[0],
            'vlr_count': struct.unpack_from('<I', header, 100)[0],
            'scales': scales.tolist(), 'offsets': offsets.tolist(),
            'bounds_min': bounds[:, 1].tolist(), 'bounds_max': bounds[:, 0].tolist(),
            'crs': 'absent' if not struct.unpack_from('<I', header, 100)[0] else 'not_parsed',
            'coordinate_units': 'unverified; registration thresholds assume metres'}


def sample_las_records(path, *, max_points=1_500_000, chunk_points=500_000, seed=0):
    """Choose one deterministic random record per stratum, then read in chunks.

    Sampling is independent of geometry, colour, pose, and registration errors.
    Withheld and class-7 noise records are excluded, with counts recorded.
    Return XYZ, metadata, and corresponding GPS times (None for formats 0/2).
    """
    if not 1 <= max_points <= 5_000_000 or chunk_points < 1:
        raise ValueError('Expected 1--5000000 sample points and a positive chunk size')
    path = Path(path)
    before = path.stat()
    header = read_header(path)
    count = header['point_count']
    step = max(1, int(np.ceil(count / max_points)))
    starts = np.arange(0, count, step, dtype=np.int64)
    indices = starts + np.random.default_rng(seed).integers(0, step, len(starts))
    indices = np.minimum(indices, count - 1)
    fields = {'names': ['raw_xyz', 'classification'],
              'formats': [('<i4', 3), 'u1'], 'offsets': [0, 15],
              'itemsize': header['record_size']}
    has_time = header['point_format'] in (1, 3)
    if has_time:
        fields['names'].append('gps_time')
        fields['formats'].append('<f8')
        fields['offsets'].append(20)
    dtype = np.dtype(fields)
    xyz_parts, time_parts = [], []
    rejected = 0
    with path.open('rb') as stream:
        stream.seek(header['point_data_offset'])
        for start in range(0, count, chunk_points):
            size = min(chunk_points, count - start)
            records = np.fromfile(stream, dtype=dtype, count=size)
            if len(records) != size:
                raise ValueError('LAS changed or was truncated while sampling')
            lo, hi = np.searchsorted(indices, [start, start + size])
            selected = records[indices[lo:hi] - start]
            flags = selected['classification']
            keep = ((flags & 128) == 0) & ((flags & 31) != 7)
            rejected += int(np.count_nonzero(~keep))
            selected = selected[keep]
            xyz_parts.append(selected['raw_xyz'] * np.asarray(header['scales'])
                             + np.asarray(header['offsets']))
            if has_time:
                time_parts.append(selected['gps_time'])
    after = path.stat()
    signature = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    if signature(before) != signature(after):
        raise ValueError('LAS changed while sampling')
    xyz = np.concatenate(xyz_parts)
    if len(xyz) < 30 or not np.isfinite(xyz).all():
        raise ValueError('Too few finite LAS samples')
    time_summary, times = None, None
    if has_time:
        times = np.concatenate(time_parts)
        finite = np.isfinite(times)
        time_summary = {'finite_count': int(finite.sum()),
                        'min': float(times[finite].min()) if finite.any() else None,
                        'max': float(times[finite].max()) if finite.any() else None,
                        'encoding': 'adjusted_standard' if header['global_encoding'] & 1 else 'gps_week'}
    return xyz, {**header, 'sampling': 'one random record per ordinal stratum; fixed seed',
                 'sample_seed': seed, 'record_stride': step, 'sampled_records': len(indices),
                 'sampled_points': len(xyz), 'excluded_withheld_or_noise': rejected,
                 'chunk_points': chunk_points, 'gps_time': time_summary}, times


def sample_las(path, *, max_points=1_500_000, chunk_points=500_000, seed=0):
    """Sample coordinates and metadata without exposing per-record attributes."""
    xyz, metadata, _ = sample_las_records(path, max_points=max_points, chunk_points=chunk_points, seed=seed)
    return xyz, metadata
