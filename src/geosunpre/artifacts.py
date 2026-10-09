"""File metadata and atomic publication shared by pipeline stages."""

import json
from tempfile import TemporaryDirectory
from uuid import uuid4
from contextlib import contextmanager
from pathlib import Path

def file_metadata(path):
    """Describe a local file for resuming work, without reading its contents."""
    stat = Path(path).stat()
    return {'size_bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def write_frozen_json(path, value):
    """Reject changed inputs before replacing an existing completed artifact."""
    path = Path(path)
    if path.exists():
        # Normalize tuples and integer keys to their on-disk JSON representation.
        if json.loads(path.read_text()) != json.loads(json.dumps(value, allow_nan=False)):
            raise ValueError(f'Frozen inputs changed; use a fresh work directory: {path}')
    else:
        write_json(path, value)


def archive_interrupted(path):
    """Preserve an uncommitted, pipeline-owned artifact before rebuilding it."""
    path = Path(path)
    if path.exists():
        archive = path.with_name(f'{path.name}.interrupted-{uuid4().hex[:12]}')
        path.rename(archive)
        for suffix in ('-wal', '-shm', '-journal'):
            companion = path.with_name(path.name + suffix)
            if companion.exists():
                companion.rename(archive.with_name(archive.name + suffix))
        return archive


@contextmanager
def staged_directory(destination):
    """Publish a complete stage; callers must only replace uncommitted outputs."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f'.{destination.name}-', dir=destination.parent) as temporary:
        stage = Path(temporary)
        yield stage
        if destination.is_dir() and not any(destination.iterdir()):
            destination.rmdir()
        else:
            archive_interrupted(destination)
        stage.rename(destination)
