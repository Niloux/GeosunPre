"""Content identities and atomic metadata writes shared by pipeline stages."""

import hashlib
import json
import os
from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

_digest_cache = ContextVar('file_digest_cache', default=None)


@contextmanager
def digest_cache():
    """Reuse hashes only within one invocation and while file metadata is stable."""
    token = _digest_cache.set(OrderedDict())
    try:
        yield
    finally:
        _digest_cache.reset(token)


def _file_signature(stat):
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def file_digest(path):
    path = Path(path)
    cache = _digest_cache.get()
    signature = _file_signature(path.stat())
    key = (str(path.absolute()), signature)
    if cache is not None and key in cache:
        cache.move_to_end(key)
        return cache[key]
    result = hashlib.sha256()
    with path.open('rb') as stream:
        if _file_signature(os.fstat(stream.fileno())) != signature:
            raise ValueError(f'File changed while hashing: {path}')
        while block := stream.read(1024 * 1024):
            result.update(block)
        if _file_signature(os.fstat(stream.fileno())) != signature:
            raise ValueError(f'File changed while hashing: {path}')
    if _file_signature(path.stat()) != signature:
        raise ValueError(f'File changed while hashing: {path}')
    value = result.hexdigest()
    if cache is not None:
        cache[key] = value
        if len(cache) > 16384:
            cache.popitem(last=False)
    return value


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
        if digest(json.loads(path.read_text())) != digest(value):
            raise ValueError(f'Frozen inputs changed; use a fresh work directory: {path}')
    else:
        write_json(path, value)
