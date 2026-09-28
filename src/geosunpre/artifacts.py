"""Content identities and atomic metadata writes shared by pipeline stages."""

import hashlib
import json
from pathlib import Path


def digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def file_digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(1024 * 1024):
            result.update(block)
    return result.hexdigest()


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
