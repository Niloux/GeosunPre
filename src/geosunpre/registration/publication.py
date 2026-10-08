"""Commit and verify complete registration stages before their atomic publication."""

import json
from pathlib import Path

from geosunpre.artifacts import file_digest, write_json


def _inventory(directory):
    result = {}
    for path in sorted(Path(directory).rglob('*')):
        relative = path.relative_to(directory)
        if relative.as_posix() == 'completion.json':
            continue
        if path.is_file():
            result[relative.as_posix()] = file_digest(path)
    return result


def complete_publication(directory, context_fingerprint):
    write_json(Path(directory) / 'completion.json', {
        'version': 1, 'context_fingerprint': context_fingerprint,
        'file_sha256': _inventory(directory)})


def verify_publication(directory, context_fingerprint):
    directory = Path(directory)
    path = directory / 'completion.json'
    if not path.is_file():
        raise ValueError(f'Registration stage has no committed identity: {directory}')
    saved = json.loads(path.read_text())
    if (saved.get('version') != 1 or saved.get('context_fingerprint') != context_fingerprint
            or saved.get('file_sha256') != _inventory(directory)):
        raise ValueError(f'Completed registration stage changed: {directory}')
    return saved
