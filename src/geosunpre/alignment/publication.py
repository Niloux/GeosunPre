"""Commit and verify complete registration stages before their atomic publication."""

import json
from pathlib import Path

from geosunpre.artifacts import file_digest, write_json
from geosunpre.colmap import verify_colmap_export


def _inventory(directory):
    result = {}
    for path in sorted(Path(directory).rglob('*')):
        relative = path.relative_to(directory)
        # Image and mask identities are verified by each COLMAP manifest below.
        if relative.as_posix() == 'completion.json' or any(
                part in ('images', 'masks') for part in relative.parts):
            continue
        if path.is_file():
            result[relative.as_posix()] = file_digest(path)
    return result


def complete_publication(directory, context_fingerprint):
    if context_fingerprint is not None:
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
    for manifest in directory.rglob('colmap_export.json'):
        verify_colmap_export(manifest.parent)
    return saved
