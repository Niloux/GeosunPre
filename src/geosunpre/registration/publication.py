"""Commit and verify complete registration stages before their atomic publication."""

import json
from pathlib import Path

from geosunpre.artifacts import file_metadata, write_json


def _inventory(directory):
    result = {}
    for path in sorted(Path(directory).rglob('*')):
        relative = path.relative_to(directory)
        if relative.as_posix() == 'completion.json':
            continue
        if path.is_file():
            result[relative.as_posix()] = file_metadata(path)
    return result


def complete_publication(directory, context):
    write_json(Path(directory) / 'completion.json', {
        'version': 1, 'context': context,
        'files': _inventory(directory)})


def verify_publication(directory, context):
    directory = Path(directory)
    path = directory / 'completion.json'
    if not path.is_file():
        raise ValueError(f'Registration stage has no committed identity: {directory}')
    saved = json.loads(path.read_text(encoding='utf-8'))
    if (saved.get('version') != 1 or saved.get('context') != context
            or saved.get('files') != _inventory(directory)):
        raise ValueError(f'Completed registration stage changed: {directory}')
    return saved
