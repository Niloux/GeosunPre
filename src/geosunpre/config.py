"""Resource and delivery limits for the single production recipe."""

from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path
import tomllib

from geosunpre.artifacts import write_frozen_json

CONFIG_FILE = 'pipeline_config.json'
CONFIG_VERSION = 8


@dataclass(frozen=True)
class PipelineConfig:
    workers: int = 4
    registration_las: str | None = None
    quality_max_median_px: float = 2.
    quality_max_p90_px: float = 4.
    registration_min_support_fraction: float = .25
    registration_max_median_m: float = .1
    registration_max_p90_m: float = .3

    def __post_init__(self):
        if type(self.workers) is not int or not 1 <= self.workers <= 16:
            raise ValueError('workers must be an integer between 1 and 16')
        if self.registration_las is not None and (
                not isinstance(self.registration_las, str) or not self.registration_las.strip()):
            raise ValueError('registration_las must be a nonempty path')
        for key in ('quality_max_median_px', 'quality_max_p90_px',
                    'registration_min_support_fraction', 'registration_max_median_m',
                    'registration_max_p90_m'):
            value = getattr(self, key)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{key} must be a finite positive number')
        if self.registration_min_support_fraction > 1:
            raise ValueError('registration_min_support_fraction must be <= 1')
        for prefix, suffix in (('quality_max_', '_px'), ('registration_max_', '_m')):
            if getattr(self, prefix + 'median' + suffix) > getattr(self, prefix + 'p90' + suffix):
                raise ValueError(prefix + 'median' + suffix + ' must not exceed the P90 limit')

    @classmethod
    def from_file(cls, path):
        with Path(path).open('rb') as stream:
            data = tomllib.load(stream)
        unknown = set(data) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f'Unknown pipeline settings: {sorted(unknown)}')
        return cls(**data)

    def check_scene(self, scene):
        if {row['camera'] for row in scene['images']} != {'L', 'R'}:
            raise ValueError('Selected input must contain both L/R cameras')


def save_config(work, config):
    path = Path(work) / CONFIG_FILE
    if path.exists():
        if load_config(work) != config:
            raise ValueError('Frozen inputs changed: pipeline configuration; use a fresh work directory')
        return
    value = {'version': CONFIG_VERSION, 'settings': asdict(config)}
    write_frozen_json(path, value)


def load_config(work):
    path = Path(work) / CONFIG_FILE
    if not path.is_file():
        raise ValueError('Missing pipeline_config.json; start a run in a fresh work directory')
    value = json.loads(path.read_text())
    if value.get('version') != CONFIG_VERSION:
        raise ValueError('Unsupported pipeline configuration; use a fresh work directory')
    if set(value.get('settings', {})) != {field.name for field in fields(PipelineConfig)}:
        raise ValueError('Invalid pipeline configuration fields')
    return PipelineConfig(**value['settings'])
