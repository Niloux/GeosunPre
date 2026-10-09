"""Scene settings for the single production recipe."""

from collections import Counter
from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path
import tomllib

from geosunpre.artifacts import write_frozen_json

CONFIG_FILE = 'pipeline_config.json'
CONFIG_VERSION = 7


@dataclass(frozen=True)
class PipelineConfig:
    expected_images: int | None = None
    calibration_frames: tuple[int, int] | None = None
    calibration_frame_count: int = 25
    workers: int = 4
    registration_las: str | None = None
    quality_max_median_px: float = 2.
    quality_max_p90_px: float = 4.
    registration_min_support_fraction: float = .25
    registration_max_median_m: float = .1
    registration_max_p90_m: float = .3

    def __post_init__(self):
        for key, low, high in (('expected_images', 2, None),
                               ('calibration_frame_count', 2, None), ('workers', 1, 16)):
            value = getattr(self, key)
            if key == 'expected_images' and value is None:
                continue
            if type(value) is not int or value < low or (high is not None and value > high):
                raise ValueError(f'{key} must be an integer >= {low}' +
                                 (f' and <= {high}' if high is not None else ''))
        if self.calibration_frames is not None:
            value = self.calibration_frames
            if (not isinstance(value, (list, tuple)) or len(value) != 2
                    or any(type(v) is not int for v in value)
                    or value[0] < 0 or value[1] <= value[0]):
                raise ValueError('calibration_frames must be [first, last], with first < last')
            object.__setattr__(self, 'calibration_frames', tuple(value))
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
        count = len(scene['images'])
        if self.expected_images is not None and count != self.expected_images:
            raise ValueError(f'Expected {self.expected_images} selected images, got {count}')
        if set(Counter(row['camera'] for row in scene['images'])) != {'L', 'R'}:
            raise ValueError('Selected input must contain both L/R cameras')

    def calibration_images(self, scene):
        self.check_scene(scene)
        if self.calibration_frames is None:
            frames = sorted({row['frame_index'] for row in scene['images']})
            images = []
            for start in range(len(frames) - 1):
                selected = set(frames[start:start + self.calibration_frame_count])
                candidate = [row for row in scene['images'] if row['frame_index'] in selected]
                if {row['camera'] for row in candidate} == {'L', 'R'}:
                    images = candidate
                    break
        else:
            low, high = self.calibration_frames
            images = [row for row in scene['images'] if low <= row['frame_index'] <= high]
        if ({row['camera'] for row in images} != {'L', 'R'}
                or len({row['frame_index'] for row in images}) < 2):
            raise ValueError('Calibration selection needs both L/R cameras and at least two frame indices')
        return images


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
