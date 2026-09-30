"""Scene selection and matching backend, frozen for each pipeline run."""

from collections import Counter
from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path
import tomllib

from geosunpre.artifacts import digest, write_frozen_json


CONFIG_FILE = 'pipeline_config.json'
CONFIG_VERSION = 4


@dataclass(frozen=True)
class PipelineConfig:
    expected_images: int | None = None
    calibration_frames: tuple[int, int] | None = None
    calibration_frame_count: int = 25
    matcher_attention: str = 'flash'
    calibration_mode: str = 'local'
    pipeline_mode: str = 'production'
    quality_max_median_px: float | None = None
    quality_max_p90_px: float | None = None
    quality_bounds_policy: str = 'warn'

    def __post_init__(self):
        if self.quality_bounds_policy not in ('warn', 'reject'):
            raise ValueError('quality_bounds_policy must be warn or reject')
        for key in ('quality_max_median_px', 'quality_max_p90_px'):
            value = getattr(self, key)
            if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value <= 0):
                raise ValueError(f'{key} must be a finite positive number')
        if self.pipeline_mode not in ('production', 'validation'):
            raise ValueError('pipeline_mode must be production or validation')
        if self.calibration_mode not in ('local', 'input'):
            raise ValueError('calibration_mode must be local or input')
        if self.matcher_attention not in ('math', 'flash'):
            raise ValueError('matcher_attention must be math or flash')
        if self.expected_images is not None and (
                type(self.expected_images) is not int or self.expected_images < 2):
            raise ValueError('expected_images must be an integer >= 2')
        if type(self.calibration_frame_count) is not int or self.calibration_frame_count < 2:
            raise ValueError('calibration_frame_count must be an integer >= 2')
        if self.calibration_frames is not None:
            value = self.calibration_frames
            if (not isinstance(value, (list, tuple)) or len(value) != 2
                    or any(type(v) is not int for v in value)
                    or value[0] < 0 or value[1] <= value[0]):
                raise ValueError('calibration_frames must be [first, last], with first < last')
            object.__setattr__(self, 'calibration_frames', tuple(value))

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
        counts = Counter(row['camera'] for row in scene['images'])
        if set(counts) != {'L', 'R'}:
            raise ValueError('Selected input must contain both L/R cameras')

    def calibration_images(self, scene):
        self.check_scene(scene)
        if self.calibration_mode == 'input':
            return []
        if self.calibration_frames is None:
            frames = sorted({row['frame_index'] for row in scene['images']})
            if len(frames) < 2:
                raise ValueError('Calibration requires at least two distinct frame indices')
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
        counts = Counter(row['camera'] for row in images)
        frames = {row['frame_index'] for row in images}
        if set(counts) != {'L', 'R'} or len(frames) < 2:
            raise ValueError('Calibration selection needs both L/R cameras and at least two frame indices')
        return images


def save_config(work, config):
    path = Path(work) / CONFIG_FILE
    if path.exists():
        # Older frozen configurations imply math attention. Compare decoded
        # settings so their bytes and downstream identity remain unchanged.
        if load_config(work) != config:
            raise ValueError('Frozen inputs changed: pipeline configuration; use a fresh work directory')
        return
    value = {'version': CONFIG_VERSION, 'settings': asdict(config)}
    write_frozen_json(path, {**value, 'fingerprint': digest(value)})


def load_config(work):
    path = Path(work) / CONFIG_FILE
    if not path.is_file():
        raise ValueError('Missing pipeline_config.json; start a run in a fresh work directory')
    value = json.loads(path.read_text())
    if (value.get('version') not in (1, 2, 3, CONFIG_VERSION) or value.get('fingerprint') != digest(
            {key: val for key, val in value.items() if key != 'fingerprint'})):
        raise ValueError('Pipeline configuration identity changed')
    settings = dict(value['settings'])
    if value['version'] == 1:
        settings.setdefault('matcher_attention', 'math')
    elif 'matcher_attention' not in settings:
        raise ValueError('Missing frozen matcher_attention')
    if value['version'] < 3:
        if settings.get('calibration_mode', 'local') != 'local':
            raise ValueError('Legacy configuration cannot select input calibration')
        settings.setdefault('calibration_mode', 'local')
    elif 'calibration_mode' not in settings:
        raise ValueError('Missing frozen calibration_mode')
    if value['version'] < 4:
        if settings.get('pipeline_mode', 'validation') != 'validation':
            raise ValueError('Legacy configuration cannot select production mode')
        settings.setdefault('pipeline_mode', 'validation')
    elif 'pipeline_mode' not in settings:
        raise ValueError('Missing frozen pipeline_mode')
    return PipelineConfig(**settings)
