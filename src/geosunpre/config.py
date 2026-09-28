"""Scene selection settings; the validated fisheye solver recipe is fixed."""

from collections import Counter
from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path
import tomllib

from geosunpre.artifacts import digest, write_frozen_json


CONFIG_FILE = 'pipeline_config.json'
CONFIG_VERSION = 1


@dataclass(frozen=True)
class PipelineConfig:
    expected_images: int | None = None
    calibration_frames: tuple[int, int] | None = None
    calibration_frame_count: int = 25

    def __post_init__(self):
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
        if self.calibration_frames is None:
            frames = sorted({row['frame_index'] for row in scene['images']})
            if len(frames) < 2:
                raise ValueError('Calibration requires at least two distinct frame indices')
            selected = set(frames[:self.calibration_frame_count])
            images = [row for row in scene['images'] if row['frame_index'] in selected]
        else:
            low, high = self.calibration_frames
            images = [row for row in scene['images'] if low <= row['frame_index'] <= high]
        counts = Counter(row['camera'] for row in images)
        frames = {row['frame_index'] for row in images}
        if set(counts) != {'L', 'R'} or len(frames) < 2:
            raise ValueError('Calibration selection needs both L/R cameras and at least two frame indices')
        return images


def save_config(work, config):
    value = {'version': CONFIG_VERSION, 'settings': asdict(config)}
    write_frozen_json(Path(work) / CONFIG_FILE, {**value, 'fingerprint': digest(value)})


def load_config(work):
    path = Path(work) / CONFIG_FILE
    if not path.is_file():
        raise ValueError('Missing pipeline_config.json; start a run in a fresh work directory')
    value = json.loads(path.read_text())
    if (value.get('version') != CONFIG_VERSION or value.get('fingerprint') != digest(
            {key: val for key, val in value.items() if key != 'fingerprint'})):
        raise ValueError('Pipeline configuration identity changed')
    return PipelineConfig(**value['settings'])
