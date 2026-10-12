"""Report installed dependencies, CUDA access and the native extension."""
from importlib import import_module
import json
from pathlib import Path
import platform
import sys

from geosunpre.artifacts import write_json
from geosunpre.pipeline.task import runtime_environment


def main():
    for module in ('cv2', 'numpy', 'scipy', 'pycolmap', 'transformers', 'loma',
                   'geosunpre.pipeline.runner'):
        import_module(module)
    import geosunpre_native as native
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; install the locked CUDA dependencies and check the NVIDIA driver')
    result = {
        **runtime_environment(),
        'platform': platform.platform(),
        'executable': sys.executable,
        'cuda': {'runtime': torch.version.cuda, 'gpu': torch.cuda.get_device_name()},
        'native': {'implementation': native.implementation_version},
    }
    root = Path(__file__).resolve().parents[1]
    write_json(root / '.cache/environment_check.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
