"""Transport to the standalone solver while the remaining stages migrate."""
from functools import lru_cache
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory


def backend():
    value = os.environ.get('GEOSUNPRE_NUMERIC_BACKEND', 'cpp' if sys.platform == 'win32' else 'python')
    if value not in ('cpp', 'python'):
        raise ValueError('GEOSUNPRE_NUMERIC_BACKEND must be cpp or python')
    return value


def executable():
    root = Path(__file__).resolve().parents[2]
    path = Path(os.environ.get('GEOSUNPRE_NATIVE_EXECUTABLE',
        str(root / '.cache/native/build/bin/geosunpre-native.exe'))).resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Native solver missing: {path}; run setup-native.ps1')
    return path


@lru_cache(maxsize=1)
def info():
    program = executable()
    value = json.loads(subprocess.check_output([str(program), 'info'], encoding='utf-8'))
    if value['implementation'] != 1 or value['colmap'] != '4.2.0':
        raise ValueError('Native solver does not match the current numerical interface')
    return {'executable': str(program), **value}


def bundle_adjustment(rec, stage, initial_cameras, *, protect_depth=False,
                      show_progress=False, max_iterations=None):
    program = executable()
    info()
    with TemporaryDirectory(prefix='geosunpre-ba-') as temporary:
        root = Path(temporary)
        source = root / 'input'
        source.mkdir()
        rec.write(source)
        initial = root / 'cameras.json'
        initial.write_text(json.dumps({str(cid): list(values) for cid, values in initial_cameras.items()},
                                      allow_nan=False), encoding='utf-8')
        output = root / 'output'
        command = [str(program), 'ba', '--input', str(source), '--initial-cameras', str(initial),
                   '--output', str(output), '--stage', stage]
        if protect_depth:
            command.append('--protect-depth')
        if show_progress:
            command.append('--progress')
        if max_iterations is not None:
            command += ['--iterations', str(max_iterations)]
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL if not show_progress else None)
        report = json.loads((output / 'ba.json').read_text(encoding='utf-8'))
        from geosunpre.profiling import record_native_resources
        record_native_resources(report)
        rec.read(output / 'sparse/0')
        return report


def map_from_inputs(scene, database, seed):
    import pycolmap
    program = executable()
    info()
    with TemporaryDirectory(prefix='geosunpre-mapper-') as temporary:
        output = Path(temporary) / 'output'
        subprocess.run([str(program), 'map', '--database', str(database), '--seed', str(seed),
                        '--images', str(Path(scene['input_root']) / 'Image'), '--output', str(output)],
                       check=True)
        from geosunpre.profiling import record_native_resources
        record_native_resources(json.loads((output / 'mapping.json').read_text(encoding='utf-8')))
        return {0: pycolmap.Reconstruction(output / 'sparse/0')}
