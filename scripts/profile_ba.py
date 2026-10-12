"""Replay a real frozen BA input without writing to its work directory."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import shutil
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--initial-model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage', choices=('fixed', 'principal'), default='principal')
    parser.add_argument('--iterations', type=int, default=5)
    parser.add_argument('--blas-library', type=Path)
    parser.add_argument('--site-root', type=Path)
    args = parser.parse_args()
    os.environ['GEOSUNPRE_NUMERIC_BACKEND'] = 'python'
    if args.output.exists():
        raise FileExistsError(args.output)
    libraries = []
    if args.site_root:
        sys.path.insert(0, str(args.site_root.resolve()))
    if args.blas_library:
        if sys.platform != 'win32':
            raise ValueError('BLAS DLL substitution is a Windows diagnostic only')
        site = Path(sys.executable).parents[1] / 'Lib/site-packages'
        preload = args.output.parent / 'blas-preload'
        if not args.blas_library.is_dir():
            preload.mkdir(parents=True, exist_ok=True)
        handles = []
        if args.blas_library.is_dir():
            handles.append(os.add_dll_directory(str(args.blas_library.resolve())))
        if args.site_root:
            handles.append(os.add_dll_directory(str(args.site_root.resolve())))
        # Preload under the wheel's dependency names in this process only.
        # The installed environment and original DLLs are never replaced.
        names = list(site.glob('openblas*.dll')) + list((site / 'pycolmap.libs').glob('openblas*.dll'))
        if not names:
            raise FileNotFoundError('No wheel OpenBLAS DLLs found')
        for original in names:
            if args.blas_library.is_dir():
                destination = args.blas_library / original.name
            else:
                destination = preload / original.name
                shutil.copyfile(args.blas_library, destination)
            library = ctypes.CDLL(str(destination.resolve()))
            handles.append(library)
            library.openblas_set_num_threads(1)
            library.openblas_get_config.restype = ctypes.c_char_p
            libraries.append({'library': str(destination), 'config': library.openblas_get_config().decode()})
    import pycolmap
    from geosunpre.sfm.reconstruction import bounded_bundle_adjustment
    initial = pycolmap.Reconstruction(args.initial_model)
    model = pycolmap.Reconstruction(args.model)
    started = time.perf_counter()
    report = bounded_bundle_adjustment(model, args.stage,
        {cid: camera.params.copy() for cid, camera in initial.cameras.items()},
        protect_depth=True, max_iterations=args.iterations)
    result = {'model': str(args.model.resolve()), 'initial_model': str(args.initial_model.resolve()),
              'stage': args.stage, 'requested_iterations': args.iterations,
              'wall_seconds': time.perf_counter() - started, 'blas': libraries, 'ba': report}
    result['colmap_module'] = pycolmap.__file__
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('stage', 'wall_seconds', 'blas')}))
    print(report['solver_details']['full_report'])


if __name__ == '__main__':
    main()
