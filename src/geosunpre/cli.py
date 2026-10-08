"""Run, export and independently check the production COLMAP workflow."""

import argparse
import json
from pathlib import Path

from geosunpre.pipeline import STAGES


def _delivery_options(parser):
    parser.add_argument('--output', help='COLMAP destination; default WORK/colmap')
    parser.add_argument('--image-mode', choices=('copy', 'link'),
                        help='Copy images for portable delivery (default), or link locally')
    parser.add_argument('--include-masks', action=argparse.BooleanOptionalAction, default=None,
                        help='Deliver combined masks (default: enabled)')
    parser.add_argument('--include-ply', action=argparse.BooleanOptionalAction, default=None,
                        help='Deliver voxel-downsampled LAS for 3DGS initialization (default: enabled)')
    parser.add_argument('--ply-voxel-size', type=float,
                        help='LAS initialization voxel size in metres (default: 0.05)')
    parser.add_argument('--model-cache', help='Hugging Face cache root; inherited on resume')


def main(argv=None):
    parser = argparse.ArgumentParser(description='Geosun L/R fisheye and LAS preprocessing for 3DGS')
    sub = parser.add_subparsers(dest='command', required=True)
    run = sub.add_parser('run', help='Run or resume the production pipeline')
    run.add_argument('--input', help='Raw input directory; required for a new run')
    run.add_argument('--work', required=True, help='Fresh or resumable work directory')
    run.add_argument('--config', help='Scene TOML; frozen on first run')
    run.add_argument('--stage', choices=(*STAGES, 'all'), default='all')
    _delivery_options(run)
    export = sub.add_parser('export', help='Export an accepted pipeline model')
    export.add_argument('--work', required=True)
    _delivery_options(export)
    check = sub.add_parser('check', help='Verify a portable COLMAP delivery independently')
    location = check.add_mutually_exclusive_group(required=True)
    location.add_argument('--output', help='COLMAP export directory')
    location.add_argument('--work', help='Check the frozen task delivery destination')
    args = parser.parse_args(argv)
    try:
        if args.command in ('run', 'export'):
            from geosunpre.pipeline.runner import run_pipeline
            result = run_pipeline(args.work,
                input_root=getattr(args, 'input', None), config=getattr(args, 'config', None),
                stage=getattr(args, 'stage', 'export'), output=args.output,
                image_mode=args.image_mode, include_masks=args.include_masks,
                include_ply=args.include_ply, ply_voxel_size=args.ply_voxel_size,
                model_cache=args.model_cache)
        else:
            from geosunpre.colmap import verify_colmap_export
            if args.output:
                output = args.output
            else:
                task = json.loads((Path(args.work).expanduser() / 'pipeline_task.json').read_text())
                output = task['delivery']['output']
            result = verify_colmap_export(output)
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        parser.exit(1, f'geosunpre: {exc}\n')
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
