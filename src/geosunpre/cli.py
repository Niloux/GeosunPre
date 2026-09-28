"""Public commands for reconstruction, COLMAP packaging, and independent checking."""

import argparse
import json
from pathlib import Path

from geosunpre.pipeline import STAGES


def _export_options(parser):
    parser.add_argument('--image-mode', choices=('copy', 'link'), default='copy',
                        help='Copy original images for delivery (default), or link them locally')
    parser.add_argument('--include-masks', action='store_true',
                        help='Include combined FOV, semantic, and device masks')
    parser.add_argument('--include-ply', action='store_true',
                        help='Also write all sparse points as PLY')


def main(argv=None):
    parser = argparse.ArgumentParser(description='Automatic Geosun L/R fisheye preprocessing to COLMAP')
    sub = parser.add_subparsers(dest='command', required=True)
    run = sub.add_parser('run', help='Run or resume the input-to-COLMAP pipeline')
    run.add_argument('--input', help='Raw input directory; required for a new run')
    run.add_argument('--work', required=True, help='Fresh or resumable work directory')
    run.add_argument('--config', help='Scene selection TOML; frozen on first run and reused on resume')
    run.add_argument('--stage', choices=(*STAGES, 'all'), default='all')
    run.add_argument('--output', help='COLMAP destination; default WORK/colmap')
    run.add_argument('--model-cache', help='Hugging Face cache for SegFormer weights')
    _export_options(run)

    export = sub.add_parser('export', help='Package an existing sparse model and scene as COLMAP')
    export.add_argument('--model', required=True, help='COLMAP sparse model directory')
    export.add_argument('--scene', required=True, help='Prepared scene manifest JSON')
    export.add_argument('--output', required=True)
    _export_options(export)

    check = sub.add_parser('check', help='Verify a delivered COLMAP dataset independently')
    location = check.add_mutually_exclusive_group(required=True)
    location.add_argument('--output', help='COLMAP export directory')
    location.add_argument('--work', help='Check WORK/colmap')
    args = parser.parse_args(argv)
    if args.command == 'run':
        from geosunpre.pipeline.runner import run_pipeline
        result = run_pipeline(
            args.work, input_root=args.input, config=args.config, stage=args.stage,
            output=args.output, image_mode=args.image_mode,
            include_masks=args.include_masks, include_ply=args.include_ply,
            model_cache=args.model_cache)
    elif args.command == 'export':
        from geosunpre.artifacts import file_digest
        from geosunpre.colmap import export_colmap
        scene = json.loads(Path(args.scene).read_text())
        result = export_colmap(
            args.model, scene, args.output, image_mode=args.image_mode,
            include_masks=args.include_masks, include_ply=args.include_ply,
            provenance={'scene_manifest_sha256': file_digest(args.scene)})
    else:
        from geosunpre.colmap import verify_colmap_export
        result = verify_colmap_export(args.output or Path(args.work) / 'colmap')
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
