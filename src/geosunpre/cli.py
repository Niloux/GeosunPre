"""Reconstruction, COLMAP packaging, checking, and post-SfM registration diagnostics."""

import argparse
import json
from pathlib import Path

from geosunpre.pipeline import STAGES


def _export_options(parser, *, resume=False):
    parser.add_argument('--image-mode', choices=('copy', 'link'), default=None if resume else 'copy',
                        help='Copy original images for delivery (default), or link them locally')
    parser.add_argument('--include-masks', action=argparse.BooleanOptionalAction,
                        default=None if resume else False,
                        help='Include combined FOV, semantic, and device masks')
    parser.add_argument('--include-ply', action=argparse.BooleanOptionalAction,
                        default=None if resume else False,
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
    _export_options(run, resume=True)

    export = sub.add_parser('export', help='Package an existing sparse model and scene as COLMAP')
    export.add_argument('--model', required=True, help='COLMAP sparse model directory')
    export.add_argument('--scene', required=True, help='Prepared scene manifest JSON')
    export.add_argument('--output', required=True)
    _export_options(export)

    check = sub.add_parser('check', help='Verify a delivered COLMAP dataset independently')
    location = check.add_mutually_exclusive_group(required=True)
    location.add_argument('--output', help='COLMAP export directory')
    location.add_argument('--work', help='Check WORK/colmap')

    align = sub.add_parser('align', help='Diagnose a global visual-to-LAS registration candidate')
    align.add_argument('--colmap', required=True, help='Existing delivered visual COLMAP dataset')
    align.add_argument('--las', required=True, help='Uncompressed LAS 1.2 file (point formats 0--3)')
    align.add_argument('--output', required=True, help='Fresh alignment diagnostic directory')
    initialization = align.add_mutually_exclusive_group(required=True)
    initialization.add_argument('--input', help='Raw Geosun input with CameraPos and both navigation files')
    initialization.add_argument('--initial-transform', help='JSON with a 4x4 visual-to-LAS matrix')
    align.add_argument('--max-las-points', type=int, default=1_500_000, help='Maximum LAS sample count')
    align.add_argument('--workers', type=int, default=4, help='CPU workers for spatial queries (default 4)')

    inspect = sub.add_parser('inspect-alignment', help='Inspect structures and low support in a frozen alignment')
    inspect.add_argument('--alignment', required=True, help='Completed alignment diagnostic directory')
    inspect.add_argument('--output', required=True, help='Fresh structural inspection directory')
    inspect.add_argument('--workers', type=int, default=4, help='CPU workers for spatial queries (default 4)')

    refine = sub.add_parser('refine-alignment', help='Refine a reviewed candidate using coherent structures')
    refine.add_argument('--alignment', required=True, help='Existing alignment candidate; frozen points/split are retained')
    refine.add_argument('--output', required=True, help='Fresh refinement diagnostic directory')
    refine.add_argument('--workers', type=int, default=4, help='CPU workers for spatial queries (default 4)')

    joint = sub.add_parser('joint-alignment', help='Compare fixed-intrinsics image / pose / LAS joint BA candidates')
    joint.add_argument('--alignment', required=True, help='Existing visual-to-LAS candidate with frozen points and split')
    joint.add_argument('--input', required=True, help='Raw input with named CameraPos and navigation coordinate files')
    joint.add_argument('--output', required=True, help='Fresh joint BA experiment directory')
    joint.add_argument('--workers', type=int, default=4)
    joint.add_argument('--outer-iterations', type=int, default=2)
    joint.add_argument('--iterations', type=int, default=20, help='Ceres iterations per fixed-correspondence solve')
    joint.add_argument('--pose-std', type=float, default=.5, help='Conservative position noise proxy in metres')
    joint.add_argument('--las-std', type=float, default=.05, help='Conservative surface noise proxy in metres')
    joint.add_argument('--pose-weight', type=float, default=.001, help='Position objective budget relative to image observations')
    joint.add_argument('--las-weight', type=float, default=.005, help='LAS objective budget relative to image observations')
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
    elif args.command == 'align':
        from geosunpre.alignment.runner import run_alignment
        result = run_alignment(args.colmap, args.las, args.output, input_root=args.input,
                               initial_transform=args.initial_transform,
                               max_las_points=args.max_las_points, workers=args.workers)
    elif args.command == 'inspect-alignment':
        from geosunpre.alignment.inspection import inspect_alignment
        result = inspect_alignment(args.alignment, args.output, workers=args.workers)
    elif args.command == 'refine-alignment':
        from geosunpre.alignment.runner import refine_alignment
        result = refine_alignment(args.alignment, args.output, workers=args.workers)
    elif args.command == 'joint-alignment':
        from geosunpre.alignment.joint import run_joint_alignment
        result = run_joint_alignment(args.alignment, args.input, args.output, workers=args.workers,
            outer_iterations=args.outer_iterations, iterations=args.iterations,
            pose_std=args.pose_std, las_std=args.las_std, pose_weight=args.pose_weight, las_weight=args.las_weight)
    else:
        from geosunpre.colmap import verify_colmap_export
        result = verify_colmap_export(args.output or Path(args.work) / 'colmap')
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
