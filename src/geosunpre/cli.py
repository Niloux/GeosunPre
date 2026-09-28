"""Stable CLI for the selected asynchronous L/R SfM pipeline."""

import argparse
import json
from pathlib import Path
import sys


STABLE_COMMANDS = frozenset({
    'prepare', 'global-data', 'global-sfm', 'build-baseline',
    'export-colmap', 'check-colmap', 'check-computed-baseline',
    'baseline-check', 'export-baseline',
})


def _export_options(parser):
    parser.add_argument('--image-mode', choices=('link', 'copy'), default='link',
                        help='Link original images locally or copy them for a portable dataset')
    parser.add_argument('--include-masks', action='store_true',
                        help='Export the combined feature-domain masks as an optional extra')
    parser.add_argument('--include-ply', action='store_true',
                        help='Also export all sparse points as PLY')


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] not in STABLE_COMMANDS and argv[0] not in ('-h', '--help'):
        # Existing reports use `geosunpre <experiment>`; retain those commands
        # while presenting new experiments through the separate executable.
        from .experiments.cli import main as experiment_main
        return experiment_main(argv)

    parser = argparse.ArgumentParser(
        description='Selected asynchronous L/R SfM pipeline and frozen baseline; '
                    'historical diagnostics: geosunpre-exp --help')
    sub = parser.add_subparsers(dest='command', required=True)
    prepare = sub.add_parser('prepare', help='Select L/R images with matching pose records')
    prepare.add_argument('--input', required=True)
    prepare.add_argument('--work', required=True)
    prepare.add_argument('--fov', type=float, default=160)
    prepare.add_argument('--principal-y', choices=['up', 'down'], default='up')
    prepare.add_argument('--mask-root')

    data = sub.add_parser('global-data', help='Prepare full-scene masks, features, and matches')
    data.add_argument('--work', required=True)
    data.add_argument('--stage', choices=['masks', 'features', 'matches', 'all'], default='all')
    sfm = sub.add_parser('global-sfm', help='Map and optimize selected asynchronous L/R images')
    sfm.add_argument('--work', required=True)
    sfm.add_argument('--stage', choices=['mapping', 'mapping-eval', 'solve', 'compare', 'all'],
                     default='all')
    sfm.add_argument('--stop-after', choices=['fixed_calibration', 'triangle_supported',
                                              'shared_calibration'],
                     default='shared_calibration')

    build = sub.add_parser('build-baseline',
                           help='Run the unattended basketball SfM and native COLMAP export')
    build.add_argument('--input', help='Raw basketball input; required for prepare/all')
    build.add_argument('--work', required=True, help='Fresh or resumable work directory')
    build.add_argument('--stage', choices=['prepare', 'global-data', 'holdout', 'local-calibration',
                                           'global-sfm', 'equal-ba', 'export', 'all'],
                       default='all')
    build.add_argument('--output', help='COLMAP output; default is WORK/colmap')
    _export_options(build)
    colmap = sub.add_parser('export-colmap', help='Package an existing model and scene as COLMAP')
    colmap.add_argument('--model', required=True, help='Source COLMAP sparse model directory')
    colmap.add_argument('--scene', required=True, help='Prepared scene manifest JSON')
    colmap.add_argument('--output', required=True)
    _export_options(colmap)
    computed = sub.add_parser('check-colmap', aliases=['check-computed-baseline'],
                              help='Validate an exported COLMAP dataset without the original work directory')
    computed.add_argument('--work', help='Use WORK/colmap if --output is omitted')
    computed.add_argument('--output', help='Export directory to validate independently')

    check = sub.add_parser('baseline-check', help='Verify the adopted development baseline')
    check.add_argument('--work', help='Override the work directory in the baseline manifest')
    check.add_argument('--manifest', default='baselines/basketball/baseline.json')
    check.add_argument('--full', action='store_true',
                       help='Hash all original images and masks as well as models')
    export = sub.add_parser('export-baseline',
                            help='Package the frozen source model as native COLMAP')
    export.add_argument('--work', help='Override the work directory in the baseline manifest')
    export.add_argument('--manifest', default='baselines/basketball/baseline.json')
    export.add_argument('--output', help='Destination; default is WORK/colmap')
    _export_options(export)

    args = parser.parse_args(argv)
    if args.command == 'prepare':
        from .scene import prepare as prepare_scene
        scene = prepare_scene(args.input, args.work, args.fov, args.principal_y,
                              args.mask_root)
        result = {'images': len(scene['images']), 'pose_model': scene['pose_model'],
                  'excluded': scene['excluded'],
                  'time_audit': scene['navigation_time_audit'],
                  'manifest': args.work + '/manifest.json'}
    elif args.command == 'global-data':
        from .pipeline.global_data import (prepare_global, extract_global_features,
                                           match_global_features)
        if args.stage in ('masks', 'all'):
            prepare_global(args.work)
        if args.stage in ('features', 'all'):
            extract_global_features(args.work)
        if args.stage in ('matches', 'all'):
            match_global_features(args.work)
        return
    elif args.command == 'global-sfm':
        from .pipeline.global_reconstruction import (
            build_global_mapping, evaluate_global_mapping, solve_global,
            compare_global_stages)
        if args.stage in ('mapping', 'all'):
            build_global_mapping(args.work)
        if args.stage in ('mapping-eval', 'all'):
            evaluate_global_mapping(args.work)
        if args.stage in ('solve', 'all'):
            solve_global(args.work, stop_after=args.stop_after)
        if args.stage == 'compare':
            compare_global_stages(args.work)
        return
    elif args.command == 'build-baseline':
        from .pipeline.baseline_build import run_baseline
        result = run_baseline(args.work, input_root=args.input, stage=args.stage,
                              output=args.output, image_mode=args.image_mode,
                              include_masks=args.include_masks, include_ply=args.include_ply)
    elif args.command == 'export-colmap':
        from .pipeline.colmap_export import export_colmap
        from .scene import file_digest
        scene = json.loads(Path(args.scene).read_text())
        result = export_colmap(args.model, scene, args.output,
                               image_mode=args.image_mode, include_masks=args.include_masks,
                               include_ply=args.include_ply,
                               provenance={'scene_manifest_sha256': file_digest(args.scene)})
    elif args.command in ('check-colmap', 'check-computed-baseline'):
        if not args.work and not args.output:
            parser.error('check-colmap requires --output or --work')
        from .pipeline.colmap_export import verify_colmap_export
        output = args.output or Path(args.work) / 'colmap'
        result = verify_colmap_export(output)
    elif args.command == 'baseline-check':
        from .pipeline.baseline import check_baseline
        result = check_baseline(manifest_path=args.manifest, work_override=args.work,
                                full=args.full)
    else:
        from .pipeline.baseline import export_baseline
        result = export_baseline(manifest_path=args.manifest, work_override=args.work,
                                 output=args.output, image_mode=args.image_mode,
                                 include_masks=args.include_masks, include_ply=args.include_ply)
    print(json.dumps(result, ensure_ascii=False, indent=2))
