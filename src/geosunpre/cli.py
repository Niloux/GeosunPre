"""Stable CLI for the selected asynchronous L/R SfM pipeline."""

import argparse
import json
import sys


STABLE_COMMANDS = frozenset({
    'prepare', 'global-data', 'global-sfm', 'baseline-check', 'export-baseline',
})


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

    check = sub.add_parser('baseline-check', help='Verify the adopted development baseline')
    check.add_argument('--work', help='Override the work directory in the baseline manifest')
    check.add_argument('--manifest', default='baselines/basketball/baseline.json')
    check.add_argument('--full', action='store_true',
                       help='Hash all original images and masks as well as models')
    export = sub.add_parser('export-baseline',
                            help='Package the verified equal-control COLMAP branch for LFS')
    export.add_argument('--work', help='Override the work directory in the baseline manifest')
    export.add_argument('--manifest', default='baselines/basketball/baseline.json')
    export.add_argument('--output', help='Destination; default is work/basketball/baseline_colmap')
    export.add_argument('--full', action='store_true',
                        help='Hash all original images and masks before packaging')

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
    elif args.command == 'baseline-check':
        from .pipeline.baseline import check_baseline
        result = check_baseline(manifest_path=args.manifest, work_override=args.work,
                                full=args.full)
    else:
        from .pipeline.baseline import export_baseline
        result = export_baseline(manifest_path=args.manifest, work_override=args.work,
                                 output=args.output, full=args.full)
    print(json.dumps(result, ensure_ascii=False, indent=2))
