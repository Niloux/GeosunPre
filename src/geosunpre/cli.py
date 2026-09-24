import argparse
import json

from .scene import prepare


def main():
    parser = argparse.ArgumentParser(description='Asynchronous dual-fisheye SfM preparation and LoMa diagnostics')
    sub = parser.add_subparsers(dest='command',required=True)
    p = sub.add_parser('prepare',help='Validate L/R pose-bound inputs and write independent exposure manifest')
    p.add_argument('--input',required=True)
    p.add_argument('--work',required=True)
    p.add_argument('--fov',type=float,default=160)
    p.add_argument('--principal-y',choices=['up','down'],default='up',help='Explicit initial calibration convention; not claimed verified')
    p.add_argument('--mask-root',help='Optional white-valid semantic masks, named L/0000/<original.JPG>.png')
    d = sub.add_parser('diagnose',help='Run bounded LoMa extraction, native fisheye verification and pairwise holdout')
    d.add_argument('--work',required=True)
    d.add_argument('--frames',help='Comma-separated frame indices; default is the basketball representative probe')
    d.add_argument('--detector-size',type=int,default=1024)
    d.add_argument('--keypoints',type=int,default=2048)
    d.add_argument('--match-threshold',type=float,default=.1)
    d.add_argument('--max-error',type=float,default=3.,help='Native/original pixel geometric threshold')
    d.add_argument('--seed',type=int,default=42)
    local = sub.add_parser('local-sfm',help='Run two continuous, masked local SfM and calibration experiments')
    local.add_argument('--work',required=True)
    local.add_argument('--stage',choices=['masks','frontend','solve','all'],default='all')
    transfer=sub.add_parser('transfer-check',help='Check fixed candidate calibrations on a third disjoint segment')
    transfer.add_argument('--work',required=True)
    transfer.add_argument('--stage',choices=['masks','frontend','solve','all'],default='all')
    transfer.add_argument('--continue-iteration-limit',action='store_true',help='Archive an iteration-limited candidate and perform one documented train-only continuation')
    review=sub.add_parser('review-residuals',help='Render native-image crops for heldout residual diagnosis')
    review.add_argument('--work',required=True)
    refine=sub.add_parser('refinement-probe',help='Compare native training-point refinement with frozen calibration and validation')
    refine.add_argument('--work',required=True)
    ablation=sub.add_parser('principal-ablation',help='Isolate the transferred R cy effect on the third segment')
    ablation.add_argument('--work',required=True)
    track=sub.add_parser('track-probe',help='Check training-only closed-triangle support with frozen validation')
    track.add_argument('--work',required=True)
    global_data=sub.add_parser('global-data',help='Prepare masks/features/matches for all selected asynchronous L/R images')
    global_data.add_argument('--work',required=True)
    global_data.add_argument('--stage',choices=['masks','features','matches','all'],default='all')
    global_sfm=sub.add_parser('global-sfm',help='Freeze global holdout, map all selected L/R images, and compare training-track support')
    global_sfm.add_argument('--work',required=True)
    global_sfm.add_argument('--stage',choices=['mapping','mapping-eval','solve','compare','all'],default='all')
    global_sfm.add_argument('--stop-after',choices=['fixed_calibration','triangle_supported','shared_calibration'],default='shared_calibration',help='Checkpoint after this global BA stage')
    trajectory=sub.add_parser('trajectory-review',help='Render scale-free L/R trajectory diagnostics from a global model')
    trajectory.add_argument('--work',required=True)
    trajectory.add_argument('--stage',default='mapping')
    photometric=sub.add_parser('photometric-review',help='Measure native JPEG differences on geometry-verified LoMa correspondences')
    photometric.add_argument('--work',required=True)
    global_review=sub.add_parser('global-review',help='Inspect selected full-scene withheld correspondences in native image crops')
    global_review.add_argument('--work',required=True)
    global_review.add_argument('--candidate',choices=['triangle_supported','shared_calibration'],default='triangle_supported')
    static_review=sub.add_parser('static-review',help='Freeze geometry-blind heldout candidates and render native-image manual review')
    static_review.add_argument('--work',required=True)
    static_eval=sub.add_parser('static-evaluate',help='Compare saved SfM stages only on manually confirmed static correspondences')
    static_eval.add_argument('--work',required=True)
    static_eval.add_argument('--annotations',required=True)
    ground_probe=sub.add_parser('ground-pose-probe',help='Compare equal and finite near-floor training weights in pose-only SfM refinement')
    ground_probe.add_argument('--work',required=True)
    ground_analyze=sub.add_parser('ground-pose-analyze',help='Analyze saved near-floor pose candidates by training and withheld manual groups')
    ground_analyze.add_argument('--work',required=True)
    joint_ground=sub.add_parser('ground-joint-ba',help='Compare equal-control and finite near-floor weights in full native fisheye BA')
    joint_ground.add_argument('--work',required=True)
    joint_analyze=sub.add_parser('ground-joint-analyze',help='Audit saved weighted full BA models and manual static groups')
    joint_analyze.add_argument('--work',required=True)
    ground_photo=sub.add_parser('ground-photo-proxy',help='Compare withheld near-floor native-image patch alignment for saved BA models')
    ground_photo.add_argument('--work',required=True)
    cleanup=sub.add_parser('sfm-track-cleanup',help='Filter weak and conflicting training tracks, then compare native-fisheye BA on frozen geometry checks')
    cleanup.add_argument('--work',required=True)
    cleanup.add_argument('--stage',choices=['audit','solve','all'],default='all')
    evidence=sub.add_parser('sfm-match-evidence',help='Audit frozen LoMa and native-image match evidence against manual labels without changing SfM')
    evidence.add_argument('--work',required=True)
    residuals=sub.add_parser('sfm-residual-structure',help='Audit saved SfM residual spatial structure without changing poses or points')
    residuals.add_argument('--work',required=True)
    lk_probe=sub.add_parser('sfm-lk-probe',help='Check cached LoMa training matches against bidirectional local image tracking')
    lk_probe.add_argument('--work',required=True)
    lk_refinement=sub.add_parser('sfm-lk-refinement',help='Compare conservative training-only LK coordinate refinement on one local SfM segment')
    lk_refinement.add_argument('--work',required=True)
    lk_refinement.add_argument('--stage',choices=['audit','all'],default='all')
    resolution=sub.add_parser('sfm-resolution-probe',help='Compare cached LoMa 1536 detector with separate 2048 detector on four adjacent image pairs')
    resolution.add_argument('--work',required=True)
    virtual=sub.add_parser('sfm-virtual-view-probe',help='Compare native LoMa features with virtual-pinhole features mapped back to fisheye pixels')
    virtual.add_argument('--work',required=True)
    virtual_audit=sub.add_parser('sfm-virtual-view-audit',help='Audit both frontends against one fixed visual pose model and native-image coverage')
    virtual_audit.add_argument('--work',required=True)
    virtual_cycles=sub.add_parser('sfm-virtual-cycle-probe',help='Compare native and virtual-view LoMa tracks over four three-image sequences')
    virtual_cycles.add_argument('--work',required=True)
    court_region=sub.add_parser('sfm-court-region-probe',help='Render reviewable court-color region proposals on six L images')
    court_region.add_argument('--work',required=True)
    bottom_mask=sub.add_parser('sfm-bottom-mask-probe',help='Audit a conservative court-pixel restoration of the legacy bottom device strip')
    bottom_mask.add_argument('--work',required=True)
    fov_audit=sub.add_parser('sfm-fov-audit',help='Separate candidate 160-degree fisheye cutoff from the legacy bottom device mask')
    fov_audit.add_argument('--work',required=True)
    fov_candidate=sub.add_parser('sfm-fov179-masks',help='Prepare isolated 179-degree L-camera candidate masks with conservative court restoration')
    fov_candidate.add_argument('--work',required=True)
    fov_frontend=sub.add_parser('sfm-fov179-frontend-probe',help='Compare native LoMa triplets after court-only fisheye FOV expansion')
    fov_frontend.add_argument('--work',required=True)
    court_flow=sub.add_parser('sfm-court-flow-probe',help='Check independent court corners with half-resolution flow mapped to native fisheye pixels')
    court_flow.add_argument('--work',required=True)
    court_structural=sub.add_parser('sfm-court-structural-probe',help='Check blurred structural court corners with bidirectional flow')
    court_structural.add_argument('--work',required=True)
    court_structure_sift=sub.add_parser('sfm-court-structure-sift-probe',help='Check SIFT matches around coarse court markings and color seams')
    court_structure_sift.add_argument('--work',required=True)
    court_down_loma=sub.add_parser('sfm-court-down-loma-probe',help='Sample LoMa features in a ground-focused virtual view and return to native fisheye')
    court_down_loma.add_argument('--work',required=True)
    court_down_loma.add_argument('--stage',choices=('prepare','match','all'),default='all')
    court_down_audit=sub.add_parser('sfm-court-down-loma-audit',help='Audit ground-view LoMa tracks with native-image bidirectional LK')
    court_down_audit.add_argument('--work',required=True)
    court_down_review=sub.add_parser('sfm-court-down-loma-review',help='Render native-pixel crops for lower-image novel ground-view LoMa cycles')
    court_down_review.add_argument('--work',required=True)
    court_plane=sub.add_parser('sfm-court-plane-probe',help='Fit image-only court homographies from ground-view LoMa and check color boundaries')
    court_plane.add_argument('--work',required=True)
    court_plane_pose=sub.add_parser('sfm-court-plane-pose-audit',help='Compare court homographies to the best plane warp allowed by saved visual poses')
    court_plane_pose.add_argument('--work',required=True)
    court_plane_common=sub.add_parser('sfm-court-plane-common-pose-audit',help='Fit one court plane against all three saved-pose pair warps')
    court_plane_common.add_argument('--work',required=True)
    court_trajectory=sub.add_parser('sfm-court-trajectory-probe',help='Scan L/R court visibility and audit cached LoMa plane motion along the trajectory')
    court_trajectory.add_argument('--work',required=True)
    court_trajectory.add_argument('--stage',choices=('scan','pairs','corroborate','cycles','review','all'),default='all')
    court_dense=sub.add_parser('sfm-court-dense-probe',help='Refine court homographies with isolated dense image registration and check graph cycles')
    court_dense.add_argument('--work',required=True)
    court_dense.add_argument('--stage',choices=('probe','review','all'),default='all')
    court_anchor=sub.add_parser('sfm-court-anchor-probe',help='Audit red/green/blue court junction identity in native and temporary view pixels')
    court_anchor.add_argument('--work',required=True)
    court_anchor_coverage=sub.add_parser('sfm-court-anchor-coverage',help='Check native-pixel junction holdout and existing visual-track coverage')
    court_anchor_coverage.add_argument('--work',required=True)
    court_native_anchor=sub.add_parser('sfm-court-native-anchor-probe',help='Audit two image-selected paint corners over L231-L242 in native fisheye pixels')
    court_native_anchor.add_argument('--work',required=True)
    court_native_eligibility=sub.add_parser('sfm-court-native-anchor-eligibility',help='Check native court corners against saved masks and fitted fisheye FOV')
    court_native_eligibility.add_argument('--work',required=True)
    court_paint_pair=sub.add_parser('sfm-court-paint-pair-probe',help='Scan paired court-paint corners over all pose-selected native L/R images')
    court_paint_pair.add_argument('--work',required=True)
    court_paint_pair.add_argument('--stage',choices=('scan','audit','review','stability'),default='audit')
    court_white_line=sub.add_parser('sfm-court-white-line-probe',help='Fit native-image court white-line junctions and audit fixed visual-model holdouts')
    court_white_line.add_argument('--work',required=True)
    court_white_line.add_argument('--stage',choices=('geometry','stability','coverage','model-review','pair-mask-review','all'),default='geometry')
    r_static_depth=sub.add_parser('sfm-r-static-depth-probe',help='Freeze reviewed near/far R-frame LoMa cycles and audit saved visual poses')
    r_static_depth.add_argument('--work',required=True)
    r_static_depth.add_argument('--stage',choices=('select','audit','all'),default='all')
    r277_static=sub.add_parser('sfm-r277-static-probe',help='Check image-reviewed fixed structures with R277 held out')
    r277_static.add_argument('--work',required=True)
    r277_static.add_argument('--stage',choices=('select','audit','all'),default='all')
    l150_depth=sub.add_parser('sfm-l150-depth-probe',help='Audit L150 court LoMa coverage and frozen near/far image cycles')
    l150_depth.add_argument('--work',required=True)
    l150_depth.add_argument('--stage',choices=('select','audit','all'),default='all')
    l150_crop=sub.add_parser('sfm-l150-crop-loma-probe',help='Sample LoMa on five native court crops around a white paint patch')
    l150_crop.add_argument('--work',required=True)
    l150_crop_audit=sub.add_parser('sfm-l150-crop-audit',help='Freeze reviewed crop LoMa tracks and check saved visual poses')
    l150_crop_audit.add_argument('--work',required=True)
    l150_crop_audit.add_argument('--stage',choices=('select','audit','all'),default='all')
    l154_static=sub.add_parser('sfm-l154-static-control-probe',help='Check frozen fixed-structure tracks with L154 held out')
    l154_static.add_argument('--work',required=True)
    l154_static.add_argument('--stage',choices=('select','audit','compare','all'),default='all')
    l154_corner=sub.add_parser('sfm-l154-right-corner-probe',help='Freeze and audit the right court white-line corner')
    l154_corner.add_argument('--work',required=True)
    l154_corner.add_argument('--stage',choices=('select','audit','sensitivity','pose','mapped-impact','all'),default='all')
    lfs_ab=sub.add_parser('lfs-export-ab',help='Export same-image, same-mask, same-point COLMAP inputs for LFS full-train A/B')
    lfs_ab.add_argument('--work',required=True)
    lfs_ab.add_argument('--lfs-template',default='/home/wuyou/LichtFeld-Studio/configs/篮球场.json',
                        help='Complete LFS config template; navigation/LiDAR paths are removed')
    lfs_analysis=sub.add_parser('lfs-analyze-ab',help='Compare matched LFS training-view renders at saved evaluation steps')
    lfs_analysis.add_argument('--work',required=True)
    args=parser.parse_args()
    if args.command=='prepare':
        scene=prepare(args.input,args.work,args.fov,args.principal_y,args.mask_root)
        print(json.dumps({'images':len(scene['images']),'pose_model':scene['pose_model'],
                          'excluded':scene['excluded'],'time_audit':scene['navigation_time_audit'],
                          'manifest':args.work+'/manifest.json'},ensure_ascii=False,indent=2))
    elif args.command=='global-review':
        from .global_residual_review import build_global_residual_review
        build_global_residual_review(args.work,args.candidate)
    elif args.command=='static-review':
        from .static_validation import prepare_static_review
        prepare_static_review(args.work)
    elif args.command=='static-evaluate':
        from .static_validation import evaluate_static_review
        evaluate_static_review(args.work,args.annotations)
    elif args.command=='ground-pose-probe':
        from .ground_pose_probe import run_ground_pose_probe
        run_ground_pose_probe(args.work)
    elif args.command=='ground-pose-analyze':
        from .ground_pose_probe import analyze_ground_pose_probe
        analyze_ground_pose_probe(args.work)
    elif args.command=='ground-joint-ba':
        from .ground_joint_ba import run_ground_joint_ba
        run_ground_joint_ba(args.work)
    elif args.command=='ground-joint-analyze':
        from .ground_joint_ba import analyze_ground_joint_ba
        analyze_ground_joint_ba(args.work)
    elif args.command=='ground-photo-proxy':
        from .ground_photo_proxy import run_ground_photo_proxy
        run_ground_photo_proxy(args.work)
    elif args.command=='sfm-track-cleanup':
        from .sfm_track_cleanup import run_sfm_track_cleanup
        result=run_sfm_track_cleanup(args.work,args.stage)
        if args.stage=='audit':
            audit=result['filter_audit']
            summary={'output':result['output'],
                     'points':audit['after']['points'],
                     'observations':audit['after']['observations'],
                     'coverage_accepted':audit['coverage_accepted']}
        elif args.stage=='solve':
            summary={'output':result['output'],'ba':result['ba']}
        else:
            summary={'output':str(args.work)+'/global_sfm/sfm_track_cleanup/comparison.json',
                     'ba':result['ba']['termination_type'],
                     'manual_paired':result['manual_paired'],
                     'triangle_common_support_queries':result['triangle_common_support']['queries']}
        print(json.dumps(summary,ensure_ascii=False,indent=2))
    elif args.command=='sfm-match-evidence':
        from .sfm_match_evidence import run_match_evidence_audit
        result=run_match_evidence_audit(args.work)
        print(json.dumps({'output':str(args.work)+'/global_sfm/sfm_match_evidence/results.json',
                          'labels':result['summary']['labels'],
                          'auc':{name:value['auc_static_vs_wrong_or_dynamic']
                                 for name,value in result['summary']['metrics'].items()}},
                         ensure_ascii=False,indent=2))
    elif args.command=='sfm-residual-structure':
        from .sfm_residual_structure import run_residual_structure_audit
        result=run_residual_structure_audit(args.work)
        print(json.dumps({'output':str(args.work)+'/global_sfm/sfm_residual_structure/results.json',
                          'manual_cases':result['per_image_pseudo_correction']['manual_cases_with_other_tracks'],
                          'pseudo_correction':result['per_image_pseudo_correction']},
                         ensure_ascii=False,indent=2))
    elif args.command=='sfm-lk-probe':
        from .sfm_lk_probe import run_lk_probe
        result=run_lk_probe(args.work)
        print(json.dumps({'output':str(args.work)+'/global_sfm/sfm_lk_probe/results.json',
                          'summary':result['summary']},ensure_ascii=False,indent=2))
    elif args.command=='sfm-lk-refinement':
        from .sfm_lk_refinement import run_lk_refinement
        result=run_lk_refinement(args.work,args.stage)
        if args.stage=='audit':
            print(json.dumps({'output':result['output'],
                              'mapped_observations':result['audit']['total_mapped_observations'],
                              'updated_observations':result['audit']['updated_observations'],
                              'updated_images':result['audit']['updated_images']},
                             ensure_ascii=False,indent=2))
        else:
            print(json.dumps({'output':str(args.work)+'/sfm_lk_refinement/normal_070_094/comparison.json',
                              'updated_observations':result['audit']['updated_observations'],
                              'triangle_common_support_queries':result['triangle_common_support']['queries'],
                              'triangle_stages':{name:value['triangle']['all_candidate_queries']
                                                 for name,value in result['stages'].items()}},
                             ensure_ascii=False,indent=2))
    elif args.command=='sfm-resolution-probe':
        from .sfm_resolution_probe import run_resolution_probe
        result=run_resolution_probe(args.work)
        print(json.dumps({'output':str(args.work)+'/global_sfm/sfm_resolution_probe/results.json',
                          'pairs':len(result['rows']),
                          'cache_stats':result['cache_stats']},ensure_ascii=False,indent=2))
    elif args.command=='sfm-virtual-view-probe':
        from .sfm_virtual_view_probe import run_virtual_view_probe
        result=run_virtual_view_probe(args.work)
        print(json.dumps({'output':str(args.work)+'/global_sfm/sfm_virtual_view_probe/results.json',
                          'pairs':len(result['rows']),
                          'cache_stats':result['cache_stats']},ensure_ascii=False,indent=2))
    elif args.command=='sfm-virtual-view-audit':
        from .sfm_virtual_view_probe import run_virtual_view_audit
        result=run_virtual_view_audit(args.work)
        print(json.dumps({'output':str(args.work)+'/global_sfm/sfm_virtual_view_probe/fixed_model_audit.json',
                          'pairs':len(result['rows'])},ensure_ascii=False,indent=2))
    elif args.command=='sfm-virtual-cycle-probe':
        from .sfm_virtual_cycle_probe import run_virtual_cycle_probe
        result=run_virtual_cycle_probe(args.work)
        print(json.dumps({'output':str(args.work)+'/global_sfm/sfm_virtual_cycle_probe/results.json',
                          'triplets':len(result['rows']),
                          'cache_stats':result['cache_stats']},ensure_ascii=False,indent=2))
    elif args.command=='sfm-court-region-probe':
        from .sfm_hybrid_court_probe import run_court_region_probe
        result=run_court_region_probe(args.work)
        print(json.dumps({'output':str(args.work)+'/global_sfm/sfm_hybrid_court_probe/court_review.json',
                          'images':len(result['rows'])},ensure_ascii=False,indent=2))
    elif args.command=='sfm-bottom-mask-probe':
        from .sfm_bottom_mask_probe import run_bottom_mask_probe
        result=run_bottom_mask_probe(args.work)
        print(json.dumps({'output':str(args.work)+'/global_sfm/sfm_bottom_mask_probe/audit.json',
                          'images':len(result['rows']),
                          'restored_pixels':sum(row['restored_court_color_pixels'] for row in result['rows'])},
                         ensure_ascii=False,indent=2))
    elif args.command=='sfm-fov-audit':
        from .sfm_bottom_mask_probe import run_fov_audit
        result=run_fov_audit(args.work)
        print(json.dumps({'output':str(args.work)+'/global_sfm/sfm_bottom_mask_probe/fov_audit.json',
                          'images':len(result['rows'])},ensure_ascii=False,indent=2))
    elif args.command=='sfm-fov179-masks':
        from .sfm_bottom_mask_probe import run_fov_candidate_masks
        result=run_fov_candidate_masks(args.work)
        print(json.dumps({'output':str(args.work)+'/global_sfm/sfm_fov179_court_candidate/mask_audit.json',
                          'images':len(result['rows'])},ensure_ascii=False,indent=2))
    elif args.command=='sfm-fov179-frontend-probe':
        from .sfm_fov_frontend_probe import run_fov_frontend_probe
        result=run_fov_frontend_probe(args.work)
        print(json.dumps({'output':str(args.work)+'/global_sfm/sfm_fov179_court_frontend_probe_v2/results.json',
                          'triplets':len(result['rows']),
                          'cache_stats':result['cache_stats']},ensure_ascii=False,indent=2))
    elif args.command=='sfm-court-flow-probe':
        from .sfm_court_flow_probe import OUTPUT, run_court_flow_probe
        result=run_court_flow_probe(args.work)
        print(json.dumps({'output':str(args.work)+'/global_sfm/'+OUTPUT+'/results.json',
                          'triplets':len(result['rows']),
                          'image_closed':sum(row['court_candidate_in_all_three'] for row in result['rows'])},
                         ensure_ascii=False,indent=2))
    elif args.command=='sfm-court-structural-probe':
        from .sfm_court_flow_probe import STRUCTURAL_OUTPUT, run_court_flow_probe
        result=run_court_flow_probe(args.work, structural=True)
        print(json.dumps({'output':str(args.work)+'/global_sfm/'+STRUCTURAL_OUTPUT+'/results.json',
                          'triplets':len(result['rows']),
                          'image_closed':sum(row['court_candidate_in_all_three'] for row in result['rows'])},
                         ensure_ascii=False,indent=2))
    elif args.command=='sfm-court-structure-sift-probe':
        from .sfm_court_structure_probe import OUTPUT, run_court_structure_probe
        result=run_court_structure_probe(args.work)
        print(json.dumps({'output':str(args.work)+'/global_sfm/'+OUTPUT+'/results.json',
                          'triplets':len(result['rows']),
                          'descriptor_closed':sum(row['three_way_descriptor_closed']
                                                  for row in result['rows'])},
                         ensure_ascii=False,indent=2))
    elif args.command=='sfm-court-down-loma-probe':
        from .sfm_court_down_loma_probe import OUTPUT, run_court_down_loma_probe
        result=run_court_down_loma_probe(args.work, stage=args.stage)
        print(json.dumps({'output':str(args.work)+'/global_sfm/'+OUTPUT+
                          ('/views.json' if args.stage=='prepare' else '/results.json'),
                          'stage':args.stage,
                          'views':len(result['views']) if args.stage=='prepare' else 6,
                          'triplets':len(result['rows']) if args.stage!='prepare' else 0},
                         ensure_ascii=False,indent=2))
    elif args.command=='sfm-court-down-loma-audit':
        from .sfm_court_down_loma_probe import OUTPUT, run_court_down_loma_audit
        result=run_court_down_loma_audit(args.work)
        print(json.dumps({'output':str(args.work)+'/global_sfm/'+OUTPUT+'/lk_audit.json',
                          'triplets':len(result['rows'])},ensure_ascii=False,indent=2))
    elif args.command=='sfm-court-down-loma-review':
        from .sfm_court_down_loma_probe import OUTPUT, run_court_down_loma_review
        result=run_court_down_loma_review(args.work)
        print(json.dumps({'output':str(args.work)+'/global_sfm/'+OUTPUT+'/near_review.json',
                          'reviewed_tracks':sum(row['tracks'] for row in result['rows'])},
                         ensure_ascii=False,indent=2))
    elif args.command=='sfm-court-plane-probe':
        from .sfm_court_plane_probe import OUTPUT, run_court_plane_probe
        result=run_court_plane_probe(args.work)
        print(json.dumps({'output':str(args.work)+'/global_sfm/'+OUTPUT+'/results.json',
                          'triplets':len(result['rows'])},ensure_ascii=False,indent=2))
    elif args.command=='sfm-court-plane-pose-audit':
        from .sfm_court_plane_probe import OUTPUT, run_court_plane_pose_audit
        result=run_court_plane_pose_audit(args.work)
        print(json.dumps({'output':str(args.work)+'/global_sfm/'+OUTPUT+'/pose_audit.json',
                          'triplets':len(result['rows'])},ensure_ascii=False,indent=2))
    elif args.command=='sfm-court-plane-common-pose-audit':
        from .sfm_court_plane_probe import OUTPUT, run_court_plane_common_pose_audit
        result=run_court_plane_common_pose_audit(args.work)
        print(json.dumps({'output':str(args.work)+'/global_sfm/'+OUTPUT+'/common_pose_audit.json',
                          'triplets':len(result['rows'])},ensure_ascii=False,indent=2))
    elif args.command=='sfm-court-trajectory-probe':
        from .sfm_court_trajectory_probe import (OUTPUT, run_court_trajectory_scan,
                                                 run_court_trajectory_pairs,
                                                 run_court_trajectory_corroboration,
                                                 run_court_trajectory_graph_cycles,
                                                 run_court_trajectory_review)
        result=(run_court_trajectory_scan(args.work) if args.stage=='scan'
                else run_court_trajectory_pairs(args.work) if args.stage=='pairs'
                else run_court_trajectory_corroboration(args.work)
                if args.stage=='corroborate' else run_court_trajectory_graph_cycles(args.work)
                if args.stage=='cycles' else run_court_trajectory_review(args.work))
        print(json.dumps({'output':str(args.work)+'/global_sfm/'+OUTPUT+
                          ('/scan.json' if args.stage=='scan' else '/pairs.json'
                           if args.stage=='pairs' else '/corroboration.json'
                           if args.stage=='corroborate' else '/graph_cycles.json'
                           if args.stage=='cycles' else '/review.json'),
                          'selected_triplets':len(result['groups']) if args.stage=='scan'
                          else len(result['triplet_cycles']) if args.stage=='pairs' else 0,
                          'pairs':0 if args.stage=='scan' else result['candidate_pair_count']
                          if args.stage=='pairs' else len(result['triangles'])
                          if args.stage=='cycles' else len(result['rows'])},
                         ensure_ascii=False,indent=2))
    elif args.command=='sfm-court-dense-probe':
        from .sfm_court_dense_probe import (OUTPUT, run_court_dense_probe,
                                            run_court_dense_review)
        result=(run_court_dense_probe(args.work) if args.stage=='probe'
                else run_court_dense_review(args.work))
        print(json.dumps({'output':str(args.work)+'/global_sfm/'+OUTPUT+
                          ('/results.json' if args.stage=='probe' else '/review.json'),
                          'pairs':len(result['pairs']) if args.stage=='probe'
                          else len(result['rows']),
                          'converged':sum(row['status']=='converged' for row in result['pairs'])
                          if args.stage=='probe' else None,
                          'graph_cycles':len(result['graph_cycles']) if args.stage=='probe'
                          else None},
                         ensure_ascii=False,indent=2))
    elif args.command=='sfm-court-anchor-probe':
        from .sfm_court_anchor_probe import OUTPUT, run_court_anchor_probe
        result=run_court_anchor_probe(args.work)
        print(json.dumps({'output':str(args.work)+'/global_sfm/'+OUTPUT+'/results.json',
                          'images_with_junction':result['images_with_junction'],
                          'associations':len(result['pair_associations']),
                          'three_view_groups':len(result['three_view_groups'])},
                         ensure_ascii=False,indent=2))
    elif args.command=='sfm-court-anchor-coverage':
        from .sfm_court_anchor_probe import OUTPUT, run_court_anchor_coverage
        result=run_court_anchor_coverage(args.work)
        print(json.dumps({'output':str(args.work)+'/global_sfm/'+OUTPUT+'/coverage_v2.json',
                          'three_view_groups':len(result['groups'])},
                         ensure_ascii=False,indent=2))
    elif args.command=='sfm-court-native-anchor-probe':
        from .sfm_court_native_anchor_probe import OUTPUT, run_court_native_anchor_probe
        result=run_court_native_anchor_probe(args.work)
        print(json.dumps({'output':str(args.work)+'/global_sfm/'+OUTPUT+'/results.json',
                          'frames':len(result['frames']),
                          'tracks':list(result['tracks'])},ensure_ascii=False,indent=2))
    elif args.command=='sfm-court-native-anchor-eligibility':
        from .sfm_court_native_anchor_probe import OUTPUT, run_court_native_anchor_eligibility
        result=run_court_native_anchor_eligibility(args.work)
        print(json.dumps({'output':str(args.work)+'/global_sfm/'+OUTPUT+'/eligibility.json',
                          'frames':len(result['rows'])},ensure_ascii=False,indent=2))
    elif args.command=='sfm-court-paint-pair-probe':
        from .sfm_court_paint_pair_probe import (OUTPUT, run_court_paint_pair_scan,
                                                 run_court_paint_pair_audit,
                                                 run_court_paint_pair_review,
                                                 run_court_paint_pair_stability)
        result=(run_court_paint_pair_scan(args.work) if args.stage=='scan'
                else run_court_paint_pair_audit(args.work) if args.stage=='audit'
                else run_court_paint_pair_review(args.work) if args.stage=='review'
                else run_court_paint_pair_stability(args.work))
        print(json.dumps({'output':str(args.work)+'/global_sfm/'+OUTPUT+'/'+
                          args.stage+'.json',
                          'runs':len(result['runs']) if args.stage=='scan'
                          else len(result['audited_runs']) if args.stage=='audit'
                          else sum(row['segments'] for row in
                                   result['contact_sheets'].values())
                          if args.stage=='review' else len(result['segments'])},
                         ensure_ascii=False,indent=2))
    elif args.command=='sfm-court-white-line-probe':
        from .sfm_court_white_line_probe import (OUTPUT, run_court_white_line_probe,
                                                 run_court_white_line_stability,
                                                 run_court_white_line_coverage,
                                                 run_court_white_line_model_review,
                                                 run_court_white_line_pair_mask_review)
        if args.stage in ('geometry','all'):
            result=run_court_white_line_probe(args.work)
        if args.stage in ('stability','all'):
            result=run_court_white_line_stability(args.work)
        if args.stage in ('coverage','all'):
            result=run_court_white_line_coverage(args.work)
        if args.stage in ('model-review','all'):
            result=run_court_white_line_model_review(args.work)
        if args.stage in ('pair-mask-review','all'):
            result=run_court_white_line_pair_mask_review(args.work)
        output_name=({'geometry':'results.json','stability':'stability.json',
                      'coverage':'coverage.json','model-review':'model_review.json',
                      'pair-mask-review':'pair_mask_review.json',
                      'all':'pair_mask_review.json'}[args.stage])
        summary={'output':str(args.work)+'/global_sfm/'+OUTPUT+'/'+output_name}
        if 'segments' in result:
            summary['segments']=len(result['segments'])
        if 'pages' in result:
            summary['review_pages']=sum(len(value) for value in result['pages'].values())
        print(json.dumps(summary,ensure_ascii=False,indent=2))
    elif args.command=='sfm-r-static-depth-probe':
        from .sfm_r_static_depth_probe import (OUTPUT, run_r_static_depth_selection,
                                               run_r_static_depth_audit)
        if args.stage in ('select','all'):
            result=run_r_static_depth_selection(args.work)
        if args.stage in ('audit','all'):
            result=run_r_static_depth_audit(args.work)
        output_name='selection.json' if args.stage=='select' else 'audit.json'
        print(json.dumps({'output':str(args.work)+'/global_sfm/'+OUTPUT+'/'+output_name,
                          'selected':len(result['selected']) if args.stage=='select'
                          else len(result['rows'])},ensure_ascii=False,indent=2))
    elif args.command=='sfm-r277-static-probe':
        from .sfm_r_static_depth_probe import (R277_OUTPUT, run_r277_static_selection,
                                               run_r277_static_audit)
        if args.stage in ('select','all'):
            result=run_r277_static_selection(args.work)
        if args.stage in ('audit','all'):
            result=run_r277_static_audit(args.work)
        output_name='selection.json' if args.stage=='select' else 'audit.json'
        print(json.dumps({'output':str(args.work)+'/global_sfm/'+R277_OUTPUT+'/'+output_name,
                          'selected':len(result['selected']) if args.stage=='select'
                          else len(result['rows'])},ensure_ascii=False,indent=2))
    elif args.command=='sfm-l150-depth-probe':
        from .sfm_l150_depth_probe import OUTPUT, run_l150_selection, run_l150_audit
        if args.stage in ('select','all'):
            result=run_l150_selection(args.work)
        if args.stage in ('audit','all'):
            result=run_l150_audit(args.work)
        output_name='selection.json' if args.stage=='select' else 'audit.json'
        print(json.dumps({'output':str(args.work)+'/global_sfm/'+OUTPUT+'/'+output_name,
                          'selected':len(result['selected']) if args.stage=='select'
                          else len(result['rows'])},ensure_ascii=False,indent=2))
    elif args.command=='sfm-l150-crop-loma-probe':
        from .sfm_l150_crop_loma_probe import OUTPUT, run_l150_crop_loma_probe
        result=run_l150_crop_loma_probe(args.work)
        print(json.dumps({'output':str(args.work)+'/global_sfm/'+OUTPUT+'/results.json',
                          'five_view':result['five_view']},ensure_ascii=False,indent=2))
    elif args.command=='sfm-l150-crop-audit':
        from .sfm_l150_crop_loma_probe import (OUTPUT, run_l150_crop_selection,
                                               run_l150_crop_audit)
        if args.stage in ('select','all'):
            result=run_l150_crop_selection(args.work)
        if args.stage in ('audit','all'):
            result=run_l150_crop_audit(args.work)
        output_name='selection.json' if args.stage=='select' else 'audit.json'
        print(json.dumps({'output':str(args.work)+'/global_sfm/'+OUTPUT+'/'+output_name,
                          'selected':len(result['selected']) if args.stage=='select'
                          else len(result['rows'])},ensure_ascii=False,indent=2))
    elif args.command=='sfm-l154-static-control-probe':
        from .sfm_l154_static_control_probe import (OUTPUT, run_l154_static_selection,
                                                   run_l154_static_audit,
                                                   run_l154_patch_comparison)
        if args.stage in ('select','all'):
            result=run_l154_static_selection(args.work)
        if args.stage in ('audit','all'):
            result=run_l154_static_audit(args.work)
        if args.stage in ('compare','all'):
            result=run_l154_patch_comparison(args.work)
        output_name={'select':'selection.json','audit':'audit.json',
                     'compare':'comparison.json','all':'comparison.json'}[args.stage]
        print(json.dumps({'output':str(args.work)+'/global_sfm/'+OUTPUT+'/'+output_name,
                          'selected':len(result['selected']) if args.stage=='select'
                          else len(result['rows']) if args.stage=='audit'
                          else len(result['ground_rows'])},ensure_ascii=False,indent=2))
    elif args.command=='sfm-l154-right-corner-probe':
        from .sfm_l154_right_corner_probe import (OUTPUT, run_right_corner_selection,
                                                 run_right_corner_audit,
                                                 run_right_corner_sensitivity,
                                                 run_right_corner_pose_hypotheses,
                                                 run_right_corner_mapped_impact)
        if args.stage in ('select','all'):
            result=run_right_corner_selection(args.work)
        if args.stage in ('audit','all'):
            result=run_right_corner_audit(args.work)
        if args.stage in ('sensitivity','all'):
            result=run_right_corner_sensitivity(args.work)
        if args.stage in ('pose','all'):
            result=run_right_corner_pose_hypotheses(args.work)
        if args.stage in ('mapped-impact','all'):
            result=run_right_corner_mapped_impact(args.work)
        output_name={'select':'selection.json','audit':'audit.json',
                     'sensitivity':'sensitivity.json',
                     'pose':'pose_hypotheses.json',
                     'mapped-impact':'mapped_impact.json','all':'mapped_impact.json'}[args.stage]
        print(json.dumps({'output':str(args.work)+'/global_sfm/'+OUTPUT+'/'+output_name,
                          'selected':len(result['selected']) if args.stage=='select'
                          else len(result['predictions']) if args.stage=='audit'
                          else len(result['combinations']) if args.stage=='sensitivity'
                          else len(result['hypotheses']) if args.stage=='pose'
                          else len(result['evaluations'])},ensure_ascii=False,indent=2))
    elif args.command=='lfs-export-ab':
        from .lfs_ab import prepare_lfs_ab
        print(json.dumps(prepare_lfs_ab(args.work,args.lfs_template),ensure_ascii=False,indent=2))
    elif args.command=='lfs-analyze-ab':
        from .lfs_ab_analysis import analyze_lfs_ab
        print(json.dumps(analyze_lfs_ab(args.work),ensure_ascii=False,indent=2))
    elif args.command=='photometric-review':
        from .photometric_review import inspect_photometric_pairs
        inspect_photometric_pairs(args.work)
    elif args.command=='trajectory-review':
        from .trajectory_review import build_trajectory_review
        build_trajectory_review(args.work,args.stage)
    elif args.command=='global-sfm':
        from .global_reconstruction import (build_global_mapping, evaluate_global_mapping,
                                            solve_global, compare_global_stages)
        if args.stage in ('mapping','all'):build_global_mapping(args.work)
        if args.stage in ('mapping-eval','all'):evaluate_global_mapping(args.work)
        if args.stage in ('solve','all'):solve_global(args.work,stop_after=args.stop_after)
        if args.stage == 'compare':compare_global_stages(args.work)
    elif args.command=='global-data':
        from .global_data import prepare_global, extract_global_features, match_global_features
        if args.stage in ('masks','all'):prepare_global(args.work)
        if args.stage in ('features','all'):extract_global_features(args.work)
        if args.stage in ('matches','all'):match_global_features(args.work)
    elif args.command=='track-probe':
        from .track_consistency import run_track_probe
        run_track_probe(args.work)
    elif args.command=='principal-ablation':
        from .principal_ablation import run_principal_ablation
        run_principal_ablation(args.work)
    elif args.command=='refinement-probe':
        from .refinement import run_refinement_probe
        run_refinement_probe(args.work)
    elif args.command=='review-residuals':
        from .residual_review import build_residual_review
        build_residual_review(args.work)
    elif args.command=='transfer-check':
        from .transfer import prepare_transfer, transfer_frontend, solve_transfer
        if args.stage in ('masks','all'):prepare_transfer(args.work)
        if args.stage in ('frontend','all'):transfer_frontend(args.work)
        if args.stage in ('solve','all'):solve_transfer(args.work,continue_iteration_limit=args.continue_iteration_limit)
    elif args.command=='local-sfm':
        from .local_data import prepare_masks, build_frontend
        if args.stage in ('masks','all'):
            prepare_masks(args.work)
        if args.stage in ('frontend','all'):
            build_frontend(args.work)
        if args.stage in ('solve','all'):
            from .local_reconstruction import run_local_reconstruction
            run_local_reconstruction(args.work)
    else:
        from .diagnostic import diagnose,DEFAULT_FRAMES
        from .frontend import FrontendConfig
        if args.detector_size < 256 or args.keypoints < 32 or args.max_error <= 0 or not 0 <= args.match_threshold <= 1:
            parser.error('Invalid extraction or geometry parameter')
        frames=[int(v) for v in args.frames.split(',')] if args.frames else DEFAULT_FRAMES
        if not frames:
            parser.error('At least one frame index is required')
        config=FrontendConfig(detector_size=args.detector_size,num_keypoints=args.keypoints,match_threshold=args.match_threshold)
        out,report=diagnose(args.work,config,frames,args.max_error,args.seed)
        print(json.dumps({'output':str(out),'summary':report['summary'],'tracks':report['tracks']},ensure_ascii=False,indent=2))
