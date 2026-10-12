"""One fixed-intrinsics image, pose and LAS refinement with guarded updates."""

import json
from pathlib import Path

import numpy as np
import pycolmap

from geosunpre.artifacts import file_metadata, staged_directory, write_json
from geosunpre.colmap import model_files
from geosunpre.registration.geometry import (quantiles, residuals,
    transform_points, validate_similarity)
from geosunpre.registration.global_alignment import MAX_LAS_POINTS, projection_invariance, validate_output
from geosunpre.registration.joint_solver import (RECIPE, IMAGE_GATES, ImageObservations,
    JointConstraintsUnavailable, assert_identity, geometry_gate, image_gate, interpolate,
    las_correspondences, mixed_objective, native_costs, solve_fixed)
from geosunpre.registration.las import sample_las
from geosunpre.registration.publication import complete_publication
from geosunpre.registration.structural import coherent_las
from geosunpre.scene import load_poses


def _positions(rec, audit):
    paths = [Path(path) for path in audit['input_files'] if path.endswith('_CameraPos_C2E.txt')]
    if len(paths) != 1:
        raise ValueError('Expected exactly one audited camera-position file')
    poses = load_poses(paths[0])
    names = {image.image_id: Path(image.name).name for image in rec.images.values()}
    if any(name not in poses for name in names.values()):
        raise ValueError('Joint BA requires a named input position for every registered image')
    matrix = validate_similarity(audit['navigation_to_candidate_las_matrix'])
    return {iid: transform_points(matrix, np.array([poses[name]['center_raw']]))[0]
            for iid, name in sorted(names.items())}


def _run_stage(label, base, initial_matrix, positions, surface, fit_ids, tiles_fit,
               observations, input_errors, recipe, directory, initial_matches):
    current, matrix = pycolmap.Reconstruction(base), initial_matrix.copy()
    steps = []
    directory.mkdir()
    for iteration in range(recipe['outer_iterations']):
        matches = initial_matches if iteration == 0 else las_correspondences(
            current, matrix, fit_ids, tiles_fit, surface,
            maximum=surface['correspondence_distance'], workers=recipe['workers'])
        errors_before, depth_before = observations.errors(current)
        objective_before = mixed_objective(current, matrix, errors_before, positions, matches, recipe)
        proposal = pycolmap.Reconstruction(current)
        print(f'Joint BA {label}: round {iteration + 1}, '
              f'{len(matches["point_ids"]) if matches is not None else 0} LAS constraints', flush=True)
        proposal_matrix, solver = solve_fixed(proposal, matrix, base, initial_matrix, positions, matches, recipe)
        print(f'Joint BA {label}: {solver["solver"]}', flush=True)
        trials, accepted = [], False
        for fraction in recipe['backtrack_fractions']:
            candidate, candidate_matrix = (proposal, proposal_matrix) if fraction == 1. else interpolate(
                current, proposal, matrix, proposal_matrix, fraction)
            candidate_errors, depths = observations.errors(candidate)
            visual = image_gate(observations, input_errors, candidate_errors)
            physical = geometry_gate(base, candidate, initial_matrix, candidate_matrix, recipe)
            depth_guard = bool(np.isfinite(depths).all() and np.all(depths >= .05 * depth_before))
            cost = mixed_objective(candidate, candidate_matrix, candidate_errors, positions, matches, recipe)
            descent = bool(np.isfinite(cost) and cost <= objective_before + 1e-6)
            passed = bool(solver['solution_usable'] and visual['passed'] and physical['passed'] and depth_guard and descent)
            trials.append({'fraction': fraction, 'passed': passed, 'image_gate': visual,
                'geometry_gate': physical, 'depth_guard_passed': depth_guard,
                'mixed_objective': cost if np.isfinite(cost) else None, 'descent': descent})
            if passed:
                current, matrix, accepted = candidate, candidate_matrix, True
                break
        step = {'round': iteration + 1, 'accepted_update': accepted, 'solver': solver,
                'correspondences': matches['summary'] if matches is not None else None,
                'mixed_objective_before': objective_before, 'backtracking': trials}
        if not accepted:
            rejected = directory / f'rejected_round_{iteration + 1}'
            rejected.mkdir(); proposal.write(rejected)
            write_json(rejected / 'global_transform.json', {'matrix': proposal_matrix.tolist()})
            step['rejected_proposal'] = rejected.name
        steps.append(step)
        write_json(directory / f'round_{iteration + 1}.json', step)
        if not accepted:
            break
    errors, depths = observations.errors(current)
    assert_identity(base, current)
    return current, matrix, errors, {'steps': steps, 'accepted_updates': sum(step['accepted_update'] for step in steps),
        'image_statistics': observations.summary(errors),
        'input_image_gate': image_gate(observations, input_errors, errors),
        'geometry_gate': geometry_gate(base, current, initial_matrix, matrix, recipe),
        'invalid_depth_observations': int((~np.isfinite(depths) | (depths <= 1e-9)).sum())}


def refine_joint(alignment, inputs, output, *, workers=4, context=None):
    """Compute one joint candidate and compare it to the unchanged source model."""
    alignment, output = Path(alignment).resolve(), Path(output).resolve()
    validate_output(output, alignment, inputs['input_root'], inputs['las'])
    native_costs()
    reference_files = {name: file_metadata(alignment / name)
                        for name in ('report.json', 'transform.json', 'registration_points.npz')}
    reference = json.loads((alignment / 'report.json').read_text(encoding='utf-8'))
    transform = json.loads((alignment / 'transform.json').read_text(encoding='utf-8'))
    if (reference['inputs_file'] != inputs['inputs_file']
            or transform['report_file'] != reference_files['report.json']):
        raise ValueError('Global registration identity mismatch')
    model_path = Path(reference['inputs']['model'])
    if model_files(model_path) != reference['inputs']['visual_model_files']:
        raise ValueError('Visual source model changed before joint refinement')
    base = pycolmap.Reconstruction(model_path)
    initial_matrix = validate_similarity(transform['matrix'])
    positions = _positions(base, reference['initialization'])
    centres = np.array(list(positions.values()))
    singular = np.linalg.svd(centres - centres.mean(0), compute_uv=False)
    if len(positions) < 3 or singular[1] < singular[0] * 1e-4:
        raise JointConstraintsUnavailable('Noncollinear position priors are required for joint refinement')
    with np.load(alignment / 'registration_points.npz', allow_pickle=False) as archive:
        points = {name: archive[name] for name in archive.files}
    ids, tiles = points['point_ids'], points['tiles']
    regions, scales = reference['regions'], reference['sampling_scales']
    repeated_tiles = np.floor((points['partition_xyz'] - np.asarray(regions['origin']))
                              / regions['tile_size']).astype(np.int64)
    source = np.array([base.points3D[int(pid)].xyz for pid in ids])
    initial_points = transform_points(initial_matrix, source)
    if (not np.array_equal(tiles, repeated_tiles)
            or len(np.unique(ids)) != len(ids)
            or not np.allclose(transform_points(transform['initial_matrix'], source), points['partition_xyz'], atol=1e-10, rtol=0)
            or not np.allclose(initial_points, points['final_xyz'], atol=1e-10, rtol=0)):
        raise ValueError('Global registration points or spatial partition do not replay')
    target, metadata = sample_las(inputs['las'], max_points=MAX_LAS_POINTS,
                                  seed=reference['las']['sample_seed'])
    if metadata != reference['las']:
        raise ValueError('LAS sample changed since global registration')
    surface = coherent_las(target, voxel_size=scales['voxel_sizes'][-1], workers=workers)
    surface['correspondence_distance'] = max(scales['correspondence_distances'][-1], scales['support_distance'])
    distances, plane, nearest = residuals(initial_points, surface, workers=workers)
    if (not np.allclose(distances, points['nearest_distance'], atol=1e-10, rtol=0)
            or not np.allclose(plane, points['signed_plane_error'], atol=1e-10, rtol=0)
            or not np.array_equal((distances <= scales['support_distance']) & surface['valid'][nearest], points['stable_surface_supported'])):
        raise ValueError('Global registration surface support does not replay')
    matches = las_correspondences(base, initial_matrix, ids, tiles, surface,
        maximum=surface['correspondence_distance'], workers=workers)
    summary = matches['summary']
    if summary['similarity_information_rank'] < 7:
        raise JointConstraintsUnavailable('Joint support cannot constrain all similarity directions', diagnostic=summary)
    if (summary['relative_plane_error_sigma']['median'] <= 1.
            and summary['relative_plane_error_sigma']['p90'] <= 3.):
        raise JointConstraintsUnavailable('Reliable surfaces already fit within relative noise proxies', diagnostic=summary)
    observations = ImageObservations(base, transform['initial_matrix'], regions=regions)
    input_errors, depths = observations.errors(base)
    if not np.isfinite(input_errors).all() or np.any(depths <= 1e-9):
        raise ValueError('Visual source has invalid projections')
    recipe = {**RECIPE, 'workers': workers}
    with staged_directory(output) as stage:
        rec, matrix, errors, diagnostic = _run_stage('pose_las', base, initial_matrix, positions, surface,
            ids, tiles, observations, input_errors, recipe, stage / 'solver', matches)
        diagnostic['accepted_for_delivery'] = bool(diagnostic['input_image_gate']['passed']
            and diagnostic['geometry_gate']['passed'] and diagnostic['accepted_updates'] > 0
            and diagnostic['invalid_depth_observations'] == 0)
        diagnostic['recipe'] = recipe
        diagnostic['position_residual_m'] = quantiles(np.linalg.norm(
            transform_points(matrix, np.array([rec.images[iid].projection_center() for iid in positions]))
            - np.array(list(positions.values())), axis=1))
        aligned = pycolmap.Reconstruction(rec)
        aligned.transform(pycolmap.Sim3d(matrix[:3]))
        model = stage / 'sparse/0'
        model.mkdir(parents=True)
        aligned.write(model)
        diagnostic['projection_invariance'] = projection_invariance(rec, pycolmap.Reconstruction(model))
        diagnostic['model_files'] = model_files(model)
        candidate_points = transform_points(matrix, np.array([rec.points3D[int(pid)].xyz for pid in ids]))
        distances, plane, nearest = residuals(candidate_points, surface, workers=workers)
        np.savez_compressed(stage / 'las_points.npz', point_ids=ids, tiles=tiles,
            initial_xyz=initial_points, final_xyz=candidate_points, nearest_distance=distances,
            signed_plane_error=plane, stable_surface_supported=surface['valid'][nearest] & (distances <= scales['support_distance']))
        np.savez_compressed(stage / 'image_observations.npz', point_ids=observations.point_ids,
            image_ids=observations.image_ids, point2D_indices=observations.feature_ids,
            observed_pixels=observations.pixels, input_error=input_errors, candidate_error=errors)
        write_json(stage / 'diagnostic.json', diagnostic)
        write_json(stage / 'transform.json', {'matrix': matrix.tolist(), 'recipe': recipe,
                                             'reference_files': reference_files, 'image_gates': IMAGE_GATES})
        if (model_files(model_path) != reference['inputs']['visual_model_files']
                or any(file_metadata(path) != expected for path, expected in inputs['input_files'].items())
                or any(file_metadata(alignment / name) != expected for name, expected in reference_files.items())):
            raise ValueError('Joint refinement sources changed during fitting')
        complete_publication(stage, context)
    return diagnostic
