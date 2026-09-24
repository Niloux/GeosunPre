"""Three-view consistency check for native and virtual-view LoMa features.

The saved baseline visual model only supplies a fixed coordinate system for
evaluation. No candidate features, matches, or poses enter production SfM.
"""

from datetime import datetime, timezone
import json
from pathlib import Path

import cv2
import numpy as np
import pycolmap
from PIL import Image, ImageDraw

from ..camera import Camera
from ..frontend import LoMaFrontend
from ..geometry import stats
from ..scene import digest, file_digest, load_scene, write_json
from .sfm_resolution_probe import PAIR_FRAMES, _cached_feature, _cached_match
from .sfm_virtual_view_probe import CONFIG, OUTPUT, _virtual_feature


OUT_NAME = 'sfm_virtual_cycle_probe'
HIGH_CONFIDENCE = .9
SUPPORT_MAX_PX = 3.
QUERY_MAX_PX = 3.
MIN_ANGLE_DEG = 1.


def _pair_entry(catalog, a, b):
    selected = [entry for entry in catalog['pairs'] if entry['a'] == a and entry['b'] == b]
    if len(selected) != 1:
        raise ValueError(f'Expected one baseline match pair: {a} -> {b}')
    return selected[0]


def _closed_cycles(matches):
    ab, ac, bc = matches
    def high_map(match):
        take = match['scores'] >= HIGH_CONFIDENCE
        return {int(i): int(j) for i, j in match['matches'][take]}
    ab_map, ac_map, bc_map = [high_map(match) for match in (ab, ac, bc)]
    rows = [(ia, ib, ac_map[ia]) for ia, ib in ab_map.items()
            if ia in ac_map and bc_map.get(ib) == ac_map[ia]]
    return np.asarray(rows, dtype=np.int64).reshape(-1, 3)


def _three_view_geometry(names, features, cycles, images_by_name, camera,
                         region_mask=None, return_masks=False):
    if not len(cycles):
        result = {'high_confidence_closed_cycles': 0, 'support_ok': 0,
                'three_view_ok': 0, 'lower_half_three_view_ok': 0,
                'query_error_px_on_support_ok': stats([]),
                'fraction_query_le_3px_on_support_ok': None}
        if region_mask is not None:
            result.update(region_closed_cycles=0, region_support_ok=0, region_three_view_ok=0)
        if return_masks:
            empty = np.zeros(0, dtype=bool)
            return result, {'support_ok': empty, 'three_view_ok': empty,
                            'query_error_px': np.empty(0), 'positive_depth': empty,
                            'angle_deg': np.empty(0),
                            'support_error_px': np.empty(0)}
        return result
    a, b, c = names
    xy_a, xy_b, xy_c = [features[name]['keypoints'][cycles[:, idx]].astype(float)
                         for idx, name in enumerate(names)]
    pa, pb, pc = [images_by_name[name].cam_from_world().matrix() for name in names]
    ra, rc = camera.rays(xy_a), camera.rays(xy_c)
    hom = cv2.triangulatePoints(pa, pc, (ra[:, :2] / ra[:, 2:]).T,
                                (rc[:, :2] / rc[:, 2:]).T).T
    valid_h = np.abs(hom[:, 3]) > 1e-12
    xyz = np.divide(hom[:, :3], hom[:, 3:], out=np.full((len(hom), 3), np.nan),
                    where=valid_h[:, None])
    finite = valid_h & np.isfinite(xyz).all(axis=1)
    local_a, local_b, local_c = [xyz @ pose[:, :3].T + pose[:, 3]
                                 for pose in (pa, pb, pc)]
    positive = (finite & (local_a[:, 2] > 0) & (local_b[:, 2] > 0)
                & (local_c[:, 2] > 0))
    ea, eb, ec = [np.full(len(cycles), np.inf) for _ in range(3)]
    if positive.any():
        for error, local, pixels in ((ea, local_a, xy_a), (eb, local_b, xy_b),
                                     (ec, local_c, xy_c)):
            error[positive] = np.linalg.norm(camera.project(local[positive])
                                              - pixels[positive], axis=1)
    world_a, world_c = ra @ pa[:, :3], rc @ pc[:, :3]
    angle = np.rad2deg(np.arccos(np.clip(np.sum(world_a * world_c, axis=1), -1, 1)))
    support_ok = positive & (angle >= MIN_ANGLE_DEG) & (np.maximum(ea, ec) <= SUPPORT_MAX_PX)
    three_view_ok = support_ok & (eb <= QUERY_MAX_PX)
    lower = xy_a[:, 1] >= camera.height / 2
    result = {'high_confidence_closed_cycles': int(len(cycles)),
            'lower_half_closed_cycles': int(lower.sum()),
            'support_ok': int(support_ok.sum()),
            'lower_half_support_ok': int((support_ok & lower).sum()),
            'three_view_ok': int(three_view_ok.sum()),
            'lower_half_three_view_ok': int((three_view_ok & lower).sum()),
            'query_error_px_on_support_ok': stats(eb[support_ok]),
            'fraction_query_le_3px_on_support_ok': float(np.mean(eb[support_ok] <= QUERY_MAX_PX))
            if support_ok.any() else None}
    if region_mask is not None:
        ij = np.floor(xy_a / [camera.width, camera.height]
                      * [region_mask.shape[1], region_mask.shape[0]]).astype(int)
        ij[:, 0] = np.clip(ij[:, 0], 0, region_mask.shape[1] - 1)
        ij[:, 1] = np.clip(ij[:, 1], 0, region_mask.shape[0] - 1)
        inside = region_mask[ij[:, 1], ij[:, 0]]
        result.update(region_closed_cycles=int(inside.sum()),
                      region_support_ok=int((support_ok & inside).sum()),
                      region_three_view_ok=int((three_view_ok & inside).sum()))
    if return_masks:
        return result, {'support_ok': support_ok, 'three_view_ok': three_view_ok,
                        'query_error_px': eb, 'positive_depth': positive,
                        'angle_deg': angle, 'support_error_px': np.maximum(ea, ec)}
    return result


def _render_cycle_review(output, record, old_xy, virtual_xy, camera):
    """Show first-image locations of descriptor-closed tracks before 3D gating."""
    scale_x, scale_y = 750 / camera.width, 1000 / camera.height
    with Image.open(record['source']) as image:
        original = image.convert('RGB').resize((750, 1000), Image.Resampling.LANCZOS)
    canvas = Image.new('RGB', (1500, 1040), (24, 28, 35))
    draw = ImageDraw.Draw(canvas)
    for index, (label, points) in enumerate((('Native LoMa', old_xy),
                                             ('Virtual views', virtual_xy))):
        x_offset = 750 * index
        canvas.paste(original, (x_offset, 40))
        draw.text((x_offset + 10, 10), f'{label}: {len(points)} closed cycles',
                  fill=(245, 245, 245))
        midline = 40 + camera.height / 2 * scale_y
        draw.line((x_offset, midline, x_offset + 750, midline),
                  fill=(255, 255, 255), width=1)
        for x, y in points:
            cx, cy = x_offset + x * scale_x, 40 + y * scale_y
            color = (255, 195, 35) if y < camera.height / 2 else (35, 240, 210)
            draw.ellipse((cx - 1.5, cy - 1.5, cx + 1.5, cy + 1.5), fill=color)
    path = output / 'review' / f'{record["camera"]}{record["frame_index"]:05d}.jpg'
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=90)
    return str(path)


def run_virtual_cycle_probe(work):
    cv2.setNumThreads(4)
    work = Path(work).resolve()
    root = work / 'global_sfm'
    parent = root / OUTPUT
    parent_protocol = json.loads((parent / 'protocol.json').read_text())
    scene = load_scene(root)
    records = {record['name']: record for record in scene['images']}
    baseline = json.loads((root / 'frontend.json').read_text())
    if file_digest(root / 'frontend.json') != parent_protocol['baseline_frontend_sha256']:
        raise ValueError('Baseline frontend changed since pair probe')
    by_camera_frame = {(record['camera'], record['frame_index']): record['name']
                       for record in scene['images']}
    triplets = [tuple(by_camera_frame[(label, frame + offset)] for offset in range(3))
                for label, frame in PAIR_FRAMES]
    all_names = sorted({name for names in triplets for name in names})
    for name in all_names:
        if name in parent_protocol['native_image_sha256']:
            if (file_digest(records[name]['source']) != parent_protocol['native_image_sha256'][name]
                    or file_digest(records[name]['semantic_mask']) != parent_protocol['native_mask_sha256'][name]):
                raise ValueError(f'Virtual-view input changed since pair probe: {name}')
    entries = {names: tuple(_pair_entry(baseline, names[i], names[j])
                            for i, j in ((0, 1), (0, 2), (1, 2)))
               for names in triplets}
    source_model = root / 'ground_joint_ba/equal_control/sparse/0'
    model_hashes = {path.name: file_digest(path) for path in sorted(source_model.glob('*.bin'))}
    if model_hashes != parent_protocol['camera_model_hashes']:
        raise ValueError('Fixed visual model changed since pair probe')
    protocol = {'version': 1, 'parent_probe_fingerprint': parent_protocol['fingerprint'],
                'triplets': triplets,
                'image_sha256': {name: file_digest(records[name]['source']) for name in all_names},
                'mask_sha256': {name: file_digest(records[name]['semantic_mask']) for name in all_names},
                'baseline_match_tokens': {f'{a}|{b}': entry['token'] for triple in entries.values()
                                          for entry in triple for a, b in [(entry['a'], entry['b'])]},
                'fixed_model_hashes': model_hashes,
                'confidence_min': HIGH_CONFIDENCE,
                'max_support_reprojection_px': SUPPORT_MAX_PX,
                'max_query_reprojection_px': QUERY_MAX_PX,
                'min_support_angle_degrees': MIN_ANGLE_DEG,
                'uses_navigation_las_or_manual_labels': False,
                'scope': 'Four consecutive same-camera image triplets; fixed baseline visual poses are diagnostic only.'}
    fingerprint = digest(protocol)
    output = root / OUT_NAME
    output.mkdir(exist_ok=True)
    protocol_path = output / 'protocol.json'
    if protocol_path.exists():
        if json.loads(protocol_path.read_text())['fingerprint'] != fingerprint:
            raise ValueError('Existing cycle probe belongs to changed inputs')
    else:
        write_json(protocol_path, {**protocol, 'fingerprint': fingerprint,
                                   'frozen_at_utc': datetime.now(timezone.utc).isoformat()})
    rec = pycolmap.Reconstruction(source_model)
    images_by_name = {image.name: image for image in rec.images.values()}
    cameras = {label: Camera(label, rec.cameras[1 if label == 'L' else 2].width,
                             rec.cameras[1 if label == 'L' else 2].height,
                             tuple(rec.cameras[1 if label == 'L' else 2].params), (0, 0))
               for label in ('L', 'R')}
    frontend = LoMaFrontend(work, scene, CONFIG)
    old_features = {name: _cached_feature(baseline['features'][name]) for name in all_names}
    for name in all_names:
        meta = json.loads(str(old_features[name]['metadata']))
        if (meta['image_sha256'] != protocol['image_sha256'][name]
                or meta['mask_sha256'] != protocol['mask_sha256'][name]):
            raise ValueError(f'Baseline feature input changed since extraction: {name}')
    virtual_features = {}
    for index, name in enumerate(all_names, 1):
        virtual_features[name] = _virtual_feature(parent, records[name], cameras[name[0]],
                                                   scene, frontend, parent_protocol['fingerprint'])
        print(f'Virtual cycle features {index}/{len(all_names)} {name}', flush=True)
    rows = []
    for index, names in enumerate(triplets, 1):
        old_matches = [_cached_match(entry) for entry in entries[names]]
        virtual_matches = [frontend.match(virtual_features[entry['a']],
                                          virtual_features[entry['b']])
                           for entry in entries[names]]
        old_cycles = _closed_cycles(old_matches)
        virtual_cycles = _closed_cycles(virtual_matches)
        camera = cameras[names[0][0]]
        review = _render_cycle_review(output, records[names[0]],
                                      old_features[names[0]]['keypoints'][old_cycles[:, 0]],
                                      virtual_features[names[0]]['keypoints'][virtual_cycles[:, 0]],
                                      camera)
        rows.append({'camera': names[0][0], 'start_frame': records[names[0]]['frame_index'],
                     'images': names, 'review_image': review,
                     'baseline': _three_view_geometry(names, old_features, old_cycles,
                                                      images_by_name, camera),
                     'virtual_views': _three_view_geometry(names, virtual_features, virtual_cycles,
                                                           images_by_name, camera),
                     'virtual_match_tokens': [str(match['token']) for match in virtual_matches]})
        print(f'Virtual cycle comparison {index}/{len(triplets)}', flush=True)
    result = {'protocol_fingerprint': fingerprint, 'rows': rows,
              'cache_stats': frontend.cache_stats,
              'guard': 'Three-view closure rejects some pair ambiguity but can still preserve systematic errors. Baseline model poses were trained using baseline features, so this fixed-pose comparison favors baseline. Lower native image half is a screen proxy, not a ground semantic mask or 3DGS evaluation.'}
    write_json(output / 'results.json', result)
    return result
