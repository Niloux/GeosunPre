"""Read-only identity checks and single-branch packaging for a frozen SfM baseline."""

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path


MODEL_FILES = ('cameras.bin', 'images.bin', 'points3D.bin', 'rigs.bin', 'frames.bin')
DEFAULT_MANIFEST = Path('baselines/basketball/baseline.json')


def _digest(path):
    digest = sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_file(path, expected):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = _digest(path)
    if actual != expected:
        raise ValueError(f'Baseline identity changed: {path} ({actual} != {expected})')


def _work_path(work, relative):
    path = (work / relative).resolve()
    if not path.is_relative_to(work.resolve()):
        raise ValueError(f'Baseline path escapes work root: {relative}')
    return path


def _load_paths(repo_root, manifest_path, work_override):
    repo_root = Path(repo_root).resolve()
    manifest_path = Path(manifest_path)
    if not manifest_path.is_absolute():
        manifest_path = repo_root / manifest_path
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('schema_version') != 1 or manifest.get('status') != 'development_baseline':
        raise ValueError('Expected a version-1 frozen development baseline manifest')
    work = Path(work_override).resolve() if work_override else (repo_root / manifest['work_root']).resolve()
    if not work.is_dir():
        raise FileNotFoundError(work)
    paths = {key: _work_path(work, manifest[key]) for key in
             ('scene_manifest', 'source_model', 'lfs_dataset', 'lfs_shared',
              'lfs_protocol', 'lfs_config')}
    return manifest, manifest_path, work, paths


def check_baseline(repo_root='.', manifest_path=DEFAULT_MANIFEST, work_override=None,
                   *, full=False, _verify_release=True):
    """Verify the accepted development artifact without modifying any inputs."""
    from geosunpre.scene import digest, file_digest

    manifest, manifest_path, work, paths = _load_paths(repo_root, manifest_path, work_override)
    _verify_file(paths['scene_manifest'], manifest['scene_manifest_sha256'])
    _verify_file(paths['lfs_protocol'], manifest['lfs_protocol_sha256'])
    _verify_file(paths['lfs_config'], manifest['lfs_config_sha256'])
    protocol = json.loads(paths['lfs_protocol'].read_text())
    scene = json.loads(paths['scene_manifest'].read_text())
    if scene.get('fingerprint') != manifest['scene_fingerprint'] or protocol['scene_fingerprint'] != scene['fingerprint']:
        raise ValueError('Scene fingerprint differs from the frozen baseline')
    if (protocol['fingerprint'] != manifest['lfs_protocol_fingerprint']
            or protocol['models']['equal_control'] !=
            {name: manifest['model_sha256'][name] for name in ('cameras.bin', 'images.bin', 'points3D.bin')}
            or protocol['common_point_cloud_sha256'] != manifest['common_point_cloud_sha256']
            or protocol['originals_and_masks_sha256'] != manifest['originals_and_masks_sha256']
            or protocol['lfs_config_sha256'] != manifest['lfs_config_sha256']):
        raise ValueError('LFS protocol differs from the frozen baseline')
    if (protocol['images'] != manifest['image_count']
            or protocol['lenses'] != manifest['lenses']
            or protocol['all_selected_images_train'] is not True
            or protocol['test_split'] is not False
            or protocol['input_navigation_pose_values_or_las_used'] is not False
            or 'OPENCV_FISHEYE' not in protocol['camera_model']):
        raise ValueError('Camera, training split, or prior-use contract changed')

    for name in MODEL_FILES:
        expected = manifest['model_sha256'][name]
        _verify_file(paths['source_model'] / name, expected)
        _verify_file(paths['lfs_dataset'] / 'sparse' / '0' / name, expected)
    shared_point_cloud = paths['lfs_shared'] / 'points3D.ply'
    _verify_file(shared_point_cloud, manifest['common_point_cloud_sha256'])
    if (paths['lfs_dataset'] / 'sparse/0/points3D.ply').resolve() != shared_point_cloud:
        raise ValueError('LFS point-cloud link changed')
    for folder in ('images', 'masks'):
        if (paths['lfs_dataset'] / folder).resolve() != paths['lfs_shared'] / folder:
            raise ValueError(f'LFS {folder} link changed')
    release = work / 'baseline_colmap'
    if release.exists() and _verify_release:
        identity_path = release / 'baseline_export.json'
        identity = json.loads(identity_path.read_text())
        if (identity.get('baseline_name') != manifest['name']
                or identity.get('baseline_manifest_sha256') != _digest(manifest_path)
                or identity.get('model_sha256') != manifest['model_sha256']):
            raise ValueError('Packaged baseline identity changed')
        for name in MODEL_FILES:
            _verify_file(release / 'sparse/0' / name, manifest['model_sha256'][name])
        _verify_file(release / 'sparse/0/points3D.ply', manifest['common_point_cloud_sha256'])
        _verify_file(release / 'lfs_config.json', manifest['lfs_config_sha256'])

    import pycolmap
    model = pycolmap.Reconstruction(paths['source_model'])
    exported = pycolmap.Reconstruction(paths['lfs_dataset'] / 'sparse/0')
    records = {row['name']: row for row in scene['images']}
    names = {model.images[image_id].name for image_id in model.reg_image_ids()}
    export_names = {exported.images[image_id].name for image_id in exported.reg_image_ids()}
    if (len(records) != manifest['image_count'] or names != set(records)
            or export_names != names or model.num_reg_images() != len(records)
            or exported.num_reg_images() != len(records)):
        raise ValueError('Selected or registered image membership changed')
    if Counter(row['camera'] for row in records.values()) != manifest['lenses']:
        raise ValueError('L/R image membership changed')
    if (len(model.cameras) != 2 or len(exported.cameras) != 2
            or any(camera.model.name != manifest['camera_model']
                   for camera in (*model.cameras.values(), *exported.cameras.values()))):
        raise ValueError('Expected two native OPENCV_FISHEYE cameras')

    images_and_masks = []
    for name, record in sorted(records.items()):
        original = Path(record['source']).resolve()
        semantic = Path(record['semantic_mask']).resolve()
        linked = paths['lfs_dataset'] / 'images' / name
        mask = paths['lfs_dataset'] / 'masks' / (name + '.png')
        if (not original.is_file() or not semantic.is_file() or not linked.is_file()
                or linked.resolve() != original or not mask.is_file()):
            raise FileNotFoundError(f'Baseline original or mask missing: {name}')
        if release.exists() and _verify_release:
            released_image = release / 'images' / name
            released_mask = release / 'masks' / (name + '.png')
            if (not released_image.is_file() or released_image.resolve() != original
                    or not released_mask.is_file()):
                raise FileNotFoundError(f'Packaged baseline image or mask missing: {name}')
            if full and file_digest(released_mask) != file_digest(mask):
                raise ValueError(f'Packaged baseline mask differs: {name}')
        if full:
            images_and_masks.append({
                'name': name,
                'image_sha256': file_digest(original),
                'source_semantic_mask_sha256': file_digest(semantic),
                'combined_mask_sha256': file_digest(mask),
            })
    if full and digest(images_and_masks) != manifest['originals_and_masks_sha256']:
        raise ValueError('Original image or mask contents changed')
    return {
        'name': manifest['name'], 'status': manifest['status'],
        'manifest': str(manifest_path), 'work': str(work),
        'images': len(records), 'lenses': dict(Counter(row['camera'] for row in records.values())),
        'model': str(paths['source_model']), 'lfs_dataset': str(paths['lfs_dataset']),
        'release_dataset': str(release) if release.exists() else None,
        'protocol_fingerprint': protocol['fingerprint'], 'full_content_check': full,
    }


def export_baseline(repo_root='.', manifest_path=DEFAULT_MANIFEST, work_override=None,
                    output=None, *, image_mode='link', include_masks=False, include_ply=False):
    """Export the frozen source geometry without depending on LFS assets/config."""
    from .colmap_export import export_colmap

    manifest, manifest_path, work, paths = _load_paths(repo_root, manifest_path, work_override)
    _verify_file(paths['scene_manifest'], manifest['scene_manifest_sha256'])
    for name in MODEL_FILES:
        _verify_file(paths['source_model'] / name, manifest['model_sha256'][name])
    scene = json.loads(paths['scene_manifest'].read_text())
    if (scene.get('fingerprint') != manifest['scene_fingerprint']
            or len(scene['images']) != manifest['image_count']
            or Counter(row['camera'] for row in scene['images']) != manifest['lenses']):
        raise ValueError('Frozen scene membership changed')
    output = Path(output).resolve() if output else work / 'colmap'
    if output == work or output.is_relative_to(work / 'global_sfm'):
        raise ValueError('Choose a separate COLMAP export directory')
    return export_colmap(paths['source_model'], scene, output, image_mode=image_mode,
                         include_masks=include_masks, include_ply=include_ply,
                         provenance={'kind': 'frozen_visual_baseline',
                                     'baseline_name': manifest['name'],
                                     'baseline_manifest_sha256': _digest(manifest_path),
                                     'source_model_sha256': manifest['model_sha256']})
