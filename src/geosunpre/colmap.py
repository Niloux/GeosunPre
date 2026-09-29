"""Consumer-independent COLMAP datasets, published only after validation."""

import json
import os
from pathlib import Path, PurePosixPath
import shutil
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image
import pycolmap

from geosunpre.camera import Camera
from geosunpre.artifacts import file_digest
from geosunpre.artifacts import write_json
from geosunpre.parallel import ordered_map


MODEL_FILES = ('cameras.bin', 'images.bin', 'points3D.bin', 'rigs.bin', 'frames.bin')
IDENTITY_FILE = 'colmap_export.json'


def _relative_path(value):
    path = PurePosixPath(value)
    if not value or path.is_absolute() or '..' in path.parts or '\\' in value or str(path) != value:
        raise ValueError(f'Expected a canonical relative dataset path: {value}')
    return Path(*path.parts)


def _contained_file(root, relative, *, image_link=False):
    path = root / _relative_path(relative)
    if not path.parent.resolve().is_relative_to(root):
        raise ValueError(f'Dataset directory escapes export root: {relative}')
    if not image_link and (path.is_symlink() or not path.resolve().is_relative_to(root)):
        raise ValueError(f'Expected a file inside the export: {relative}')
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _read_model(path):
    model = pycolmap.Reconstruction(path)
    if not model.is_valid() or not model.num_reg_images():
        raise ValueError('Expected an internally consistent, registered COLMAP reconstruction')
    if model.num_images() != model.num_reg_images():
        raise ValueError('Export model contains unregistered images')
    names = [image.name for image in model.images.values()]
    if len(names) != len(set(names)):
        raise ValueError('Duplicate COLMAP image names')
    for camera in model.cameras.values():
        if camera.width <= 0 or camera.height <= 0 or not np.isfinite(camera.params).all():
            raise ValueError(f'Invalid camera: {camera.camera_id}')
    for image in model.images.values():
        _relative_path(image.name)
        if not np.isfinite(image.cam_from_world().matrix()).all():
            raise ValueError(f'Nonfinite image pose: {image.name}')
        if any(not np.isfinite(point.xy).all() for point in image.points2D):
            raise ValueError(f'Nonfinite image observation: {image.name}')
    if any(not np.isfinite(point.xyz).all() for point in model.points3D.values()):
        raise ValueError('Nonfinite COLMAP point coordinates')
    return model


def verify_colmap_export(output):
    """Validate a dataset using only its exported files and image links."""
    output = Path(output).resolve()
    identity_path = _contained_file(output, IDENTITY_FILE)
    identity = json.loads(identity_path.read_text())
    if (identity.get('schema_version') != 1 or identity.get('format') != 'COLMAP'
            or identity.get('image_mode') not in ('link', 'copy')
            or set(identity.get('model_sha256', {})) != set(MODEL_FILES)):
        raise ValueError('Unsupported COLMAP export contract')
    for name, expected in identity['model_sha256'].items():
        path = _contained_file(output, f'sparse/0/{name}')
        if file_digest(path) != expected:
            raise ValueError(f'Exported COLMAP model changed: {name}')
    model = _read_model(output / 'sparse/0')
    records = identity['images']
    by_name = {image.name: image for image in model.images.values()}
    if (len(records) != len(by_name) or {row['name'] for row in records} != set(by_name)
            or model.num_points3D() != identity['points']):
        raise ValueError('Exported model membership differs from the manifest')
    for record in records:
        name = record['name']
        image = by_name[name]
        camera = model.cameras[image.camera_id]
        if (record['camera_id'] != image.camera_id
                or record['size'] != [camera.width, camera.height]):
            raise ValueError(f'Exported camera/image identity changed: {name}')
        original = _contained_file(output, f'images/{name}',
                                   image_link=identity['image_mode'] == 'link')
        if file_digest(original) != record['sha256']:
            raise ValueError(f'Exported image content changed: {name}')
        with Image.open(original) as source:
            if source.size != (camera.width, camera.height) or source.getexif().get(274, 1) != 1:
                raise ValueError(f'Image dimensions or orientation differ: {name}')
        if identity['include_masks']:
            mask = _contained_file(output, f'masks/{name}.png')
            if file_digest(mask) != record['mask_sha256']:
                raise ValueError(f'Exported mask content changed: {name}')
            with Image.open(mask) as source:
                if source.size != (camera.width, camera.height) or source.mode != 'L':
                    raise ValueError(f'Invalid exported mask: {name}')
    for relative, expected in identity['supplemental_sha256'].items():
        if file_digest(_contained_file(output, relative)) != expected:
            raise ValueError(f'Exported supplementary file changed: {relative}')
    return {'output': str(output), 'identity': str(identity_path),
            'model': str(output / 'sparse/0'), 'images': len(records),
            'points': model.num_points3D(), 'image_mode': identity['image_mode'],
            'include_masks': identity['include_masks'], 'full_content_check': True}


def _export_mask(record, scene, destination):
    camera = Camera.from_dict(scene['cameras'][record['camera']])
    valid = camera.mask(scene['fov_degrees']) > 0
    if record.get('semantic_mask'):
        with Image.open(record['semantic_mask']) as mask:
            if mask.size != (camera.width, camera.height):
                raise ValueError(f'Mask/image dimensions differ: {record["name"]}')
            valid &= np.asarray(mask.convert('L')) >= 128
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(valid.astype(np.uint8) * 255).save(destination)


def export_colmap(model_path, scene, output, *, image_mode='copy', include_masks=False,
                  include_ply=False, provenance=None, quality_report=None):
    """Export native images and the full sparse model without training policy.

    Image copies are portable. Image links retain their external source dependency.
    Optional masks preserve the feature-extraction domain from the scene manifest.
    Existing datasets are reusable only when their complete identity is identical.
    """
    if image_mode not in ('link', 'copy'):
        raise ValueError('image_mode must be link or copy')
    model_path, output = Path(model_path).resolve(), Path(output).resolve()
    if output.is_relative_to(model_path) or model_path.is_relative_to(output):
        raise ValueError('Choose an export directory separate from the source model')
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        if not (output / IDENTITY_FILE).is_file():
            raise ValueError('Destination is not a COLMAP export; choose a new output directory')
        verify_colmap_export(output)
    source_hashes = {name: file_digest(model_path / name) for name in MODEL_FILES
                     if (model_path / name).is_file()}
    expected_hashes = (provenance or {}).get('source_model_sha256')
    if expected_hashes is not None and source_hashes != expected_hashes:
        raise ValueError('Source COLMAP model changed after provenance validation')
    model = _read_model(model_path)
    records = {row['name']: row for row in scene['images']}
    by_name = {image.name: image for image in model.images.values()}
    if len(records) != len(scene['images']) or set(records) != set(by_name):
        raise ValueError('Scene images must exactly match the registered model images')
    for record in records.values():
        source = Path(record['source']).resolve()
        if source.is_relative_to(output):
            raise ValueError('Export directory contains a source image')
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f'.{output.name}-', dir=output.parent) as temporary:
        stage = Path(temporary)
        sparse = stage / 'sparse/0'
        sparse.mkdir(parents=True)
        if set(source_hashes) == set(MODEL_FILES):
            for name in MODEL_FILES:
                shutil.copyfile(model_path / name, sparse / name)
                if file_digest(sparse / name) != source_hashes[name]:
                    raise ValueError(f'Source COLMAP model changed during export: {name}')
        else:
            # Older COLMAP models omit rigs/frames. Emit the complete current model.
            model.write(sparse)
        camera_details = {name: (image.camera_id, model.cameras[image.camera_id].width,
                                model.cameras[image.camera_id].height)
                          for name, image in by_name.items()}
        def copy_image(item):
            name, record = item
            source = Path(record['source']).resolve()
            destination = stage / 'images' / _relative_path(name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            source_hash = file_digest(source)
            if image_mode == 'copy':
                shutil.copyfile(source, destination)
            else:
                # Staging and final directories have the same depth and parent.
                destination.symlink_to(os.path.relpath(source, destination.parent))
            camera_id, width, height = camera_details[name]
            row = {'name': name, 'camera_id': camera_id,
                   'size': [width, height], 'sha256': source_hash}
            if include_masks:
                mask_path = stage / 'masks' / (name + '.png')
                _export_mask(record, scene, mask_path)
                row['mask_sha256'] = file_digest(mask_path)
            return row
        images = list(ordered_map(copy_image, sorted(records.items())))
        supplements = {}
        if include_ply:
            model.export_PLY(sparse / 'points3D.ply')
            supplements['sparse/0/points3D.ply'] = file_digest(sparse / 'points3D.ply')
        if quality_report is not None:
            shutil.copyfile(quality_report, stage / 'quality.json')
            supplements['quality.json'] = file_digest(stage / 'quality.json')
        identity = {
            'schema_version': 1, 'format': 'COLMAP',
            'image_mode': image_mode, 'include_masks': include_masks,
            'model_sha256': {name: file_digest(sparse / name) for name in MODEL_FILES},
            'images': images, 'points': model.num_points3D(),
            'supplemental_sha256': supplements, 'provenance': provenance or {},
        }
        write_json(stage / IDENTITY_FILE, identity)
        verify_colmap_export(stage)
        if (output / IDENTITY_FILE).exists():
            if json.loads((output / IDENTITY_FILE).read_text()) != identity:
                raise ValueError('Existing COLMAP export differs; choose a new output directory')
        else:
            # A completed dataset appears atomically; interrupted staging is never reused.
            stage.rename(output)
    return verify_colmap_export(output)
