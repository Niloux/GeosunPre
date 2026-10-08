import os
from pathlib import Path
from threading import get_ident
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from geosunpre import masks
from geosunpre.artifacts import file_digest
from geosunpre.features import FRONTEND_CONFIG, feature_input
from geosunpre.scene import prepare, prepared_image_digest
from helpers import raw_input


def test_prepare_freezes_bytes_even_when_size_and_mtime_are_restored(tmp_path):
    scene = prepare(raw_input(tmp_path), tmp_path / 'work')
    record = scene['images'][0]
    source = Path(record['source'])
    assert prepared_image_digest(record) == record['source_sha256']
    original = source.read_bytes()
    # A JPEG metadata byte can change without changing dimensions or file size.
    changed = bytearray(original)
    changed[15] ^= 1
    source.write_bytes(changed)
    os.utime(source, ns=(source.stat().st_atime_ns, record['mtime_ns']))
    with pytest.raises(ValueError, match='content changed since prepare'):
        prepared_image_digest(record)
    with pytest.raises(ValueError, match='content changed since prepare'):
        feature_input(tmp_path / 'work', scene, FRONTEND_CONFIG, record)
    with pytest.raises(ValueError, match='Frozen inputs changed'):
        prepare(scene['input_root'], tmp_path / 'work')


def test_prepare_checks_fisheye_domain_before_publishing_inputs(tmp_path):
    root = raw_input(tmp_path)
    path = root / 'scene.pro'
    project = json.loads(path.read_text())
    project['cameraDevices']['devices'][0]['dataSensorParams']['k1'] = -1.
    path.write_text(json.dumps(project))
    work = tmp_path / 'work'
    with pytest.raises(ValueError, match='not monotonic'):
        prepare(root, work)
    assert not (work / 'manifest.json').exists()


def test_mask_pipeline_keeps_inference_order_and_reuses_decode_for_review(tmp_path, monkeypatch):
    root = raw_input(tmp_path)
    for i, path in enumerate(sorted((root / 'Image').rglob('*.JPG'))):
        Image.new('RGB', (8,8), (20 + i * 50, 20, 30)).save(path)
    scene = prepare(root, tmp_path / 'work')
    expected_values = []
    for row in scene['images']:
        with Image.open(row['source']) as im:
            expected_values.append(int(np.asarray(im.convert('RGB'))[0,0,0]))
    output = tmp_path / 'work/sfm/masks_v2'
    caller = get_ident()
    preparation_threads, inference_threads, inference_values = [], [], []
    open_image = Image.open

    def decode(path, *args, **kwargs):
        if str(path) in {r['source'] for r in scene['images']}:
            preparation_threads.append(get_ident())
        return open_image(path, *args, **kwargs)

    def sampling(cam):
        yy, xx = np.mgrid[:8, :8]
        source = np.stack((xx, yy), axis=-1).astype(np.float32)
        views = [(source, xx.astype(np.float32), yy.astype(np.float32), np.ones((8,8), np.float32))] * 5
        return views, np.ones((8,8,1), np.float32)

    def predict(views, geometry, model, processor):
        inference_threads.append(get_ident())
        inference_values.append(int(views[0][0,0,0]))
        assert len(views) == 5 and all(v.shape == (8,8,3) for v in views)
        probability = np.zeros((8,8,2), np.float32)
        probability[:2,:,0] = .95
        probability[4:6,2:4,1] = .6
        return probability

    monkeypatch.setattr(Image, 'open', decode)
    monkeypatch.setattr(masks, 'fusion_geometry', sampling)
    monkeypatch.setattr(masks, '_load_segmentation', lambda cache: (object(), SimpleNamespace(config=SimpleNamespace(id2label={0:'sky', 1:'person'}))))
    monkeypatch.setattr(masks, 'predict_sky_person', predict)
    frames = sorted({r['frame_index'] for r in scene['images']})
    records, metadata = masks.generate_masks(scene, output, frames)
    assert inference_threads == [caller] * len(records)
    assert inference_values == expected_values
    assert len(preparation_threads) == len(records) and all(t != caller for t in preparation_threads)
    assert [r['name'] for r in records] == [r['name'] for r in scene['images']]
    before = {str(p): file_digest(p) for p in output.parent.rglob('*') if p.is_file()}

    def forbidden(*args, **kwargs):
        raise AssertionError('Completed masks and previews must not decode images or load a model')

    monkeypatch.setattr(Image, 'open', forbidden)
    monkeypatch.setattr(masks, '_load_segmentation', forbidden)
    monkeypatch.setattr(masks, 'fusion_geometry', forbidden)
    assert masks.generate_masks(scene, output, frames) == (records, metadata)
    report = masks.mask_review({**scene, 'images': records}, output.parent)
    assert report['images'] == len(records)
    assert all(file_digest(p) == expected for p, expected in before.items())
    assert all(masks.mask_cached(r, output) for r in records)
    # Preview loss needs only a CPU redraw; it does not invalidate semantic masks.
    first = records[0]
    (output.parent / 'mask_review' / f'{first["camera"]}{first["frame_index"]:05d}.jpg').unlink()
    monkeypatch.setattr(Image, 'open', open_image)
    masks.mask_review({**scene, 'images': records}, output.parent)
    assert masks.mask_cached(first, output)
    labels = output / 'labels' / (first['name'] + '.png')
    labels.write_bytes(b'changed labels')
    with pytest.raises(ValueError, match='Committed mask artifact changed'):
        masks.generate_masks(scene, output, frames)


@pytest.mark.parametrize('artifact', ['valid', 'labels'])
def test_committed_mask_damage_is_not_silently_reinferred(tmp_path, artifact):
    scene = prepare(raw_input(tmp_path), tmp_path / 'work')
    record = scene['images'][0]
    output = tmp_path / 'masks'
    paths = {folder: output / folder / (record['name'] + '.png') for folder in ('valid', 'labels')}
    for p in paths.values():
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.new('L', (8,8), 255).save(p)
    assert not masks.mask_cached(record, output)
    from geosunpre.artifacts import write_json
    write_json(output / 'stats' / (record['name'] + '.json'),
               {'source_sha256': record['source_sha256'], 'mask_sha256': file_digest(paths['valid']),
                'labels_sha256': file_digest(paths['labels'])})
    assert masks.mask_cached(record, output)
    paths[artifact].unlink()
    with pytest.raises(ValueError, match='Committed mask artifact changed'):
        masks.mask_cached(record, output)
import json
