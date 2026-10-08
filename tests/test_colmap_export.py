import json
from pathlib import Path
import shutil
import numpy as np

from PIL import Image
import pycolmap
import pytest

from geosunpre.cli import main
from geosunpre.pipeline import finalization as build
from geosunpre.colmap import MODEL_FILES
from geosunpre.colmap import export_colmap
from geosunpre.colmap import verify_colmap_export
from geosunpre.artifacts import file_digest
from geosunpre.artifacts import write_json
from helpers import tiny_visual_work, make_las


def test_copy_export_is_portable_and_preserves_full_model(tmp_path):
    work, root, scene, _, records = tiny_visual_work(tmp_path)
    output = tmp_path / 'export'
    las = tmp_path / 'input.las'
    make_las(las, np.random.default_rng(1).normal(size=(40, 3)))
    original_hashes = {name: file_digest(work / build.SOURCE / name) for name in MODEL_FILES}
    exported = export_colmap(work / build.SOURCE, scene, output, image_mode='copy',
                             include_masks=True, include_ply=True, initialization_las=las)
    assert exported['points'] == 20
    assert (output / 'sparse/0/points3D.ply').is_file()
    assert exported['initialization']['source'] == 'las'
    assert exported['initialization']['points'] == 40
    assert not (output / 'lfs_config.json').exists()
    assert not (output / 'images/L/a.jpg').is_symlink()
    assert {name: file_digest(output / 'sparse/0' / name) for name in MODEL_FILES} == original_hashes
    relocated = tmp_path / 'delivery/nested/dataset'
    relocated.parent.mkdir(parents=True)
    output.rename(relocated)
    shutil.rmtree(work)
    las.unlink()
    for record in records.values():
        Path(record['source']).unlink()
        Path(record['semantic_mask']).unlink()
    result = verify_colmap_export(relocated)
    assert result['images'] == 2
    assert result['points'] == 20
    assert result['include_masks'] is True
    assert pycolmap.Reconstruction(relocated / 'sparse/0').is_valid()


@pytest.mark.parametrize('relative', ['images/L/a.jpg', 'masks/L/a.jpg.png', 'sparse/0/points3D.bin',
                                      'sparse/0/points3D.ply', 'initialization.json'])
def test_content_changes_are_rejected(tmp_path, relative):
    work, _, scene, _, _ = tiny_visual_work(tmp_path)
    output = tmp_path / 'export'
    las = tmp_path / 'input.las'
    make_las(las, np.random.default_rng(1).normal(size=(40, 3)))
    export_colmap(work / build.SOURCE, scene, output, image_mode='copy', include_masks=True,
                  include_ply=True, initialization_las=las)
    path = output / relative
    with path.open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='changed'):
        verify_colmap_export(output)


def test_failed_export_does_not_publish_a_partial_dataset(tmp_path):
    work, _, scene, _, records = tiny_visual_work(tmp_path)
    Image.new('RGB', (9, 8)).save(records['R/b.jpg']['source'])
    output = tmp_path / 'export'
    with pytest.raises(ValueError, match='dimensions'):
        export_colmap(work / build.SOURCE, scene, output, image_mode='copy')
    assert not output.exists()
    assert not list(tmp_path.glob('.export-*'))


def test_existing_export_is_immutable_and_nonempty_directory_is_protected(tmp_path):
    work, _, scene, _, _ = tiny_visual_work(tmp_path)
    output = tmp_path / 'export'
    original = export_colmap(work / build.SOURCE, scene, output)
    assert export_colmap(work / build.SOURCE, scene, output) == original
    identity_hash = file_digest(output / 'colmap_export.json')
    with pytest.raises(ValueError, match='Existing COLMAP export differs'):
        export_colmap(work / build.SOURCE, scene, output, image_mode='link')
    assert file_digest(output / 'colmap_export.json') == identity_hash
    foreign = tmp_path / 'foreign'
    write_json(foreign / 'other.json', {'status': 'frozen'})
    with pytest.raises(ValueError, match='choose a new output directory'):
        export_colmap(work / build.SOURCE, scene, foreign)
    assert list(foreign.iterdir()) == [foreign / 'other.json']


def test_cli_checks_portable_delivery_without_a_work_directory(tmp_path, capsys):
    work, root, _, _, _ = tiny_visual_work(tmp_path)
    output = tmp_path / 'export'
    export_colmap(work / build.SOURCE, json.loads((root / 'manifest.json').read_text()), output)
    shutil.rmtree(work)
    main(['check', '--output', str(output)])
    assert json.loads(capsys.readouterr().out)['full_content_check'] is True


def test_cli_work_check_uses_the_frozen_custom_delivery_path(tmp_path, capsys):
    from geosunpre.config import PipelineConfig, save_config
    from geosunpre.pipeline.task import bind_task

    work, _, scene, _, _ = tiny_visual_work(tmp_path)
    output = tmp_path / 'custom_delivery'
    save_config(work, PipelineConfig())
    bind_task(work, output=output)
    export_colmap(work / build.SOURCE, scene, output)
    main(['check', '--work', str(work)])
    result = json.loads(capsys.readouterr().out)
    assert result['output'] == str(output) and result['full_content_check']
