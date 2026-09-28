import json
from pathlib import Path
import shutil

from PIL import Image
import pycolmap
import pytest

from geosunpre.cli import main
from geosunpre.sfm.evaluation import evaluate_reconstruction
from geosunpre.pipeline import runner as build
from geosunpre.colmap import MODEL_FILES
from geosunpre.colmap import export_colmap
from geosunpre.colmap import verify_colmap_export
from geosunpre.artifacts import file_digest
from geosunpre.artifacts import write_json
from test_pipeline import tiny_visual_work


def test_copy_export_is_portable_and_preserves_full_model(tmp_path):
    work, root, scene, _, records = tiny_visual_work(tmp_path)
    output = tmp_path / 'export'
    original_hashes = {name: file_digest(work / build.SOURCE / name) for name in MODEL_FILES}
    exported = export_colmap(work / build.SOURCE, scene, output, image_mode='copy',
                             include_masks=True, include_ply=True)
    assert exported['points'] == 20
    assert (output / 'sparse/0/points3D.ply').is_file()
    assert not (output / 'lfs_config.json').exists()
    assert not (output / 'images/L/a.jpg').is_symlink()
    assert {name: file_digest(output / 'sparse/0' / name) for name in MODEL_FILES} == original_hashes
    relocated = tmp_path / 'delivery/nested/dataset'
    relocated.parent.mkdir(parents=True)
    output.rename(relocated)
    shutil.rmtree(work)
    for record in records.values():
        Path(record['source']).unlink()
        Path(record['semantic_mask']).unlink()
    result = verify_colmap_export(relocated)
    assert result['images'] == 2
    assert result['points'] == 20
    assert result['include_masks'] is True
    assert pycolmap.Reconstruction(relocated / 'sparse/0').is_valid()


@pytest.mark.parametrize('relative', ['images/L/a.jpg', 'masks/L/a.jpg.png', 'sparse/0/points3D.bin'])
def test_content_changes_are_rejected(tmp_path, relative):
    work, _, scene, _, _ = tiny_visual_work(tmp_path)
    output = tmp_path / 'export'
    export_colmap(work / build.SOURCE, scene, output, image_mode='copy', include_masks=True)
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






def test_cli_exports_and_checks_without_a_work_directory(tmp_path, capsys):
    work, root, _, _, _ = tiny_visual_work(tmp_path)
    output = tmp_path / 'export'
    main(['export', '--scene', str(root / 'manifest.json'),
          '--model', str(work / build.SOURCE), '--output', str(output), '--image-mode', 'copy'])
    assert json.loads(capsys.readouterr().out)['images'] == 2
    shutil.rmtree(work)
    main(['check', '--output', str(output)])
    assert json.loads(capsys.readouterr().out)['full_content_check'] is True


def test_nonempty_evaluation_creates_directory_and_retains_failed_queries(tmp_path):
    work, _, scene, features, _ = tiny_visual_work(tmp_path)
    # A missing support view is a legitimate failed query and stays in the denominator.
    split = {'tracks': [{'id': 'missing-support',
                         'support': [('L/a.jpg', 0), ('missing.jpg', 0)],
                         'queries': [('R/b.jpg', 0)]}]}
    output = tmp_path / 'new/validation'
    model = pycolmap.Reconstruction(work / build.SOURCE)
    report = evaluate_reconstruction(model, scene, features, split, output)
    assert report['status'] == 'evaluated'
    assert report['all_candidate_queries']['count'] == 1
    assert report['all_candidate_queries']['finite_count'] == 0
    assert report['all_candidate_queries']['fraction_le_1px'] == 0
    assert (output / 'validation_observations.npz').is_file()
