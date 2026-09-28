import json
from pathlib import Path
import shutil

from PIL import Image
import pycolmap
import pytest

from geosunpre.cli import main
from geosunpre.local_evaluation import evaluate_reconstruction
from geosunpre.pipeline import baseline_build as build
from geosunpre.pipeline.baseline import export_baseline
from geosunpre.pipeline.colmap_export import MODEL_FILES, export_colmap, verify_colmap_export
from geosunpre.scene import file_digest, write_json
from test_baseline_build import tiny_visual_work


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


def test_existing_export_is_immutable_and_legacy_directory_is_protected(tmp_path):
    work, _, scene, _, _ = tiny_visual_work(tmp_path)
    output = tmp_path / 'export'
    original = export_colmap(work / build.SOURCE, scene, output)
    assert export_colmap(work / build.SOURCE, scene, output) == original
    identity_hash = file_digest(output / 'colmap_export.json')
    with pytest.raises(ValueError, match='Existing COLMAP export differs'):
        export_colmap(work / build.SOURCE, scene, output, image_mode='copy')
    assert file_digest(output / 'colmap_export.json') == identity_hash
    legacy = tmp_path / 'legacy'
    write_json(legacy / 'baseline_export.json', {'status': 'frozen'})
    with pytest.raises(ValueError, match='choose a new output directory'):
        export_colmap(work / build.SOURCE, scene, legacy)
    assert list(legacy.iterdir()) == [legacy / 'baseline_export.json']


def test_frozen_source_export_needs_no_lfs_files(tmp_path):
    work, root, scene, _, _ = tiny_visual_work(tmp_path)
    manifest = tmp_path / 'baseline.json'
    write_json(manifest, {
        'schema_version': 1, 'status': 'development_baseline', 'name': 'tiny',
        'work_root': str(work), 'scene_manifest': 'global_sfm/manifest.json',
        'scene_manifest_sha256': file_digest(root / 'manifest.json'),
        'scene_fingerprint': scene['fingerprint'], 'source_model': str(build.SOURCE),
        'model_sha256': {name: file_digest(work / build.SOURCE / name) for name in MODEL_FILES},
        'image_count': 2, 'lenses': {'L': 1, 'R': 1},
        **{key: 'missing/' + key for key in ('lfs_dataset', 'lfs_shared', 'lfs_protocol', 'lfs_config')},
    })
    result = export_baseline(manifest_path=manifest)
    assert result['images'] == 2
    assert not (work / 'colmap/lfs_config.json').exists()


def test_joint_experiment_refuses_to_overwrite_the_frozen_baseline(tmp_path):
    from geosunpre.experiments.ground_joint_ba import run_ground_joint_ba

    write_json(tmp_path / 'global_sfm/lfs_ab/protocol.json', {'status': 'frozen'})
    with pytest.raises(ValueError, match='frozen historical baseline'):
        run_ground_joint_ba(tmp_path)
    assert not (tmp_path / 'global_sfm/ground_joint_ba').exists()


def test_cli_exports_and_checks_without_a_work_directory(tmp_path, capsys):
    work, root, _, _, _ = tiny_visual_work(tmp_path)
    output = tmp_path / 'export'
    main(['export-colmap', '--scene', str(root / 'manifest.json'),
          '--model', str(work / build.SOURCE), '--output', str(output), '--image-mode', 'copy'])
    assert json.loads(capsys.readouterr().out)['images'] == 2
    shutil.rmtree(work)
    main(['check-colmap', '--output', str(output)])
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
