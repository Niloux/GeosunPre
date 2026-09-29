import os
from pathlib import Path

import pytest

from geosunpre.artifacts import digest_cache, file_digest
from geosunpre.pipeline.runtime import StageRecorder


def test_hash_cache_reuses_reads_and_detects_restored_mtime(tmp_path, monkeypatch):
    path = tmp_path / 'data'
    path.write_bytes(b'old content')
    original_open = Path.open
    reads = []
    def tracked_open(self, *args, **kwargs):
        if self == path and args and args[0] == 'rb':
            reads.append(self)
        return original_open(self, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', tracked_open)
    with digest_cache():
        before = file_digest(path)
        assert file_digest(path) == before
        assert len(reads) == 1
        stat = path.stat()
        path.write_bytes(b'new content')
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        assert file_digest(path) != before
        assert len(reads) == 2
    file_digest(path)
    assert len(reads) == 3


def test_hash_cache_detects_atomic_replacement(tmp_path):
    path = tmp_path / 'data'
    path.write_bytes(b'first')
    with digest_cache():
        before = file_digest(path)
        other = tmp_path / 'replacement'
        other.write_bytes(b'other')
        other.replace(path)
        assert file_digest(path) != before


def test_stage_records_resources_nested_timings_and_failures(tmp_path):
    from geosunpre.profiling import operation
    recorder = StageRecorder(tmp_path, ['example'])
    with pytest.raises(ValueError, match='example'):
        with recorder.stage('example'):
            with operation('parent'):
                with operation('child'):
                    raise ValueError('example')
    entry = recorder.value['stages'][0]
    assert entry['status'] == 'failed'
    assert [o['path'] for o in entry['operations']] == [['parent'], ['parent', 'child']]
    assert all(o['status'] == 'failed' for o in entry['operations'])
    assert entry['resources']['samples'] >= 2
    assert entry['resources']['rss_sampled_peak_bytes'] > 0
    assert entry['resources']['cpu_seconds'] >= 0


def test_feature_cache_hits_do_not_load_cpu_data_and_eviction_is_bounded(tmp_path):
    torch = pytest.importorskip('torch')
    import numpy as np
    from geosunpre.features import FrontendConfig, LoMaFrontend
    frontend = LoMaFrontend(tmp_path, {}, FrontendConfig())
    frontend.device = torch.device('cpu')
    frontend._gpu_feature_limit = 32
    loaded = []
    def loader(token):
        loaded.append(token)
        return {'normalized_keypoints': np.zeros((1, 2), dtype=np.float32),
                'descriptors': np.ones((1, 2), dtype=np.float32)}
    first = frontend._matching_tensors({'token': 'a'}, loader)
    frontend._matching_tensors({'token': 'b'}, loader)
    assert frontend._matching_tensors({'token': 'a'}, loader) is first
    frontend._matching_tensors({'token': 'c'}, loader)
    assert loaded == ['a', 'b', 'c']
    assert list(frontend._gpu_features) == ['a', 'c']
    assert frontend.cache_stats['gpu_feature_peak_bytes'] == 32
    frontend._matching_tensors({'token': 'b'}, loader)
    assert loaded == ['a', 'b', 'c', 'b']
    assert frontend._gpu_feature_bytes == 32


def test_compact_tracks_keep_lexical_ties_and_reject_image_conflicts():
    import numpy as np
    from geosunpre.sfm.geometry import build_tracks
    images=[{'name':name,'camera':'L' if name!='c' else 'R'} for name in ['c','b','a']]
    pairs=[('b','a',np.array([[0,0],[1,1]]),np.array([.9,.9])),
           ('a','c',np.array([[0,0],[1,0]]),np.array([.8,.8]))]
    summary,tracks=build_tracks(iter(pairs),iter(images))
    assert tracks==[[('a',0),('b',0),('c',0)],[('a',1),('b',1)]]
    assert summary['accepted_merges']==3
    assert summary['conflicting_edge_merges_rejected']==1
    assert summary['cross_camera_tracks']==1


def test_compact_tracks_support_more_than_64_images_and_empty_edges():
    import numpy as np
    from geosunpre.sfm.geometry import build_tracks
    images=[{'name':f'{i:03d}','camera':'L'} for i in range(70)]
    pairs=[('000','064',np.array([[0,0]]),np.array([1.])),
           ('064','069',np.array([[0,0]]),np.array([1.]))]
    assert build_tracks(pairs,images)[1]==[[('000',0),('064',0),('069',0)]]
    assert build_tracks([],images)[1]==[]
