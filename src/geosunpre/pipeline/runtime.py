"""Exclusive work-directory ownership and observable stage completion."""

from contextlib import contextmanager
from datetime import datetime, timezone
import errno
import json
import os
from pathlib import Path
import time
from uuid import uuid4

if os.name == 'nt':
    import msvcrt
else:
    import fcntl

from geosunpre.artifacts import write_json
from geosunpre.profiling import StageResources, release_gpu_cache, reclaim_host_memory


def utc_now():
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def work_lock(work):
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    with (work / '.pipeline.lock').open('a+b') as stream:
        try:
            if os.name == 'nt':
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                raise
            raise RuntimeError(f'Another pipeline process is using {work}') from exc
        try:
            stream.seek(0)
            stream.truncate()
            stream.write(json.dumps({'pid': os.getpid(), 'started_at_utc': utc_now()}).encode('utf-8'))
            stream.flush()
            yield
        finally:
            if os.name == 'nt':
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


class StageRecorder:
    def __init__(self, work, stages):
        self.started = time.perf_counter()
        self.path = Path(work) / 'pipeline_status.json'
        self.value = {'version': 2, 'invocation_id': uuid4().hex, 'pid': os.getpid(), 'started_at_utc': utc_now(),
                      'timing_scope': 'recorded stages and orchestration; excludes CLI/config/task initialization',
                      'requested_stages': list(stages), 'status': 'running', 'stages': []}
        write_json(self.path, self.value)

    @contextmanager
    def stage(self, name):
        started = time.perf_counter()
        entry = {'name': name, 'status': 'running', 'started_at_utc': utc_now()}
        self.value['stages'].append(entry)
        self.value['active_stage'] = name
        write_json(self.path, self.value)
        print(f'Pipeline stage: {name}', flush=True)
        resources = StageResources()
        resources.start()
        try:
            yield
        except BaseException as exc:
            state = 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed'
            entry.update(status=state, error=f'{type(exc).__name__}: {exc}')
            self.value.update(status=state, finished_at_utc=utc_now())
            raise
        else:
            entry['status'] = 'completed'
        finally:
            if name == 'matches':
                try:
                    release_gpu_cache()
                except RuntimeError as exc:
                    entry['gpu_cleanup_error'] = str(exc)
            if name in ('features', 'matches', 'observations'):
                entry['host_memory_reclaim'] = reclaim_host_memory()
            entry['resources'] = resources.stop()
            entry['operations'] = resources.operations
            entry.update(elapsed_seconds=time.perf_counter() - started, finished_at_utc=utc_now())
            self.value['active_stage'] = None
            self.value.update(self.timing())
            write_json(self.path, self.value)
            print(f'Pipeline stage {name}: {entry["status"]} ({entry["elapsed_seconds"]:.1f}s)', flush=True)

    def __enter__(self):
        return self

    def __exit__(self, kind, error, traceback):
        if error is not None:
            self.value.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                              error=f'{type(error).__name__}: {error}', active_stage=None,
                              finished_at_utc=utc_now(), **self.timing())
            write_json(self.path, self.value)

    def complete(self, timing=None):
        self.value.update(status='completed', finished_at_utc=utc_now(), **(timing or self.timing()))
        write_json(self.path, self.value)

    def timing(self):
        elapsed = time.perf_counter() - self.started
        stages = sum(entry.get('elapsed_seconds', 0) for entry in self.value['stages'])
        return {'elapsed_seconds': elapsed, 'stage_elapsed_seconds': stages,
                'overhead_seconds': elapsed - stages, 'clock': 'perf_counter'}
