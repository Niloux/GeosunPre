"""Exclusive work-directory ownership and observable stage completion."""

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import time

from geosunpre.artifacts import write_json


def utc_now():
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def work_lock(work):
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    with (work / '.pipeline.lock').open('a+') as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f'Another pipeline process is using {work}') from exc
        stream.seek(0)
        stream.truncate()
        stream.write(json.dumps({'pid': os.getpid(), 'started_at_utc': utc_now()}))
        stream.flush()
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


class StageRecorder:
    def __init__(self, work, stages):
        self.path = Path(work) / 'pipeline_status.json'
        self.value = {'version': 1, 'pid': os.getpid(), 'started_at_utc': utc_now(),
                      'requested_stages': list(stages), 'status': 'running', 'stages': []}
        write_json(self.path, self.value)

    @contextmanager
    def stage(self, name):
        entry = {'name': name, 'status': 'running', 'started_at_utc': utc_now()}
        self.value['stages'].append(entry)
        self.value['active_stage'] = name
        write_json(self.path, self.value)
        print(f'Pipeline stage: {name}', flush=True)
        started = time.monotonic()
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
            entry.update(finished_at_utc=utc_now(), elapsed_seconds=time.monotonic() - started)
            self.value['active_stage'] = None
            write_json(self.path, self.value)
            print(f'Pipeline stage {name}: {entry["status"]} ({entry["elapsed_seconds"]:.1f}s)', flush=True)

    def complete(self):
        self.value.update(status='completed', finished_at_utc=utc_now())
        write_json(self.path, self.value)
