"""Low-overhead stage resource sampling and nested operation timings."""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
import resource
import gc
import sys
from threading import Event, Thread
import time

_operations = ContextVar('profile_operations', default=None)
_parents = ContextVar('profile_parents', default=())


@contextmanager
def operation(name):
    entries = _operations.get()
    if entries is None:
        yield
        return
    path = (*_parents.get(), name)
    token = _parents.set(path)
    started = time.perf_counter()
    entry = {'path': list(path), 'status': 'running', 'rss_start_bytes': _rss_bytes()}
    entries.append(entry)
    try:
        yield
    except BaseException:
        entry['status'] = 'failed'
        raise
    else:
        entry['status'] = 'completed'
    finally:
        entry['elapsed_seconds'] = time.perf_counter() - started
        entry['rss_end_bytes'] = _rss_bytes()
        _parents.reset(token)


def profiled(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        with operation(f'{function.__module__}.{function.__name__}'):
            return function(*args, **kwargs)
    return wrapped


def _cuda():
    torch = sys.modules.get('torch')
    cuda = getattr(torch, 'cuda', None)
    if cuda is not None and getattr(cuda, 'is_initialized', lambda: False)():
        return cuda
    return None


def _rss_bytes():
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('VmRSS:'):
            return int(line.split()[1]) * 1024
    return 0


def release_gpu_cache():
    """Matching is the last GPU stage; release allocator reserves afterwards."""
    cuda = _cuda()
    if cuda is not None:
        gc.collect()
        cuda.empty_cache()


def reclaim_host_memory():
    """Return free glibc heap pages after memory-heavy stages, when supported."""
    import ctypes
    gc.collect()
    before = _rss_bytes()
    try:
        trim = ctypes.CDLL(None).malloc_trim
    except AttributeError:
        return {'supported': False, 'rss_before_bytes': before, 'rss_after_bytes': _rss_bytes()}
    trim.argtypes = [ctypes.c_size_t]
    trim.restype = ctypes.c_int
    released = bool(trim(0))
    return {'supported': True, 'allocator_released_pages': released,
            'rss_before_bytes': before, 'rss_after_bytes': _rss_bytes()}


class StageResources:
    """RSS is sampled; CUDA allocator peaks exclude non-PyTorch allocations."""
    def start(self):
        self.stop_event = Event()
        self.started = time.perf_counter()
        self.cpu_started = resource.getrusage(resource.RUSAGE_SELF)
        self.values = {'sample_interval_seconds': .1, 'samples': 0,
                       'rss_sampled_peak_bytes': 0,
                       'cuda_allocated_peak_bytes': 0, 'cuda_reserved_peak_bytes': 0}
        self.operations = []
        self.token = _operations.set(self.operations)
        cuda = _cuda()
        if cuda is not None:
            cuda.reset_peak_memory_stats()
        self._safe_sample()
        self.values['rss_start_bytes'] = self.values['rss_sampled_peak_bytes']
        self.thread = Thread(target=self._poll, name='stage-resources', daemon=True)
        self.thread.start()

    def _sample(self):
        self.values['samples'] += 1
        self.values['rss_sampled_peak_bytes'] = max(self.values['rss_sampled_peak_bytes'], _rss_bytes())
        cuda = _cuda()
        if cuda is not None:
            for key, value in [('cuda_allocated_peak_bytes', cuda.max_memory_allocated()),
                               ('cuda_reserved_peak_bytes', cuda.max_memory_reserved())]:
                self.values[key] = max(self.values[key], value)

    def _poll(self):
        while not self.stop_event.wait(.1):
            self._safe_sample()

    def _safe_sample(self):
        try:
            self._sample()
        except (OSError, RuntimeError) as exc:
            self.values['sampling_error'] = str(exc)

    def stop(self):
        self.stop_event.set()
        self.thread.join()
        self._safe_sample()
        cpu = resource.getrusage(resource.RUSAGE_SELF)
        elapsed = time.perf_counter() - self.started
        self.values.update(rss_end_bytes=_rss_bytes(),
                           process_rss_high_water_bytes=cpu.ru_maxrss * 1024,
                           cpu_seconds=(cpu.ru_utime + cpu.ru_stime -
                                        self.cpu_started.ru_utime - self.cpu_started.ru_stime))
        cuda = _cuda()
        if cuda is not None:
            self.values.update(cuda_allocated_end_bytes=cuda.memory_allocated(),
                               cuda_reserved_end_bytes=cuda.memory_reserved())
        self.values['average_cpu_cores'] = self.values['cpu_seconds'] / max(elapsed, 1e-9)
        _operations.reset(self.token)
        return self.values
