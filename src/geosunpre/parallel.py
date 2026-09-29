"""Ordered CPU work with a bounded queue and exception propagation."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from itertools import islice


def ordered_map(function, items, *, workers=4):
    """Keep at most two tasks per worker alive, yielding in input order."""
    iterator = iter(items)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = deque(pool.submit(function, item) for item in islice(iterator, 2 * workers))
        while pending:
            yield pending.popleft().result()
            for item in islice(iterator, 1):
                pending.append(pool.submit(function, item))
