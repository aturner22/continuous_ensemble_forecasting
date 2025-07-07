from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from contextlib import nullcontext

import os

def _resolve_worker_count(default_max: int = 8) -> int:
    import multiprocessing as mp
    try:
        requested = int(os.getenv("N_WORKERS", "0"))
        if requested > 0:
            return min(requested, default_max)
    except ValueError:
        pass
    return min(mp.cpu_count() // 2, default_max)

N_WORKERS = _resolve_worker_count()
BACKEND = os.getenv("PARALLEL_BACKEND", "process")

_executor_cls = ThreadPoolExecutor if BACKEND == "thread" else ProcessPoolExecutor

def parallel_map(func, iterable):
    if N_WORKERS == 1:
        for item in iterable:
            yield func(item)
    else:
        with _executor_cls(max_workers=N_WORKERS) as pool:
            futures = [pool.submit(func, x) for x in iterable]
            for f in as_completed(futures):
                yield f.result()
