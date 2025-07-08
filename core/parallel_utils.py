from __future__ import annotations

import importlib.util as _importlib_util
import multiprocessing as _multiprocessing
import os as _os
from concurrent.futures import (
    ProcessPoolExecutor as _ProcessPoolExecutor,
    ThreadPoolExecutor as _ThreadPoolExecutor,
    as_completed as _as_completed,
)


def _safe_positive_int(text: str, fallback: int) -> int:
    try:
        value = int(text)
        return value if value > 0 else fallback
    except ValueError:
        return fallback


def _determine_worker_quota(upper_bound: int = 8) -> int:
    explicit = _safe_positive_int(_os.getenv("N_WORKERS", "0"), 0)
    if explicit:
        return min(explicit, upper_bound)
    cpu_total = _multiprocessing.cpu_count()
    if cpu_total <= 2:
        return 1
    return min(max(cpu_total // 2, 1), upper_bound)


def _ray_available() -> bool:
    if _importlib_util.find_spec("ray") is None:
        return False
    if _os.getenv("PARALLEL_BACKEND", "").lower() == "ray":
        return True
    return bool(_os.getenv("RAY_ADDRESS") or _os.getenv("RAY_CLUSTER"))


def _initialise_ray() -> None:
    import ray  # type: ignore
    if ray.is_initialized():
        return
    address = _os.getenv("RAY_ADDRESS", "auto")
    try:
        ray.init(address=address, namespace="abc_gibbs", ignore_reinit_error=True)
    except Exception:
        ray.init(namespace="abc_gibbs", ignore_reinit_error=True)


N_WORKERS: int = _determine_worker_quota()
_BACKEND_ENV: str = _os.getenv("PARALLEL_BACKEND", "auto").lower()

if _BACKEND_ENV == "auto":
    if _ray_available():
        _BACKEND_ENV = "ray"
    elif N_WORKERS == 1:
        _BACKEND_ENV = "sequential"
    else:
        _BACKEND_ENV = "process"

if _BACKEND_ENV not in {"process", "thread", "ray", "sequential"}:
    raise ValueError(f"Unsupported backend {_BACKEND_ENV!r}")

if _BACKEND_ENV == "thread":
    _executor_cls = _ThreadPoolExecutor
elif _BACKEND_ENV == "process":
    _executor_cls = _ProcessPoolExecutor
else:
    _executor_cls = None  # type: ignore[assignment]


def _local_parallel_map(callable_function, argument_iterable):
    if N_WORKERS == 1:
        for argument in argument_iterable:
            yield callable_function(argument)
        return
    with _executor_cls(max_workers=N_WORKERS) as pool:  # type: ignore[misc]
        pending_futures = [pool.submit(callable_function, arg) for arg in argument_iterable]
        for completed_future in _as_completed(pending_futures):
            yield completed_future.result()


def _ray_parallel_map(callable_function, argument_iterable):
    _initialise_ray()
    import ray  # type: ignore
    remote_handle = ray.remote(callable_function)
    object_references = [remote_handle.remote(arg) for arg in argument_iterable]
    for reference in object_references:
        yield ray.get(reference)


def parallel_map(callable_function, argument_iterable):
    if _BACKEND_ENV == "sequential":
        for argument in argument_iterable:
            yield callable_function(argument)
    elif _BACKEND_ENV == "ray":
        yield from _ray_parallel_map(callable_function, argument_iterable)
    else:
        yield from _local_parallel_map(callable_function, argument_iterable)
