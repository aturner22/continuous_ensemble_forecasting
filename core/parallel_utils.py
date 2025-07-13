from __future__ import annotations

import importlib.util as _imp
import multiprocessing as _mp
import os as _os
import resource as _resource
import sys as _sys
from contextlib import suppress as _suppress
from concurrent.futures import (
    ProcessPoolExecutor as _ProcessPoolExecutor,
    ThreadPoolExecutor as _ThreadPoolExecutor,
    as_completed as _as_completed,
)

# ───────────────────────────────────────────── helpers ──────────────────────────────────────────────

def _positive_int(text: str, default: int) -> int:
    with _suppress(ValueError):
        val = int(text)
        if val > 0:
            return val
    return default


def _soft_fd_limit() -> int:
    with _suppress(Exception):
        return _resource.getrlimit(_resource.RLIMIT_NOFILE)[0]
    return 1024


def _torch_use_file_system_sharing() -> None:
    with _suppress(ImportError):
        import torch.multiprocessing as _tmp  # type: ignore
        if _tmp.get_sharing_strategy() != "file_system":
            _tmp.set_sharing_strategy("file_system")


def _ray_requested_or_available() -> bool:
    if _imp.find_spec("ray") is None:
        return False
    return bool(
        _os.getenv("PARALLEL_BACKEND", "").lower() == "ray"
        or _os.getenv("RAY_ADDRESS")
        or _os.getenv("RAY_CLUSTER")
    )


def _init_ray() -> None:  # pragma: no cover
    import ray  # type: ignore
    if not ray.is_initialized():
        ray.init(
            address=_os.getenv("RAY_ADDRESS", "auto"),
            namespace="abc_gibbs",
            ignore_reinit_error=True,
        )


# ───────────────────────────────────── worker-quota & backend ───────────────────────────────────────

def _worker_quota(cap: int = 8) -> int:
    requested = _positive_int(_os.getenv("N_WORKERS", "0"), 0)
    if requested:
        return min(requested, cap)
    cpu_total = _mp.cpu_count()
    if cpu_total <= 2:
        return 1
    return min(max(cpu_total // 2, 1), cap)


N_WORKERS: int = _worker_quota()
_FD_LIMIT: int = _soft_fd_limit()
_BACKEND_ENV: str = _os.getenv("PARALLEL_BACKEND", "auto").lower()

if _BACKEND_ENV == "auto":
    if _ray_requested_or_available():
        _BACKEND_ENV = "ray"
    elif _FD_LIMIT < 4096:            # tight RLIMIT ⇒ FD-neutral threads
        _BACKEND_ENV = "thread"
    elif N_WORKERS == 1:
        _BACKEND_ENV = "sequential"
    else:
        _BACKEND_ENV = "process"

if _BACKEND_ENV not in {"process", "thread", "ray", "sequential"}:
    raise ValueError(f"Unsupported backend {_BACKEND_ENV!r}")

_torch_use_file_system_sharing()

if _BACKEND_ENV == "thread":
    _executor_cls = _ThreadPoolExecutor
elif _BACKEND_ENV == "process":
    _executor_cls = lambda **kw: _ProcessPoolExecutor(  # type: ignore
        mp_context=_mp.get_context("spawn"), **kw
    )
else:
    _executor_cls = None  # type: ignore[assignment]

# ───────────────────────────────────────────── core mappers ─────────────────────────────────────────

def _pool_map(fn, iterable):
    if N_WORKERS == 1:
        for item in iterable:
            yield fn(item)
        return
    with _executor_cls(max_workers=N_WORKERS) as pool:  # type: ignore[misc]
        futures = [pool.submit(fn, x) for x in iterable]
        for fut in _as_completed(futures):
            yield fut.result()


def _ray_map(fn, iterable):  # pragma: no cover
    _init_ray()
    import ray  # type: ignore
    remote = ray.remote(fn)
    refs = [remote.remote(x) for x in iterable]
    for r in refs:
        yield ray.get(r)


def parallel_map(fn, iterable):
    """
    Resource-adaptive, fault-tolerant parallel map.

    * **sequential**    enforced when `N_WORKERS==1`
    * **thread**        default under low `RLIMIT_NOFILE`
    * **process**       shared-memory, uses torch file_system strategy
    * **ray**           distributed cluster when requested/available

    On `OSError [Errno 24]` (FD exhaustion) the backend downgrades once
    from **process**→**thread** and transparently retries.
    """
    global _BACKEND_ENV, _executor_cls  # pylint: disable=global-statement
    try:
        if _BACKEND_ENV == "sequential":
            return (fn(x) for x in iterable)
        if _BACKEND_ENV == "thread":
            return _pool_map(fn, iterable)
        if _BACKEND_ENV == "process":
            return _pool_map(fn, iterable)
        if _BACKEND_ENV == "ray":
            return _ray_map(fn, iterable)
        raise RuntimeError(f"Unknown backend {_BACKEND_ENV!r}")
    except OSError as exc:
        if exc.errno != 24 or _BACKEND_ENV == "thread":
            raise
        _BACKEND_ENV = "thread"
        _executor_cls = _ThreadPoolExecutor
        _sys.stderr.write(
            "[parallel_utils] Warning: file-descriptor exhaustion detected; "
            "retrying with ThreadPoolExecutor.\n"
        )
        return _pool_map(fn, iterable)
