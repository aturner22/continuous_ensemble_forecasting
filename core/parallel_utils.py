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


# ──────────────────────────────────────────────────────────────────────────────
#  Low-level helpers
# ──────────────────────────────────────────────────────────────────────────────

def _positive_int(text: str, default: int) -> int:
    with _suppress(ValueError):
        value = int(text)
        if value > 0:
            return value
    return default


def _soft_fd_limit() -> int:
    with _suppress(Exception):
        return _resource.getrlimit(_resource.RLIMIT_NOFILE)[0]
    return 1024


def _torch_fs_sharing() -> None:
    with _suppress(ImportError):
        import torch.multiprocessing as _tmp  # type: ignore

        if _tmp.get_sharing_strategy() != "file_system":
            _tmp.set_sharing_strategy("file_system")


def _ray_available() -> bool:
    if _imp.find_spec("ray") is None:
        return False
    return bool(
        _os.getenv("PARALLEL_BACKEND", "").lower() == "ray"
        or _os.getenv("RAY_ADDRESS")
        or _os.getenv("RAY_CLUSTER")
    )


def _init_ray() -> None:  # pragma: no cover
    import ray  # type: ignore

    if ray.is_initialized():
        return
    ray.init(address=_os.getenv("RAY_ADDRESS", "auto"),
             namespace="abc_gibbs",
             ignore_reinit_error=True)


# ──────────────────────────────────────────────────────────────────────────────
#  Worker-quota determination
# ──────────────────────────────────────────────────────────────────────────────

def _worker_quota(cap: int = 8) -> int:
    requested = _positive_int(_os.getenv("N_WORKERS", "0"), 0)
    if requested:
        return min(requested, cap)
    cpu_total = _mp.cpu_count()
    if cpu_total <= 2:
        return 1
    return min(max(cpu_total // 2, 1), cap)


N_WORKERS: int = _worker_quota()

# ──────────────────────────────────────────────────────────────────────────────
#  Backend selection
# ──────────────────────────────────────────────────────────────────────────────

_BACKEND_ENV: str = _os.getenv("PARALLEL_BACKEND", "auto").lower()
_FD_LIMIT: int = _soft_fd_limit()

if _BACKEND_ENV == "auto":
    if _ray_available():
        _BACKEND_ENV = "ray"
    elif _FD_LIMIT < 4096:                 # heuristic: 4 k FDs per task is comfortable
        _BACKEND_ENV = "thread"            # avoids FD usage almost entirely
    elif N_WORKERS == 1:
        _BACKEND_ENV = "sequential"
    else:
        _BACKEND_ENV = "process"

if _BACKEND_ENV not in {"process", "thread", "ray", "sequential"}:
    raise ValueError(f"Unsupported PARALLEL_BACKEND={_BACKEND_ENV!r}")

# prepare torch sharing before any pools are spawned
_torch_fs_sharing()

if _BACKEND_ENV == "thread":
    _executor_cls = _ThreadPoolExecutor
elif _BACKEND_ENV == "process":
    # use *spawn* to avoid inheriting superfluous FDs from the parent
    _executor_cls = lambda **kw: _ProcessPoolExecutor(  # type: ignore
        mp_context=_mp.get_context("spawn"), **kw
    )
else:
    _executor_cls = None  # type: ignore[assignment]


# ──────────────────────────────────────────────────────────────────────────────
#  Parallel map implementations
# ──────────────────────────────────────────────────────────────────────────────

def _local_pool_map(fn, it):
    if N_WORKERS == 1:
        for x in it:
            yield fn(x)
        return
    with _executor_cls(max_workers=N_WORKERS) as pool:  # type: ignore[misc]
        futures = [pool.submit(fn, x) for x in it]
        for fut in _as_completed(futures):
            yield fut.result()


def _ray_pool_map(fn, it):  # pragma: no cover
    _init_ray()
    import ray  # type: ignore

    remote = ray.remote(fn)
    refs = [remote.remote(x) for x in it]
    for r in refs:
        yield ray.get(r)


def parallel_map(fn, it):
    """
    Parallel evaluation with automatic fallback.
    On *Errno 24* the computation is transparently re-executed in a
    thread pool without user intervention.
    """
    try:
        if _BACKEND_ENV == "sequential":
            return (fn(x) for x in it)
        if _BACKEND_ENV == "ray":
            return _ray_pool_map(fn, it)
        return _local_pool_map(fn, it)
    except OSError as exc if isinstance(exc, OSError) else False:  # type: ignore[sintax]
        if exc.errno != 24 or _BACKEND_ENV == "thread":
            raise
        # downgrade once and re-run
        global _BACKEND_ENV, _executor_cls  # noqa: PLW0603
        _BACKEND_ENV = "thread"
        _executor_cls = _ThreadPoolExecutor
        _sys._
