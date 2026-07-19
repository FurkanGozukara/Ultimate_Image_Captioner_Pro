"""Portable parallel TorchInductor worker configuration.

TorchInductor normally compiles generated kernels in background worker
processes. PyTorch defaults to a single worker on Windows, and its default
``subprocess`` launcher is not compatible with Windows ``pass_fds`` handling.
This module configures a bounded worker pool and uses ``spawn`` on Windows,
matching the proven strategy used by ACE-Step Premium.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from typing import Any, MutableMapping


MIN_COMPILE_THREADS = 1
MAX_COMPILE_THREADS = 32
DEFAULT_COMPILE_THREADS = 8
COMPILE_THREADS_ENV = "TORCHINDUCTOR_COMPILE_THREADS"
WORKER_START_ENV = "TORCHINDUCTOR_WORKER_START"
_SUPPORTED_WORKER_START_METHODS = {"subprocess", "fork", "spawn"}

_CONFIG_LOCK = threading.Lock()


@dataclass(frozen=True)
class CompileWorkerSettings:
    threads: int
    worker_start: str


@dataclass(frozen=True)
class CompileWorkerWarmupHandle:
    threads: int
    worker_start: str
    started_at: float
    pool: Any = None
    futures: tuple[Any, ...] = ()
    error: str = ""


@dataclass(frozen=True)
class CompileWorkerWarmupResult:
    requested_threads: int
    active_workers: int
    worker_start: str
    elapsed_seconds: float
    ready: bool
    detail: str


def normalize_compile_threads(
    value: Any = None,
    *,
    env: MutableMapping[str, str] | None = None,
) -> int:
    """Resolve and clamp a UI/environment worker count."""

    if value is None or str(value).strip() == "":
        source_env = os.environ if env is None else env
        value = source_env.get(COMPILE_THREADS_ENV, DEFAULT_COMPILE_THREADS)
    try:
        threads = int(round(float(value)))
    except (TypeError, ValueError):
        threads = DEFAULT_COMPILE_THREADS
    return max(MIN_COMPILE_THREADS, min(MAX_COMPILE_THREADS, threads))


def prepare_compile_worker_env(
    env: MutableMapping[str, str],
    value: Any = None,
    *,
    platform: str | None = None,
) -> CompileWorkerSettings:
    """Write platform-correct Inductor worker settings to an environment."""

    resolved_platform = sys.platform if platform is None else platform
    threads = normalize_compile_threads(value, env=env)
    if resolved_platform == "win32":
        worker_start = "spawn"
    else:
        worker_start = str(env.get(WORKER_START_ENV, "subprocess")).strip() or "subprocess"
        if worker_start not in _SUPPORTED_WORKER_START_METHODS:
            worker_start = "subprocess"
    env[COMPILE_THREADS_ENV] = str(threads)
    env[WORKER_START_ENV] = worker_start
    return CompileWorkerSettings(threads=threads, worker_start=worker_start)


def configure_compile_workers(value: Any = None) -> CompileWorkerSettings:
    """Configure TorchInductor before its first asynchronous compilation."""

    with _CONFIG_LOCK:
        settings = prepare_compile_worker_env(os.environ, value)
        _apply_loaded_inductor_config(settings.threads, settings.worker_start)
    return settings


def _compile_worker_ready_probe(delay_seconds: float) -> int:
    """Keep a probe busy briefly so the executor starts the complete pool."""

    if delay_seconds > 0:
        time.sleep(delay_seconds)
    return os.getpid()


def _load_async_compile_runtime() -> tuple[Any, int]:
    from torch._inductor.async_compile import AsyncCompile, get_compile_threads

    return AsyncCompile, int(get_compile_threads())


def start_compile_worker_warmup(value: Any = None) -> CompileWorkerWarmupHandle:
    """Start every configured Inductor process without blocking generation."""

    settings = configure_compile_workers(value)
    started_at = time.perf_counter()
    if settings.threads <= 1:
        return CompileWorkerWarmupHandle(settings.threads, settings.worker_start, started_at)

    try:
        async_compile, configured_threads = _load_async_compile_runtime()
        if configured_threads != settings.threads:
            return CompileWorkerWarmupHandle(
                settings.threads,
                settings.worker_start,
                started_at,
                error=(
                    "TorchInductor worker mismatch after configuration: "
                    f"requested {settings.threads}, runtime reports {configured_threads}"
                ),
            )
        pool = async_compile.process_pool()
        futures = tuple(
            pool.submit(_compile_worker_ready_probe, 0.2)
            for _ in range(settings.threads)
        )
        async_compile.use_process_pool()
        return CompileWorkerWarmupHandle(
            settings.threads,
            settings.worker_start,
            started_at,
            pool=pool,
            futures=futures,
        )
    except Exception as exc:
        return CompileWorkerWarmupHandle(
            settings.threads,
            settings.worker_start,
            started_at,
            error=f"{type(exc).__name__}: {exc}",
        )


def finish_compile_worker_warmup(
    handle: CompileWorkerWarmupHandle,
    *,
    timeout_seconds: float = 120.0,
) -> CompileWorkerWarmupResult:
    """Wait for process startup and confirm Inductor marked its pool ready."""

    if handle.error:
        return _warmup_result(handle, 0, False, handle.error)
    if handle.threads <= 1:
        return _warmup_result(handle, 1, True, "serial compilation")

    deadline = time.perf_counter() + max(0.1, float(timeout_seconds))
    worker_pids: set[int] = set()
    try:
        for future in handle.futures:
            remaining = max(0.0, deadline - time.perf_counter())
            worker_pids.add(int(future.result(timeout=remaining)))
    except (FuturesTimeoutError, TimeoutError) as exc:
        return _warmup_result(handle, len(worker_pids), False, f"worker startup timed out: {exc}")
    except Exception as exc:
        return _warmup_result(
            handle,
            len(worker_pids),
            False,
            f"worker startup failed: {type(exc).__name__}: {exc}",
        )

    active_workers = len(worker_pids)
    processes = getattr(handle.pool, "_processes", None)
    if isinstance(processes, dict):
        active_workers = max(
            active_workers,
            sum(
                1
                for process in processes.values()
                if process is not None and process.is_alive()
            ),
        )
    try:
        async_compile, _configured_threads = _load_async_compile_runtime()
        while not async_compile.use_process_pool():
            if time.perf_counter() >= deadline:
                return _warmup_result(
                    handle,
                    active_workers,
                    False,
                    "workers started but Inductor did not mark the pool ready",
                )
            time.sleep(0.01)
    except Exception as exc:
        return _warmup_result(
            handle,
            active_workers,
            False,
            f"Inductor pool readiness check failed: {type(exc).__name__}: {exc}",
        )
    return _warmup_result(
        handle,
        active_workers,
        True,
        f"worker probes completed on {active_workers} process(es)",
    )


def _warmup_result(
    handle: CompileWorkerWarmupHandle,
    active_workers: int,
    ready: bool,
    detail: str,
) -> CompileWorkerWarmupResult:
    return CompileWorkerWarmupResult(
        requested_threads=handle.threads,
        active_workers=active_workers,
        worker_start=handle.worker_start,
        elapsed_seconds=max(0.0, time.perf_counter() - handle.started_at),
        ready=ready,
        detail=detail,
    )


def _apply_loaded_inductor_config(threads: int, worker_start: str) -> None:
    """Update a loaded Inductor config and recycle a stale idle pool."""

    try:
        import torch._inductor.config as inductor_config
    except (ImportError, AttributeError):
        return

    current_threads = int(getattr(inductor_config, "compile_threads", threads) or threads)
    current_start = str(getattr(inductor_config, "worker_start_method", worker_start))
    if current_threads != threads or current_start != worker_start:
        try:
            from torch._inductor.async_compile import AsyncCompile, shutdown_compile_workers
        except (ImportError, AttributeError):
            pass
        else:
            shutdown_compile_workers()
            if AsyncCompile.pool.cache_info().currsize:
                AsyncCompile.pool().shutdown(wait=True)
                AsyncCompile.pool.cache_clear()

    inductor_config.compile_threads = threads
    inductor_config.worker_start_method = worker_start


__all__ = [
    "COMPILE_THREADS_ENV",
    "CompileWorkerSettings",
    "CompileWorkerWarmupHandle",
    "CompileWorkerWarmupResult",
    "DEFAULT_COMPILE_THREADS",
    "MAX_COMPILE_THREADS",
    "MIN_COMPILE_THREADS",
    "WORKER_START_ENV",
    "configure_compile_workers",
    "finish_compile_worker_warmup",
    "normalize_compile_threads",
    "prepare_compile_worker_env",
    "start_compile_worker_warmup",
]
