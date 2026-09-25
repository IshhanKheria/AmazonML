"""Process-based parallelism for shard-bound stages (fork on Linux).

Stages are sharded by Source 1 / candidate shard, so each shard is independent
and can be processed in a separate process. On Linux the default ``fork`` start
method lets workers inherit large read-only inputs (prepared frames, fitted
blockers, embedding caches) copy-on-write instead of pickling them per task.

When ``fork`` is unavailable (macOS/Windows) or ``workers <= 1``, everything
falls back to a plain in-process loop, so behaviour is identical.
"""

from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from typing import Callable, Iterable, Iterator, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def fork_available() -> bool:
    return "fork" in mp.get_all_start_methods()


def _context():
    return mp.get_context("fork")


def parallel_map(func: Callable[[T], R], items: Iterable[T], workers: int) -> Iterator[R]:
    """Map over a small, already-materialised sequence; yield as tasks finish.

    Intended for shard stages where ``items`` is a list of lightweight keys.
    """
    materialised = list(items)
    if workers <= 1 or len(materialised) <= 1 or not fork_available():
        for item in materialised:
            yield func(item)
        return
    with ProcessPoolExecutor(max_workers=min(workers, len(materialised)), mp_context=_context()) as pool:
        futures = [pool.submit(func, item) for item in materialised]
        for future in _as_completed_bounded(futures):
            yield future.result()


def parallel_imap(func: Callable[[T], R], items: Iterable[T], workers: int, max_inflight: int | None = None) -> Iterator[R]:
    """Lazily map a large/streamed iterable, bounding in-flight work.

    The producer runs in the parent; at most ``max_inflight`` items are read and
    submitted at once, so memory stays bounded for stages like ``prepare``.
    """
    if workers <= 1 or not fork_available():
        for item in items:
            yield func(item)
        return
    max_inflight = max(workers, max_inflight or workers * 2)
    iterator = iter(items)
    pending: dict = {}
    with ProcessPoolExecutor(max_workers=workers, mp_context=_context()) as pool:

        def fill() -> None:
            while len(pending) < max_inflight:
                try:
                    item = next(iterator)
                except StopIteration:
                    return
                pending[pool.submit(func, item)] = True

        fill()
        while pending:
            done, _ = wait(list(pending), return_when=FIRST_COMPLETED)
            for future in done:
                pending.pop(future, None)
                yield future.result()
            fill()


def _as_completed_bounded(futures) -> Iterator:
    remaining = list(futures)
    while remaining:
        done, _ = wait(remaining, return_when=FIRST_COMPLETED)
        for future in done:
            remaining.remove(future)
            yield future
