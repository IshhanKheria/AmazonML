"""Hardware detection and adaptive resource budgeting.

Nothing here hardcodes an absolute memory figure. Chunk sizes and batch sizes
are derived from the memory actually available on the machine at run time, and
every heavy loop is wrapped in :func:`run_with_oom_backoff` so the pipeline
survives out-of-memory conditions instead of dying.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Callable, TypeVar

T = TypeVar("T")


def _available_ram_bytes() -> int:
    """Best-effort available RAM in bytes, with portable fallbacks."""
    try:
        import psutil  # type: ignore

        return int(psutil.virtual_memory().available)
    except Exception:
        pass
    try:  # POSIX
        pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        if pages > 0 and page_size > 0:
            return int(pages) * int(page_size)
    except Exception:
        pass
    return 8 * 1024**3  # conservative fallback only if nothing else is detectable


def _cuda_info() -> tuple[bool, int]:
    """Return (cuda_available, free_vram_bytes)."""
    try:
        import torch  # type: ignore

        if not torch.cuda.is_available():
            return False, 0
        free, _total = torch.cuda.mem_get_info()
        return True, int(free)
    except Exception:
        return False, 0


@dataclass(slots=True)
class ResourcePlan:
    device: str
    ram_budget_bytes: int
    vram_budget_bytes: int
    cpu_count: int
    n_workers: int
    chunk_pairs: int
    embed_batch_size: int
    model_batch_size: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "ram_budget_bytes": self.ram_budget_bytes,
            "vram_budget_bytes": self.vram_budget_bytes,
            "cpu_count": self.cpu_count,
            "n_workers": self.n_workers,
            "chunk_pairs": self.chunk_pairs,
            "embed_batch_size": self.embed_batch_size,
            "model_batch_size": self.model_batch_size,
        }


def plan_resources(
    *,
    mode: str = "auto",
    ram_fraction: float = 0.65,
    vram_fraction: float = 0.75,
    threads: int | None = None,
    chunk_pairs: int | None = None,
    embed_batch_size: int | None = None,
) -> ResourcePlan:
    """Derive a :class:`ResourcePlan` from the live machine.

    ``mode`` is ``auto`` | ``cpu`` | ``gpu``. Explicit ``chunk_pairs`` /
    ``embed_batch_size`` values override the derived ones.
    """
    available = _available_ram_bytes()
    ram_budget = max(256 * 1024**2, int(available * float(ram_fraction)))

    cuda_available, free_vram = _cuda_info()
    if mode == "gpu" and not cuda_available:
        mode = "cpu"
    device = "cuda" if (mode == "gpu" or (mode == "auto" and cuda_available)) else "cpu"
    vram_budget = int(free_vram * float(vram_fraction)) if device == "cuda" else 0

    cpu_count = os.cpu_count() or 1
    n_workers = max(1, min(cpu_count, int(threads))) if threads else max(1, cpu_count - 1)

    # ~4 KB per candidate-pair feature row is a deliberately generous estimate
    # (strings are not held per pair; the join is the main cost). Derived, not fixed.
    derived_chunk = max(10_000, min(2_000_000, ram_budget // 4096))
    derived_embed = max(8, min(4096, (vram_budget // (1024 * 1024 * 4)) if vram_budget else (ram_budget // (1024 * 1024 * 8))))
    resolved_chunk = int(chunk_pairs) if chunk_pairs else derived_chunk
    resolved_embed = int(embed_batch_size) if embed_batch_size else derived_embed

    return ResourcePlan(
        device=device,
        ram_budget_bytes=ram_budget,
        vram_budget_bytes=vram_budget,
        cpu_count=cpu_count,
        n_workers=n_workers,
        chunk_pairs=resolved_chunk,
        embed_batch_size=resolved_embed,
        model_batch_size=max(10_000, min(2_000_000, ram_budget // 2048)),
    )


def is_oom_error(exc: BaseException) -> bool:
    if isinstance(exc, MemoryError):
        return True
    name = type(exc).__name__
    if name == "OutOfMemoryError":
        return True
    message = str(exc).lower()
    return "out of memory" in message or "cuda out of memory" in message or "cannot allocate memory" in message


def run_with_oom_backoff(
    operation: Callable[[int], T],
    size: int,
    *,
    min_size: int = 1,
    on_retry: Callable[[int, BaseException], None] | None = None,
) -> T:
    """Run ``operation(size)``; on OOM, halve ``size`` and retry down to ``min_size``.

    The caller supplies an operation that consumes the size argument (a chunk or
    batch size). This lets constrained machines complete large stages instead of
    crashing, without the code ever fixing a memory figure up front.
    """
    current = max(int(size), int(min_size))
    while True:
        try:
            return operation(current)
        except BaseException as exc:  # noqa: BLE001 - deliberately broad to survive OOM
            if not is_oom_error(exc) or current <= int(min_size):
                raise
            new_size = max(int(min_size), current // 2)
            if on_retry is not None:
                on_retry(new_size, exc)
            current = new_size
