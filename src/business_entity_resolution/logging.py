"""Continuous, crash-resilient run logging for long stages.

Every logger writes to stdout *and* to a per-stage log file, flushed on every
record so that a crash or kill leaves the tail of the log on disk. Progress,
resource plans, per-shard rows/elapsed/ETA/peak memory, and model metrics
(training loss, AUC, threshold sweeps) all flow through here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import time
from typing import Any


def _peak_rss_bytes() -> int:
    try:
        import resource  # type: ignore

        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    except Exception:
        pass
    try:
        import psutil  # type: ignore

        return int(psutil.Process().memory_info().rss)
    except Exception:
        return 0


def _format_bytes(value: int) -> str:
    size = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.2f}{unit}"
        size /= 1024
    return f"{size:.2f}TB"


class StageLogger:
    """A tiny append-only logger with stdout mirroring and flush-per-line."""

    def __init__(self, path: str | Path | None = None, name: str = "run", echo: bool = True) -> None:
        self.name = name
        self.echo = echo
        self.path = Path(path) if path is not None else None
        self.started = time.time()
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def _emit(self, line: str) -> None:
        stamped = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {line}"
        if self.echo:
            print(stamped, flush=True)
        if self.path is not None:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(stamped + "\n")
                handle.flush()

    def info(self, message: str, **fields: Any) -> None:
        self._emit(self._compose("INFO", message, fields))

    def event(self, event: str, **fields: Any) -> None:
        self._emit(self._compose("EVENT", event, fields))

    def metric(self, name: str, value: Any, **fields: Any) -> None:
        self._emit(self._compose("METRIC", name, {"value": value, **fields}))

    def progress(self, stage: str, done: int, total: int, extra: dict[str, Any] | None = None) -> None:
        elapsed = time.time() - self.started
        rate = done / elapsed if elapsed > 0 and done > 0 else 0.0
        eta = (total - done) / rate if rate > 0 else float("inf")
        fields: dict[str, Any] = {
            "done": done,
            "total": total,
            "elapsed_s": round(elapsed, 1),
            "eta_s": round(eta, 1) if eta != float("inf") else None,
            "rate_per_s": round(rate, 2),
            "peak_rss": _format_bytes(_peak_rss_bytes()),
        }
        if extra:
            fields.update(extra)
        self._emit(self._compose("PROGRESS", stage, fields))

    def resource_plan(self, plan: Any) -> None:
        data = plan.as_dict() if hasattr(plan, "as_dict") else dict(plan)
        if "ram_budget_bytes" in data:
            data["ram_budget"] = _format_bytes(int(data.pop("ram_budget_bytes")))
        if "vram_budget_bytes" in data:
            data["vram_budget"] = _format_bytes(int(data.pop("vram_budget_bytes")))
        self._emit(self._compose("RESOURCES", "plan", data))

    def stage_start(self, stage: str, **fields: Any) -> None:
        self.started = time.time()
        self._emit(self._compose("STAGE_START", stage, fields))

    def stage_end(self, stage: str, **fields: Any) -> None:
        fields.setdefault("elapsed_s", round(time.time() - self.started, 1))
        fields.setdefault("peak_rss", _format_bytes(_peak_rss_bytes()))
        self._emit(self._compose("STAGE_END", stage, fields))

    def _compose(self, level: str, message: str, fields: dict[str, Any]) -> str:
        suffix = " ".join(f"{key}={_stringify(value)}" for key, value in fields.items() if value is not None)
        return f"[{self.name}][{level}] {message}" + (f" | {suffix}" if suffix else "")


def _stringify(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


@dataclass(slots=True)
class NullLogger:
    """No-op logger so call sites never branch on availability."""

    name: str = "null"
    echo: bool = False

    def info(self, *_a: Any, **_k: Any) -> None: ...
    def event(self, *_a: Any, **_k: Any) -> None: ...
    def metric(self, *_a: Any, **_k: Any) -> None: ...
    def progress(self, *_a: Any, **_k: Any) -> None: ...
    def resource_plan(self, *_a: Any, **_k: Any) -> None: ...
    def stage_start(self, *_a: Any, **_k: Any) -> None: ...
    def stage_end(self, *_a: Any, **_k: Any) -> None: ...


def stage_logger(config: Any, stage: str, echo: bool = True) -> StageLogger:
    """Create a logger rooted at ``<artifact_root>/<run_id>/logs/<stage>.log``."""
    try:
        path = config.artifact_dir("logs") / f"{stage}.log"
    except Exception:
        path = None
    return StageLogger(path, name=stage, echo=echo and sys.stdout is not None)
