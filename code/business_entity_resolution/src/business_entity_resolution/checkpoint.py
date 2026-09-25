"""Crash-safe, idempotent shard checkpointing for long pipeline stages.

A stage processes deterministic shards (by S1-ID range or input part index).
Each shard is written atomically to ``parts/part-XXXXX.parquet`` together with a
``parts/part-XXXXX.manifest.json`` carrying the config fingerprint, an input
fingerprint, the shard key, and the row count. Re-running a stage skips every
shard whose manifest matches the current fingerprint(s), so a crash or kill only
costs the in-flight shard. A top-level ``manifest.json`` with ``complete=true``
marks the stage finished.
"""

from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
from typing import Any, Iterable, Iterator

import pandas as pd

from .artifacts import config_fingerprint, read_frame, write_frame


def shard_for_id(entity_id: str, n_shards: int) -> int:
    """Deterministically map an entity ID to a shard in ``[0, n_shards)``.

    Uses the numeric suffix when present (stable, order-independent) and falls
    back to a hash of the raw string otherwise.
    """
    if n_shards <= 1:
        return 0
    digits = entity_id.split("-", 1)[-1]
    if digits.isdigit():
        return int(digits) % n_shards
    digest = hashlib.blake2b(str(entity_id).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % n_shards


def shard_ids(ids: Iterable[str], n_shards: int) -> dict[int, list[str]]:
    buckets: dict[int, list[str]] = {index: [] for index in range(n_shards)}
    for entity_id in ids:
        buckets[shard_for_id(str(entity_id), n_shards)].append(str(entity_id))
    return buckets


def input_fingerprint(paths: Iterable[str | Path]) -> str:
    """Fingerprint a set of input files by path, size, and mtime."""
    entries = []
    for path in sorted(Path(p) for p in paths):
        try:
            stat = path.stat()
            entries.append((str(path), int(stat.st_size), int(stat.st_mtime)))
        except OSError:
            entries.append((str(path), -1, -1))
    return config_fingerprint({"inputs": entries})


def stage_fingerprint(stage_dir: str | Path) -> str:
    """Return the fingerprint recorded in a stage manifest, or ``""`` if absent.

    Readers use this so they can attach to whatever fingerprint the writer
    actually used (which may depend on options like embeddings being enabled),
    without having to recompute every variant.
    """
    path = Path(stage_dir) / "manifest.json"
    if not path.is_file():
        return ""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    return str(data.get("fingerprint", ""))


class ShardStore:
    """Manage resumable, fingerprinted shards for one stage directory."""

    def __init__(self, stage_dir: str | Path, fingerprint: str) -> None:
        self.stage_dir = Path(stage_dir)
        self.parts_dir = self.stage_dir / "parts"
        self.fingerprint = fingerprint
        self.parts_dir.mkdir(parents=True, exist_ok=True)

    # -- paths -----------------------------------------------------------------
    def part_path(self, key: int) -> Path:
        return self.parts_dir / f"part-{int(key):05d}.parquet"

    def manifest_path(self, key: int) -> Path:
        return self.parts_dir / f"part-{int(key):05d}.manifest.json"

    def stage_manifest_path(self) -> Path:
        return self.stage_dir / "manifest.json"

    # -- introspection ---------------------------------------------------------
    def completed_keys(self) -> set[int]:
        done: set[int] = set()
        for manifest_path in self.parts_dir.glob("part-*.manifest.json"):
            try:
                data = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if data.get("complete") and data.get("fingerprint") == self.fingerprint and data.get("key") is not None:
                if self.part_path(int(data["key"])).is_file():
                    done.add(int(data["key"]))
        return done

    def pending_keys(self, all_keys: Iterable[int]) -> list[int]:
        done = self.completed_keys()
        return sorted(int(key) for key in all_keys if int(key) not in done)

    def is_complete(self) -> bool:
        path = self.stage_manifest_path()
        if not path.is_file():
            return False
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return bool(data.get("complete")) and data.get("fingerprint") == self.fingerprint

    # -- mutation --------------------------------------------------------------
    def commit(self, key: int, frame: pd.DataFrame, *, rows: int | None = None, extra: dict[str, Any] | None = None) -> Path:
        part = write_frame(frame, self.part_path(key))
        manifest = {
            "key": int(key),
            "fingerprint": self.fingerprint,
            "rows": int(len(frame) if rows is None else rows),
            "schema": list(frame.columns),
            "complete": True,
        }
        if extra:
            manifest.update(extra)
        temporary = self.manifest_path(key).with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8")
        os.replace(temporary, self.manifest_path(key))
        return part

    def mark_complete(self, rows: int, schema: list[str], extra: dict[str, Any] | None = None) -> Path:
        manifest: dict[str, Any] = {
            "fingerprint": self.fingerprint,
            "rows": int(rows),
            "schema": list(schema),
            "parts": len(list(self.parts_dir.glob("part-*.parquet"))),
            "complete": True,
        }
        if extra:
            manifest.update(extra)
        path = self.stage_manifest_path()
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8")
        os.replace(temporary, path)
        return path

    def read_all(self, columns: list[str] | None = None) -> pd.DataFrame:
        paths = sorted(self.parts_dir.glob("part-*.parquet"))
        if not paths:
            return pd.DataFrame(columns=columns) if columns else pd.DataFrame()
        return pd.concat([read_frame(path, columns=columns) for path in paths], ignore_index=True)

    def iter_parts(self) -> Iterator[Path]:
        yield from sorted(self.parts_dir.glob("part-*.parquet"))

    def total_rows(self) -> int:
        total = 0
        for manifest_path in self.parts_dir.glob("part-*.manifest.json"):
            try:
                data = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if data.get("fingerprint") == self.fingerprint:
                total += int(data.get("rows", 0))
        return total
