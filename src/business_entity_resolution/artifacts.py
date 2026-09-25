"""Artifact persistence, fingerprints, and manifests."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any
import pandas as pd


def sha256_file(path: str | Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def config_fingerprint(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_frame(frame: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, engine="pyarrow")
    os.replace(temporary, path)
    return path


def read_frame(path: str | Path, columns: list[str] | None = None) -> pd.DataFrame:
    return pd.read_parquet(Path(path), columns=columns, engine="pyarrow")


def write_manifest(path: str | Path, *, stage: str, config: dict[str, Any], rows: int, schema: list[str], inputs: list[str] | None = None, metrics: dict[str, Any] | None = None, started_at: float | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Merge with any manifest a ShardStore already wrote (which carries the
    # input `fingerprint`, `parts`, etc.), so stage completion stays detectable.
    existing: dict[str, Any] = {}
    if path.is_file():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = {}
    manifest = {
        **existing,
        "stage": stage,
        "config_fingerprint": config_fingerprint(config),
        "config": config,
        "rows": int(rows),
        "schema": schema,
        "inputs": inputs or [],
        "metrics": metrics or {},
        "python": sys.version,
        "platform": platform.platform(),
        "started_at_epoch": started_at,
        "completed_at_epoch": time.time(),
        "complete": True,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8")
    os.replace(temporary, path)
    return path

