"""Environment-neutral configuration loading."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ProjectConfig:
    data_root: Path
    artifact_root: Path
    output_root: Path
    run_id: str = "default"
    seed: int = 2026
    batch_size: int = 100_000
    threads: int = 1
    device: str = "cpu"
    max_rows: int | None = None
    model: dict[str, Any] = field(default_factory=dict)
    blocking: dict[str, Any] = field(default_factory=dict)
    config_path: Path | None = None

    @classmethod
    def load(cls, path: str | Path, overrides: dict[str, Any] | None = None) -> "ProjectConfig":
        config_path = Path(path).expanduser().resolve()
        with config_path.open(encoding="utf-8") as handle:
            raw = json.load(handle)
        env_map = {
            "data_root": os.getenv("BER_DATA_ROOT"),
            "artifact_root": os.getenv("BER_ARTIFACT_ROOT"),
            "output_root": os.getenv("BER_OUTPUT_ROOT"),
            "run_id": os.getenv("BER_RUN_ID"),
            "device": os.getenv("BER_DEVICE"),
            "threads": os.getenv("BER_THREADS"),
        }
        raw.update({k: v for k, v in env_map.items() if v not in (None, "")})
        if overrides:
            raw.update({k: v for k, v in overrides.items() if v is not None})

        def path_value(name: str, default: str) -> Path:
            value = Path(raw.get(name, default)).expanduser()
            return value if value.is_absolute() else (Path.cwd() / value).resolve()

        return cls(
            data_root=path_value("data_root", "dataset"),
            artifact_root=path_value("artifact_root", "artifacts"),
            output_root=path_value("output_root", "output"),
            run_id=str(raw.get("run_id", "default")),
            seed=int(raw.get("seed", 2026)),
            batch_size=int(raw.get("batch_size", 100_000)),
            threads=int(raw.get("threads", 1)),
            device=str(raw.get("device", "cpu")),
            max_rows=None if raw.get("max_rows") is None else int(raw["max_rows"]),
            model=dict(raw.get("model", {})),
            blocking=dict(raw.get("blocking", {})),
            config_path=config_path,
        )

    def artifact_dir(self, stage: str) -> Path:
        return self.artifact_root / self.run_id / stage

    def as_dict(self) -> dict[str, Any]:
        return {
            "data_root": str(self.data_root),
            "artifact_root": str(self.artifact_root),
            "output_root": str(self.output_root),
            "run_id": self.run_id,
            "seed": self.seed,
            "batch_size": self.batch_size,
            "threads": self.threads,
            "device": self.device,
            "max_rows": self.max_rows,
            "model": self.model,
            "blocking": self.blocking,
        }

