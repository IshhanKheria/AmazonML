#!/usr/bin/env python3
"""Minimal deterministic executor for the generated code cells.

This avoids adding a notebook-runtime dependency to the parity gate. It is not a
replacement for Jupyter; it executes the exact Python source stored in each code
cell in one shared namespace, matching top-to-bottom kernel semantics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", type=Path)
    args = parser.parse_args()
    notebook = json.loads(args.notebook.read_text(encoding="utf-8"))
    namespace: dict[str, object] = {"__name__": "__main__"}
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        try:
            exec(compile(source, f"{args.notebook}#cell-{index}", "exec"), namespace)
        except Exception as exc:
            raise RuntimeError(f"notebook cell {index} failed") from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
