#!/usr/bin/env python3
"""Generate the two self-contained, source-readable branch notebooks.

The source of truth is each pinned Git commit. Every relevant source/config file
is shown in a Markdown cell and also embedded as base64 in the following code
cell, which verifies SHA-256 before materializing the temporary runtime package.
"""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
BRANCHES = {
    "m1": {
        "sha": "e04336d23d6da988218ad1f5afe4abb413dd5472",
        "output": ROOT / "branch_m1" / "notebook" / "m1_reproducible.ipynb",
        "title": "m1 Reproducible Pipeline",
    },
    "codex/competition-hardening": {
        "sha": "a658723ef81f5c7c70fd15f720c2e440f50f4694",
        "output": ROOT / "branch_codex_competition_hardening" / "notebook" / "hardening_reproducible.ipynb",
        "title": "codex/competition-hardening Reproducible Pipeline",
    },
}

INCLUDE_PREFIXES = (
    "code/business_entity_resolution/src/",
    "code/business_entity_resolution/configs/",
    "code/business_entity_resolution/tests/",
)
INCLUDE_FILES = {
    "code/business_entity_resolution/run_pipeline.py",
    "code/business_entity_resolution/pyproject.toml",
    "code/business_entity_resolution/requirements.txt",
    "code/business_entity_resolution/requirements-remote.txt",
    "code/business_entity_resolution/requirements-embeddings.txt",
    "code/business_entity_resolution/THIRD_PARTY_LICENSES.md",
    "code/business_entity_resolution/README.md",
}


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={ROOT.as_posix()}", *args],
        cwd=ROOT,
        text=True,
        encoding="utf-8",
    )


def source_paths(sha: str) -> list[str]:
    paths = git("ls-tree", "-r", "--name-only", sha).splitlines()
    return sorted(
        path
        for path in paths
        if path in INCLUDE_FILES or any(path.startswith(prefix) for prefix in INCLUDE_PREFIXES)
    )


def source_at(sha: str, path: str) -> bytes:
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={ROOT.as_posix()}", "show", f"{sha}:{path}"],
        cwd=ROOT,
    )


def lines(text: str) -> list[str]:
    return text.splitlines(keepends=True) or [""]


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": lines(text)}


def code(text: str, tags: Iterable[str] = ()) -> dict:
    metadata = {"tags": list(tags)} if tags else {}
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": metadata,
        "outputs": [],
        "source": lines(text),
    }


def build(branch: str, spec: dict[str, object]) -> None:
    sha = str(spec["sha"])
    output = Path(spec["output"])
    cells: list[dict] = [
        markdown(
            f"# {spec['title']}\n\n"
            f"Pinned commit: `{sha}`  \n"
            "Experiment 0 label: **REPRODUCTION ONLY — NOT A COMPARATIVE SCORE**\n\n"
            "This notebook embeds the complete branch implementation. It materializes a run-scoped "
            "package and never imports the repository checkout. External embedding weights are disabled."
        ),
        code(
            "from __future__ import annotations\n"
            "import base64, hashlib, json, os, platform, shutil, subprocess, sys, tempfile, time\n"
            "from pathlib import Path\n\n"
            f"BRANCH_NAME = {branch!r}\n"
            f"BRANCH_SHA = {sha!r}\n"
            "SEED = int(os.environ.get('BER_SEED', '2026'))\n"
            "DATA_ROOT = Path(os.environ.get('BER_DATA_ROOT', 'dataset')).expanduser().resolve()\n"
            "RUN_ROOT = Path(os.environ.get('BER_RUN_ROOT', f'artifacts/notebook-{BRANCH_SHA[:8]}')).expanduser().resolve()\n"
            "OUTPUT_ROOT = Path(os.environ.get('BER_OUTPUT_ROOT', str(RUN_ROOT / 'output'))).expanduser().resolve()\n"
            "VALIDATOR_PATH = Path(os.environ.get('BER_VALIDATOR_PATH', 'utils/validate_submission.py')).expanduser().resolve()\n"
            "RUN_STAGES = {s.strip() for s in os.environ.get('BER_RUN_STAGES', 'inspect').split(',') if s.strip()}\n"
            "RUN_TEST = os.environ.get('BER_RUN_TEST', '0').lower() in {'1','true','yes'}\n"
            "INSTALL_DEPS = os.environ.get('BER_INSTALL_DEPS', '0').lower() in {'1','true','yes'}\n"
            "PACKAGE_ROOT = RUN_ROOT / '_embedded_package' / BRANCH_SHA\n"
            "SOURCE_ROOT = PACKAGE_ROOT / 'code' / 'business_entity_resolution'\n"
            "PACKAGE_ROOT.mkdir(parents=True, exist_ok=True)\n"
            "print({'branch': BRANCH_NAME, 'sha': BRANCH_SHA, 'data_root': str(DATA_ROOT), "
            "'run_root': str(RUN_ROOT), 'run_stages': sorted(RUN_STAGES), 'run_test': RUN_TEST})\n"
        ),
        code(
            "def materialize(relative_path: str, payload_b64: str, expected_sha256: str) -> Path:\n"
            "    payload = base64.b64decode(payload_b64.encode('ascii'))\n"
            "    actual = hashlib.sha256(payload).hexdigest()\n"
            "    if actual != expected_sha256:\n"
            "        raise RuntimeError(f'embedded source hash mismatch for {relative_path}: {actual}')\n"
            "    destination = PACKAGE_ROOT / relative_path\n"
            "    destination.parent.mkdir(parents=True, exist_ok=True)\n"
            "    destination.write_bytes(payload)\n"
            "    return destination\n"
        ),
    ]

    source_manifest = []
    for path in source_paths(sha):
        payload = source_at(sha, path)
        digest = hashlib.sha256(payload).hexdigest()
        source_manifest.append({"path": path, "bytes": len(payload), "sha256": digest})
        language = "python" if path.endswith(".py") else "text"
        rendered = payload.decode("utf-8")
        cells.append(markdown(f"## Embedded source: `{path}`\n\n~~~~{language}\n{rendered}\n~~~~"))
        encoded = base64.b64encode(payload).decode("ascii")
        cells.append(code(f"materialize({path!r}, {encoded!r}, {digest!r})"))

    cells.extend(
        [
            markdown("## Dependency installation and environment record"),
            code(
                "requirements = SOURCE_ROOT / 'requirements.txt'\n"
                "remote_requirements = SOURCE_ROOT / 'requirements-remote.txt'\n"
                "if INSTALL_DEPS:\n"
                "    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', str(requirements)])\n"
                "    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', str(remote_requirements)])\n"
                "environment = {'python': sys.version, 'platform': platform.platform(), 'branch': BRANCH_NAME, 'sha': BRANCH_SHA}\n"
                "RUN_ROOT.mkdir(parents=True, exist_ok=True)\n"
                "(RUN_ROOT / 'environment.json').write_text(json.dumps(environment, indent=2), encoding='utf-8')\n"
                "environment"
            ),
            markdown("## Import and source-integrity gate"),
            code(
                "sys.path.insert(0, str(SOURCE_ROOT / 'src'))\n"
                "from business_entity_resolution.cli import main as cli_main\n"
                "from business_entity_resolution.config import ProjectConfig\n"
                "from business_entity_resolution.data import load_source, load_ground_truth\n"
                "assert DATA_ROOT.joinpath('train', 'train_source1.tsv').is_file(), DATA_ROOT\n"
                "train_preview = load_source(DATA_ROOT, 'train', 1, nrows=5)\n"
                "print(train_preview.to_string(index=False))\n"
                "print('embedded package import and real training-data read: PASS')"
            ),
            markdown("## Runtime configuration"),
            code(
                "base = json.loads((SOURCE_ROOT / 'configs' / 'base.json').read_text(encoding='utf-8'))\n"
                "base.update({'data_root': str(DATA_ROOT), 'artifact_root': str(RUN_ROOT / 'artifacts'), "
                "'output_root': str(OUTPUT_ROOT), 'run_id': os.environ.get('BER_RUN_ID', 'notebook'), 'seed': SEED})\n"
                "base.setdefault('embeddings', {})['enable'] = 'false'\n"
                "base.setdefault('resources', {})['enable_embeddings'] = 'false'\n"
                "runtime_config = RUN_ROOT / 'runtime_config.json'\n"
                "runtime_config.write_text(json.dumps(base, indent=2), encoding='utf-8')\n"
                "runtime_config"
            ),
            markdown("## Branch-native stage runner"),
            code(
                "def run_cli(*arguments: str) -> None:\n"
                "    print('CLI:', ' '.join(arguments), flush=True)\n"
                "    result = int(cli_main(list(arguments)))\n"
                "    if result:\n"
                "        raise RuntimeError(f'branch CLI failed with exit code {result}: {arguments}')\n\n"
                "def run_functional_smoke() -> None:\n"
                "    # Native synthetic smoke covers normalization, blocking, features, SGD training, inference, and TSV formatting.\n"
                "    run_cli('smoke', '--config', str(runtime_config))\n\n"
                "def run_native_full() -> None:\n"
                "    if not RUN_TEST:\n"
                "        raise RuntimeError('Full native pipeline includes test inference; set BER_RUN_TEST=1 only after freeze')\n"
                "    env = os.environ.copy()\n"
                "    env.update({'BER_DATA_ROOT': str(DATA_ROOT), 'BER_ARTIFACT_ROOT': str(RUN_ROOT / 'artifacts'), "
                "'BER_OUTPUT_ROOT': str(OUTPUT_ROOT), 'BER_RUN_ID': 'native-full'})\n"
                "    subprocess.check_call([sys.executable, str(SOURCE_ROOT / 'run_pipeline.py'), '--profile', 'full', "
                "'--skip-embeddings', '--validator', str(VALIDATOR_PATH)], env=env)\n\n"
                "if 'functional' in RUN_STAGES:\n"
                "    run_functional_smoke()\n"
                "if 'native-full' in RUN_STAGES:\n"
                "    run_native_full()"
            ),
            markdown(
                "## Common evaluation contract\n\n"
                "Authoritative comparison is executed by the shared, branch-isolated runner using the same fold manifest. "
                "The runner calls only this embedded branch package for normalization, blocking, features, models, and decisions. "
                "It never creates candidates. Experiment 0 output remains **REPRODUCTION ONLY — NOT A COMPARATIVE SCORE**."
            ),
            markdown("## Frozen test inference and submission validation"),
            code(
                "def require_test_unlock() -> dict:\n"
                "    unlock = RUN_ROOT / 'FROZEN_TEST_UNLOCK.json'\n"
                "    if not RUN_TEST or not unlock.is_file():\n"
                "        raise RuntimeError('test inference remains locked until RUN_TEST=1 and FROZEN_TEST_UNLOCK.json exists')\n"
                "    payload = json.loads(unlock.read_text(encoding='utf-8'))\n"
                "    if payload.get('branch_sha') != BRANCH_SHA:\n"
                "        raise RuntimeError('test unlock SHA does not match embedded branch')\n"
                "    return payload\n\n"
                "if 'test' in RUN_STAGES:\n"
                "    require_test_unlock()\n"
                "    run_native_full()"
            ),
            markdown("## Experiment manifest"),
            code(
                f"embedded_manifest = {json.dumps(source_manifest, sort_keys=True)!r}\n"
                "embedded_manifest = json.loads(embedded_manifest)\n"
                "manifest = {'branch': BRANCH_NAME, 'branch_sha': BRANCH_SHA, 'seed': SEED, "
                "'data_root': str(DATA_ROOT), 'run_root': str(RUN_ROOT), 'test_unlocked': RUN_TEST, "
                "'source_files': embedded_manifest, 'experiment_0_label': 'REPRODUCTION ONLY — NOT A COMPARATIVE SCORE'}\n"
                "manifest_path = RUN_ROOT / 'notebook_manifest.json'\n"
                "manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding='utf-8')\n"
                "print(manifest_path)"
            ),
        ]
    )

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
            "ber": {"branch": branch, "commit": sha, "self_contained": True},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {output} ({output.stat().st_size} bytes, {len(cells)} cells)")


def main() -> int:
    for branch, spec in BRANCHES.items():
        build(branch, spec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
