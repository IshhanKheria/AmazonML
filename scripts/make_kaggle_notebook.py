#!/usr/bin/env python3
"""Generate notebooks/kaggle_run.ipynb — a self-contained Kaggle runner.

Run from the repo root:  python scripts/make_kaggle_notebook.py
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).resolve().parents[1] / "notebooks" / "kaggle_run.ipynb"

MARKDOWN_INTRO = """# Amazon ML 2026 — Business Entity Resolution (Kaggle runner)

Runs the pipeline on Kaggle and writes `matching_results.tsv` + `candidate_pairs.tsv`.

## Before running
1. **Accelerator:** None (CPU).
2. **Internet:** Settings → Internet **ON** (needed for the public repo, git-lfs, and pip).
3. **Data:** either attach the TSVs as a Kaggle Dataset (recommended), or let this
   notebook fetch them from the public repo via Git LFS.
4. **Runtime:** the full run takes several hours on ~4 vCPU / 30 GB. Run the mini
   cell first to sanity-check, and see the notes at the end to subsample.
"""

CELL_INSTALL = """!pip install -q lightgbm==4.6.0 rapidfuzz==3.12.2 psutil==6.1.0
import sys, os, glob, subprocess
print("python", sys.version.split()[0], "| cpus", os.cpu_count())
"""

CELL_CLONE = """REPO_URL = "https://github.com/IshhanKheria/AmazonML.git"
BRANCH   = "linux"          # <- branch with the latest code; data LFS lives on main
WORK     = "/kaggle/working"
REPO     = os.path.join(WORK, "AmazonML")

if not os.path.isdir(REPO):
    subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH, REPO_URL, REPO], check=True)
print("repo:", REPO)
print(sorted(os.listdir(REPO))[:15])
"""

CELL_DATA = """# Prefer an attached Kaggle Dataset; otherwise fetch the real TSVs via Git LFS.
attached = sorted(glob.glob("/kaggle/input/**/train/train_source1.tsv", recursive=True))
if attached:
    DATA_ROOT = os.path.dirname(os.path.dirname(attached[0]))
    print("using attached dataset:", DATA_ROOT)
else:
    if subprocess.run(["git", "lfs", "version"], capture_output=True).returncode != 0:
        subprocess.run(
            "curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash",
            shell=True, check=True,
        )
        subprocess.run(["sudo", "apt-get", "install", "-y", "git-lfs"], check=True)
    subprocess.run(["bash", "scripts/fetch_data.sh"], cwd=REPO, check=True)
    DATA_ROOT = os.path.join(REPO, "dataset")

assert os.path.isfile(os.path.join(DATA_ROOT, "train", "train_source1.tsv")), DATA_ROOT
print("DATA_ROOT =", DATA_ROOT)
"""

CELL_ENV = """ENV = os.environ.copy()
CPUS = max(1, (os.cpu_count() or 2) - 1)
ENV.update({
    "SKIP_SETUP": "1",
    "SKIP_EMBED": "1",                 # CPU run: skip BGE-M3 download/encoding
    "BER_DATA_ROOT": DATA_ROOT,
    "BER_ARTIFACT_ROOT": os.path.join(WORK, "artifacts"),
    "BER_OUTPUT_ROOT": os.path.join(WORK, "output"),
    "BER_THREADS": str(CPUS),
    "BER_WORKERS": str(CPUS),          # set to "1" if the parallel stages hang
})
print({k: ENV[k] for k in ("BER_THREADS", "BER_WORKERS", "BER_DATA_ROOT", "BER_OUTPUT_ROOT")})
"""

CELL_MINI = """# Quick sanity check (15 records) — seconds. Optional for the full run.
subprocess.run([sys.executable, "run_pipeline.py", "--profile", "mini"], cwd=REPO, env=ENV, check=True)
"""

CELL_FULL = """# Full run. Checkpointed and resumable: safe to re-run after a timeout.
subprocess.run([sys.executable, "run_pipeline.py", "--profile", "full"], cwd=REPO, env=ENV, check=True)
"""

CELL_RESULTS = """out = ENV["BER_OUTPUT_ROOT"]
for name in ("matching_results.tsv", "candidate_pairs.tsv"):
    path = os.path.join(out, name)
    with open(path, encoding="utf-8") as handle:
        lines = sum(1 for _ in handle)
    print(f"{name}: {lines} lines")
print("--- matching_results.tsv (head) ---")
print(open(os.path.join(out, "matching_results.tsv"), encoding="utf-8").read(300))
"""

MARKDOWN_NOTES = """## Notes & limits

- **Timeout:** Kaggle CPU sessions cap at ~12 h. The pipeline is checkpointed, so
  re-running this notebook resumes. If you keep timing out, subsample by setting
  `"max_s1_rows": 200000` in `configs/remote_full.json` (Source 1 only; targets stay full).
- **Memory:** 30 GB is tight for the full corpus. If candidates are OOM-killed, lower
  `per_source_cap` / `tfidf_top_k` in `configs/remote_full.json`.
- **Hangs:** if a parallel stage freezes the kernel (fork + Jupyter), set
  `ENV["BER_WORKERS"] = "1"` and re-run — serial but safe.
- **Embeddings:** disabled here (`SKIP_EMBED=1`). Turn on with a GPU only if you accept
  the download/encode time.
- **Outputs** land in `/kaggle/working/output/`; download them from the notebook's
  Output tab, or run the `package` command to produce the submission zip.
"""


def main() -> None:
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
        "language_info": {"name": "python"},
    }
    notebook["cells"] = [
        nbf.v4.new_markdown_cell(MARKDOWN_INTRO),
        nbf.v4.new_code_cell(CELL_INSTALL),
        nbf.v4.new_code_cell(CELL_CLONE),
        nbf.v4.new_code_cell(CELL_DATA),
        nbf.v4.new_code_cell(CELL_ENV),
        nbf.v4.new_code_cell(CELL_MINI),
        nbf.v4.new_code_cell(CELL_FULL),
        nbf.v4.new_code_cell(CELL_RESULTS),
        nbf.v4.new_markdown_cell(MARKDOWN_NOTES),
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, str(OUT))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
