# Amazon ML 2026 Business Entity Resolution

This package implements a reproducible pipeline for linking every Source 1 entity to zero, one, or multiple Source 2/Source 3 records. It uses only the supplied challenge data and performs no external business lookup, geocoding, registry access, or internet augmentation.

Current state: complete codebase, unit-tested and smoke-tested on synthetic data. No full-scale model has been trained and no final test inference has been run.

## Architecture

```text
Raw TSV
  → strict audit and schema validation
  → raw-preserving normalization
  → leakage-safe S1 folds
  → complementary candidate generation
  → candidate-only pair features
  → deterministic / SGD / optional LightGBM scorer
  → configurable decision thresholds
  → matching_results.tsv + candidate_pairs.tsv
  → internal preflight + official validator
```

The final candidate table is the exact table passed to feature extraction and scoring. It is also the source for `candidate_pairs.tsv`.

## Directory layout

```text
src/business_entity_resolution/  Production package and CLI
tests/                           Unit and integration tests
configs/                         Portable local/remote JSON configurations
notebooks/                       Thin exploratory clients of the package
requirements.txt                 Core runtime dependencies
requirements-dev.txt             Test and notebook dependencies
requirements-remote.txt          Optional full-scale LightGBM backend
```

Generated artifacts are written to configured `artifacts/` and `output/` roots.

## Installation

From `code/business_entity_resolution/`:

```bash
python -m pip install -e .
python -m pip install -r requirements-dev.txt
```

For the optional MIT-licensed LightGBM backend:

```bash
python -m pip install -r requirements-remote.txt
```

No GPU is required. CPU is the default and reference execution mode.

## Configuration

Precedence:

```text
CLI override > BER_* environment variable > JSON config > default
```

Environment variables:

```text
BER_DATA_ROOT
BER_ARTIFACT_ROOT
BER_OUTPUT_ROOT
BER_RUN_ID
BER_DEVICE
BER_THREADS
```

The pipeline never contains a username, drive letter, Colab path, or AWS-specific path. Point these variables at local disk, a mounted Colab drive, EBS/EFS, or another mounted filesystem.

## Input contract

```text
dataset/
├── train/
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   └── train_ground_truth.tsv
└── test/
    ├── test_source1.tsv
    ├── test_source2.tsv
    └── test_source3.tsv
```

All files are UTF-8 and tab-separated. Blank strings are preserved. Country is an arbitrary open-set string; no fixed `{US, India}` enumeration exists.

## Local verification

PowerShell, from the package directory:

```powershell
$env:PYTHONPATH = (Resolve-Path '.\src')
python -m pytest -q
python -m business_entity_resolution smoke --config '.\configs\local_smoke.json'
```

Bounded audit of the actual challenge files:

```powershell
$env:BER_DATA_ROOT = '<workspace>/6ab10eb3b23ba_student_resource/student_resource/dataset'
$env:BER_ARTIFACT_ROOT = '<workspace>/artifacts'
$env:PYTHONPATH = (Resolve-Path '.\src')
python -m business_entity_resolution audit --config '.\configs\local_smoke.json' --max-rows 50000
```

The smoke command creates a four-S1 synthetic dataset, performs blocking, builds features, trains a tiny SGD model, runs inference, writes both required TSVs, and runs strict internal preflight. Its metric is a software smoke result, not a competition score.

## Remote full-scale workflow

Use the same source code on a machine with 16–32 vCPU, 64–128 GB RAM, and fast temporary storage:

```bash
export BER_DATA_ROOT=/mounted/challenge/dataset
export BER_ARTIFACT_ROOT=/mounted/work/artifacts
export BER_OUTPUT_ROOT=/mounted/work/output
export BER_THREADS=16

python -m business_entity_resolution audit --config configs/remote_full.json
python -m business_entity_resolution prepare --config configs/remote_full.json --split both
python -m business_entity_resolution make-splits --config configs/remote_full.json --folds 10
python -m business_entity_resolution generate-candidates --config configs/remote_full.json --split train
python -m business_entity_resolution build-features --config configs/remote_full.json --split train
python -m business_entity_resolution train --config configs/remote_full.json --model sgd --validation-fold 0
python -m business_entity_resolution score --config configs/remote_full.json --split train --model sgd --validation-fold 0
python -m business_entity_resolution tune-decision --config configs/remote_full.json
python -m business_entity_resolution evaluate --config configs/remote_full.json
```

After selecting and freezing a validated model and threshold:

```bash
python -m business_entity_resolution train --config configs/remote_full.json --model sgd --all-training-data
python -m business_entity_resolution prepare --config configs/remote_full.json --split test
python -m business_entity_resolution generate-candidates --config configs/remote_full.json --split test
python -m business_entity_resolution build-features --config configs/remote_full.json --split test
python -m business_entity_resolution infer --config configs/remote_full.json --model sgd --model-path /mounted/work/artifacts/remote-full/models/sgd.joblib
python -m business_entity_resolution preflight --config configs/remote_full.json --official-validator /mounted/challenge/utils/validate_submission.py --check-ids
python -m business_entity_resolution package --config configs/remote_full.json --team-name YOUR_TEAM --documentation /mounted/challenge/Documentation_template.md
```

Complete the supplied documentation template before packaging. The package command refuses to create a submission archive without it.

Use `--model lightgbm` only after installing `requirements-remote.txt` and validating it on held-out data.

## Outputs

```text
output/
├── matching_results.tsv
└── candidate_pairs.tsv
```

Both files contain exactly one row per test S1, including blank rows. Internal preflight enforces target existence, duplicate rules, candidate containment, exact columns, and empty-list handling before the official validator runs.

## Reproducibility and compliance

- Seed defaults to 2026.
- Folds and negative sampling use stable cryptographic hashes.
- Candidate ordering and output lists are deterministic.
- Stage artifacts use explicit schemas and manifests.
- Raw challenge files are never modified.
- Supplied challenge data only; no pretrained models or external address/entity data.
- LightGBM is optional and MIT licensed; any use must be recorded in the run manifest.
- A final model is not selected until held-out experiments run on remote compute.
