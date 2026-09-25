# Amazon ML 2026 Business Entity Resolution

This package implements a reproducible pipeline for linking every Source 1 entity to zero, one, or multiple Source 2/Source 3 records. It uses only the supplied challenge data and performs no external business lookup, geocoding, registry access, or internet augmentation.

## Robustness guarantees

- **Crash-safe resume.** Every long stage (prepare, candidates, embeddings, features) processes work in shards and commits each shard atomically (temp file + `os.replace`) with a config/input fingerprint. Re-running the same command skips completed shards and resumes from the last committed one — no full restart. Use `--force` to redo a stage.
- **Hardware-adaptive resources.** `resources.py` detects available RAM, CPU count, and CUDA/VRAM at run time and derives chunk/batch sizes from them. Nothing is hardcoded.
- **Bounded pair processing.** Candidate queries, pair features, scoring, and output writing are sharded. Target blocker indexes remain source-sized and must be capacity-tested before a full run.
- **Continuous logging.** Each stage writes to stdout and to `artifacts/<run_id>/logs/<stage>.log`, flushed per line, including per-shard progress, ETA, peak memory, and LightGBM training metrics.
- **Format guarantee.** Before writing, outputs pass an in-process gate (one row per test S1, no dupes, no whitespace, valid prefixes, matches ⊆ candidates). `write-output` can regenerate formatting from cached scores/candidates without re-running the model.

## Architecture

```text
Raw TSV
  → strict audit and schema validation
  → raw-preserving normalization (US / India / France legal suffixes)
  → leakage-safe S1 folds
  → complementary candidate generation (7-way union, resumable)
  → optional experimental embedding features (disabled by default)
  → candidate-only pair features (39 features)
  → deterministic / SGD / optional LightGBM scorer
  → macro-F0.5 threshold tuning on complete held-out S1 folds
  → format-gated matching_results.tsv + candidate_pairs.tsv
  → internal preflight + official validator
```

The final candidate table is the exact table passed to feature extraction and scoring. It is also the source for `candidate_pairs.tsv`.

## Models and licenses

- LightGBM (`lightgbm==4.6.0`), MIT licensed.
- The optional BGE-M3 integration is experimental and disabled in every supplied profile. Do not enable it unless the organizers explicitly confirm that pretrained representations satisfy the supplied-data-only rule.
- `country` is treated strictly as an open-set string label; it is never hardcoded.


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

Provision a fresh machine with the helper script (detects GPU, picks the right torch wheel):

```bash
bash setup.sh                      # compliant text-feature pipeline
ENABLE_EMBED=1 bash setup.sh       # experimental; requires organizer approval
```

Or install manually from `code/business_entity_resolution/`:

```bash
python -m pip install -e .
python -m pip install -r requirements-dev.txt
python -m pip install -r requirements-remote.txt   # LightGBM backend
```

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

`resources` and `embeddings` config blocks control device mode (`auto`/`cpu`/`gpu`),
the RAM/VRAM fractional budgets, and whether BGE-M3 features are enabled
(`true`/`false`; supplied profiles use `false`). `n_shards` controls checkpoint granularity.

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

## Mini end-to-end cycle (run this first)

`make-mini` samples 15 realistic Source 1 entities (6 US / 6 India / 3 France,
including singletons, single matches, multi-matches, and both-source matches)
plus decoys, straight from the real files, into `dataset/mini/` with the exact
real schema. Use it to verify the whole cycle end-to-end before the full run.

```bash
python run_pipeline.py --profile mini --skip-embeddings \
  --validator /path/to/student_resource/utils/validate_submission.py
```

The runner tunes the decision threshold on fold 0, reports an unbiased estimate
on a separately trained fold-1 model, then trains the final model on all rows.

Any of these commands can be re-run after a crash and will resume.

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
python -m business_entity_resolution train --config configs/remote_full.json --exclude-folds 0,1 --output-name lightgbm-validation-fold0
python -m business_entity_resolution score --config configs/remote_full.json --split train --model-path /mounted/work/artifacts/remote-full/models/lightgbm-validation-fold0.txt --validation-fold 0 --output-name validation-fold0.parquet
python -m business_entity_resolution tune-decision --config configs/remote_full.json --score-file validation-fold0.parquet --validation-fold 0

# Independent evaluation: train without fold 1, then score/evaluate fold 1.
python -m business_entity_resolution train --config configs/remote_full.json --validation-fold 1 --output-name lightgbm-validation-fold1
python -m business_entity_resolution score --config configs/remote_full.json --split train --model-path /mounted/work/artifacts/remote-full/models/lightgbm-validation-fold1.txt --validation-fold 1 --output-name evaluation-fold1.parquet
python -m business_entity_resolution evaluate --config configs/remote_full.json --score-file evaluation-fold1.parquet --validation-fold 1
```

After selecting and freezing a validated model and threshold:

```bash
python -m business_entity_resolution train --config configs/remote_full.json --all-training-data --output-name lightgbm-final
python -m business_entity_resolution prepare --config configs/remote_full.json --split test
python -m business_entity_resolution generate-candidates --config configs/remote_full.json --split test
python -m business_entity_resolution build-features --config configs/remote_full.json --split test
python -m business_entity_resolution infer --config configs/remote_full.json \
    --model-path /mounted/work/artifacts/remote-full/models/lightgbm-final.txt
python -m business_entity_resolution preflight --config configs/remote_full.json \
    --official-validator /mounted/challenge/utils/validate_submission.py --check-ids
python -m business_entity_resolution package --config configs/remote_full.json --team-name YOUR_TEAM \
    --documentation /mounted/challenge/Documentation_template.md
```

Keep embedding features disabled for competition runs unless the supplied-data-only and model-license interpretation is confirmed in writing.

Complete the supplied documentation template before packaging. The package command refuses to create a submission archive without it.

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
- Supplied challenge data only in the default and recommended workflows; no external address/entity lookup.
- LightGBM is optional and MIT licensed; any use must be recorded in the run manifest.
- A final model is not selected until held-out experiments run on remote compute.
