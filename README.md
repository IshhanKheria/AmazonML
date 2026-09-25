# Amazon ML 2026 Business Entity Resolution

This repository implements a reproducible pipeline for linking every Source 1 entity to zero, one, or multiple Source 2/Source 3 records. It uses only the supplied challenge data and performs no external business lookup, geocoding, registry access, or internet augmentation.

Optimization target: entity-level macro **F0.5** (precision-heavy).

## Quickstart

```bash
make setup     # create .venv and install dependencies
make data      # fetch the real TSVs via git-lfs and verify them
make mini      # 15-record end-to-end sanity run
make full      # the real dataset
make help      # list every target
```

`make` is a thin, self-documenting wrapper around the CLI. You can always call the
pipeline directly (see [CLI stages](#cli-stages)).

## Repository layout

```text
.
├── Makefile                     # one command surface (setup/data/mini/full/test/...)
├── pyproject.toml               # packaging + pytest config
├── requirements*.txt            # pinned core/dev/remote dependencies
├── setup.sh  run.sh  run_pipeline.py
├── configs/                     # single source of truth for run configuration
│   └── project.json  base.json  mini.json  local_smoke.json  remote_full.json
├── src/business_entity_resolution/   # production package and CLI
├── tests/                       # unit and integration tests
├── scripts/
│   ├── fetch_data.sh            # git-lfs fetch + pointer verification
│   └── doctor.sh                # environment health check
├── dataset/                     # working data root (real TSVs once fetched)
├── 6ab10eb3b23ba_student_resource/student_resource/   # original bundle (read-only)
├── artifacts/                   # generated intermediate data/models
├── output/                      # required TSV outputs
├── reports/  experiments/  docs/
└── notebooks/                   # thin exploratory clients
```

`make package` assembles the challenge-required
`code/business_entity_resolution/{src,README.md,requirements.txt}` layout inside the
submission zip, so the repo itself can stay flat.

## Data

The real challenge files are tracked with **Git LFS** on `main`. Fetch them into the
working `dataset/` root:

```bash
make data                       # wraps scripts/fetch_data.sh
BER_DATA_ROOT=/mounted/dataset make full   # or point elsewhere
```

`scripts/fetch_data.sh` runs `git lfs fetch origin main` for the seven dataset
paths, writes the real TSVs into `dataset/{train,test}/`, and verifies the files
are not Git-LFS pointer stubs. It requires `git-lfs` to be installed.

The original bundle under `6ab10eb3b23ba_student_resource/` is read-only; the pipeline
accepts their location via `BER_DATA_ROOT`/`BER_SOURCE_DATA_ROOT` or auto-detects it.

## CLI stages

`run.sh` provisions/reuses `.venv` and calls `run_pipeline.py`, which runs every stage
in order (resumable):

```bash
bash run.sh mini
bash run.sh full
SKIP_EMBED=1 bash run.sh full      # text features only
VALIDATOR=/path/to/validate_submission.py bash run.sh full
```

Individual stages are available through the package CLI. With the venv active:

```bash
python -m business_entity_resolution make-mini            --config configs/mini.json
python -m business_entity_resolution prepare              --config configs/mini.json --split both
python -m business_entity_resolution make-splits          --config configs/mini.json --folds 5
python -m business_entity_resolution generate-candidates  --config configs/mini.json --split train
python -m business_entity_resolution build-features       --config configs/mini.json --split train
python -m business_entity_resolution train                --config configs/mini.json --model lightgbm --all-training-data
python -m business_entity_resolution tune-decision        --config configs/mini.json
python -m business_entity_resolution evaluate             --config configs/mini.json
python -m business_entity_resolution infer                --config configs/mini.json --model lightgbm
python -m business_entity_resolution preflight            --config configs/mini.json \
    --official-validator 6ab10eb3b23ba_student_resource/student_resource/utils/validate_submission.py --check-ids
```

Any stage can be re-run after a crash and will resume. Use `--force` to redo a stage.

### Honest tuning

`run_pipeline.py` trains on all folds except fold 0, scores/`tune-decision`/`evaluate`
on fold 0, then retrains on all data for `infer`. `tune-decision` persists the chosen
global and per-source thresholds to `artifacts/<profile>/<run_id>/decisions/best_thresholds.json`;
`evaluate`/`infer`/`write-output` pick it up automatically (CLI `--threshold` overrides).

## Configuration

Precedence: `CLI override > BER_* environment variable > JSON config > default`.

```text
BER_DATA_ROOT        dataset location      (mini: dataset/mini, full: dataset)
BER_ARTIFACT_ROOT    intermediate artifacts (default: artifacts/<profile>)
BER_OUTPUT_ROOT      final TSV output dir   (default: output/<profile>)
BER_RUN_ID  BER_DEVICE  BER_THREADS
```

`configs/*.json` controls blocking caps, LightGBM params, the decision threshold,
negative sampling, and whether BGE-M3 embeddings are on (`auto`/`true`/`false`).

## Architecture

```text
Raw TSV
  → strict audit and schema validation
  → raw-preserving normalization (US / India / France legal suffixes)
  → leakage-safe S1 folds
  → complementary candidate generation (exact + rare-token + name/address TF-IDF)
  → optional BGE-M3 pair cosine features
  → candidate-only pair features (36 features)
  → deterministic / SGD / optional LightGBM scorer
  → macro-F0.5 threshold tuning (global + per-source) + conservative France rule
  → format-gated matching_results.tsv + candidate_pairs.tsv
  → internal preflight + official validator
```

## Outputs

```text
output/
├── matching_results.tsv
└── candidate_pairs.tsv
```

Both files contain exactly one row per test S1, including blank rows. Internal
preflight enforces target existence, duplicate rules, candidate containment, exact
columns, and empty-list handling before the official validator runs.

## Submission package

The challenge expects a single archive named `<team_name>_submission.zip`:

```text
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/   # src/, README.md, requirements.txt, ...
└── Documentation_template.md
```

Team identity lives in exactly one place, `configs/submission.json`:

```json
{
  "team_name": "Blackhats",
  "team_members": ["Aayush Kumar", "Samarth Manoj Naik", "Ishhan Khera"],
  "submission_date": "",
  "documentation": "docs/methodology.md",
  "validator": "6ab10eb3b23ba_student_resource/student_resource/utils/validate_submission.py"
}
```

Then everything is one command:

```bash
make full
make package
```

`make package` (i.e. `ber package`):

1. reads `configs/submission.json` (CLI `--team-name`/`--documentation` override it),
2. seeds `docs/methodology.md` from the challenge template if it does not exist,
3. runs the internal preflight and the official validator,
4. injects Team Name / Members / Date into the doc header and writes
   `artifacts/full/remote-full/package/<team_name>_submission.zip`
   with `output/`, `code/business_entity_resolution/`, and `Documentation_template.md`.

`submission_date` defaults to today if left blank. The two output TSVs contain no
team metadata — only `source1_entity_id` and the ID list. `make validate` runs just
the preflight step on its own.

## Development

```bash
make test      # pytest
make lint      # ruff check
make fmt       # ruff --fix + format
make doctor    # environment / data health check
make clean     # drop generated artifacts and caches
```

## Reproducibility and compliance

- Seed defaults to 2026.
- Folds and negative sampling use stable cryptographic hashes.
- Candidate ordering and output lists are deterministic.
- Raw challenge files are never modified.
- Supplied challenge data only; no pretrained models or external address/entity data.
- LightGBM (MIT) and BGE-M3 (MIT, ~568M params) are within the license/size rules.
- A final model is not selected until held-out experiments run.
