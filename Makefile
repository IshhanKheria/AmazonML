# Amazon ML 2026 — Business Entity Resolution
# Single command surface. Run `make` or `make help` to list targets.
SHELL := /usr/bin/env bash
ROOT  := $(CURDIR)
VENV  := $(ROOT)/.venv
PY    := $(VENV)/bin/python
BER    = PYTHONPATH=$(ROOT)/src $(PY) -m business_entity_resolution

.DEFAULT_GOAL := help
.PHONY: help setup lfs data doctor mini full score-mini experiment test lint fmt validate package clean clean-all

help: ## list available targets
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup: ## create .venv and install all dependencies
	bash setup.sh

lfs: ## enable git-lfs and fetch the dataset objects
	@command -v git-lfs >/dev/null || { echo "git-lfs not installed"; exit 1; }
	git lfs install --local
	git lfs fetch origin main --include="6ab10eb3b23ba_student_resource/student_resource/dataset/**"

data: ## fetch + verify real TSVs into ./dataset
	bash scripts/fetch_data.sh

doctor: ## environment and data health check
	bash scripts/doctor.sh

mini: ## 15-record end-to-end sanity run (embeddings off)
	SKIP_SETUP=1 SKIP_EMBED=1 bash run.sh mini

full: ## real dataset end-to-end run
	SKIP_SETUP=1 bash run.sh full

score-mini: ## score output/mini against the labeled dataset/mini/test_ground_truth.tsv
	$(BER) evaluate-output --config configs/mini.json

experiment: ## measured iteration (holdout fit/score/tune/evaluate) -> experiments/experiment_log.tsv
	$(BER) experiment --config configs/remote_full.json $(if $(NAME),--name "$(NAME)",) $(if $(NOTES),--notes "$(NOTES)",)

test: ## run the test suite
	PYTHONPATH=$(ROOT)/src $(PY) -m pytest -q

lint: ## ruff static checks
	$(VENV)/bin/ruff check src tests

fmt: ## ruff autofix + format
	$(VENV)/bin/ruff check --fix src tests
	$(VENV)/bin/ruff format src tests

validate: ## internal + official preflight on output/ (validator from configs/submission.json)
	$(BER) preflight --config configs/remote_full.json --check-ids

package: ## validate, then build <team>_submission.zip (all settings from configs/submission.json)
	$(BER) package --config configs/remote_full.json

clean: ## remove generated artifacts, outputs and caches
	rm -rf artifacts/* output/*
	find src tests -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache
	@echo "cleaned (kept .venv; use 'make clean-all' to drop it too)"

clean-all: clean ## also remove the virtualenv
	rm -rf .venv
