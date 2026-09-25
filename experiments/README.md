# Experiment protocol

Use one row per meaningful experiment in `experiment_log.tsv`. Keep the log append-only; if a run is invalid, mark it invalid in `status` and explain why in `notes` rather than deleting it.

Required run discipline:

1. Use an entity-level train/validation split and record the split seed/definition.
2. Evaluate blocking independently before classifier tuning.
3. Report candidate recall and reduction ratio.
4. Report entity-level macro precision, recall, F0.5, singleton accuracy, false merges, and missed matches.
5. Record source-pair and country slices when statistically meaningful.
6. Record runtime, peak-memory estimate, dependency/model license, and artifact paths.
7. Never use test labels or external entity information.

## Per-run artifacts

Each run should create `experiments/<experiment_id>/` containing the resolved configuration, input/artifact manifests, candidate metrics, validation and threshold reports, runtime/memory notes, error-category counts, license notes, and a keep/reject conclusion.

Do not enter placeholder scores in `experiment_log.tsv`. Add a row only after a command produced the corresponding measured artifact.
