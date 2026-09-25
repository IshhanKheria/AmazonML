# Amazon ML 2026 — Business Entity Resolution

This repository is the working area for a competition-compliant entity-resolution pipeline that maps every Source 1 test record to zero, one, or multiple Source 2/Source 3 records.

## Status

Initialization and dataset forensics only. No final model has been trained and no test predictions have been produced.

## Non-negotiable constraints

- Use only the supplied challenge dataset and locally installed, permissively licensed software.
- Never perform external business lookup, geocoding, registry lookup, API enrichment, or internet-based data augmentation.
- Treat `country` as an open-set string; France is present in test but absent from training.
- Optimize entity-level macro F0.5, with conservative final decisions and explicit singleton handling.
- Preserve high recall in the final candidate set presented to the matcher.
- Every final match must be present in `candidate_pairs.tsv`.
- Final outputs must satisfy the exact TSV contract and pass the supplied validator.
- Any final pretrained model must be MIT/Apache-2.0 licensed and no larger than 8B parameters.

## Repository layout

```text
.
├── 6ab10eb3b23ba_student_resource/student_resource/  # original challenge bundle; read-only
├── code/business_entity_resolution/
│   ├── README.md
│   ├── requirements.txt
│   ├── src/business_entity_resolution/
│   └── tests/
├── configs/                 # versioned run configuration
├── docs/                    # project initialization and operating decisions
├── experiments/             # append-only experiment registry and notes
├── reports/                 # generated audits and evaluation reports
├── artifacts/               # generated intermediate data/models; never source data
└── output/                  # eventual required TSV outputs
```

The challenge source files are not copied, renamed, edited, or deleted. Code must receive their location through configuration or CLI arguments.

## Source data

Default local data root:

```text
6ab10eb3b23ba_student_resource/student_resource/dataset
```

All TSVs must be read with an explicit tab delimiter and UTF-8 encoding.

## Initialization artifacts

- Dataset audit: `reports/dataset_audit.md`
- Machine-readable audit: `reports/dataset_audit.json` (when generated)
- Experiment protocol: `experiments/README.md`
- Experiment registry: `experiments/experiment_log.tsv`
- Agent/skill activation record: `docs/initialization.md`

## Next phase

The planning phase should define an entity-level validation protocol, reconstruct pair labels, benchmark complementary high-recall blocking strategies, and specify macro-F0.5 threshold/singleton evaluation before any final model is trained.

