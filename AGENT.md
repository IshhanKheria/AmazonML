# AGENT.md

## Mission

Build a reproducible, competition-compliant machine-learning pipeline for the Amazon ML 2026 **Business Entity Resolution Challenge**.

The system must learn from the provided training data and generate entity matches for every Source 1 record in the test set by linking it to zero, one, or many records from Source 2 and/or Source 3.

The primary optimization target is the challenge's macro-averaged **F_0.5** score, which is precision-heavy and therefore makes false merges especially costly.

## Ground Rules

1. **Use only the supplied challenge data and locally installed/open-source libraries.**
2. **Never perform external entity lookup or data augmentation.**
   - No web lookup of businesses.
   - No government/company registries.
   - No geocoding APIs.
   - No commercial entity-resolution APIs.
   - No external business databases.
3. Do not leak test labels or use any hidden/external source of truth.
4. Treat `country` as an open-set categorical/string field. Never hard-code the training countries.
5. Preserve the distinction between Source 1, Source 2, and Source 3.
6. Every test Source 1 entity must appear exactly once in the final output.
7. Final matches may reference only existing Source 2/Source 3 test IDs.
8. Every final match must be present in `candidate_pairs.tsv`.
9. Empty predictions are valid and important because singleton Source 1 entities receive full credit when correctly predicted as having no matches.
10. Use the supplied validation utility before considering a submission complete.

## Competition Objective

Optimize:

`F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)`

The score is computed per Source 1 entity and then macro-averaged.

Therefore:
- prioritize precision over recall during final decision-making;
- nevertheless maintain high candidate recall during blocking;
- explicitly evaluate singleton detection;
- do not assume every Source 1 record has a match;
- tune decision thresholds using held-out training validation rather than intuition alone.

## Required Pipeline

The agent should develop and maintain the following stages:

1. **Data audit**
   - discover actual dataset structure;
   - inspect row counts, nulls, duplicates, country distributions, string lengths, and source sizes;
   - verify TSV parsing.

2. **Ground-truth reconstruction**
   - parse `train_ground_truth.tsv`;
   - convert the comma-separated match lists into pair-level labels;
   - preserve the original Source 1 → matched IDs relationship.

3. **Entity normalization**
   - create deterministic, auditable normalized representations of business names and addresses;
   - preserve raw fields;
   - experiment with multiple normalization levels rather than destroying information with one aggressive transformation.

4. **Candidate generation / blocking**
   - generate high-recall candidate pairs from Source 1 to Source 2/3;
   - use multiple complementary blocking strategies;
   - retain the final candidate set actually passed to the matching model;
   - measure candidate recall and reduction ratio.

5. **Pairwise feature engineering**
   Useful feature families include:
   - exact normalized-name agreement;
   - character n-gram TF-IDF cosine similarity;
   - token/Jaccard similarities;
   - edit-distance-derived similarities;
   - token-set / token-sort similarities;
   - address similarity at multiple normalization levels;
   - digit overlap and numeric-token agreement;
   - postal/PIN-like token agreement where present;
   - country equality;
   - name/address length statistics;
   - missingness indicators;
   - source-pair indicators;
   - interaction features such as name similarity × address similarity;
   - strong exact/near-exact signals.

6. **Pair classifier / scorer**
   Compare appropriate models using validation:
   - logistic regression;
   - gradient-boosted tree models;
   - random forest / extra trees where useful;
   - other locally available permissively licensed models if compliant.

   A model is not mandatory if a deterministic similarity system demonstrably performs better, but the final solution must be evidence-driven.

7. **Decision layer**
   - convert pair scores into final multi-match predictions;
   - tune thresholds for F_0.5;
   - consider separate thresholds by source pair or confidence regime only when validation supports it;
   - include a singleton/no-match decision;
   - avoid arbitrary fixed matching counts.

8. **Validation**
   Use a realistic entity-level holdout, not random pair splitting that causes leakage.

   Report:
   - macro F_0.5;
   - macro precision;
   - macro recall;
   - singleton accuracy;
   - false merges;
   - missed matches;
   - candidate recall;
   - candidate reduction ratio;
   - score distribution;
   - performance by country and source pair where statistically meaningful.

9. **Test inference**
   - fit the selected pipeline on all available labeled training data;
   - run candidate generation;
   - score candidates;
   - apply the validated decision rule;
   - generate both required TSV outputs.

10. **Submission validation**
    Run the official validator against:
    - `output/matching_results.tsv`
    - `output/candidate_pairs.tsv`
    - `dataset/test/`

11. **Reproducibility**
    Maintain:
    - `src/`
    - `README.md`
    - pinned `requirements.txt`
    - configuration files where useful;
    - deterministic random seeds where applicable;
    - experiment logs/results;
    - methodology documentation.

## Architecture Principles

### Candidate generation is a first-class ML problem

The matching model cannot recover a true pair that blocking removed. Therefore candidate recall should be measured independently and optimized before aggressively tuning the classifier.

Use several blocking strategies and take their union.

### Precision-heavy final decisions

Because F_0.5 weights precision more strongly than recall, a candidate being merely plausible is not enough to make it a final match.

Prefer:
- strong multi-field evidence;
- calibrated confidence;
- conservative thresholds;
- explicit no-match handling.

### Multiple matches are allowed

A Source 1 entity may map to multiple Source 2 and/or Source 3 records. Never impose a one-to-one assumption unless the training data proves that such a constraint is valid.

### Source-specific behavior

Do not assume Source 2 and Source 3 have identical noise patterns. Measure their behavior separately.

### Country generalization

France appears in test data but not training data. The pipeline must not fail because a country is unseen during training.

Avoid preprocessing that assumes only `US` and `India`.

## Experimentation Rules

Every meaningful experiment should record:

- experiment ID;
- data split;
- candidate strategy;
- features;
- model;
- hyperparameters;
- threshold;
- candidate recall;
- precision;
- recall;
- F_0.5;
- singleton metrics;
- runtime;
- memory considerations;
- observations.

Do not select a model based on a single metric without checking false merges and singleton behavior.

## Coding Standards

- Python 3.
- Prefer pandas/numpy/scikit-learn and other permissively licensed dependencies.
- Keep preprocessing deterministic.
- Separate data loading, normalization, blocking, feature engineering, training, inference, evaluation, and output writing.
- Avoid notebooks as the only implementation; notebooks may be used for exploration.
- Avoid hard-coded absolute paths.
- Use command-line/configurable paths.
- Add assertions for schema and output integrity.
- Never silently swallow errors.

## Required Output Contract

`matching_results.tsv`

```text
source1_entity_id<TAB>matched_entity_ids
```

`candidate_pairs.tsv`

```text
source1_entity_id<TAB>candidate_entity_ids
```

For both:
- one row per test Source 1 ID;
- empty list when there are no IDs;
- comma-separated IDs within the list;
- no duplicate IDs;
- only valid S2/S3 test IDs;
- every final match must be in the candidate set.

## Definition of Done

The solution is not complete until:

- the full pipeline runs end-to-end from the supplied dataset;
- validation metrics are reported;
- the chosen approach is justified using held-out data;
- the test outputs are generated;
- the official validator passes;
- the final package is reproducible;
- methodology documentation is complete;
- no prohibited external data lookup was used.
