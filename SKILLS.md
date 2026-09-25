# SKILLS.md

## Purpose

This file defines the technical skills the Amazon ML 2026 entity-resolution agent should activate when solving the challenge.

---

## Skill 1 — Dataset Forensics

### Goal
Understand the actual dataset before designing the model.

### Tasks
- locate train/test files;
- verify TSV schemas;
- count records;
- inspect nulls and duplicates;
- inspect source/country distributions;
- inspect string lengths and unusual characters;
- analyze ground-truth match cardinality;
- estimate singleton prevalence;
- identify possible data artifacts.

### Deliverables
- dataset profile;
- schema report;
- match-cardinality report;
- data-quality risks.

---

## Skill 2 — Entity Resolution Fundamentals

### Goal
Model the problem as Source 1 → Source 2/3 entity linking.

### Tasks
- distinguish record linkage from ordinary classification;
- construct positive/negative pair examples;
- account for one-to-many matches;
- avoid assuming one-to-one correspondence;
- identify singleton/no-match cases.

### Key principle
The unit of prediction is a candidate pair, but the final output is a set of matches for each Source 1 entity.

---

## Skill 3 — Text Normalization

### Goal
Create robust representations of noisy business names and addresses.

### Techniques
- Unicode normalization;
- case normalization;
- punctuation handling;
- whitespace normalization;
- legal suffix handling;
- abbreviation-aware transformations;
- `&` / `and` handling;
- token normalization;
- digit extraction;
- transliteration-aware representations where possible using local libraries;
- conservative versus aggressive normalization variants.

### Rule
Never replace raw fields. Add normalized/derived fields alongside them.

---

## Skill 4 — String Similarity

### Goal
Create strong pairwise similarity features.

### Candidate features
- exact match;
- normalized exact match;
- character n-gram similarity;
- TF-IDF cosine similarity;
- token Jaccard;
- token overlap;
- token-sort similarity;
- edit distance;
- normalized edit similarity;
- longest-common-subsequence-like signals;
- digit overlap;
- numeric-token agreement.

Evaluate each feature family independently and in combinations.

---

## Skill 5 — Address Resolution

### Goal
Handle noisy and incomplete addresses.

### Consider
- road/street abbreviations;
- apartment/unit numbers;
- postal/PIN-like numbers;
- state/city tokens;
- landmarks;
- reordered components;
- missing components;
- numeric overlap;
- character-level similarity.

Do not use external geocoding or external address databases.

---

## Skill 6 — Blocking / Candidate Generation

### Goal
Reduce the Cartesian product while preserving extremely high recall.

### Possible blocks
- exact normalized name;
- name prefix/suffix;
- character n-gram retrieval;
- rare-token overlap;
- address token overlap;
- numeric-token overlap;
- country-compatible blocking where appropriate;
- combinations of name/address signals;
- approximate nearest-neighbor retrieval if locally available and compliant.

### Evaluation
For each blocking strategy measure:
- candidate recall;
- candidate count;
- reduction ratio;
- average candidates per Source 1;
- worst-case candidate counts.

Take the union of complementary high-value blocks when necessary.

---

## Skill 7 — Pairwise ML Classification

### Goal
Learn whether a candidate pair refers to the same business.

### Models to test
- logistic regression;
- gradient boosting;
- random forest / extra trees;
- other locally available permissively licensed tabular models.

### Requirements
- train only on training-derived labels;
- handle severe class imbalance;
- use appropriate negative sampling;
- prevent entity leakage between train/validation;
- compare models using the challenge metric.

---

## Skill 8 — Hard Negative Mining

### Goal
Teach the classifier to distinguish genuinely confusing non-matches.

### Process
1. Train a baseline.
2. Find high-scoring false positives.
3. Add them as informative negatives.
4. Retrain.
5. Re-evaluate.

Pay special attention to:
- same/similar business names;
- chains/branches;
- common surnames;
- similar addresses;
- businesses in the same city;
- transliteration collisions.

---

## Skill 9 — Validation Design

### Goal
Produce trustworthy offline estimates.

### Preferred approach
Use an entity-level holdout so validation entities are not trivially memorized through pair construction.

Evaluate multiple seeds/splits if computationally practical.

### Required metrics
- macro F_0.5;
- precision;
- recall;
- singleton accuracy;
- false-positive/false-merge count;
- candidate recall;
- reduction ratio.

Never optimize only pair-level accuracy.

---

## Skill 10 — Threshold Optimization

### Goal
Translate pair scores into final match lists.

### Tasks
- sweep score thresholds;
- calculate macro F_0.5;
- inspect precision-recall trade-offs;
- test source-specific thresholds only if justified;
- analyze singleton/no-match behavior;
- consider confidence gaps between top candidates;
- verify that thresholds generalize across validation splits.

Do not assume 0.5 is an appropriate probability threshold.

---

## Skill 11 — Error Analysis

### Goal
Understand why the system succeeds or fails.

### Create buckets
- exact name match;
- typo;
- abbreviation;
- legal suffix variation;
- address variation;
- transliteration;
- missing fields;
- branch ambiguity;
- common-name collision;
- singleton false positive;
- candidate-generation miss;
- classifier false positive;
- classifier false negative.

Use these buckets to drive the next experiment.

---

## Skill 12 — Open-Set Country Handling

### Goal
Ensure unseen test countries do not break inference.

### Requirements
- no fixed `{US, India}` assumptions;
- treat country as an arbitrary string/category;
- avoid preprocessing that errors on unseen categories;
- use country as a feature only in a way that safely handles unseen values;
- test the pipeline explicitly with an unseen country value.

---

## Skill 13 — Submission Engineering

### Goal
Generate exactly the required artifacts.

### Must produce
```text
output/
├── matching_results.tsv
└── candidate_pairs.tsv
```

### Checks
- all test Source 1 IDs present exactly once;
- no S1 IDs in match lists;
- all IDs exist;
- no duplicates;
- matches are subsets of candidates;
- valid TSV encoding;
- correct column names;
- empty lists represented correctly.

Run the official validation utility.

---

## Skill 14 — Reproducible ML Engineering

### Goal
Make the solution auditable and rerunnable.

### Practices
- modular source tree;
- config-driven paths;
- pinned dependencies;
- deterministic seeds;
- command-line entry point;
- saved model/artifacts when useful;
- logging;
- experiment records;
- README with exact commands.

---

## Skill 15 — Competition Optimization

### Goal
Iteratively improve the leaderboard-relevant metric without violating the rules.

### Loop

```text
Profile
→ Baseline
→ Validate
→ Analyze errors
→ Improve blocking
→ Improve features
→ Improve model
→ Tune threshold
→ Validate again
→ Freeze candidate strategy
→ Retrain on full train
→ Infer test
→ Validate submission
```

Do not chase tiny improvements without checking stability across validation splits.

---

## Skill 16 — License / Fair-Play Compliance

### Goal
Ensure the final model and dependencies satisfy the challenge requirements.

### Requirements
- final model must comply with the stated MIT/Apache 2.0 model requirement;
- model size must be ≤ 8B parameters;
- no external business/entity data;
- document important dependencies and licenses;
- keep the entire approach reproducible from supplied data.

When uncertain about a model/dependency's license, do not assume compliance; verify its license from local package/model metadata or the official competition materials available to the team.
