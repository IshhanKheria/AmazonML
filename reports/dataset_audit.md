# Dataset and schema audit

Audit date: 2026-09-25  
Scope: read-only initialization audit; no model training or test inference.

## Canonical data location

`6ab10eb3b23ba_student_resource/student_resource/dataset/`

All files are UTF-8 tab-separated text. Parsing must use an explicit tab delimiter and preserve blank strings (`keep_default_na=False` or an equivalent rule).

## Dimensions and countries

| Split | File | Rows | Columns | US | India | France |
|---|---|---:|---:|---:|---:|---:|
| Train | `train_source1.tsv` | 2,206,821 | 4 | 1,323,633 | 883,188 | 0 |
| Train | `train_source2.tsv` | 5,034,616 | 4 | 3,016,817 | 2,017,799 | 0 |
| Train | `train_source3.tsv` | 5,285,603 | 4 | 3,170,056 | 2,115,547 | 0 |
| Train | `train_ground_truth.tsv` | 2,206,821 | 2 | — | — | — |
| Test | `test_source1.tsv` | 1,732,544 | 4 | 663,106 | 809,986 | 259,452 |
| Test | `test_source2.tsv` | 4,887,273 | 4 | 1,871,330 | 2,312,565 | 703,378 |
| Test | `test_source3.tsv` | 5,082,316 | 4 | 1,945,701 | 2,405,000 | 731,615 |

There are 26,435,994 rows across the seven TSVs (24,229,173 source-record rows plus 2,206,821 ground-truth rows). Country counts sum exactly to every source-file row count; no blank or additional country labels were observed. France is absent from training but represents 14.98% of test Source 1, so open-set behavior is a primary design requirement rather than a corner case.

## Schema

Every source file has four string fields:

| Field | Meaning | Integrity rule |
|---|---|---|
| `entity_id` | Record identifier | Prefix must match file source: `S1-`, `S2-`, or `S3-` |
| `business_name` | Raw business name | Preserve raw value; derive normalized variants separately |
| `business_address` | Raw address | May be blank in auxiliary sources |
| `country` | Country label | Treat as an arbitrary/open-set string |

The ground-truth file has:

| Field | Meaning |
|---|---|
| `source1_entity_id` | Training Source 1 identifier |
| `matched_entity_ids` | Comma-separated S2/S3 IDs; blank means no matches |

The observed source rows have four fields and the expected ID prefixes. Exact source-file duplicate and whitespace checks remain a planning-phase audit item.

## Missingness and text observations

- Source 1 has no blank fields in train or test.
- Only `business_address` was observed blank in auxiliary source files: train S2 168,967 (3.3561%), train S3 175,916 (3.3282%), test S2 129,408 (2.6479%), and test S3 136,098 (2.6779%). IDs, names, and countries were nonblank in the streaming scan.
- Unicode/non-Latin names occur (including Devanagari), so normalization and similarity must be Unicode-safe and raw text must be retained.
- Names contain punctuation, leading noise, apostrophes, legal suffix variants, and domain-like strings.
- Addresses contain commas, mixed case, reordered components, abbreviations, landmarks, and missing components.
- The extracted archive includes `.DS_Store` and `__MACOSX/._*` metadata. Preserve but exclude these files from data ingestion and final packaging.

## Ground-truth match distribution

Ground truth contains 2,206,821 unique Source 1 rows and 7,638,365 positive links.

| Statistic | Count | Share of S1 |
|---|---:|---:|
| Empty match set / singleton | 123,247 | 5.5848% |
| Non-empty match set | 2,083,574 | 94.4152% |
| Any S2 match | 1,919,076 | 86.9611% |
| Any S3 match | 1,940,545 | 87.9340% |
| Both S2 and S3 | 1,776,047 | 80.4799% |
| S2 only | 143,029 | 6.4812% |
| S3 only | 164,498 | 7.4541% |

Positive links comprise 3,693,619 S2 links and 3,944,746 S3 links. Mean total matches per Source 1 is 3.4613; maximum is 11.

| Total matches per S1 | S1 rows |
|---:|---:|
| 0 | 123,247 |
| 1 | 119,157 |
| 2 | 375,212 |
| 3 | 530,841 |
| 4 | 484,115 |
| 5 | 321,957 |
| 6 | 164,868 |
| 7 | 63,968 |
| 8 | 18,680 |
| 9 | 4,205 |
| 10 | 534 |
| 11 | 37 |

No duplicate IDs within a match list, malformed match lists, duplicate ground-truth Source 1 rows, or globally reused target IDs were observed. The latter is a training-set property only and must not be imposed as a test-time one-to-one constraint without validation evidence.

## Scale implications

- Full train S1×(S2+S3) comparison is about 22.77 trillion pairs.
- Full test S1×(S2+S3) comparison is about 17.27 trillion pairs.
- Chunked I/O, sparse representations, and staged high-recall retrieval are mandatory.
- `candidate_pairs.tsv` must contain the exact last-stage candidates actually scored, not an earlier superset.

## Output and scoring implications

- Macro F0.5 must be computed at the Source 1 set level.
- Both truth and prediction empty must score 1.0; exactly one empty must score 0.0. Generic classifier metrics do not implement this competition behavior automatically.
- One-to-many matching is common; top-1 matching is invalid.
- Correct singleton decisions matter even though singletons are only 5.58% of training S1.
- The supplied validator treats candidate absence and candidate-containment failures as warnings and skips target-ID existence unless `--check-ids` is used. The pipeline therefore needs a stricter, memory-aware preflight in addition to the official validator.

## Remaining audit checks before modeling

1. Produce a persistent raw-file checksum manifest.
2. Re-run exact source-level ID/full-row/identity-key duplicate checks with UTF-8 file output.
3. Record name/address length quantiles, leading/trailing whitespace, control characters, and possible mojibake without printing raw business strings.
4. Verify exact set equality between ground-truth S1 IDs and train Source 1 IDs.
5. Verify every ground-truth target ID exists in the corresponding train S2/S3 file.
6. Check train/test ID overlap and derive ground-truth cardinality by country.

These checks are required planning gates; the current audit is sufficient to establish schema, scale, open-set, cardinality, and submission-design requirements.
