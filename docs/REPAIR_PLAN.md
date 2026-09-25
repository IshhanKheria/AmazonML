# Competition hardening plan

This branch starts from the teammate `m1` implementation and merges `main` so
the challenge data and requested local artifacts remain available through Git
LFS. The repair order follows the read-only audit.

## Completed in this branch

1. Evaluate every held-out Source 1 entity, including zero-candidate entities.
2. Separate threshold-selection fold 0 from independently reported fold 1.
3. Route model-specific parameters safely and honor `model.kind` by default.
4. Treat dual missing address/numeric/postal fields as neutral evidence.
5. Remove insertion-order truncation from rare-token aggregation and make
   fallback provenance candidate-specific.
6. Add accent-folded blocking/features while preserving raw representations.
7. Integrate deterministic per-entity negative sampling and balanced weights.
8. Scale SGD features and train SGD from feature shards in two bounded passes.
9. Fix prepared-shard overwrites and stale chunk reuse.
10. Process target sources separately during blocking and stream candidate
    metrics instead of concatenating the complete candidate table.
11. Build pair features in bounded chunks and score/output test shards without
    materializing the full test pair table.
12. Persist selected thresholds, verify exact candidate/feature key equality,
    and make packaging clean, preflight-gated, and correctly named.

## Required remote experiments before final submission

- Measure candidate recall, complete-entity recall, zero-candidate rate,
  p95/p99/max candidates, wall time, and peak RAM per blocker/source/country.
- Compare deterministic, scaled SGD, and MIT-licensed LightGBM using the fixed
  fold protocol. Record variance across additional folds before freezing.
- Mine high-scoring false candidates from out-of-fold predictions and rerun the
  comparison with hard negatives.
- Tune global and S2/S3 thresholds only from held-out evidence. France-specific
  margins and overrides are disabled because no labeled France examples exist.
- Run full official validation and verify dependency/model licenses in the
  methodology document.

## Remaining engineering risks

- Each blocker still builds an in-memory index for one complete target source.
- Character TF-IDF now multiplies against bounded target blocks before retaining
  top-K. Benchmark the configured target-block size on representative real
  shards and reduce it if peak memory is unsafe.
- Threshold tuning and strict preflight still load their validation/output
  mappings in memory; final candidate volume determines whether a disk-backed
  implementation is necessary.
- The optional pretrained embedding integration is disabled. It should remain
  disabled unless the organizers explicitly approve it under the supplied-data
  and license rules.
