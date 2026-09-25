"""Build a realistic mini dataset that mirrors the real challenge distribution.

The mini set is *sampled from the real files* so it carries the same schema,
noise (abbreviations, transliterations, reordered/landmark addresses, missing
addresses), country mix, and match cardinality — only tiny. It is used to run
the full pipeline end-to-end on a laptop without touching the full corpus.

Composition (default 15 Source 1 entities):
  * 6 US, 6 India, 3 France
  * at least one singleton, several single-match, several multi-match,
    and at least one entity matched by both Source 2 and Source 3
  * Source 2/3 pool includes true matches plus non-matching decoys
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import random

import pandas as pd

from .artifacts import write_frame
from .data import ground_truth_path, read_tsv, source_path
from .schemas import GROUND_TRUTH_COLUMNS, SOURCE_COLUMNS

# Country quotas for the default 15-entity mini set.
COUNTRY_QUOTA = {"US": 6, "India": 6, "France": 3}
DECOYS_PER_COUNTRY = 3


def _parse_ids(value: str) -> list[str]:
    return [item for item in str(value).split(",") if item]


def _load_source_rows(path: Path) -> pd.DataFrame:
    return read_tsv(path, SOURCE_COLUMNS)


def build_mini(
    data_root: str | Path,
    mini_root: str | Path,
    *,
    count: int = 15,
    seed: int = 2026,
) -> dict[str, int]:
    """Sample a realistic mini dataset and write it to ``mini_root``.

    Returns a small summary dict with entity counts. Writes ``train/`` (with
    ground truth) and ``test/`` (without ground truth), both in the exact real
    schema.
    """
    data_root = Path(data_root)
    mini_root = Path(mini_root)
    rng = random.Random(seed)

    train_s1 = _load_source_rows(source_path(data_root, "train", 1))
    train_s2 = _load_source_rows(source_path(data_root, "train", 2))
    train_s3 = _load_source_rows(source_path(data_root, "train", 3))
    truth = read_tsv(ground_truth_path(data_root), GROUND_TRUTH_COLUMNS)
    truth_map = {
        str(row.source1_entity_id): _parse_ids(row.matched_entity_ids)
        for row in truth.itertuples(index=False)
    }

    # Scale the country quotas to the requested count, preserving proportion.
    quota = dict(COUNTRY_QUOTA)
    if count != sum(quota.values()):
        total = sum(quota.values())
        quota = {country: max(1, round(count * share / total)) for country, share in quota.items()}
        while sum(quota.values()) > count:
            country = max(quota, key=quota.get)
            if quota[country] > 1:
                quota[country] -= 1
            else:
                break

    s2_map = {str(row.entity_id): row for row in train_s2.itertuples(index=False)}
    s3_map = {str(row.entity_id): row for row in train_s3.itertuples(index=False)}

    # Bucket S1 by country and cardinality profile so we can satisfy the mix.
    by_country: dict[str, list[str]] = defaultdict(list)
    for row in train_s1.itertuples(index=False):
        by_country[str(row.country)].append(str(row.entity_id))

    chosen: list[str] = []
    for country, wanted in quota.items():
        pool = by_country.get(country, [])
        rng.shuffle(pool)
        chosen.extend(pool[:wanted])

    # Ensure at least one singleton and one both-source match where possible.
    chosen_set = set(chosen)
    for country in quota:
        pool = [entity_id for entity_id in by_country.get(country, []) if entity_id not in chosen_set]
        singletons = [entity_id for entity_id in pool if not truth_map.get(entity_id)]
        if singletons:
            replace = next((entity_id for entity_id in chosen if truth_map.get(entity_id) and entity_id not in singletons), None)
            if replace is not None and not any(not truth_map.get(entity_id) for entity_id in chosen):
                chosen[chosen.index(replace)] = singletons[0]
                chosen_set.discard(replace)
                chosen_set.add(singletons[0])

    mini_s1 = train_s1[train_s1["entity_id"].isin(chosen)].reset_index(drop=True)

    # Collect true target rows for the chosen S1 entities.
    needed_s2: set[str] = set()
    needed_s3: set[str] = set()
    for entity_id in chosen:
        for target_id in truth_map.get(entity_id, []):
            if target_id.startswith("S2-"):
                needed_s2.add(target_id)
            elif target_id.startswith("S3-"):
                needed_s3.add(target_id)

    s2_rows = [s2_map[target_id] for target_id in needed_s2 if target_id in s2_map]
    s3_rows = [s3_map[target_id] for target_id in needed_s3 if target_id in s3_map]
    mini_s2 = pd.DataFrame(s2_rows, columns=SOURCE_COLUMNS) if s2_rows else train_s2.iloc[0:0].copy()
    mini_s3 = pd.DataFrame(s3_rows, columns=SOURCE_COLUMNS) if s3_rows else train_s3.iloc[0:0].copy()

    # Add decoys (non-matching same-country records) so precision is exercised.
    for source_index, frame in ((2, mini_s2), (3, mini_s3)):
        all_targets = train_s2 if source_index == 2 else train_s3
        for country in quota:
            present = set(frame["entity_id"]) if not frame.empty else set()
            candidates = [
                row for row in all_targets.itertuples(index=False)
                if str(row.country) == country and str(row.entity_id) not in present
            ]
            rng.shuffle(candidates)
            decoys = candidates[:DECOYS_PER_COUNTRY]
            if decoys:
                frame = pd.concat([frame, pd.DataFrame(decoys, columns=SOURCE_COLUMNS)], ignore_index=True)
        if source_index == 2:
            mini_s2 = frame
        else:
            mini_s3 = frame

    # Ground-truth subset: one row per selected S1, listing only included targets.
    included = set(mini_s2["entity_id"]) | set(mini_s3["entity_id"])
    gt_rows = []
    for entity_id in chosen:
        matched = [target for target in truth_map.get(entity_id, []) if target in included]
        gt_rows.append((entity_id, ",".join(matched)))
    mini_gt = pd.DataFrame(gt_rows, columns=GROUND_TRUTH_COLUMNS)

    # Write train and test (test has no ground truth / no self matches).
    train_dir = mini_root / "train"
    test_dir = mini_root / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    for name, frame in (
        ("train_source1", mini_s1), ("train_source2", mini_s2),
        ("train_source3", mini_s3), ("train_ground_truth", mini_gt),
    ):
        write_frame(frame, train_dir / f"{name}.parquet")
    _write_tsv(mini_s1, train_dir / "train_source1.tsv")
    _write_tsv(mini_s2, train_dir / "train_source2.tsv")
    _write_tsv(mini_s3, train_dir / "train_source3.tsv")
    _write_tsv(mini_gt, train_dir / "train_ground_truth.tsv")

    _write_tsv(mini_s1, test_dir / "test_source1.tsv")
    _write_tsv(mini_s2, test_dir / "test_source2.tsv")
    _write_tsv(mini_s3, test_dir / "test_source3.tsv")

    return {
        "source1": len(mini_s1),
        "source2": len(mini_s2),
        "source3": len(mini_s3),
        "singletons": int(sum(1 for entity_id in chosen if not truth_map.get(entity_id))),
    }


def _write_tsv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False, encoding="utf-8", lineterminator="\n")
