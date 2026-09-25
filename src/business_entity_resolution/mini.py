"""Build a realistic mini dataset that mirrors the real challenge distribution.

The mini set is *sampled from the real files* so it carries the same schema,
noise (abbreviations, transliterations, reordered/landmark addresses, missing
addresses), country mix, and match cardinality — only tiny.

Composition (default quotas):
  * ``train/``  from the real training files (US + India) with ground truth.
  * ``test/``   US + India **sampled from the training files** so it carries
    real matches *and* ground truth (``test_ground_truth.tsv``), plus France
    entities sampled from the real test files as open-set singletons (they have
    no training labels, so their expected match set is empty).

This makes the mini test split scoreable end-to-end via
``business_entity_resolution evaluate-output`` while still exercising unseen
``France`` handling. The real challenge test files remain unlabeled, as always.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import random

import pandas as pd

from .data import ground_truth_path, iter_tsv, source_path
from .schemas import GROUND_TRUTH_COLUMNS, SOURCE_COLUMNS

TRAIN_QUOTA = {"US": 6, "India": 6}
TEST_QUOTA = {"US": 5, "India": 5, "France": 5}
S1_BUFFER_PER_COUNTRY = 60
DECOYS_PER_COUNTRY = 4


def _parse_ids(value: str) -> list[str]:
    return [item for item in str(value).split(",") if item]


def _collect_s1(chunk, quotas: dict[str, int], buffers: dict[str, list[str]], rows: dict[str, dict]) -> None:
    for row in chunk.itertuples(index=False):
        country = str(row.country)
        if country in quotas and len(buffers[country]) < S1_BUFFER_PER_COUNTRY:
            entity_id = str(row.entity_id)
            buffers[country].append(entity_id)
            rows[entity_id] = {"entity_id": entity_id, "business_name": row.business_name,
                               "business_address": row.business_address, "country": row.country}


def _buffers_ready(buffers: dict[str, list[str]], quotas: dict[str, int]) -> bool:
    return all(len(buffers.get(c, [])) >= min(quotas[c] * 3, S1_BUFFER_PER_COUNTRY) for c in quotas)


def _choose(buffers: dict[str, list[str]], quotas: dict[str, int], rng: random.Random) -> list[str]:
    chosen: list[str] = []
    for country, wanted in quotas.items():
        pool = sorted(buffers.get(country, []))
        rng.shuffle(pool)
        chosen.extend(pool[:wanted])
    return chosen


def _scan_targets(
    data_root: Path,
    read_split: str,
    source: int,
    needed: set[str],
    quotas: dict[str, int],
    rng: random.Random,
) -> tuple[list[dict], dict[str, list[dict]]]:
    """Return (rows for needed IDs, decoys per country) from one target source."""
    lookup: dict[str, dict] = {}
    decoys: dict[str, list[dict]] = defaultdict(list)
    remaining = set(needed)
    for chunk in iter_tsv(source_path(data_root, read_split, source), SOURCE_COLUMNS, batch_size=100_000):
        if remaining:
            selected = chunk[chunk["entity_id"].isin(remaining)]
            for row in selected.itertuples(index=False):
                lookup[str(row.entity_id)] = {"entity_id": str(row.entity_id), "business_name": row.business_name,
                                              "business_address": row.business_address, "country": row.country}
                remaining.discard(str(row.entity_id))
        for country in quotas:
            if len(decoys[country]) < DECOYS_PER_COUNTRY:
                pool = chunk[chunk["country"].eq(country) & ~chunk["entity_id"].isin(needed)]
                if not pool.empty:
                    take = pool.sample(min(DECOYS_PER_COUNTRY - len(decoys[country]), len(pool)), random_state=rng.randint(0, 2**31))
                    decoys[country].extend(
                        {"entity_id": str(r.entity_id), "business_name": r.business_name,
                         "business_address": r.business_address, "country": r.country}
                        for r in take.itertuples(index=False)
                    )
        if not remaining and all(len(decoys[c]) >= DECOYS_PER_COUNTRY for c in quotas):
            break
    return list(lookup.values()), decoys


def _build_split(
    data_root: Path,
    out_split: str,
    quotas: dict[str, int],
    rng: random.Random,
    with_ground_truth: bool,
    read_split: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame | None, dict[str, object]]:
    """Sample one split and return (source1, source2, source3, ground_truth, summary)."""
    read_split = read_split or out_split
    # 1) Stream Source 1 until each country buffer is full.
    buffers: dict[str, list[str]] = defaultdict(list)
    s1_rows: dict[str, dict] = {}
    for chunk in iter_tsv(source_path(data_root, read_split, 1), SOURCE_COLUMNS, batch_size=100_000):
        _collect_s1(chunk, quotas, buffers, s1_rows)
        if _buffers_ready(buffers, quotas):
            break

    chosen = _choose(buffers, quotas, rng)
    chosen_set = set(chosen)
    truth_map: dict[str, list[str]] = {}

    if with_ground_truth:
        pool = {e for ids in buffers.values() for e in ids}
        remaining = set(chosen) | pool
        for chunk in iter_tsv(ground_truth_path(data_root), GROUND_TRUTH_COLUMNS, batch_size=200_000):
            matches = chunk[chunk["source1_entity_id"].isin(remaining)]
            for row in matches.itertuples(index=False):
                truth_map[str(row.source1_entity_id)] = _parse_ids(row.matched_entity_ids)
                remaining.discard(str(row.source1_entity_id))
            if not remaining:
                break
        for entity_id in chosen:
            truth_map.setdefault(entity_id, [])

        # Guarantee a singleton and a both-source match when the buffer allows.
        def _profile(ids):
            singleton = any(not truth_map[i] for i in ids)
            both = any(
                any(t.startswith("S2-") for t in truth_map[i]) and any(t.startswith("S3-") for t in truth_map[i])
                for i in ids
            )
            return singleton, both

        def _swap(predicate) -> None:
            for country in quotas:
                for replacement in buffers.get(country, []):
                    if replacement in chosen_set or not predicate(truth_map[replacement]):
                        continue
                    for index in reversed(range(len(chosen))):
                        if str(s1_rows.get(chosen[index], {}).get("country")) == country:
                            chosen_set.discard(chosen[index])
                            chosen[index] = replacement
                            chosen_set.add(replacement)
                            return

        has_singleton, has_both = _profile(chosen)
        if not has_singleton:
            _swap(lambda m: not m)
        if not has_both:
            _swap(lambda m: any(t.startswith("S2-") for t in m) and any(t.startswith("S3-") for t in m))

    mini_s1 = pd.DataFrame([s1_rows[e] for e in chosen if e in s1_rows], columns=SOURCE_COLUMNS)

    # 2) Collect target rows (true matches + decoys) for both target sources.
    needed_s2 = {t for e in chosen for t in truth_map.get(e, []) if t.startswith("S2-")}
    needed_s3 = {t for e in chosen for t in truth_map.get(e, []) if t.startswith("S3-")}
    s2_rows, s2_decoys = _scan_targets(data_root, read_split, 2, needed_s2, quotas, rng)
    s3_rows, s3_decoys = _scan_targets(data_root, read_split, 3, needed_s3, quotas, rng)
    for country in quotas:
        s2_rows.extend(s2_decoys.get(country, []))
        s3_rows.extend(s3_decoys.get(country, []))
    mini_s2 = pd.DataFrame(s2_rows, columns=SOURCE_COLUMNS) if s2_rows else pd.DataFrame(columns=SOURCE_COLUMNS)
    mini_s3 = pd.DataFrame(s3_rows, columns=SOURCE_COLUMNS) if s3_rows else pd.DataFrame(columns=SOURCE_COLUMNS)

    mini_gt: pd.DataFrame | None = None
    if with_ground_truth:
        included = set(mini_s2["entity_id"]) | set(mini_s3["entity_id"])
        gt_rows = [(e, ",".join(t for t in truth_map.get(e, []) if t in included)) for e in chosen]
        mini_gt = pd.DataFrame(gt_rows, columns=GROUND_TRUTH_COLUMNS)

    countries = mini_s1["country"].value_counts().to_dict() if not mini_s1.empty else {}
    summary = {
        "source1": len(mini_s1),
        "source2": len(mini_s2),
        "source3": len(mini_s3),
        "singletons": int(sum(1 for e in chosen if not truth_map.get(e))) if with_ground_truth else 0,
        "countries": countries,
    }
    return mini_s1, mini_s2, mini_s3, mini_gt, summary


def _dedupe(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    return frame.drop_duplicates(subset="entity_id", keep="first").reset_index(drop=True)


def _write_split(out_dir: Path, out_split: str, s1: pd.DataFrame, s2: pd.DataFrame, s3: pd.DataFrame, gt: pd.DataFrame | None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_tsv(s1, out_dir / f"{out_split}_source1.tsv")
    _write_tsv(s2, out_dir / f"{out_split}_source2.tsv")
    _write_tsv(s3, out_dir / f"{out_split}_source3.tsv")
    if gt is not None:
        name = "train_ground_truth.tsv" if out_split == "train" else "test_ground_truth.tsv"
        _write_tsv(gt, out_dir / name)


def build_mini(
    data_root: str | Path,
    mini_root: str | Path,
    *,
    count: int = 15,
    seed: int = 2026,
) -> dict[str, object]:
    """Sample a realistic, scoreable mini train/test dataset into ``mini_root``."""
    data_root = Path(data_root)
    mini_root = Path(mini_root)
    rng = random.Random(seed)

    train_s1, train_s2, train_s3, train_gt, train_summary = _build_split(
        data_root, "train", dict(TRAIN_QUOTA), rng, with_ground_truth=True
    )
    _write_split(mini_root / "train", "train", train_s1, train_s2, train_s3, train_gt)

    # Labeled test: US/India sampled from the *training* files so real matches
    # and ground truth exist.
    label_quota = {country: TEST_QUOTA[country] for country in ("US", "India")}
    lab_s1, lab_s2, lab_s3, lab_gt, _ = _build_split(
        data_root, "test", label_quota, rng, with_ground_truth=True, read_split="train"
    )
    # Open-set extras: France entities from the *test* files, expected singletons.
    fr_s1, fr_s2, fr_s3, _, _ = _build_split(
        data_root, "test", {"France": TEST_QUOTA["France"]}, rng, with_ground_truth=False, read_split="test"
    )

    test_s1 = pd.concat([lab_s1, fr_s1], ignore_index=True)
    test_s2 = _dedupe(pd.concat([lab_s2, fr_s2], ignore_index=True))
    test_s3 = _dedupe(pd.concat([lab_s3, fr_s3], ignore_index=True))
    france_gt = pd.DataFrame([(e, "") for e in fr_s1["entity_id"]], columns=GROUND_TRUTH_COLUMNS)
    test_gt = pd.concat([lab_gt, france_gt], ignore_index=True) if lab_gt is not None else france_gt
    _write_split(mini_root / "test", "test", test_s1, test_s2, test_s3, test_gt)

    test_summary = {
        "source1": len(test_s1),
        "source2": len(test_s2),
        "source3": len(test_s3),
        "labeled": len(lab_s1),
        "singletons": int(sum(1 for value in test_gt["matched_entity_ids"] if not value)),
        "countries": test_s1["country"].value_counts().to_dict() if not test_s1.empty else {},
    }
    return {"train": train_summary, "test": test_summary}


def _write_tsv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False, encoding="utf-8", lineterminator="\n")