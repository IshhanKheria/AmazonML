"""Build a realistic mini dataset that mirrors the real challenge distribution.

The mini set is *sampled from the real files* so it carries the same schema,
noise (abbreviations, transliterations, reordered/landmark addresses, missing
addresses), country mix, and match cardinality — only tiny. It is used to run
the full pipeline end-to-end on a laptop without touching the full corpus.

Key realism point: France appears only in the **test** files, not in training.
The mini set therefore builds:
  * ``train/``  from the real training files (US + India) with ground truth
  * ``test/``   from the real test files (US + India + France), no ground truth

Composition (default 15 Source 1 entities per split):
  * several US, India in train; US + India + France in test
  * at least one singleton, several single-match, several multi-match,
    and at least one entity matched by both Source 2 and Source 3
  * Source 2/3 pool includes true matches plus non-matching decoys

Scans are streamed and stop as soon as enough records are collected, so this is
fast regardless of the full dataset size.
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
    split: str,
    source: int,
    needed: set[str],
    quotas: dict[str, int],
    rng: random.Random,
) -> tuple[list[dict], dict[str, list[dict]]]:
    """Return (rows for needed IDs, decoys per country) from one target source."""
    lookup: dict[str, dict] = {}
    decoys: dict[str, list[dict]] = defaultdict(list)
    remaining = set(needed)
    for chunk in iter_tsv(source_path(data_root, split, source), SOURCE_COLUMNS, batch_size=100_000):
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
    out_dir: Path,
    split: str,
    quotas: dict[str, int],
    rng: random.Random,
    with_ground_truth: bool,
) -> dict[str, int]:
    # 1) Stream Source 1 until each country buffer is full.
    buffers: dict[str, list[str]] = defaultdict(list)
    s1_rows: dict[str, dict] = {}
    for chunk in iter_tsv(source_path(data_root, split, 1), SOURCE_COLUMNS, batch_size=100_000):
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
    s2_rows, s2_decoys = _scan_targets(data_root, split, 2, needed_s2, quotas, rng)
    s3_rows, s3_decoys = _scan_targets(data_root, split, 3, needed_s3, quotas, rng)
    for country in quotas:
        s2_rows.extend(s2_decoys.get(country, []))
        s3_rows.extend(s3_decoys.get(country, []))
    mini_s2 = pd.DataFrame(s2_rows, columns=SOURCE_COLUMNS) if s2_rows else pd.DataFrame(columns=SOURCE_COLUMNS)
    mini_s3 = pd.DataFrame(s3_rows, columns=SOURCE_COLUMNS) if s3_rows else pd.DataFrame(columns=SOURCE_COLUMNS)

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_tsv(mini_s1, out_dir / f"{split}_source1.tsv")
    _write_tsv(mini_s2, out_dir / f"{split}_source2.tsv")
    _write_tsv(mini_s3, out_dir / f"{split}_source3.tsv")

    if with_ground_truth:
        included = set(mini_s2["entity_id"]) | set(mini_s3["entity_id"])
        gt_rows = [(e, ",".join(t for t in truth_map.get(e, []) if t in included)) for e in chosen]
        mini_gt = pd.DataFrame(gt_rows, columns=GROUND_TRUTH_COLUMNS)
        _write_tsv(mini_gt, out_dir / "train_ground_truth.tsv")

    countries = mini_s1["country"].value_counts().to_dict() if not mini_s1.empty else {}
    return {"source1": len(mini_s1), "source2": len(mini_s2), "source3": len(mini_s3),
            "singletons": int(sum(1 for e in chosen if not truth_map.get(e))) if with_ground_truth else 0,
            "countries": countries}


def build_mini(
    data_root: str | Path,
    mini_root: str | Path,
    *,
    count: int = 15,
    seed: int = 2026,
) -> dict[str, object]:
    """Sample a realistic mini train/test dataset into ``mini_root``."""
    data_root = Path(data_root)
    mini_root = Path(mini_root)
    rng = random.Random(seed)

    train_summary = _build_split(data_root, mini_root / "train", "train", dict(TRAIN_QUOTA), rng, with_ground_truth=True)
    test_summary = _build_split(data_root, mini_root / "test", "test", dict(TEST_QUOTA), rng, with_ground_truth=False)
    return {"train": train_summary, "test": test_summary}


def _write_tsv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False, encoding="utf-8", lineterminator="\n")
