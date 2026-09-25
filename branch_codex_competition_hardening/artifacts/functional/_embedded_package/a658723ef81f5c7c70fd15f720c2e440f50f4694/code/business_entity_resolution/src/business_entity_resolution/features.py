"""Memory-conscious pairwise feature construction over candidate pairs."""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
from rapidfuzz import fuzz


def _set(value: object) -> set[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return set()
    if isinstance(value, str):
        return set(value.split()) if value else set()
    return set(value)


def _jaccard(left: object, right: object) -> float:
    a, b = _set(left), _set(right)
    # Missing evidence is not positive evidence. Empty/empty used to score as
    # a perfect match, which was particularly harmful for blank addresses.
    return 0.0 if not a or not b else len(a & b) / len(a | b)


def _containment(left: object, right: object) -> float:
    a, b = _set(left), _set(right)
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def _ratio(left: str, right: str) -> float:
    return 0.0 if not left or not right else fuzz.ratio(left, right) / 100.0


def _partial(left: str, right: str) -> float:
    return 0.0 if not left or not right else fuzz.partial_ratio(left, right) / 100.0


FEATURE_COLUMNS = [
    "name_raw_exact", "name_canonical_exact", "name_compact_exact", "name_core_exact",
    "name_token_sorted_exact", "name_accent_folded_exact", "name_accent_folded_ratio",
    "name_ratio", "name_partial", "name_token_sort_ratio",
    "name_token_set_ratio", "name_token_jaccard", "name_token_containment", "name_length_ratio",
    "name_jaro_winkler",
    "address_canonical_exact", "address_compact_exact", "address_ratio", "address_partial",
    "address_token_jaccard", "address_token_containment", "address_length_ratio",
    "numeric_exact", "numeric_jaccard", "numeric_overlap", "numeric_conflicts",
    "postal_exact", "postal_jaccard", "country_equal", "candidate_is_s3",
    "s1_missing_address", "candidate_missing_address", "blocking_reason_count",
    "retrieval_score", "retrieval_rank_inverse", "name_address_interaction",
    "name_numeric_interaction",
    "bge_name_cosine", "bge_addr_cosine",
]

# Features that only exist when an embedding backend and cache are available.
EMBEDDING_COLUMNS = ["bge_name_cosine", "bge_addr_cosine"]


def _length_ratio(left: str, right: str) -> float:
    a, b = len(left or ""), len(right or "")
    return 0.0 if not a or not b else min(a, b) / max(a, b)


def _jaro_winkler(left: str, right: str) -> float:
    a, b = left or "", right or ""
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    match_distance = max(len(a), len(b)) // 2 - 1
    a_matches = [False] * len(a)
    b_matches = [False] * len(b)
    matches = 0
    for i, char in enumerate(a):
        start, end = max(0, i - match_distance), min(i + match_distance + 1, len(b))
        for j in range(start, end):
            if b_matches[j] or b[j] != char:
                continue
            a_matches[i] = b_matches[j] = True
            matches += 1
            break
    if matches == 0:
        return 0.0
    transpositions = _transpositions(a, b, a_matches, b_matches)
    jaro = (matches / len(a) + matches / len(b) + (matches - transpositions) / matches) / 3.0
    prefix = 0
    for x, y in zip(a, b):
        if x != y or prefix == 4:
            break
        prefix += 1
    return jaro + prefix * 0.1 * (1 - jaro)


def _transpositions(a: str, b: str, a_matches: list[bool], b_matches: list[bool]) -> int:
    a_chars = [a[i] for i in range(len(a)) if a_matches[i]]
    b_chars = [b[j] for j in range(len(b)) if b_matches[j]]
    return sum(1 for x, y in zip(a_chars, b_chars) if x != y) // 2


def build_pair_features(
    candidates: pd.DataFrame,
    source1: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    embed_names: dict[str, np.ndarray] | None = None,
    embed_addrs: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    required = {"source1_entity_id", "candidate_entity_id"}
    if not required.issubset(candidates.columns):
        raise ValueError(f"candidate table missing columns: {sorted(required - set(candidates.columns))}")
    if candidates.empty:
        columns = ["source1_entity_id", "candidate_entity_id", "candidate_source", *FEATURE_COLUMNS]
        for passthrough in ("label", "sample_weight", "forced_positive"):
            if passthrough in candidates:
                columns.append(passthrough)
        return pd.DataFrame(columns=columns)
    s1 = source1.add_prefix("s1_").rename(columns={"s1_entity_id": "source1_entity_id"})
    target = targets.add_prefix("c_").rename(columns={"c_entity_id": "candidate_entity_id"})
    pairs = candidates.merge(s1, on="source1_entity_id", how="left", validate="many_to_one")
    pairs = pairs.merge(target, on="candidate_entity_id", how="left", validate="many_to_one")
    if pairs["s1_name_canonical"].isna().any() or pairs["c_name_canonical"].isna().any():
        raise ValueError("candidate table references an unknown Source 1 or target ID")

    # Precompute embedding cosines vectorized (batched gather + dot) to avoid
    # per-row Python overhead; falls back to zeros when vectors are unavailable.
    bge_name = np.zeros(len(pairs), dtype=np.float32)
    bge_addr = np.zeros(len(pairs), dtype=np.float32)
    if embed_names is not None and embed_addrs is not None:
        bge_name = _pair_cosine(pairs["source1_entity_id"], pairs["candidate_entity_id"], embed_names)
        bge_addr = _pair_cosine(pairs["source1_entity_id"], pairs["candidate_entity_id"], embed_addrs)

    records: list[dict[str, object]] = []
    for position, row in enumerate(pairs.itertuples(index=False)):
        s1_name, c_name = row.s1_name_canonical, row.c_name_canonical
        s1_addr, c_addr = row.s1_address_canonical, row.c_address_canonical
        numeric_a, numeric_b = _set(row.s1_numeric_tokens), _set(row.c_numeric_tokens)
        name_ratio, address_ratio = _ratio(s1_name, c_name), _ratio(s1_addr, c_addr)
        reason_mask = str(getattr(row, "reason_mask", ""))
        records.append({
            "source1_entity_id": row.source1_entity_id,
            "candidate_entity_id": row.candidate_entity_id,
            "candidate_source": getattr(row, "candidate_source", row.candidate_entity_id[:2]),
            "name_raw_exact": float(row.s1_business_name_raw == row.c_business_name_raw),
            "name_canonical_exact": float(s1_name == c_name),
            "name_compact_exact": float(row.s1_name_compact == row.c_name_compact),
            "name_core_exact": float(row.s1_name_core == row.c_name_core),
            "name_token_sorted_exact": float(row.s1_name_token_sorted == row.c_name_token_sorted),
            "name_accent_folded_exact": float(
                bool(row.s1_name_accent_folded)
                and row.s1_name_accent_folded == row.c_name_accent_folded
            ),
            "name_accent_folded_ratio": _ratio(row.s1_name_accent_folded, row.c_name_accent_folded),
            "name_ratio": name_ratio,
            "name_partial": _partial(s1_name, c_name),
            "name_token_sort_ratio": fuzz.token_sort_ratio(s1_name, c_name) / 100.0,
            "name_token_set_ratio": fuzz.token_set_ratio(s1_name, c_name) / 100.0,
            "name_token_jaccard": _jaccard(row.s1_name_tokens, row.c_name_tokens),
            "name_token_containment": _containment(row.s1_name_tokens, row.c_name_tokens),
            "name_length_ratio": _length_ratio(s1_name, c_name),
            "name_jaro_winkler": _jaro_winkler(s1_name, c_name),
            "address_canonical_exact": float(s1_addr == c_addr and bool(s1_addr)),
            "address_compact_exact": float(row.s1_address_compact == row.c_address_compact and bool(row.s1_address_compact)),
            "address_ratio": address_ratio,
            "address_partial": _partial(s1_addr, c_addr),
            "address_token_jaccard": _jaccard(row.s1_address_tokens, row.c_address_tokens),
            "address_token_containment": _containment(row.s1_address_tokens, row.c_address_tokens),
            "address_length_ratio": _length_ratio(s1_addr, c_addr),
            "numeric_exact": float(numeric_a == numeric_b and bool(numeric_a)),
            "numeric_jaccard": _jaccard(numeric_a, numeric_b),
            "numeric_overlap": float(bool(numeric_a & numeric_b)),
            "numeric_conflicts": float(len(numeric_a ^ numeric_b)),
            "postal_exact": float(_set(row.s1_postal_like_tokens) == _set(row.c_postal_like_tokens) and bool(_set(row.s1_postal_like_tokens))),
            "postal_jaccard": _jaccard(row.s1_postal_like_tokens, row.c_postal_like_tokens),
            "country_equal": float(row.s1_country_norm == row.c_country_norm),
            "candidate_is_s3": float(str(row.candidate_entity_id).startswith("S3-")),
            "s1_missing_address": float(row.s1_missing_address),
            "candidate_missing_address": float(row.c_missing_address),
            "blocking_reason_count": float(len([part for part in reason_mask.split("|") if part])),
            "retrieval_score": float(getattr(row, "retrieval_score", 0.0)),
            "retrieval_rank_inverse": 1.0 / max(1, int(getattr(row, "retrieval_rank", 1))),
            "name_address_interaction": name_ratio * address_ratio,
            "name_numeric_interaction": name_ratio * float(bool(numeric_a & numeric_b)),
            "bge_name_cosine": float(bge_name[position]),
            "bge_addr_cosine": float(bge_addr[position]),
        })
    out = pd.DataFrame.from_records(records)
    for column in FEATURE_COLUMNS:
        out[column] = out[column].astype(np.float32)
    for passthrough in ("label", "sample_weight", "forced_positive"):
        if passthrough in candidates:
            out[passthrough] = candidates[passthrough].to_numpy()
    return out


def _pair_cosine(s1_ids: pd.Series, c_ids: pd.Series, vectors: dict[str, np.ndarray]) -> np.ndarray:
    """Vectorized cosine between aligned S1 and candidate embedding vectors."""
    if not vectors:
        return np.zeros(len(s1_ids), dtype=np.float32)
    dim = len(next(iter(vectors.values())))
    left_ids = s1_ids.astype(str).to_numpy()
    right_ids = c_ids.astype(str).to_numpy()
    left = np.zeros((len(left_ids), dim), dtype=np.float32)
    right = np.zeros((len(right_ids), dim), dtype=np.float32)
    for index, entity_id in enumerate(left_ids):
        vector = vectors.get(entity_id)
        if vector is not None:
            left[index] = vector
    for index, entity_id in enumerate(right_ids):
        vector = vectors.get(entity_id)
        if vector is not None:
            right[index] = vector
    left_norm = left / (np.linalg.norm(left, axis=1, keepdims=True) + 1e-9)
    right_norm = right / (np.linalg.norm(right, axis=1, keepdims=True) + 1e-9)
    return np.einsum("ij,ij->i", left_norm, right_norm).astype(np.float32)


def feature_matrix(frame: pd.DataFrame) -> np.ndarray:
    missing = set(FEATURE_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"feature table missing columns: {sorted(missing)}")
    return frame[FEATURE_COLUMNS].to_numpy(dtype=np.float32, copy=False)

