import pandas as pd
import numpy as np

from business_entity_resolution.blocking import CandidateGenerator
from business_entity_resolution.normalize import normalize_records


def test_blocking_union_is_deterministic_and_open_set(smoke_frames):
    s1, s2, _, _ = smoke_frames
    ns1, n2 = normalize_records(s1), normalize_records(s2)
    config = {"rare_token_max_df": 20, "token_posting_cap": 20, "tfidf_top_k": 3, "per_source_cap": 10, "batch_size": 2}
    first = CandidateGenerator(config).fit(n2).transform(ns1)
    second = CandidateGenerator(config).fit(n2).transform(ns1)
    pd.testing.assert_frame_equal(first, second)
    alpha = first[first.source1_entity_id == "S1-1"]
    assert "S2-10" in set(alpha.candidate_entity_id)
    assert len(first[first.source1_entity_id == "S1-1"]) <= 10


def test_duplicate_content_ids_are_preserved():
    s1 = normalize_records(pd.DataFrame([("S1-1", "Acme", "1 Road", "US")], columns=["entity_id", "business_name", "business_address", "country"]))
    targets = normalize_records(pd.DataFrame([
        ("S2-1", "Acme", "1 Road", "US"), ("S2-2", "Acme", "1 Road", "US")
    ], columns=["entity_id", "business_name", "business_address", "country"]))
    candidates = CandidateGenerator({"token_posting_cap": 10, "per_source_cap": 10}, include_tfidf=False).fit(targets).transform(s1)
    assert set(candidates.candidate_entity_id) == {"S2-1", "S2-2"}


def test_blockers_accept_arrow_roundtrip_array_tokens(smoke_frames):
    s1, s2, _, _ = smoke_frames
    ns1, n2 = normalize_records(s1), normalize_records(s2)
    for frame in (ns1, n2):
        for column in ("name_tokens", "address_tokens", "numeric_tokens"):
            frame[column] = frame[column].map(np.asarray)
    candidates = CandidateGenerator({"token_posting_cap": 10, "per_source_cap": 10}, include_tfidf=False).fit(n2).transform(ns1)
    assert not candidates.empty
