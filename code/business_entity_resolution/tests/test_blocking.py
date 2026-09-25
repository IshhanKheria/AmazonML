import pandas as pd
import numpy as np

from business_entity_resolution.blocking import CandidateGenerator
from business_entity_resolution.blocking.base import CandidateReason
from business_entity_resolution.blocking.tokens import RareTokenBlocker
from business_entity_resolution.blocking.tfidf import TfidfTopKBlocker
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


def test_token_posting_cap_is_independent_of_target_row_order():
    columns = ["entity_id", "country_norm", "name_tokens"]
    targets = pd.DataFrame([
        (f"S2-{index}", "us", ("shared",)) for index in range(1, 301)
    ], columns=columns)
    query = pd.DataFrame([("S1-1", "us", ("shared",))], columns=columns)
    first = RareTokenBlocker("name_tokens", CandidateReason.RARE_NAME_TOKEN, 500, 25).fit(targets).transform(query)
    second = RareTokenBlocker("name_tokens", CandidateReason.RARE_NAME_TOKEN, 500, 25).fit(targets.iloc[::-1]).transform(query)
    assert first["candidate_entity_id"].tolist() == second["candidate_entity_id"].tolist()


def test_country_fallback_reason_is_candidate_specific():
    columns = ["entity_id", "country_norm", "name_tokens"]
    targets = pd.DataFrame([
        ("S2-1", "france", ("local",)),
        ("S2-2", "us", ("global",)),
    ], columns=columns)
    query = pd.DataFrame([
        ("S1-1", "france", ("local", "global")),
    ], columns=columns)
    result = RareTokenBlocker("name_tokens", CandidateReason.RARE_NAME_TOKEN, 10, 10).fit(targets).transform(query)
    bits = dict(zip(result["candidate_entity_id"], result["reason_bits"]))
    assert not bits["S2-1"] & int(CandidateReason.COUNTRY_FALLBACK)
    assert bits["S2-2"] & int(CandidateReason.COUNTRY_FALLBACK)


def test_chunked_tfidf_matches_single_target_block(smoke_frames):
    s1, s2, _, _ = smoke_frames
    queries, targets = normalize_records(s1), normalize_records(s2)
    chunked = TfidfTopKBlocker(top_k=3, min_score=0.0, batch_size=2, target_batch_size=2).fit(targets).transform(queries)
    single = TfidfTopKBlocker(top_k=3, min_score=0.0, batch_size=2, target_batch_size=100).fit(targets).transform(queries)
    pd.testing.assert_frame_equal(chunked, single)
