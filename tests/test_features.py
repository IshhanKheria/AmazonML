import numpy as np
import pandas as pd

from business_entity_resolution.features import FEATURE_COLUMNS, build_pair_features, feature_matrix
from business_entity_resolution.normalize import normalize_records


def test_pair_features_use_only_candidates(smoke_frames):
    s1, s2, _, _ = smoke_frames
    candidates = pd.DataFrame([{
        "source1_entity_id": "S1-1", "candidate_entity_id": "S2-10", "candidate_source": "S2",
        "reason_mask": "exact_name", "retrieval_score": 1.0, "retrieval_rank": 1,
    }])
    features = build_pair_features(candidates, normalize_records(s1), normalize_records(s2))
    assert len(features) == 1
    assert set(FEATURE_COLUMNS).issubset(features.columns)
    assert feature_matrix(features).dtype == np.float32
    assert features.loc[0, "country_equal"] == 1.0

