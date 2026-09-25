from __future__ import annotations

import pandas as pd
import pytest

from business_entity_resolution.pipeline import make_smoke_fixture


@pytest.fixture
def smoke_frames():
    return make_smoke_fixture()


@pytest.fixture
def tiny_source():
    return pd.DataFrame([
        ("S1-1", "Acme & Sons Ltd", "12 Main Rd, Paris 75001", "France"),
        ("S1-2", "राम मार्केटिंग", "10 MG Road 560001", "India"),
    ], columns=["entity_id", "business_name", "business_address", "country"])

