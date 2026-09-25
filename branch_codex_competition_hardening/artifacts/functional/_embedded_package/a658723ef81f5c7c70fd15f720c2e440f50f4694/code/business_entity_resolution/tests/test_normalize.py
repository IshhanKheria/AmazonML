from business_entity_resolution.normalize import canonical_text, core_name, normalize_records, numeric_tokens


def test_normalization_is_unicode_safe_and_deterministic(tiny_source):
    first = normalize_records(tiny_source)
    second = normalize_records(tiny_source)
    assert first.to_dict("records") == second.to_dict("records")
    assert first.loc[0, "country_norm"] == "france"
    assert "राम" in first.loc[1, "name_canonical"]
    assert tiny_source.loc[0, "business_name"] == "Acme & Sons Ltd"


def test_name_and_numeric_variants():
    assert canonical_text(" ACME & Sons, Ltd. ") == "acme and sons ltd"
    assert core_name("Acme Corporation") == "acme"
    assert numeric_tokens("12-A Road 75001") == ("12", "75001")

