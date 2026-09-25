"""Deterministic, raw-preserving entity text normalization."""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable
import pandas as pd

# Legal designators are an open set: the challenge test set introduces France,
# which never appears in training. Stripping these stops identical roots from
# scoring lower purely because of a mismatched designation.
LEGAL_SUFFIXES = frozenset(
    {
        # US
        "corp", "corporation", "inc", "incorporated", "llc", "llp", "lp", "co", "company",
        # India / UK-style
        "pvt", "private", "ltd", "limited", "enterprises", "enterprise", "opc",
        # France
        "sarl", "sas", "sasu", "sci", "eurl", "sa", "snc", "gie", "scop", "scp",
    }
)


def _clean_controls(value: str) -> str:
    return "".join(" " if unicodedata.category(char).startswith("C") else char for char in value)


def _unicode_tokens(value: str) -> tuple[str, ...]:
    """Tokenize letters/numbers while retaining combining marks used by Indic scripts."""
    result: list[str] = []
    current: list[str] = []
    for char in value:
        category = unicodedata.category(char)
        if category[0] in {"L", "N", "M"}:
            current.append(char)
        elif current:
            result.append("".join(current))
            current = []
    if current:
        result.append("".join(current))
    return tuple(result)


def canonical_text(value: str, ampersand_to_and: bool = True) -> str:
    value = unicodedata.normalize("NFKC", str(value or ""))
    value = _clean_controls(value).casefold()
    if ampersand_to_and:
        value = value.replace("&", " and ")
    return " ".join(_unicode_tokens(value))


def compact_text(value: str) -> str:
    return "".join(canonical_text(value).split())


def accent_folded_text(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", canonical_text(value))
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def tokens(value: str) -> tuple[str, ...]:
    normalized = canonical_text(value)
    return tuple(normalized.split()) if normalized else ()


def core_name(value: str, legal_suffixes: Iterable[str] = LEGAL_SUFFIXES) -> str:
    suffixes = frozenset(legal_suffixes)
    parts = list(tokens(value))
    while parts and parts[-1] in suffixes:
        parts.pop()
    return " ".join(parts)


def token_sorted(value: str) -> str:
    return " ".join(sorted(tokens(value)))


def numeric_tokens(value: str) -> tuple[str, ...]:
    result: list[str] = []
    current: list[str] = []
    for char in unicodedata.normalize("NFKC", str(value or "")):
        if char.isdecimal():
            current.append(char)
        elif current:
            result.append("".join(current))
            current = []
    if current:
        result.append("".join(current))
    return tuple(result)


def postal_like_tokens(value: str) -> tuple[str, ...]:
    return tuple(
        token for token in tokens(value)
        if 3 <= len(token) <= 10 and any(char.isdigit() for char in token)
    )


def normalize_country(value: str) -> str:
    return canonical_text(value, ampersand_to_and=False)


def normalize_records(frame: pd.DataFrame, legal_suffixes: Iterable[str] = LEGAL_SUFFIXES) -> pd.DataFrame:
    required = {"entity_id", "business_name", "business_address", "country"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing columns for normalization: {sorted(missing)}")
    out = frame.copy()
    out["business_name_raw"] = out["business_name"]
    out["business_address_raw"] = out["business_address"]
    out["country_raw"] = out["country"]
    out["country_norm"] = out["country"].map(normalize_country)
    out["name_canonical"] = out["business_name"].map(canonical_text)
    out["name_compact"] = out["business_name"].map(compact_text)
    out["name_core"] = out["business_name"].map(lambda value: core_name(value, legal_suffixes))
    out["name_token_sorted"] = out["business_name"].map(token_sorted)
    out["name_accent_folded"] = out["business_name"].map(accent_folded_text)
    out["address_canonical"] = out["business_address"].map(canonical_text)
    out["address_compact"] = out["business_address"].map(compact_text)
    out["address_token_sorted"] = out["business_address"].map(token_sorted)
    out["name_tokens"] = out["business_name"].map(tokens)
    out["address_tokens"] = out["business_address"].map(tokens)
    out["numeric_tokens"] = out["business_address"].map(numeric_tokens)
    out["postal_like_tokens"] = out["business_address"].map(postal_like_tokens)
    out["missing_address"] = out["business_address"].eq("")
    return out
