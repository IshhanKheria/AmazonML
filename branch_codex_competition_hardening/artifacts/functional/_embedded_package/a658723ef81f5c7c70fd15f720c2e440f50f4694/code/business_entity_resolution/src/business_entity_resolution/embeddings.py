"""Optional BGE-M3 embedding features (pair-level cosine only).

The challenge permits only permissively licensed models up to 8B parameters.
``BAAI/bge-m3`` is MIT licensed and ~568M parameters. Embeddings are computed
once per entity (name and address separately), cached under ``artifacts/`` as
fp16 vectors, and then reused to produce pairwise cosine features via batched
gather + dot products. No external data or services are used.

When ``torch``/``transformers``/``FlagEmbedding`` are unavailable, every entry
point returns ``None`` and the pipeline continues without embedding features.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .artifacts import read_frame
from .checkpoint import ShardStore, input_fingerprint, shard_for_id

DEFAULT_MODEL = "BAAI/bge-m3"
_EMPTY_SENTINEL = "__EMPTY__"


def _import_backend():
    """Return (torch, model_class) or (None, None) when unavailable."""
    try:
        import torch  # type: ignore
    except Exception:
        return None, None
    try:
        from FlagEmbedding import BGEM3FlagModel  # type: ignore

        return torch, BGEM3FlagModel
    except Exception:
        pass
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore

        return torch, SentenceTransformer
    except Exception:
        return None, None


def backend_available() -> bool:
    torch, model_class = _import_backend()
    return torch is not None and model_class is not None


def _embed_dir(config: Any, split: str, source: int) -> Path:
    return config.artifact_dir("embeddings") / f"{split}_source{source}"


def load_embedding_vectors(config: Any, split: str) -> tuple[dict[str, np.ndarray] | None, dict[str, np.ndarray] | None]:
    """Return (name_vectors, address_vectors) keyed by entity_id, or (None, None).

    Vectors for all three sources are loaded from the cached embedding shards.
    Source 1 vectors are required because pair cosine compares an S1 entity
    against an S2/S3 candidate. Returns ``(None, None)`` when the Source 2/3
    caches are not complete, so feature building proceeds without them.
    """
    names: dict[str, np.ndarray] = {}
    addrs: dict[str, np.ndarray] = {}
    # Source 2 and 3 must be complete for the features to be meaningful.
    for source in (2, 3):
        store = ShardStore(_embed_dir(config, split, source), _embedding_fingerprint(config, split, source))
        if not store.is_complete():
            return None, None
    for source in (1, 2, 3):
        store = ShardStore(_embed_dir(config, split, source), _embedding_fingerprint(config, split, source))
        if not store.is_complete():
            continue
        for part in store.iter_parts():
            frame = read_frame(part)
            if frame.empty:
                continue
            name_matrix = np.vstack(frame["vec_name"].to_numpy())
            addr_matrix = np.vstack(frame["vec_addr"].to_numpy())
            for index, entity_id in enumerate(frame["entity_id"].to_numpy()):
                names[str(entity_id)] = name_matrix[index].astype(np.float32)
                addrs[str(entity_id)] = addr_matrix[index].astype(np.float32)
    return names, addrs


def _embedding_fingerprint(config: Any, split: str, source: int) -> str:
    """Fingerprint the prepared shards plus the model name for cache validity."""
    model = str((config.embeddings or {}).get("model", DEFAULT_MODEL))
    prepared = config.artifact_dir("prepared") / f"{split}_source{source}"
    return input_fingerprint([prepared]) + f":model={model}"


def _encode_texts(model: Any, backend: str, texts: list[str], batch_size: int) -> np.ndarray:
    if backend == "bge":
        output = model.encode(texts, batch_size=batch_size, max_length=256)["dense_vecs"]
        return np.asarray(output, dtype=np.float32)
    return np.asarray(model.encode(texts, batch_size=batch_size, show_progress_bar=False), dtype=np.float32)


def _load_model(config: Any, logger: Any):
    torch, model_class = _import_backend()
    if torch is None or model_class is None:
        logger.event("embeddings_unavailable", reason="backend_import_failed")
        return None, None, None
    model_name = str((config.embeddings or {}).get("model", DEFAULT_MODEL))
    plan = config.resource_plan()
    device = "cuda" if plan.device == "cuda" else "cpu"
    if model_class.__name__ == "BGEM3FlagModel":
        model = model_class(model_name, use_fp16=(device == "cuda"))
        backend = "bge"
    else:
        model = model_class(model_name, device=device)
        backend = "st"
    return model, backend, device


def embed_split(config: Any, split: str, logger: Any | None = None) -> dict[str, object]:
    """Compute and cache BGE-M3 name/address vectors for Source 2 and 3.

    Sharded by S1-ID-range style bucketing over target entity IDs and fully
    resumable. Returns a summary dict. When embeddings are disabled or the
    backend is unavailable, this is a no-op.
    """
    from .logging import NullLogger

    logger = logger if logger is not None else NullLogger()

    if not config.embeddings_enabled():
        logger.event("embeddings_skipped", split=split, reason="disabled_or_unavailable")
        return {"enabled": False, "split": split}

    model, backend, device = _load_model(config, logger)
    if model is None:
        return {"enabled": False, "split": split}

    plan = config.resource_plan()
    batch_size = int((config.embeddings or {}).get("batch_size", plan.embed_batch_size))
    total_vectors = 0
    for source in (1, 2, 3):
        store = ShardStore(_embed_dir(config, split, source), _embedding_fingerprint(config, split, source))
        if store.is_complete():
            logger.info("embeddings already complete", split=split, source=source)
            continue
        prepared_dir = config.artifact_dir("prepared") / f"{split}_source{source}"
        prepared_store = ShardStore(prepared_dir, input_fingerprint([prepared_dir]))
        frame = prepared_store.read_all(columns=["entity_id", "business_name", "business_address"])
        n_shards = config.n_shards
        keys = frame["entity_id"].map(lambda value: shard_for_id(value, n_shards))
        buckets = {int(key): group for key, group in frame.groupby(keys, sort=False)}
        pending = store.pending_keys(range(n_shards))
        source_vectors = 0
        for index, key in enumerate(pending, start=1):
            group = buckets.get(int(key), frame.iloc[0:0])
            name_texts = [str(value) if value else _EMPTY_SENTINEL for value in group["business_name"].tolist()]
            addr_texts = [str(value) if value else _EMPTY_SENTINEL for value in group["business_address"].tolist()]

            def _run(chunk_size: int, _name=name_texts, _addr=addr_texts, _group=group) -> pd.DataFrame:
                name_matrix = _encode_texts(model, backend, _name, chunk_size)
                addr_matrix = _encode_texts(model, backend, _addr, chunk_size)
                return pd.DataFrame({
                    "entity_id": _group["entity_id"].to_numpy(),
                    "vec_name": list(name_matrix.astype(np.float16)),
                    "vec_addr": list(addr_matrix.astype(np.float16)),
                })

            from .resources import run_with_oom_backoff

            vectors = run_with_oom_backoff(
                _run,
                max(1, batch_size),
                on_retry=lambda size, exc: logger.event("oom_backoff", stage="embeddings", new_batch=size, error=type(exc).__name__),
            )
            store.commit(int(key), vectors)
            source_vectors += len(vectors)
            logger.progress("embeddings", index, len(pending), extra={"split": split, "source": source, "shard": int(key), "device": device})
        store.mark_complete(source_vectors, ["entity_id", "vec_name", "vec_addr"], extra={"split": split, "source": source, "model": str((config.embeddings or {}).get("model", DEFAULT_MODEL))})
        total_vectors += source_vectors
    return {"enabled": True, "split": split, "vectors": total_vectors, "device": device}
