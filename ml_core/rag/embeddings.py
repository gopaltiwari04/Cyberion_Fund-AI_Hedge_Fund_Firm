"""
Text embeddings via fastembed (ONNX, no torch dependency).

The same model is used for indexing (passage embeddings) and for queries
(query embeddings with the model's retrieval prefix). ``RAG_EMBEDDINGS=off``
disables semantic search, in which case retrieval falls back to PostgreSQL
full-text search only.
"""

from __future__ import annotations

import threading
from typing import Sequence

from common.config import get_settings
from common.logs import get_logger

logger = get_logger(__name__)

EMBEDDING_DIM = 384
_model = None
_failed = False
_lock = threading.Lock()


def _load():
    global _model, _failed
    if _model is not None or _failed:
        return _model
    with _lock:
        if _model is not None or _failed:
            return _model
        settings = get_settings()
        if settings.rag_embeddings == "off":
            _failed = True
            return None
        try:
            from fastembed import TextEmbedding

            _model = TextEmbedding(settings.embedding_model)
        except Exception as error:  # noqa: BLE001 - optional dependency / model download
            logger.warning("Embedding model unavailable, using full-text search only: %s", error)
            _failed = True
    return _model


def available() -> bool:
    return _load() is not None


def model_name() -> str:
    return get_settings().embedding_model


def embed_passages(texts: Sequence[str], batch_size: int = 64) -> list[list[float]]:
    model = _load()
    if model is None:
        raise RuntimeError("Embedding model unavailable")
    return [vector.tolist() for vector in model.passage_embed(list(texts), batch_size=batch_size)]


def embed_query(text: str) -> list[float] | None:
    model = _load()
    if model is None:
        return None
    return next(iter(model.query_embed(text))).tolist()


def to_pgvector(vector: Sequence[float]) -> str:
    return "[" + ",".join(f"{value:.6f}" for value in vector) + "]"
