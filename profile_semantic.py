"""Ollama embedding client with validation and a bounded process cache.

Persistent fact vectors live in PostgreSQL/pgvector. The process cache mainly
avoids recomputing identical query vectors; retrieval still falls back to the
deterministic selector when the provider is disabled or unavailable.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
import math
import os
from threading import Lock
from typing import Any
from urllib import error, request

from memory_retriever import ProfileMemory


DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_EMBEDDING_MODEL = "embeddinggemma:latest"


@dataclass(frozen=True)
class SemanticSettings:
    provider: str
    model: str
    base_url: str
    timeout_seconds: float
    max_candidates: int
    min_similarity: float
    cache_size: int

    @property
    def enabled(self) -> bool:
        return self.provider != "none"


@dataclass(frozen=True)
class SemanticScoreResult:
    scores: dict[str, float]
    provider: str
    model: str | None
    enabled: bool
    available: bool
    candidate_count: int
    reason: str | None = None


_embedding_cache: OrderedDict[tuple[str, str], tuple[float, ...]] = OrderedDict()
_cache_lock = Lock()


def _bounded_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    return min(maximum, max(minimum, value))


def _bounded_float(
    name: str,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError:
        return default
    if not math.isfinite(value):
        return default
    return min(maximum, max(minimum, value))


def semantic_settings() -> SemanticSettings:
    provider = os.getenv("MEMORY_PROFILE_SEMANTIC_PROVIDER", "none").strip().lower()
    if provider not in {"none", "ollama"}:
        provider = "none"
    return SemanticSettings(
        provider=provider,
        model=os.getenv(
            "MEMORY_PROFILE_EMBEDDING_MODEL",
            DEFAULT_EMBEDDING_MODEL,
        ).strip(),
        base_url=os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL).rstrip("/"),
        timeout_seconds=_bounded_float(
            "MEMORY_PROFILE_EMBEDDING_TIMEOUT_SECONDS", 10.0, 0.5, 60.0
        ),
        max_candidates=_bounded_int(
            "MEMORY_PROFILE_SEMANTIC_MAX_CANDIDATES", 100, 1, 500
        ),
        min_similarity=_bounded_float(
            "MEMORY_PROFILE_SEMANTIC_MIN_SIMILARITY", 0.40, -1.0, 1.0
        ),
        cache_size=_bounded_int(
            "MEMORY_PROFILE_EMBEDDING_CACHE_SIZE", 256, 0, 20_000
        ),
    )


def profile_memory_text(memory: ProfileMemory) -> str:
    value = json.dumps(memory.value, ensure_ascii=False, sort_keys=True)
    return f"category: {memory.category}\nkey: {memory.key}\nvalue: {value}"


def _validate_embeddings(payload: Any, expected_count: int) -> list[tuple[float, ...]]:
    if not isinstance(payload, dict) or not isinstance(payload.get("embeddings"), list):
        raise ValueError("embedding response has no embeddings array")
    raw_embeddings = payload["embeddings"]
    if len(raw_embeddings) != expected_count:
        raise ValueError("embedding response count does not match request")
    embeddings: list[tuple[float, ...]] = []
    dimensions: int | None = None
    for raw in raw_embeddings:
        if not isinstance(raw, list) or not raw:
            raise ValueError("embedding vector is empty")
        vector = tuple(float(item) for item in raw)
        if not all(math.isfinite(item) for item in vector):
            raise ValueError("embedding vector contains a non-finite value")
        if dimensions is None:
            dimensions = len(vector)
        elif len(vector) != dimensions:
            raise ValueError("embedding vectors use different dimensions")
        embeddings.append(vector)
    return embeddings


def _request_ollama_embeddings(
    texts: list[str], settings: SemanticSettings
) -> list[tuple[float, ...]]:
    body = json.dumps(
        {"model": settings.model, "input": texts, "truncate": True},
        ensure_ascii=False,
    ).encode("utf-8")
    call = request.Request(
        f"{settings.base_url}/api/embed",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(call, timeout=settings.timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Ollama embedding request failed: {exc}") from exc
    return _validate_embeddings(payload, len(texts))


def _get_cached(model: str, text: str) -> tuple[float, ...] | None:
    key = (model, hashlib.sha256(text.encode("utf-8")).hexdigest())
    with _cache_lock:
        vector = _embedding_cache.get(key)
        if vector is not None:
            _embedding_cache.move_to_end(key)
        return vector


def _put_cached(
    model: str,
    text: str,
    vector: tuple[float, ...],
    cache_size: int,
) -> None:
    if cache_size <= 0:
        return
    key = (model, hashlib.sha256(text.encode("utf-8")).hexdigest())
    with _cache_lock:
        _embedding_cache[key] = vector
        _embedding_cache.move_to_end(key)
        while len(_embedding_cache) > cache_size:
            _embedding_cache.popitem(last=False)


def _embeddings_for_texts(
    texts: list[str], settings: SemanticSettings
) -> list[tuple[float, ...]]:
    result: list[tuple[float, ...] | None] = [None] * len(texts)
    missing_texts: list[str] = []
    missing_indexes: list[int] = []
    for index, text in enumerate(texts):
        cached = _get_cached(settings.model, text)
        if cached is None:
            missing_texts.append(text)
            missing_indexes.append(index)
        else:
            result[index] = cached

    if missing_texts:
        if settings.provider != "ollama":
            raise RuntimeError(f"Unsupported semantic provider: {settings.provider}")
        generated = _request_ollama_embeddings(missing_texts, settings)
        for index, text, vector in zip(
            missing_indexes, missing_texts, generated, strict=True
        ):
            result[index] = vector
            _put_cached(settings.model, text, vector, settings.cache_size)

    return [vector for vector in result if vector is not None]


def embed_texts(
    texts: list[str],
    *,
    settings: SemanticSettings | None = None,
) -> list[tuple[float, ...]]:
    """Generate validated embeddings through the configured local provider."""
    active_settings = settings or semantic_settings()
    if not active_settings.enabled:
        raise RuntimeError("semantic embedding provider is disabled")
    if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("embedding input must contain nonempty strings")
    return _embeddings_for_texts(texts, active_settings)


def _cosine_similarity(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if len(left) != len(right):
        raise ValueError("embedding dimensions do not match")
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return max(-1.0, min(1.0, dot / (left_norm * right_norm)))


def score_profile_memories_semantically(
    memories: list[ProfileMemory],
    query: str | None,
    *,
    settings: SemanticSettings | None = None,
) -> SemanticScoreResult:
    active_settings = settings or semantic_settings()
    if not active_settings.enabled:
        return SemanticScoreResult(
            scores={},
            provider="none",
            model=None,
            enabled=False,
            available=False,
            candidate_count=0,
            reason="disabled",
        )
    if not query or not query.strip():
        return SemanticScoreResult(
            scores={},
            provider=active_settings.provider,
            model=active_settings.model,
            enabled=True,
            available=False,
            candidate_count=0,
            reason="query_missing",
        )
    candidates = memories[: active_settings.max_candidates]
    if not candidates:
        return SemanticScoreResult(
            scores={},
            provider=active_settings.provider,
            model=active_settings.model,
            enabled=True,
            available=True,
            candidate_count=0,
        )

    texts = [query, *(profile_memory_text(memory) for memory in candidates)]
    try:
        vectors = _embeddings_for_texts(texts, active_settings)
        if len(vectors) != len(texts):
            raise ValueError("embedding result was incomplete")
        query_vector = vectors[0]
        scores = {
            memory.id: _cosine_similarity(query_vector, vector)
            for memory, vector in zip(candidates, vectors[1:], strict=True)
        }
    except (RuntimeError, ValueError, TypeError) as exc:
        return SemanticScoreResult(
            scores={},
            provider=active_settings.provider,
            model=active_settings.model,
            enabled=True,
            available=False,
            candidate_count=len(candidates),
            reason=str(exc),
        )
    return SemanticScoreResult(
        scores=scores,
        provider=active_settings.provider,
        model=active_settings.model,
        enabled=True,
        available=True,
        candidate_count=len(candidates),
    )


def clear_embedding_cache() -> None:
    """Clear the bounded cache; exposed for deterministic tests."""
    with _cache_lock:
        _embedding_cache.clear()
