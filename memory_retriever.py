"""Deterministic, bounded retrieval for structured long-term profile facts."""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


ESTIMATED_CHARACTERS_PER_TOKEN = 3
TURKISH_STOP_WORDS = {
    "acaba",
    "ben",
    "benim",
    "bir",
    "bu",
    "da",
    "de",
    "icin",
    "ile",
    "mi",
    "mı",
    "mu",
    "mü",
    "nasil",
    "ne",
    "su",
    "ve",
}


@dataclass(frozen=True)
class ProfileMemory:
    id: str
    category: str
    key: str
    value: Any
    confidence: float
    verification_status: str
    updated_at: datetime


@dataclass(frozen=True)
class SelectedProfileMemory:
    memory: ProfileMemory
    score: float
    estimated_tokens: int
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class ProfileRetrievalResult:
    selected: tuple[SelectedProfileMemory, ...]
    eligible_fact_count: int
    candidate_fact_count: int
    estimated_tokens: int
    max_facts: int
    max_tokens: int
    query_used: bool

    @property
    def omitted_fact_count(self) -> int:
        return self.eligible_fact_count - len(self.selected)


def normalize_text(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value.casefold())
    ascii_like = "".join(
        character
        for character in decomposed
        if not unicodedata.combining(character)
    ).replace("ı", "i")
    return re.sub(r"\s+", " ", ascii_like).strip()


def tokenize(value: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", normalize_text(value))
        if len(token) >= 2 and token not in TURKISH_STOP_WORDS
    }


def estimate_tokens(value: Any) -> int:
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return max(
        1,
        (len(serialized) + ESTIMATED_CHARACTERS_PER_TOKEN - 1)
        // ESTIMATED_CHARACTERS_PER_TOKEN,
    )


def _token_matches(query_token: str, memory_token: str) -> bool:
    if query_token == memory_token:
        return True
    if min(len(query_token), len(memory_token)) < 5:
        return False
    return query_token[:5] == memory_token[:5]


def _matched_query_tokens(
    query_tokens: set[str],
    memory_tokens: set[str],
) -> set[str]:
    return {
        query_token
        for query_token in query_tokens
        if any(
            _token_matches(query_token, memory_token)
            for memory_token in memory_tokens
        )
    }


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _score_memory(
    memory: ProfileMemory,
    query_tokens: set[str],
    pinned_categories: set[str],
    now: datetime,
) -> tuple[float, tuple[str, ...], bool]:
    category_tokens = tokenize(memory.category)
    key_tokens = tokenize(memory.key.replace("_", " "))
    value_tokens = tokenize(
        json.dumps(memory.value, ensure_ascii=False, sort_keys=True)
    )
    category_matches = _matched_query_tokens(query_tokens, category_tokens)
    key_matches = _matched_query_tokens(query_tokens, key_tokens)
    value_matches = _matched_query_tokens(query_tokens, value_tokens)
    matched_tokens = category_matches | key_matches | value_matches

    reasons: list[str] = []
    relevance = 0.0
    if query_tokens:
        relevance += 45.0 * len(category_matches) / len(query_tokens)
        relevance += 35.0 * len(key_matches) / len(query_tokens)
        relevance += 20.0 * len(value_matches) / len(query_tokens)
        if matched_tokens:
            reasons.append("query_overlap")

    normalized_category = normalize_text(memory.category).replace(" ", "_")
    pinned = normalized_category in pinned_categories
    if pinned:
        relevance += 100.0
        reasons.append("pinned_category")

    updated_at = _aware_utc(memory.updated_at)
    age_days = max(0.0, (now - updated_at).total_seconds() / 86_400)
    relevance += 5.0 / (1.0 + age_days / 30.0)
    relevance += max(0.0, min(1.0, memory.confidence)) * 3.0
    if memory.verification_status in {"user_confirmed", "caregiver_confirmed"}:
        relevance += 4.0
        reasons.append("verified")
    reasons.append("recency_confidence")
    return relevance, tuple(reasons), bool(matched_tokens)


def select_profile_memories(
    memories: list[ProfileMemory],
    *,
    query: str | None,
    max_facts: int,
    max_tokens: int,
    pinned_categories: set[str],
    now: datetime | None = None,
) -> ProfileRetrievalResult:
    selection_time = now or datetime.now(timezone.utc)
    query_tokens = tokenize(query or "")
    normalized_pinned = {
        normalize_text(category).replace(" ", "_")
        for category in pinned_categories
    }

    scored: list[tuple[ProfileMemory, float, tuple[str, ...], bool, int]] = []
    for memory in memories:
        score, reasons, query_match = _score_memory(
            memory,
            query_tokens,
            normalized_pinned,
            selection_time,
        )
        token_cost = estimate_tokens(
            {memory.category: {memory.key: memory.value}}
        )
        scored.append((memory, score, reasons, query_match, token_cost))

    if query_tokens:
        candidates = [
            item
            for item in scored
            if item[3] or "pinned_category" in item[2]
        ]
    else:
        candidates = scored
    candidates.sort(
        key=lambda item: (
            item[1],
            _aware_utc(item[0].updated_at),
            item[0].id,
        ),
        reverse=True,
    )

    selected: list[SelectedProfileMemory] = []
    used_tokens = 0
    for memory, score, reasons, _, token_cost in candidates:
        if len(selected) >= max_facts:
            break
        if token_cost > max_tokens - used_tokens:
            continue
        selected.append(
            SelectedProfileMemory(
                memory=memory,
                score=round(score, 3),
                estimated_tokens=token_cost,
                reasons=reasons,
            )
        )
        used_tokens += token_cost

    return ProfileRetrievalResult(
        selected=tuple(selected),
        eligible_fact_count=len(memories),
        candidate_fact_count=len(candidates),
        estimated_tokens=used_tokens,
        max_facts=max_facts,
        max_tokens=max_tokens,
        query_used=bool(query_tokens),
    )
