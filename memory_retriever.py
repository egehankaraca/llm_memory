"""Deterministic, bounded retrieval for structured long-term profile facts."""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping


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
    "her",
    "mi",
    "mı",
    "mu",
    "mü",
    "nasil",
    "ne",
    "su",
    "ve",
    # Generic time words are too weak to connect unrelated profile domains.
    # Domain aliases/semantic similarity still recover genuinely relevant
    # routines such as wake times and medication questions.
    "aksam",
    "aksamlari",
    "bugun",
    "sabah",
    "sabahlari",
    "saat",
    "simdi",
    "yarin",
}

# Retrieval aliases are intentionally separate from the extraction policy. They
# do not decide whether a sentence becomes memory; they only bridge common
# Turkish utterances and the stable English category/key names stored by the
# extractor. Semantic embeddings remain optional, while these aliases provide a
# deterministic and inexpensive baseline for small profile stores.
CONCEPT_ALIASES: dict[str, tuple[str, ...]] = {
    "form_of_address": (
        "form of address",
        "form_of_address",
        "hitap",
        "seslen",
        "bana de",
        "beni cagir",
        "adimi kullan",
    ),
    "tea": ("tea", "cay", "cayimi"),
    "tea_sweetener": (
        "sugar",
        "sweetener",
        "seker",
        "sekersiz",
        "tatlandirici",
        "tatlandir",
    ),
    "coffee": ("coffee", "kahve", "turk kahvesi"),
    "wake_time": (
        "wake time",
        "wake_up",
        "uyan",
        "kalk",
        "sabah kacta",
    ),
    "sleep_time": ("sleep time", "bedtime", "uyu", "yatma saati"),
    "medication": ("medication", "medicine", "ilac", "tablet", "hap"),
    "health": (
        "health",
        "saglik",
        "tansiyon",
        "agri",
        "hastalik",
        "doktor",
    ),
    "emergency_contact": (
        "emergency contact",
        "emergency_contact",
        "acil kisi",
        "acil durumda",
        "kimi ara",
        "acil",
        "yardim",
        "dustum",
        "kaydim",
        "kafami vurdum",
        "basimi vurdum",
        "kafami carptim",
        "basimi carptim",
        "kan",
        "kanama",
        "yaralandim",
        "nefes alamiyorum",
        "gogsum agriyor",
        "bilincimi kaybettim",
        "felc",
        "nobet",
        "ambulans",
        "112",
    ),
    "family": (
        "family",
        "aile",
        "kizim",
        "oglum",
        "esim",
        "kuzen",
        "torun",
    ),
    "food": (
        "food",
        "meal",
        "yemek",
        "yiyecek",
        "yesem",
        "yiyeyim",
        "yiyebilirim",
        "yemeliyim",
        "aciktim",
        "acim",
        "kahvalti",
        "ogle yemegi",
        "aksam yemegi",
    ),
    "drink": (
        "drink", "beverage", "icecek", "icerim", "icerdim", "iciyorum",
    ),
    "preference": ("preference", "tercih", "severim", "sevmem", "hoslan"),
    "accessibility": (
        "accessibility",
        "erisilebilirlik",
        "yavas konus",
        "kisa cumle",
        "sesli",
    ),
}


PROTECTED_QUERY_CONCEPTS = {
    "emergency_contact": frozenset({"emergency_contact"}),
    "health": frozenset({"health", "medication"}),
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
    alias_used: bool = False
    semantic_used: bool = False
    semantic_candidate_count: int = 0

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


def _concepts_for_text(value: str) -> set[str]:
    normalized = normalize_text(value).replace("_", " ")
    tokens = tokenize(normalized)
    concepts: set[str] = set()
    for concept, aliases in CONCEPT_ALIASES.items():
        for alias in aliases:
            normalized_alias = normalize_text(alias).replace("_", " ")
            alias_tokens = tokenize(normalized_alias)
            if not alias_tokens:
                continue
            if len(alias_tokens) == 1:
                alias_token = next(iter(alias_tokens))
                if any(_token_matches(alias_token, token) for token in tokens):
                    concepts.add(concept)
                    break
                continue
            if all(
                any(_token_matches(alias_token, token) for token in tokens)
                for alias_token in alias_tokens
            ):
                concepts.add(concept)
                break
    return concepts


def _required_query_concepts(memory: ProfileMemory) -> frozenset[str]:
    """Require an explicit domain signal before returning protected facts.

    Values are deliberately excluded from this check: a person's name or a
    generic time word inside a sensitive value must not make that value broadly
    retrievable. Category/key are the structured policy boundary.
    """
    identity_concepts = _concepts_for_text(
        f"{memory.category} {memory.key.replace('_', ' ')}"
    )
    if "emergency_contact" in identity_concepts:
        return PROTECTED_QUERY_CONCEPTS["emergency_contact"]
    if identity_concepts & {"health", "medication"}:
        return PROTECTED_QUERY_CONCEPTS["health"]
    return frozenset()


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _score_memory(
    memory: ProfileMemory,
    query_tokens: set[str],
    query_concepts: set[str],
    pinned_categories: set[str],
    now: datetime,
) -> tuple[float, tuple[str, ...], bool, bool]:
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

    memory_concepts = _concepts_for_text(
        " ".join(
            (
                memory.category,
                memory.key.replace("_", " "),
                json.dumps(memory.value, ensure_ascii=False, sort_keys=True),
            )
        )
    )
    concept_matches = query_concepts & memory_concepts
    if query_concepts and concept_matches:
        relevance += 55.0 * len(concept_matches) / len(query_concepts)
        reasons.append("concept_alias")

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
    return (
        relevance,
        tuple(reasons),
        bool(matched_tokens),
        bool(concept_matches),
    )


def select_profile_memories(
    memories: list[ProfileMemory],
    *,
    query: str | None,
    max_facts: int,
    max_tokens: int,
    pinned_categories: set[str],
    semantic_scores: Mapping[str, float] | None = None,
    semantic_min_similarity: float = 0.40,
    enforce_protected_query_gate: bool = True,
    now: datetime | None = None,
) -> ProfileRetrievalResult:
    selection_time = now or datetime.now(timezone.utc)
    query_tokens = tokenize(query or "")
    query_concepts = _concepts_for_text(query or "")
    bounded_semantic_scores = {
        memory_id: max(-1.0, min(1.0, float(score)))
        for memory_id, score in (semantic_scores or {}).items()
    }
    normalized_pinned = {
        normalize_text(category).replace(" ", "_")
        for category in pinned_categories
    }

    scored: list[
        tuple[ProfileMemory, float, tuple[str, ...], bool, bool, bool, int]
    ] = []
    for memory in memories:
        required_concepts = _required_query_concepts(memory)
        if (
            enforce_protected_query_gate
            and query_tokens
            and required_concepts
            and not query_concepts.intersection(required_concepts)
        ):
            continue
        score, reasons, query_match, alias_match = _score_memory(
            memory,
            query_tokens,
            query_concepts,
            normalized_pinned,
            selection_time,
        )
        semantic_score = bounded_semantic_scores.get(memory.id)
        semantic_match = (
            semantic_score is not None
            and semantic_score >= semantic_min_similarity
        )
        if semantic_match:
            score += 80.0 * semantic_score
            reasons = (*reasons, "semantic_similarity")
        token_cost = estimate_tokens(
            {memory.category: {memory.key: memory.value}}
        )
        scored.append(
            (
                memory,
                score,
                reasons,
                query_match,
                alias_match,
                semantic_match,
                token_cost,
            )
        )

    if query_tokens:
        candidates = [
            item
            for item in scored
            if item[3] or item[4] or item[5] or "pinned_category" in item[2]
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
    for memory, score, reasons, _, _, _, token_cost in candidates:
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
        alias_used=any(item[4] for item in candidates),
        semantic_used=any(item[5] for item in candidates),
        semantic_candidate_count=sum(1 for item in candidates if item[5]),
    )
