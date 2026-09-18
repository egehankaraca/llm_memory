"""Deterministic conflict detection and profile-memory consolidation policy."""

from __future__ import annotations

import enum
import json
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any


class ConsolidationAction(str, enum.Enum):
    CREATE = "create"
    UNCHANGED = "unchanged"
    SUPERSEDE = "supersede"


@dataclass(frozen=True)
class ActiveMemory:
    id: str
    category: str
    key: str
    value: Any


@dataclass(frozen=True)
class ConsolidationPlan:
    action: ConsolidationAction
    category: str
    key: str
    matched_fact_id: str | None
    matched_by: str | None
    similarity: float | None
    reason: str


CATEGORY_ALIASES = {
    "preferences": "preference",
    "preference": "preference",
    "habits": "routine",
    "habit": "routine",
    "routines": "routine",
    "routine": "routine",
    "behavior": "routine",
    "behaviour": "routine",
    "communications": "communication",
    "communication": "communication",
    "relationships": "relationship",
    "relationship": "relationship",
    "family": "relationship",
    "medications": "medication",
    "medicines": "medication",
    "medicine": "medication",
    "medication": "medication",
}

SLOT_ALIASES = {
    ("communication", "title_of_address"): "form_of_address",
    ("communication", "addressing_preference"): "form_of_address",
    ("communication", "preferred_form_of_address"): "form_of_address",
    ("routine", "morning_wake_up_time"): "morning_wake_time",
    ("routine", "morning_wakeup_time"): "morning_wake_time",
    ("routine", "wake_up_hour"): "morning_wake_time",
    ("routine", "wakeup_hour"): "morning_wake_time",
}

TOKEN_ALIASES = {
    "hours": "time",
    "hour": "time",
    "saat": "time",
    "zaman": "time",
    "wakeup": "wake",
    "waking": "wake",
    "kalkis": "wake",
    "kalkma": "wake",
    "sabah": "morning",
}

KEY_STOP_WORDS = {"of", "the", "user", "users", "up"}
CORRECTION_PATTERN = re.compile(
    r"\b(artik|bundan sonra|eskiden|degil|yerine|guncelle|duzelt)\b"
)


def normalize_text(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value.casefold())
    ascii_like = "".join(
        character
        for character in decomposed
        if not unicodedata.combining(character)
    ).replace("ı", "i")
    return re.sub(r"\s+", " ", ascii_like).strip()


def normalize_identifier(value: str, fallback: str) -> str:
    normalized = normalize_text(value)
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")
    return (normalized or fallback)[:100]


def canonical_category(category: str) -> str:
    normalized = normalize_identifier(category, "uncategorized")
    return CATEGORY_ALIASES.get(normalized, normalized)


def canonical_key(category: str, key: str) -> str:
    normalized = normalize_identifier(key, "unknown")
    return SLOT_ALIASES.get((canonical_category(category), normalized), normalized)


def key_tokens(category: str, key: str) -> set[str]:
    canonical = canonical_key(category, key)
    return {
        TOKEN_ALIASES.get(token, token)
        for token in canonical.split("_")
        if token and token not in KEY_STOP_WORDS
    }


def slot_similarity(
    left_category: str,
    left_key: str,
    right_category: str,
    right_key: str,
) -> float:
    if canonical_category(left_category) != canonical_category(right_category):
        return 0.0
    left = canonical_key(left_category, left_key)
    right = canonical_key(right_category, right_key)
    if left == right:
        return 1.0
    left_tokens = key_tokens(left_category, left_key)
    right_tokens = key_tokens(right_category, right_key)
    union = left_tokens | right_tokens
    token_score = len(left_tokens & right_tokens) / len(union) if union else 0.0
    sequence_score = SequenceMatcher(None, left, right).ratio()
    return max(token_score, sequence_score)


def normalized_value(
    value: Any,
    *,
    ignore_terminal_sentence_punctuation: bool = False,
) -> str:
    if isinstance(value, str) and ignore_terminal_sentence_punctuation:
        # Extractors are allowed to preserve or omit terminal sentence
        # punctuation.  That presentation-only variance must not turn an
        # otherwise identical profile fact into a conflict.  Internal
        # punctuation (times, decimals, phone numbers, etc.) stays intact.
        return normalize_text(value).rstrip(" \t\r\n.\u2026")
    try:
        serialized = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError):
        serialized = str(value)
    return normalize_text(serialized)


def has_explicit_correction(text: str) -> bool:
    return CORRECTION_PATTERN.search(normalize_text(text)) is not None


def _hinted_match(
    memories: list[ActiveMemory],
    matched_memory_id: str | None,
    category: str,
    key: str,
) -> ActiveMemory | None:
    if not matched_memory_id:
        return None
    hinted = next(
        (memory for memory in memories if memory.id == matched_memory_id),
        None,
    )
    if hinted is None:
        return None
    if (
        canonical_category(hinted.category) == canonical_category(category)
        and slot_similarity(hinted.category, hinted.key, category, key) >= 0.35
    ):
        return hinted
    return None


def _similar_match(
    memories: list[ActiveMemory],
    category: str,
    key: str,
) -> tuple[ActiveMemory | None, float | None, str | None]:
    scored = sorted(
        (
            (slot_similarity(category, key, memory.category, memory.key), memory)
            for memory in memories
        ),
        key=lambda item: item[0],
        reverse=True,
    )
    if not scored or scored[0][0] < 0.65:
        return None, None, None
    if len(scored) > 1 and scored[1][0] >= scored[0][0] - 0.08:
        return None, None, None
    score, memory = scored[0]
    matched_by = "canonical_slot" if score == 1.0 else "key_similarity"
    return memory, score, matched_by


def resolve_consolidation(
    *,
    category: str,
    key: str,
    value: Any,
    source_text: str,
    active_memories: list[ActiveMemory],
    matched_memory_id: str | None = None,
    relation_to_existing: str = "none",
    ignore_terminal_sentence_punctuation: bool = False,
) -> ConsolidationPlan:
    hinted = _hinted_match(active_memories, matched_memory_id, category, key)
    if hinted is not None:
        matched = hinted
        similarity = slot_similarity(category, key, hinted.category, hinted.key)
        matched_by = "extractor_reference"
    else:
        matched, similarity, matched_by = _similar_match(
            active_memories,
            category,
            key,
        )

    if matched is None:
        return ConsolidationPlan(
            action=ConsolidationAction.CREATE,
            category=canonical_category(category),
            key=canonical_key(category, key),
            matched_fact_id=None,
            matched_by=None,
            similarity=None,
            reason="Aynı kavramı temsil eden aktif memory bulunamadı.",
        )

    if normalized_value(
        value,
        ignore_terminal_sentence_punctuation=ignore_terminal_sentence_punctuation,
    ) == normalized_value(
        matched.value,
        ignore_terminal_sentence_punctuation=ignore_terminal_sentence_punctuation,
    ):
        return ConsolidationPlan(
            action=ConsolidationAction.UNCHANGED,
            category=matched.category,
            key=matched.key,
            matched_fact_id=matched.id,
            matched_by=matched_by,
            similarity=similarity,
            reason="Aynı memory değeri zaten aktif; yeni fact oluşturulmadı.",
        )

    return ConsolidationPlan(
        action=ConsolidationAction.SUPERSEDE,
        category=matched.category,
        key=matched.key,
        matched_fact_id=matched.id,
        matched_by=matched_by,
        similarity=similarity,
        reason=(
            "Kullanıcı mevcut memory için açık bir güncelleme belirtti."
            if has_explicit_correction(source_text)
            else "Aynı memory slotundaki yeni açık kullanıcı beyanı önceki değerin yerini aldı."
        ),
    )
