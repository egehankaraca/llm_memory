"""Small deterministic emergency gate for the local assistant demo.

This module deliberately does not place real telephone calls.  It detects a
small set of high-signal emergencies, selects a confirmed emergency contact
from the already-built memory context, and returns an explicit call simulation.
Memory extraction remains asynchronous and independent from this fast path.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata
from typing import Any


PHONE_PATTERN = re.compile(
    r"(?<!\d)(?:\+?90[\s().-]*)?0?5\d{2}(?:[\s().-]*\d){7}(?!\d)"
)
PAST_ONLY_PATTERN = re.compile(
    r"\b(?:dun|onceki gun|gecen (?:hafta|ay|yil)|eskiden)\b"
)
CURRENT_MARKER_PATTERN = re.compile(r"\b(?:simdi|su an|hala|halen)\b")
FALL_PATTERN = re.compile(
    r"\b(?:dust[uü]m|dustu|yere dustum|kayip dustum|kalkamiyorum)\b"
)
BLEEDING_PATTERN = re.compile(
    r"\b(?:kaniyor|kanama|kan kaybediyorum|kan durmuyor|her yer kan|"
    r"kan icinde|kan oldu)\b"
)
SEVERE_BLEEDING_PATTERN = re.compile(
    r"\b(?:kan durmuyor|cok kaniyor|her yer kan|kan kaybediyorum|kan icinde)\b"
)
BREATHING_PATTERN = re.compile(
    r"\b(?:nefes alamiyorum|nefesim kesildi|boguluyorum)\b"
)
UNCONSCIOUS_PATTERN = re.compile(
    r"\b(?:bayildim|bilincimi kaybettim|bilinci kapali)\b"
)
CHEST_PATTERN = re.compile(
    r"\b(?:gogsum|gogusum|kalbim)\s+(?:cok\s+)?(?:agriyor|sikisiyor)\b"
)
NEGATED_PATTERN = re.compile(
    r"\b(?:dusmedim|kanamiyor|kanama yok)\b"
)
NAME_PATTERN = re.compile(
    r"\b(?:kızım|oğlum|eşim|kardeşim|komşum|arkadaşım|"
    r"kizim|oglum|esim|kardesim|komsum|arkadasim)\s+"
    r"([A-Za-zÇĞİÖŞÜçğıöşü]+)\b",
    re.IGNORECASE,
)


def normalize(text: str) -> str:
    # Unicode decomposition removes accents but does not transliterate Turkish
    # dotless i. Translate it explicitly before applying the ASCII-like rules.
    folded = text.casefold().translate(str.maketrans({"ı": "i"}))
    decomposed = unicodedata.normalize("NFKD", folded)
    return "".join(character for character in decomposed if not unicodedata.combining(character))


def canonical_phone(value: str) -> str | None:
    match = PHONE_PATTERN.search(value)
    if match is None:
        return None
    digits = re.sub(r"\D", "", match.group(0))
    if len(digits) == 12 and digits.startswith("90"):
        return f"+{digits}"
    if len(digits) == 11 and digits.startswith("05"):
        return f"+90{digits[1:]}"
    if len(digits) == 10 and digits.startswith("5"):
        return f"+90{digits}"
    return None


def mask_phone(phone: str) -> str:
    return f"{phone[:5]} *** ** {phone[-2:]}"


@dataclass(frozen=True)
class EmergencyContact:
    fact_id: str
    name: str
    phone: str

    @property
    def masked_phone(self) -> str:
        return mask_phone(self.phone)


@dataclass(frozen=True)
class EmergencyAction:
    reason: str
    action: str
    reply: str
    contact_name: str | None = None
    masked_phone: str | None = None
    simulated: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "reason": self.reason,
            "action": self.action,
            "contact_name": self.contact_name,
            "masked_phone": self.masked_phone,
            "simulated": self.simulated,
        }


def emergency_reason(text: str, recent_messages: list[dict[str, Any]] | None = None) -> str | None:
    """Return a bounded high-signal emergency reason, otherwise ``None``."""
    current = normalize(text)
    past_only = bool(PAST_ONLY_PATTERN.search(current)) and not CURRENT_MARKER_PATTERN.search(current)
    if BREATHING_PATTERN.search(current):
        return "breathing_difficulty"
    if UNCONSCIOUS_PATTERN.search(current):
        return "loss_of_consciousness"
    if CHEST_PATTERN.search(current):
        return "chest_pain"
    if past_only:
        return None
    if SEVERE_BLEEDING_PATTERN.search(current):
        return "severe_bleeding"
    if NEGATED_PATTERN.search(current):
        return None
    if FALL_PATTERN.search(current) and BLEEDING_PATTERN.search(current):
        return "fall_with_bleeding"

    # A short follow-up such as "Kolum kanıyor" may rely on the preceding fall.
    if BLEEDING_PATTERN.search(current):
        recent_user_text = " ".join(
            str(message.get("content", ""))
            for message in (recent_messages or [])[-4:]
            if isinstance(message, dict) and message.get("role") == "user"
        )
        if FALL_PATTERN.search(normalize(recent_user_text)):
            return "fall_with_bleeding"
    return None


def actionable_emergency_contact(context: dict[str, Any]) -> EmergencyContact | None:
    """Select the first explicitly supplied, callable emergency contact."""
    facts = context.get("profile_facts", [])
    if not isinstance(facts, list):
        return None
    for fact in facts:
        if not isinstance(fact, dict) or fact.get("category") != "emergency_contact":
            continue
        provenance = fact.get("provenance")
        if not isinstance(provenance, dict) or provenance.get("verification_status") not in {
            "user_asserted", "user_confirmed", "caregiver_confirmed", "system_verified",
        }:
            continue
        value = fact.get("value")
        if isinstance(value, dict):
            phone_source = str(value.get("phone") or value.get("statement") or "")
            name = str(value.get("name") or value.get("relationship") or "Kayıtlı acil kişi")
        elif isinstance(value, str):
            phone_source = value
            name_match = NAME_PATTERN.search(value)
            name = name_match.group(1) if name_match else "Kayıtlı acil kişi"
        else:
            continue
        phone = canonical_phone(phone_source)
        fact_id = fact.get("fact_id")
        if phone is not None and isinstance(fact_id, str) and fact_id:
            return EmergencyContact(fact_id=fact_id, name=name, phone=phone)
    return None


def evaluate_emergency(
    text: str,
    context: dict[str, Any],
    *,
    action_already_started: bool = False,
) -> EmergencyAction | None:
    reason = emergency_reason(text, context.get("recent_messages"))
    if reason is None:
        return None
    contact = actionable_emergency_contact(context)
    if action_already_started:
        return EmergencyAction(
            reason=reason,
            action="already_active",
            reply=(
                "Acil durum işlemi bu oturumda zaten başlatıldı. "
                "Bu demo gerçek arama yapmaz; ciddi veya devam eden tehlikede 112'yi arayın."
            ),
            contact_name=contact.name if contact else None,
            masked_phone=contact.masked_phone if contact else None,
        )
    if contact is None:
        return EmergencyAction(
            reason=reason,
            action="no_emergency_contact",
            reply=(
                "Acil bir durum olabileceğini algıladım ancak kullanılabilir bir acil kişi "
                "bulamadım. Bu demo gerçek arama yapmaz; ciddi veya devam eden tehlikede "
                "112'yi arayın."
            ),
        )
    return EmergencyAction(
        reason=reason,
        action="call_simulated",
        reply=(
            f"Acil bir durum olabileceğini algıladım. {contact.name} için arama "
            "simülasyonu başlatıldı. Bu demo gerçek arama yapmaz; ciddi veya devam eden "
            "tehlikede 112'yi arayın."
        ),
        contact_name=contact.name,
        masked_phone=contact.masked_phone,
    )
