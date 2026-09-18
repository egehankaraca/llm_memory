"""Small independent evidence/grammar checks, not a replacement for extraction."""

import re
import unicodedata


def normalize(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value.casefold())
    return "".join(c for c in decomposed if not unicodedata.combining(c)).replace("ı", "i")


QUESTION_WORDS = re.compile(
    r"\b(?:kac(?:ta|te|tan|ten|inci)?|nasil|neden|niye|hangi(?:si|sini|leri)?|"
    r"kim(?:in|i|e|den)?|nere(?:de|den|ye)|ne(?:yi|ye|ler)?|"
    r"m[ui](?:y?(?:im|um|iz|uz)|s(?:in|un|iniz|unuz)|"
    r"y?d[ui](?:m|n|k|niz|nuz|ler|lar)?|"
    r"y?s[ae](?:m|n|k|niz|nuz|ler|lar)?|"
    r"y?m[ui]s(?:im|um|sin|sun|iz|uz|siniz|sunuz|ler|lar)?|"
    r"d[ui]r)?)\b"
)
ENGLISH_QUESTION_START = re.compile(
    r"^\s*(?:who|what|when|where|why|how|which|is|are|am|was|were|"
    r"do|does|did|have|has|had|can|could|will|would|shall|should|may|might)\b",
    re.IGNORECASE,
)
CLAUSE_BOUNDARY = re.compile(r"[!?;\n]|(?<!\d)\.|\.(?!\d)")


def is_question_like(text: str) -> bool:
    return (
        "?" in text
        or bool(QUESTION_WORDS.search(normalize(text)))
        or bool(ENGLISH_QUESTION_START.search(text))
    )


def is_fragment(text: str) -> bool:
    return len(re.findall(r"\w+", text, flags=re.UNICODE)) < 2


def has_reported_speech(text: str) -> bool:
    """Conservative fallback guard for quoted/reported third-party statements."""
    normalized = normalize(text)
    reporting_verb = re.search(
        r"\b(?:dedi|soyledi|anlatti|belirtti|aktardi|said|told|reported)\b",
        normalized,
    )
    quotation = re.search(r"['\"“”‘’].+['\"“”‘’]", text)
    return bool(reporting_verb and quotation)


def supported_quote(quote: str, source: str) -> bool:
    # Whitespace/case normalization only: don't fabricate or repair spelling.
    clean = lambda value: re.sub(r"\s+", " ", unicodedata.normalize("NFC", value).casefold()).strip()
    return bool(quote.strip()) and clean(quote) in clean(source)


def evidence_clause(quote: str, source: str) -> str:
    # casefold can change character counts (e.g. Turkish capital İ). Match on
    # the original string so offsets cannot jump into a different clause.
    match = next((
        candidate for candidate in re.finditer(re.escape(quote), source, re.IGNORECASE)
        if supported_quote(quote, candidate.group())
    ), None)
    if match is None:
        return source  # Conservative for quotes matched only after whitespace normalization.
    position, end = match.span()
    # A quote may itself include its final punctuation. Do not accidentally
    # attach the next sentence (which can be a question) to that valid quote.
    # Dots inside a numeric time/decimal are not sentence boundaries.
    boundaries = [match.start() for match in CLAUSE_BOUNDARY.finditer(source)]
    left = max((index for index in boundaries if index < position), default=-1) + 1
    right = min((index for index in boundaries if index >= end - 1), default=len(source))
    return source[left:right + 1]


def has_mixed_question_clauses(text: str) -> bool:
    clauses = []
    start = 0
    for match in CLAUSE_BOUNDARY.finditer(text):
        clauses.append(text[start:match.end()])
        start = match.end()
    clauses.append(text[start:])
    clauses = [clause.strip() for clause in clauses if clause.strip()]
    return any(is_question_like(clause) for clause in clauses) and any(
        not is_question_like(clause) and not is_fragment(clause) for clause in clauses
    )


DAY_NAMES = {
    "pazartesi": "monday", "sali": "tuesday", "carsamba": "wednesday",
    "persembe": "thursday", "cuma": "friday", "cumartesi": "saturday", "pazar": "sunday",
}


def temporal_atoms(text: str) -> set[str]:
    normalized = normalize(text)
    atoms = set()
    if re.search(r"\b(?:her (?:gun|sabah|aksam)|gunluk|daily|every (?:day|morning|evening))\b", normalized):
        atoms.add("daily")
    if re.search(r"\b(?:her hafta|haftalik|weekly)\b", normalized):
        atoms.add("weekly")
    for turkish, english in DAY_NAMES.items():
        if re.search(rf"\b(?:{turkish}(?:leri|lari)?|{english})\b", normalized):
            atoms.update({"weekly", f"day:{english}"})
    return atoms


def numeric_atoms(text: str) -> set[int]:
    # Treat 7, 07:00 and 7.00 alike, without asserting complete semantic entailment.
    atoms = set()
    for token in re.findall(r"\d+(?:[:.]\d+)?", text):
        parts = re.split(r"[:.]", token)
        atoms.add(int(parts[0]))
        if len(parts) > 1 and int(parts[1]) != 0:
            atoms.add(int(parts[1]))
    return atoms


MEDICATION_DOSE_AMOUNT = re.compile(
    r"\b(?:\d+(?:[.,]\d+)?|bir|iki|uc|dort|bes|alti|yedi|sekiz|dokuz|on|"
    r"yarim|ceyrek|one|two|three|four|five|half|quarter)\s*"
    r"(?:mg|mcg|mikrogram|gr|gram|ml|mililitre|unit|units|unite|"
    r"tablet(?:i)?|hap(?:i)?|kapsul(?:u)?|tane|adet|damla|olcek)\b"
)


def has_medication_dose_amount(text: str) -> bool:
    """Detect an explicit dose amount without treating a clock as a dose.

    A bare number is intentionally insufficient: in a medication routine,
    ``saat 8`` is schedule evidence, while ``iki tablet`` or ``5 mg`` is a dose
    amount that stays behind the review gate.
    """
    return MEDICATION_DOSE_AMOUNT.search(normalize(text)) is not None


def protected_domain_hint(text: str) -> str | None:
    """Bounded DLP hints for explicit protected data; not semantic coverage."""
    normalized = normalize(text)
    phone = re.search(r"(?<!\d)(?:\+?90[\s()-]*)?0?5\d{2}[\s()-]*\d{3}[\s()-]*\d{2}[\s()-]*\d{2}(?!\d)", normalized)
    if re.search(r"\bacil\b", normalized) and (
        phone or re.search(r"\b(?:kisi(?:si)?|iletisim|irtibat|numara(?:si|sindan)?|ara)\b", normalized)
    ):
        return "emergency_contact"
    if re.search(r"\badres(?:im|imiz|i)?\b", normalized) or (
        re.search(r"\b(?:sokak|sokagi|cadde|caddesi|mahallesi|apartmani)\b", normalized)
        and numeric_atoms(text)
    ):
        return "location"
    if re.search(r"\b(?:iban(?:im)?|hesap numaram|banka hesabim)\b", normalized) or re.search(r"\bTR\d{2}(?:[ -]?\d){22}\b", text, re.IGNORECASE):
        return "financial"
    if re.search(r"\b(?:sifre(?:m|miz)?|parola(?:m|miz)?|password|api[_ -]?key|token(?:im)?|pin(?:im)?)\b", normalized):
        return "credential"
    return None
