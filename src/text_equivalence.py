"""Value/polarity guard shared by the duplicate mergers.

Two records can be worded almost identically and still state different facts
("limit is 100 rpm" vs "limit is 500 rpm", "backup is enabled" vs "backup is
not enabled"). Similarity (character ratio or embedding cosine) alone must
never be allowed to merge them: the texts have to agree on every value they
carry and on their polarity as well.
"""
from __future__ import annotations

import re

TOKEN_RE = re.compile(r"\w+", re.UNICODE)

NEGATIONS = frozenset({
    "not", "no", "never", "without", "cannot", "dont", "doesnt", "isnt", "wont", "disabled",
    "не", "нет", "без", "никогда", "отключен", "отключено", "выключен",
})


def value_signature(text: str) -> tuple[frozenset[str], int]:
    """Tokens that carry a value (anything with a digit) and the polarity of the text."""
    words = TOKEN_RE.findall((text or "").lower())
    values = frozenset(word for word in words if any(ch.isdigit() for ch in word))
    negations = sum(1 for word in words if word in NEGATIONS)
    return values, negations % 2


def same_values(text_a: str, text_b: str) -> bool:
    """True when both texts carry the same values and the same polarity."""
    return value_signature(text_a) == value_signature(text_b)
