"""Detect user corrections / pushback in a prompt and draft an insight from them.

A correction ("I already told you...", "я же говорил", "сколько раз", swearing,
a shouted ALL-CAPS prompt, or a repeat marker plus a negated action such as
"again you didn't run the tests") is the cheapest signal that the agent keeps
making a mistake. :func:`capture` turns it into a *draft* insight in the
existing ``insights`` pipeline: an active row with low importance and
confidence plus the ``draft`` tag, so it surfaces in ``self_insight(list)``,
can be upvoted, and becomes promotable to a rule only after real upvotes.

Detection is pure regex (well under a millisecond per prompt); nothing here
calls a model.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Literal, Protocol

from memory_core.timestamps import format_utc
from rule_surfaces import tokenize
from secret_redaction import redact_secrets

LOG = logging.getLogger(__name__)

Strength = Literal["strong", "weak"]

CORRECTION_CATEGORY = "user_correction"
DRAFT_TAG = "draft"
CORRECTION_TAG = "user-correction"
STRENGTH_TAG_PREFIX = "strength:"
# Below the self_insight defaults (2 / 0.5) and far below promotion (5 / 0.8).
DRAFT_IMPORTANCE = 1
DRAFT_CONFIDENCE = 0.3
REPEAT_IMPORTANCE_STEP = 1
REPEAT_CONFIDENCE_STEP = 0.05
MAX_CONFIDENCE = 1.0
DEDUP_WINDOW = timedelta(minutes=90)
# Share of the smaller token set found in the other one.
DEDUP_MIN_OVERLAP = 0.5
MAX_INSIGHT_TEXT_CHARS = 400
# Pushback sits at the start or the end of a prompt; a pasted log in between is
# not scanned, which keeps detection well under a millisecond.
SCAN_HEAD_CHARS = 1000
SCAN_TAIL_CHARS = 500
SHOUT_MIN_LETTERS = 8
SHOUT_MIN_WORDS = 2
SHOUT_MIN_UPPER_RATIO = 0.7
DEFAULT_PROJECT = "general"

# Matched against casefolded text, so no IGNORECASE (it slows every branch).
_STRONG_BRANCHES: tuple[tuple[str, str], ...] = (
    ("already_told",
     (r"\bi(?:'ve|\s+have)?\s+(?:already|just|literally)\s+(?:told|said|asked|explained)\b"
      r"|\bi\s+(?:told|asked)\s+you\s+(?:already|before|twice|not\s+to)\b"
      r"|\bas\s+i\s+(?:already\s+)?(?:said|told\s+you)\b")),
    ("how_many_times",
     r"\bhow\s+many\s+times\b|\bfor\s+the\s+(?:last|hundredth|tenth|third|nth)\s+time\b"),
    ("ru_already_told",
     (r"\bя\s+(?:же|уже|ведь)\s+(?:тебе\s+)?(?:говорил|сказал|просил|писал|объяснял)а?\b"
      r"|\b(?:говорил|сказал|просил)а?\s+же\b"
      r"|\bтебе\s+(?:же\s+)?(?:говорили|сказали)\b")),
    ("ru_how_many_times",
     r"\bсколько\s+(?:раз|можно)\b|\bв\s+(?:сотый|десятый|который)\s+раз\b"),
    ("profanity",
     (r"\b(?:fuck\w*|shit\w*|wtf|bullshit|goddamn\w*|damn\s+it)\b"
      r"|\b(?:бля|блять|блядь|бляха|сука|суки|пиздец|пизд\w*|нахуй|нахуя|похуй|хуйня|"
      r"хуй\w*|ху[её]во|еба\w*|ёба\w*|ебан\w*|ёб\w*|заебал\w*|нахрен|охренел\w*)\b")),
)
_STRONG = re.compile("|".join(f"(?P<{name}>{branch})" for name, branch in _STRONG_BRANCHES))
_REPEAT_MARKER = re.compile(
    r"\b(?:again|once\s+more|still|keep\s+(?:doing|making|forgetting)|every\s+time|"
    r"опять|снова|ещ[её]\s+раз|вс[её]\s+ещ[её]|по-прежнему|постоянно|каждый\s+раз)\b"
)
_NEGATED_ACTION = re.compile(
    r"\b(?:don't|dont|do\s+not|didn't|didnt|did\s+not|doesn't|does\s+not|haven't|"
    r"not|never|stop|quit|won't|no\s+more|"
    r"не\s+[а-яё]+|нельзя|хватит|перестань\w*|прекрати\w*|никогда)\b"
)
_UPPER = re.compile(r"[A-ZА-ЯЁ]")
_LOWER = re.compile(r"[a-zа-яё]")
_WORD = re.compile(r"[^\W\d_]{2,}")
_WHITESPACE = re.compile(r"\s+")
_APOSTROPHES = str.maketrans({"\u2019": "'", "\u2018": "'", "\u02bc": "'"})


@dataclass(frozen=True)
class CorrectionSignal:
    strength: Strength
    markers: tuple[str, ...]
    excerpt: str


class Database(Protocol):
    """The DB-API subset used here (sqlite3.Connection or tam_db PgConnection)."""

    def execute(self, sql: str, parameters: Sequence[Any] = ..., /) -> Any: ...

    def commit(self) -> None: ...


def _is_shout(text: str) -> bool:
    upper = len(_UPPER.findall(text))
    cased = upper + len(_LOWER.findall(text))
    if cased < SHOUT_MIN_LETTERS or len(_WORD.findall(text)) < SHOUT_MIN_WORDS:
        return False
    return upper / cased >= SHOUT_MIN_UPPER_RATIO


def _excerpt(text: str) -> str:
    # Prompts can carry pasted credentials; the draft insight must not.
    return redact_secrets(_WHITESPACE.sub(" ", text).strip())[0][:MAX_INSIGHT_TEXT_CHARS]


def _scanned(text: str) -> str:
    if len(text) <= SCAN_HEAD_CHARS + SCAN_TAIL_CHARS:
        return text
    return f"{text[:SCAN_HEAD_CHARS]}\n{text[-SCAN_TAIL_CHARS:]}"


def detect(text: str | None) -> CorrectionSignal | None:
    """Correction signal in ``text``, or ``None``.

    Strong: "already told you" phrases (EN/RU), "how many times" / "сколько раз",
    profanity, or a shouted prompt (>= 70% upper-case letters). Weak: a repeat
    marker ("again", "опять", "снова") together with a negated action.
    """
    if not text or not text.strip():
        return None
    scanned = _scanned(text).translate(_APOSTROPHES)
    folded = scanned.casefold()
    strong = list(dict.fromkeys(
        name for match in _STRONG.finditer(folded) for name, value in match.groupdict().items()
        if value is not None))
    if _is_shout(scanned):
        strong.append("shout")
    if strong:
        return CorrectionSignal("strong", tuple(strong), _excerpt(text))
    if _REPEAT_MARKER.search(folded) and _NEGATED_ACTION.search(folded):
        return CorrectionSignal("weak", ("repeat", "negated_action"), _excerpt(text))
    return None


def _topic_tokens(text: str) -> set[str]:
    """Tokens of ``text`` without the correction phrasing, so that two corrections
    match on what they are about, not on "I already told you"."""
    folded = text.translate(_APOSTROPHES).casefold()
    for pattern in (_STRONG, _REPEAT_MARKER, _NEGATED_ACTION):
        folded = pattern.sub(" ", folded)
    return set(tokenize(folded))


def _overlap(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / min(len(left), len(right))


def _utc_now() -> datetime:
    return datetime.now(UTC)


def capture(db: Database, text: str | None, project: str | None, session_id: str,
            *, clock: Callable[[], datetime] = _utc_now) -> int | None:
    """Draft insight id for a correction in ``text``; ``None`` when there is none.

    A draft of the same project from the last 90 minutes with overlapping tokens
    is reused (and gains importance, since the user had to repeat themselves)
    instead of creating a new row.
    """
    signal = detect(text)
    if signal is None:
        return None
    project = project or DEFAULT_PROJECT
    now = clock()
    stamp = format_utc(now)
    tokens = _topic_tokens(signal.excerpt)

    recent = db.execute(
        "SELECT id, content FROM insights WHERE project = ? AND category = ? AND status = 'active' "
        "AND created_at >= ? ORDER BY created_at DESC",
        (project, CORRECTION_CATEGORY, format_utc(now - DEDUP_WINDOW))).fetchall()
    for row in recent:
        if _overlap(tokens, _topic_tokens(str(row[1]))) >= DEDUP_MIN_OVERLAP:
            insight_id = int(row[0])
            db.execute(
                "UPDATE insights SET importance = importance + ?, "
                "confidence = MIN(?, confidence + ?), updated_at = ? WHERE id = ?",
                (REPEAT_IMPORTANCE_STEP, MAX_CONFIDENCE, REPEAT_CONFIDENCE_STEP, stamp, insight_id))
            db.commit()
            LOG.info("correction capture: repeat of draft insight %s (project=%s)", insight_id, project)
            return insight_id

    tags = [DRAFT_TAG, CORRECTION_TAG, f"{STRENGTH_TAG_PREFIX}{signal.strength}"]
    context = f"Auto-captured user correction ({signal.strength}: {', '.join(signal.markers)})."
    cursor = db.execute(
        "INSERT INTO insights (session_id, content, context, category, importance, confidence, "
        "source_error_ids, project, tags, status, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, '[]', ?, ?, 'active', ?, ?)",
        (session_id, signal.excerpt, context, CORRECTION_CATEGORY, DRAFT_IMPORTANCE,
         DRAFT_CONFIDENCE, project, json.dumps(tags), stamp, stamp))
    db.commit()
    insight_id = int(cursor.lastrowid)
    LOG.info("correction capture: drafted insight %s (project=%s, %s)", insight_id, project,
             signal.strength)
    return insight_id
