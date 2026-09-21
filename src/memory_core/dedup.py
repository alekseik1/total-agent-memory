"""v11.0 Phase 3 — Deterministic dedup helpers.

Two cheap operations that run in the save hot path:

* :func:`exact_dedup` — returns (sha256 of raw content, sha256 of
  normalized content). The normalized hash collapses whitespace, lowercases
  and strips so trivially different inputs share a key.
* :func:`repeats` — whether a new record says exactly what a stored one
  says. `Store._find_duplicate` finds candidates with FTS5 and keeps only
  those that pass this check.
* :func:`find_duplicate` — wraps `Store._find_duplicate`. The wrapper
  isolates that call site so future replacements don't need to touch
  every save path.

No LLM, no network. Pure local logic.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Any, Optional

_RE_WS = re.compile(r"\s+")
_RE_TOKEN = re.compile(r"\w+(?:[./:-]\w+)*", re.UNICODE)


def normalize(text: str) -> str:
    """Lowercase, strip, collapse whitespace runs to single spaces."""
    if not text:
        return ""
    return _RE_WS.sub(" ", text.strip().lower())


# A repeat contains every word of the new text, so a few of them are enough
# to find the candidates.
MAX_CANDIDATE_TERMS = 32
MIN_CANDIDATE_TERM_LEN = 3


def _tokens(text: str) -> list[str]:
    folded = unicodedata.normalize("NFKC", text or "").casefold().replace("ё", "е")
    return _RE_TOKEN.findall(folded)


def candidate_match_query(text: str) -> str:
    """FTS5 MATCH expression that finds records containing the words of `text`.

    Words with "е" are left out: `repeats` treats "ё" and "е" as one letter,
    the FTS5 index does not. Short words are left out when longer ones
    remain. Empty when no word is usable.
    """
    words = [token for token in _tokens(text) if "е" not in token]
    # Short words ("на", "7") are in most records and only slow the match.
    selective = [word for word in words if len(word) >= MIN_CANDIDATE_TERM_LEN] or words
    terms = list(dict.fromkeys(selective))[:MAX_CANDIDATE_TERMS]
    return " ".join('"' + term.replace('"', '""') + '"' for term in terms)


def repeats(new: str, stored: str) -> bool:
    """True when `new` states nothing that `stored` does not.

    Both texts must have the same words in the same order, ignoring case,
    punctuation and whitespace. Articles count: "Monk" and "The Monk" can be
    different works. A similarity score is the wrong test: "Messi's
    citizenship is Argentina" and "... is Armenia" score 0.96 and are an
    update, not a repeat.
    """
    tokens = _tokens(new)
    return bool(tokens) and tokens == _tokens(stored)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def exact_dedup(content: str) -> tuple[str, str]:
    """Return (raw_sha256, normalized_sha256) for the given content."""
    return _sha256(content or ""), _sha256(normalize(content or ""))


def find_duplicate(
    db_or_store: Any,
    content: str,
    ktype: str,
    project: str,
) -> Optional[int]:
    """Look up a near-duplicate via the legacy `Store._find_duplicate`.

    Accepts either a `Store` instance (preferred — gives access to the
    full FTS+Jaccard ladder) or a raw sqlite connection (falls back to a
    minimal FTS-only check).
    """
    # Preferred: full store-level dedup.
    finder = getattr(db_or_store, "_find_duplicate", None)
    if callable(finder):
        try:
            return finder(content, ktype, project)
        except Exception:  # noqa: BLE001 — never let dedup raise into save
            return None

    # Fallback: ad-hoc FTS5 nearest-match if we only have a connection.
    try:
        words = [w for w in (content or "").split()[:10] if len(w) > 2]
        if not words:
            return None
        fts_q = " OR ".join(
            re.sub(r'[^a-zA-Z0-9_]+', "", w) or w for w in words
        )
        rows = db_or_store.execute(
            """
            SELECT k.id FROM knowledge_fts f
            JOIN knowledge k ON k.id = f.rowid
            WHERE f.content MATCH ? AND k.status='active'
              AND k.project=? AND k.type=?
            ORDER BY rank LIMIT 1
            """,
            (fts_q, project, ktype),
        ).fetchall()
        if rows:
            return int(rows[0][0])
    except Exception:  # noqa: BLE001
        pass
    return None


__all__ = ["candidate_match_query", "exact_dedup", "find_duplicate", "normalize", "repeats"]
