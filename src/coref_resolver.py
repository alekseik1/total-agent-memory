"""Self-contained coreference rewrite — v10 P0.4.

Inspired by Beever Atlas's preprocessor stage, which rewrites pronouns and
deictics ("after this it broke", "we deprecated that") into self-contained
prose ("after migration 422000001 batchUpsert broke") *before* the record
hits semantic indexing. Without this, fragments retrieved by embedding
similarity arrive devoid of the discourse context that gave them meaning,
and the user reads gibberish.

Behaviour:

  * Cheap regex pre-filter (`needs_resolution()`) skips the LLM call when
    no pronouns or deictics are present — the bulk of saves never trigger
    a rewrite.
  * When triggered, the resolver pulls the last N records from the same
    session and asks the LLM to expand the input so it stands alone.
  * Failures fall through silently: the original content is preserved and
    a `[coref]` log line is emitted.

The resolver is **opt-in** by default: callers must pass `coref=True`
to `save_knowledge`, OR set `MEMORY_COREF_ENABLED=true`. This is
deliberately conservative — most explicit `memory_save` calls are
already self-contained, and the LLM round-trip on the hot path costs ~1s.
The auto-extract / transcript ingestion path is the obvious user.
"""

from __future__ import annotations

import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Any

import config

LOG = lambda msg: sys.stderr.write(f"[coref] {msg}\n")

_DEFAULT_HISTORY_LIMIT = 20
_DEFAULT_TIMEOUT = 25.0
_MAX_INPUT_CHARS = 4000
_MAX_HISTORY_CHARS_PER_RECORD = 240


# ──────────────────────────────────────────────
# Pre-filter (fast, regex-only)
# ──────────────────────────────────────────────

_PRONOUN_PATTERN = re.compile(
    r"\b("
    # English pronouns / deictics
    r"it|its|this|that|these|those|there|then|here|"
    r"he|she|him|her|them|they|their|theirs|"
    # Russian pronouns / deictics — most-common forms
    r"он|она|они|оно|его|её|ее|их|"
    r"это|этот|эта|эти|то|та|те|тот|"
    r"туда|сюда|здесь|тут|там|"
    r"потом|тогда|после|"
    # Common 'after this'/'before that' phrases
    r"after\s+(?:this|that)|before\s+(?:this|that)"
    r")\b",
    re.IGNORECASE | re.UNICODE,
)


def needs_resolution(content: str) -> bool:
    """True iff content contains at least one pronoun/deictic worth resolving."""
    if not content:
        return False
    return bool(_PRONOUN_PATTERN.search(content))


# ──────────────────────────────────────────────
# Env knobs
# ──────────────────────────────────────────────


def _enabled_default() -> bool:
    return os.environ.get("MEMORY_COREF_ENABLED", "false").strip().lower() in (
        "1", "true", "yes", "on", "force",
    )


def _history_limit() -> int:
    raw = os.environ.get("MEMORY_COREF_HISTORY_LIMIT")
    if not raw:
        return _DEFAULT_HISTORY_LIMIT
    try:
        return max(0, int(raw))
    except ValueError:
        return _DEFAULT_HISTORY_LIMIT


def _timeout() -> float:
    raw = os.environ.get("MEMORY_COREF_TIMEOUT_SEC")
    if not raw:
        return _DEFAULT_TIMEOUT
    try:
        return max(1.0, float(raw))
    except ValueError:
        return _DEFAULT_TIMEOUT


# ──────────────────────────────────────────────
# Provider
# ──────────────────────────────────────────────

_provider_cache: dict[str, Any] = {}


def _get_provider():
    cached = _provider_cache.get("coref")
    if cached is not None:
        return cached
    from llm_provider import make_provider
    name = os.environ.get(
        "MEMORY_COREF_PROVIDER",
        config.get_phase_provider("enrich"),
    )
    provider = make_provider(name)
    _provider_cache["coref"] = provider
    return provider


def _model_name() -> str | None:
    return os.environ.get("MEMORY_COREF_MODEL") or config.get_phase_model("enrich")


def _reset_provider_cache() -> None:
    _provider_cache.clear()


# ──────────────────────────────────────────────
# History fetch
# ──────────────────────────────────────────────


def _fetch_session_history(
    db, session_id: str, limit: int, project: str | None = None,
) -> list[str]:
    """Recent saves in the same session, oldest-first, content only."""
    if not session_id or limit <= 0 or project is None:
        return []
    try:
        rows = db.execute(
            """SELECT content FROM knowledge
               WHERE session_id = ? AND project = ? AND status = 'active'
               ORDER BY id DESC LIMIT ?""",
            (session_id, project, limit),
        ).fetchall()
    except Exception as exc:
        LOG(f"history fetch failed: {exc}")
        return []
    snippets: list[str] = []
    for row in reversed(rows):  # oldest-first reads more naturally for LLMs
        text = (row["content"] if isinstance(row, dict) or hasattr(row, "keys") else row[0]) or ""
        text = text.strip().replace("\n", " ")
        if not text:
            continue
        if len(text) > _MAX_HISTORY_CHARS_PER_RECORD:
            text = text[:_MAX_HISTORY_CHARS_PER_RECORD] + "…"
        snippets.append(text)
    return snippets


# ──────────────────────────────────────────────
# Prompt
# ──────────────────────────────────────────────


_REWRITE_PROMPT = """You rewrite a single memory record so it is fully
self-contained — every pronoun, deictic ("this", "that", "after that",
"его", "это", "там"), and dangling reference is replaced with the
concrete subject from the session context. Preserve every fact, file
path, number, code identifier, and proper noun exactly as written;
only resolve references.

LANGUAGE: keep the exact same language as the input record. If the
record is in Russian, output Russian. If English — English. Do NOT
translate. Do NOT switch language even partially.

If the context does not unambiguously identify a referent, leave that
particular pronoun alone — never invent. Do not add new facts. Do not
shorten, summarise, or reformat.

Recent records from the same session, oldest first:
---
{history}
---

Record to rewrite:
---
{content}
---

Output only the rewritten record body, no preamble, no markdown fence."""


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────


@dataclass
class CorefResult:
    decision: str           # 'rewritten' | 'noop' | 'skip' | 'error'
    content: str            # always populated — original on noop/skip/error
    reason: str
    latency_ms: int | None = None


def resolve(
    content: str,
    *,
    db,
    session_id: str | None,
    coref: bool | None = None,
    project: str | None = None,
) -> CorefResult:
    """Run coreference resolution on `content`.

    Caller passes the SQLite connection and session id so the resolver
    can read recent records. Returns the rewritten content (or original
    on no-op / skip / error). Never raises — `decision` carries the
    classification.
    """
    if not content:
        return CorefResult("skip", content or "", "empty content")

    enabled = _enabled_default() if coref is None else bool(coref)
    if not enabled:
        return CorefResult("skip", content, "coref disabled")

    if not needs_resolution(content):
        return CorefResult("noop", content, "no pronouns/deictics detected")

    if len(content) > _MAX_INPUT_CHARS:
        # Long records are usually structured (code/logs); rewriting them
        # risks subtle data corruption. Skip and rely on the original.
        return CorefResult(
            "skip", content,
            f"content > {_MAX_INPUT_CHARS} chars — too long for safe rewrite",
        )

    try:
        provider = _get_provider()
    except Exception as exc:
        return CorefResult("error", content, f"provider build failed: {exc}")

    if not provider.available():
        return CorefResult(
            "skip", content,
            f"provider '{getattr(provider, 'name', '?')}' unavailable",
        )

    history = _fetch_session_history(db, session_id or "", _history_limit(), project)
    if not history:
        # Without context the LLM cannot resolve anything reliably.
        return CorefResult("skip", content, "no session history available")

    prompt = _REWRITE_PROMPT.format(
        history="\n".join(f"- {snippet}" for snippet in history),
        content=content,
    )

    started = time.monotonic()
    try:
        rewritten = provider.complete(
            prompt,
            model=_model_name(),
            max_tokens=min(1024, len(content) // 2 + 200),
            temperature=0.0,
            timeout=_timeout(),
        )
    except Exception as exc:
        latency = int((time.monotonic() - started) * 1000)
        LOG(f"LLM call failed after {latency}ms: {exc}")
        return CorefResult("error", content, f"LLM error: {exc}", latency_ms=latency)

    latency = int((time.monotonic() - started) * 1000)
    rewritten = (rewritten or "").strip()
    # Strip accidental markdown fences if the model added them.
    if rewritten.startswith("```"):
        rewritten = re.sub(r"^```[a-zA-Z]*\n?", "", rewritten)
        rewritten = re.sub(r"\n?```$", "", rewritten).strip()

    if not rewritten:
        return CorefResult("error", content, "LLM returned empty rewrite", latency_ms=latency)

    # Sanity check: rewritten content must be at least as long as the
    # original. The whole point is *expansion*; a shorter result almost
    # always means truncation, which violates the prompt contract.
    if len(rewritten) < int(len(content) * 0.7):
        return CorefResult(
            "error", content,
            f"rewrite shorter than 70% of original "
            f"({len(rewritten)} vs {len(content)} chars) — suspected truncation",
            latency_ms=latency,
        )

    if rewritten == content:
        return CorefResult("noop", content, "LLM returned identical text", latency_ms=latency)

    return CorefResult("rewritten", rewritten, "rewritten by LLM", latency_ms=latency)
