"""Quality gate — pre-save scorer applying the "6-Month Test".

Inspired by Beever Atlas's `FactExtractor` quality scoring. Every record
that hits `save_knowledge` is scored synchronously against three axes
before deduplication runs:

  * specificity   — concrete (file paths, numbers, names) vs vague
  * actionability — would a future reader know what to *do* with it
  * verifiability — can it be checked against the codebase / a system

Each axis is 0.0–1.0; the total is the arithmetic mean. Records below
`MEMORY_QUALITY_THRESHOLD` (default 0.5) are rejected and journaled to
`quality_gate_log` so the prompt and threshold can be tuned against real
rejection data.

Behavior is controlled by environment variables (all optional):

  * MEMORY_QUALITY_GATE_ENABLED   — auto|true|false (default: auto)
  * MEMORY_QUALITY_THRESHOLD      — float (default: 0.5)
  * MEMORY_QUALITY_MIN_CHARS      — skip below this size (default: 80)
  * MEMORY_QUALITY_BYPASS_TYPES   — csv list never scored
                                    (default: transcript,raw,observation)
  * MEMORY_QUALITY_LOG_ALL        — 1 to journal passes too (default: 0)
  * MEMORY_QUALITY_TIMEOUT_SEC    — per-call LLM timeout (default: 20)

The gate is **graceful**: any provider error, JSON-parse failure, or
unavailable LLM yields a `skip` decision so the underlying save still
proceeds. Better to occasionally let noise through than to drop a real
fact because Ollama is offline.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Any

import config

LOG = lambda msg: sys.stderr.write(f"[quality-gate] {msg}\n")

_DEFAULT_BYPASS_TYPES = ("transcript", "raw", "observation")
_DEFAULT_THRESHOLD = 0.5
_DEFAULT_MIN_CHARS = 80
_DEFAULT_TIMEOUT = 20.0
_MAX_LLM_INPUT_CHARS = 4000  # truncate long content before scoring
_PREVIEW_CHARS = 240


# ──────────────────────────────────────────────
# Result types
# ──────────────────────────────────────────────


@dataclass
class QualityScore:
    """Outcome of a single quality-gate evaluation."""

    decision: str           # 'pass' | 'drop' | 'skip' | 'error'
    total: float | None     # mean of the three axes, or None when skipped
    specificity: float | None
    actionability: float | None
    verifiability: float | None
    reason: str
    threshold: float
    provider: str | None = None
    model: str | None = None
    latency_ms: int | None = None
    llm_down: bool = False  # skipped because the provider could not be reached

    @property
    def retryable(self) -> bool:
        """True when the verdict reflects an LLM failure, not a judgement of the content."""
        return self.decision == "error" or self.llm_down

    @property
    def passed(self) -> bool:
        # 'skip' and 'error' both let the save through — the gate fails open.
        return self.decision in ("pass", "skip", "error")


# ──────────────────────────────────────────────
# Env knobs
# ──────────────────────────────────────────────


def _enabled_mode() -> str:
    return os.environ.get("MEMORY_QUALITY_GATE_ENABLED", "auto").strip().lower()


def _threshold() -> float:
    raw = os.environ.get("MEMORY_QUALITY_THRESHOLD")
    if not raw:
        return _DEFAULT_THRESHOLD
    try:
        v = float(raw)
    except ValueError:
        return _DEFAULT_THRESHOLD
    # Clamp to [0, 1] — anything else makes the gate trivially pass/fail.
    return max(0.0, min(1.0, v))


def _min_chars() -> int:
    raw = os.environ.get("MEMORY_QUALITY_MIN_CHARS")
    if not raw:
        return _DEFAULT_MIN_CHARS
    try:
        return max(0, int(raw))
    except ValueError:
        return _DEFAULT_MIN_CHARS


def _bypass_types() -> tuple[str, ...]:
    raw = os.environ.get("MEMORY_QUALITY_BYPASS_TYPES")
    if raw is None:
        return _DEFAULT_BYPASS_TYPES
    items = tuple(t.strip().lower() for t in raw.split(",") if t.strip())
    return items


def _log_all() -> bool:
    return os.environ.get("MEMORY_QUALITY_LOG_ALL", "0").strip() == "1"


def _timeout() -> float:
    raw = os.environ.get("MEMORY_QUALITY_TIMEOUT_SEC")
    if not raw:
        return _DEFAULT_TIMEOUT
    try:
        return max(1.0, float(raw))
    except ValueError:
        return _DEFAULT_TIMEOUT


# ──────────────────────────────────────────────
# Provider plumbing (cached, lazy)
# ──────────────────────────────────────────────

_provider_cache: dict[str, Any] = {}


def _get_provider():
    """Resolve the LLM provider for quality scoring.

    Reuses the same per-phase routing as deep_enricher: callers who want
    quality scoring on a different model than enrichment can set
    `MEMORY_QUALITY_PROVIDER` directly.
    """
    cached = _provider_cache.get("quality")
    if cached is not None:
        return cached
    from llm_provider import make_provider
    provider_name = os.environ.get(
        "MEMORY_QUALITY_PROVIDER",
        config.get_phase_provider("enrich"),
    )
    provider = make_provider(provider_name)
    _provider_cache["quality"] = provider
    return provider


def _model_name() -> str | None:
    return os.environ.get("MEMORY_QUALITY_MODEL") or config.get_phase_model("enrich")


def _reset_provider_cache() -> None:
    """Test helper — clear the provider cache."""
    _provider_cache.clear()


# ──────────────────────────────────────────────
# Prompt
# ──────────────────────────────────────────────


_SCORE_PROMPT = """You are evaluating a memory record before it is saved to a long-term
knowledge base used by an autonomous coding assistant. Apply the
"6-Month Test": would a teammate joining the project six months from
now still benefit from this fact?

Score the record on three axes from 0.0 to 1.0:

  - specificity:   concrete details (file paths, numbers, names,
                   commit SHAs, error messages) score high; vague
                   reflections or generic statements score low.
  - actionability: a reader can act on it — apply a fix, follow a
                   procedure, avoid a pitfall — score high; pure
                   commentary scores low.
  - verifiability: can be checked against the code, a system, or
                   external reality (URL, test, dashboard) score
                   high; opinions and feelings score low.

Type of record: {ktype}
Project:        {project}

Record content (may be truncated):
---
{content}
---

Respond with a single JSON object on one line, no prose around it:
{{"specificity": 0.0-1.0, "actionability": 0.0-1.0, "verifiability": 0.0-1.0, "reason": "one short sentence"}}
"""


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────


def should_score(content: str, ktype: str | None) -> tuple[bool, str]:
    """Quick gate — decide whether to spend an LLM call at all.

    Returns (should_run, reason_when_skipping).
    """
    mode = _enabled_mode()
    if mode in ("false", "0", "off", "no"):
        return False, "disabled by MEMORY_QUALITY_GATE_ENABLED"

    if not content or len(content) < _min_chars():
        return False, f"content < min_chars ({_min_chars()})"

    bypass = _bypass_types()
    if ktype and ktype.lower() in bypass:
        return False, f"type '{ktype}' in bypass list"

    if mode == "auto":
        # Auto: only run when an LLM is actually available.
        try:
            if not config.has_llm("enrich"):
                return False, "LLM unavailable (auto-mode)"
        except Exception as exc:  # pragma: no cover — defensive
            return False, f"has_llm check failed: {exc}"

    return True, ""


def _truncate_for_llm(content: str) -> str:
    if len(content) <= _MAX_LLM_INPUT_CHARS:
        return content
    return content[: _MAX_LLM_INPUT_CHARS] + "\n…[truncated]"


def _parse_score(raw: str) -> dict[str, float | str] | None:
    """Extract the JSON object from an LLM response. Tolerant of prose."""
    if not raw:
        return None
    # Find the first {...} block — handles models that wrap JSON in prose.
    match = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    return obj


def _coerce_axis(value: Any) -> float | None:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, v))


def score_quality(
    content: str,
    ktype: str | None = None,
    project: str | None = None,
) -> QualityScore:
    """Run the quality gate against `content`. Synchronous — caller must be
    prepared for an LLM round-trip on the hot path.

    Returns a :class:`QualityScore`; callers should consult `passed` to
    decide whether to proceed with the save and `decision`/`total` for
    audit-log persistence.
    """
    threshold = _threshold()

    should_run, reason = should_score(content, ktype)
    if not should_run:
        return QualityScore(
            decision="skip",
            total=None,
            specificity=None,
            actionability=None,
            verifiability=None,
            reason=reason,
            threshold=threshold,
        )

    try:
        provider = _get_provider()
    except Exception as exc:
        LOG(f"provider build failed: {exc}")
        return QualityScore(
            decision="skip",
            total=None,
            specificity=None,
            actionability=None,
            verifiability=None,
            reason=f"provider build failed: {exc}",
            threshold=threshold,
            llm_down=True,
        )

    if not provider.available():
        return QualityScore(
            decision="skip",
            total=None,
            specificity=None,
            actionability=None,
            verifiability=None,
            reason=f"provider '{getattr(provider, 'name', '?')}' unavailable",
            threshold=threshold,
            llm_down=True,
        )

    prompt = _SCORE_PROMPT.format(
        ktype=ktype or "fact",
        project=project or "general",
        content=_truncate_for_llm(content),
    )

    started = time.monotonic()
    try:
        raw = provider.complete(
            prompt,
            model=_model_name(),
            max_tokens=160,
            temperature=0.0,
            timeout=_timeout(),
        )
    except Exception as exc:
        latency = int((time.monotonic() - started) * 1000)
        LOG(f"LLM call failed after {latency}ms: {exc}")
        return QualityScore(
            decision="error",
            total=None,
            specificity=None,
            actionability=None,
            verifiability=None,
            reason=f"LLM error: {exc}",
            threshold=threshold,
            provider=getattr(provider, "name", None),
            model=_model_name(),
            latency_ms=latency,
        )

    latency = int((time.monotonic() - started) * 1000)
    parsed = _parse_score(raw)
    if not parsed:
        return QualityScore(
            decision="error",
            total=None,
            specificity=None,
            actionability=None,
            verifiability=None,
            reason=f"unparsable LLM response: {raw[:140]!r}",
            threshold=threshold,
            provider=getattr(provider, "name", None),
            model=_model_name(),
            latency_ms=latency,
        )

    spec = _coerce_axis(parsed.get("specificity"))
    act = _coerce_axis(parsed.get("actionability"))
    ver = _coerce_axis(parsed.get("verifiability"))
    reason_text = str(parsed.get("reason", "")).strip()[:240]

    axes = [a for a in (spec, act, ver) if a is not None]
    if len(axes) < 3:
        return QualityScore(
            decision="error",
            total=None,
            specificity=spec,
            actionability=act,
            verifiability=ver,
            reason=f"missing axes in response (got {len(axes)}/3): {raw[:140]!r}",
            threshold=threshold,
            provider=getattr(provider, "name", None),
            model=_model_name(),
            latency_ms=latency,
        )

    total = sum(axes) / 3.0
    decision = "pass" if total >= threshold else "drop"
    return QualityScore(
        decision=decision,
        total=total,
        specificity=spec,
        actionability=act,
        verifiability=ver,
        reason=reason_text or "scored",
        threshold=threshold,
        provider=getattr(provider, "name", None),
        model=_model_name(),
        latency_ms=latency,
    )


# ──────────────────────────────────────────────
# Audit-log persistence
# ──────────────────────────────────────────────


def log_decision(
    db,
    score: QualityScore,
    *,
    project: str | None,
    ktype: str | None,
    content: str,
    knowledge_id: int | None = None,
) -> int | None:
    """Persist a single gate decision to `quality_gate_log`; returns the row id.

    Always logs `drop` and `error` decisions; logs `pass`/`skip` only when
    `MEMORY_QUALITY_LOG_ALL=1` (sampling for prompt tuning). Returns None when
    nothing was written.
    """
    if score.decision in ("pass", "skip") and not _log_all():
        return None

    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    preview = (content or "")[:_PREVIEW_CHARS]

    try:
        cursor = db.execute(
            """
            INSERT INTO quality_gate_log (
                created_at, decision, score_total,
                score_specificity, score_actionability, score_verifiability,
                threshold, project, type, content_chars, content_preview,
                reason, knowledge_id, provider, model, latency_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                now,
                score.decision,
                score.total,
                score.specificity,
                score.actionability,
                score.verifiability,
                score.threshold,
                project,
                ktype,
                len(content or ""),
                preview,
                score.reason,
                knowledge_id,
                score.provider,
                score.model,
                score.latency_ms,
            ),
        )
        db.commit()
        return cursor.lastrowid
    except Exception as exc:
        LOG(f"audit log insert failed: {exc}")
        return None
