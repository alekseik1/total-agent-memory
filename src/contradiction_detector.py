"""Auto-contradiction detection — v10 P1.5.

When a new `decision` or `solution` is saved, walk the top-K most
semantically-similar pre-existing records of the same type within the same
project, ask an LLM "does the new one supersede the old one?", and either:

  * **superseded** (LLM confidence ≥ MEMORY_CONTRADICTION_LLM_THRESHOLD,
    default 0.8) → mark the old record `status='superseded'`,
    `superseded_by=<new_id>`, write to `contradiction_log` as `superseded`.
  * **flagged** (0.5 ≤ confidence < 0.8) → leave both records active but
    write to `contradiction_log` as `flagged`. The dashboard can list these
    for human review.
  * **rejected** (< 0.5) → log as `rejected` and move on.

The detector is **fail-open**: any embedding/LLM/SQL error logs an `error`
decision and lets the underlying save complete normally. Better to leave a
contradiction unresolved than to break the write path.

Dependency injection is deliberate: `embed_fn`, `llm_fn`, and
`candidates_fn` are passed in by the caller. This keeps the module
testable without booting Store + FastEmbed + Ollama.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

LOG = lambda msg: sys.stderr.write(f"[contradiction] {msg}\n")

# Reason returned by should_run() when the LLM is down; callers that persist
# "done" must treat it as a deferral, not as a completed check.
LLM_UNAVAILABLE_REASON = "LLM unavailable (auto-mode)"

_DEFAULT_TYPES = ("decision", "solution")
_DEFAULT_TOP_K = 5
_DEFAULT_MIN_COSINE = 0.55
_DEFAULT_LLM_THRESHOLD = 0.8
_DEFAULT_FLAG_THRESHOLD = 0.5
_DEFAULT_TIMEOUT = 25.0
_MAX_LLM_INPUT_CHARS = 2200  # per record snippet


# ──────────────────────────────────────────────
# Result types
# ──────────────────────────────────────────────


@dataclass
class ContradictionVerdict:
    """One LLM comparison result. Multiple verdicts come back per save."""

    candidate_id: int
    cosine: float
    llm_confidence: float | None
    decision: str       # 'superseded' | 'flagged' | 'rejected' | 'skip' | 'error'
    reason: str
    provider: str | None = None
    model: str | None = None
    latency_ms: int | None = None

    @property
    def retryable(self) -> bool:
        """True when the comparison failed (LLM error, unparsable reply) and must be re-run later."""
        return self.decision == "error"


# ──────────────────────────────────────────────
# Env knobs
# ──────────────────────────────────────────────


def _enabled_mode() -> str:
    return os.environ.get("MEMORY_CONTRADICTION_DETECT_ENABLED", "auto").strip().lower()


def _enabled_types() -> tuple[str, ...]:
    raw = os.environ.get("MEMORY_CONTRADICTION_TYPES")
    if not raw:
        return _DEFAULT_TYPES
    return tuple(t.strip().lower() for t in raw.split(",") if t.strip())


def _top_k() -> int:
    raw = os.environ.get("MEMORY_CONTRADICTION_TOP_K")
    if not raw:
        return _DEFAULT_TOP_K
    try:
        return max(1, int(raw))
    except ValueError:
        return _DEFAULT_TOP_K


def _min_cosine() -> float:
    raw = os.environ.get("MEMORY_CONTRADICTION_MIN_COSINE")
    if not raw:
        return _DEFAULT_MIN_COSINE
    try:
        return max(0.0, min(1.0, float(raw)))
    except ValueError:
        return _DEFAULT_MIN_COSINE


def _llm_threshold() -> float:
    raw = os.environ.get("MEMORY_CONTRADICTION_LLM_THRESHOLD")
    if not raw:
        return _DEFAULT_LLM_THRESHOLD
    try:
        return max(0.0, min(1.0, float(raw)))
    except ValueError:
        return _DEFAULT_LLM_THRESHOLD


def _flag_threshold() -> float:
    raw = os.environ.get("MEMORY_CONTRADICTION_FLAG_THRESHOLD")
    if not raw:
        return _DEFAULT_FLAG_THRESHOLD
    try:
        return max(0.0, min(1.0, float(raw)))
    except ValueError:
        return _DEFAULT_FLAG_THRESHOLD


def _timeout() -> float:
    raw = os.environ.get("MEMORY_CONTRADICTION_TIMEOUT_SEC")
    if not raw:
        return _DEFAULT_TIMEOUT
    try:
        return max(1.0, float(raw))
    except ValueError:
        return _DEFAULT_TIMEOUT


# ──────────────────────────────────────────────
# Provider plumbing (for the production call site)
# ──────────────────────────────────────────────


_provider_cache: dict[str, Any] = {}


def _get_provider():
    cached = _provider_cache.get("contradiction")
    if cached is not None:
        return cached
    import config
    from llm_provider import make_provider
    name = os.environ.get(
        "MEMORY_CONTRADICTION_PROVIDER",
        config.get_phase_provider("enrich"),
    )
    provider = make_provider(name)
    _provider_cache["contradiction"] = provider
    return provider


def _model_name() -> str | None:
    import config
    return os.environ.get("MEMORY_CONTRADICTION_MODEL") or config.get_phase_model("enrich")


def _reset_provider_cache() -> None:
    _provider_cache.clear()


# ──────────────────────────────────────────────
# Prompt
# ──────────────────────────────────────────────


_COMPARE_PROMPT = """You compare two memory records and decide whether the
NEW one *supersedes* the OLD one. Supersession means: a competent reader,
told to follow only one of the two, would pick the NEW one because the OLD
one is now wrong, obsolete, or replaced.

Examples that count as supersession:
  - "We use Redis for caching" → "We migrated from Redis to Memcached"
  - "MySQL primary, Postgres replica" → "Postgres-only after the cutover"
  - "Set TIMEOUT=10" → "TIMEOUT is now 30 (10 was insufficient)"

Examples that DO NOT count as supersession (output 0.0):
  - Unrelated topics that happen to mention the same component
  - Two complementary facts about different parts of the same system
  - The new record is just a more detailed restatement of the old one

OLD record (id={old_id}, type={old_type}):
---
{old_content}
---

NEW record (id=just-saved, type={new_type}):
---
{new_content}
---

Respond with one JSON object on a single line, no prose:
{{"contradicts": true|false, "confidence": 0.0-1.0, "reason": "one short sentence"}}
"""


# ──────────────────────────────────────────────
# JSON parsing helper
# ──────────────────────────────────────────────


def _parse_verdict(raw: str) -> dict[str, Any] | None:
    if not raw:
        return None
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


def _coerce_confidence(value: Any) -> float | None:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, v))


# ──────────────────────────────────────────────
# Production call-site adapters
# ──────────────────────────────────────────────


def production_llm_call(prompt: str) -> str:
    """Default `llm_fn` used in the production hot path."""
    provider = _get_provider()
    if not provider.available():
        raise RuntimeError(
            f"contradiction provider '{getattr(provider, 'name', '?')}' unavailable"
        )
    return provider.complete(
        prompt,
        model=_model_name(),
        max_tokens=160,
        temperature=0.0,
        timeout=_timeout(),
    )


def production_candidates_query(
    db,
    *,
    project: str,
    ktype: str,
    candidate_ids: list[int],
) -> list[dict[str, Any]]:
    """Fetch the candidate rows referenced by id, filtered to active records
    in the same project + type. Returns dicts with id/content/type/project."""
    if not candidate_ids:
        return []
    placeholders = ",".join("?" * len(candidate_ids))
    rows = db.execute(
        f"""SELECT id, content, type, project FROM knowledge
           WHERE id IN ({placeholders})
             AND status='active'
             AND project=?
             AND type=?""",
        (*candidate_ids, project, ktype),
    ).fetchall()
    return [dict(r) if hasattr(r, "keys") else dict(zip(("id", "content", "type", "project"), r)) for r in rows]


# ──────────────────────────────────────────────
# Core detector
# ──────────────────────────────────────────────


def should_run(ktype: str | None) -> tuple[bool, str]:
    """Quick gate before any expensive work."""
    mode = _enabled_mode()
    if mode in ("false", "0", "off", "no"):
        return False, "disabled by MEMORY_CONTRADICTION_DETECT_ENABLED"
    if not ktype:
        return False, "no type — skipping"
    if ktype.lower() not in _enabled_types():
        return False, f"type '{ktype}' not in enabled types"

    if mode == "auto":
        try:
            import config
            if not config.has_llm("enrich"):
                return False, LLM_UNAVAILABLE_REASON
        except Exception as exc:  # pragma: no cover — defensive
            return False, f"has_llm check failed: {exc}"
    return True, ""


def detect_contradictions(
    content: str,
    *,
    ktype: str,
    project: str,
    candidate_pool: list[tuple[int, float]],
    fetch_candidates: Callable[[list[int]], list[dict[str, Any]]],
    llm_fn: Callable[[str], str] = production_llm_call,
) -> list[ContradictionVerdict]:
    """Run the contradiction comparison loop.

    Parameters
    ----------
    content : freshly-saved record content (already privacy/coref/filter applied)
    ktype : record type (must be in `_enabled_types()` — caller checks)
    project : project scope
    candidate_pool : `[(knowledge_id, cosine), ...]` sorted by cosine desc.
                     Caller runs the embedding search; the detector only
                     iterates and asks the LLM.
    fetch_candidates : callable that pulls full rows for a given list of ids
                       (DI for testability). Should already filter to
                       active + same project + same type.
    llm_fn : callable returning the raw LLM text. Defaults to the production
             provider call; tests inject a fake.

    Returns
    -------
    A list of `ContradictionVerdict`s, one per candidate that was actually
    compared. Empty when nothing crossed `min_cosine`.
    """
    min_cos = _min_cosine()
    llm_thr = _llm_threshold()
    flag_thr = _flag_threshold()

    # Filter by minimum cosine
    above_floor = [(cid, cos) for cid, cos in candidate_pool if cos >= min_cos]
    if not above_floor:
        return []

    # Fetch full rows in one shot (the helper enforces project + type + status filters).
    candidate_ids = [cid for cid, _ in above_floor]
    rows = fetch_candidates(candidate_ids)
    if not rows:
        return []

    # Index rows by id for quick joining with cosine values.
    by_id = {row["id"]: row for row in rows}

    verdicts: list[ContradictionVerdict] = []
    for cid, cos in above_floor:
        row = by_id.get(cid)
        if row is None:
            # Caller's pool included a candidate that the fetcher filtered out
            # (different project/type/status). Quietly skip — not an error.
            continue
        prompt = _COMPARE_PROMPT.format(
            old_id=cid,
            old_type=row.get("type", ktype),
            old_content=(row.get("content") or "")[:_MAX_LLM_INPUT_CHARS],
            new_type=ktype,
            new_content=(content or "")[:_MAX_LLM_INPUT_CHARS],
        )
        started = time.monotonic()
        try:
            raw = llm_fn(prompt)
        except Exception as exc:
            latency = int((time.monotonic() - started) * 1000)
            LOG(f"LLM call failed after {latency}ms for candidate {cid}: {exc}")
            verdicts.append(ContradictionVerdict(
                candidate_id=cid, cosine=cos, llm_confidence=None,
                decision="error", reason=f"LLM error: {exc}",
                latency_ms=latency,
            ))
            continue
        latency = int((time.monotonic() - started) * 1000)
        parsed = _parse_verdict(raw)
        if not parsed:
            verdicts.append(ContradictionVerdict(
                candidate_id=cid, cosine=cos, llm_confidence=None,
                decision="error",
                reason=f"unparsable LLM response: {raw[:120]!r}",
                latency_ms=latency,
            ))
            continue
        confidence = _coerce_confidence(parsed.get("confidence"))
        contradicts = bool(parsed.get("contradicts"))
        reason = str(parsed.get("reason", "")).strip()[:240]

        if not contradicts or confidence is None:
            verdicts.append(ContradictionVerdict(
                candidate_id=cid, cosine=cos, llm_confidence=confidence,
                decision="rejected",
                reason=reason or "LLM said no contradiction",
                latency_ms=latency,
            ))
            continue

        if confidence >= llm_thr:
            decision = "superseded"
        elif confidence >= flag_thr:
            decision = "flagged"
        else:
            decision = "rejected"

        verdicts.append(ContradictionVerdict(
            candidate_id=cid, cosine=cos, llm_confidence=confidence,
            decision=decision,
            reason=reason or f"confidence={confidence:.2f}",
            latency_ms=latency,
        ))
    return verdicts


# ──────────────────────────────────────────────
# Application
# ──────────────────────────────────────────────


def apply_supersession(db, old_id: int, new_id: int) -> bool:
    """Mark `old_id` as superseded by `new_id`. Returns True on success."""
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    try:
        cur = db.execute(
            "UPDATE knowledge SET status='superseded', superseded_by=?, "
            "last_confirmed=? WHERE id=? AND status='active'",
            (new_id, now, old_id),
        )
        db.commit()
        return cur.rowcount > 0
    except Exception as exc:
        LOG(f"apply_supersession({old_id}→{new_id}) failed: {exc}")
        return False


def log_verdict(
    db,
    verdict: ContradictionVerdict,
    *,
    new_id: int,
    provider: str | None = None,
    model: str | None = None,
) -> None:
    """Persist one verdict row to `contradiction_log`."""
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    try:
        db.execute(
            """INSERT INTO contradiction_log (
                created_at, new_knowledge_id, candidate_knowledge_id,
                cosine_similarity, llm_confidence, decision, reason,
                provider, model, latency_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                now, new_id, verdict.candidate_id,
                verdict.cosine, verdict.llm_confidence, verdict.decision,
                verdict.reason, provider or verdict.provider,
                model or verdict.model, verdict.latency_ms,
            ),
        )
        db.commit()
    except Exception as exc:
        LOG(f"log_verdict insert failed: {exc}")


def apply_and_log(db, verdicts: list[ContradictionVerdict], *, new_id: int,
                  provider: str | None = None, model: str | None = None) -> dict[str, int]:
    """Sweep verdicts: supersede where decided, log everything. Returns counts."""
    counts = {"superseded": 0, "flagged": 0, "rejected": 0, "error": 0, "skip": 0}
    for v in verdicts:
        counts[v.decision] = counts.get(v.decision, 0) + 1
        if v.decision == "superseded":
            ok = apply_supersession(db, v.candidate_id, new_id)
            if not ok:
                # Row was already superseded by someone else — re-classify
                # so the audit log doesn't claim a write that didn't happen.
                v.decision = "skip"
                v.reason = (v.reason or "") + " | already non-active"
                counts["superseded"] -= 1
                counts["skip"] += 1
        log_verdict(db, v, new_id=new_id, provider=provider, model=model)
    return counts
