"""Outbox / WriteIntent journal for save_knowledge — v10 P1.6.

Beever Atlas's persister opens a `WriteIntent` document in MongoDB before
touching its three stores (Weaviate / Neo4j / queues), then closes it on
success. A background `WriteReconciler` retries any intents that were
left open by a crash. We do the same in SQLite.

Lifecycle:

  pending       — created at the top of save_knowledge
     │
     ├── on success → committed(knowledge_id, committed_at)
     ├── on hard failure → failed(last_error)
     └── on duplicate dedup → superseded (the dedup path returned a
                              previously-saved record id; the intent is
                              honoured but no new row was inserted)

The reconciler runs in Store.__init__ after migrations apply. It picks up
`pending` rows older than the grace window (default 60s — long enough for
genuinely in-flight saves not to be stomped on), parses the payload, and
calls a replay function the caller injects. The replay must be idempotent
under content_hash dedup; save_knowledge already is.

Concurrency (migration 039): ``claim_pending`` leases intents atomically
(``lease_until``), so two reconcilers never replay the same intent; an
expired lease (crashed reconciler) is claimable again. Before replaying,
``find_existing_by_hash`` detects a record that was inserted by a save that
crashed before ``mark_committed`` and closes the intent without a replay.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from memory_core.timestamps import format_utc, utc_now

LOG = lambda msg: sys.stderr.write(f"[outbox] {msg}\n")

_DEFAULT_GRACE_SECONDS = 60
_DEFAULT_MAX_ATTEMPTS = 3
_DEFAULT_RECONCILE_LIMIT = 50
# Longer than a replay can take (save_knowledge incl. embedding retries).
_DEFAULT_LEASE_SECONDS = 600
# save_knowledge stamps the knowledge row before it opens the intent; candidates
# for find_existing_by_hash may be this much older than the intent.
_CREATED_AT_SLACK_SECONDS = 300


# ──────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────


@dataclass
class WriteIntent:
    id: int
    intent_uuid: str
    session_id: str | None
    content_hash: str
    payload: dict[str, Any]
    status: str
    knowledge_id: int | None
    attempts: int
    last_error: str | None
    created_at: str
    updated_at: str
    committed_at: str | None


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────


def _now() -> str:
    return utc_now()


def compute_content_hash(content: str, ktype: str | None, project: str | None) -> str:
    """sha1 of (type || project || content) — used for replay idempotency."""
    blob = f"{(ktype or '')}|{(project or '')}|{content or ''}".encode("utf-8")
    return hashlib.sha1(blob).hexdigest()


def _grace_seconds() -> int:
    raw = os.environ.get("MEMORY_OUTBOX_GRACE_SEC")
    if not raw:
        return _DEFAULT_GRACE_SECONDS
    try:
        return max(0, int(raw))
    except ValueError:
        return _DEFAULT_GRACE_SECONDS


def _max_attempts() -> int:
    raw = os.environ.get("MEMORY_OUTBOX_MAX_ATTEMPTS")
    if not raw:
        return _DEFAULT_MAX_ATTEMPTS
    try:
        return max(1, int(raw))
    except ValueError:
        return _DEFAULT_MAX_ATTEMPTS


def _enabled() -> bool:
    raw = os.environ.get("MEMORY_OUTBOX_ENABLED", "true").strip().lower()
    return raw not in ("0", "false", "off", "no")


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────


def is_enabled() -> bool:
    return _enabled()


def create_intent(
    db,
    *,
    payload: dict[str, Any],
    session_id: str | None,
    content: str,
    ktype: str | None,
    project: str | None,
) -> WriteIntent | None:
    """Insert a `pending` write_intents row. Returns None when outbox
    is disabled or the table is missing — caller should still proceed
    with the save (graceful degradation)."""
    if not _enabled():
        return None
    intent_uuid = uuid.uuid4().hex[:16]
    chash = compute_content_hash(content, ktype, project)
    now = _now()
    try:
        cur = db.execute(
            """INSERT INTO write_intents (
                intent_uuid, session_id, content_hash, payload_json,
                status, attempts, created_at, updated_at
            ) VALUES (?, ?, ?, ?, 'pending', 0, ?, ?)""",
            (intent_uuid, session_id, chash, json.dumps(payload), now, now),
        )
        db.commit()
        return WriteIntent(
            id=cur.lastrowid, intent_uuid=intent_uuid,
            session_id=session_id, content_hash=chash,
            payload=payload, status="pending",
            knowledge_id=None, attempts=0, last_error=None,
            created_at=now, updated_at=now, committed_at=None,
        )
    except Exception as exc:
        LOG(f"create_intent failed (continuing without outbox): {exc}")
        return None


def mark_committed(db, intent: WriteIntent | None, knowledge_id: int | None) -> None:
    if intent is None:
        return
    now = _now()
    try:
        db.execute(
            "UPDATE write_intents SET status='committed', knowledge_id=?, "
            "committed_at=?, updated_at=?, payload_json='{}' WHERE id=?",
            (knowledge_id, now, now, intent.id),
        )
        db.commit()
    except Exception as exc:
        LOG(f"mark_committed({intent.id}) failed: {exc}")


def mark_superseded(db, intent: WriteIntent | None, knowledge_id: int | None) -> None:
    """Use when save_knowledge returned an existing dedup target — we want
    the intent recorded as resolved, but with a distinct status so audits
    can tell genuine commits apart from no-op dedups."""
    if intent is None:
        return
    now = _now()
    try:
        db.execute(
            "UPDATE write_intents SET status='superseded', knowledge_id=?, "
            "committed_at=?, updated_at=?, payload_json='{}' WHERE id=?",
            (knowledge_id, now, now, intent.id),
        )
        db.commit()
    except Exception as exc:
        LOG(f"mark_superseded({intent.id}) failed: {exc}")


def mark_failed(db, intent: WriteIntent | None, error: str) -> None:
    if intent is None:
        return
    now = _now()
    try:
        db.execute(
            "UPDATE write_intents SET attempts=attempts+1, last_error=?, "
            "updated_at=? WHERE id=?",
            (error[:1000], now, intent.id),
        )
        # If we've blown the attempt budget, freeze the intent in 'failed'.
        db.execute(
            "UPDATE write_intents SET status='failed' "
            "WHERE id=? AND attempts >= ?",
            (intent.id, _max_attempts()),
        )
        db.commit()
    except Exception as exc:
        LOG(f"mark_failed({intent.id}) failed: {exc}")


def _lease_seconds() -> int:
    raw = os.environ.get("MEMORY_OUTBOX_LEASE_SEC")
    if not raw:
        return _DEFAULT_LEASE_SECONDS
    try:
        return max(1, int(raw))
    except ValueError:
        return _DEFAULT_LEASE_SECONDS


def _iso_at(offset_seconds: float) -> str:
    return format_utc(datetime.now(timezone.utc) + timedelta(seconds=offset_seconds))


def _shift(stamp: str | None, offset_seconds: float) -> str:
    """Canonical ``stamp`` moved by ``offset_seconds``; "" (matches everything) when absent."""
    if not stamp:
        return ""
    moment = datetime.fromisoformat(stamp)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return format_utc(moment + timedelta(seconds=offset_seconds))


def _row_to_intent(row) -> WriteIntent:
    try:
        payload = json.loads(row[4]) if row[4] else {}
    except (TypeError, ValueError) as exc:
        LOG(f"intent {row[1]}: unreadable payload ({exc}); replay will fail and count an attempt")
        payload = {}
    return WriteIntent(
        id=row[0], intent_uuid=row[1], session_id=row[2],
        content_hash=row[3], payload=payload, status=row[5],
        knowledge_id=row[6], attempts=row[7], last_error=row[8],
        created_at=row[9], updated_at=row[10], committed_at=row[11],
    )


def claim_pending(
    db, *, grace_seconds: int | None = None, limit: int | None = None,
    lease_seconds: int | None = None,
) -> list[WriteIntent]:
    """Atomically lease pending intents older than the grace window.

    One ``UPDATE ... RETURNING`` sets ``lease_until`` on up to ``limit``
    intents whose lease is absent or expired; the outer WHERE repeats that
    condition, so a concurrent reconciler (serialized by SQLite's write lock,
    or re-checked by PostgreSQL after the row lock) cannot claim the same
    intent. A reconciler that dies mid-replay leaves an expired lease behind
    and the intent becomes claimable again after ``lease_seconds``.
    """
    grace = _grace_seconds() if grace_seconds is None else grace_seconds
    cap = limit if limit is not None else _DEFAULT_RECONCILE_LIMIT
    lease = _lease_seconds() if lease_seconds is None else lease_seconds
    now_iso = _iso_at(0)
    cutoff_iso = _iso_at(-grace)
    lease_iso = _iso_at(lease)
    try:
        rows = db.execute(
            """UPDATE write_intents SET lease_until = ?, updated_at = ?
                WHERE id IN (
                    SELECT id FROM write_intents
                     WHERE status = 'pending' AND created_at < ?
                       AND (lease_until IS NULL OR lease_until < ?)
                     ORDER BY created_at ASC, id ASC LIMIT ?)
                  AND status = 'pending'
                  AND (lease_until IS NULL OR lease_until < ?)
            RETURNING id, intent_uuid, session_id, content_hash, payload_json,
                      status, knowledge_id, attempts, last_error,
                      created_at, updated_at, committed_at""",
            (lease_iso, now_iso, cutoff_iso, now_iso, cap, now_iso),
        ).fetchall()
        db.commit()
    except Exception as exc:  # noqa: BLE001 — reconcile is best-effort; the next start retries
        try:
            db.rollback()
        except Exception as rollback_exc:  # noqa: BLE001 — report both, keep the original
            LOG(f"claim_pending rollback failed: {rollback_exc}")
        LOG(f"claim_pending failed: {exc}")
        return []
    return sorted((_row_to_intent(row) for row in rows), key=lambda i: (i.created_at, i.id))


def find_existing_by_hash(
    db, content_hash: str, payload: Mapping[str, Any], *, since: str | None = None,
) -> int | None:
    """Id of a knowledge row this intent already produced, or None.

    Covers a crash after the INSERT but before ``mark_committed``: replaying
    would insert the record a second time. Candidates come from the
    ``idx_knowledge_neighbors`` index — (project, session_id) equality, rows
    created at or after the intent — and must reproduce ``content_hash``
    exactly (type, project and content), so a different record of the same
    session never matches. Any status counts: a committed record that was
    later superseded or deleted must not be resurrected by a replay.
    """
    content = payload.get("content")
    ktype = payload.get("ktype")
    project = payload.get("project") or "general"
    session_id = payload.get("sid")
    if not content or not ktype or session_id is None:
        return None
    if compute_content_hash(content, ktype, project) != content_hash:
        return None
    rows = db.execute(
        """SELECT id, type, project, content FROM knowledge
            WHERE project = ? AND session_id = ? AND created_at >= ?
            ORDER BY created_at, id""",
        (project, session_id, _shift(since, -_CREATED_AT_SLACK_SECONDS)),
    ).fetchall()
    for row in rows:
        if compute_content_hash(row[3], row[1], row[2]) == content_hash:
            return int(row[0])
    return None


def reconcile_pending(
    db,
    replay_fn: Callable[[dict[str, Any]], int | None],
    *,
    grace_seconds: int | None = None,
    limit: int | None = None,
    lease_seconds: int | None = None,
    on_recovered: Callable[[int], None] | None = None,
) -> dict[str, int]:
    """Walk pending intents, replay each via `replay_fn(payload)`.

    Intents are leased first (``claim_pending``), so concurrent reconcilers
    split the work instead of replaying the same intent twice. Before a
    replay, ``find_existing_by_hash`` checks whether the record was already
    inserted; if so the intent is closed as committed without a replay and
    ``on_recovered(knowledge_id)`` lets the caller finish side effects the
    crash may have skipped (e.g. queue a missing embedding).

    `replay_fn` should call save_knowledge(**payload, _from_outbox=True)
    and return the committed knowledge_id (or None on dedup). It must
    raise on hard failures so we can record `attempts`.

    Returns counts of {replayed, dedup, failed, skipped, recovered}.
    """
    counts = {"replayed": 0, "dedup": 0, "failed": 0, "skipped": 0, "recovered": 0}
    pending = claim_pending(db, grace_seconds=grace_seconds, limit=limit,
                            lease_seconds=lease_seconds)
    if not pending:
        return counts

    for intent in pending:
        if intent.attempts >= _max_attempts():
            mark_failed(db, intent, "max attempts already reached")
            counts["failed"] += 1
            counts["skipped"] += 1
            continue

        try:
            existing = find_existing_by_hash(db, intent.content_hash, intent.payload,
                                             since=intent.created_at)
        except Exception as exc:  # noqa: BLE001 — counted as a failed attempt
            LOG(f"find_existing_by_hash failed for intent {intent.intent_uuid}: {exc}")
            mark_failed(db, intent, f"existence check failed: {exc}")
            counts["failed"] += 1
            continue
        if existing is not None:
            mark_committed(db, intent, existing)
            counts["recovered"] += 1
            if on_recovered is not None:
                try:
                    on_recovered(existing)
                except Exception as exc:  # noqa: BLE001 — caller hook must not stop the reconcile
                    LOG(f"on_recovered({existing}) failed: {exc}")
            continue

        try:
            new_id = replay_fn(intent.payload)
        except Exception as exc:
            LOG(
                f"replay failed for intent {intent.intent_uuid} "
                f"(attempt {intent.attempts + 1}): {exc}"
            )
            mark_failed(db, intent, str(exc))
            counts["failed"] += 1
            continue

        if new_id is None:
            # Dedup hit — the underlying save returned a previously
            # existing record. Mark accordingly.
            mark_superseded(db, intent, None)
            counts["dedup"] += 1
            continue

        mark_committed(db, intent, new_id)
        counts["replayed"] += 1

    return counts


# ──────────────────────────────────────────────
# Stats helpers (for dashboards / memory_stats)
# ──────────────────────────────────────────────


def get_status_counts(db) -> dict[str, int]:
    try:
        rows = db.execute(
            "SELECT status, COUNT(*) FROM write_intents GROUP BY status"
        ).fetchall()
    except Exception:
        return {}
    return {row[0]: row[1] for row in rows}
