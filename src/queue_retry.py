"""Shared retry/backoff helpers for the durable background queues.

Every queue that talks to an LLM or a network service (enrichment,
deep enrichment, triple extraction, ingest) uses the same schedule so an
outage of a few seconds never burns through a row's attempts: a failed row is
parked until `next_attempt_at` and only claimed again afterwards.
"""
from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timedelta, timezone

DEFAULT_RETRY_BASE_SEC = 30.0
DEFAULT_RETRY_MAX_SEC = 3600.0
MAX_BACKOFF_EXPONENT = 20

_COLUMN_DDL = "ALTER TABLE {table} ADD COLUMN {column} {ddl}"


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return max(0.0, float(raw))
    except ValueError:
        return default


def retry_base_sec() -> float:
    return _env_float("MEMORY_ENRICH_RETRY_BASE_SEC", DEFAULT_RETRY_BASE_SEC)


def retry_max_sec() -> float:
    return _env_float("MEMORY_ENRICH_RETRY_MAX_SEC", DEFAULT_RETRY_MAX_SEC)


def iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def now_iso() -> str:
    return iso(datetime.now(timezone.utc))


def backoff_seconds(attempts: int) -> float:
    """Exponential delay for the n-th failed attempt (1-based), capped."""
    exponent = min(max(attempts, 1) - 1, MAX_BACKOFF_EXPONENT)
    return min(retry_base_sec() * (2 ** exponent), retry_max_sec())


def next_attempt_iso(attempts: int, now: datetime | None = None) -> str:
    """Timestamp before which a row that failed `attempts` times must not be claimed."""
    return iso((now or datetime.now(timezone.utc)) + timedelta(seconds=backoff_seconds(attempts)))


def ensure_column(db: sqlite3.Connection, table: str, column: str, ddl: str) -> None:
    """Add `column` to `table` when the schema predates migration 045 (idempotent)."""
    columns = {row[1] for row in db.execute(f"PRAGMA table_info({table})")}
    if not columns or column in columns:
        return
    try:
        db.execute(_COLUMN_DDL.format(table=table, column=column, ddl=ddl))
        db.commit()
    except sqlite3.OperationalError as error:
        if "duplicate column" not in str(error).lower():
            raise


MAX_DEFERRALS = 50
STALE_PROCESSING_MINUTES = 30
_QUEUE_TABLES = frozenset({"deep_enrichment_queue", "triple_extraction_queue"})
_QUEUE_STAMP = "%Y-%m-%dT%H:%M:%SZ"


def ensure_queue_columns(db: sqlite3.Connection, table: str) -> None:
    """Bring a queue table created before migration 045 up to date."""
    ensure_column(db, table, "next_attempt_at", "TEXT")
    ensure_column(db, table, "deferrals", "INTEGER NOT NULL DEFAULT 0")


def _check_table(table: str) -> None:
    if table not in _QUEUE_TABLES:
        raise ValueError(f"unsupported queue table: {table}")


def _replace_state(db: sqlite3.Connection, table: str, item_id: int, knowledge_id: int, status: str) -> None:
    # UNIQUE(knowledge_id, status): drop an older row occupying the target state.
    db.execute(
        f"DELETE FROM {table} WHERE knowledge_id = ? AND status = ? AND id != ?",  # noqa: S608 - table is allowlisted
        (knowledge_id, status, item_id),
    )


def defer_row(db: sqlite3.Connection, table: str, item_id: int, error: str, max_attempts: int) -> str:
    """Park a row whose LLM/provider was unreachable. Does not consume an attempt.

    Returns the new status. After MAX_DEFERRALS the row is failed so a
    permanently broken provider cannot keep a row alive forever.
    """
    _check_table(table)
    row = db.execute(
        f"SELECT knowledge_id, attempts, deferrals FROM {table} WHERE id = ?", (item_id,)  # noqa: S608
    ).fetchone()
    if row is None:
        return "missing"
    deferrals = (row[2] or 0) + 1
    if deferrals >= MAX_DEFERRALS:
        status, attempts, due = "failed", max(row[1] or 0, max_attempts), None
    else:
        status, attempts, due = "pending", row[1] or 0, next_attempt_iso(deferrals)
    _replace_state(db, table, item_id, row[0], status)
    db.execute(
        f"UPDATE {table} SET status = ?, attempts = ?, deferrals = ?, next_attempt_at = ?, "  # noqa: S608
        "last_error = ?, claimed_at = NULL WHERE id = ?",
        (status, attempts, deferrals, due, (error or "")[:500], item_id),
    )
    db.commit()
    return status


def reclaim_stale_processing(db: sqlite3.Connection, table: str, max_attempts: int,
                             stale_minutes: int = STALE_PROCESSING_MINUTES) -> int:
    """Return rows stuck in 'processing' (killed worker) to 'pending', or fail them past max_attempts."""
    _check_table(table)
    cutoff = (datetime.now(timezone.utc) - timedelta(minutes=stale_minutes)).strftime(_QUEUE_STAMP)
    stale = db.execute(
        f"SELECT id, knowledge_id, attempts FROM {table} "  # noqa: S608
        "WHERE status = 'processing' AND (claimed_at IS NULL OR claimed_at < ?)",
        (cutoff,),
    ).fetchall()
    for item_id, knowledge_id, attempts in stale:
        if db.execute(
            f"SELECT 1 FROM {table} WHERE knowledge_id = ? AND status = 'pending' LIMIT 1",  # noqa: S608
            (knowledge_id,),
        ).fetchone():
            db.execute(f"DELETE FROM {table} WHERE id = ?", (item_id,))  # noqa: S608
            continue
        used = (attempts or 0) + 1
        status = "failed" if used >= max_attempts else "pending"
        _replace_state(db, table, item_id, knowledge_id, status)
        db.execute(
            f"UPDATE {table} SET status = ?, attempts = ?, next_attempt_at = NULL, "  # noqa: S608
            "last_error = 'reclaimed: worker did not finish' WHERE id = ?",
            (status, used, item_id),
        )
    db.commit()
    return len(stale)
