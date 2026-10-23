"""Retry queue for records saved without a vector (table ``embedding_pending``, migration 039).

When the embedding provider is down or times out, ``save_knowledge`` still
commits the record (text search keeps working) and enqueues its id here. The
drain runs from the reflection runner and opportunistically at server start:

  * ``claim`` atomically leases due rows by pushing ``next_attempt_at`` forward
    by ``CLAIM_LEASE_SECONDS`` in one ``UPDATE ... RETURNING``; the outer WHERE
    repeats the due condition, so a concurrent claimer (SQLite write lock, or
    PostgreSQL's re-check after a row lock) never gets the same row. A crashed
    drainer's lease simply expires.
  * success deletes the row; a vanished / inactive record deletes it too;
  * failure backs off ``BASE_BACKOFF_SECONDS * 2**(attempts-1)`` capped at
    ``MAX_BACKOFF_SECONDS``; after ``MAX_ATTEMPTS`` the row becomes a dead letter
    (``status='failed'``) until ``retry_dead_letter`` puts it back.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any, Protocol

from memory_core.timestamps import format_utc

BASE_BACKOFF_SECONDS = 60
MAX_BACKOFF_SECONDS = 6 * 3600
MAX_ATTEMPTS = 8
CLAIM_LEASE_SECONDS = 900
DEFAULT_DRAIN_LIMIT = 50
STARTUP_DRAIN_LIMIT = 20
LAST_ERROR_MAX_CHARS = 500

STATUS_PENDING = "pending"
STATUS_FAILED = "failed"

LOG = lambda msg: sys.stderr.write(f"[embedding-backfill] {msg}\n")


class EmbeddingUnavailable(RuntimeError):
    """The embedding provider produced no vector for a record."""


class BackfillOutcome(StrEnum):
    EMBEDDED = "embedded"
    GONE = "gone"


class _Connection(Protocol):
    def execute(self, sql: str, parameters: Any = (), /) -> Any: ...

    def commit(self) -> None: ...

    def rollback(self) -> None: ...


@dataclass(frozen=True)
class PendingEmbedding:
    knowledge_id: int
    attempts: int


@dataclass
class DrainResult:
    claimed: int = 0
    embedded: int = 0
    gone: int = 0
    retried: int = 0
    dead_lettered: int = 0
    deferred: int = 0

    def as_dict(self) -> dict[str, int]:
        return {"claimed": self.claimed, "embedded": self.embedded, "gone": self.gone,
                "retried": self.retried, "dead_lettered": self.dead_lettered,
                "deferred": self.deferred}


EmbedOne = Callable[[int], BackfillOutcome]
ErrorSink = Callable[[str, BaseException, str], None]


def backoff_seconds(attempts: int, *, base: int = BASE_BACKOFF_SECONDS,
                    cap: int = MAX_BACKOFF_SECONDS) -> int:
    """Delay before the next attempt after ``attempts`` failures (1-based)."""
    exponent = max(0, attempts - 1)
    # Bound the exponent so the multiplication stays small before the cap applies.
    return min(cap, base * (2 ** min(exponent, cap.bit_length())))


class EmbeddingBackfill:
    """``embedding_pending`` operations over an injected connection."""

    def __init__(self, db: _Connection, *, clock: Callable[[], float] = time.time,
                 max_attempts: int = MAX_ATTEMPTS, base_backoff: int = BASE_BACKOFF_SECONDS,
                 max_backoff: int = MAX_BACKOFF_SECONDS, lease_seconds: int = CLAIM_LEASE_SECONDS) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        self.db = db
        self._clock = clock
        self.max_attempts = max_attempts
        self.base_backoff = base_backoff
        self.max_backoff = max_backoff
        self.lease_seconds = lease_seconds

    def _at(self, offset_seconds: float = 0.0) -> str:
        return format_utc(datetime.fromtimestamp(self._clock(), tz=UTC) + timedelta(seconds=offset_seconds))

    def enqueue(self, knowledge_id: int, error: str = "") -> bool:
        """Queue a record for embedding; True when a new row was added. Idempotent."""
        now = self._at()
        cursor = self.db.execute(
            "INSERT OR IGNORE INTO embedding_pending "
            "(knowledge_id, attempts, last_error, next_attempt_at, status, created_at, updated_at) "
            "VALUES (?, 0, ?, ?, 'pending', ?, ?)",
            (int(knowledge_id), (error or "")[:LAST_ERROR_MAX_CHARS] or None, now, now, now))
        self.db.commit()
        return int(cursor.rowcount or 0) == 1

    def claim(self, limit: int = DEFAULT_DRAIN_LIMIT) -> list[PendingEmbedding]:
        """Lease up to ``limit`` due rows; no other claimer sees them until the lease expires."""
        if limit <= 0:
            return []
        now = self._at()
        lease_until = self._at(self.lease_seconds)
        try:
            rows = self.db.execute(
                "UPDATE embedding_pending SET next_attempt_at = ?, updated_at = ? "
                "WHERE knowledge_id IN ("
                "  SELECT knowledge_id FROM embedding_pending "
                "  WHERE status = 'pending' AND next_attempt_at <= ? "
                "  ORDER BY next_attempt_at, knowledge_id LIMIT ?"
                ") AND status = 'pending' AND next_attempt_at <= ? "
                "RETURNING knowledge_id, attempts",
                (lease_until, now, now, int(limit), now)).fetchall()
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise
        return sorted((PendingEmbedding(int(row[0]), int(row[1])) for row in rows),
                      key=lambda item: item.knowledge_id)

    def mark_done(self, knowledge_id: int) -> None:
        self.db.execute("DELETE FROM embedding_pending WHERE knowledge_id = ?", (int(knowledge_id),))
        self.db.commit()

    def mark_failed(self, item: PendingEmbedding, error: str) -> str:
        """Record a failed attempt; returns the new status (pending or failed)."""
        attempts = item.attempts + 1
        status = STATUS_FAILED if attempts >= self.max_attempts else STATUS_PENDING
        delay = backoff_seconds(attempts, base=self.base_backoff, cap=self.max_backoff)
        now = self._at()
        self.db.execute(
            "UPDATE embedding_pending SET attempts = ?, last_error = ?, next_attempt_at = ?, "
            "status = ?, updated_at = ? WHERE knowledge_id = ?",
            (attempts, (error or "")[:LAST_ERROR_MAX_CHARS], self._at(delay), status, now,
             item.knowledge_id))
        self.db.commit()
        return status

    def defer(self, item: PendingEmbedding, delay_seconds: int) -> None:
        """Push a claimed row back without spending an attempt (provider known to be down)."""
        now = self._at()
        self.db.execute(
            "UPDATE embedding_pending SET next_attempt_at = ?, updated_at = ? WHERE knowledge_id = ? "
            "AND status = 'pending'", (self._at(delay_seconds), now, item.knowledge_id))
        self.db.commit()

    def retry_dead_letter(self, knowledge_ids: Iterable[int] | None = None) -> int:
        """Return dead letters (all, or the given ids) to the queue with a fresh attempt budget."""
        now = self._at()
        if knowledge_ids is None:
            cursor = self.db.execute(
                "UPDATE embedding_pending SET status = 'pending', attempts = 0, next_attempt_at = ?, "
                "updated_at = ? WHERE status = 'failed'", (now, now))
        else:
            ids = sorted({int(kid) for kid in knowledge_ids})
            if not ids:
                return 0
            marks = ",".join("?" for _ in ids)
            cursor = self.db.execute(
                "UPDATE embedding_pending SET status = 'pending', attempts = 0, next_attempt_at = ?, "
                f"updated_at = ? WHERE status = 'failed' AND knowledge_id IN ({marks})", (now, now, *ids))
        self.db.commit()
        return int(cursor.rowcount or 0)

    def counts(self) -> dict[str, int]:
        rows = self.db.execute("SELECT status, COUNT(*) FROM embedding_pending GROUP BY status").fetchall()
        out = {STATUS_PENDING: 0, STATUS_FAILED: 0}
        for row in rows:
            out[row[0]] = int(row[1])
        out["due"] = int(self.db.execute(
            "SELECT COUNT(*) FROM embedding_pending WHERE status = 'pending' AND next_attempt_at <= ?",
            (self._at(),)).fetchone()[0])
        return out

    def drain(self, embed_one: EmbedOne, *, limit: int = DEFAULT_DRAIN_LIMIT,
              stop_on_failure: bool = True, on_error: ErrorSink | None = None) -> DrainResult:
        """Claim due rows and embed each through ``embed_one``.

        ``embed_one(knowledge_id)`` returns EMBEDDED or GONE, and raises on
        failure (EmbeddingUnavailable when the provider gave no vector). With
        ``stop_on_failure`` (default) the first failure ends the pass: the
        provider is most likely down, so the remaining claimed rows are
        deferred by the same backoff without spending their attempts.
        """
        result = DrainResult()
        items = self.claim(limit)
        result.claimed = len(items)
        for index, item in enumerate(items):
            try:
                outcome = embed_one(item.knowledge_id)
            except Exception as error:  # noqa: BLE001 — any provider failure is a retryable attempt
                status = self.mark_failed(item, f"{type(error).__name__}: {error}")
                if status == STATUS_FAILED:
                    result.dead_lettered += 1
                else:
                    result.retried += 1
                if on_error is not None:
                    on_error("embedding_backfill", error,
                             f"backfill attempt {item.attempts + 1} for knowledge id={item.knowledge_id}")
                if stop_on_failure:
                    delay = backoff_seconds(item.attempts + 1, base=self.base_backoff, cap=self.max_backoff)
                    for rest in items[index + 1:]:
                        self.defer(rest, delay)
                        result.deferred += 1
                    break
                continue
            self.mark_done(item.knowledge_id)
            if outcome is BackfillOutcome.GONE:
                result.gone += 1
            else:
                result.embedded += 1
        if result.claimed:
            LOG(f"drain: {result.as_dict()}")
        return result
