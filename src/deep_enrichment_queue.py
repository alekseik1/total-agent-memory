"""Async deep enrichment queue.

Mirrors `triple_extraction_queue` but drives `deep_enricher.deep_enrich()`
(entities / intent / topics via LLM). Results are persisted into
`knowledge_enrichment` for metadata filtering at retrieval time.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timezone
from typing import Any, Callable

from queue_retry import (
    defer_row,
    ensure_queue_columns,
    next_attempt_iso,
    now_iso,
    reclaim_stale_processing,
)

LOG = lambda msg: sys.stderr.write(f"[deep-enr-queue] {msg}\n")

STALE_PROCESSING_MINUTES = 30


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# Enricher signature: (content_text, base_metadata) -> {"entities","intent","topics"}
EnricherFn = Callable[[str, dict[str, Any] | None], dict[str, Any]]


class DeepEnrichmentQueue:
    """Durable FIFO queue for LLM-driven metadata enrichment."""

    def __init__(self, db: sqlite3.Connection, max_attempts: int = 3) -> None:
        self.db = db
        self.max_attempts = max(1, max_attempts)
        ensure_queue_columns(db, "deep_enrichment_queue")

    def enqueue(self, knowledge_id: int) -> bool:
        existing = self.db.execute(
            "SELECT id FROM deep_enrichment_queue "
            "WHERE knowledge_id = ? AND status = 'pending' LIMIT 1",
            (knowledge_id,),
        ).fetchone()
        if existing:
            return False
        self.db.execute(
            "INSERT INTO deep_enrichment_queue (knowledge_id, status, created_at) "
            "VALUES (?, 'pending', ?)",
            (knowledge_id, _now()),
        )
        self.db.commit()
        return True

    def reclaim_stale(self, stale_minutes: int = STALE_PROCESSING_MINUTES) -> int:
        """Return rows left in 'processing' by a killed drain to 'pending'."""
        return reclaim_stale_processing(
            self.db, "deep_enrichment_queue", self.max_attempts, stale_minutes
        )

    def claim_next(self) -> dict | None:
        self.reclaim_stale()
        row = self.db.execute(
            "SELECT id, knowledge_id, attempts FROM deep_enrichment_queue "
            "WHERE status = 'pending' AND (next_attempt_at IS NULL OR next_attempt_at <= ?) "
            "ORDER BY id ASC LIMIT 1",
            (now_iso(),),
        ).fetchone()
        if row is None:
            return None
        self.db.execute(
            "UPDATE deep_enrichment_queue SET status='processing', claimed_at=? WHERE id=?",
            (_now(), row["id"]),
        )
        self.db.commit()
        return {
            "id": row["id"],
            "knowledge_id": row["knowledge_id"],
            "attempts": row["attempts"],
            "status": "processing",
        }

    def mark_done(self, item_id: int) -> None:
        row = self.db.execute(
            "SELECT knowledge_id FROM deep_enrichment_queue WHERE id=?", (item_id,)
        ).fetchone()
        if row is not None:
            self.db.execute(
                "DELETE FROM deep_enrichment_queue "
                "WHERE knowledge_id = ? AND status = 'done' AND id != ?",
                (row["knowledge_id"], item_id),
            )
        self.db.execute(
            "UPDATE deep_enrichment_queue SET status='done', processed_at=? WHERE id=?",
            (_now(), item_id),
        )
        self.db.commit()

    def defer(self, item_id: int, error: str) -> str:
        """Park an item whose LLM was unreachable; it is retried later without losing an attempt."""
        return defer_row(self.db, "deep_enrichment_queue", item_id, error, self.max_attempts)

    def mark_failed(self, item_id: int, error: str) -> None:
        row = self.db.execute(
            "SELECT attempts, knowledge_id FROM deep_enrichment_queue WHERE id=?",
            (item_id,),
        ).fetchone()
        if row is None:
            return
        next_attempts = (row["attempts"] or 0) + 1
        new_status = "failed" if next_attempts >= self.max_attempts else "pending"
        self.db.execute(
            "DELETE FROM deep_enrichment_queue "
            "WHERE knowledge_id = ? AND status = ? AND id != ?",
            (row["knowledge_id"], new_status, item_id),
        )
        due = next_attempt_iso(next_attempts) if new_status == "pending" else None
        self.db.execute(
            "UPDATE deep_enrichment_queue "
            "SET status=?, attempts=?, last_error=?, processed_at=?, next_attempt_at=? WHERE id=?",
            (new_status, next_attempts, (error or "")[:500], _now(), due, item_id),
        )
        self.db.commit()

    # ──────────────────────────────────────────────
    # Worker
    # ──────────────────────────────────────────────

    def process_pending(
        self, enricher: EnricherFn, limit: int = 10
    ) -> dict[str, int]:
        stats = {"processed": 0, "failed": 0, "skipped": 0, "deferred": 0}

        for _ in range(limit):
            item = self.claim_next()
            if item is None:
                break

            kid = item["knowledge_id"]
            content_row = self.db.execute(
                "SELECT content FROM knowledge WHERE id=?", (kid,)
            ).fetchone()
            if content_row is None:
                self.mark_done(item["id"])
                stats["skipped"] += 1
                continue

            try:
                result = enricher(content_row["content"] or "", None) or {}
                if result.get("degraded"):
                    # The LLM failed: storing the empty defaults as "done" would
                    # make the loss permanent. Park the item and stop draining,
                    # the provider is down for the rest of the batch too.
                    self.defer(item["id"], str(result.get("degraded_reason") or "LLM unavailable"))
                    stats["deferred"] += 1
                    break
                self._store(kid, result)
                self.mark_done(item["id"])
                stats["processed"] += 1
            except Exception as e:  # noqa: BLE001
                LOG(f"enrich failed for knowledge_id={kid}: {e}")
                self.mark_failed(item["id"], str(e))
                stats["failed"] += 1

        return stats

    def _store(self, knowledge_id: int, enrichment: dict) -> None:
        entities = json.dumps(enrichment.get("entities") or [])
        intent = str(enrichment.get("intent") or "unknown")
        topics = json.dumps(enrichment.get("topics") or [])
        self.db.execute(
            """INSERT INTO knowledge_enrichment
                 (knowledge_id, entities, intent, topics, updated_at)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(knowledge_id) DO UPDATE SET
                 entities   = excluded.entities,
                 intent     = excluded.intent,
                 topics     = excluded.topics,
                 updated_at = excluded.updated_at""",
            (knowledge_id, entities, intent, topics, _now()),
        )
        self.db.commit()

    # ──────────────────────────────────────────────
    # Observability
    # ──────────────────────────────────────────────

    def stats(self) -> dict[str, int]:
        rows = self.db.execute(
            "SELECT status, COUNT(*) AS c FROM deep_enrichment_queue GROUP BY status"
        ).fetchall()
        out = {"pending": 0, "processing": 0, "done": 0, "failed": 0}
        for r in rows:
            out[r["status"]] = r["c"]
        return out
