"""Regression tests for triple_extraction_queue deadlock recovery.

Prior bug: stuck processing row with no timeout blocked the queue indefinitely.
A new pending row for the same knowledge_id would hit UNIQUE(knowledge_id,
status) on the UPDATE to processing.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from triple_extraction_queue import TripleExtractionQueue
from base_schema import apply_full_schema


def _setup_db() -> sqlite3.Connection:
    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    apply_full_schema(db)
    return db


def test_reclaim_stale_removes_old_processing_rows():
    db = _setup_db()
    q = TripleExtractionQueue(db)
    old = (datetime.now(timezone.utc) - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    db.execute(
        "INSERT INTO triple_extraction_queue (knowledge_id, status, created_at, claimed_at) "
        "VALUES (42, 'processing', ?, ?)",
        (old, old),
    )
    db.commit()
    cleaned = q.reclaim_stale(stale_minutes=30)
    assert cleaned == 1
    assert db.execute(
        "SELECT COUNT(*) FROM triple_extraction_queue WHERE status='processing'"
    ).fetchone()[0] == 0


def test_reclaim_stale_keeps_fresh_processing_rows():
    db = _setup_db()
    q = TripleExtractionQueue(db)
    fresh = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    db.execute(
        "INSERT INTO triple_extraction_queue (knowledge_id, status, created_at, claimed_at) "
        "VALUES (42, 'processing', ?, ?)",
        (fresh, fresh),
    )
    db.commit()
    cleaned = q.reclaim_stale(stale_minutes=30)
    assert cleaned == 0
    assert db.execute(
        "SELECT COUNT(*) FROM triple_extraction_queue WHERE status='processing'"
    ).fetchone()[0] == 1


def test_claim_next_unblocks_when_processing_is_stale():
    """Regression: previously, a stuck processing row for knowledge_id=K
    blocked the queue because UPDATE of new pending row for K to 'processing'
    collided on UNIQUE(knowledge_id, status)."""
    db = _setup_db()
    q = TripleExtractionQueue(db)
    old = (datetime.now(timezone.utc) - timedelta(hours=5)).strftime("%Y-%m-%dT%H:%M:%SZ")
    # Old stuck processing row for kid=42
    db.execute(
        "INSERT INTO triple_extraction_queue (knowledge_id, status, created_at, claimed_at) "
        "VALUES (42, 'processing', ?, ?)",
        (old, old),
    )
    # Fresh pending row for the same kid=42
    db.execute(
        "INSERT INTO triple_extraction_queue (knowledge_id, status, created_at) "
        "VALUES (42, 'pending', ?)",
        (datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),),
    )
    db.commit()

    claimed = q.claim_next()
    assert claimed is not None, "claim_next should succeed after reclaim_stale"
    assert claimed["knowledge_id"] == 42
    assert claimed["status"] == "processing"
    # Only one processing row should remain (the one we just claimed)
    count = db.execute(
        "SELECT COUNT(*) FROM triple_extraction_queue WHERE status='processing'"
    ).fetchone()[0]
    assert count == 1


def test_claim_next_no_reclaim_needed_when_no_stale():
    db = _setup_db()
    q = TripleExtractionQueue(db)
    db.execute(
        "INSERT INTO triple_extraction_queue (knowledge_id, status, created_at) "
        "VALUES (10, 'pending', ?)",
        (datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),),
    )
    db.commit()
    claimed = q.claim_next()
    assert claimed["knowledge_id"] == 10
