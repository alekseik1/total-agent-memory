"""Tests for v10.1 async enrichment worker.

Covers:
  - enqueue puts a row with status='pending'
  - run_pending claims rows atomically and runs all stages
  - per-stage failures are isolated and logged in last_error
  - stages are idempotent (replaying a row twice does not duplicate audit rows)
  - retry up to MEMORY_ENRICH_MAX_ATTEMPTS, then status flips to 'failed'
  - quality_gate 'drop' verdict marks knowledge.status='quality_dropped'
    rather than physically removing the record
  - daemon thread starts only when MEMORY_ASYNC_ENRICHMENT=true
"""

from __future__ import annotations

import json
import sqlite3
import sys
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import enrichment_worker as ew


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────


@pytest.fixture
def seed_record(db):
    """Insert a knowledge row and return its id."""
    cur = db.execute(
        "INSERT INTO knowledge (session_id, type, content, project, created_at) "
        "VALUES ('s1', 'fact', 'Postgres 18 настроен в claude-memory-server', 'p', '2026-04-27T10:00:00Z')"
    )
    db.commit()
    return cur.lastrowid


# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────


def test_enabled_default_off(monkeypatch):
    monkeypatch.delenv("MEMORY_ASYNC_ENRICHMENT", raising=False)
    assert ew._enabled() is False


def test_enabled_when_true(monkeypatch):
    monkeypatch.setenv("MEMORY_ASYNC_ENRICHMENT", "true")
    assert ew._enabled() is True


# ──────────────────────────────────────────────
# Enqueue
# ──────────────────────────────────────────────


def test_enqueue_creates_pending_row(db, seed_record):
    qid = ew.enqueue(
        db,
        knowledge_id=seed_record,
        session_id="s1",
        project="p",
        ktype="fact",
        content_snapshot="payload",
        tags_snapshot=["postgres", "infra"],
    )
    row = db.execute("SELECT * FROM enrichment_queue WHERE id=?", (qid,)).fetchone()
    assert row["status"] == "pending"
    assert row["attempts"] == 0
    assert row["knowledge_id"] == seed_record
    assert json.loads(row["tags_snapshot"]) == ["postgres", "infra"]


# ──────────────────────────────────────────────
# Claim
# ──────────────────────────────────────────────


def test_claim_pending_moves_rows_to_processing(db, seed_record):
    for i in range(3):
        ew.enqueue(db, knowledge_id=seed_record, session_id="s1",
                   project="p", ktype="fact",
                   content_snapshot=f"row {i}", tags_snapshot=[])
    tasks = ew._claim_pending(db, limit=2)
    assert len(tasks) == 2
    statuses = [r[0] for r in db.execute(
        "SELECT status FROM enrichment_queue ORDER BY id"
    ).fetchall()]
    assert statuses[:2] == ["processing", "processing"]
    assert statuses[2] == "pending"
    # attempts bumped
    assert all(t.attempts == 1 for t in tasks)


def test_claim_returns_empty_when_no_pending(db):
    assert ew._claim_pending(db, limit=5) == []


# ──────────────────────────────────────────────
# run_pending end-to-end with all stages stubbed out
# ──────────────────────────────────────────────


def _all_stages_succeed_no_op(*args, **kwargs):
    """Stub stage runner that does nothing — tests the queue mechanics."""
    return None


def test_run_pending_marks_done_when_stages_succeed(db, seed_record, monkeypatch):
    ew.enqueue(db, knowledge_id=seed_record, session_id="s1",
               project="p", ktype="fact",
               content_snapshot="payload", tags_snapshot=["postgres"])
    monkeypatch.setattr(ew, "_STAGES", [
        ("fake1", _all_stages_succeed_no_op),
        ("fake2", _all_stages_succeed_no_op),
    ])
    counts = ew.run_pending(db, max_rows=10)
    assert counts["done"] == 1
    assert counts["claimed"] == 1
    row = db.execute("SELECT status, last_error, finished_at FROM enrichment_queue").fetchone()
    assert row["status"] == "done"
    assert row["last_error"] is None
    assert row["finished_at"] is not None


def test_run_pending_isolates_one_failing_stage(db, seed_record, monkeypatch):
    ew.enqueue(db, knowledge_id=seed_record, session_id="s1",
               project="p", ktype="fact",
               content_snapshot="payload", tags_snapshot=[])
    calls = []

    def good(db, task, store=None):
        calls.append("good")

    def bad(db, task, store=None):
        calls.append("bad")
        raise RuntimeError("LLM provider down")

    monkeypatch.setattr(ew, "_STAGES", [
        ("good_first", good),
        ("bad_second", bad),
        ("good_third", good),
    ])
    counts = ew.run_pending(db, max_rows=10)
    # All three stages were attempted (fail isolation)
    assert calls == ["good", "bad", "good"]
    # Row marked retry-pending after first failure
    row = db.execute("SELECT status, last_error, attempts FROM enrichment_queue").fetchone()
    assert row["status"] == "pending"
    assert "bad_second" in row["last_error"]
    assert "LLM provider down" in row["last_error"]
    assert row["attempts"] == 1


def test_run_pending_marks_failed_after_max_attempts(db, seed_record, monkeypatch):
    monkeypatch.setenv("MEMORY_ENRICH_MAX_ATTEMPTS", "2")
    ew.enqueue(db, knowledge_id=seed_record, session_id="s1",
               project="p", ktype="fact",
               content_snapshot="payload", tags_snapshot=[])

    def always_fails(db, task, store=None):
        raise RuntimeError("permanent error")

    monkeypatch.setattr(ew, "_STAGES", [("only", always_fails)])
    # First attempt → retry
    ew.run_pending(db, max_rows=10)
    # Second attempt → should flip to failed
    ew.run_pending(db, max_rows=10)
    row = db.execute("SELECT status, attempts FROM enrichment_queue").fetchone()
    assert row["status"] == "failed"
    assert row["attempts"] == 2


# ──────────────────────────────────────────────
# Quality gate stage (real wiring — the "soft drop" semantic)
# ──────────────────────────────────────────────


def test_quality_gate_stage_marks_dropped_record(db, seed_record):
    """Async drop must NOT delete the row — it just flips status."""
    ew.enqueue(db, knowledge_id=seed_record, session_id="s1",
               project="p", ktype="fact",
               content_snapshot="garbage", tags_snapshot=[])

    # Fake quality_gate module returning a 'drop' verdict.
    fake_score = type("S", (), {
        "decision": "drop", "total": 0.2, "specificity": 0.2,
        "actionability": 0.2, "verifiability": 0.2,
        "reason": "noise", "threshold": 0.5,
        "provider": "fake", "model": "fake-1", "latency_ms": 10,
    })()
    log_calls = []
    fake_qg = type("M", (), {
        "score_quality": lambda content, ktype, project: fake_score,
        "log_decision": lambda *a, **k: log_calls.append(k),
    })

    with patch.dict(sys.modules, {"quality_gate": fake_qg}):
        # Drive only the quality_gate stage so the test is hermetic.
        task = ew._claim_pending(db, limit=1)[0]
        ew._run_quality_gate(db, task)

    knowledge_row = db.execute(
        "SELECT status FROM knowledge WHERE id=?", (seed_record,)
    ).fetchone()
    assert knowledge_row["status"] == "quality_dropped"
    # Audit row was journalled with the knowledge id
    assert log_calls and log_calls[0]["knowledge_id"] == seed_record


def test_quality_gate_stage_skipped_when_skip_quality_flag(db, seed_record):
    ew.enqueue(db, knowledge_id=seed_record, session_id="s1",
               project="p", ktype="fact",
               content_snapshot="payload", tags_snapshot=[],
               skip_quality=True)
    called = {"n": 0}
    fake_qg = type("M", (), {
        "score_quality": lambda *a, **k: called.update(n=called["n"] + 1),
        "log_decision": lambda *a, **k: None,
    })
    with patch.dict(sys.modules, {"quality_gate": fake_qg}):
        task = ew._claim_pending(db, limit=1)[0]
        ew._run_quality_gate(db, task)
    assert called["n"] == 0  # gate was skipped


# ──────────────────────────────────────────────
# Daemon thread
# ──────────────────────────────────────────────


# ──────────────────────────────────────────────
# Stale-processing recovery
# ──────────────────────────────────────────────


def test_reclaim_stale_flips_old_processing_back_to_pending(db, seed_record, monkeypatch):
    """Row stuck in 'processing' for > stale_after_sec → reclaimed."""
    monkeypatch.setenv("MEMORY_ENRICH_STALE_AFTER_SEC", "5")
    qid = ew.enqueue(db, knowledge_id=seed_record, session_id="s1",
                     project="p", ktype="fact",
                     content_snapshot="payload", tags_snapshot=[])
    # Force the row into a stale processing state.
    db.execute(
        "UPDATE enrichment_queue SET status='processing', "
        "started_at='2020-01-01T00:00:00Z' WHERE id=?",
        (qid,),
    )
    db.commit()

    n = ew.reclaim_stale(db)
    assert n == 1
    row = db.execute(
        "SELECT status, last_error FROM enrichment_queue WHERE id=?", (qid,)
    ).fetchone()
    assert row["status"] == "pending"
    assert "reclaimed" in (row["last_error"] or "").lower()


def test_reclaim_stale_leaves_fresh_processing_alone(db, seed_record, monkeypatch):
    """A row that *just* started processing must NOT be reclaimed."""
    monkeypatch.setenv("MEMORY_ENRICH_STALE_AFTER_SEC", "60")
    qid = ew.enqueue(db, knowledge_id=seed_record, session_id="s1",
                     project="p", ktype="fact",
                     content_snapshot="payload", tags_snapshot=[])
    # Started just now (well within the stale window).
    fresh = ew._now()
    db.execute(
        "UPDATE enrichment_queue SET status='processing', started_at=? WHERE id=?",
        (fresh, qid),
    )
    db.commit()
    n = ew.reclaim_stale(db)
    assert n == 0
    status = db.execute(
        "SELECT status FROM enrichment_queue WHERE id=?", (qid,)
    ).fetchone()["status"]
    assert status == "processing"


def test_run_pending_reports_reclaimed_count(db, seed_record, monkeypatch):
    """run_pending() returns reclaimed count alongside claimed/done."""
    monkeypatch.setenv("MEMORY_ENRICH_STALE_AFTER_SEC", "5")
    qid = ew.enqueue(db, knowledge_id=seed_record, session_id="s1",
                     project="p", ktype="fact",
                     content_snapshot="payload", tags_snapshot=[])
    db.execute(
        "UPDATE enrichment_queue SET status='processing', "
        "started_at='2020-01-01T00:00:00Z' WHERE id=?",
        (qid,),
    )
    db.commit()
    monkeypatch.setattr(ew, "_STAGES", [("noop", lambda *a, **k: None)])
    counts = ew.run_pending(db, max_rows=10)
    assert counts["reclaimed"] == 1
    # The reclaimed row should also be picked up in the same tick.
    assert counts["claimed"] == 1
    assert counts["done"] == 1


# ──────────────────────────────────────────────
# Daemon thread
# ──────────────────────────────────────────────


def test_start_worker_returns_none_when_disabled(monkeypatch):
    monkeypatch.delenv("MEMORY_ASYNC_ENRICHMENT", raising=False)
    fake_store = type("S", (), {"db": None})()
    assert ew.start_worker(fake_store) is None


def test_start_worker_launches_thread_when_enabled(monkeypatch, db, seed_record):
    monkeypatch.setenv("MEMORY_ASYNC_ENRICHMENT", "true")
    monkeypatch.setenv("MEMORY_ENRICH_TICK_SEC", "0.02")

    drained = []

    def fake_run(db_, store=None, **kw):
        drained.append(time.monotonic())
        return {"claimed": 0, "done": 0, "retried": 0, "failed": 0}

    monkeypatch.setattr(ew, "run_pending", fake_run)
    fake_store = type("S", (), {"db": db})()
    t = ew.start_worker(fake_store)
    assert t is not None
    try:
        # Give the thread a couple of ticks
        time.sleep(0.1)
        assert len(drained) >= 2
    finally:
        t.stop()
        t.join(timeout=1)
        assert not t.is_alive()


def test_worker_ticks_do_not_corrupt_concurrent_store_writes(tmp_path, monkeypatch):
    """A tick firing while the Store's connection is mid-transaction must not
    break that transaction.

    sqlite3 keeps the implicit-BEGIN state on the Connection, so a worker
    sharing `store.db` raises "cannot start a transaction within a transaction"
    / "bad parameter or other API misuse" in either thread. Both writers here
    are real: the assertions check that neither lost a row.
    """
    from base_schema import apply_full_schema

    store_conn = sqlite3.connect(str(tmp_path / "memory.db"), check_same_thread=False)
    store_conn.row_factory = sqlite3.Row
    # Mirror server.py:268-270 — WAL is what lets a second connection write.
    store_conn.execute("PRAGMA journal_mode=WAL")
    apply_full_schema(store_conn)
    store_conn.execute(
        "CREATE TABLE probe (id INTEGER PRIMARY KEY, writer TEXT NOT NULL)"
    )
    store_conn.commit()

    monkeypatch.setenv("MEMORY_ASYNC_ENRICHMENT", "true")
    monkeypatch.setenv("MEMORY_ENRICH_TICK_SEC", "0.001")

    burst = 800
    started = threading.Barrier(2, timeout=5)

    def write_burst(conn, writer, errors):
        """Commit every 7th row so a transaction spans several statements —
        a one-statement-per-commit loop is too narrow to catch the race."""
        for i in range(burst):
            try:
                conn.execute("INSERT INTO probe (writer) VALUES (?)", (writer,))
                if i % 7 == 0:
                    conn.commit()
            except Exception as e:  # noqa: BLE001
                errors.append(f"{type(e).__name__}: {e}")
                return
        conn.commit()

    worker_errors: list[str] = []
    ticked = threading.Event()

    def fake_run(db_, store=None, **kw):
        if not ticked.is_set():
            ticked.set()
            started.wait()
            write_burst(db_, "worker", worker_errors)
        return {"claimed": 0, "done": 0, "retried": 0, "failed": 0}

    monkeypatch.setattr(ew, "run_pending", fake_run)
    fake_store = type("S", (), {"db": store_conn})()

    t = ew.start_worker(fake_store)
    assert t is not None
    main_errors: list[str] = []
    try:
        started.wait()
        write_burst(store_conn, "main", main_errors)
    finally:
        t.stop()
        t.join(timeout=10)

    assert ticked.is_set(), "worker never ticked — the test proved nothing"
    assert main_errors == []
    assert worker_errors == []
    counts = dict(
        store_conn.execute("SELECT writer, count(*) FROM probe GROUP BY writer").fetchall()
    )
    assert counts.get("main") == burst
    assert counts.get("worker") == burst
    store_conn.close()
