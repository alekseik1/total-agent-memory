"""Regression tests: enrichment retry/backoff, provider-outage deferral, poison rows, stage idempotency."""

from __future__ import annotations

import sqlite3
import sys
import urllib.error
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import enrichment_worker as ew  # noqa: E402
from llm_provider import ProviderUnavailable  # noqa: E402

OUTAGE_SEC = 2.0
BASE_SEC = 0.05


def _make_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE enrichment_queue (
            id INTEGER PRIMARY KEY AUTOINCREMENT, knowledge_id INTEGER NOT NULL, session_id TEXT,
            project TEXT NOT NULL DEFAULT 'general', ktype TEXT NOT NULL, content_snapshot TEXT NOT NULL,
            tags_snapshot TEXT NOT NULL DEFAULT '[]', importance TEXT NOT NULL DEFAULT 'medium',
            skip_quality INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'pending'
                CHECK (status IN ('pending','processing','done','failed')),
            attempts INTEGER NOT NULL DEFAULT 0, last_error TEXT, enqueued_at TEXT NOT NULL,
            lease_token TEXT, heartbeat_at TEXT, atomic_only INTEGER DEFAULT 0,
            started_at TEXT, finished_at TEXT
        );
        """
    )
    return conn


@pytest.fixture
def db():
    conn = _make_db()
    yield conn
    conn.close()


def _enqueue(db) -> int:
    return ew.enqueue(db, knowledge_id=1, session_id="s", project="p", ktype="fact",
                      content_snapshot="payload", tags_snapshot=[])


def _row(db):
    return db.execute("SELECT * FROM enrichment_queue").fetchone()


def test_failure_is_parked_behind_backoff_not_retried_next_tick(db, monkeypatch):
    monkeypatch.setattr(ew, "_STAGES", [("s", lambda db, task, store=None: (_ for _ in ()).throw(RuntimeError("boom")))])
    _enqueue(db)
    first = ew.run_pending(db)
    assert first["retried"] == 1
    # The very next tick (default backoff 30 s) must not claim it again.
    assert ew.run_pending(db)["claimed"] == 0
    assert _row(db)["status"] == "pending"
    assert _row(db)["next_attempt_at"] > ew._now()[:19]


def test_row_survives_a_short_provider_outage(db, monkeypatch):
    """Stage fails for OUTAGE_SEC then succeeds: the row must end 'done', not 'failed'."""
    monkeypatch.setenv("MEMORY_ENRICH_RETRY_BASE_SEC", str(BASE_SEC))
    clock = {"now": datetime.now(timezone.utc)}
    outage_ends = clock["now"] + timedelta(seconds=OUTAGE_SEC)

    def flaky(db, task, store=None):
        if clock["now"] < outage_ends:
            raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(ew, "_STAGES", [("atomic_facts", flaky)])
    monkeypatch.setattr(ew, "now_iso", lambda: clock["now"].strftime("%Y-%m-%dT%H:%M:%S.%fZ"))
    _enqueue(db)
    for _ in range(200):
        ew.run_pending(db)
        if _row(db)["status"] == "done":
            break
        clock["now"] += timedelta(seconds=0.1)
    assert _row(db)["status"] == "done"


def test_provider_outage_does_not_consume_attempts(db, monkeypatch):
    monkeypatch.setenv("MEMORY_ENRICH_RETRY_BASE_SEC", "0")
    monkeypatch.setattr(ew, "_STAGES", [("s", lambda db, task, store=None: (_ for _ in ()).throw(ProviderUnavailable("down")))])
    _enqueue(db)
    for _ in range(ew._max_attempts() + 3):
        ew.run_pending(db)
    row = _row(db)
    assert row["status"] == "pending"
    assert row["attempts"] == 0
    assert row["deferrals"] == ew._max_attempts() + 3


def test_non_transient_failures_still_fail_after_max_attempts(db, monkeypatch):
    monkeypatch.setenv("MEMORY_ENRICH_RETRY_BASE_SEC", "0")
    monkeypatch.setattr(ew, "_STAGES", [("s", lambda db, task, store=None: (_ for _ in ()).throw(ValueError("bad")))])
    _enqueue(db)
    for _ in range(ew._max_attempts() + 2):
        ew.run_pending(db)
    assert _row(db)["status"] == "failed"


def test_failed_rows_are_requeued_by_retry_failed(db, monkeypatch):
    monkeypatch.setenv("MEMORY_ENRICH_RETRY_BASE_SEC", "0")
    monkeypatch.setenv("MEMORY_ENRICH_REQUEUE_AFTER_SEC", "0")
    monkeypatch.setattr(ew, "_STAGES", [("s", lambda db, task, store=None: (_ for _ in ()).throw(ValueError("bad")))])
    _enqueue(db)
    for _ in range(ew._max_attempts()):
        ew.run_pending(db)
    assert _row(db)["status"] == "failed"
    assert ew.retry_failed(db) == 1
    assert _row(db)["status"] == "pending"
    assert _row(db)["attempts"] == 0


def test_failed_rows_are_requeued_automatically_after_cooldown(db, monkeypatch):
    monkeypatch.setenv("MEMORY_ENRICH_RETRY_BASE_SEC", "0")
    monkeypatch.setenv("MEMORY_ENRICH_REQUEUE_AFTER_SEC", "60")
    monkeypatch.setattr(ew, "_STAGES", [("s", lambda db, task, store=None: (_ for _ in ()).throw(ValueError("bad")))])
    _enqueue(db)
    for _ in range(ew._max_attempts()):
        ew.run_pending(db)
    assert _row(db)["status"] == "failed"
    assert ew.requeue_failed_due(db) == 0  # cool-down not over yet
    old = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    db.execute("UPDATE enrichment_queue SET finished_at=?", (old,))
    assert ew.requeue_failed_due(db) == 1
    assert _row(db)["requeues"] == 1


def test_requeue_is_bounded(db, monkeypatch):
    monkeypatch.setenv("MEMORY_ENRICH_REQUEUE_AFTER_SEC", "1")
    _enqueue(db)
    ew._ensure_schema(db)
    old = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    db.execute("UPDATE enrichment_queue SET status='failed', finished_at=?, requeues=?", (old, ew.MAX_REQUEUES))
    assert ew.requeue_failed_due(db) == 0


def test_poison_row_is_failed_on_reclaim_not_retried_forever(db, monkeypatch):
    """A row whose stage kills the process never reaches mark_failed; reclaim must cap it."""
    monkeypatch.setenv("MEMORY_ENRICH_STALE_AFTER_SEC", "5")
    _enqueue(db)
    ew._ensure_schema(db)
    old = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    db.execute(
        "UPDATE enrichment_queue SET status='processing', attempts=?, started_at=?, lease_token='x'",
        (ew._max_attempts(), old),
    )
    ew.reclaim_stale(db)
    assert _row(db)["status"] == "failed"


def test_claim_skips_rows_at_the_attempt_limit(db):
    _enqueue(db)
    ew._ensure_schema(db)
    db.execute("UPDATE enrichment_queue SET attempts=?", (ew._max_attempts(),))
    assert ew._claim_pending(db, 5) == []


def test_completed_stages_are_not_rerun_on_retry(db, monkeypatch):
    monkeypatch.setenv("MEMORY_ENRICH_RETRY_BASE_SEC", "0")
    calls = {"episodic": 0, "flaky": 0}

    def episodic(db, task, store=None):
        calls["episodic"] += 1

    def flaky(db, task, store=None):
        calls["flaky"] += 1
        if calls["flaky"] < 3:
            raise ValueError("transient bug")

    monkeypatch.setattr(ew, "_STAGES", [("episodic", episodic), ("flaky", flaky)])
    _enqueue(db)
    for _ in range(3):
        ew.run_pending(db)
    assert _row(db)["status"] == "done"
    assert calls == {"episodic": 1, "flaky": 3}
    assert db.execute("SELECT COUNT(*) FROM enrichment_stage_done").fetchone()[0] == 0


def test_retryable_quality_verdict_fails_the_stage(db, monkeypatch):
    import types

    score = types.SimpleNamespace(retryable=True, reason="LLM error: down", decision="error")
    fake = types.SimpleNamespace(score_quality=lambda *a, **k: score, log_decision=lambda *a, **k: None)
    monkeypatch.setitem(sys.modules, "quality_gate", fake)
    _enqueue(db)
    task = ew._claim_pending(db, 1)[0]
    with pytest.raises(ProviderUnavailable):
        ew._run_quality_gate(db, task)


def test_error_verdicts_from_contradiction_detector_fail_the_stage(db, monkeypatch):
    import types

    import contradiction_detector as cd

    verdict = cd.ContradictionVerdict(candidate_id=2, cosine=0.9, llm_confidence=None,
                                      decision="error", reason="LLM error: down")
    monkeypatch.setattr(cd, "should_run", lambda ktype: (True, ""))
    monkeypatch.setattr(cd, "detect_contradictions", lambda *a, **k: [verdict])
    monkeypatch.setattr(cd, "apply_and_log", lambda *a, **k: {"error": 1})
    store = types.SimpleNamespace(embed=lambda texts: [[0.1]], _binary_search=lambda *a, **k: [(2, 0.9)])
    _enqueue(db)
    task = ew._claim_pending(db, 1)[0]
    with pytest.raises(ProviderUnavailable):
        ew._run_contradiction_detector(db, task, store=store)


def test_llm_down_contradiction_gate_is_a_deferral_not_a_pass(db, monkeypatch):
    import contradiction_detector as cd

    monkeypatch.setattr(cd, "should_run", lambda ktype: (False, cd.LLM_UNAVAILABLE_REASON))
    _enqueue(db)
    task = ew._claim_pending(db, 1)[0]
    with pytest.raises(ProviderUnavailable):
        ew._run_contradiction_detector(db, task, store=object())


def test_retry_failed_cli(tmp_path, monkeypatch, capsys):
    path = tmp_path / "memory.db"
    conn = _make_db()
    conn.close()
    disk = sqlite3.connect(path)
    disk.row_factory = sqlite3.Row
    disk.executescript(
        "CREATE TABLE enrichment_queue (id INTEGER PRIMARY KEY AUTOINCREMENT, knowledge_id INTEGER NOT NULL,"
        " session_id TEXT, project TEXT NOT NULL DEFAULT 'general', ktype TEXT NOT NULL,"
        " content_snapshot TEXT NOT NULL, tags_snapshot TEXT NOT NULL DEFAULT '[]',"
        " importance TEXT NOT NULL DEFAULT 'medium', skip_quality INTEGER NOT NULL DEFAULT 0,"
        " status TEXT NOT NULL DEFAULT 'pending', attempts INTEGER NOT NULL DEFAULT 0, last_error TEXT,"
        " enqueued_at TEXT NOT NULL, lease_token TEXT, heartbeat_at TEXT, atomic_only INTEGER DEFAULT 0,"
        " started_at TEXT, finished_at TEXT);"
    )
    _enqueue(disk)
    disk.execute("UPDATE enrichment_queue SET status='failed', finished_at='2026-01-01T00:00:00Z'")
    disk.commit()
    disk.close()
    import paths
    monkeypatch.setattr(paths, "memory_db", lambda: path)
    assert ew.main(["--retry-failed"]) == 0
    assert "re-queued 1" in capsys.readouterr().out
