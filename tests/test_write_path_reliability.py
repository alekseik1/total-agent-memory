"""Store-level write-path reliability (migration 039), on SQLite and PostgreSQL.

* embedding backfill: a save whose embedding fails is queued, retried with
  backoff, dead-lettered after the attempt budget, and embedded once the
  provider recovers;
* outbox: a crash after the INSERT but before mark_committed does not
  duplicate (or resurrect) the record on restart;
* error journal and importance budget surface in memory_stats.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from tests.pg_store_support import store_backend  # noqa: F401 — fixture

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

SESSION = "write-path-session"
PROJECT = "reliability"
CONTENT = "Billing exports run nightly at 02:00 UTC from the replica database"
BACKDATE_SECONDS = 120


@pytest.fixture
def memory_dir(monkeypatch, tmp_path):
    import server

    monkeypatch.setattr(server, "MEMORY_DIR", tmp_path)
    monkeypatch.setenv("MEMORY_ASYNC_ENRICHMENT", "false")
    return tmp_path


@pytest.fixture
def make_store(store_backend, memory_dir):  # noqa: F811 — pytest fixture injection
    import server

    opened = []

    def make():
        store = server.Store(database=store_backend)
        opened.append(store)
        return store

    yield make
    for store in opened:
        thread = getattr(store, "_embedding_backfill_thread", None)
        if thread is not None:
            thread.join()
        store.db.close()


class FlakyEmbedder:
    """Fails until `recover()`; then returns a fixed vector of the store's dimension."""

    def __init__(self, store):
        self.real = store.embed
        self.healthy = False
        self.calls = 0

    def recover(self):
        self.healthy = True

    def __call__(self, texts):
        self.calls += 1
        if not self.healthy:
            return None
        return self.real(texts)


def _save(store, content=CONTENT, **kwargs):
    kwargs.setdefault("skip_quality", True)
    rid, *_ = store.save_knowledge(SESSION, content, kwargs.pop("ktype", "fact"), project=PROJECT, **kwargs)
    return rid


def _has_vector(store, rid) -> bool:
    return store.db.execute("SELECT COUNT(*) FROM embeddings WHERE knowledge_id=?", (rid,)).fetchone()[0] == 1


def _make_due(store, rid):
    past = (datetime.now(UTC) - timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    store.db.execute("UPDATE embedding_pending SET next_attempt_at=? WHERE knowledge_id=?", (past, rid))
    store.db.commit()


def _pending(store, rid):
    row = store.db.execute(
        "SELECT attempts, status, last_error, next_attempt_at FROM embedding_pending WHERE knowledge_id=?",
        (rid,)).fetchone()
    return dict(row) if row is not None else None


# ─── embedding backfill ─────────────────────────────────────────────────────

def test_failed_embedding_is_queued_retried_and_backfilled(make_store, monkeypatch):
    store = make_store()
    flaky = FlakyEmbedder(store)
    monkeypatch.setattr(store, "embed", flaky)

    rid = _save(store)
    assert rid is not None and not _has_vector(store, rid)
    queued = _pending(store, rid)
    assert queued["status"] == "pending" and queued["attempts"] == 0
    # The outbox intent is committed: the backfill queue owns the retry now.
    assert store.db.execute("SELECT status FROM write_intents WHERE knowledge_id=?", (rid,)).fetchone()[0] == "committed"

    # Provider still down: one attempt is spent and the row backs off.
    counts = store.drain_embedding_backfill()
    assert counts["claimed"] == 1 and counts["retried"] == 1 and counts["embedded"] == 0
    after_failure = _pending(store, rid)
    assert after_failure["attempts"] == 1 and after_failure["status"] == "pending"
    assert after_failure["next_attempt_at"] > datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    assert "no vector" in after_failure["last_error"]
    # Not due yet: a second drain does nothing.
    assert store.drain_embedding_backfill()["claimed"] == 0

    flaky.recover()
    _make_due(store, rid)
    counts = store.drain_embedding_backfill()
    assert counts["embedded"] == 1
    assert _has_vector(store, rid)
    assert _pending(store, rid) is None


def test_backfill_dead_letters_after_max_attempts_and_can_be_retried(make_store, monkeypatch):
    from embedding_backfill import MAX_ATTEMPTS

    store = make_store()
    flaky = FlakyEmbedder(store)
    monkeypatch.setattr(store, "embed", flaky)
    rid = _save(store)
    for _ in range(MAX_ATTEMPTS):
        _make_due(store, rid)
        store.drain_embedding_backfill()
    dead = _pending(store, rid)
    assert dead["status"] == "failed" and dead["attempts"] == MAX_ATTEMPTS
    _make_due(store, rid)
    assert store.drain_embedding_backfill()["claimed"] == 0  # dead letters are not drained

    assert store._embedding_backfill().retry_dead_letter() == 1
    flaky.recover()
    assert store.drain_embedding_backfill()["embedded"] == 1
    assert _has_vector(store, rid) and _pending(store, rid) is None


def test_backfill_drops_rows_of_records_that_are_no_longer_active(make_store, monkeypatch):
    store = make_store()
    monkeypatch.setattr(store, "embed", FlakyEmbedder(store))
    rid = _save(store)
    store.delete_knowledge(rid)
    counts = store.drain_embedding_backfill()
    assert counts["gone"] == 1 and _pending(store, rid) is None


def test_save_with_working_provider_queues_nothing(make_store):
    store = make_store()
    rid = _save(store)
    assert _has_vector(store, rid) and _pending(store, rid) is None


def test_startup_drains_due_rows_in_the_background(make_store, monkeypatch):
    first = make_store()
    monkeypatch.setattr(first, "embed", FlakyEmbedder(first))
    rid = _save(first)
    assert _pending(first, rid) is not None

    second = make_store()  # provider healthy again: the startup thread embeds the row
    assert second._embedding_backfill_thread is not None
    second._embedding_backfill_thread.join()
    assert _has_vector(second, rid) and _pending(second, rid) is None


def test_memory_stats_report_backfill_outbox_errors_and_budget(make_store, monkeypatch):
    import server

    store = make_store()
    monkeypatch.setattr(store, "embed", FlakyEmbedder(store))
    _save(store)
    store.error_journal.log_error("save.auto_link", RuntimeError("graph down"), "Auto-link error")
    stats = server.Recall(store).stats()["write_path"]
    assert stats["embedding_backfill"]["pending"] == 1 and stats["embedding_backfill"]["failed"] == 0
    assert stats["outbox"].get("committed", 0) >= 1
    assert stats["errors_24h"]["by_component"]["save.auto_link"] == 1
    assert stats["importance_budget"]["budgets"]["critical"] >= 0


# ─── outbox crash recovery ──────────────────────────────────────────────────

def _crash_before_mark(store, monkeypatch, content=CONTENT):
    """Save with mark_committed failing to run, as if the process died right before it."""
    import outbox

    with monkeypatch.context() as patch:
        patch.setattr(outbox, "mark_committed", lambda db, intent, knowledge_id: None)
        rid = _save(store, content)
    intent = store.db.execute(
        "SELECT id, status FROM write_intents ORDER BY id DESC LIMIT 1").fetchone()
    assert intent[1] == "pending"
    old = (datetime.now(UTC) - timedelta(seconds=BACKDATE_SECONDS)).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    store.db.execute("UPDATE write_intents SET created_at=? WHERE id=?", (old, intent[0]))
    store.db.commit()
    return rid, intent[0]


def test_crash_after_insert_before_mark_does_not_duplicate(make_store, monkeypatch):
    store = make_store()
    rid, intent_id = _crash_before_mark(store, monkeypatch)

    restarted = make_store()  # Store.__init__ runs the reconciler
    rows = restarted.db.execute(
        "SELECT id FROM knowledge WHERE project=? AND content=?", (PROJECT, CONTENT)).fetchall()
    assert [row[0] for row in rows] == [rid]
    intent = restarted.db.execute(
        "SELECT status, knowledge_id FROM write_intents WHERE id=?", (intent_id,)).fetchone()
    assert (intent[0], intent[1]) == ("committed", rid)


def test_crash_then_delete_is_not_resurrected_by_the_reconciler(make_store, monkeypatch):
    store = make_store()
    rid, _ = _crash_before_mark(store, monkeypatch)
    store.delete_knowledge(rid)

    restarted = make_store()
    assert restarted.db.execute(
        "SELECT COUNT(*) FROM knowledge WHERE content=?", (CONTENT,)).fetchone()[0] == 1
    assert restarted.db.execute(
        "SELECT status FROM knowledge WHERE id=?", (rid,)).fetchone()[0] == "deleted"


def test_crash_that_also_skipped_the_embedding_queues_a_backfill(make_store, monkeypatch):
    store = make_store()
    rid, _ = _crash_before_mark(store, monkeypatch)
    store.db.execute("DELETE FROM embeddings WHERE knowledge_id=?", (rid,))
    store.db.commit()

    restarted = make_store()
    thread = restarted._embedding_backfill_thread
    if thread is not None:
        thread.join()
    # Queued by the reconciler; the drain (startup thread or reflection) embeds it.
    if _pending(restarted, rid) is not None:
        restarted.drain_embedding_backfill()
    assert _has_vector(restarted, rid)


# ─── importance budget ──────────────────────────────────────────────────────

def test_critical_saves_beyond_the_daily_budget_are_lowered_with_a_note(make_store, monkeypatch):
    monkeypatch.setenv("MEMORY_CRITICAL_DAILY_BUDGET", "2")
    store = make_store()
    metas = []
    for n in range(3):
        rid, _, _, _, meta = store.save_knowledge(
            SESSION, f"Decision {n}: shard the events table by tenant id {n}", "decision",
            project=PROJECT, importance="critical", skip_quality=True)
        metas.append((rid, meta))
    levels = [store.db.execute("SELECT importance FROM knowledge WHERE id=?", (rid,)).fetchone()[0]
              for rid, _ in metas]
    assert levels == ["critical", "critical", "high"]
    note = metas[2][1]["importance_budget"]
    assert note["requested"] == "critical" and note["applied"] == "high" and note["budget"] == 2
    assert "critical" in note["note"] and "high" in note["note"]
    audit = store.db.execute(
        "SELECT knowledge_id, requested, applied FROM importance_budget_log").fetchall()
    assert [tuple(row) for row in audit] == [(metas[2][0], "critical", "high")]


def test_conventions_are_exempt_and_projects_have_separate_budgets(make_store, monkeypatch):
    monkeypatch.setenv("MEMORY_CRITICAL_DAILY_BUDGET", "1")
    store = make_store()
    for n in range(2):
        rid, *_ = store.save_knowledge(SESSION, f"Convention {n}: API errors use RFC 9457 bodies {n}",
                                       "convention", project=PROJECT, importance="critical", skip_quality=True)
        assert store.db.execute("SELECT importance FROM knowledge WHERE id=?", (rid,)).fetchone()[0] == "critical"
    for project in ("alpha", "beta"):
        rid, *_ = store.save_knowledge(SESSION, f"Decision for {project}: keep Redis 7 for queues",
                                       "decision", project=project, importance="critical", skip_quality=True)
        assert store.db.execute("SELECT importance FROM knowledge WHERE id=?", (rid,)).fetchone()[0] == "critical"


def test_budget_note_survives_async_enrichment(make_store, store_backend, monkeypatch):  # noqa: F811
    if store_backend is not None:
        pytest.skip("async enrichment is SQLite-only (enrichment_worker.reject_postgres)")
    monkeypatch.setenv("MEMORY_ASYNC_ENRICHMENT", "true")
    monkeypatch.setenv("MEMORY_CRITICAL_DAILY_BUDGET", "1")
    store = make_store()
    try:
        metas = [store.save_knowledge(SESSION, f"Decision {n}: retire the legacy cron host {n}", "decision",
                                      project=PROJECT, importance="critical")[4] for n in range(2)]
    finally:
        worker = getattr(store, "_enrich_worker", None)
        if worker is not None:
            worker.stop()
            worker.join()
    assert metas[1]["decision"] == "pending"
    assert metas[1]["importance_budget"]["applied"] == "high"


@pytest.mark.parametrize("tool", ["memory_save", "memory_save_fast"])
def test_save_tools_return_the_importance_note(make_store, monkeypatch, tool):
    import asyncio
    import json

    import server

    monkeypatch.setenv("MEMORY_CRITICAL_DAILY_BUDGET", "1")
    monkeypatch.setenv("MEMORY_QUALITY_GATE_ENABLED", "false")
    store = make_store()
    monkeypatch.setattr(server, "store", store, raising=False)
    monkeypatch.setattr(server, "recall", server.Recall(store), raising=False)
    monkeypatch.setattr(server, "SID", SESSION, raising=False)
    monkeypatch.setattr(server, "BRANCH", "", raising=False)

    def call(content):
        return json.loads(asyncio.run(server._do(tool, {
            "content": content, "type": "decision", "project": PROJECT, "importance": "critical"})))

    first = call("Decision: the ledger service owns all currency conversions")
    assert first["saved"] is True and "importance_note" not in first
    second = call("Decision: invoices are immutable once issued to a customer")
    assert second["importance"] == "high"
    assert "lowered from critical to high" in second["importance_note"]

    stats = json.loads(asyncio.run(server._do("memory_stats", {})))
    assert stats["write_path"]["importance_budget"]["clamped_today"] == {PROJECT: {"critical": 1}}
