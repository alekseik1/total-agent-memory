"""Tests for the v10 outbox / WriteIntent journal."""

from __future__ import annotations

import json
import sqlite3
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import outbox

KNOWLEDGE_DDL = """
CREATE TABLE knowledge (
    id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, type TEXT NOT NULL,
    content TEXT NOT NULL, project TEXT DEFAULT 'general', status TEXT DEFAULT 'active',
    created_at TEXT NOT NULL
);
CREATE INDEX idx_knowledge_neighbors ON knowledge(project, session_id, status, created_at, id);
"""


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("MEMORY_OUTBOX_ENABLED", "true")
    monkeypatch.delenv("MEMORY_OUTBOX_GRACE_SEC", raising=False)
    monkeypatch.delenv("MEMORY_OUTBOX_MAX_ATTEMPTS", raising=False)
    yield


@pytest.fixture
def odb():
    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    migrations = Path(__file__).parent.parent / "migrations"
    db.executescript((migrations / "017_outbox.sql").read_text())
    # The columns of `knowledge` that find_existing_by_hash reads.
    db.executescript(KNOWLEDGE_DDL)
    db.executescript((migrations / "039_write_path_reliability.sql").read_text())
    yield db
    db.close()


def _payload():
    return {
        "sid": "sess-1",
        "content": "Migration 015 adds importance and quality_gate_log tables",
        "ktype": "decision",
        "project": "vito",
        "tags": ["database"],
        "context": "Why: Beever-style audit",
        "branch": "main",
        "importance": "high",
    }


# ──────────────────────────────────────────────
# Hash + envelope
# ──────────────────────────────────────────────


def test_compute_content_hash_is_deterministic():
    a = outbox.compute_content_hash("hello", "fact", "vito")
    b = outbox.compute_content_hash("hello", "fact", "vito")
    assert a == b
    assert outbox.compute_content_hash("hello", "fact", "other") != a
    assert outbox.compute_content_hash("hello", "lesson", "vito") != a


# ──────────────────────────────────────────────
# create + transitions
# ──────────────────────────────────────────────


def test_create_intent_inserts_pending_row(odb):
    p = _payload()
    intent = outbox.create_intent(
        odb, payload=p, session_id="sess-1",
        content=p["content"], ktype=p["ktype"], project=p["project"],
    )
    assert intent is not None
    assert intent.status == "pending"
    assert intent.attempts == 0
    assert intent.knowledge_id is None
    row = odb.execute(
        "SELECT status, knowledge_id, payload_json FROM write_intents WHERE id=?",
        (intent.id,),
    ).fetchone()
    assert row["status"] == "pending"
    assert json.loads(row["payload_json"]) == p


def test_create_intent_returns_none_when_disabled(odb, monkeypatch):
    monkeypatch.setenv("MEMORY_OUTBOX_ENABLED", "false")
    intent = outbox.create_intent(
        odb, payload=_payload(), session_id="s1",
        content="x", ktype="fact", project="vito",
    )
    assert intent is None


def test_mark_committed_promotes_intent(odb):
    p = _payload()
    intent = outbox.create_intent(
        odb, payload=p, session_id="s1",
        content=p["content"], ktype=p["ktype"], project=p["project"],
    )
    outbox.mark_committed(odb, intent, knowledge_id=42)
    row = odb.execute(
        "SELECT status, knowledge_id, committed_at FROM write_intents WHERE id=?",
        (intent.id,),
    ).fetchone()
    assert row["status"] == "committed"
    assert row["knowledge_id"] == 42
    assert row["committed_at"] is not None


def test_mark_superseded_records_dedup_outcome(odb):
    p = _payload()
    intent = outbox.create_intent(
        odb, payload=p, session_id="s1",
        content=p["content"], ktype=p["ktype"], project=p["project"],
    )
    outbox.mark_superseded(odb, intent, knowledge_id=10)
    row = odb.execute(
        "SELECT status FROM write_intents WHERE id=?", (intent.id,),
    ).fetchone()
    assert row["status"] == "superseded"


def test_mark_failed_increments_and_freezes_after_max_attempts(odb, monkeypatch):
    monkeypatch.setenv("MEMORY_OUTBOX_MAX_ATTEMPTS", "2")
    p = _payload()
    intent = outbox.create_intent(
        odb, payload=p, session_id="s1",
        content=p["content"], ktype=p["ktype"], project=p["project"],
    )
    outbox.mark_failed(odb, intent, "first try boom")
    row = odb.execute(
        "SELECT status, attempts, last_error FROM write_intents WHERE id=?",
        (intent.id,),
    ).fetchone()
    assert row["status"] == "pending"
    assert row["attempts"] == 1
    assert row["last_error"] == "first try boom"

    outbox.mark_failed(odb, intent, "second try also boom")
    row = odb.execute(
        "SELECT status, attempts FROM write_intents WHERE id=?", (intent.id,),
    ).fetchone()
    assert row["status"] == "failed"
    assert row["attempts"] == 2


# ──────────────────────────────────────────────
# claim_pending — grace window
# ──────────────────────────────────────────────


def _backdate(odb, intent_id: int, seconds_ago: int):
    """Move the created_at timestamp into the past so claim_pending sees it."""
    older = datetime.now(timezone.utc) - timedelta(seconds=seconds_ago)
    odb.execute(
        "UPDATE write_intents SET created_at=? WHERE id=?",
        (older.isoformat().replace("+00:00", "Z"), intent_id),
    )
    odb.commit()


def test_claim_pending_respects_grace_window(odb, monkeypatch):
    monkeypatch.setenv("MEMORY_OUTBOX_GRACE_SEC", "30")
    p = _payload()
    fresh = outbox.create_intent(
        odb, payload=p, session_id="s1",
        content=p["content"], ktype=p["ktype"], project=p["project"],
    )
    older_p = {**p, "content": p["content"] + " v2"}
    older = outbox.create_intent(
        odb, payload=older_p, session_id="s1",
        content=older_p["content"], ktype=p["ktype"], project=p["project"],
    )
    _backdate(odb, older.id, seconds_ago=120)

    pending = outbox.claim_pending(odb)
    pending_ids = [p.id for p in pending]
    # Fresh intent (created just now) is inside the grace window — skip.
    assert fresh.id not in pending_ids
    # Backdated intent passes — needs replay.
    assert older.id in pending_ids


def test_claim_pending_skips_committed(odb):
    p = _payload()
    intent = outbox.create_intent(
        odb, payload=p, session_id="s1",
        content=p["content"], ktype=p["ktype"], project=p["project"],
    )
    outbox.mark_committed(odb, intent, knowledge_id=1)
    _backdate(odb, intent.id, seconds_ago=300)
    assert outbox.claim_pending(odb, grace_seconds=10) == []


# ──────────────────────────────────────────────
# reconcile_pending
# ──────────────────────────────────────────────


def test_reconcile_replays_and_marks_committed(odb):
    p = _payload()
    intent = outbox.create_intent(
        odb, payload=p, session_id="s1",
        content=p["content"], ktype=p["ktype"], project=p["project"],
    )
    _backdate(odb, intent.id, seconds_ago=120)

    seen = []

    def replay(payload):
        seen.append(payload)
        return 99  # pretend save_knowledge returned id=99

    counts = outbox.reconcile_pending(odb, replay_fn=replay)
    assert counts == {"replayed": 1, "dedup": 0, "failed": 0, "skipped": 0, "recovered": 0}
    assert seen == [p]
    row = odb.execute(
        "SELECT status, knowledge_id FROM write_intents WHERE id=?", (intent.id,),
    ).fetchone()
    assert row["status"] == "committed"
    assert row["knowledge_id"] == 99


def test_reconcile_marks_dedup_when_replay_returns_none(odb):
    p = _payload()
    intent = outbox.create_intent(
        odb, payload=p, session_id="s1",
        content=p["content"], ktype=p["ktype"], project=p["project"],
    )
    _backdate(odb, intent.id, seconds_ago=120)

    counts = outbox.reconcile_pending(odb, replay_fn=lambda payload: None)
    assert counts["dedup"] == 1
    row = odb.execute(
        "SELECT status FROM write_intents WHERE id=?", (intent.id,),
    ).fetchone()
    assert row["status"] == "superseded"


def test_reconcile_records_failures(odb):
    p = _payload()
    intent = outbox.create_intent(
        odb, payload=p, session_id="s1",
        content=p["content"], ktype=p["ktype"], project=p["project"],
    )
    _backdate(odb, intent.id, seconds_ago=120)

    def boom(payload):
        raise RuntimeError("provider down")

    counts = outbox.reconcile_pending(odb, replay_fn=boom)
    assert counts == {"replayed": 0, "dedup": 0, "failed": 1, "skipped": 0, "recovered": 0}
    row = odb.execute(
        "SELECT status, attempts, last_error FROM write_intents WHERE id=?",
        (intent.id,),
    ).fetchone()
    assert row["status"] == "pending"
    assert row["attempts"] == 1
    assert "provider down" in row["last_error"]


def test_reconcile_skips_intents_already_at_max_attempts(odb, monkeypatch):
    monkeypatch.setenv("MEMORY_OUTBOX_MAX_ATTEMPTS", "1")
    p = _payload()
    intent = outbox.create_intent(
        odb, payload=p, session_id="s1",
        content=p["content"], ktype=p["ktype"], project=p["project"],
    )
    _backdate(odb, intent.id, seconds_ago=120)
    odb.execute("UPDATE write_intents SET attempts=1 WHERE id=?", (intent.id,))
    odb.commit()

    seen = []
    counts = outbox.reconcile_pending(odb, replay_fn=lambda p: seen.append(p) or 1)
    assert seen == []   # never called
    assert counts["skipped"] == 1
    row = odb.execute(
        "SELECT status FROM write_intents WHERE id=?", (intent.id,),
    ).fetchone()
    assert row["status"] == "failed"


def test_get_status_counts(odb):
    p = _payload()
    a = outbox.create_intent(
        odb, payload=p, session_id="s1",
        content=p["content"], ktype=p["ktype"], project=p["project"],
    )
    b = outbox.create_intent(
        odb, payload={**p, "content": p["content"] + "B"},
        session_id="s1", content=p["content"] + "B",
        ktype=p["ktype"], project=p["project"],
    )
    outbox.mark_committed(odb, a, knowledge_id=1)
    counts = outbox.get_status_counts(odb)
    assert counts.get("committed") == 1
    assert counts.get("pending") == 1


# ──────────────────────────────────────────────
# Leases (migration 039)
# ──────────────────────────────────────────────


def _canonical(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _pending_intent(db, content="lease me", seconds_ago=120):
    p = {**_payload(), "content": content}
    intent = outbox.create_intent(
        db, payload=p, session_id=p["sid"],
        content=p["content"], ktype=p["ktype"], project=p["project"],
    )
    _backdate(db, intent.id, seconds_ago=seconds_ago)
    return intent


def test_claim_pending_leases_so_a_second_claim_gets_nothing(odb):
    intent = _pending_intent(odb)
    first = outbox.claim_pending(odb, grace_seconds=10)
    assert [i.id for i in first] == [intent.id]
    assert outbox.claim_pending(odb, grace_seconds=10) == []
    row = odb.execute("SELECT lease_until, status FROM write_intents WHERE id=?",
                      (intent.id,)).fetchone()
    assert row["status"] == "pending" and row["lease_until"] is not None


def test_expired_lease_is_claimable_again(odb):
    intent = _pending_intent(odb)
    assert outbox.claim_pending(odb, grace_seconds=10)
    expired = _canonical(datetime.now(timezone.utc) - timedelta(seconds=1))
    odb.execute("UPDATE write_intents SET lease_until=? WHERE id=?", (expired, intent.id))
    odb.commit()
    assert [i.id for i in outbox.claim_pending(odb, grace_seconds=10)] == [intent.id]


def _file_db(path):
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    migrations = Path(__file__).parent.parent / "migrations"
    conn.executescript((migrations / "017_outbox.sql").read_text())
    conn.executescript(KNOWLEDGE_DDL)
    conn.executescript((migrations / "039_write_path_reliability.sql").read_text())
    return conn


INTENT_COUNT = 20
REPLAY_SECONDS = 0.01
LOCK_WAIT_SECONDS = 10


def test_two_reconcilers_on_separate_connections_never_replay_the_same_intent(tmp_path):
    import threading

    path = tmp_path / "outbox.db"
    setup = _file_db(path)
    for n in range(INTENT_COUNT):
        _pending_intent(setup, content=f"intent {n}")
    setup.close()

    replayed: list[str] = []
    lock = threading.Lock()
    barrier = threading.Barrier(2)
    errors: list[BaseException] = []

    def run():
        conn = sqlite3.connect(path, timeout=LOCK_WAIT_SECONDS)
        try:
            barrier.wait()

            def replay(payload):
                with lock:
                    replayed.append(payload["content"])
                time.sleep(REPLAY_SECONDS)
                return 1

            outbox.reconcile_pending(conn, replay_fn=replay, grace_seconds=10, limit=INTENT_COUNT)
        except BaseException as exc:  # noqa: BLE001 — surfaced by the assertion below
            errors.append(exc)
        finally:
            conn.close()

    threads = [threading.Thread(target=run) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert sorted(replayed) == sorted(f"intent {n}" for n in range(INTENT_COUNT))
    assert len(replayed) == len(set(replayed))


# ──────────────────────────────────────────────
# find_existing_by_hash
# ──────────────────────────────────────────────


def _insert_knowledge(db, payload, created_at):
    cur = db.execute(
        "INSERT INTO knowledge (session_id, type, content, project, created_at) VALUES (?, ?, ?, ?, ?)",
        (payload["sid"], payload["ktype"], payload["content"], payload["project"], created_at))
    db.commit()
    return cur.lastrowid


def test_find_existing_by_hash_matches_only_the_same_record(odb):
    p = _payload()
    chash = outbox.compute_content_hash(p["content"], p["ktype"], p["project"])
    stamp = _canonical(datetime.now(timezone.utc))
    assert outbox.find_existing_by_hash(odb, chash, p, since=stamp) is None
    _insert_knowledge(odb, {**p, "content": p["content"] + " (other)"}, stamp)
    _insert_knowledge(odb, {**p, "ktype": "fact"}, stamp)
    _insert_knowledge(odb, {**p, "project": "elsewhere"}, stamp)
    _insert_knowledge(odb, {**p, "sid": "another-session"}, stamp)
    assert outbox.find_existing_by_hash(odb, chash, p, since=stamp) is None
    rid = _insert_knowledge(odb, p, stamp)
    assert outbox.find_existing_by_hash(odb, chash, p, since=stamp) == rid
    # A hash that does not belong to the payload never matches.
    assert outbox.find_existing_by_hash(odb, "0" * 40, p, since=stamp) is None


def test_find_existing_by_hash_uses_the_neighbours_index(odb):
    plan = " ".join(row[3] for row in odb.execute(
        "EXPLAIN QUERY PLAN SELECT id, type, project, content FROM knowledge "
        "WHERE project = ? AND session_id = ? AND created_at >= ? ORDER BY created_at, id",
        ("p", "s", "")))
    assert "idx_knowledge_neighbors" in plan


def test_reconcile_closes_intent_whose_record_was_inserted_before_the_crash(odb):
    intent = _pending_intent(odb)
    # The save stamps the row a moment before it opens the intent.
    rid = _insert_knowledge(odb, intent.payload, _canonical(datetime.now(timezone.utc) - timedelta(seconds=125)))
    recovered = []

    def replay(payload):
        raise AssertionError("must not replay a record that was already inserted")

    counts = outbox.reconcile_pending(odb, replay_fn=replay, grace_seconds=10,
                                      on_recovered=recovered.append)
    assert counts["recovered"] == 1 and counts["replayed"] == 0
    assert recovered == [rid]
    row = odb.execute("SELECT status, knowledge_id FROM write_intents WHERE id=?",
                      (intent.id,)).fetchone()
    assert (row["status"], row["knowledge_id"]) == ("committed", rid)
    assert odb.execute("SELECT COUNT(*) FROM knowledge").fetchone()[0] == 1
