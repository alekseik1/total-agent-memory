"""A tool call whose write fails must not leave the shared connection inside a transaction.

A leaked implicit transaction pins a stale snapshot: every later write through the
same connection fails with SQLITE_BUSY_SNAPSHOT until the process restarts.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def live_server(monkeypatch, tmp_path):
    for sub in ("raw", "blobs", "chroma"):
        (tmp_path / sub).mkdir(exist_ok=True)
    import server
    monkeypatch.setattr(server, "MEMORY_DIR", tmp_path)
    store = server.Store()
    monkeypatch.setattr(server, "store", store)
    monkeypatch.setattr(server, "recall", server.Recall(store))
    monkeypatch.setattr(server, "SID", "sess-leak")
    monkeypatch.setattr(server, "BRANCH", "")
    monkeypatch.setattr(server, "_v5_modules", {})
    store.db.execute("INSERT INTO sessions (id, started_at, project, status) VALUES (?, ?, ?, ?)",
                     (server.SID, "2026-09-28T00:00:00Z", "p", "open"))
    store.db.commit()
    yield server, store
    store.db.close()


def call(server, tool, **args):
    content, is_error = asyncio.run(server._call_tool_impl(tool, args))
    text = content[0].text
    try:
        return json.loads(text), is_error
    except ValueError:
        return text, is_error


def test_locked_write_does_not_leave_the_transaction_open(live_server):
    server, store = live_server
    store.db.execute("PRAGMA busy_timeout=50")
    blocker = sqlite3.connect(str(store.db_path), isolation_level=None)
    try:
        blocker.execute("BEGIN IMMEDIATE")
        blocker.execute("UPDATE sessions SET status='closed' WHERE id=?", (server.SID,))

        result, is_error = call(server, "memory_save", type="fact", project="p",
                                content="Deploys go through the blue green pipeline on Fridays")

        assert is_error, result
        assert "database is locked" in result
        assert not store.db.in_transaction
        blocker.execute("COMMIT")
    finally:
        blocker.rollback()
        blocker.close()
    result, is_error = call(server, "memory_save", type="fact", project="p",
                            content="Release notes are drafted by the on-call engineer")
    assert not is_error, result
    assert result["saved"] is True


def test_write_error_swallowed_by_a_helper_does_not_leave_the_transaction_open(live_server, monkeypatch):
    monkeypatch.setenv("MEMORY_OUTBOX_ENABLED", "true")
    server, store = live_server
    store.db.execute("CREATE TRIGGER fail_intent_update BEFORE UPDATE ON write_intents "
                     "BEGIN SELECT RAISE(ABORT, 'intent update failed'); END")
    store.db.commit()

    result, is_error = call(server, "memory_save", type="fact", project="p",
                            content="Deploys go through the blue green pipeline on Fridays")

    assert not is_error, result
    assert not store.db.in_transaction
    statuses = store.db.execute("SELECT status FROM write_intents WHERE session_id=?",
                                (server.SID,)).fetchall()
    assert [s for (s,) in statuses] == ["pending"]
