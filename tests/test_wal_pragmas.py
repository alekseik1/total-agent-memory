"""Tests for WAL bloat / write-lock fixes in Store.__init__ (journal_size_limit,
busy_timeout, opportunistic startup checkpoint)."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def memory_dir(monkeypatch, tmp_path):
    import server
    monkeypatch.setattr(server, "MEMORY_DIR", tmp_path)
    return server, tmp_path


def test_journal_size_limit_and_busy_timeout_applied(memory_dir):
    server, tmp_path = memory_dir
    store = server.Store()
    try:
        limit = store.db.execute("PRAGMA journal_size_limit").fetchone()[0]
        timeout = store.db.execute("PRAGMA busy_timeout").fetchone()[0]
        assert limit == server.DB_JOURNAL_SIZE_LIMIT_BYTES
        assert timeout == server.DB_BUSY_TIMEOUT_MS
    finally:
        store.db.close()


def test_store_init_survives_busy_startup_checkpoint(memory_dir):
    server, tmp_path = memory_dir
    warmup = server.Store()
    warmup.db.execute(
        "INSERT INTO sessions (id, started_at, project, status) "
        "VALUES ('sess-wal-test', '2026-04-19T00:00:00Z', 'myproj', 'open')"
    )
    warmup.db.commit()
    warmup.db.close()

    db_path = str(tmp_path / "memory.db")
    blocker = sqlite3.connect(db_path)
    blocker.execute("BEGIN")
    blocker.execute("SELECT count(*) FROM sessions").fetchone()

    sneak = sqlite3.connect(db_path)
    sneak.execute("PRAGMA busy_timeout=2000")
    sneak.execute(
        "INSERT INTO sessions (id, started_at, project, status) "
        "VALUES ('sess-wal-test-2', '2026-04-19T00:00:01Z', 'myproj', 'open')"
    )
    sneak.commit()
    sneak.close()

    try:
        store = server.Store()
        try:
            ids = {row[0] for row in store.db.execute("SELECT id FROM sessions")}
            assert {"sess-wal-test", "sess-wal-test-2"} <= ids
            timeout = store.db.execute("PRAGMA busy_timeout").fetchone()[0]
            assert timeout == server.DB_BUSY_TIMEOUT_MS
        finally:
            store.db.close()
    finally:
        blocker.close()
