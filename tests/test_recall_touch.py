"""Recall usage statistics are buffered and written off the read path (memory_core.recall_touch)."""

from __future__ import annotations

import signal
import sqlite3
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

import server
from memory_core import recall_touch
from memory_core.recall_touch import RecallTouchBuffer
from tests.pg_store_support import store_backend  # noqa: F401 — fixture

SRC = Path(__file__).resolve().parent.parent / "src"
POLL_SECONDS = 0.01
POLL_LIMIT_SECONDS = 5.0
SLOW_FLUSH_MS = "600000"


def _wait_for(predicate) -> bool:
    deadline = time.monotonic() + POLL_LIMIT_SECONDS
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(POLL_SECONDS)
    return predicate()


@pytest.fixture
def db_file(tmp_path):
    path = tmp_path / "touch.db"
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE knowledge (id INTEGER PRIMARY KEY, recall_count INTEGER DEFAULT 0, "
                       "last_recalled TEXT)")
    connection.executemany("INSERT INTO knowledge(id) VALUES (?)", [(1,), (2,), (3,)])
    connection.commit()
    connection.close()
    return path


def _connector(path):
    return lambda: sqlite3.connect(path, check_same_thread=False)


def _counts(path) -> dict[int, tuple[int, str | None]]:
    connection = sqlite3.connect(path)
    try:
        return {row[0]: (row[1], row[2]) for row in connection.execute(
            "SELECT id, recall_count, last_recalled FROM knowledge ORDER BY id")}
    finally:
        connection.close()


def test_touch_coalesces_and_flush_writes_once(db_file):
    stamps = iter(["2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z"])
    buffer = RecallTouchBuffer(_connector(db_file), flush_interval=600, clock=lambda: next(stamps))
    try:
        buffer.touch([1, 2, 1])
        buffer.touch([1])
        assert buffer.pending() == 2
        assert _counts(db_file)[1] == (0, None)
        assert buffer.flush() == 2
        counts = _counts(db_file)
        assert counts[1] == (3, "2026-01-02T00:00:00Z")
        assert counts[2] == (1, "2026-01-01T00:00:00Z")
        assert counts[3] == (0, None)
        assert buffer.flush() == 0
    finally:
        buffer.close()


def test_background_thread_flushes_and_stops_when_idle(db_file):
    buffer = RecallTouchBuffer(_connector(db_file), flush_interval=0.02)
    try:
        buffer.touch([2, 3])
        assert _wait_for(lambda: _counts(db_file)[2][0] == 1 and _counts(db_file)[3][0] == 1)
        assert _wait_for(lambda: buffer._thread is None), "flusher must exit when nothing is pending"
        buffer.touch([2])
        assert _wait_for(lambda: _counts(db_file)[2][0] == 2), "next touch restarts the flusher"
    finally:
        buffer.close()


def test_close_flushes_pending_and_later_touches_write_directly(db_file):
    buffer = RecallTouchBuffer(_connector(db_file), flush_interval=600)
    buffer.touch([1, 1, 3])
    buffer.close()
    assert _counts(db_file)[1][0] == 2
    assert _counts(db_file)[3][0] == 1
    buffer.close()
    buffer.touch([3])
    assert _counts(db_file)[3][0] == 2
    assert buffer.data_version() is None


def test_close_all_flushes_every_open_buffer(db_file):
    buffer = RecallTouchBuffer(_connector(db_file), flush_interval=600)
    buffer.touch([2])
    recall_touch.close_all()
    assert _counts(db_file)[2][0] == 1
    assert buffer.closed


def test_failed_flush_keeps_counts_and_stops_the_thread(db_file, caplog):
    attempts = []

    def flaky():
        attempts.append(1)
        if len(attempts) == 1:
            raise sqlite3.OperationalError("database is locked")
        return sqlite3.connect(db_file, check_same_thread=False)

    buffer = RecallTouchBuffer(flaky, flush_interval=0.02)
    try:
        buffer.touch([1])
        assert _wait_for(lambda: buffer._thread is None and attempts)
        assert buffer.pending() == 1
        assert "Recall usage flush failed" in caplog.text
        buffer.touch([1])
        assert _wait_for(lambda: _counts(db_file)[1][0] == 2), "both recalls survive the failed flush"
    finally:
        buffer.close()


def test_failed_write_rolls_back_and_merges(db_file):
    buffer = RecallTouchBuffer(_connector(db_file), flush_interval=600)
    try:
        buffer.touch([1])
        connection = buffer._connection_locked()
        connection.execute("DROP TABLE knowledge")
        connection.commit()
        with pytest.raises(sqlite3.OperationalError):
            buffer.flush()
        assert buffer.pending() == 1
    finally:
        buffer._pending.clear()
        buffer.close()


def test_data_version_ignores_own_flushes_but_sees_other_commits(db_file):
    buffer = RecallTouchBuffer(_connector(db_file), flush_interval=600)
    other = sqlite3.connect(db_file)
    try:
        before = buffer.data_version()
        buffer.touch([1])
        buffer.flush()
        assert buffer.data_version() == before
        other.execute("UPDATE knowledge SET recall_count=7 WHERE id=2")
        other.commit()
        assert buffer.data_version() != before
    finally:
        other.close()
        buffer.close()


def test_data_version_does_not_wait_for_a_flush(db_file):
    buffer = RecallTouchBuffer(_connector(db_file), flush_interval=600)
    try:
        with buffer._io_lock:
            assert buffer.data_version() is None
        assert buffer.data_version() is not None
    finally:
        buffer.close()


def test_rejects_non_positive_interval(db_file):
    with pytest.raises(ValueError):
        RecallTouchBuffer(_connector(db_file), flush_interval=0)


# ── Recall integration ───────────────────────────────────────────────────────

@pytest.fixture
def seeded(store_backend, tmp_path, monkeypatch):  # noqa: F811 — pytest fixture injection
    monkeypatch.setattr(server, "MEMORY_DIR", tmp_path)
    monkeypatch.setenv("MEMORY_RECALL_TOUCH_FLUSH_MS", SLOW_FLUSH_MS)
    store = server.Store(database=store_backend)
    store._embed_mode = "none"
    store.session_start("s1", project="usage")
    for text in ("PostgreSQL uses UUID v7 primary keys", "RabbitMQ carries invoice events",
                 "Docker Compose runs the deploys"):
        store.save_knowledge(sid="s1", content=text, ktype="fact", project="usage", skip_dedup=True)
    recall = server.Recall(store)
    try:
        yield store, recall
    finally:
        recall.close()
        store.db.close()


def _total(store) -> int:
    return sum(row[0] for row in store.db.execute(
        "SELECT recall_count FROM knowledge WHERE project='usage'").fetchall())


def test_search_writes_nothing_and_flush_lands_counts(seeded):
    store, recall = seeded
    statements: list[str] = []
    if not store.is_postgres:
        store.db.set_trace_callback(statements.append)
    result = recall.search("UUID primary keys", project="usage", limit=5)
    if not store.is_postgres:
        store.db.set_trace_callback(None)
        assert not [sql for sql in statements if sql.lstrip().upper().startswith(("UPDATE", "COMMIT", "INSERT"))]
    assert result["total"] >= 1
    assert _total(store) == 0, "the search itself must not write usage"
    assert recall.usage_buffer().pending() >= 1
    recall.flush_usage()
    assert _total(store) == result["total"]


def test_shutdown_flushes_buffered_usage(seeded):
    store, recall = seeded
    recall.search("RabbitMQ invoice", project="usage", limit=5)
    assert _total(store) == 0
    recall.close()
    assert _total(store) >= 1


def test_record_usage_false_buffers_nothing(seeded):
    store, recall = seeded
    recall.search("RabbitMQ invoice", project="usage", limit=5, record_usage=False)
    assert recall.usage_buffer().pending() == 0
    recall.close()
    assert _total(store) == 0


def test_cache_hit_counts_usage_too(seeded):
    store, recall = seeded
    first = recall.search("Docker deploys", project="usage", limit=5)
    second = recall.search("Docker deploys", project="usage", limit=5)
    assert first == second
    recall.flush_usage()
    assert _total(store) == 2 * first["total"]


def test_usage_flush_keeps_result_cache_valid(seeded, monkeypatch):
    store, recall = seeded
    if store.is_postgres or store.cache is None:
        pytest.skip("SQLite result cache only")
    recall.search("Docker deploys", project="usage", limit=5)
    recall.flush_usage()
    invalidations = []
    monkeypatch.setattr(store.cache, "invalidate", lambda *a, **k: invalidations.append(1))
    recall.search("Docker deploys", project="usage", limit=5)
    assert invalidations == [], "a usage flush must not invalidate cached results"
    other = sqlite3.connect(store.db_path)
    try:
        other.execute("UPDATE knowledge SET content=content || ' (edited)' WHERE project='usage'")
        other.commit()
    finally:
        other.close()
    recall.search("Docker deploys", project="usage", limit=5)
    assert invalidations, "a commit by another connection must invalidate cached results"


def test_flush_runs_on_a_background_thread(seeded, monkeypatch):
    store, recall = seeded
    monkeypatch.setenv("MEMORY_RECALL_TOUCH_FLUSH_MS", "20")
    fresh = server.Recall(store)
    writers: list[str] = []
    original = RecallTouchBuffer._write

    def spy(self, batch):
        writers.append(threading.current_thread().name)
        return original(self, batch)

    monkeypatch.setattr(RecallTouchBuffer, "_write", spy)
    try:
        fresh.search("UUID primary keys", project="usage", limit=5)
        assert _wait_for(lambda: _total(store) >= 1)
        assert writers and all(name == "recall-touch-flush" for name in writers)
    finally:
        fresh.close()


def test_sigterm_handler_turns_into_system_exit(monkeypatch):
    installed = {}
    monkeypatch.setattr(server.signal, "getsignal", lambda signum: server.signal.SIG_DFL)
    monkeypatch.setattr(server.signal, "signal", lambda signum, handler: installed.setdefault(signum, handler))
    server._flush_on_sigterm()
    handler = installed[server.signal.SIGTERM]
    with pytest.raises(SystemExit) as raised:
        handler(server.signal.SIGTERM, None)
    assert raised.value.code == 128 + server.signal.SIGTERM


def test_sigterm_handler_keeps_a_handler_someone_else_installed(monkeypatch):
    def theirs(signum, frame):
        return None

    calls = []
    monkeypatch.setattr(server.signal, "getsignal", lambda signum: theirs)
    monkeypatch.setattr(server.signal, "signal", lambda *args: calls.append(args))
    server._flush_on_sigterm()
    assert calls == []


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
def test_sigterm_still_flushes_buffered_usage(db_file):
    # Regression: the default SIGTERM action skipped atexit, so counts still in
    # the buffer were lost when a client stopped the server.
    script = textwrap.dedent(f"""
        import os, signal, sqlite3, sys, time
        sys.path.insert(0, {str(SRC)!r})
        import server
        from memory_core.recall_touch import RecallTouchBuffer
        buffer = RecallTouchBuffer(lambda: sqlite3.connect({str(db_file)!r}, check_same_thread=False),
                                   flush_interval=600)
        buffer.touch([1, 2])
        server._flush_on_sigterm()
        os.kill(os.getpid(), signal.SIGTERM)
        time.sleep(30)
    """)
    completed = subprocess.run([sys.executable, "-c", script], timeout=60, capture_output=True, text=True)
    assert completed.returncode == 128 + signal.SIGTERM, completed.stderr
    counts = _counts(db_file)
    assert counts[1][0] == 1 and counts[2][0] == 1 and counts[3][0] == 0


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
def test_sigterm_reaches_the_event_loop_as_a_callback():
    # Regression: the plain handler raised inside anyio's task bookkeeping now and
    # then; the task group broke, the main task never finished and the server hung
    # (1 in 12 stdio exits on macOS). Inside the loop SIGTERM must be a loop callback.
    script = textwrap.dedent(f"""
        import asyncio, os, signal, sys
        sys.path.insert(0, {str(SRC)!r})
        import server
        server._flush_on_sigterm()
        seen = {{}}

        async def finishes():
            seen["plain_handler_inside_loop"] = signal.getsignal(signal.SIGTERM) is server._exit_on_sigterm

        asyncio.run(server._with_sigterm_in_loop(finishes()))
        assert seen["plain_handler_inside_loop"] is False, "SIGTERM still raises at any bytecode in the loop"
        assert signal.getsignal(signal.SIGTERM) is server._exit_on_sigterm, "handler not restored after the loop"

        async def killed():
            os.kill(os.getpid(), signal.SIGTERM)
            await asyncio.sleep(30)

        try:
            asyncio.run(server._with_sigterm_in_loop(killed()))
        except server._Terminated as stop:
            sys.exit(stop.code)
        sys.exit(1)
    """)
    completed = subprocess.run([sys.executable, "-c", script], timeout=60, capture_output=True, text=True)
    assert completed.returncode == 128 + signal.SIGTERM, completed.stderr
