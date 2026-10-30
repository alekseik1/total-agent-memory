"""Tests for the swallowed-error journal (src/error_journal.py)."""

from __future__ import annotations

import sqlite3
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import error_journal as ej
from error_journal import ErrorJournal

MIGRATIONS = Path(__file__).parent.parent / "migrations"
T0 = 1_800_000_000.0
DAY = 86_400


class Clock:
    def __init__(self, now: float):
        self.now = now

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def db():
    conn = sqlite3.connect(":memory:")
    conn.executescript((MIGRATIONS / "017_outbox.sql").read_text(encoding="utf-8"))
    conn.executescript((MIGRATIONS / "039_write_path_reliability.sql").read_text(encoding="utf-8"))
    yield conn
    conn.close()


@pytest.fixture
def clocks():
    return Clock(T0), Clock(0.0)


@pytest.fixture
def journal(db, clocks):
    wall, mono = clocks
    return ErrorJournal(db, clock=wall, monotonic=mono)


def _rows(db):
    return db.execute(
        "SELECT component, exc_type, message, occurrences, trace_head FROM error_log ORDER BY id").fetchall()


def _raise_and_catch(exc):
    try:
        raise exc
    except Exception as caught:  # noqa: BLE001 — the point of the helper
        return caught


def test_log_error_writes_component_type_message_and_trace(journal, db):
    journal.log_error("save.auto_link", _raise_and_catch(KeyError("node")), "Auto-link error")
    ((component, exc_type, message, occurrences, trace),) = _rows(db)
    assert (component, exc_type, occurrences) == ("save.auto_link", "KeyError", 1)
    assert message.startswith("Auto-link error: ") and "node" in message
    assert "_raise_and_catch" in trace


def test_floods_are_rate_limited_and_rolled_up(journal, db, clocks):
    _, mono = clocks
    for _ in range(ej.RATE_LIMIT_MAX_PER_WINDOW + 25):
        journal.log_error("save.coref", ValueError("bad"), "coref resolver skipped")
    journal.log_error("save.coref", TypeError("other type"), "coref resolver skipped")
    assert len(_rows(db)) == ej.RATE_LIMIT_MAX_PER_WINDOW + 1

    mono.now += ej.RATE_LIMIT_WINDOW_SECONDS
    journal.flush()
    rollup = _rows(db)[-1]
    assert rollup[1] == "ValueError" and rollup[3] == 25 and "rate limited" in rollup[2]
    counts = journal.recent_counts()
    assert counts["by_component"] == {"save.coref": ej.RATE_LIMIT_MAX_PER_WINDOW + 25 + 1}


def test_entries_wait_while_the_caller_holds_a_transaction(journal, db):
    db.execute("CREATE TABLE caller (x INTEGER)")
    db.commit()
    db.execute("INSERT INTO caller VALUES (1)")  # implicit transaction left open by the caller
    journal.log_error("save.filter_savings", RuntimeError("locked"), "filter_savings log error")
    assert db.in_transaction  # the journal did not commit the caller's work
    db.rollback()
    assert db.execute("SELECT COUNT(*) FROM caller").fetchone()[0] == 0
    assert _rows(db) == []
    assert journal.flush() == 1
    assert len(_rows(db)) == 1


def test_other_threads_buffer_until_the_owner_flushes(journal, db):
    worker = threading.Thread(target=journal.log_error, args=("bg", RuntimeError("x"), "background"))
    worker.start()
    worker.join()
    assert _rows(db) == []
    assert journal.flush() == 1


def test_retention_purges_rows_older_than_fourteen_days(journal, db, clocks):
    wall, _ = clocks
    journal.log_error("old", RuntimeError("x"), "old failure")
    wall.now += (ej.ERROR_RETENTION_DAYS + 1) * DAY
    journal.log_error("new", RuntimeError("y"), "new failure")
    journal.purge()
    db.commit()
    assert [row[0] for row in _rows(db)] == ["new"]


def test_log_error_never_raises_without_a_table(clocks):
    wall, mono = clocks
    conn = sqlite3.connect(":memory:")
    journal = ErrorJournal(conn, clock=wall, monotonic=mono)
    journal.log_error("anything", RuntimeError("x"), "no table")  # must not raise
    assert journal.recent_counts()["total"] == 0


def test_module_level_log_error_uses_the_installed_journal(journal, db, capsys):
    ej.install(None)
    ej.log_error("unbound", RuntimeError("x"), "goes to stderr")
    assert "unbound" in capsys.readouterr().err
    ej.install(journal)
    try:
        ej.log_error("bound", RuntimeError("y"), "goes to the table")
    finally:
        ej.install(None)
    assert [row[0] for row in _rows(db)] == ["bound"]
