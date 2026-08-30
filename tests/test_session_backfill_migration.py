"""Migration 030 repairs session rows the old session_end never closed.

Only sessions that actually produced a summary are closed: one without a
summary was never ended, and stamping an end time on it would be an invention.
"""

import sqlite3
from pathlib import Path

import pytest

from base_schema import apply_full_schema

MIGRATION = (
    Path(__file__).resolve().parent.parent / "migrations" / "030_backfill_session_rows.sql"
)


@pytest.fixture
def db():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    apply_full_schema(conn)
    yield conn
    conn.close()


def _session(db, sid, *, project="general", branch=""):
    db.execute(
        "INSERT INTO sessions (id, started_at, project, branch) VALUES (?,?,?,?)",
        (sid, "2026-08-01T00:00:00Z", project, branch),
    )


def _summary(db, sid, *, project="core-monorepo", branch="dev", ended_at="2026-08-01T10:00:00Z"):
    db.execute(
        """INSERT INTO session_summaries
           (id, session_id, project, branch, summary, highlights, pitfalls,
            next_steps, open_questions, context_blob, started_at, ended_at,
            consumed, created_at)
           VALUES (?, ?, ?, ?, 's', '[]', '[]', '[]', '[]', NULL, NULL, ?, 0, ?)""",
        (f"sum_{sid}_{ended_at}", sid, project, branch, ended_at, ended_at),
    )


def _apply(db):
    db.executescript(MIGRATION.read_text())
    db.commit()


def test_a_session_with_a_summary_is_closed_and_named(db):
    _session(db, "s1")
    _summary(db, "s1")
    db.commit()

    _apply(db)

    row = db.execute("SELECT * FROM sessions WHERE id='s1'").fetchone()
    assert row["ended_at"] == "2026-08-01T10:00:00Z"
    assert row["project"] == "core-monorepo"
    assert row["branch"] == "dev"


def test_a_session_without_a_summary_stays_open(db):
    """It was never ended — the migration must not invent an end time."""
    _session(db, "s2")
    db.commit()

    _apply(db)

    row = db.execute("SELECT ended_at, project FROM sessions WHERE id='s2'").fetchone()
    assert row["ended_at"] is None
    assert row["project"] == "general"


def test_an_already_closed_session_is_left_alone(db):
    _session(db, "s3", project="paperless-gost", branch="main")
    db.execute("UPDATE sessions SET ended_at='2026-07-01T00:00:00Z' WHERE id='s3'")
    _summary(db, "s3", project="core-monorepo", branch="dev")
    db.commit()

    _apply(db)

    row = db.execute("SELECT * FROM sessions WHERE id='s3'").fetchone()
    assert row["ended_at"] == "2026-07-01T00:00:00Z"
    assert row["project"] == "paperless-gost"
    assert row["branch"] == "main"


def test_the_newest_summary_wins_when_a_session_has_several(db):
    _session(db, "s4")
    _summary(db, "s4", project="older-project", ended_at="2026-08-01T09:00:00Z")
    _summary(db, "s4", project="newer-project", ended_at="2026-08-01T18:00:00Z")
    db.commit()

    _apply(db)

    row = db.execute("SELECT ended_at, project FROM sessions WHERE id='s4'").fetchone()
    assert row["ended_at"] == "2026-08-01T18:00:00Z"
    assert row["project"] == "newer-project"


def test_a_summary_that_only_says_general_does_not_overwrite_the_row(db):
    """'general' in the summary is the same placeholder — not an improvement."""
    _session(db, "s5")
    _summary(db, "s5", project="general", branch="")
    db.commit()

    _apply(db)

    row = db.execute("SELECT ended_at, project FROM sessions WHERE id='s5'").fetchone()
    assert row["ended_at"] == "2026-08-01T10:00:00Z"   # still closed
    assert row["project"] == "general"                  # nothing better to say


def test_running_it_twice_changes_nothing(db):
    _session(db, "s6")
    _summary(db, "s6")
    db.commit()

    _apply(db)
    first = dict(db.execute("SELECT * FROM sessions WHERE id='s6'").fetchone())
    _apply(db)
    second = dict(db.execute("SELECT * FROM sessions WHERE id='s6'").fetchone())

    assert first == second
