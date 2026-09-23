"""migrations/*_backfill_orphan_session_rows.sql gives a session row to
summaries that never had one.

*_backfill_session_rows.sql could only repair rows that existed. These are
the sessions that ended under an identity the MCP server does not issue - a
hook's Claude Code uuid, an id a caller invented - and so had no row at all.
"""

import sqlite3
from pathlib import Path

import pytest

from base_schema import apply_full_schema

ROOT = Path(__file__).resolve().parent.parent
MIGRATION = next((ROOT / "migrations").glob("*_backfill_orphan_session_rows.sql"))


@pytest.fixture
def db():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    apply_full_schema(conn)
    yield conn
    conn.close()


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


def test_an_orphan_summary_gets_a_closed_row(db):
    _summary(db, "7733731d-4bba-4fa2-95df-7e4dbf51474f")
    db.commit()

    _apply(db)

    row = db.execute(
        "SELECT * FROM sessions WHERE id='7733731d-4bba-4fa2-95df-7e4dbf51474f'"
    ).fetchone()
    assert row["ended_at"] == "2026-08-01T10:00:00Z"
    assert row["project"] == "core-monorepo"
    assert row["branch"] == "dev"


def test_the_row_spans_first_to_last_summary(db):
    """started_at is the earliest end seen, not a guess at when it began."""
    _summary(db, "s_multi", ended_at="2026-08-01T09:00:00Z", project="general")
    _summary(db, "s_multi", ended_at="2026-08-01T18:00:00Z", project="app-hub")
    db.commit()

    _apply(db)

    row = db.execute("SELECT * FROM sessions WHERE id='s_multi'").fetchone()
    assert row["started_at"] == "2026-08-01T09:00:00Z"
    assert row["ended_at"] == "2026-08-01T18:00:00Z"
    assert row["project"] == "app-hub"          # newest summary that names one


def test_an_existing_session_is_left_alone(db):
    db.execute(
        "INSERT INTO sessions (id, started_at, project, branch) VALUES "
        "('s_known', '2026-07-01T00:00:00Z', 'paperless-gost', 'main')"
    )
    _summary(db, "s_known", project="core-monorepo", branch="dev")
    db.commit()

    _apply(db)

    rows = db.execute("SELECT * FROM sessions WHERE id='s_known'").fetchall()
    assert len(rows) == 1
    assert rows[0]["started_at"] == "2026-07-01T00:00:00Z"
    assert rows[0]["project"] == "paperless-gost"


def test_a_summary_naming_only_general_still_gets_a_row(db):
    _summary(db, "s_plain", project="general", branch="")
    db.commit()

    _apply(db)

    row = db.execute("SELECT * FROM sessions WHERE id='s_plain'").fetchone()
    assert row["ended_at"] == "2026-08-01T10:00:00Z"
    assert row["project"] == "general"
    assert row["branch"] == ""


def test_running_it_twice_changes_nothing(db):
    _summary(db, "s_idem")
    db.commit()

    _apply(db)
    first = [dict(r) for r in db.execute("SELECT * FROM sessions").fetchall()]
    _apply(db)
    second = [dict(r) for r in db.execute("SELECT * FROM sessions").fetchall()]

    assert first == second


def test_no_summary_is_left_without_a_session_row(db):
    for sid in ("uuid-shaped-0000", "mcp_20260830_hand_written", "session_01ABC"):
        _summary(db, sid)
    db.commit()

    _apply(db)

    orphans = db.execute(
        "SELECT COUNT(*) FROM session_summaries ss "
        "WHERE NOT EXISTS (SELECT 1 FROM sessions s WHERE s.id = ss.session_id)"
    ).fetchone()[0]
    assert orphans == 0
