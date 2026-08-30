"""Tests for src/session_continuity.py — v7.0 Phase G."""

import sqlite3

import pytest

from session_continuity import SessionContinuity
from base_schema import apply_full_schema


@pytest.fixture
def sc_db():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    apply_full_schema(conn)
    yield conn
    conn.close()


@pytest.fixture
def sc(sc_db):
    return SessionContinuity(sc_db)


# ──────────────────────────────────────────────
# session_end
# ──────────────────────────────────────────────

def test_session_end_stores_summary(sc):
    r = sc.session_end(
        "sess_1", "Worked on temporal KG",
        highlights=["Phase A done"], pitfalls=["sqlite lock"],
        next_steps=["wire up MCP"], open_questions=["How to migrate?"],
        project="claude-total-memory",
    )
    assert r["id"]
    assert r["next_steps_count"] == 1


def test_session_end_validates_input(sc):
    with pytest.raises(ValueError):
        sc.session_end("", "summary")
    with pytest.raises(ValueError):
        sc.session_end("sid", "")


def test_session_end_infers_project_from_prior_summary(sc):
    sc.session_end("s_inf", "first", project="real-proj")
    r = sc.session_end("s_inf", "second")  # project omitted → default general
    assert r["project"] == "real-proj"
    assert r["project_inferred"] is True


def test_session_end_infers_project_from_knowledge_table(sc, sc_db):
    sc_db.execute(
        "INSERT INTO knowledge (session_id, type, content, project, created_at) "
        "VALUES ('s_k', 'fact', 'x', 'proj-from-knowledge', '2026-01-01T00:00:00Z')"
    )
    r = sc.session_end("s_k", "summary")
    assert r["project"] == "proj-from-knowledge"
    assert r["project_inferred"] is True


def test_session_end_keeps_general_when_nothing_to_infer(sc):
    r = sc.session_end("s_none", "summary")
    assert r["project"] == "general"
    assert "project_inferred" not in r


# ──────────────────────────────────────────────
# session_init
# ──────────────────────────────────────────────

def test_session_init_returns_none_when_empty(sc):
    assert sc.session_init(project="x") is None


def test_session_init_returns_most_recent_unconsumed(sc):
    sc.session_end("s1", "older", project="p")
    sc.session_end("s2", "newest", project="p",
                   next_steps=["continue"])
    result = sc.session_init(project="p")
    assert result["session_id"] == "s2"
    assert result["next_steps"] == ["continue"]


def test_session_init_marks_consumed_by_default(sc):
    sc.session_end("s1", "summary", project="p")
    first = sc.session_init(project="p")
    assert first is not None
    # Second call → no unconsumed remaining
    assert sc.session_init(project="p") is None


def test_session_init_can_skip_consumption(sc):
    sc.session_end("s1", "summary", project="p")
    sc.session_init(project="p", mark_consumed=False)
    # Still available
    result2 = sc.session_init(project="p")
    assert result2 is not None


def test_session_init_filters_by_project(sc):
    sc.session_end("s1", "for p1", project="p1")
    sc.session_end("s2", "for p2", project="p2")
    r1 = sc.session_init(project="p1")
    r2 = sc.session_init(project="p2")
    assert r1["summary"] == "for p1"
    assert r2["summary"] == "for p2"


def test_session_init_can_exclude_pitfalls(sc):
    sc.session_end("s1", "sum", pitfalls=["watch out"], project="p")
    r = sc.session_init(project="p", include_pitfalls=False)
    assert r["pitfalls"] == []


def test_session_init_parses_json_fields(sc):
    sc.session_end(
        "s1", "sum",
        highlights=["h1", "h2"],
        pitfalls=["p1"],
        next_steps=["n1", "n2", "n3"],
        open_questions=["q1"],
        project="p",
    )
    r = sc.session_init(project="p")
    assert r["highlights"] == ["h1", "h2"]
    assert r["pitfalls"] == ["p1"]
    assert r["next_steps"] == ["n1", "n2", "n3"]
    assert r["open_questions"] == ["q1"]


# ──────────────────────────────────────────────
# Listing / stats
# ──────────────────────────────────────────────

def test_list_summaries(sc):
    for i in range(3):
        sc.session_end(f"s{i}", f"summary {i}", project="p")
    rows = sc.list_summaries(project="p")
    assert len(rows) == 3
    assert rows[0]["summary"] == "summary 2"  # newest first


def test_stats(sc):
    sc.session_end("s1", "a", project="p")
    sc.session_end("s2", "b", project="p")
    sc.session_init(project="p")  # consume most recent
    s = sc.stats(project="p")
    assert s["total_summaries"] == 2
    assert s["pending"] == 1
    assert s["consumed"] == 1


def test_mark_unconsumed_replays_summary(sc):
    r = sc.session_end("s1", "sum", project="p")
    sc.session_init(project="p")
    # Nothing left
    assert sc.session_init(project="p") is None
    # Reopen
    assert sc.mark_unconsumed(r["id"]) is True
    assert sc.session_init(project="p") is not None


# ──────────────────────────────────────────────
# session_end closes the session row (not just the summary)
# ──────────────────────────────────────────────

def _open_session(db, sid, *, project="general", branch=""):
    db.execute(
        "INSERT INTO sessions (id, started_at, project, branch) VALUES (?,?,?,?)",
        (sid, "2026-08-30T06:00:00Z", project, branch),
    )
    db.commit()


def test_session_end_stamps_ended_at_on_the_session_row(sc, sc_db):
    """The row in `sessions` must stop looking like a running session.

    Writing only `session_summaries` left every row open forever, so
    memory_timeline reported long-finished sessions as still running.
    """
    _open_session(sc_db, "sess_close")

    r = sc.session_end("sess_close", "did the thing", project="claude-memory-server")

    row = sc_db.execute(
        "SELECT ended_at, project FROM sessions WHERE id='sess_close'"
    ).fetchone()
    assert row["ended_at"] == r["ended_at"]
    assert row["project"] == "claude-memory-server"


def test_session_end_does_not_downgrade_a_project_the_row_already_knows(sc, sc_db):
    """A caller who omits `project` must not overwrite a real one with 'general'."""
    _open_session(sc_db, "sess_known", project="core-monorepo")

    sc.session_end("sess_known", "no project passed")

    row = sc_db.execute(
        "SELECT project FROM sessions WHERE id='sess_known'"
    ).fetchone()
    assert row["project"] == "core-monorepo"


def test_session_end_fills_branch_only_when_the_row_has_none(sc, sc_db):
    _open_session(sc_db, "sess_branch", branch="main")
    _open_session(sc_db, "sess_nobranch")

    sc.session_end("sess_branch", "s", project="p", branch="feature/x")
    sc.session_end("sess_nobranch", "s", project="p", branch="feature/x")

    kept = sc_db.execute("SELECT branch FROM sessions WHERE id='sess_branch'").fetchone()
    filled = sc_db.execute("SELECT branch FROM sessions WHERE id='sess_nobranch'").fetchone()
    assert kept["branch"] == "main"
    assert filled["branch"] == "feature/x"


def test_session_end_without_a_session_row_still_saves_the_summary(sc, sc_db):
    """Hook-driven ends can name a session the server never inserted."""
    r = sc.session_end("sess_absent", "summary survives", project="p")

    assert r["id"]
    assert sc_db.execute(
        "SELECT COUNT(*) FROM session_summaries WHERE session_id='sess_absent'"
    ).fetchone()[0] == 1
