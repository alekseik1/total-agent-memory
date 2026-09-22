"""Tests for src/session_continuity.py — v7.0 Phase G."""

import sqlite3

import pytest

import session_continuity
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


def test_session_end_creates_the_row_for_a_session_it_never_opened(sc, sc_db):
    """Ends arrive under identities the MCP process does not own.

    A hook names the Claude Code session, a subagent names itself, a caller
    invents an id. Each is a real session; without a row it has nowhere to
    belong, which is how 231 of 256 stored summaries ended up orphaned.
    """
    r = sc.session_end(
        "7733731d-4bba-4fa2-95df-7e4dbf51474f",
        "ended by a hook",
        project="core-monorepo",
        branch="dev",
    )

    row = sc_db.execute(
        "SELECT * FROM sessions WHERE id='7733731d-4bba-4fa2-95df-7e4dbf51474f'"
    ).fetchone()
    assert row is not None
    assert row["ended_at"] == r["ended_at"]
    assert row["project"] == "core-monorepo"
    assert row["branch"] == "dev"


def test_a_created_row_starts_no_earlier_than_the_end_it_was_created_for(sc, sc_db):
    """No invented start time: the end is the only moment known for certain."""
    r = sc.session_end("unknown_sess", "s", project="p")

    row = sc_db.execute(
        "SELECT started_at, ended_at FROM sessions WHERE id='unknown_sess'"
    ).fetchone()
    assert row["started_at"] == r["ended_at"]


def test_an_explicit_started_at_is_used_for_the_created_row(sc, sc_db):
    sc.session_end("with_start", "s", project="p", started_at="2026-08-30T06:00:00Z")

    row = sc_db.execute(
        "SELECT started_at FROM sessions WHERE id='with_start'"
    ).fetchone()
    assert row["started_at"] == "2026-08-30T06:00:00Z"


def test_an_existing_row_is_not_replaced_by_the_insert(sc, sc_db):
    """INSERT OR IGNORE must leave a real session's own start time alone."""
    _open_session(sc_db, "sess_real", project="core-monorepo")
    sc_db.execute(
        "UPDATE sessions SET started_at='2026-08-30T06:00:00Z' WHERE id='sess_real'"
    )
    sc_db.commit()

    sc.session_end("sess_real", "s", project="core-monorepo")

    row = sc_db.execute(
        "SELECT started_at FROM sessions WHERE id='sess_real'"
    ).fetchone()
    assert row["started_at"] == "2026-08-30T06:00:00Z"


# ──────────────────────────────────────────────
# _dedup_action's docstring has the full policy. All tests here control
# the clock via session_continuity._now (real-time resolution is 1s, too
# coarse to prove a 300s window deterministically).
# ──────────────────────────────────────────────

def _at(monkeypatch, ts: str) -> None:
    monkeypatch.setattr(session_continuity, "_now", lambda: ts)


def _summaries(sc_db, session_id: str) -> list[str]:
    rows = sc_db.execute(
        "SELECT summary FROM session_summaries WHERE session_id = ? "
        "ORDER BY rowid", (session_id,),
    ).fetchall()
    return [dict(r)["summary"] for r in rows]


def test_a_later_close_outside_the_window_still_moves_ended_at_forward(sc, sc_db, monkeypatch):
    """The hook fires legitimately on /clear and /compact as well as exit,
    so a session can close more than once - the second close, outside the
    dedup window, must still move `ended_at` forward and add its own row."""
    _open_session(sc_db, "sess_twice")

    _at(monkeypatch, "2026-09-21T10:00:00Z")
    r1 = sc.session_end("sess_twice", "first close", project="real-proj")
    _at(monkeypatch, "2026-09-21T10:10:00Z")  # +600s, outside the 300s window
    r2 = sc.session_end("sess_twice", "second close", project="real-proj")

    row = sc_db.execute("SELECT ended_at FROM sessions WHERE id='sess_twice'").fetchone()
    assert row["ended_at"] == r2["ended_at"]
    assert row["ended_at"] != r1["ended_at"]
    assert "deduped" not in r2
    assert _summaries(sc_db, "sess_twice") == ["first close", "second close"]


def test_a_session_never_closed_before_is_never_deduped(sc, sc_db):
    """A session_summaries row can predate any close for this id (e.g. an
    id collision, or a row written by some other path) - _dedup_action must
    not treat that as a reason to skip or overwrite on the real first close,
    or a later real summary could lose to whatever wrote the earlier row."""
    _open_session(sc_db, "sess_never_closed")
    sc_db.execute(
        "INSERT INTO session_summaries (id, session_id, project, summary, "
        "ended_at, consumed, created_at) VALUES "
        "('s1', 'sess_never_closed', 'p', 'earlier', ?, 0, ?)",
        (session_continuity._now(), session_continuity._now()),
    )
    sc_db.commit()

    r = sc.session_end("sess_never_closed", "the real one", project="p")

    assert "deduped" not in r
    assert _summaries(sc_db, "sess_never_closed") == ["earlier", "the real one"]


def test_hook_then_tool_the_real_summary_survives(sc, sc_db, monkeypatch):
    """The gap this round closes: a hook floor write lands first and closes
    the session itself, then a real summary arrives within the window - the
    floor must not own the slot (see _dedup_action's docstring)."""
    _open_session(sc_db, "sess_ht")

    _at(monkeypatch, "2026-09-21T10:00:00Z")
    r1 = sc.session_end("sess_ht", "floor text", project="p", producer="hook")
    _at(monkeypatch, "2026-09-21T10:00:30Z")  # +30s, inside the window
    r2 = sc.session_end("sess_ht", "the real summary", project="p", producer="tool")

    assert "deduped" not in r1
    assert "deduped" not in r2
    assert r2["id"] == r1["id"]
    assert _summaries(sc_db, "sess_ht") == ["the real summary"]


def test_tool_then_hook_the_real_summary_survives(sc, sc_db, monkeypatch):
    """A hook fire chasing a real summary (e.g. /clear right after the agent
    already called session_end itself) must not overwrite it with a floor."""
    _open_session(sc_db, "sess_th")

    _at(monkeypatch, "2026-09-21T10:00:00Z")
    r1 = sc.session_end("sess_th", "the real summary", project="p", producer="tool")
    _at(monkeypatch, "2026-09-21T10:00:30Z")
    r2 = sc.session_end("sess_th", "floor text", project="p", producer="hook")

    assert "deduped" not in r1
    assert r2.get("deduped") is True
    assert _summaries(sc_db, "sess_th") == ["the real summary"]


def test_hook_then_hook_the_first_floor_survives_but_still_closes(sc, sc_db, monkeypatch):
    """/clear and /compact can both fire the hook seconds apart - the second
    floor write must not replace the first (nothing to gain, and overwriting
    would reset `consumed` on a row session_init may already have
    delivered), but the session row must still close on both calls."""
    _open_session(sc_db, "sess_hh")

    _at(monkeypatch, "2026-09-21T10:00:00Z")
    r1 = sc.session_end("sess_hh", "first floor", project="p", producer="hook")
    _at(monkeypatch, "2026-09-21T10:00:30Z")
    r2 = sc.session_end("sess_hh", "second floor", project="p", producer="hook")

    assert "deduped" not in r1
    assert r2.get("deduped") is True
    # S1: a skip must not touch the markdown projection either, so r2 lacks
    # the active_context_* keys r1's real write produced.
    assert set(r2.keys()) - {"deduped"} == set(r1.keys()) - {
        "active_context_path", "active_context_error"
    }
    assert _summaries(sc_db, "sess_hh") == ["first floor"]
    row = sc_db.execute("SELECT ended_at FROM sessions WHERE id='sess_hh'").fetchone()
    assert row["ended_at"] == r2["ended_at"]  # the close still ran, timestamp moved
    assert row["ended_at"] != r1["ended_at"]


def test_tool_then_tool_the_second_summary_overwrites_the_first(sc, sc_db, monkeypatch):
    """Two real session_end calls seconds apart collapse to one row - the
    row's own producer is never stored, so a tool call can't tell an
    earlier real summary from a floor and always overwrites (same tradeoff
    the old skip-based dedup already made for this pair, just now keeping
    the newer content instead of the older)."""
    _open_session(sc_db, "sess_tt")

    _at(monkeypatch, "2026-09-21T10:00:00Z")
    r1 = sc.session_end("sess_tt", "first", project="p")
    _at(monkeypatch, "2026-09-21T10:00:30Z")
    r2 = sc.session_end("sess_tt", "second", project="p")

    assert "deduped" not in r1
    assert "deduped" not in r2
    assert r2["id"] == r1["id"]
    assert _summaries(sc_db, "sess_tt") == ["second"]
    row = sc_db.execute("SELECT ended_at FROM sessions WHERE id='sess_tt'").fetchone()
    assert row["ended_at"] == r2["ended_at"]
    assert row["ended_at"] != r1["ended_at"]


def test_session_end_for_a_different_project_does_not_collapse_into_the_first(sc, sc_db, monkeypatch):
    """Two ends under one session_id for two different projects inside the
    window are two different events, not a retry of the same close - the
    second must get its own row, not repurpose the first project's."""
    _open_session(sc_db, "sess_multiproj")

    _at(monkeypatch, "2026-09-21T10:00:00Z")
    r1 = sc.session_end("sess_multiproj", "proj a summary", project="proj-a")
    _at(monkeypatch, "2026-09-21T10:00:30Z")  # inside the window
    r2 = sc.session_end("sess_multiproj", "proj b summary", project="proj-b")

    assert "deduped" not in r1
    assert "deduped" not in r2
    assert r2["id"] != r1["id"]
    rows = sc_db.execute(
        "SELECT project, summary FROM session_summaries WHERE session_id = ? "
        "ORDER BY rowid", ("sess_multiproj",),
    ).fetchall()
    assert [dict(r) for r in rows] == [
        {"project": "proj-a", "summary": "proj a summary"},
        {"project": "proj-b", "summary": "proj b summary"},
    ]


def test_tool_overwrite_skips_a_row_already_consumed(sc, sc_db, monkeypatch):
    """A row session_init already delivered ends its burst - a further tool
    call inside the window must not resurrect it (reset consumed=0) and risk
    the same summary being served to a second session; it starts a new row
    instead."""
    _open_session(sc_db, "sess_consumed")

    _at(monkeypatch, "2026-09-21T10:00:00Z")
    r1 = sc.session_end("sess_consumed", "first", project="p")
    delivered = sc.session_init(project="p")  # marks r1's row consumed
    assert delivered["id"] == r1["id"]

    _at(monkeypatch, "2026-09-21T10:00:30Z")  # inside the window
    r2 = sc.session_end("sess_consumed", "second", project="p")

    assert "deduped" not in r2
    assert r2["id"] != r1["id"]
    assert _summaries(sc_db, "sess_consumed") == ["first", "second"]
    row1 = sc_db.execute(
        "SELECT consumed FROM session_summaries WHERE id = ?", (r1["id"],)
    ).fetchone()
    assert row1["consumed"] == 1  # never reset by the second call


def test_validation_runs_before_dedup_is_even_checked(sc, sc_db, monkeypatch):
    """The dedup return must not jump the summary-required check - a
    skipped-looking call with a bad argument still raises."""
    _open_session(sc_db, "sess_bad_arg")
    _at(monkeypatch, "2026-09-21T10:00:00Z")
    sc.session_end("sess_bad_arg", "first", project="p", producer="hook")
    _at(monkeypatch, "2026-09-21T10:00:05Z")  # inside the window: would skip

    with pytest.raises(ValueError):
        sc.session_end("sess_bad_arg", "", project="p", producer="hook")


# ──────────────────────────────────────────────
# Scope-audit fix round: S1 (skip must not repaint the live markdown doc),
# S2 (skip must return an addressable id), S3 (a naive stored timestamp
# must not crash dedup), S6 (overwrite must not null a field the second
# call in the burst simply omitted).
# ──────────────────────────────────────────────

def test_a_skipped_hook_call_does_not_repaint_the_live_document(sc, sc_db, monkeypatch):
    """A deduped hook floor must not overwrite the markdown projection a
    real summary already wrote there for this burst."""
    from active_context import active_context_path
    from config import get_active_context_vault

    _open_session(sc_db, "sess_md")
    _at(monkeypatch, "2026-09-21T10:00:00Z")
    sc.session_end("sess_md", "the real summary", project="p", producer="tool")

    doc_path = active_context_path("p", vault_root=get_active_context_vault())
    before = doc_path.read_text()

    _at(monkeypatch, "2026-09-21T10:00:30Z")  # inside the window
    r = sc.session_end("sess_md", "hook floor text", project="p", producer="hook")

    assert r.get("deduped") is True
    assert doc_path.read_text() == before


def test_a_skipped_call_returns_the_id_of_the_row_left_alone(sc, sc_db, monkeypatch):
    """A skip must still name the surviving row - mark_unconsumed takes an
    id, and the caller otherwise has no way to address what's actually
    there."""
    _open_session(sc_db, "sess_skip_id")
    _at(monkeypatch, "2026-09-21T10:00:00Z")
    r1 = sc.session_end("sess_skip_id", "real summary", project="p", producer="tool")
    _at(monkeypatch, "2026-09-21T10:00:30Z")
    r2 = sc.session_end("sess_skip_id", "hook floor", project="p", producer="hook")

    assert r2.get("deduped") is True
    assert r2["id"] == r1["id"]
    assert r2["id"] is not None
    sc.session_init(project="p")  # marks r1's row consumed
    assert sc.mark_unconsumed(r2["id"]) is True
    assert sc.session_init(project="p") is not None


def test_a_naive_stored_timestamp_does_not_crash_dedup(sc, sc_db, monkeypatch):
    """ended_at without a timezone offset parses without raising, so the
    TypeError from subtracting an aware `now` from it must be caught too,
    not just the ValueError a malformed string would raise - dedup falls
    back to "insert" rather than crashing the whole session_end call."""
    _open_session(sc_db, "sess_naive")
    _at(monkeypatch, "2026-09-21T10:00:00Z")
    sc.session_end("sess_naive", "earlier", project="p")
    # Simulate a legacy row stored without a timezone offset.
    sc_db.execute(
        "UPDATE session_summaries SET ended_at = '2026-09-21T10:00:00' "
        "WHERE session_id = 'sess_naive'"
    )
    sc_db.commit()

    _at(monkeypatch, "2026-09-21T10:00:30Z")  # inside the window
    r = sc.session_end("sess_naive", "the real one", project="p")

    assert "deduped" not in r
    assert _summaries(sc_db, "sess_naive") == ["earlier", "the real one"]


def test_overwrite_does_not_null_fields_a_second_call_omits(sc, sc_db, monkeypatch):
    """A second tool call in the same burst that omits branch/started_at
    must not wipe what the first call already stored for this row."""
    _open_session(sc_db, "sess_overwrite_fields")
    _at(monkeypatch, "2026-09-21T10:00:00Z")
    r1 = sc.session_end(
        "sess_overwrite_fields", "first", project="p",
        branch="feature/x", started_at="2026-09-20T09:00:00Z",
    )
    _at(monkeypatch, "2026-09-21T10:00:30Z")  # inside the window
    r2 = sc.session_end("sess_overwrite_fields", "second", project="p")

    assert r2["id"] == r1["id"]
    row = sc_db.execute(
        "SELECT branch, started_at, summary FROM session_summaries WHERE id = ?",
        (r1["id"],),
    ).fetchone()
    assert row["branch"] == "feature/x"
    assert row["started_at"] == "2026-09-20T09:00:00Z"
    assert row["summary"] == "second"  # content itself still overwrites
