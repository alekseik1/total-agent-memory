"""The hook's end-of-session capture writes a summary and a session row.

The hook had been calling this script for 664 logged session ends while the
file did not exist - `hook_run_script` fails silently and the hook logs success
either way, so nothing was ever written.
"""

import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from base_schema import apply_full_schema

SCRIPT = Path(__file__).resolve().parent.parent / "src" / "auto_session_end.py"


@pytest.fixture
def memory_dir(tmp_path):
    db = sqlite3.connect(tmp_path / "memory.db")
    apply_full_schema(db)
    db.commit()
    db.close()
    return tmp_path


def _run(memory_dir, _env=None, **kw):
    args = [sys.executable, str(SCRIPT)]
    for k, v in kw.items():
        args += [f"--{k.replace('_', '-')}", v]
    env = {"PATH": "/usr/bin:/bin", "TAM_MEMORY_DIR": str(memory_dir),
           "CLAUDE_MEMORY_DIR": str(memory_dir),
           # Off unless a test says otherwise: the probe would reach for a live
           # Ollama and make the result depend on the machine.
           "MEMORY_LLM_ENABLED": "false"}
    env.update(_env or {})
    return subprocess.run(args, capture_output=True, text=True, env=env)


def _rows(memory_dir, sql):
    db = sqlite3.connect(memory_dir / "memory.db")
    db.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in db.execute(sql).fetchall()]
    finally:
        db.close()


def test_it_writes_a_summary_and_a_session_row(memory_dir):
    r = _run(
        memory_dir,
        session_id="7733731d-4bba-4fa2-95df-7e4dbf51474f",
        project="core-monorepo",
        branch="dev",
        reason="User exited",
        user_context="fix the wearables CSV",
    )
    assert r.returncode == 0, r.stderr

    summaries = _rows(memory_dir, "SELECT * FROM session_summaries")
    assert len(summaries) == 1
    assert summaries[0]["project"] == "core-monorepo"
    assert "fix the wearables CSV" in summaries[0]["summary"]

    # The base schema seeds its own `fact-merge` row, so look for this one.
    sessions = _rows(
        memory_dir,
        "SELECT * FROM sessions WHERE id='7733731d-4bba-4fa2-95df-7e4dbf51474f'",
    )
    assert len(sessions) == 1
    assert sessions[0]["ended_at"] is not None
    assert sessions[0]["project"] == "core-monorepo"
    assert sessions[0]["branch"] == "dev"


def test_a_session_with_nothing_extracted_writes_nothing(memory_dir):
    """A summary saying only "session ended" tells the next session nothing,
    but the `sessions` row must still close - S4: an early return before
    that would leave ended_at NULL for every session where the hook
    extracted nothing at all."""
    r = _run(memory_dir, session_id="empty_sess", project="p", reason="User exited")

    assert r.returncode == 0, r.stderr
    assert _rows(memory_dir, "SELECT * FROM session_summaries") == []
    sessions = _rows(memory_dir, "SELECT * FROM sessions WHERE id='empty_sess'")
    assert len(sessions) == 1
    assert sessions[0]["ended_at"] is not None
    assert sessions[0]["project"] == "p"


def test_a_second_end_seconds_later_is_not_written_twice(memory_dir):
    """/clear and /compact fire the hook too, and can land seconds apart."""
    for _ in range(2):
        r = _run(memory_dir, session_id="dup_sess", project="p",
                 reason="User ran /clear", user_context="same work")
        assert r.returncode == 0, r.stderr

    assert len(_rows(memory_dir, "SELECT * FROM session_summaries")) == 1


def test_a_missing_database_is_reported_not_crashed(tmp_path):
    r = _run(tmp_path, session_id="s", project="p", user_context="x")

    assert r.returncode == 1
    assert "Memory DB not found" in r.stderr


def test_the_deterministic_summary_survives_a_failed_compression(memory_dir, monkeypatch):
    """`session_end` only asks the LLM when `summary` is None, so the fallback
    cannot be passed alongside it - it is written afterwards instead. Without
    that, a compression that yields nothing stores an empty summary."""
    r = _run(
        memory_dir,
        session_id="compress_sess",
        project="p",
        reason="User exited",
        user_context="the work that must survive",
        _env={"MEMORY_LLM_ENABLED": "force", "MEMORY_LLM_API_BASE": "http://127.0.0.1:9"},
    )
    assert r.returncode == 0, r.stderr

    rows = _rows(memory_dir, "SELECT summary FROM session_summaries")
    assert len(rows) == 1
    assert "the work that must survive" in rows[0]["summary"]


def test_the_hook_producer_argument_is_actually_wired(memory_dir):
    """producer="hook" is this file's entire point in the session_end call,
    and nothing else here pins it: if the kwarg were dropped, session_end
    would default to producer="tool", which - unlike "hook" - starts a
    fresh row once the existing one in the burst is already consumed. This
    proves the literal actually reaches session_end rather than the
    default."""
    r = _run(memory_dir, session_id="pin_sess", project="p", reason="User exited",
              user_context="first work")
    assert r.returncode == 0, r.stderr

    db = sqlite3.connect(memory_dir / "memory.db")
    db.execute("UPDATE session_summaries SET consumed = 1")
    db.commit()
    db.close()

    r2 = _run(memory_dir, session_id="pin_sess", project="p", reason="User ran /clear",
               user_context="second work")
    assert r2.returncode == 0, r2.stderr

    # producer="hook" always skips within the window, regardless of
    # `consumed` - if the kwarg were dropped (defaulting to "tool"), a
    # consumed existing row makes dedup start a fresh one instead, and this
    # would be 2.
    assert len(_rows(memory_dir, "SELECT * FROM session_summaries")) == 1
