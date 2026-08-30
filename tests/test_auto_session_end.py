"""The hook's end-of-session capture writes a summary and a session row.

The hook had been calling this script for 664 logged session ends while the
file did not exist — `hook_run_script` fails silently and the hook logs success
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


def _run(memory_dir, **kw):
    args = [sys.executable, str(SCRIPT)]
    for k, v in kw.items():
        args += [f"--{k.replace('_', '-')}", v]
    return subprocess.run(
        args,
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "TAM_MEMORY_DIR": str(memory_dir),
             "CLAUDE_MEMORY_DIR": str(memory_dir)},
    )


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
    """A summary saying only "session ended" tells the next session nothing."""
    r = _run(memory_dir, session_id="empty_sess", project="p", reason="User exited")

    assert r.returncode == 0, r.stderr
    assert _rows(memory_dir, "SELECT * FROM session_summaries") == []


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
