"""Tests for src/intents.py — save_intent, list_intents, search_intents."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from intents import (
    DEDUP_WINDOW_SECONDS,
    _sha256,
    list_intents,
    save_intent,
    search_intents,
)


@pytest.fixture
def intents_db(tmp_path: Path) -> Path:
    """Fresh sqlite file with the intents table applied from the migration."""
    db_path = tmp_path / "memory.db"
    conn = sqlite3.connect(str(db_path))
    # migrations table is referenced by the SQL file — create it first
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS migrations (
            version TEXT PRIMARY KEY,
            description TEXT,
            applied_at TEXT
        );
        """
    )
    mig = Path(__file__).parent.parent / "migrations" / "013_intents.sql"
    conn.executescript(mig.read_text(encoding="utf-8"))
    conn.commit()
    conn.close()
    return db_path


def _rows(db_path: Path) -> list[sqlite3.Row]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        return list(conn.execute("SELECT * FROM intents ORDER BY id"))
    finally:
        conn.close()


def test_save_intent_inserts_row(intents_db: Path) -> None:
    rid = save_intent(intents_db, "fix auth", "sess-1", "proj-a")
    assert rid > 0

    rows = _rows(intents_db)
    assert len(rows) == 1
    r = rows[0]
    assert r["prompt"] == "fix auth"
    assert r["session_id"] == "sess-1"
    assert r["project"] == "proj-a"
    assert r["turn_index"] == 0
    assert r["prompt_hash"] == _sha256("fix auth")
    assert r["created_at"].endswith("Z")


def test_save_intent_turn_index_autoincrements(intents_db: Path) -> None:
    save_intent(intents_db, "first", "sess-1", "proj")
    save_intent(intents_db, "second", "sess-1", "proj")
    save_intent(intents_db, "third", "sess-1", "proj")
    rows = _rows(intents_db)
    assert [r["turn_index"] for r in rows] == [0, 1, 2]


def test_save_intent_dedup_same_prompt_within_5min(intents_db: Path) -> None:
    rid1 = save_intent(intents_db, "fix bug", "sess-1", "proj")
    rid2 = save_intent(intents_db, "fix bug", "sess-1", "proj")
    # Same prompt within window → no new row, existing id returned.
    assert rid1 == rid2
    assert len(_rows(intents_db)) == 1


def test_save_intent_different_sessions_no_dedup(intents_db: Path) -> None:
    rid1 = save_intent(intents_db, "same text", "sess-A", "proj")
    rid2 = save_intent(intents_db, "same text", "sess-B", "proj")
    assert rid1 != rid2
    assert len(_rows(intents_db)) == 2


def test_save_intent_empty_is_noop(intents_db: Path) -> None:
    assert save_intent(intents_db, "", "sess", "proj") == 0
    assert save_intent(intents_db, "   \n\t  ", "sess", "proj") == 0
    assert _rows(intents_db) == []


def test_save_intent_dedup_expires_after_window(intents_db: Path) -> None:
    """A duplicate older than DEDUP_WINDOW_SECONDS should NOT be collapsed."""
    # Insert a row manually with an old created_at, then re-save same prompt.
    old_ts = (
        datetime.now(timezone.utc) - timedelta(seconds=DEDUP_WINDOW_SECONDS + 60)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    conn = sqlite3.connect(str(intents_db))
    conn.execute(
        "INSERT INTO intents (session_id, project, prompt, created_at, turn_index, prompt_hash) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("sess-1", "proj", "old prompt", old_ts, 0, _sha256("old prompt")),
    )
    conn.commit()
    conn.close()

    rid = save_intent(intents_db, "old prompt", "sess-1", "proj")
    assert rid > 0
    assert len(_rows(intents_db)) == 2


def test_list_intents_by_project(intents_db: Path) -> None:
    save_intent(intents_db, "a", "s1", "alpha")
    save_intent(intents_db, "b", "s2", "beta")
    save_intent(intents_db, "c", "s3", "alpha")

    alpha = list_intents(intents_db, project="alpha")
    assert [x["prompt"] for x in alpha] == ["c", "a"]  # newest first
    assert all(x["project"] == "alpha" for x in alpha)


def test_list_intents_by_session(intents_db: Path) -> None:
    save_intent(intents_db, "p1", "s1", "proj")
    save_intent(intents_db, "p2", "s2", "proj")
    save_intent(intents_db, "p3", "s1", "proj")

    s1 = list_intents(intents_db, session_id="s1")
    assert [x["prompt"] for x in s1] == ["p3", "p1"]


def test_list_intents_limit(intents_db: Path) -> None:
    for i in range(5):
        save_intent(intents_db, f"prompt {i}", f"sess-{i}", "proj")
    out = list_intents(intents_db, project="proj", limit=3)
    assert len(out) == 3


def test_search_intents_fts(intents_db: Path) -> None:
    save_intent(intents_db, "please fix the auth flow", "s1", "proj")
    save_intent(intents_db, "write docs for billing", "s2", "proj")
    save_intent(intents_db, "refactor auth middleware", "s3", "proj")

    results = search_intents(intents_db, query="auth")
    prompts = [r["prompt"] for r in results]
    assert "please fix the auth flow" in prompts
    assert "refactor auth middleware" in prompts
    assert "write docs for billing" not in prompts


def test_search_intents_project_filter(intents_db: Path) -> None:
    save_intent(intents_db, "fix auth", "s1", "proj-a")
    save_intent(intents_db, "fix auth", "s2", "proj-b")
    res_a = search_intents(intents_db, query="fix", project="proj-a")
    assert len(res_a) == 1
    assert res_a[0]["project"] == "proj-a"


def test_search_intents_empty_query(intents_db: Path) -> None:
    save_intent(intents_db, "anything", "s", "p")
    assert search_intents(intents_db, query="") == []
    assert search_intents(intents_db, query="   ") == []
