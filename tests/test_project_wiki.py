"""Tests for the v10 project wiki generator."""

from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import project_wiki as pw
from base_schema import apply_full_schema


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMORY_WIKI_ENABLED", "true")
    monkeypatch.setenv("MEMORY_WIKI_DIR", str(tmp_path / "wikis"))
    monkeypatch.delenv("MEMORY_WIKI_RECENT_DAYS", raising=False)
    monkeypatch.delenv("MEMORY_WIKI_AUTO_REFRESH_EVERY_N", raising=False)
    yield


@pytest.fixture
def wdb():
    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    apply_full_schema(db)
    yield db
    db.close()


def _add(db, *, kid, ktype, content, project="vito",
         tags=None, importance="medium", created="2026-04-27T10:00:00Z",
         status="active"):
    db.execute(
        """INSERT INTO knowledge (id, session_id, type, content, project,
            tags, status, importance, created_at, last_confirmed)
           VALUES (?, 's1', ?, ?, ?, ?, ?, ?, ?, ?)""",
        (kid, ktype, content, project, json.dumps(tags or []),
         status, importance, created, created),
    )
    db.commit()


# ──────────────────────────────────────────────
# Section queries
# ──────────────────────────────────────────────


def test_top_decisions_filters_by_importance_and_type(wdb):
    _add(wdb, kid=1, ktype="decision",
         content="Migrate to Postgres", importance="critical")
    _add(wdb, kid=2, ktype="decision",
         content="Random small choice", importance="low")
    _add(wdb, kid=3, ktype="solution",
         content="Fixed memcached", importance="critical")
    _add(wdb, kid=4, ktype="decision",
         content="Use NEWSEQUENTIALID", importance="high")

    rows = pw._top_decisions(wdb, "vito")
    ids = [r["id"] for r in rows]
    # Only critical/high decisions, in importance-then-recency order
    assert ids == [1, 4]


def test_active_solutions_returns_only_active_solutions(wdb):
    _add(wdb, kid=1, ktype="solution", content="Set-based carry-forward")
    _add(wdb, kid=2, ktype="solution", content="Old approach", status="superseded")
    _add(wdb, kid=3, ktype="decision", content="Decision row")

    rows = pw._active_solutions(wdb, "vito")
    assert [r["id"] for r in rows] == [1]


def test_recent_changes_window_filter(wdb):
    today = datetime.now(timezone.utc)
    _add(wdb, kid=1, ktype="fact", content="Recent",
         created=today.isoformat().replace("+00:00", "Z"))
    _add(wdb, kid=2, ktype="fact", content="Old",
         created=(today - timedelta(days=60)).isoformat().replace("+00:00", "Z"))

    rows = pw._recent_changes(wdb, "vito", days=14, limit=10)
    assert [r["id"] for r in rows] == [1]


def test_conventions_returns_all(wdb):
    _add(wdb, kid=1, ktype="convention", content="Use Conventional Commits")
    _add(wdb, kid=2, ktype="convention", content="Read-only git for Claude")
    rows = pw._conventions(wdb, "vito")
    assert {r["id"] for r in rows} == {1, 2}


# ──────────────────────────────────────────────
# Rendering
# ──────────────────────────────────────────────


def test_first_line_truncates_at_limit():
    s = "x" * 200
    assert len(pw._first_line(s, limit=110)) <= 110
    assert pw._first_line(s, limit=110).endswith("…")


def test_first_line_skips_blank_lines():
    assert pw._first_line("\n\nReal content here\n") == "Real content here"


def test_render_entry_includes_id_title_date():
    row = {
        "id": 42, "content": "Migrate to Postgres", "importance": "critical",
        "created_at": "2026-04-27T10:00:00Z", "tags": '["database","azure-sql"]',
    }
    out = pw._render_entry(row)
    assert "[#42]" in out
    assert "Migrate to Postgres" in out
    assert "2026-04-27" in out
    assert "critical" in out
    assert "tags: database, azure-sql" in out


def test_render_entry_omits_medium_importance():
    row = {"id": 1, "content": "x", "importance": "medium",
           "created_at": "2026-04-27T10:00:00Z", "tags": "[]"}
    out = pw._render_entry(row)
    assert "medium" not in out


def test_render_wiki_contains_all_sections(wdb):
    _add(wdb, kid=1, ktype="decision", content="Migrate to Postgres",
         importance="critical")
    _add(wdb, kid=2, ktype="solution", content="Set-based carry-forward")
    _add(wdb, kid=3, ktype="convention", content="Use Conventional Commits")
    _add(wdb, kid=4, ktype="fact", content="Recently observed",
         created=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))

    body = pw.render_wiki(wdb, "vito")
    assert body.startswith("# Project: vito")
    assert "## Top Decisions" in body
    assert "## Active Solutions" in body
    assert "## Conventions" in body
    assert "## Recent Changes" in body
    assert "[#1]" in body
    assert "[#2]" in body
    assert "[#3]" in body
    assert "[#4]" in body


def test_render_wiki_handles_empty_project(wdb):
    body = pw.render_wiki(wdb, "ghost-project")
    assert "## Top Decisions" in body
    assert "_No critical or high-importance decisions yet._" in body
    assert "_No active solutions recorded._" in body


# ──────────────────────────────────────────────
# generate_wiki — persistence
# ──────────────────────────────────────────────


def test_generate_wiki_writes_markdown_file(wdb, tmp_path):
    _add(wdb, kid=1, ktype="decision", content="Migration plan",
         importance="critical")
    out = tmp_path / "wikis"
    res = pw.generate_wiki(wdb, "vito", output_dir=out)
    assert res is not None
    assert Path(res.path).exists()
    content = Path(res.path).read_text()
    assert "Migration plan" in content
    assert content.startswith("# Project: vito")
    assert res.chars == len(content)


def test_generate_wiki_returns_none_for_empty_project(wdb, tmp_path):
    res = pw.generate_wiki(wdb, "no-such-project", output_dir=tmp_path / "wikis")
    assert res is None
    assert not (tmp_path / "wikis" / "no-such-project.md").exists()


def test_generate_wiki_disabled_returns_none(wdb, tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORY_WIKI_ENABLED", "false")
    _add(wdb, kid=1, ktype="decision", content="Whatever", importance="high")
    res = pw.generate_wiki(wdb, "vito", output_dir=tmp_path / "wikis")
    assert res is None


def test_generate_wiki_sanitises_filename(wdb, tmp_path):
    _add(wdb, kid=1, ktype="decision", content="x", importance="critical",
         project="weird/project name")
    res = pw.generate_wiki(wdb, "weird/project name", output_dir=tmp_path / "wikis")
    assert res is not None
    # Slashes replaced with underscores so we never escape the wikis dir.
    assert "/" not in Path(res.path).name
    assert Path(res.path).name == "weird_project_name.md"


# ──────────────────────────────────────────────
# list_projects + generate_all
# ──────────────────────────────────────────────


def test_list_projects_returns_distinct_active(wdb):
    _add(wdb, kid=1, ktype="fact", content="x", project="vito")
    _add(wdb, kid=2, ktype="fact", content="x", project="floatytv")
    _add(wdb, kid=3, ktype="fact", content="x", project="vito")
    _add(wdb, kid=4, ktype="fact", content="x", project="archived",
         status="superseded")

    out = pw.list_projects(wdb)
    assert out == ["floatytv", "vito"]


def test_generate_all_creates_one_file_per_project(wdb, tmp_path):
    _add(wdb, kid=1, ktype="decision", content="A", importance="high",
         project="vito")
    _add(wdb, kid=2, ktype="decision", content="B", importance="critical",
         project="floatytv")

    out_dir = tmp_path / "wikis"
    results = pw.generate_all(wdb, output_dir=out_dir)
    assert {r.project for r in results} == {"vito", "floatytv"}
    assert (out_dir / "vito.md").exists()
    assert (out_dir / "floatytv.md").exists()


# ──────────────────────────────────────────────
# Auto-refresh
# ──────────────────────────────────────────────


def test_maybe_auto_refresh_off_by_default(wdb, tmp_path):
    _add(wdb, kid=10, ktype="decision", content="x", importance="critical")
    res = pw.maybe_auto_refresh(
        wdb, project="vito", save_count=10, output_dir=tmp_path / "wikis"
    )
    assert res is None  # disabled by default (N=0)


def test_maybe_auto_refresh_fires_on_multiples_of_n(wdb, tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORY_WIKI_AUTO_REFRESH_EVERY_N", "5")
    _add(wdb, kid=5, ktype="decision", content="x", importance="critical")
    out_dir = tmp_path / "wikis"
    # save_count=5 → refresh
    res = pw.maybe_auto_refresh(wdb, project="vito", save_count=5, output_dir=out_dir)
    assert res is not None
    assert (out_dir / "vito.md").exists()


def test_maybe_auto_refresh_skips_non_multiples(wdb, tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORY_WIKI_AUTO_REFRESH_EVERY_N", "5")
    _add(wdb, kid=7, ktype="decision", content="x", importance="critical")
    res = pw.maybe_auto_refresh(
        wdb, project="vito", save_count=7, output_dir=tmp_path / "wikis"
    )
    assert res is None
