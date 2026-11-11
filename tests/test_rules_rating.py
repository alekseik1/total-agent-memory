"""Tests for self_rules rating: success_rate uses rated events, not fire_count.

Regression: fire_count is inflated by bulk auto-fires (self_rules_context fires
every loaded rule each session), so the first rating of a long-lived rule used
to yield success_rate ~ 1/700 and instantly trip auto-suspend.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def store(monkeypatch, tmp_path):
    """Fresh Store on an isolated MEMORY_DIR (no prod data)."""
    (tmp_path / "blobs").mkdir(exist_ok=True)
    (tmp_path / "chroma").mkdir(exist_ok=True)

    import server  # lazy import so MEMORY_DIR override sticks
    monkeypatch.setattr(server, "MEMORY_DIR", tmp_path)

    s = server.Store()
    # Need an active session for rules (rules.session_id NOT NULL).
    s.db.execute(
        "INSERT INTO sessions (id, started_at, project, status) "
        "VALUES ('sess-rating-test', '2026-04-19T00:00:00Z', 'myproj', 'open')"
    )
    s.db.commit()
    yield s
    try:
        s.db.close()
    except Exception:
        pass


def _add_rule(store, content, project="myproj", scope=None,
              priority=5, tags=None):
    """Helper: insert a rule, return rule_id."""
    if scope is None:
        scope = f"project:{project}" if project != "general" else "global"
    now = "2026-04-19T00:00:00Z"
    cur = store.db.execute(
        """INSERT INTO rules (session_id, content, context, category, scope,
                              priority, project, tags, status, created_at, updated_at)
           VALUES ('sess-rating-test', ?, '', 'manual', ?, ?, ?, ?, 'active', ?, ?)""",
        (content, scope, priority, project, json.dumps(tags or []), now, now),
    )
    store.db.commit()
    return cur.lastrowid


def _rule_row(store, rid):
    return store.db.execute(
        "SELECT * FROM rules WHERE id=?", (rid,)).fetchone()


def test_first_success_rating_does_not_auto_suspend_high_fire_rule(store):
    """Rule with 700 auto-fires rated success once stays active, rate == 1.0."""
    rid = _add_rule(store, "long-lived good rule")
    store.db.execute("UPDATE rules SET fire_count=700 WHERE id=?", (rid,))
    store.db.commit()

    r = store.manage_rule("sess-rating-test", "rate", id=rid, success=True)
    assert r.get("auto_suspended") is not True
    assert r["success_rate"] == 1.0

    row = store.db.execute(
        "SELECT status, success_rate FROM rules WHERE id=?", (rid,)).fetchone()
    assert row["status"] == "active"
    assert row["success_rate"] == 1.0


def test_update_replaces_content_and_keeps_counters(store):
    """update edits content + updated_at only; rating history survives."""
    rid = _add_rule(store, "old rule text")
    store.manage_rule("sess-rating-test", "rate", id=rid, success=True)
    store.db.execute(
        "UPDATE rules SET updated_at='2020-01-01T00:00:00Z' WHERE id=?", (rid,))
    store.db.commit()
    before = _rule_row(store, rid)

    r = store.manage_rule("sess-rating-test", "update", id=rid,
                          content="new rule text")
    assert r == {"updated": True, "id": rid}

    after = _rule_row(store, rid)
    assert after["content"] == "new rule text"
    assert after["updated_at"] != before["updated_at"]
    for col in ("fire_count", "success_count", "fail_count", "shown_count",
                "success_rate", "status", "priority", "scope", "project"):
        assert after[col] == before[col]


def test_update_unknown_id_returns_not_found(store):
    r = store.manage_rule("sess-rating-test", "update", id=99999,
                          content="new text")
    assert r == {"error": "Rule 99999 not found", "id": 99999}


def test_update_requires_id_and_non_empty_content(store):
    rid = _add_rule(store, "keep me")
    for kwargs in ({"id": rid}, {"content": "x"},
                   {"id": rid, "content": ""},
                   {"id": rid, "content": "   "}):
        r = store.manage_rule("sess-rating-test", "update", **kwargs)
        assert "error" in r and "Unknown action" not in r["error"]
    assert _rule_row(store, rid)["content"] == "keep me"
