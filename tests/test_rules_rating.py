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


def test_auto_suspend_after_many_failed_ratings(store):
    """10 failed ratings → success_rate 0.0 → auto-suspend."""
    rid = _add_rule(store, "consistently bad rule")

    for _ in range(9):
        r = store.manage_rule("sess-rating-test", "rate", id=rid, success=False)
        assert r.get("auto_suspended") is not True

    r = store.manage_rule("sess-rating-test", "rate", id=rid, success=False)
    assert r["auto_suspended"] is True
    assert r["reason"] == "success_rate < 0.2 after 10+ ratings"

    row = store.db.execute(
        "SELECT status FROM rules WHERE id=?", (rid,)).fetchone()
    assert row["status"] == "suspended"
