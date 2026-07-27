"""Tests for self_rules_context result truncation (get_rules_for_context).

Regression: `ORDER BY priority DESC, success_rate DESC LIMIT 20` silently
dropped active rules past position 20, and buried unrated (new) rules below
rules with recorded failures because success_rate=0.0 for both. Phase
filtering also ran after the SQL LIMIT, so a phase-tagged rule ranked below
the cap could never load.

Follow-up (review): MEMORY_RULES_LIMIT parsing must never crash the server
at import, negative `limit` must be rejected like a bad `phase`, the
fire_count update must be batched, and the response must expose
`total_matched` so a caller can tell "all matching rules" from "top N".
"""

from __future__ import annotations

import asyncio
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
        "VALUES ('sess-limit-test', '2026-04-19T00:00:00Z', 'myproj', 'open')"
    )
    s.db.commit()
    yield s
    try:
        s.db.close()
    except Exception:
        pass


@pytest.fixture
def live_store(monkeypatch, tmp_path):
    """Fresh Store wired into the module-level globals `_do()` reads,
    for handler-level (dispatch) tests."""
    (tmp_path / "blobs").mkdir(exist_ok=True)
    (tmp_path / "chroma").mkdir(exist_ok=True)

    import server
    monkeypatch.setattr(server, "MEMORY_DIR", tmp_path)
    s = server.Store()
    server.store = s
    server.recall = server.Recall(s)
    server.SID = "sess-limit-handler"
    server.BRANCH = ""
    s.db.execute(
        "INSERT INTO sessions (id, started_at, project, status) VALUES (?, ?, ?, ?)",
        (server.SID, "2026-04-19T00:00:00Z", "myproj", "open"),
    )
    s.db.commit()
    yield s, server
    try:
        s.db.close()
    except Exception:
        pass


def _add_rule(store, content, project="myproj", scope=None,
              priority=5, tags=None, created_at="2026-04-19T00:00:00Z"):
    """Helper: insert a rule, return rule_id."""
    if scope is None:
        scope = f"project:{project}" if project != "general" else "global"
    cur = store.db.execute(
        """INSERT INTO rules (session_id, content, context, category, scope,
                              priority, project, tags, status, created_at, updated_at)
           VALUES ('sess-limit-test', ?, '', 'manual', ?, ?, ?, ?, 'active', ?, ?)""",
        (content, scope, priority, project, json.dumps(tags or []), created_at, created_at),
    )
    store.db.commit()
    return cur.lastrowid


def test_no_cap_returns_all_25_active_rules(store):
    """25 active rules seeded → all 25 returned (no silent cap)."""
    for i in range(25):
        _add_rule(store, f"rule {i}")

    r = store.get_rules_for_context(project="myproj")
    assert r["rules_count"] == 25
    assert len(r["rules"]) == 25


def test_explicit_limit_parameter_caps_result(store):
    """limit=10 parameter → exactly 10 returned."""
    for i in range(25):
        _add_rule(store, f"rule {i}")

    r = store.get_rules_for_context(project="myproj", limit=10)
    assert r["rules_count"] == 10
    assert len(r["rules"]) == 10


def test_read_rules_limit_parses_env_var():
    """_read_rules_limit is the real MEMORY_RULES_LIMIT parsing contract."""
    import server
    assert server._read_rules_limit({"MEMORY_RULES_LIMIT": "7"}) == 7


def test_read_rules_limit_unset_returns_none():
    import server
    assert server._read_rules_limit({}) is None


def test_read_rules_limit_empty_string_returns_none():
    import server
    assert server._read_rules_limit({"MEMORY_RULES_LIMIT": ""}) is None


def test_read_rules_limit_zero_returns_zero():
    """'0' means empty result, not 'no cap' — must not collapse to None."""
    import server
    assert server._read_rules_limit({"MEMORY_RULES_LIMIT": "0"}) == 0


def test_read_rules_limit_invalid_value_returns_none_without_raising():
    """MEMORY_RULES_LIMIT=twenty must not crash the server at import."""
    import server
    assert server._read_rules_limit({"MEMORY_RULES_LIMIT": "twenty"}) is None


def test_module_attribute_fallback_used_when_no_explicit_limit(store, monkeypatch):
    """get_rules_for_context falls back to the module-level RULES_CONTEXT_LIMIT
    attribute (set at import from MEMORY_RULES_LIMIT via _read_rules_limit)
    when no explicit `limit` argument is passed."""
    import server
    monkeypatch.setattr(server, "RULES_CONTEXT_LIMIT", 7)

    for i in range(25):
        _add_rule(store, f"rule {i}")

    r = store.get_rules_for_context(project="myproj")
    assert r["rules_count"] == 7
    assert len(r["rules"]) == 7


def test_explicit_limit_overrides_env_derived_default(store, monkeypatch):
    """An explicit `limit` argument wins over the RULES_CONTEXT_LIMIT default,
    even when both are set to different values."""
    import server
    monkeypatch.setattr(server, "RULES_CONTEXT_LIMIT", 20)

    for i in range(25):
        _add_rule(store, f"rule {i}")

    r = store.get_rules_for_context(project="myproj", limit=5)
    assert r["rules_count"] == 5
    assert len(r["rules"]) == 5


def test_unrated_rule_ranks_above_failed_rule_at_same_priority(store):
    """An unrated rule (0 successes, 0 fails) ranks ABOVE a rule with recorded
    failures at the same priority — unrated is neutral (0.5), not worst."""
    failed_id = _add_rule(store, "failed rule", priority=5)
    store.db.execute(
        "UPDATE rules SET fail_count=5, success_rate=0.0 WHERE id=?", (failed_id,))
    unrated_id = _add_rule(store, "unrated rule", priority=5)
    store.db.commit()

    r = store.get_rules_for_context(project="myproj")
    ids = [x["id"] for x in r["rules"]]
    assert ids.index(unrated_id) < ids.index(failed_id)


def test_created_at_desc_tiebreak_newer_rule_first(store):
    """Two rules, same priority, both unrated → newer created_at ranks first."""
    older_id = _add_rule(store, "older rule", priority=5, created_at="2026-04-19T00:00:00Z")
    newer_id = _add_rule(store, "newer rule", priority=5, created_at="2026-04-20T00:00:00Z")

    r = store.get_rules_for_context(project="myproj")
    ids = [x["id"] for x in r["rules"]]
    assert ids.index(newer_id) < ids.index(older_id)


def test_phase_filter_applied_before_truncation(store):
    """Seed 19 core rules (priority 10) + 5 phase:plan rules (priority 9) +
    1 phase:build rule (priority 8) = 25 total. Pre-filter ranking puts the
    build rule outside the top 20 (19 cores + top plan rule fill the cap).
    Filtering phase='plan' rules out BEFORE truncation frees a slot for the
    build rule; the old bug truncated first, so the build rule was dropped
    before the phase filter ever saw it."""
    core_ids = [_add_rule(store, f"core {i}", priority=10) for i in range(19)]
    plan_ids = [_add_rule(store, f"plan {i}", priority=9, tags=["phase:plan"]) for i in range(5)]
    build_rule_id = _add_rule(store, "build-only rule", priority=8, tags=["phase:build"])

    r = store.get_rules_for_context(project="myproj", phase="build", limit=20)
    ids = [x["id"] for x in r["rules"]]
    # Identity check, not a count — 20 is also the old hardcoded cap, so a
    # count-only assertion could pass for the wrong reason.
    assert build_rule_id in ids
    assert not (set(plan_ids) & set(ids))
    assert set(ids) == set(core_ids) | {build_rule_id}


def test_explicit_limit_zero_returns_no_rules(store):
    """limit=0 → zero rules returned, and no rule's fire_count is incremented."""
    ids = [_add_rule(store, f"rule {i}") for i in range(5)]

    r = store.get_rules_for_context(project="myproj", limit=0)
    assert r["rules_count"] == 0
    assert r["rules"] == []

    for rid in ids:
        row = store.db.execute(
            "SELECT fire_count FROM rules WHERE id=?", (rid,)).fetchone()
        assert row["fire_count"] == 0


def test_negative_limit_returns_error_and_does_not_bump_fire_count(store):
    """A negative `limit` returns the same {"error": ...} shape as a bad
    `phase`, and neither truncates nor bumps fire_count."""
    ids = [_add_rule(store, f"rule {i}") for i in range(5)]

    r = store.get_rules_for_context(project="myproj", limit=-1)
    assert "error" in r
    assert "rules" not in r

    for rid in ids:
        row = store.db.execute(
            "SELECT fire_count FROM rules WHERE id=?", (rid,)).fetchone()
        assert row["fire_count"] == 0


def test_total_matched_reflects_pre_truncation_count(store):
    """`total_matched` is the count after the phase filter, before `limit`
    truncation; `rules_count` is what was actually returned."""
    for i in range(25):
        _add_rule(store, f"rule {i}")

    r = store.get_rules_for_context(project="myproj", limit=10)
    assert r["total_matched"] == 25
    assert r["rules_count"] == 10


def test_fire_count_incremented_only_for_returned_rules(store):
    """A rule cut by `limit` keeps fire_count == 0."""
    ids = [_add_rule(store, f"rule {i}") for i in range(5)]

    r = store.get_rules_for_context(project="myproj", limit=3)
    returned_ids = {x["id"] for x in r["rules"]}
    assert len(returned_ids) == 3

    for rid in ids:
        row = store.db.execute(
            "SELECT fire_count FROM rules WHERE id=?", (rid,)).fetchone()
        if rid in returned_ids:
            assert row["fire_count"] == 1
        else:
            assert row["fire_count"] == 0


def test_self_rules_context_handler_honours_limit_via_real_inject_path(live_store):
    """Drive the MCP dispatcher end-to-end (`server._do`), not the Store
    method directly, so a wiring/signature regression in the handler would
    be caught."""
    s, server = live_store
    for i in range(25):
        _add_rule(s, f"rule {i}")

    raw = asyncio.run(server._do("self_rules_context", {"project": "myproj", "limit": 3}))
    out = json.loads(raw)
    assert out["rules_count"] == 3
    assert out["total_matched"] == 25


def test_manage_rule_list_no_cap_returns_all_35_active_rules(store):
    """35 active rules seeded → manage_rule(action="list") returns all 35
    (regression against the old hardcoded LIMIT 30)."""
    for i in range(35):
        _add_rule(store, f"rule {i}")

    r = store.manage_rule("sess-limit-test", "list", project="myproj")
    assert r["total"] == 35
    assert r["total_matched"] == 35
    assert len(r["rules"]) == 35


def test_manage_rule_list_explicit_limit_caps_result(store):
    """limit=10 parameter → exactly 10 returned, total_matched stays 35."""
    for i in range(35):
        _add_rule(store, f"rule {i}")

    r = store.manage_rule("sess-limit-test", "list", project="myproj", limit=10)
    assert r["total"] == 10
    assert r["total_matched"] == 35
    assert len(r["rules"]) == 10


def test_manage_rule_list_negative_limit_returns_error(store):
    """A negative `limit` returns the same {"error": ...} shape as
    get_rules_for_context's bad-limit guard, and does not truncate."""
    for i in range(5):
        _add_rule(store, f"rule {i}")

    r = store.manage_rule("sess-limit-test", "list", project="myproj", limit=-1)
    assert "error" in r
    assert "rules" not in r


def test_self_rules_handler_list_honours_limit_via_real_inject_path(live_store):
    """Drive the MCP dispatcher end-to-end (`server._do`) for self_rules,
    not the Store method directly, so a wiring/signature regression in the
    handler would be caught."""
    s, server = live_store
    for i in range(35):
        _add_rule(s, f"rule {i}")

    raw = asyncio.run(server._do("self_rules", {"action": "list", "project": "myproj", "limit": 10}))
    out = json.loads(raw)
    assert out["total"] == 10
    assert out["total_matched"] == 35
