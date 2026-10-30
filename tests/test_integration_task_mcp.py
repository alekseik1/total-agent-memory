"""Integration tests: MCP wiring for task_classifier + task_phases."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path

import pytest


@pytest.fixture
def srv(monkeypatch, tmp_path):
    """Real MCP server with temp storage + task_phases migration applied."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

    (tmp_path / "blobs").mkdir(exist_ok=True)
    (tmp_path / "chroma").mkdir(exist_ok=True)

    import server
    monkeypatch.setattr(server, "MEMORY_DIR", tmp_path)

    s = server.Store()
    # Apply migration 012 so task_phases table exists in the ephemeral store.
    mig = Path(__file__).parent.parent / "migrations" / "012_task_phases.sql"
    s.db.executescript(mig.read_text(encoding="utf-8"))
    s.db.commit()

    # Register the store globally so MCP dispatchers see it.
    server.store = s
    server.SID = "test-session-integration"
    # Ensure session row exists for logging calls.
    s.db.execute(
        "INSERT OR IGNORE INTO sessions (id, started_at, project, status) VALUES (?, ?, ?, ?)",
        (server.SID, "2026-04-19T00:00:00Z", "test", "open"),
    )
    s.db.commit()

    yield server
    try:
        s.db.close()
    except Exception:
        pass


def _call(server_mod, name, args):
    """Invoke the private MCP dispatcher and return decoded JSON."""
    raw = asyncio.run(server_mod._do(name, args))
    return json.loads(raw)


# ──────────────────────────────────────────────
# Tool discovery — tools must be registered
# ──────────────────────────────────────────────

def test_new_tools_registered(srv):
    tools = asyncio.run(srv.list_tools())
    names = {t.name for t in tools}
    for expected in ("classify_task", "task_create",
                     "phase_transition", "task_phases_list"):
        assert expected in names, f"{expected} missing from MCP tools list"


# ──────────────────────────────────────────────
# classify_task tool
# ──────────────────────────────────────────────

def test_mcp_classify_task_tool_returns_structured_dict(srv):
    result = _call(srv, "classify_task",
                   {"description": "refactor auth middleware to JWT-only"})
    assert result["level"] == 3
    assert "creative" in result["suggested_phases"]
    assert result["confidence"] > 0.0
    assert "rationale" in result


# ──────────────────────────────────────────────
# task_create → phase_transition → list
# ──────────────────────────────────────────────

def test_mcp_phase_transition_persists_artifacts(srv):
    created = _call(srv, "task_create", {
        "task_id": "mcp-task-1",
        "description": "add /users endpoint",
        "level": 2,
    })
    assert created["phase"] == "van"

    moved = _call(srv, "phase_transition", {
        "task_id": "mcp-task-1",
        "new_phase": "plan",
        "artifacts": {"files": ["api.py"], "decisions": ["REST v2"]},
        "notes": "planning done",
    })
    assert moved["from_phase"] == "van"
    assert moved["to_phase"] == "plan"

    listed = _call(srv, "task_phases_list", {"task_id": "mcp-task-1"})
    phases = listed["phases"]
    assert [p["phase"] for p in phases] == ["van", "plan"]
    # artifacts_json was decoded into "artifacts" dict
    plan_row = phases[1]
    assert plan_row["artifacts"]["files"] == ["api.py"]
    assert plan_row["artifacts"]["decisions"] == ["REST v2"]
    assert plan_row["exited_at"] is None  # still open
