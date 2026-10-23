"""MCP initialize instructions and the compact wire form of tool results."""

from __future__ import annotations

import asyncio
import json
import re
import sys
from datetime import date, datetime
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import server
import server_instructions
import tool_profiles

# Compact JSON must save at least this share of the indented form on a recall answer.
MIN_RECALL_SAVING = 0.2
TODAY = date(2026, 10, 13)
TOOL_NAME = re.compile(r"\b(?:memory|session|self|kg)_[a-z_]+\b|\bfile_context\b|\blearn_error\b")


# ── instructions ───────────────────────────────────────────────────────────

def test_server_hands_out_instructions_with_todays_date():
    text = server.app.instructions
    assert text is not None
    assert len(text) < server_instructions.MAX_INSTRUCTIONS_CHARS
    assert datetime.now().astimezone().date().isoformat() in text
    for phrase in ("memory_recall", "session_init", "Do not save", "ephemeral"):
        assert phrase in text
    assert server.app.create_initialization_options().instructions == text


def test_instructions_only_name_tools_every_profile_lists():
    named = set(TOOL_NAME.findall(server_instructions.build({}, date(2026, 1, 2))))
    assert named and named <= tool_profiles.MINIMAL_TOOLS


def test_instructions_carry_the_given_date():
    assert "Server start date: 2026-01-02." in server_instructions.build({}, date(2026, 1, 2))


@pytest.mark.parametrize("value", ["false", "0", "off", "FALSE"])
def test_instructions_can_be_disabled(value):
    assert server_instructions.build({server_instructions.INSTRUCTIONS_ENV: value}, TODAY) is None


@pytest.mark.parametrize("value", ["", "true", "on"])
def test_instructions_enabled_by_default(value):
    assert server_instructions.build({server_instructions.INSTRUCTIONS_ENV: value}, TODAY)


def test_unrecognised_value_keeps_the_default_and_warns(caplog):
    assert server_instructions.build({server_instructions.INSTRUCTIONS_ENV: "maybe"}, TODAY)
    assert "not true/false" in caplog.text


def test_instructions_setting_is_in_the_dashboard_catalogue():
    from settings_catalog import LOCAL_CATALOG

    keys = {spec.key: spec for spec in LOCAL_CATALOG}
    assert keys[server_instructions.INSTRUCTIONS_ENV].choices == ("true", "false")
    assert keys[server_instructions.INSTRUCTIONS_ENV].group == "agent"


# ── compact responses ─────────────────────────────────────────────────────

SAMPLE = {"query": "база данных", "total": 2, "nothing": None, "results": {"decision": [
    {"id": 1, "content": "Решение: PostgreSQL 18", "tags": ["db"], "valid_to": None, "score": 1.5},
    {"id": 2, "content": "Redis 7 — кэш", "tags": [], "valid_to": "2026-01-01", "score": 0.25},
]}}


def test_render_is_compact_and_parses_identically():
    compact = server._render(SAMPLE)
    indented = json.dumps(SAMPLE, ensure_ascii=False, indent=2, default=str)
    assert "\n" not in compact and ": " not in compact.replace("Решение: ", "")
    assert "база данных" in compact  # UTF-8 kept, not \\u-escaped
    assert json.loads(compact) == json.loads(indented)
    assert len(compact.encode()) < len(indented.encode())


def test_render_falls_back_to_str_for_unknown_types():
    assert json.loads(server._render({"at": date(2026, 1, 2)})) == {"at": "2026-01-02"}


def test_null_fields_leave_hits_but_not_the_top_level():
    out = server._without_null_hit_fields(SAMPLE)
    assert out["nothing"] is None
    first, second = out["results"]["decision"]
    assert "valid_to" not in first and second["valid_to"] == "2026-01-01"
    assert first["tags"] == ["db"] and second["tags"] == []
    assert SAMPLE["results"]["decision"][0]["valid_to"] is None  # input untouched


def test_null_fields_leave_flat_hit_lists():
    out = server._without_null_hit_fields({"results": [{"id": 1, "x": None}], "error": None})
    assert out == {"results": [{"id": 1}], "error": None}


@pytest.mark.parametrize("value", [None, [], "text", {"results": "none"}, {"no_results": 1}])
def test_null_stripping_leaves_other_shapes_alone(value):
    assert server._without_null_hit_fields(value) == value


@pytest.fixture
def live_store(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "MEMORY_DIR", tmp_path)
    store = server.Store()
    monkeypatch.setattr(server, "store", store)
    monkeypatch.setattr(server, "recall", server.Recall(store))
    monkeypatch.setattr(server, "SID", "sess-compact")
    monkeypatch.setattr(server, "BRANCH", "")
    store.db.execute("INSERT INTO sessions (id, started_at, project, status) VALUES (?, ?, ?, ?)",
                     ("sess-compact", "2026-04-14T00:00:00Z", "demo", "open"))
    store.db.commit()
    yield store
    store.db.close()


def test_recall_answer_is_compact_and_smaller(live_store):
    for text in ("Use PostgreSQL 18 for orders because of logical replication.",
                 "RabbitMQ 4 quorum queues replaced mirrored queues.",
                 "Решение: хранить вебхуки в отдельной таблице."):
        live_store.save_knowledge(sid="sess-compact", content=text, ktype="decision", project="demo")
    raw = asyncio.run(server._do("memory_recall", {"query": "database queue", "project": "demo"}))
    parsed = json.loads(raw)
    assert "\n" not in raw
    assert parsed["total"] >= 1
    hits = [hit for group in parsed["results"].values() for hit in group]
    assert hits and all(value is not None for hit in hits for value in hit.values())
    indented = json.dumps(parsed, ensure_ascii=False, indent=2)
    saving = 1 - len(raw.encode()) / len(indented.encode())
    assert saving >= MIN_RECALL_SAVING, f"compact saved only {saving:.0%}"
