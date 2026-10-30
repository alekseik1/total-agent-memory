"""Regression tests for the 14.10.0 correctness review (H1, H2, M1-M3, L4, L5)."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import queue_retry  # noqa: E402
from memory_core.schema_migration import Migration, MigrationRunner  # noqa: E402

ROOT = Path(__file__).parent.parent
MIGRATION_045 = next((ROOT / "migrations").glob("045_*.sql"))


@pytest.fixture
def live(monkeypatch, tmp_path):
    for sub in ("raw", "blobs", "chroma", "backups"):
        (tmp_path / sub).mkdir(exist_ok=True)
    import server
    monkeypatch.setattr(server, "MEMORY_DIR", tmp_path)
    store = server.Store()
    monkeypatch.setattr(server, "store", store)
    monkeypatch.setattr(server, "SID", "sess-fix")
    monkeypatch.setattr(server, "BRANCH", "")
    store.db.execute("INSERT INTO sessions (id, started_at, project, status) VALUES (?, ?, ?, ?)",
                     (server.SID, "2026-09-28T00:00:00Z", "p", "open"))
    store.db.commit()
    yield server, store, tmp_path
    store.db.close()


# H1 ---------------------------------------------------------------------------

def _pre_045_queues() -> sqlite3.Connection:
    db = sqlite3.connect(":memory:")
    db.executescript("""
        CREATE TABLE migrations(version TEXT PRIMARY KEY, description TEXT NOT NULL, applied_at TEXT NOT NULL);
        CREATE TABLE enrichment_queue(id INTEGER PRIMARY KEY, status TEXT);
        CREATE TABLE deep_enrichment_queue(id INTEGER PRIMARY KEY, status TEXT);
        CREATE TABLE triple_extraction_queue(id INTEGER PRIMARY KEY, status TEXT);
        CREATE TABLE ingest_queue(id INTEGER PRIMARY KEY, status TEXT);
    """)
    db.commit()
    return db


def test_migration_045_tolerates_columns_added_by_runtime_helpers():
    db = _pre_045_queues()
    queue_retry.ensure_queue_columns(db, "deep_enrichment_queue")
    queue_retry.ensure_queue_columns(db, "triple_extraction_queue")
    MigrationRunner(db).apply(Migration("045", "queue retry", MIGRATION_045.read_text(encoding="utf-8")))
    assert db.execute("SELECT 1 FROM migrations WHERE version='045'").fetchone()
    columns = {row[1] for row in db.execute("PRAGMA table_info(enrichment_queue)")}
    assert {"next_attempt_at", "deferrals", "requeues"} <= columns
    assert db.execute("SELECT COUNT(*) FROM enrichment_stage_done").fetchone()[0] == 0


def test_store_starts_when_queue_columns_were_added_before_migration_045(monkeypatch, tmp_path):
    for sub in ("raw", "blobs", "chroma", "backups"):
        (tmp_path / sub).mkdir()
    import server
    monkeypatch.setattr(server, "MEMORY_DIR", tmp_path)
    first = server.Store()
    db_path = first.db_path
    first.db.execute("DELETE FROM migrations WHERE version='045'")
    first.db.commit()
    first.db.close()
    # Columns stay in place (SQLite cannot drop them cheaply): exactly the state a
    # reflection daemon leaves behind after ensure_column ran on new code.
    store = server.Store()
    try:
        assert store.db.execute("SELECT 1 FROM migrations WHERE version='045'").fetchone()
    finally:
        store.db.close()
    assert db_path


# H2 ---------------------------------------------------------------------------

class _FakeOldServer:
    """mcp < 1.10: call_tool() takes no validate_input and the SDK validates nothing."""

    def __init__(self):
        self.registered = {}

    def list_tools(self):
        return lambda fn: self.registered.setdefault("list", fn)

    def call_tool(self):
        return lambda fn: self.registered.setdefault("call", fn)


class _FakeNewServer(_FakeOldServer):
    def call_tool(self, *, validate_input=True):
        self.registered["validate_input"] = validate_input
        return lambda fn: self.registered.setdefault("call", fn)


def test_registers_call_tool_on_sdk_without_validate_input():
    import server
    fake = _FakeOldServer()
    assert server._register_mcp_handlers(fake) == "1.x"
    assert fake.registered["call"] is server._call_tool_raising


def test_registers_validating_wrapper_on_sdk_with_validate_input():
    import server
    fake = _FakeNewServer()
    assert server._register_mcp_handlers(fake) == "1.x"
    assert fake.registered["validate_input"] is False
    assert fake.registered["call"] is not server._call_tool_raising


def test_missing_jsonschema_skips_validation_instead_of_raising(monkeypatch):
    import server
    monkeypatch.setitem(sys.modules, "jsonschema", None)
    asyncio.run(server._validate_tool_arguments("memory_save", {"bogus": 1}))


# M1 ---------------------------------------------------------------------------

def test_rejected_update_leaves_no_replayable_intent(live):
    server, store, _ = live
    rid, *_ = store.save_knowledge(server.SID, "Cache TTL is sixty seconds for the catalog", "fact", "p")
    store.db.execute("UPDATE knowledge SET status='superseded' WHERE id=?", (rid,))
    store.db.commit()
    with pytest.raises(ValueError, match="no longer active"):
        store.save_knowledge(server.SID, "Cache TTL is ninety seconds for the catalog", "fact", "p",
                             skip_dedup=True, skip_quality=True, supersedes=rid)
    pending = store.db.execute("SELECT COUNT(*) FROM write_intents WHERE status='pending'").fetchone()[0]
    assert pending == 0


def test_update_intent_payload_carries_supersedes(live):
    server, store, _ = live
    rid, *_ = store.save_knowledge(server.SID, "Retention window is thirty days for audit logs", "fact", "p")
    store.save_knowledge(server.SID, "Retention window is ninety days for audit logs", "fact", "p",
                         skip_dedup=True, skip_quality=True, supersedes=rid)
    # The intent of a successful write is closed; a crash before that must replay the supersede.
    import outbox
    seen = {}
    real = outbox.create_intent

    def spy(db, **kwargs):
        seen.update(kwargs["payload"])
        return real(db, **kwargs)

    outbox.create_intent = spy
    try:
        new_id, *_ = store.save_knowledge(server.SID, "Retention window is one year for audit logs", "fact", "p",
                                          skip_dedup=True, skip_quality=True, supersedes=rid + 1)
    finally:
        outbox.create_intent = real
    assert seen["supersedes"] == rid + 1
    assert new_id


# M2 ---------------------------------------------------------------------------

def test_auto_filter_keeps_non_prose_runs_instead_of_on_empty_placeholder(live):
    server, store, _ = live
    text = ("The deploy failed after the migration step ran on staging.\n- step 1: 10 rpm\n- step 2: 20 rpm\n"
            "Here is the full trace from the worker process.\n"
            'Traceback (most recent call last):\n  File "worker.py", line 10, in run\n'
            "ValueError: boom")
    from autofilter import detect_filter
    assert detect_filter(text) == "stack_trace"
    result, is_error = asyncio.run(server._call_tool_impl("memory_save", {"content": text, "type": "fact",
                                                                          "project": "p"}))
    assert not is_error
    saved = json.loads(result[0].text)
    content = store.db.execute("SELECT content FROM knowledge WHERE id=?", (saved["id"],)).fetchone()[0]
    assert "(no stack trace)" not in content
    assert "10 rpm" in content and "20 rpm" in content


# M3 ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["Pytest", "git-status", "stack-trace"])
def test_unusual_but_well_formed_filter_name_warns_and_stores(live, name):
    server, store, _ = live
    result, is_error = asyncio.run(server._call_tool_impl(
        "memory_save", {"content": f"Unfiltered body for filter {name} stays intact", "type": "fact",
                        "project": "p", "filter": name}))
    assert not is_error, result[0].text
    payload = json.loads(result[0].text)
    assert "filter_warning" in payload


@pytest.mark.parametrize("name", ["../secret", "a/b", "a\\b", ".."])
def test_path_like_filter_name_is_rejected(live, name):
    server, store, _ = live
    result, is_error = asyncio.run(server._call_tool_impl(
        "memory_save", {"content": "body that must not be stored", "type": "fact", "project": "p",
                        "filter": name}))
    assert is_error
    assert store.db.execute("SELECT COUNT(*) FROM knowledge WHERE content LIKE 'body that must%'").fetchone()[0] == 0


# L4 ---------------------------------------------------------------------------

def test_file_watcher_links_only_the_row_it_saved(tmp_path, monkeypatch):
    from ingestion import file_watcher as fw
    linked: list[int] = []
    import graph.auto_link as auto_link
    monkeypatch.setattr(auto_link, "auto_link_knowledge", lambda db, kid, text, project: linked.append(kid))
    monkeypatch.setattr(fw, "MEMORY_DIR", tmp_path)
    db = sqlite3.connect(":memory:")
    db.executescript("""
        CREATE TABLE knowledge(id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, type TEXT, content TEXT,
            context TEXT, project TEXT DEFAULT 'general', tags TEXT, status TEXT DEFAULT 'active',
            confidence REAL DEFAULT 1.0, source TEXT, created_at TEXT);
    """)
    note = tmp_path / "note.txt"
    note.write_text("Duplicate inbox note about the release checklist", encoding="utf-8")
    processor = fw.FileProcessor(str(tmp_path / "x.db"), tmp_path)
    processor._process_text(db, note)
    db.execute("INSERT INTO knowledge(content, project, status) VALUES ('unrelated newer row', 'general', 'active')")
    processor._process_text(db, note)  # deduplicated: created=False
    assert linked == [1]


# L5 ---------------------------------------------------------------------------

def test_pending_rows_over_lowered_attempt_limit_are_failed(live, monkeypatch):
    server, store, _ = live
    import enrichment_worker as ew
    rid, *_ = store.save_knowledge(server.SID, "Row left pending at three attempts", "fact", "p")
    store.db.execute("DELETE FROM enrichment_queue")
    store.db.execute(
        "INSERT INTO enrichment_queue(knowledge_id, session_id, project, ktype, content_snapshot, "
        "status, attempts, enqueued_at) VALUES (?, ?, 'p', 'fact', 'x', 'pending', 3, '2026-01-01T00:00:00Z')",
        (rid, server.SID))
    store.db.commit()
    monkeypatch.setenv("MEMORY_ENRICH_MAX_ATTEMPTS", "2")
    assert ew.fail_exhausted_pending(store.db) == 1
    status = store.db.execute("SELECT status, finished_at FROM enrichment_queue").fetchone()
    assert status[0] == "failed" and status[1]
