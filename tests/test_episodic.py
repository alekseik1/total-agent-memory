"""Tests for the v10 episodic links (Entity → Event → Fact)."""

from __future__ import annotations

import json
import sqlite3
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import episodic as ep
from base_schema import apply_full_schema


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("MEMORY_EPISODIC_ENABLED", "true")
    monkeypatch.delenv("MEMORY_EPISODIC_MAX_ENTITIES_PER_EVENT", raising=False)
    yield


@pytest.fixture
def gdb():
    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    apply_full_schema(db)
    yield db
    db.close()


def _add_node(db, nid, name, ntype, project=None, status="active"):
    props = json.dumps({"project": project} if project else {})
    db.execute(
        """INSERT INTO graph_nodes (
            id, type, name, properties, first_seen_at, last_seen_at, status
        ) VALUES (?, ?, ?, ?, '2026-04-27T00:00:00Z', '2026-04-27T00:00:00Z', ?)""",
        (nid, ntype, name, props, status),
    )
    db.commit()


def _link(db, knowledge_id, node_id, role="mentions"):
    db.execute(
        """INSERT INTO knowledge_nodes (knowledge_id, node_id, role, strength)
           VALUES (?, ?, ?, 1.0)""",
        (knowledge_id, node_id, role),
    )
    db.commit()


def _add_knowledge(db, kid, content="x", ktype="decision", project="vito"):
    db.execute(
        """INSERT INTO knowledge (id, session_id, type, content, project, created_at)
           VALUES (?, 's1', ?, ?, ?, '2026-04-27T00:00:00Z')""",
        (kid, ktype, content, project),
    )
    db.commit()


# ──────────────────────────────────────────────
# record_save_event — writer
# ──────────────────────────────────────────────


def test_record_save_event_creates_event_node_and_edges(gdb):
    _add_knowledge(gdb, 1, content="Migrated session storage to Memcached")
    _add_node(gdb, "ent1", "memcached", "technology")
    _add_node(gdb, "ent2", "redis", "technology")
    _link(gdb, 1, "ent1", role="mentions")
    _link(gdb, 1, "ent2", role="mentions")

    rec = ep.record_save_event(gdb, knowledge_id=1, project="vito", session_id="s1")
    assert rec is not None
    assert rec.entity_count == 2

    # Event node exists
    row = gdb.execute(
        "SELECT type, name FROM graph_nodes WHERE id=?", (rec.node_id,)
    ).fetchone()
    assert row["type"] == ep.EVENT_NODE_TYPE
    assert row["name"].startswith("save:1:")

    # MENTIONED_IN edges from each entity to the event
    edge_rows = gdb.execute(
        "SELECT source_id, target_id, relation_type FROM graph_edges "
        "WHERE target_id=? ORDER BY source_id",
        (rec.node_id,),
    ).fetchall()
    sources = {r["source_id"] for r in edge_rows}
    assert sources == {"ent1", "ent2"}
    assert all(r["relation_type"] == ep.MENTIONED_IN for r in edge_rows)

    # knowledge_nodes 'represents' link from knowledge to event
    rep = gdb.execute(
        "SELECT role FROM knowledge_nodes WHERE knowledge_id=1 AND node_id=?",
        (rec.node_id,),
    ).fetchone()
    assert rep["role"] == "represents"


def test_record_save_event_skips_when_no_entities(gdb):
    _add_knowledge(gdb, 1)
    # No entity-typed graph_nodes linked to k=1.
    rec = ep.record_save_event(gdb, knowledge_id=1, project="vito", session_id="s1")
    assert rec is None
    assert gdb.execute("SELECT COUNT(*) AS c FROM graph_nodes "
                       "WHERE type=?", (ep.EVENT_NODE_TYPE,)).fetchone()["c"] == 0


def test_record_save_event_disabled_returns_none(gdb, monkeypatch):
    monkeypatch.setenv("MEMORY_EPISODIC_ENABLED", "false")
    _add_knowledge(gdb, 1)
    _add_node(gdb, "ent1", "redis", "technology")
    _link(gdb, 1, "ent1")
    assert ep.record_save_event(gdb, knowledge_id=1, project="vito", session_id="s1") is None


def test_record_save_event_skips_non_entity_node_types(gdb):
    _add_knowledge(gdb, 1)
    # Only tag/project nodes — no entity types — should produce no event.
    _add_node(gdb, "tag1", "database", "concept")  # 'concept' IS in list — confirm
    _add_node(gdb, "proj1", "vito", "project")     # 'project' IS in list
    _add_node(gdb, "rule1", "no-commits", "rule")  # 'rule' not in list
    _link(gdb, 1, "rule1")  # only the rule is linked
    rec = ep.record_save_event(gdb, knowledge_id=1, project="vito", session_id="s1")
    assert rec is None


def test_record_save_event_caps_entities(gdb, monkeypatch):
    monkeypatch.setenv("MEMORY_EPISODIC_MAX_ENTITIES_PER_EVENT", "3")
    _add_knowledge(gdb, 1)
    for i in range(10):
        nid = f"e{i}"
        _add_node(gdb, nid, f"tech{i}", "technology")
        _link(gdb, 1, nid)
    rec = ep.record_save_event(gdb, knowledge_id=1, project="vito", session_id="s1")
    assert rec.entity_count == 3


# ──────────────────────────────────────────────
# Read helpers
# ──────────────────────────────────────────────


def test_find_events_for_entity_returns_recent_first(gdb):
    _add_knowledge(gdb, 1)
    _add_knowledge(gdb, 2)
    _add_node(gdb, "ent1", "Postgres", "technology")
    _link(gdb, 1, "ent1")
    _link(gdb, 2, "ent1")

    rec_a = ep.record_save_event(gdb, knowledge_id=1, project="vito", session_id="s1")
    # Force the second event to have a strictly later timestamp.
    time.sleep(1.05)  # graph_nodes.last_seen_at is second-precision
    rec_b = ep.record_save_event(gdb, knowledge_id=2, project="vito", session_id="s1")

    hits = ep.find_events_for_entity(gdb, entity_name="Postgres")
    assert len(hits) == 2
    # Newest first
    assert hits[0].event_node_id == rec_b.node_id
    assert hits[1].event_node_id == rec_a.node_id


def test_find_events_for_entity_filters_by_project(gdb):
    _add_knowledge(gdb, 1, project="vito")
    _add_knowledge(gdb, 2, project="floatytv")
    _add_node(gdb, "ent1", "Postgres", "technology")
    _link(gdb, 1, "ent1")
    _link(gdb, 2, "ent1")
    ep.record_save_event(gdb, knowledge_id=1, project="vito", session_id="s1")
    ep.record_save_event(gdb, knowledge_id=2, project="floatytv", session_id="s2")

    hits = ep.find_events_for_entity(gdb, entity_name="Postgres", project="vito")
    assert len(hits) == 1
    assert hits[0].project == "vito"


def test_find_co_mentioned_events_intersects(gdb):
    _add_knowledge(gdb, 1)
    _add_knowledge(gdb, 2)
    _add_knowledge(gdb, 3)
    _add_node(gdb, "bob", "Bob", "person")
    _add_node(gdb, "pg", "Postgres", "technology")
    _add_node(gdb, "go", "Golang", "technology")

    # k=1 → bob + postgres   ← co-mention
    _link(gdb, 1, "bob"); _link(gdb, 1, "pg")
    # k=2 → bob + go         ← only bob
    _link(gdb, 2, "bob"); _link(gdb, 2, "go")
    # k=3 → pg only
    _link(gdb, 3, "pg")

    ep.record_save_event(gdb, knowledge_id=1, project="vito", session_id="s")
    ep.record_save_event(gdb, knowledge_id=2, project="vito", session_id="s")
    ep.record_save_event(gdb, knowledge_id=3, project="vito", session_id="s")

    hits = ep.find_co_mentioned_events(gdb, entity_a="Bob", entity_b="Postgres")
    assert len(hits) == 1
    assert hits[0].knowledge_id == 1


def test_find_co_mentioned_returns_empty_when_no_overlap(gdb):
    _add_knowledge(gdb, 1)
    _add_knowledge(gdb, 2)
    _add_node(gdb, "bob", "Bob", "person")
    _add_node(gdb, "pg", "Postgres", "technology")
    _link(gdb, 1, "bob")
    _link(gdb, 2, "pg")
    ep.record_save_event(gdb, knowledge_id=1, project="vito", session_id="s")
    ep.record_save_event(gdb, knowledge_id=2, project="vito", session_id="s")

    assert ep.find_co_mentioned_events(gdb, entity_a="Bob", entity_b="Postgres") == []


def test_get_event_for_knowledge_round_trip(gdb):
    _add_knowledge(gdb, 1)
    _add_node(gdb, "ent1", "Postgres", "technology")
    _link(gdb, 1, "ent1")
    rec = ep.record_save_event(gdb, knowledge_id=1, project="vito", session_id="s1")
    hit = ep.get_event_for_knowledge(gdb, knowledge_id=1)
    assert hit is not None
    assert hit.event_node_id == rec.node_id
    assert hit.knowledge_id == 1


def test_get_event_for_knowledge_returns_none_when_no_event(gdb):
    _add_knowledge(gdb, 99)
    assert ep.get_event_for_knowledge(gdb, knowledge_id=99) is None
