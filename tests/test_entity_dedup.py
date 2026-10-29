"""Tests for the v10 pre-write entity dedup."""

from __future__ import annotations

import math
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import entity_dedup as ed


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("MEMORY_ENTITY_DEDUP_ENABLED", "true")
    monkeypatch.delenv("MEMORY_ENTITY_DEDUP_THRESHOLD", raising=False)
    monkeypatch.delenv("MEMORY_ENTITY_DEDUP_LOG_ALL", raising=False)
    yield


@pytest.fixture
def edb():
    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    migration = (
        Path(__file__).parent.parent / "migrations" / "018_entity_dedup.sql"
    ).read_text(encoding="utf-8")
    db.executescript(migration)
    yield db
    db.close()


def _vec(*components):
    """Tiny helper — easier than typing list literals everywhere."""
    return list(components)


def _make_embed_fn(mapping):
    """`mapping` is dict[str → list[float]]. Returns a callable that
    embeds in input order so tests can match results to inputs."""
    def fn(texts):
        out = []
        for t in texts:
            out.append(mapping.get(t.lower(), [0.0, 0.0, 0.0]))
        return out
    return fn


# ──────────────────────────────────────────────
# Cosine helper
# ──────────────────────────────────────────────


def test_cosine_orthogonal_returns_zero():
    assert ed.cosine([1, 0, 0], [0, 1, 0]) == 0.0


def test_cosine_identical_returns_one():
    assert ed.cosine([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)


def test_cosine_handles_zero_vector():
    assert ed.cosine([0, 0, 0], [1, 1, 1]) == 0.0
    assert ed.cosine([], [1]) == 0.0


# ──────────────────────────────────────────────
# find_dedup_candidates
# ──────────────────────────────────────────────


def test_find_returns_match_above_threshold():
    candidates = [
        ed.EntityCandidate(node_id="n1", name="Beever Atlas", type="project"),
        ed.EntityCandidate(node_id="n2", name="Postgres", type="technology"),
    ]
    embed = _make_embed_fn({
        "atlas":         _vec(1.0, 0.0, 0.0),
        "beever atlas":  _vec(0.95, 0.0, 0.05),
        "postgres":      _vec(0.0, 1.0, 0.0),
    })
    matches = ed.find_dedup_candidates(
        "Atlas", candidates=candidates, embed_fn=embed, threshold=0.7,
    )
    assert len(matches) == 1
    assert matches[0][0].name == "Beever Atlas"
    assert matches[0][1] > 0.9


def test_find_excludes_below_threshold():
    candidates = [
        ed.EntityCandidate(node_id="n2", name="Postgres", type="technology"),
    ]
    embed = _make_embed_fn({
        "atlas":    _vec(1.0, 0.0, 0.0),
        "postgres": _vec(0.0, 1.0, 0.0),
    })
    matches = ed.find_dedup_candidates(
        "Atlas", candidates=candidates, embed_fn=embed, threshold=0.85,
    )
    assert matches == []


def test_find_caches_candidate_embeddings_across_calls():
    candidates = [
        ed.EntityCandidate(node_id="n1", name="Postgres", type="technology"),
    ]
    call_log = []
    def embed(texts):
        call_log.append(list(texts))
        return [[1.0 if "postgres" in t.lower() else 0.5, 0.0] for t in texts]
    # First call — embedder must be asked for the candidate.
    ed.find_dedup_candidates("Postgres", candidates=candidates,
                             embed_fn=embed, threshold=0.5)
    assert any("Postgres" in calls for calls in call_log)
    pre_count = len(call_log)
    # Second call — embedding cache is on the EntityCandidate object.
    ed.find_dedup_candidates("Postgres", candidates=candidates,
                             embed_fn=embed, threshold=0.5)
    last_call = call_log[-1]
    # Only the input tag should have been re-embedded, not the candidate.
    assert "Postgres" not in last_call or len(last_call) == 1


def test_find_handles_embed_failure():
    candidates = [
        ed.EntityCandidate(node_id="n1", name="Postgres", type="technology"),
    ]
    matches = ed.find_dedup_candidates(
        "Atlas", candidates=candidates, embed_fn=lambda texts: None,
    )
    assert matches == []


def test_find_returns_empty_for_empty_candidate_pool():
    assert ed.find_dedup_candidates("X", candidates=[], embed_fn=lambda t: [[1, 0]]) == []


# ──────────────────────────────────────────────
# canonicalize_entity_tags
# ──────────────────────────────────────────────


def _three_tag_setup():
    candidates = [
        ed.EntityCandidate(node_id="n1", name="vitamin_all", type="project"),
        ed.EntityCandidate(node_id="n2", name="Postgres",     type="technology"),
        ed.EntityCandidate(node_id="n3", name="Bob",          type="person"),
    ]
    # Vectors are 5-D so each canonical lives in a distinct subspace and
    # we can give "alice" coordinates orthogonal to every candidate.
    embed_map = {
        "vitamin all": _vec(0.95, 0.0, 0.0, 0.0, 0.0),
        "vitamin_all": _vec(1.00, 0.0, 0.0, 0.0, 0.0),
        "stockflow":   _vec(0.92, 0.05, 0.0, 0.0, 0.0),
        "postgres":    _vec(0.0, 1.0, 0.0, 0.0, 0.0),
        "pg":          _vec(0.0, 0.93, 0.07, 0.0, 0.0),
        "bob":         _vec(0.0, 0.0, 0.0, 1.0, 0.0),
        "alice":       _vec(0.0, 0.0, 0.0, 0.0, 1.0),  # orthogonal to all
    }
    return candidates, _make_embed_fn(embed_map)


def test_canonicalize_rewrites_known_synonym():
    candidates, embed = _three_tag_setup()
    out, decisions = ed.canonicalize_entity_tags(
        ["StockFlow", "PG"], candidates=candidates,
        embed_fn=embed, threshold=0.8,
    )
    # Tag rewritten + original kept (canonical_tags-style).
    assert "vitamin_all" in out
    assert "stockflow" in out
    assert "postgres" in out
    assert "pg" in out
    assert {d.decision for d in decisions} == {"merged"}


def test_canonicalize_keeps_unmatched_verbatim():
    candidates, embed = _three_tag_setup()
    out, decisions = ed.canonicalize_entity_tags(
        ["alice"], candidates=candidates, embed_fn=embed, threshold=0.8,
    )
    assert out == ["alice"]
    assert decisions == []  # no_match silenced unless LOG_ALL=1


def test_canonicalize_logs_no_matches_when_log_all_set(monkeypatch):
    monkeypatch.setenv("MEMORY_ENTITY_DEDUP_LOG_ALL", "1")
    candidates, embed = _three_tag_setup()
    out, decisions = ed.canonicalize_entity_tags(
        ["alice"], candidates=candidates, embed_fn=embed, threshold=0.8,
    )
    assert out == ["alice"]
    assert len(decisions) == 1
    assert decisions[0].decision == "no_match"


def test_canonicalize_disabled_passes_through(monkeypatch):
    monkeypatch.setenv("MEMORY_ENTITY_DEDUP_ENABLED", "false")
    candidates, embed = _three_tag_setup()
    out, decisions = ed.canonicalize_entity_tags(
        ["StockFlow"], candidates=candidates, embed_fn=embed, threshold=0.8,
    )
    assert out == ["StockFlow"]   # untouched
    assert decisions == []


def test_canonicalize_dedups_output():
    candidates, embed = _three_tag_setup()
    out, _ = ed.canonicalize_entity_tags(
        ["vitamin_all", "Vitamin_All", "VITAMIN_ALL"],
        candidates=candidates, embed_fn=embed, threshold=0.8,
    )
    assert out.count("vitamin_all") == 1


def test_canonicalize_preserves_order():
    candidates, embed = _three_tag_setup()
    out, _ = ed.canonicalize_entity_tags(
        ["StockFlow", "Bob", "PG"], candidates=candidates,
        embed_fn=embed, threshold=0.8,
    )
    # canonical of first input must come before any canonical of later input
    assert out.index("vitamin_all") < out.index("bob")
    assert out.index("bob") < out.index("postgres")


def test_canonicalize_handles_garbage_input():
    candidates, embed = _three_tag_setup()
    out, _ = ed.canonicalize_entity_tags(
        [None, "", "  ", 42, "Bob"], candidates=candidates,
        embed_fn=embed, threshold=0.8,
    )
    assert out == ["bob"]


# ──────────────────────────────────────────────
# Audit log persistence
# ──────────────────────────────────────────────


def test_log_decisions_persists_rows(edb):
    decisions = [
        ed.DedupDecision(
            input_tag="StockFlow", decision="merged",
            matched_node_id="n1", canonical_name="vitamin_all",
            similarity=0.92, threshold=0.85, reason="cosine=0.92",
        ),
        ed.DedupDecision(
            input_tag="alice", decision="no_match",
            matched_node_id=None, canonical_name=None,
            similarity=0.0, threshold=0.85, reason="below threshold",
        ),
    ]
    ed.log_decisions(edb, decisions, knowledge_id=42, project="vito")
    rows = edb.execute(
        "SELECT input_tag, decision, canonical_name, knowledge_id "
        "FROM entity_dedup_log ORDER BY id"
    ).fetchall()
    assert len(rows) == 2
    assert rows[0]["input_tag"] == "StockFlow"
    assert rows[0]["canonical_name"] == "vitamin_all"
    assert rows[0]["knowledge_id"] == 42
    assert rows[1]["decision"] == "no_match"


def test_log_decisions_empty_is_noop(edb):
    ed.log_decisions(edb, [], knowledge_id=1, project="vito")
    row = edb.execute("SELECT COUNT(*) AS c FROM entity_dedup_log").fetchone()
    assert row["c"] == 0


# ──────────────────────────────────────────────
# Production candidates query — graph_nodes lookup
# ──────────────────────────────────────────────


def test_production_candidates_query_filters_active_entity_types():
    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    db.executescript(
        """
        CREATE TABLE graph_nodes (
            id TEXT PRIMARY KEY, type TEXT NOT NULL, name TEXT NOT NULL,
            content TEXT, properties TEXT, source TEXT,
            importance REAL DEFAULT 0.5,
            first_seen_at TEXT, last_seen_at TEXT,
            mention_count INTEGER DEFAULT 1, status TEXT DEFAULT 'active'
        );
        INSERT INTO graph_nodes (id, type, name, status, mention_count) VALUES
            ('a', 'technology', 'Postgres', 'active', 10),
            ('b', 'project',    'vitamin_all', 'active', 5),
            ('c', 'rule',       'NoCommitsByClaude', 'active', 1),  -- excluded type
            ('d', 'technology', 'OldFramework', 'archived', 0);     -- excluded status
        """
    )
    cands = ed.production_candidates_query(db)
    names = {c.name for c in cands}
    assert names == {"Postgres", "vitamin_all"}
    db.close()
