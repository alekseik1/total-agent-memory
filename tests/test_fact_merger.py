"""Tests for semantic fact merger.

Unlike reflection.digest.merge_duplicates (Jaccard >=0.85 — near duplicates),
this module finds clusters of *related but distinct* facts (cosine 0.70-0.90)
and asks an LLM to synthesize them into one consolidated fact.

Example:
  "User uses Go for backend" + "User builds microservices in Go"
    → "User's primary backend language is Go (used for microservices)"
"""

from __future__ import annotations

import pytest


@pytest.fixture
def merger_db(db):
    return db


def _add(db, content: str, project: str = "demo", confidence: float = 1.0) -> int:
    return db.execute(
        "INSERT INTO knowledge (session_id, type, content, project, confidence, created_at) "
        "VALUES ('s1', 'fact', ?, ?, ?, ?)",
        (content, project, confidence, "2026-04-14T00:00:00Z"),
    ).lastrowid


# ──────────────────────────────────────────────
# Clustering
# ──────────────────────────────────────────────


def test_find_clusters_groups_similar_records(merger_db):
    from fact_merger import FactMerger

    a = _add(merger_db, "User prefers Go for backend services")
    b = _add(merger_db, "User builds backend APIs in Go language")
    c = _add(merger_db, "Unrelated: user likes coffee on rainy days")

    # fake similarity: a~b highly similar; c dissimilar to both
    def sim(id1: int, id2: int) -> float:
        pair = frozenset({id1, id2})
        if pair == frozenset({a, b}):
            return 0.82
        return 0.2

    m = FactMerger(merger_db, similarity_fn=sim)
    clusters = m.find_clusters(project="demo", min_similarity=0.7, max_cluster_size=5)

    assert any({a, b}.issubset(set(cl)) for cl in clusters)
    for cl in clusters:
        assert c not in cl or len(cl) == 1


def test_find_clusters_skips_exact_duplicates(merger_db):
    """Exact duplicates (sim >=0.97) belong to dedup, not merger."""
    from fact_merger import FactMerger

    a = _add(merger_db, "User uses Go")
    b = _add(merger_db, "User uses Go")

    def sim(*_a: int) -> float:
        return 0.99

    m = FactMerger(merger_db, similarity_fn=sim)
    clusters = m.find_clusters(
        project="demo", min_similarity=0.7, max_similarity=0.95
    )
    assert clusters == []


def test_find_clusters_respects_max_size(merger_db):
    from fact_merger import FactMerger

    ids = [_add(merger_db, f"fact {i}") for i in range(8)]

    def sim(i1: int, i2: int) -> float:
        return 0.8

    m = FactMerger(merger_db, similarity_fn=sim)
    clusters = m.find_clusters(project="demo", max_cluster_size=3)
    for cl in clusters:
        assert len(cl) <= 3


# ──────────────────────────────────────────────
# Merge
# ──────────────────────────────────────────────


def test_merge_creates_consolidated_record_and_archives_sources(merger_db, monkeypatch):
    from fact_merger import FactMerger

    a = _add(merger_db, "User uses Go for backend")
    b = _add(merger_db, "User builds APIs in Go")

    def fake_merge(contents: list[str]) -> str:
        return "User's primary backend language is Go (used for APIs)."

    m = FactMerger(merger_db, similarity_fn=lambda *_: 0.8, llm_merge_fn=fake_merge)
    result = m.merge_cluster([a, b])

    assert result["merged_id"] is not None
    merged_row = merger_db.execute(
        "SELECT content, status, superseded_by FROM knowledge WHERE id=?",
        (result["merged_id"],),
    ).fetchone()
    assert "Go" in merged_row["content"]
    assert merged_row["status"] == "active"

    # source rows marked superseded
    for src in (a, b):
        row = merger_db.execute(
            "SELECT status, superseded_by FROM knowledge WHERE id=?", (src,)
        ).fetchone()
        assert row["status"] == "archived"
        assert row["superseded_by"] == result["merged_id"]


def test_merge_rejected_when_llm_drops_a_path(merger_db):
    from fact_merger import FactMerger

    path = "/Users/alice/project/src/server.py"
    a = _add(merger_db, f"Edit {path} to fix the bug")
    b = _add(merger_db, f"The file {path} also needs a docstring")

    def drops_path(contents: list[str]) -> str:
        return "Fix the bug and add a docstring to the server file."

    m = FactMerger(merger_db, similarity_fn=lambda *_: 0.8, llm_merge_fn=drops_path)
    result = m.merge_cluster([a, b])

    assert result["merged_id"] is None
    assert path in result["reason"]


def test_merged_record_carries_merge_origin(merger_db, monkeypatch):
    import fact_merger as fact_merger_mod
    from fact_merger import MERGE_SESSION_ID, FactMerger

    monkeypatch.setattr(fact_merger_mod, "_now", lambda: "2026-04-20T00:00:00Z")

    a = _add(merger_db, "User uses Go for backend")
    b = _add(merger_db, "User builds APIs in Go")

    m = FactMerger(
        merger_db,
        similarity_fn=lambda *_: 0.8,
        llm_merge_fn=lambda contents: "User writes Go backends and APIs.",
    )
    result = m.merge_cluster([a, b])

    row = merger_db.execute(
        "SELECT session_id, source, type, project, updated_at FROM knowledge WHERE id=?",
        (result["merged_id"],),
    ).fetchone()
    assert row["session_id"] == MERGE_SESSION_ID
    assert row["source"] == "merged"
    assert row["type"] == "fact"
    assert row["project"] == "demo"
    assert row["updated_at"] == "2026-04-20T00:00:00Z"


def test_merged_record_session_resolves_to_seeded_session_row(merger_db):
    from fact_merger import MERGE_SESSION_ID, FactMerger

    a = _add(merger_db, "User uses Go for backend")
    b = _add(merger_db, "User builds APIs in Go")

    m = FactMerger(
        merger_db,
        similarity_fn=lambda *_: 0.8,
        llm_merge_fn=lambda contents: "User writes Go backends and APIs.",
    )
    result = m.merge_cluster([a, b])

    merged_session_id = merger_db.execute(
        "SELECT session_id FROM knowledge WHERE id=?", (result["merged_id"],)
    ).fetchone()["session_id"]
    assert merged_session_id == MERGE_SESSION_ID

    session = merger_db.execute(
        "SELECT id FROM sessions WHERE id=?", (merged_session_id,)
    ).fetchone()
    assert session is not None


def test_archived_sources_get_updated_at_stamp(merger_db, monkeypatch):
    import fact_merger as fact_merger_mod
    from fact_merger import FactMerger

    monkeypatch.setattr(fact_merger_mod, "_now", lambda: "2026-04-20T00:00:00Z")

    a = _add(merger_db, "fact a")
    b = _add(merger_db, "fact b")

    m = FactMerger(
        merger_db,
        similarity_fn=lambda *_: 0.8,
        llm_merge_fn=lambda contents: "merged summary",
    )
    m.merge_cluster([a, b])

    for src in (a, b):
        row = merger_db.execute(
            "SELECT updated_at FROM knowledge WHERE id=?", (src,)
        ).fetchone()
        assert row["updated_at"] == "2026-04-20T00:00:00Z"


def test_merged_record_is_handed_to_the_hook(merger_db):
    from fact_merger import FactMerger

    a = _add(merger_db, "User uses Go for backend")
    b = _add(merger_db, "User builds APIs in Go")
    seen: list[tuple[int, str]] = []

    m = FactMerger(
        merger_db,
        similarity_fn=lambda *_: 0.8,
        llm_merge_fn=lambda contents: "User writes Go backends.",
        on_merged=lambda kid, content: seen.append((kid, content)),
    )
    result = m.merge_cluster([a, b])

    assert seen == [(result["merged_id"], "User writes Go backends.")]


def test_failing_hook_does_not_undo_the_merge(merger_db):
    from fact_merger import FactMerger

    a = _add(merger_db, "fact a")
    b = _add(merger_db, "fact b")

    def broken(_kid, _content):
        raise RuntimeError("embedder down")

    m = FactMerger(
        merger_db,
        similarity_fn=lambda *_: 0.8,
        llm_merge_fn=lambda contents: "merged summary",
        on_merged=broken,
    )
    result = m.merge_cluster([a, b])

    assert result["merged_id"] is not None
    row = merger_db.execute(
        "SELECT status FROM knowledge WHERE id=?", (result["merged_id"],)
    ).fetchone()
    assert row["status"] == "active"


def test_vectors_fn_clusters_without_pairwise_calls(merger_db):
    from fact_merger import FactMerger

    a = _add(merger_db, "cat")
    b = _add(merger_db, "kitten")
    c = _add(merger_db, "spreadsheet")
    vectors = {a: [1.0, 0.0], b: [0.8, 0.6], c: [0.0, 1.0]}

    def exploding_similarity(*_a: int) -> float:
        raise AssertionError("pairwise similarity must not be used")

    m = FactMerger(
        merger_db,
        similarity_fn=exploding_similarity,
        vectors_fn=lambda ids: vectors,
    )
    clusters = m.find_clusters(project="demo", min_similarity=0.7, max_similarity=0.95)

    assert clusters == [sorted([a, b])]


def test_vectors_fn_compares_within_each_embedding_dimension(merger_db):
    from fact_merger import FactMerger

    a = _add(merger_db, "small model a")
    b = _add(merger_db, "small model b")
    c = _add(merger_db, "big model a")
    d = _add(merger_db, "big model b")
    vectors = {
        a: [1.0, 0.0],
        b: [0.8, 0.6],
        c: [1.0, 0.0, 0.0],
        d: [0.8, 0.6, 0.0],
    }

    m = FactMerger(
        merger_db,
        similarity_fn=lambda *_: 0.0,
        vectors_fn=lambda ids: vectors,
    )
    clusters = m.find_clusters(project="demo", min_similarity=0.7, max_similarity=0.95)

    assert sorted(clusters) == [sorted([a, b]), sorted([c, d])]


def test_vectors_fn_tolerates_zero_vector(merger_db):
    from fact_merger import FactMerger

    a = _add(merger_db, "real content")
    b = _add(merger_db, "empty embedding")

    m = FactMerger(
        merger_db,
        similarity_fn=lambda *_: 0.0,
        vectors_fn=lambda ids: {a: [1.0, 0.0], b: [0.0, 0.0]},
    )

    assert m._banded_pairs([a, b], 0.0, 0.0) == [(a, b)]


def test_merge_preserves_provenance_in_audit_table(merger_db):
    from fact_merger import FactMerger

    a = _add(merger_db, "fact a")
    b = _add(merger_db, "fact b")

    m = FactMerger(
        merger_db,
        similarity_fn=lambda *_: 0.8,
        llm_merge_fn=lambda contents: "merged summary",
    )
    result = m.merge_cluster([a, b])

    audit = merger_db.execute(
        "SELECT source_ids, merged_knowledge_id FROM knowledge_merges"
    ).fetchone()
    assert audit is not None
    assert str(a) in audit["source_ids"]
    assert str(b) in audit["source_ids"]
    assert audit["merged_knowledge_id"] == result["merged_id"]


def test_merge_skips_single_item(merger_db):
    from fact_merger import FactMerger

    a = _add(merger_db, "solo")
    m = FactMerger(
        merger_db,
        similarity_fn=lambda *_: 0.8,
        llm_merge_fn=lambda c: "x",
    )
    result = m.merge_cluster([a])
    assert result["merged_id"] is None
    assert "too small" in result["reason"].lower()


def test_merge_validator_rejects_bad_llm_output(merger_db):
    """If LLM output fails validator (loses URLs/paths), abort merge."""
    from fact_merger import FactMerger

    a = _add(merger_db, "See https://docs.example.com/api for details")
    b = _add(merger_db, "Also check https://docs.example.com/api spec")

    def bad_merge(contents: list[str]) -> str:
        return "Just read the docs."  # URL dropped!

    m = FactMerger(merger_db, similarity_fn=lambda *_: 0.8, llm_merge_fn=bad_merge)
    result = m.merge_cluster([a, b])

    assert result["merged_id"] is None
    assert "validat" in result["reason"].lower()

    # Sources remain active — nothing archived
    for src in (a, b):
        row = merger_db.execute(
            "SELECT status FROM knowledge WHERE id=?", (src,)
        ).fetchone()
        assert row["status"] == "active"


# ──────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────


def test_run_processes_all_clusters(merger_db):
    from fact_merger import FactMerger

    # Two obvious clusters
    a = _add(merger_db, "User uses Go")
    b = _add(merger_db, "User programs in Go")
    c = _add(merger_db, "User deploys with Docker")
    d = _add(merger_db, "User runs Docker containers")

    def sim(i1: int, i2: int) -> float:
        go_pair = frozenset({a, b})
        docker_pair = frozenset({c, d})
        if frozenset({i1, i2}) == go_pair:
            return 0.82
        if frozenset({i1, i2}) == docker_pair:
            return 0.80
        return 0.15

    def merge(contents: list[str]) -> str:
        return " | ".join(contents)[:200]

    m = FactMerger(merger_db, similarity_fn=sim, llm_merge_fn=merge)
    stats = m.run(project="demo")

    assert stats["clusters_found"] == 2
    assert stats["merged"] == 2
