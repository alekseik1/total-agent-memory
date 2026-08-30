"""Integration test for reflection.agent.run_full new phases (triple queue + fact merger)."""

from __future__ import annotations

import asyncio
import struct

import pytest

# These tests stub the LLM seam; `llm_enabled` skips the has_llm() probe
# so they do not depend on a live Ollama being present on the machine.
pytestmark = pytest.mark.usefixtures("llm_enabled")


@pytest.fixture
def refl_db(db):
    return db


def test_run_full_drains_triple_queue_and_runs_fact_merger(refl_db, monkeypatch):
    """End-to-end: enqueue items → run_full → triples processed + merge attempted."""
    from reflection.agent import ReflectionAgent
    from triple_extraction_queue import TripleExtractionQueue

    monkeypatch.setenv("MEMORY_FACT_MERGE_ENABLED", "true")

    # Seed knowledge + queue
    kid1 = refl_db.execute(
        "INSERT INTO knowledge (session_id, type, content, project, status, created_at) "
        "VALUES ('s1', 'fact', 'User uses Go', 'demo', 'active', '2026-04-14T00:00:00Z')"
    ).lastrowid
    kid2 = refl_db.execute(
        "INSERT INTO knowledge (session_id, type, content, project, status, created_at) "
        "VALUES ('s1', 'fact', 'User prefers Go', 'demo', 'active', '2026-04-14T00:00:00Z')"
    ).lastrowid
    refl_db.commit()
    q = TripleExtractionQueue(refl_db)
    q.enqueue(kid1)
    q.enqueue(kid2)

    # Stub ConceptExtractor.extract_and_link so it doesn't hit Ollama
    import ingestion.extractor as extractor_mod

    calls: list[int] = []

    def fake_extract_and_link(self, text, knowledge_id=None, deep=False):
        calls.append(knowledge_id)
        return {"relations": [], "entities": [], "concepts": []}

    monkeypatch.setattr(extractor_mod.ConceptExtractor, "extract_and_link", fake_extract_and_link)

    agent = ReflectionAgent(refl_db)

    # Stub fact_merger's LLM dependency: no embeddings => it returns early
    # That's fine — we just want to ensure the phase runs without error.
    report = asyncio.run(agent.run_full())

    # Triple extraction processed both queue items
    assert report["triple_extraction"]["processed"] == 2
    assert set(calls) == {kid1, kid2}

    # Fact merge phase ran (no embeddings → skipped gracefully)
    assert "fact_merge" in report
    assert report["fact_merge"].get("merged", 0) == 0  # no embeddings, no merging


def test_run_full_survives_triple_extraction_error(refl_db, monkeypatch):
    from reflection.agent import ReflectionAgent

    kid = refl_db.execute(
        "INSERT INTO knowledge (session_id, type, content, project, status, created_at) "
        "VALUES ('s1', 'fact', 'any', 'demo', 'active', '2026-04-14T00:00:00Z')"
    ).lastrowid
    refl_db.commit()
    from triple_extraction_queue import TripleExtractionQueue
    TripleExtractionQueue(refl_db).enqueue(kid)

    import ingestion.extractor as extractor_mod

    def broken(self, text, knowledge_id=None, deep=False):
        raise RuntimeError("ollama down")

    monkeypatch.setattr(extractor_mod.ConceptExtractor, "extract_and_link", broken)

    agent = ReflectionAgent(refl_db)
    report = asyncio.run(agent.run_full())

    # Still returns a report, just with failed triples
    assert report["triple_extraction"]["failed"] >= 1
    assert report["triple_extraction"]["processed"] == 0


def test_failing_phase_is_persisted_in_report(refl_db, monkeypatch):
    import json

    from reflection.agent import ReflectionAgent

    def broken_similarity_fn():
        raise RuntimeError("boom")

    monkeypatch.setenv("MEMORY_FACT_MERGE_ENABLED", "true")
    agent = ReflectionAgent(refl_db)
    monkeypatch.setattr(agent, "_make_cosine_similarity_fn", broken_similarity_fn)

    report = asyncio.run(agent.run_full())

    assert "boom" in report["fact_merge"]["error"]
    stored = refl_db.execute(
        "SELECT phase_errors FROM reflection_reports WHERE id=?", (report["id"],)
    ).fetchone()
    assert "boom" in json.loads(stored["phase_errors"])["fact_merge"]


def test_fact_merge_disabled_by_default_leaves_phase_errors_clean(refl_db, monkeypatch):
    import json

    import fact_merger
    from reflection.agent import ReflectionAgent, phase_errors

    monkeypatch.delenv("MEMORY_FACT_MERGE_ENABLED", raising=False)

    def fail_if_constructed(*_args, **_kwargs):
        raise AssertionError("FactMerger must not be constructed while disabled")

    monkeypatch.setattr(fact_merger, "FactMerger", fail_if_constructed)

    agent = ReflectionAgent(refl_db)
    report = asyncio.run(agent.run_full())

    assert report["fact_merge"] == {
        "clusters_found": 0,
        "merged": 0,
        "rejected": 0,
        "disabled": True,
    }
    assert "fact_merge" not in phase_errors(report)
    stored = refl_db.execute(
        "SELECT phase_errors FROM reflection_reports WHERE id=?", (report["id"],)
    ).fetchone()
    assert "fact_merge" not in json.loads(stored["phase_errors"])


def test_fact_merge_enabled_via_env_attempts_clustering(refl_db, monkeypatch):
    from reflection.agent import ReflectionAgent

    monkeypatch.setenv("MEMORY_FACT_MERGE_ENABLED", "true")

    agent = ReflectionAgent(refl_db)
    monkeypatch.setattr(agent, "_make_cosine_similarity_fn", lambda: (lambda a, b: 0.9))
    monkeypatch.setattr(agent, "_make_llm_merge_fn", lambda: (lambda contents: "merged"))

    stats = agent._run_fact_merger()

    assert "disabled" not in stats
    assert "clusters_found" in stats


def test_save_report_without_phase_errors_column_still_inserts_row(monkeypatch):
    import sqlite3

    from reflection.agent import ReflectionAgent

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE reflection_reports (
            id TEXT PRIMARY KEY,
            period_start TEXT NOT NULL,
            period_end TEXT NOT NULL,
            type TEXT NOT NULL CHECK (type IN ('session', 'periodic', 'weekly', 'manual')),
            new_nodes INTEGER DEFAULT 0,
            patterns_found INTEGER DEFAULT 0,
            skills_refined INTEGER DEFAULT 0,
            rules_proposed INTEGER DEFAULT 0,
            contradictions INTEGER DEFAULT 0,
            archived INTEGER DEFAULT 0,
            focus_areas JSON,
            key_findings JSON,
            proposed_changes JSON,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
        );
        """
    )

    agent = ReflectionAgent(conn)
    warnings: list[str] = []
    monkeypatch.setattr("reflection.agent.LOG", warnings.append)

    report_id = agent._save_report(
        {"id": "rep1", "type": "periodic", "digest": {"error": "boom"}}
    )

    assert report_id == "rep1"
    row = conn.execute(
        "SELECT id FROM reflection_reports WHERE id=?", ("rep1",)
    ).fetchone()
    assert row is not None
    assert any("029" in w and "boom" in w for w in warnings)


@pytest.mark.parametrize(
    "report,expected",
    [
        ({"digest": {"error": "boom"}}, {"digest": "error: boom"}),
        ({"triple_extraction": {"deferred": "no_llm"}}, {"triple_extraction": "deferred: no_llm"}),
        ({"triple_extraction": {"skipped": 4}}, {}),
        (
            {"representations": {"skipped": 0, "skipped_reason": "no embedder"}},
            {"representations": "skipped_reason: no embedder"},
        ),
        (
            {"fact_merge": {"error": "e1", "deferred": "d1", "skipped": "s1"}},
            {"fact_merge": "error: e1"},
        ),
        (
            {"representations": {"skipped_reason": "no embedder", "skipped": "s1"}},
            {"representations": "skipped_reason: no embedder"},
        ),
        ({"digest": "not a dict"}, {}),
        ({"some_unknown_key": {"error": "x"}}, {}),
    ],
)
def test_phase_errors_markers(report, expected):
    from reflection.agent import phase_errors

    assert phase_errors(report) == expected


def test_merge_hook_embeds_and_queues_merged_record(refl_db, monkeypatch):
    from fact_merger import FactMerger
    from reflection.agent import ReflectionAgent

    monkeypatch.setenv("MEMORY_EMBED_PROVIDER", "fastembed")
    monkeypatch.setenv("FASTEMBED_MODEL", "test-fastembed-model")

    ids = []
    for content in ("User uses Go for backend", "User builds APIs in Go"):
        cur = refl_db.execute(
            "INSERT INTO knowledge (session_id, type, content, project, status, created_at) "
            "VALUES ('s1', 'fact', ?, 'demo', 'active', '2026-04-14T00:00:00Z')",
            (content,),
        )
        ids.append(cur.lastrowid)
    refl_db.commit()

    agent = ReflectionAgent(refl_db, embedder=lambda text: [0.5, 0.25, 0.125])
    merger = FactMerger(
        refl_db,
        similarity_fn=lambda *_: 0.8,
        llm_merge_fn=lambda contents: "User writes Go backends and APIs.",
        on_merged=agent._make_merge_hook(),
    )
    merged_id = merger.merge_cluster(ids)["merged_id"]

    embedding = refl_db.execute(
        "SELECT embed_dim, embed_model, embedding_provider FROM embeddings WHERE knowledge_id=?",
        (merged_id,),
    ).fetchone()
    assert embedding["embed_dim"] == 3
    assert embedding["embed_model"] == "test-fastembed-model"
    assert embedding["embedding_provider"] == "fastembed"

    round_tripped_vector = agent._make_vectors_fn()([merged_id])
    assert list(round_tripped_vector[merged_id]) == [0.5, 0.25, 0.125]

    queued = refl_db.execute(
        "SELECT status FROM representations_queue WHERE knowledge_id=?", (merged_id,)
    ).fetchone()
    assert queued["status"] == "pending"


def test_merge_hook_prefers_injected_embed_identity_over_config(refl_db, monkeypatch):
    from fact_merger import FactMerger
    from reflection.agent import ReflectionAgent

    monkeypatch.setenv("MEMORY_EMBED_PROVIDER", "openai")
    monkeypatch.setenv("FASTEMBED_MODEL", "config-fastembed-model")

    ids = []
    for content in ("User uses Go for backend", "User builds APIs in Go"):
        cur = refl_db.execute(
            "INSERT INTO knowledge (session_id, type, content, project, status, created_at) "
            "VALUES ('s1', 'fact', ?, 'demo', 'active', '2026-04-14T00:00:00Z')",
            (content,),
        )
        ids.append(cur.lastrowid)
    refl_db.commit()

    agent = ReflectionAgent(
        refl_db,
        embedder=lambda text: [0.5, 0.25, 0.125],
        embed_identity=("injected-model", "injected-provider"),
    )
    merger = FactMerger(
        refl_db,
        similarity_fn=lambda *_: 0.8,
        llm_merge_fn=lambda contents: "User writes Go backends and APIs.",
        on_merged=agent._make_merge_hook(),
    )
    merged_id = merger.merge_cluster(ids)["merged_id"]

    embedding = refl_db.execute(
        "SELECT embed_model, embedding_provider FROM embeddings WHERE knowledge_id=?",
        (merged_id,),
    ).fetchone()
    assert embedding["embed_model"] == "injected-model"
    assert embedding["embedding_provider"] == "injected-provider"


def test_merge_hook_absent_without_embedder(refl_db, monkeypatch):
    import server as _srv
    from fact_merger import FactMerger
    from reflection.agent import ReflectionAgent

    def raise_store(*args, **kwargs):
        raise RuntimeError("no store in this environment")

    monkeypatch.setattr(_srv, "Store", raise_store)

    agent = ReflectionAgent(refl_db, embedder=None)
    hook = agent._make_merge_hook()
    assert hook is None

    a = refl_db.execute(
        "INSERT INTO knowledge (session_id, type, content, project, status, created_at) "
        "VALUES ('s1', 'fact', 'fact a', 'demo', 'active', '2026-04-14T00:00:00Z')"
    ).lastrowid
    b = refl_db.execute(
        "INSERT INTO knowledge (session_id, type, content, project, status, created_at) "
        "VALUES ('s1', 'fact', 'fact b', 'demo', 'active', '2026-04-14T00:00:00Z')"
    ).lastrowid
    refl_db.commit()

    merger = FactMerger(
        refl_db,
        similarity_fn=lambda *_: 0.8,
        llm_merge_fn=lambda contents: "merged fact",
        on_merged=hook,
    )
    result = merger.merge_cluster([a, b])

    assert result["merged_id"] is not None
    row = refl_db.execute(
        "SELECT status FROM knowledge WHERE id=?", (result["merged_id"],)
    ).fetchone()
    assert row["status"] == "active"
    assert (
        refl_db.execute(
            "SELECT 1 FROM embeddings WHERE knowledge_id=?", (result["merged_id"],)
        ).fetchone()
        is None
    )


def test_phase_errors_omits_healthy_phases(refl_db, monkeypatch):
    import json

    import config as config_mod
    import server as server_mod
    from reflection.agent import ReflectionAgent

    monkeypatch.setattr(config_mod, "has_llm", lambda *_a, **_kw: False)
    monkeypatch.setenv("MEMORY_FACT_MERGE_ENABLED", "true")

    def raise_store(*_args, **_kwargs):
        raise RuntimeError("no store in this environment")

    monkeypatch.setattr(server_mod, "Store", raise_store)

    report = asyncio.run(ReflectionAgent(refl_db).run_full())
    stored = json.loads(
        refl_db.execute(
            "SELECT phase_errors FROM reflection_reports WHERE id=?", (report["id"],)
        ).fetchone()["phase_errors"]
    )

    assert stored == {
        "triple_extraction": "deferred: no_llm",
        "fact_merge": "skipped: deps",
        "deep_enrichment": "deferred: no_llm",
        "representations": "skipped_reason: no embedder",
    }


def test_make_vectors_fn_loads_real_embeddings_across_chunk_boundary(refl_db):
    from reflection.agent import ReflectionAgent

    dim = 3
    ids = list(range(1, 502))
    for kid in ids:
        refl_db.execute(
            "INSERT INTO embeddings "
            "(knowledge_id, binary_vector, float32_vector, embed_model, embed_dim, created_at) "
            "VALUES (?, ?, ?, 'test-model', ?, '2026-04-14T00:00:00Z')",
            (kid, b"", struct.pack(f"{dim}f", float(kid), 0.0, 1.0), dim),
        )
    refl_db.commit()

    vectors_fn = ReflectionAgent(refl_db)._make_vectors_fn()
    loaded = vectors_fn(ids)

    assert set(loaded.keys()) == set(ids)
    assert all(len(v) == dim for v in loaded.values())
    assert list(loaded[1]) == [1.0, 0.0, 1.0]
    assert list(loaded[501]) == [501.0, 0.0, 1.0]


def test_run_fact_merger_wires_hook_and_vectors_fn(refl_db, monkeypatch):
    from reflection.agent import ReflectionAgent

    monkeypatch.setenv("MEMORY_FACT_MERGE_ENABLED", "true")

    ids = []
    for content in ("User uses Go for backend", "User builds APIs in Go"):
        cur = refl_db.execute(
            "INSERT INTO knowledge (session_id, type, content, project, status, created_at) "
            "VALUES ('s1', 'fact', ?, 'demo', 'active', '2026-04-14T00:00:00Z')",
            (content,),
        )
        ids.append(cur.lastrowid)
    refl_db.commit()

    agent = ReflectionAgent(refl_db, embedder=lambda text: [0.1, 0.2])
    monkeypatch.setattr(agent, "_make_cosine_similarity_fn", lambda: (lambda a, b: 0.9))
    monkeypatch.setattr(
        agent, "_make_llm_merge_fn", lambda: (lambda contents: "User writes Go backends and APIs.")
    )

    stats = agent._run_fact_merger()

    assert stats["merged"] == 1
    merged_row = refl_db.execute(
        "SELECT id FROM knowledge WHERE source='merged'"
    ).fetchone()
    assert merged_row is not None
    merged_id = merged_row["id"]

    assert (
        refl_db.execute(
            "SELECT 1 FROM embeddings WHERE knowledge_id=?", (merged_id,)
        ).fetchone()
        is not None
    )
    queued = refl_db.execute(
        "SELECT status FROM representations_queue WHERE knowledge_id=?", (merged_id,)
    ).fetchone()
    assert queued["status"] == "pending"
