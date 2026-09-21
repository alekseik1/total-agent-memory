import json
import sqlite3
from pathlib import Path

import pytest

from ai_layer.atomic_fact_extractor import FactExtractor
from memory_core.atomic_facts import (
    FactRepository,
    InvalidExtraction,
    SourceChanged,
)
from memory_core.retrieval import SearchScope


@pytest.fixture
def repo():
    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    db.executescript("""
        CREATE TABLE knowledge(id INTEGER PRIMARY KEY, content TEXT, project TEXT,
        session_id TEXT, created_at TEXT DEFAULT '', branch TEXT DEFAULT '', status TEXT DEFAULT 'active', type TEXT DEFAULT 'fact');
        CREATE TABLE embeddings(knowledge_id INTEGER, embedding_space TEXT);
        INSERT INTO knowledge(id,content,project,session_id) VALUES
        (1,'Alice adopted a rabbit named Pip.','a','session'),
        (2,'She feeds him carrots every evening.','a','session'),
        (3,'SECRET Bob owns a hamster.','b','session');
    """)
    db.executescript(
        (Path(__file__).parents[1] / "migrations/029_atomic_facts.sql").read_text()
    )
    db.executescript(
        (Path(__file__).parents[1] / "migrations/020_async_enrichment.sql").read_text()
    )
    db.executescript(
        (
            Path(__file__).parents[1] / "migrations/030_evidence_lifecycle.sql"
        ).read_text()
    )
    yield FactRepository(db)
    db.close()


def response():
    return {
        "facts": [
            {
                "subject": "Alice",
                "predicate": "feeds Pip",
                "object": "carrots",
                "temporal_text": "every evening",
                "sources": [
                    {"id": 1, "quote": "Alice adopted a rabbit named Pip."},
                    {"id": 2, "quote": "She feeds him carrots every evening."},
                ],
            }
        ]
    }


class Provider:
    def __init__(self, data=None, hook=None):
        self.data = response() if data is None else data
        self.hook = hook
        self.calls = []

    def complete(self, prompt, **kwargs):
        self.calls.append(prompt)
        if self.hook:
            self.hook()
        return json.dumps(self.data)


def test_extract_links_references_without_replacing_original(repo):
    provider = Provider()
    extractor = FactExtractor(repo, provider, "test-model")
    assert extractor.extract(2) == 1
    assert "SECRET" not in provider.calls[0]
    assert extractor.extract(2) == 0
    assert len(provider.calls) == 1
    hits = repo.search("Pip carrots", SearchScope(project="a"))
    assert {hit["id"] for hit in hits} == {1, 2}
    assert hits[0]["content"] == "She feeds him carrots every evening."
    assert repo.search("Pip", SearchScope(project="b")) == []


@pytest.mark.parametrize("change", ["quote", "id", "target", "field"])
def test_invalid_grounding_writes_nothing(repo, change):
    data = response()
    if change == "quote":
        data["facts"][0]["sources"][0]["quote"] = "fabricated quote"
    elif change == "id":
        data["facts"][0]["sources"][0]["id"] = 3
    elif change == "target":
        data["facts"][0]["sources"].pop()
    else:
        data["facts"][0]["subject"] = []
    with pytest.raises(InvalidExtraction):
        FactExtractor(repo, Provider(data), "test").extract(2)
    assert repo.db.execute("SELECT COUNT(*) FROM atomic_facts").fetchone()[0] == 0
    assert repo.db.execute("SELECT COUNT(*) FROM atomic_fact_runs").fetchone()[0] == 0


@pytest.mark.parametrize(
    "mutation",
    [
        "UPDATE knowledge SET content='Changed' WHERE id=1",
        "UPDATE knowledge SET status='deleted' WHERE id=2",
        "UPDATE knowledge SET project='b' WHERE id=1",
        "UPDATE knowledge SET branch='private' WHERE id=1",
        "DELETE FROM knowledge WHERE id=1",
    ],
)
def test_source_changes_remove_fact_and_search_index(repo, mutation):
    FactExtractor(repo, Provider(), "test").extract(2)
    repo.db.execute(mutation)
    assert repo.search("carrots", SearchScope()) == []
    for table in (
        "atomic_facts",
        "atomic_facts_fts",
        "atomic_fact_sources",
        "atomic_fact_runs",
    ):
        assert repo.db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0


def test_changed_source_during_provider_call_rejects_commit(repo):
    provider = Provider(
        hook=lambda: repo.db.execute(
            "UPDATE knowledge SET content='Changed' WHERE id=1"
        )
    )
    with pytest.raises(SourceChanged):
        FactExtractor(repo, provider, "test").extract(2)
    assert repo.db.execute("SELECT COUNT(*) FROM atomic_facts").fetchone()[0] == 0


def test_recall_usage_does_not_invalidate_facts(repo):
    FactExtractor(repo, Provider(), "test").extract(2)
    repo.db.execute("ALTER TABLE knowledge ADD COLUMN recall_count INTEGER DEFAULT 0")
    repo.db.execute("UPDATE knowledge SET recall_count=recall_count+1")
    assert repo.search("carrots", SearchScope(project="a"))


def test_empty_extraction_is_idempotent(repo):
    provider = Provider({"facts": []})
    extractor = FactExtractor(repo, provider, "test")
    assert extractor.extract(2) == 0
    assert extractor.extract(2) == 0
    assert len(provider.calls) == 1
    assert repo.pending("a", 10) == [1]


def test_spaces_filter_before_candidate_limit(repo):
    FactExtractor(repo, Provider(), "test").extract(2)
    repo.db.execute("INSERT INTO embeddings VALUES (2,'text')")
    assert (
        repo.search("carrots", SearchScope(project="a", spaces="code"), limit=1) == []
    )
    assert (
        repo.search("carrots", SearchScope(project="a", spaces="text"), limit=1)[0][
            "id"
        ]
        == 2
    )


def test_background_stage_runs_configured_extractor(repo, monkeypatch):
    from types import SimpleNamespace

    import enrichment_worker
    import llm_provider

    provider = Provider()
    monkeypatch.setenv("MEMORY_ATOMIC_FACTS_ENABLED", "true")
    monkeypatch.setattr(llm_provider, "make_provider", lambda name: provider)
    enrichment_worker._run_atomic_facts(repo.db, SimpleNamespace(knowledge_id=2))
    assert repo.search("carrots", SearchScope(project="a"))


def test_validation_retry_is_bounded_and_can_recover(repo):
    class RetryProvider:
        calls = 0

        def complete(self, prompt, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return '{"facts":[{"subject":"invalid"}]}'
            assert "Validation failed" in prompt
            return json.dumps(response())

    provider = RetryProvider()
    assert FactExtractor(repo, provider, "test").extract(2) == 1
    assert provider.calls == 2


def test_persistent_invalid_output_stops_after_two_calls(repo):
    provider = Provider({"facts": "invalid"})
    with pytest.raises(InvalidExtraction):
        FactExtractor(repo, provider, "test").extract(2)
    assert len(provider.calls) == 2


def test_background_disabled_never_builds_provider(repo, monkeypatch):
    from types import SimpleNamespace

    import enrichment_worker
    import llm_provider

    def forbidden(name):
        raise AssertionError("Unexpected LLM setup")

    monkeypatch.setenv("MEMORY_ATOMIC_FACTS_ENABLED", "false")
    monkeypatch.setattr(llm_provider, "make_provider", forbidden)
    enrichment_worker._run_atomic_facts(repo.db, SimpleNamespace(knowledge_id=2))


def test_duplicate_facts_do_not_inflate_persisted_count(repo):
    data = response()
    data["facts"].append(dict(data["facts"][0]))
    assert FactExtractor(repo, Provider(data), "test").extract(2) == 1
    assert repo.db.execute("SELECT fact_count FROM atomic_fact_runs").fetchone()[0] == 1


def test_backfill_uses_current_schema_and_is_resumable(repo, tmp_path, monkeypatch):
    import atomic_facts_backfill

    path = tmp_path / "memory.db"
    repo.db.commit()
    with sqlite3.connect(path) as copied:
        repo.db.backup(copied)
    provider = Provider({"facts": []})
    monkeypatch.setattr(atomic_facts_backfill, "make_provider", lambda name: provider)
    assert atomic_facts_backfill.backfill(path, "a", 10) == (2, 0)
    assert atomic_facts_backfill.backfill(path, "a", 10) == (0, 0)
    assert len(provider.calls) == 2


def test_package_reports_the_runtime_version():
    import total_agent_memory
    from src.version import VERSION

    assert total_agent_memory.__version__ == VERSION


def test_fact_sources_are_loaded_in_two_queries_with_any_matching_space(repo):
    FactExtractor(repo, Provider(), "test").extract(2)
    repo.db.executemany(
        "INSERT INTO embeddings VALUES (?,?)", [(1, "code"), (1, "text"), (2, "text")]
    )
    statements = []
    repo.db.set_trace_callback(statements.append)
    hits = repo.search("carrots", SearchScope(project="a", spaces="text"))
    repo.db.set_trace_callback(None)
    assert [hit["id"] for hit in hits] == [2, 1]
    assert len([sql for sql in statements if sql.startswith("SELECT")]) == 2


def test_empty_extraction_tracks_context_and_requeues_after_change(repo):
    FactExtractor(repo, Provider({"facts": []}), "test").extract(2)
    repo.db.execute(
        "UPDATE knowledge SET content='Alice renamed her rabbit.' WHERE id=1"
    )
    assert (
        repo.db.execute("SELECT knowledge_id FROM atomic_fact_rebuild").fetchone()[0]
        == 2
    )
    assert not repo.completed(repo.sources(2)[-1])
    FactExtractor(repo, Provider({"facts": []}), "test").extract(2)
    assert not repo.db.execute("SELECT * FROM atomic_fact_rebuild").fetchall()


def test_noop_source_update_does_not_invalidate(repo):
    FactExtractor(repo, Provider(), "test").extract(2)
    repo.db.execute("UPDATE knowledge SET content=content WHERE id=1")
    assert repo.completed(repo.sources(2)[-1])


def test_changed_observation_date_invalidates_event(repo):
    repo.db.execute("UPDATE knowledge SET created_at='2026-02-03T10:00:00Z' WHERE id=2")
    data = response()
    repo.db.execute(
        "UPDATE knowledge SET content='She fed him carrots yesterday.' WHERE id=2"
    )
    data["facts"][0]["sources"][1]["quote"] = "She fed him carrots yesterday."
    data["facts"][0]["temporal_text"] = "yesterday"
    FactExtractor(repo, Provider(data), "test").extract(2)
    row = repo.db.execute(
        "SELECT observed_at,event_start,event_end,event_precision,event_key FROM atomic_facts"
    ).fetchone()
    assert tuple(row[:4]) == ("2026-02-03T10:00:00Z", "2026-02-02", "2026-02-02", "day")
    assert len(row[4]) == 64
    repo.db.execute("UPDATE knowledge SET created_at='2026-02-04T10:00:00Z' WHERE id=2")
    assert repo.db.execute("SELECT count(*) FROM atomic_facts").fetchone()[0] == 0


def test_evidence_chain_recovers_bridge_without_replacing_anchor(repo):
    from memory_core.evidence_chains import EvidenceChains

    FactExtractor(repo, Provider(), "test").extract(2)
    hits = [{"id": 2, "content": "She feeds him carrots every evening."}]
    expanded = EvidenceChains(repo.db).expand(hits, SearchScope(project="a"))
    assert [hit["id"] for hit in expanded] == [2, 1]
    assert expanded[0]["evidence_ids"] == [1, 2]
    assert "evidence_ids" not in hits[0]
    assert EvidenceChains(repo.db).expand(hits, SearchScope(project="b")) == []
    assert (
        EvidenceChains(repo.db).expand(hits, SearchScope(project="a"), max_sources=0)
        == hits
    )


def test_rebuild_enqueues_only_current_active_snapshot(repo):
    from memory_core.leases import enqueue_rebuilds

    repo.db.execute("ALTER TABLE knowledge ADD COLUMN tags TEXT DEFAULT '[]'")
    repo.db.execute("ALTER TABLE knowledge ADD COLUMN importance TEXT DEFAULT 'medium'")
    FactExtractor(repo, Provider(), "test").extract(2)
    repo.db.execute(
        "UPDATE knowledge SET content='Alice adopted another rabbit.' WHERE id=1"
    )
    assert enqueue_rebuilds(repo.db, 10, lambda: "2026-02-03T10:00:00Z") == 1
    row = repo.db.execute(
        "SELECT knowledge_id,atomic_only,content_snapshot FROM enrichment_queue"
    ).fetchone()
    assert tuple(row) == (2, 1, "She feeds him carrots every evening.")
    assert enqueue_rebuilds(repo.db, 10, lambda: "2026-02-03T10:00:00Z") == 0


def test_lost_lease_rejects_extraction_commit(repo):
    from memory_core.leases import LeaseLost

    def guard():
        raise LeaseLost("Lease transferred to another worker")

    with pytest.raises(LeaseLost):
        FactExtractor(repo, Provider(), "test", before_commit=guard).extract(2)
    assert repo.db.execute("SELECT count(*) FROM atomic_facts").fetchone()[0] == 0
    assert repo.db.execute("SELECT count(*) FROM atomic_fact_runs").fetchone()[0] == 0


def test_relative_date_uses_cited_message_date_not_later_target(repo):
    repo.db.execute(
        "UPDATE knowledge SET content='Alice adopted a rabbit yesterday.', "
        "created_at='2026-02-03' WHERE id=1"
    )
    repo.db.execute(
        "UPDATE knowledge SET content='That adoption mattered.', "
        "created_at='2026-03-10' WHERE id=2"
    )
    data = {
        "facts": [
            {
                "subject": "Alice",
                "predicate": "adopted",
                "object": "a rabbit",
                "temporal_text": "yesterday",
                "sources": [
                    {"id": 1, "quote": "Alice adopted a rabbit yesterday."},
                    {"id": 2, "quote": "That adoption mattered."},
                ],
            }
        ]
    }
    FactExtractor(repo, Provider(data), "test").extract(2)
    row = repo.db.execute(
        "SELECT observed_at,temporal_anchor_at,event_start FROM atomic_facts"
    ).fetchone()
    assert tuple(row) == ("2026-03-10", "2026-02-03", "2026-02-02")
