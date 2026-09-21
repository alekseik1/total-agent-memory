import json
import sqlite3
from pathlib import Path

import pytest

from memory_core.retrieval import SearchScope, flatten_results
from temporal_kg import TemporalKG


def test_grouped_retrieval_preserves_evidence():
    hits = [{"id": 1, "content": "Docker required"}, {"id": 2, "content": "Use Go"}]
    assert flatten_results({"results": {"rule": hits[:1], "fact": hits[1:]}}) == hits
    with pytest.raises(ValueError):
        flatten_results({"results": {"fact": "invalid"}})


def test_scope_excludes_graph_neighbor_and_wrong_branch():
    with sqlite3.connect(":memory:") as db:
        scope = SearchScope(project="a", branch="main")
        assert scope.allows({"project": "a", "branch": ""}, db)
        assert not scope.allows({"project": "b", "branch": "main"}, db)
        assert not scope.allows({"project": "a", "branch": "other"}, db)
        assert not scope.allows({"project": "a", "status": "deleted"}, db)


def test_outbox_discards_completed_payload():
    import outbox
    with sqlite3.connect(":memory:") as db:
        db.executescript((Path(__file__).parents[1] / "migrations/017_outbox.sql").read_text())
        intent = outbox.create_intent(db, payload={"content": "record"},
                                     session_id="s", content="record", ktype="fact", project="p")
        outbox.mark_committed(db, intent, 1)
        assert json.loads(db.execute("SELECT payload_json FROM write_intents").fetchone()[0]) == {}


def test_backdated_fact_does_not_replace_newer_state():
    with sqlite3.connect(":memory:") as db:
        db.row_factory = sqlite3.Row
        db.executescript((Path(__file__).parents[1] / "migrations/008_temporal_kg.sql").read_text())
        kg = TemporalKG(db)
        newer = kg.add_fact("person", "job", "engineer", valid_from="2025-06-01T00:00:00Z")
        older = kg.add_fact("person", "job", "student", valid_from="2024-06-01T00:00:00Z")
        assert [row["id"] for row in kg.get_current()] == [newer]
        old = db.execute("SELECT * FROM fact_assertions WHERE id=?", (older,)).fetchone()
        assert old["valid_to"] >= old["valid_from"]
        assert old["created_at"] > old["valid_from"]


def test_context_budget_includes_source_metadata_and_unicode():
    from memory_core.context import bound_context
    bundle = {"rules": [{"id": 1, "content": "Use Docker"}],
              "knowledge": [{"id": 2, "content": "память🧠" * 200}],
              "episodes": [], "skills": [], "blind_spots": [],
              "competency": {"content": "x" * 200}, "total_tokens": 0}
    result = bound_context(bundle, 100)
    assert result["total_tokens"] <= 100
    assert result["rules"][0]["source_ref"] == "rules:1"
    assert result["knowledge"] == [] and result["competency"] is None
    assert result["omitted_items"] == 2
    assert bundle["knowledge"]
    assert bound_context(bundle, 0)["total_tokens"] == 0
    with pytest.raises(ValueError):
        bound_context(bundle, -1)


def test_backdated_middle_assertion_splits_existing_interval():
    with sqlite3.connect(":memory:") as db:
        db.row_factory = sqlite3.Row
        db.executescript((Path(__file__).parents[1] / "migrations/008_temporal_kg.sql").read_text())
        kg = TemporalKG(db)
        for value, date in [("junior", "2023"), ("senior", "2025"), ("middle", "2024")]:
            kg.add_fact("person", "job", value, valid_from=f"{date}-01-01T00:00:00Z")
        assert [r["object"] for r in kg.query_at("2024-07-01T02:00:00+02:00")] == ["middle"]
        assert [r["object"] for r in kg.get_current()] == ["senior"]


@pytest.fixture
def real_store(tmp_path, monkeypatch):
    import server
    monkeypatch.setattr(server, "MEMORY_DIR", tmp_path)
    monkeypatch.setattr(server, "HAS_CHROMA", False)
    monkeypatch.setattr(server.Store, "_init_embed_mode", lambda self: "none")
    monkeypatch.setattr(server.Store, "_check_embed_dim_compat", lambda self: None)
    monkeypatch.setattr(server.Store, "embed", lambda self, texts: [])
    monkeypatch.setenv("MEMORY_ASYNC_ENRICHMENT", "false")
    store = server.Store()
    store.session_start("a", project="alpha")
    store.session_start("b", project="beta")
    yield store
    store.db.close()


def test_privacy_applies_before_outbox_payload(real_store, monkeypatch):
    import outbox
    original = outbox.create_intent
    captured = []
    def capture(*args, **kwargs):
        captured.append(kwargs["payload"])
        return original(*args, **kwargs)
    monkeypatch.setattr(outbox, "create_intent", capture)
    real_store.save_knowledge("a", "Docker isolates tasks <private>hidden-token</private>",
                              "fact", project="alpha", context="<private>hidden-context", skip_quality=True)
    assert captured
    assert "hidden-" not in json.dumps(captured)
    assert "hidden-" not in "\n".join(real_store.db.iterdump())


def test_recall_and_export_keep_project_boundary(real_store):
    import server
    alpha = real_store.save_knowledge("a", "Docker isolates build tasks", "fact", project="alpha", skip_quality=True)[0]
    beta = real_store.save_knowledge("b", "Secret customer account", "fact", project="beta", skip_quality=True)[0]
    real_store.add_relation(alpha, beta, "related")
    recall = server.Recall(real_store)
    hits = flatten_results(recall.search("Docker", project="alpha", detail="full", record_usage=False))
    assert hits and {hit["id"] for hit in hits} == {alpha}
    exported = real_store.export_all(project="alpha")
    assert all(s["project"] == "alpha" for s in exported["sessions"])
    assert not exported["relations"]


def test_external_delete_invalidates_recall_cache(real_store):
    import server
    kid = real_store.save_knowledge("a", "Docker isolates build tasks", "fact", project="alpha", skip_quality=True)[0]
    recall = server.Recall(real_store)
    assert flatten_results(recall.search("Docker", project="alpha", record_usage=False))
    with sqlite3.connect(real_store.db_path) as worker:
        worker.execute("UPDATE knowledge SET status='deleted' WHERE id=?", (kid,))
    assert not flatten_results(recall.search("Docker", project="alpha", record_usage=False))


def test_recall_does_not_verify_knowledge(real_store):
    kid = real_store.save_knowledge("a", "Docker isolates build tasks", "fact", project="alpha", skip_quality=True)[0]
    real_store.db.execute("UPDATE knowledge SET last_confirmed='2020-01-01T00:00:00Z' WHERE id=?", (kid,))
    real_store.db.commit()
    real_store.bump_recall([kid])
    row = real_store.q1("SELECT last_confirmed, recall_count FROM knowledge WHERE id=?", (kid,))
    assert row["last_confirmed"] == "2020-01-01T00:00:00Z"
    assert row["recall_count"] == 1


def test_atomic_index_is_used_by_public_recall(real_store):
    import server
    from ai_layer.atomic_fact_extractor import FactExtractor
    from memory_core.atomic_facts import FactRepository
    first = real_store.save_knowledge("a", "Alice adopted rabbit Pip.", "fact",
                                     project="alpha", skip_quality=True)[0]
    second = real_store.save_knowledge("a", "She feeds him carrots.", "fact",
                                      project="alpha", skip_quality=True)[0]
    class Provider:
        def complete(self, prompt, **kwargs):
            return json.dumps({"facts": [{"subject": "Alice", "predicate": "feeds Pip",
                "object": "carrots", "temporal_text": "", "sources": [
                    {"id": first, "quote": "Alice adopted rabbit Pip."},
                    {"id": second, "quote": "She feeds him carrots."}]}]})
    FactExtractor(FactRepository(real_store.db), Provider(), "test").extract(second)
    recall = server.Recall(real_store)
    hits = flatten_results(recall.search("Pip carrots", project="alpha", record_usage=False))
    assert any("atomic_facts" in hit["via"] for hit in hits)
    assert {hit["id"] for hit in hits} >= {first, second}
    real_store.db.execute("UPDATE knowledge SET content='Redacted' WHERE id=?", (first,))
    assert real_store.db.execute("SELECT COUNT(*) FROM atomic_facts").fetchone()[0] == 0


def test_mixed_dimensions_use_separate_queries(real_store, monkeypatch):
    class Provider:
        def active_model(self, space):
            return "code-model"
        def embed_query(self, query, *, space):
            assert space == "code"
            return [0.0, 1.0, 0.0]
    real_store._v11_embed_provider = Provider()
    monkeypatch.setattr(real_store, "_active_embed_model_name", lambda: "text-model")
    monkeypatch.setattr(real_store, "embed", lambda texts: [[1.0, 0.0]])
    for content, model, space, vector in [("Docker text", "text-model", "text", [1.0, 0.0]),
                                         ("Python code", "code-model", "code", [0.0, 1.0, 0.0])]:
        kid = real_store.save_knowledge("a", content, "fact", project="alpha", skip_quality=True)[0]
        real_store._upsert_embedding(kid, vector, model_name=model, embedding_space=space)
    hits = real_store._search_spaces("query", project="alpha")
    assert len(hits) == 2
    assert real_store._semantic_diagnostics == []
    monkeypatch.setattr(real_store, "_active_embed_model_name", lambda: "different-text-model")
    assert len(real_store._search_spaces("query", project="alpha")) == 1
    assert real_store._semantic_diagnostics[0]["space"] == "text"


def test_queue_concurrent_claims_are_unique(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    from representations_queue import RepresentationsQueue
    path = tmp_path / "queue.db"
    root = Path(__file__).parents[1]
    with sqlite3.connect(path) as db:
        db.executescript((root / "migrations/002_multi_representation.sql").read_text())
        db.executescript((root / "migrations/005_representations_queue.sql").read_text())
        db.row_factory = sqlite3.Row
        queue = RepresentationsQueue(db)
        for kid in range(8):
            queue.enqueue(kid)
    def claim():
        with sqlite3.connect(path, timeout=5) as db:
            db.row_factory = sqlite3.Row
            return RepresentationsQueue(db).claim_next()
    with ThreadPoolExecutor(max_workers=8) as pool:
        claims = list(pool.map(lambda _: claim(), range(8)))
    assert len({item["id"] for item in claims}) == 8
    assert claim() is None


def test_planner_uses_configured_runtime_provider(monkeypatch):
    from ai_layer import planner_client
    observed = {}
    class Provider:
        def complete(self, prompt, **kwargs):
            observed.update(kwargs)
            return '{"done": true}'
    monkeypatch.setattr(planner_client, "make_provider", lambda name: Provider())
    monkeypatch.setattr(planner_client.config, "get_llm_model_for_provider", lambda: "local-model")
    assert planner_client.PlannerClient().complete(model="configured", system="system", user="question") == '{"done": true}'
    assert observed["model"] == "local-model"
    assert observed["temperature"] == 0.0


def test_cognitive_context_excludes_other_project(real_store):
    from cognitive.engine import CognitiveEngine
    real_store.save_knowledge("a", "Docker project build", "fact", project="alpha", skip_quality=True)
    real_store.save_knowledge("b", "Docker customer secret", "fact", project="beta", skip_quality=True)
    result = CognitiveEngine(real_store.db).build_context("Docker", project="alpha")
    assert result["knowledge"]
    assert {item["project"] for item in result["knowledge"]} == {"alpha"}
    assert result["knowledge"][0]["session_id"] == "a"


def test_benchmark_blind_routing_does_not_read_gold_category(real_store, monkeypatch):
    import importlib.util
    path = Path(__file__).parents[1] / "benchmarks/locomo_bench_llm.py"
    spec = importlib.util.spec_from_file_location("v14_locomo_test", path)
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    calls = []
    class Recall:
        def search(self, **kwargs):
            calls.append(kwargs)
            return {"results": {}}
    monkeypatch.setattr(bench, "call_llm", lambda *args, **kwargs: ("YES", 0, 0))
    for category in (1, 2, 3, 4, 5):
        bench.process_qa(None, None, real_store, Recall(),
                         {"question": "Docker?", "answer": "YES", "category": category},
                         "alpha", top_k=10)
    assert len(calls) == 5 and all(call == calls[0] for call in calls)
    assert calls[0]["limit"] == 10 and calls[0]["record_usage"] is False
    assert bench.CATEGORY_NAMES[1] == "multi-hop"
    assert bench.CATEGORY_NAMES[4] == "single-hop"
    with pytest.raises(ValueError, match="oracle"):
        bench.process_qa(None, None, real_store, Recall(), {}, "alpha", top_k=10, per_cat_prompts=True)


def test_temporal_repeated_value_has_one_current_interval():
    with sqlite3.connect(":memory:") as db:
        db.row_factory = sqlite3.Row
        db.executescript((Path(__file__).parents[1] / "migrations/008_temporal_kg.sql").read_text())
        kg = TemporalKG(db)
        first = kg.add_fact("person", "job", "engineer", valid_from="2024-01-01T00:00:00Z")
        second = kg.add_fact("person", "job", "engineer", valid_from="2025-01-01T00:00:00Z")
        assert first != second
        assert [r["id"] for r in kg.get_current()] == [second]
        assert kg.add_fact("person", "job", "engineer", valid_from="2024-01-01T00:00:00Z") == first
