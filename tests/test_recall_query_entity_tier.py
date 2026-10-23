"""query_entity tier: records linked to graph entities named in the query, IDF-weighted."""

from __future__ import annotations

import math

import pytest

import config
import server
from memory_core.retrieval_tiers import (
    QUERY_ENTITY_IDF_OFFSET,
    QUERY_ENTITY_MULTI_BONUS,
    query_entity_candidates,
    query_entity_names,
)
from tests.pg_store_support import store_backend  # noqa: F401 — fixture


@pytest.fixture(autouse=True)
def _fresh_weight_cache():
    config._tier_weights_cache.clear()
    yield
    config._tier_weights_cache.clear()


def test_names_cover_case_variants_and_pairs():
    names = query_entity_names("What did Alice say about windows vista?")
    assert "Alice" in names and "alice" in names
    assert "windows vista" in names and "Windows vista" in names
    assert all(len(name) >= 3 for name in names)
    assert query_entity_names("") == []


@pytest.fixture
def graph_store(store_backend, tmp_path, monkeypatch):  # noqa: F811 — pytest fixture injection
    monkeypatch.setattr(server, "MEMORY_DIR", tmp_path)
    monkeypatch.setenv("MEMORY_CROSS_RERANK", "off")
    store = server.Store(database=store_backend)
    store._embed_mode = "none"
    contents = {
        1: "kickoff call notes",                      # alice + kafka
        2: "retro notes from friday",                 # alice
        3: "queue migration plan",                    # kafka
        4: "standup summary",                         # bob (popular)
        5: "lunch order",                             # bob
        6: "office wifi password rotation",           # bob
        7: "archived note",                           # alice, inactive
        8: "other project note",                      # alice, other project
    }
    for kid, content in contents.items():
        store.db.execute(
            "INSERT INTO knowledge(id,content,session_id,project,type,status,created_at,last_confirmed) "
            "VALUES (?,?,'s',?,'fact',?,'2026-01-01','2026-01-01')",
            (kid, content, "other" if kid == 8 else "p", "archived" if kid == 7 else "active"))
    nodes = (("n-alice", "person", "Alice"), ("n-kafka", "technology", "kafka"), ("n-bob", "person", "bob"),
             ("n-proj", "project", "alice"))
    for node_id, node_type, name in nodes:
        store.db.execute("INSERT INTO graph_nodes(id,type,name,status,first_seen_at,last_seen_at) "
                         "VALUES (?,?,?,'active','2026-01-01','2026-01-01')", (node_id, node_type, name))
    links = [(1, "n-alice"), (1, "n-kafka"), (2, "n-alice"), (3, "n-kafka"), (4, "n-bob"), (5, "n-bob"),
             (6, "n-bob"), (7, "n-alice"), (8, "n-alice"), (6, "n-proj")]
    for kid, node_id in links:
        store.db.execute("INSERT INTO knowledge_nodes(knowledge_id,node_id,role,strength) VALUES (?,?,'mentions',1.0)",
                         (kid, node_id))
    store.db.commit()
    yield store
    store.db.close()


def _idf(mentions: int) -> float:
    return 1.0 / math.log(QUERY_ENTITY_IDF_OFFSET + mentions)


def test_scores_sum_idf_with_multi_entity_bonus(graph_store):
    scored = dict(query_entity_candidates(graph_store, "What did Alice say about Kafka?", project="p", ktype="all",
                                          branch=None, limit=10))
    # alice links 4 records (1, 2, 7, 8), kafka 2 (1, 3); mentions count every linked record.
    both = (_idf(4) + _idf(2)) * (1 + QUERY_ENTITY_MULTI_BONUS)
    assert scored[1] == pytest.approx(both / (1 + both))
    assert scored[3] == pytest.approx(_idf(2) / (1 + _idf(2)))
    assert scored[2] == pytest.approx(_idf(4) / (1 + _idf(4)))
    assert set(scored) == {1, 2, 3}, "inactive and other-project records and project nodes stay out"
    assert max(scored, key=scored.get) == 1
    assert scored[3] > scored[2], "the rarer entity weighs more"


def test_hub_entities_are_skipped(graph_store, monkeypatch):
    monkeypatch.setenv("MEMORY_GRAPH_HUB_DEGREE", "2")
    scored = dict(query_entity_candidates(graph_store, "bob and kafka", project="p", ktype="all", branch=None,
                                          limit=10))
    assert set(scored) == {1, 3}


def test_no_entities_no_candidates(graph_store):
    assert query_entity_candidates(graph_store, "weather today", project="p", ktype="all", branch=None,
                                   limit=10) == []


def test_off_by_default():
    assert config.get_recall_tier_weights()["query_entity"] == 0.0


def test_tier_fuses_entity_records(graph_store):
    result = server.Recall(graph_store).search("What did Alice say about Kafka?", project="p", limit=5,
                                               record_usage=False, _explain=True,
                                               tier_weights={"query_entity": 1.0})
    assert "query_entity" in result["_explain"]["tiers_used"]
    assert [pair["id"] for pair in result["_explain"]["query_entity"]][0] == 1
    assert result["_explain"]["merged"][0]["id"] == 1


def test_weight_zero_skips_the_tier(graph_store):
    result = server.Recall(graph_store).search("What did Alice say about Kafka?", project="p", limit=5,
                                               record_usage=False, _explain=True,
                                               tier_weights={"query_entity": 0})
    assert "query_entity" not in result["_explain"]["tiers_used"]
