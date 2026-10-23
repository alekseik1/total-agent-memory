"""Compound queries: each part's best record is guaranteed a place in the top K."""

from __future__ import annotations

import pytest

import config
import server
from memory_core.query_shapes import compound_parts
from memory_core.retrieval_tiers import guarantee_slots


@pytest.mark.parametrize("query, parts", [
    ("What database do we use and how do we deploy the billing service?",
     ["What database do we use", "how do we deploy the billing service"]),
    ("Postgres version, also the Redis eviction policy", ["Postgres version", "the Redis eviction policy"]),
    ("Какая версия Postgres, а также политика вытеснения Redis?",
     ["Какая версия Postgres", "политика вытеснения Redis"]),
    ("Какой порт у API и где лежат логи nginx?", ["Какой порт у API", "где лежат логи nginx"]),
    ("What are the deploy steps? What is the rollback plan?", ["What are the deploy steps", "What is the rollback plan"]),
])
def test_splits_compound_queries(query, parts):
    assert compound_parts(query) == parts


@pytest.mark.parametrize("query", [
    "Who did Caroline and Melanie meet?",
    "cats and dogs",
    "What did Caroline do with her family and friends at the park?",
    "When did Melanie paint a sunrise?",
    "кошки и собаки",
    "What does Tim have that serves as a reminder of hard work and is his prized possession?",
    "What dessert did Joanna share with an almond flour crust, chocolate ganache, and fresh raspberries?",
    "Was September a good month for Nate and Joanna? Answer yes or no.",
    "What does Nate do that he loves and can make money from?",
    "",
])
def test_keeps_single_questions_whole(query):
    assert compound_parts(query) == []


def test_parts_are_capped():
    query = "alpha beta one; gamma delta two; epsilon zeta three; eta theta four"
    assert len(compound_parts(query)) == 3


@pytest.mark.parametrize("ranked, winners, k, expected", [
    ([1, 2, 3, 4, 5], [2], 3, [1, 2, 3, 4, 5]),
    ([1, 2, 3, 4, 5], [5], 3, [1, 2, 5, 3, 4]),
    ([1, 2, 3, 4, 5], [4, 5], 3, [1, 4, 5, 2, 3]),
    ([1, 2, 3], [9], 3, [1, 2, 9, 3]),
    ([1, 2], [9], 3, [1, 2, 9]),
    ([1, 2, 3], [1, 2, 3, 9], 3, [1, 2, 3]),
    ([1, 2, 3, 9], [1, 2, 3, 9], 3, [1, 2, 3, 9]),
    ([1, 2, 3, 4, 9], [9, 2], 3, [1, 2, 9, 3, 4]),
    ([1, 2, 3], [], 3, [1, 2, 3]),
    ([1, 2, 3], [3], 0, [1, 2, 3]),
])
def test_guarantee_slots(ranked, winners, k, expected):
    result = guarantee_slots(ranked, winners, k)
    assert result == expected
    assert len(result) == len(set(result))


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "MEMORY_DIR", tmp_path)
    monkeypatch.setenv("MEMORY_CROSS_RERANK", "off")
    instance = server.Store()
    instance._embed_mode = "none"
    # Six short records match six words of the query and crowd the single, longer helm record
    # (two words) out of a top 3; unrelated notes keep the query words rare enough to score.
    rows = [(index, f"we deploy billing postgres database use replica {index}") for index in range(1, 7)]
    rows.append((7, "helm canary rollout notes for the frontend web application release train"))
    rows += [(index, f"unrelated note about lunch number {index}") for index in range(8, 38)]
    for identity, content in rows:
        instance.db.execute(
            "INSERT INTO knowledge(id,content,session_id,project,type,status,created_at,last_confirmed) "
            "VALUES (?,?,'s','p','fact','active','2026-01-01','2026-01-01')", (identity, content))
    instance.db.commit()
    yield instance
    instance.db.close()


QUERY = "Which postgres database does billing use and how do we deploy with helm canary?"


def _ids(result):
    return [item["id"] for group in result["results"].values() for item in group]


def test_flag_off_leaves_ranking_alone(store, monkeypatch):
    monkeypatch.setenv("MEMORY_RECALL_COMPOUND_SLOTS", "off")
    result = server.Recall(store).search(QUERY, project="p", limit=3, record_usage=False, _explain=True)
    assert result["_explain"]["compound_slotted"] == []
    assert 7 not in _ids(result), "without slots the database records crowd out the deploy record"


def test_each_part_gets_a_slot(store):
    assert config.get_recall_compound_slots() is True, "on by default"
    result = server.Recall(store).search(QUERY, project="p", limit=3, record_usage=False, _explain=True)
    ids = _ids(result)
    assert len(ids) == 3
    assert 7 in ids, "the deploy record must hold a slot"
    assert any(kid != 7 for kid in ids), "the database part keeps its own slot"
    assert len(result["_explain"]["compound_parts"]) == 2
