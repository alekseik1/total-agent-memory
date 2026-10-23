"""Query intent -> record-type multipliers (decision / lesson+convention / solution)."""

from __future__ import annotations

import logging

import pytest

import config
import server
from memory_core.query_shapes import INTENT_DECISION, INTENT_HOWTO, INTENT_LESSON, query_intents
from memory_core.timestamps import utc_now


@pytest.fixture(autouse=True)
def _fresh_caches():
    config._intent_boosts_cache.clear()
    config._tier_weights_cache.clear()
    yield
    config._intent_boosts_cache.clear()
    config._tier_weights_cache.clear()


@pytest.mark.parametrize("query, intent", [
    ("What did we decide about the cache?", INTENT_DECISION),
    ("Why did we choose Postgres over MySQL?", INTENT_DECISION),
    ("why do we use RabbitMQ", INTENT_DECISION),
    ("What was the rationale for sharding?", INTENT_DECISION),
    ("Какое решение приняли по кэшу?", INTENT_DECISION),
    ("Почему выбрали Go?", INTENT_DECISION),
    ("что мы решили по деплою", INTENT_DECISION),
    ("What lessons did we learn from the outage?", INTENT_LESSON),
    ("What are the rules for naming branches?", INTENT_LESSON),
    ("Any gotchas with pgbouncer?", INTENT_LESSON),
    ("Какие правила именования веток?", INTENT_LESSON),
    ("какие грабли были с миграцией", INTENT_LESSON),
    ("чему мы научились на инциденте", INTENT_LESSON),
    ("How did we fix the OOM in the worker?", INTENT_HOWTO),
    ("How to configure nginx rate limits?", INTENT_HOWTO),
    ("workaround for the flaky e2e test", INTENT_HOWTO),
    ("Как починить сборку на CI?", INTENT_HOWTO),
    ("как мы решили проблему с памятью", INTENT_HOWTO),
])
def test_detects_intents(query, intent):
    assert intent in query_intents(query)


@pytest.mark.parametrize("query", [
    "What is Caroline's favourite colour?",
    "When did Melanie paint a sunrise?",
    "Где живёт Игорь?",
    "How many children does Melanie have?",
    "When did Jolene and her partner try scuba diving lessons?",
    "What game did John play at the gaming convention in March?",
    "Why did John start blogging about politics and policies?",
    "",
])
def test_plain_questions_have_no_intent(query):
    assert query_intents(query) == frozenset()


def test_type_boosts_follow_config(monkeypatch):
    assert config.recall_type_boosts(frozenset({INTENT_LESSON})) == {"lesson": 1.3, "convention": 1.3}
    assert config.recall_type_boosts(frozenset()) == {}
    monkeypatch.setenv("MEMORY_RECALL_INTENT_BOOSTS", "lesson=1.0,howto=2")
    assert config.recall_type_boosts(frozenset({INTENT_LESSON, INTENT_HOWTO})) == {"solution": 2.0}


@pytest.mark.parametrize("raw", ["unknown=2", "decision=0", "decision=-1", "decision=abc", "decision=nan", "decision"])
def test_malformed_boosts_are_rejected_and_ignored(raw, monkeypatch, caplog):
    with pytest.raises(ValueError):
        config.parse_intent_boosts(raw)
    monkeypatch.setenv("MEMORY_RECALL_INTENT_BOOSTS", raw)
    with caplog.at_level(logging.WARNING):
        assert config.get_recall_intent_boosts() == config.DEFAULT_RECALL_INTENT_BOOSTS
    assert "ignoring MEMORY_RECALL_INTENT_BOOSTS" in caplog.text


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "MEMORY_DIR", tmp_path)
    monkeypatch.setenv("MEMORY_CROSS_RERANK", "off")
    instance = server.Store()
    instance._embed_mode = "none"
    rows = (
        (1, "fact", "billing cache uses redis with a five minute ttl"),
        (2, "decision", "billing cache: redis chosen over memcached for persistence"),
        (3, "solution", "billing cache stampede fixed with request coalescing"),
        (4, "lesson", "billing cache: never share redis between tenants"),
    )
    # Fresh timestamps: decision/lesson are exempt from age decay (decay_policy),
    # so old rows would be ranked by decay rather than by the intent boost.
    now = utc_now()
    for identity, ktype, content in rows:
        instance.db.execute(
            "INSERT INTO knowledge(id,content,session_id,project,type,status,created_at,last_confirmed) "
            "VALUES (?,?,'s','p',?,'active',?,?)", (identity, content, ktype, now, now))
    instance.db.commit()
    yield instance
    instance.db.close()


def _ids(result):
    return [item["id"] for item in result["_explain"]["merged"]]


@pytest.mark.parametrize("query, expected_first, boosted", [
    ("billing cache decision", 2, "decision"),
    ("billing cache lessons learned", 4, "lesson"),
    ("billing cache workaround", 3, "solution"),
])
def test_intent_lifts_the_matching_type(store, monkeypatch, query, expected_first, boosted):
    monkeypatch.setenv("MEMORY_RECALL_INTENT_BOOSTS", "decision=3,lesson=3,howto=3")
    result = server.Recall(store).search(query, project="p", limit=4, record_usage=False, _explain=True)
    assert _ids(result)[0] == expected_first
    assert expected_first in result["_explain"]["type_boosted"]
    assert result["_explain"]["type_boosts"].get(boosted) == 3.0


def test_boost_of_one_changes_nothing(store, monkeypatch):
    plain = server.Recall(store).search("billing cache", project="p", limit=4, record_usage=False, _explain=True)
    monkeypatch.setenv("MEMORY_RECALL_INTENT_BOOSTS", "decision=1,lesson=1,howto=1")
    neutral = server.Recall(store).search("billing cache decision", project="p", limit=4, record_usage=False,
                                          _explain=True)
    assert neutral["_explain"]["type_boosted"] == []
    assert plain["_explain"]["intents"] == []
