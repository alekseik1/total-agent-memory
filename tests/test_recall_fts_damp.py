"""Cross-lingual FTS damp: no query content word in the index -> lower fts fusion weight."""

from __future__ import annotations

import pytest

import config
import server
from memory_core.fts_vocab import content_tokens, vocabulary_overlap
from tests.pg_store_support import store_backend  # noqa: F401 — fixture


@pytest.fixture(autouse=True)
def _fresh_weight_cache():
    config._tier_weights_cache.clear()
    yield
    config._tier_weights_cache.clear()


def test_content_tokens_drop_function_words_numbers_and_short_tokens():
    assert content_tokens("What does my sister like about the 2024 trip?") == ["sister", "trip"]
    assert content_tokens("что мне нравится в Казани") == ["нравится", "казани"]
    assert content_tokens("is it ok") == []


@pytest.fixture
def mixed_store(store_backend, tmp_path, monkeypatch):  # noqa: F811 — pytest fixture injection
    monkeypatch.setattr(server, "MEMORY_DIR", tmp_path)
    monkeypatch.setenv("MEMORY_CROSS_RERANK", "off")
    store = server.Store(database=store_backend)
    store._embed_mode = "none"
    rows = ("Маша любит зелёный цвет и живёт в Казани",
            "Олег работает в Ozon с прошлого года",
            "what does it mean for the plan")
    for identity, content in enumerate(rows, 1):
        store.db.execute(
            "INSERT INTO knowledge(id,content,session_id,project,type,status,created_at,last_confirmed) "
            "VALUES (?,?,'s','p','fact','active','2026-01-01','2026-01-01')", (identity, content))
    store.db.commit()
    yield store
    store.db.close()


def test_vocabulary_overlap(mixed_store):
    db, pg = mixed_store.db, mixed_store.is_postgres
    assert vocabulary_overlap(db, "Where does Masha live? Kazan?", is_postgres=pg) is False
    assert vocabulary_overlap(db, "Где живёт Маша?", is_postgres=pg) is True
    assert vocabulary_overlap(db, "Маше нравится Казань", is_postgres=pg) is True, "Russian stems match as prefixes"
    assert vocabulary_overlap(db, "Ozon office", is_postgres=pg) is True
    assert vocabulary_overlap(db, "what is it", is_postgres=pg) is None


def test_search_damps_fts_without_content_overlap(mixed_store):
    recall = server.Recall(mixed_store)
    overlapping = recall.search("what does Masha mean for the plan in Kazan", project="p", limit=5,
                           record_usage=False, _explain=True)
    assert overlapping["_explain"]["fts_damped"] is False, "'plan' is in the index"
    foreign = recall.search("what does it signify for Masha", project="p", limit=5, record_usage=False, _explain=True)
    assert foreign["_explain"]["fts_damped"] is True
    assert foreign["_explain"]["tier_weights"]["fts"] == pytest.approx(
        config.DEFAULT_RECALL_TIER_WEIGHTS["fts"] * config.DEFAULT_RECALL_FTS_CROSSLINGUAL_DAMP)


def test_damp_of_one_switches_it_off(mixed_store, monkeypatch):
    monkeypatch.setenv("MEMORY_RECALL_FTS_CROSSLINGUAL_DAMP", "1")
    result = server.Recall(mixed_store).search("what does it signify for Masha", project="p", limit=5,
                                               record_usage=False, _explain=True)
    assert result["_explain"]["fts_damped"] is False
    assert result["_explain"]["tier_weights"]["fts"] == config.DEFAULT_RECALL_TIER_WEIGHTS["fts"]
