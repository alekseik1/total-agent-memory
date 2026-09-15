import sqlite3
from pathlib import Path

import numpy as np
import pytest

from memory_core.retrieval import fetch_active_records
from memory_core.telemetry import counters
from memory_core.vector_search import VectorScope, VectorSearch

MIGRATION = Path(__file__).parents[1] / "migrations/032_vector_index_revision.sql"


@pytest.fixture
def db(tmp_path):
    connection = sqlite3.connect(tmp_path / "vectors.db")
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.executescript(
        "CREATE TABLE knowledge(id INTEGER PRIMARY KEY,content TEXT,project TEXT DEFAULT 'p',"
        "status TEXT DEFAULT 'active',type TEXT DEFAULT 'fact',branch TEXT DEFAULT '',recall_count INTEGER DEFAULT 0);"
        "CREATE TABLE embeddings(knowledge_id INTEGER PRIMARY KEY,binary_vector BLOB,float32_vector BLOB,"
        "embed_dim INTEGER,embed_model TEXT,embedding_space TEXT);"
    )
    connection.executescript(MIGRATION.read_text())
    yield connection
    connection.close()


def seed(db, identity, vector, project="p", space="text", model="m"):
    vector = np.asarray(vector, dtype=np.float32)
    db.execute("INSERT INTO knowledge(id,content,project) VALUES (?,?,?)", (identity, f"record {identity}", project))
    db.execute("INSERT INTO embeddings VALUES (?,?,?,?,?,?)", (
        identity, np.packbits(vector > 0).tobytes(), vector.tobytes(), len(vector), model, space,
    ))


def search(index, **kwargs):
    return index.search([1.0, 0.0], VectorScope(2, project="p"), candidates=5, limit=5, exact_limit=10, **kwargs)


def test_exact_path_has_no_binary_load_and_warm_cache_has_no_vector_load(db):
    seed(db, 1, [1, 0]); seed(db, 2, [0, 1]); db.commit()
    index = VectorSearch(db)
    sql = []
    db.set_trace_callback(sql.append)
    first = search(index)
    assert not any("binary_vector" in statement for statement in sql)
    assert any("float32_vector" in statement for statement in sql)
    sql.clear()
    assert search(index) == first
    assert sql and all("vector_index_revision" in statement for statement in sql)
    assert first[0] == (1, 1.0)


def test_usage_updates_do_not_invalidate_vector_cache(db):
    seed(db, 1, [1, 0]); db.commit()
    index = VectorSearch(db)
    search(index)
    before = counters.get("vector_pool_cache_hits")
    db.execute("UPDATE knowledge SET recall_count=recall_count+1 WHERE id=1"); db.commit()
    search(index)
    assert counters.get("vector_pool_cache_hits") == before + 1


@pytest.mark.parametrize("sql", [
    "UPDATE knowledge SET status='deleted' WHERE id=1",
    "UPDATE knowledge SET project='other' WHERE id=1",
    "DELETE FROM knowledge WHERE id=1",
    "DELETE FROM embeddings WHERE knowledge_id=1",
    "UPDATE embeddings SET embed_model='other' WHERE knowledge_id=1",
    "UPDATE embeddings SET embedding_space='code' WHERE knowledge_id=1",
])
def test_membership_changes_invalidate_cache(db, sql):
    seed(db, 1, [1, 0]); db.commit()
    index = VectorSearch(db)
    scope = VectorScope(2, project="p", model="m", spaces=("text",))
    assert index.search([1, 0], scope, candidates=5, limit=5, exact_limit=10)
    db.execute(sql); db.commit()
    assert index.search([1, 0], scope, candidates=5, limit=5, exact_limit=10) == []


def test_external_committed_vector_change_is_visible(db):
    seed(db, 1, [1, 0]); seed(db, 2, [0, 1]); db.commit()
    index = VectorSearch(db)
    assert search(index)[0][0] == 1
    filename = db.execute("PRAGMA database_list").fetchone()[2]
    with sqlite3.connect(filename) as writer:
        writer.execute("UPDATE embeddings SET float32_vector=? WHERE knowledge_id=1", (np.array([-1, 0], dtype=np.float32).tobytes(),))
    assert search(index)[0][0] == 2


def test_rollback_and_savepoint_do_not_leave_cached_uncommitted_values(db):
    seed(db, 1, [1, 0]); db.commit()
    index = VectorSearch(db)
    assert search(index)
    db.execute("BEGIN")
    db.execute("SAVEPOINT change")
    db.execute("UPDATE knowledge SET project='other' WHERE id=1")
    assert search(index) == []
    db.execute("ROLLBACK TO change")
    assert search(index)
    db.execute("UPDATE knowledge SET status='deleted' WHERE id=1")
    assert search(index) == []
    db.rollback()
    assert search(index)
    assert len(index.pools) == 1


def test_scope_is_applied_before_top_k(db):
    for identity in range(1, 61):
        seed(db, identity, [1, 0])
    seed(db, 61, [0, 1])
    db.execute("UPDATE knowledge SET branch='other',type='lesson' WHERE id<61")
    db.commit()
    index = VectorSearch(db)
    result = index.search([1, 0], VectorScope(2, project="p", kind="fact", branch="main"), candidates=1, limit=1)
    assert result == [(61, 0.0)]
    db.execute("UPDATE knowledge SET branch='other' WHERE id=61"); db.commit()
    assert index.search([1, 0], VectorScope(2, project="p", kind="fact", branch="main"), candidates=1, limit=1) == []


@pytest.mark.parametrize("exact_limit", [0, 100])
def test_streaming_matches_cache_with_zero_vectors_and_ties(db, exact_limit):
    for identity in range(1, 50):
        seed(db, identity, [identity % 3, identity % 5])
    db.commit()
    scope = VectorScope(2, project="p")
    kwargs = {"candidates": 15, "limit": 7, "exact_limit": exact_limit}
    cached = VectorSearch(db).search([1, 0], scope, **kwargs)
    streaming = VectorSearch(db, max_cache_bytes=0).search([1, 0], scope, **kwargs)
    assert streaming == cached


def test_lru_memory_limit_evicts_oldest_scope(db):
    for identity, project in enumerate(("a", "b", "c"), 1):
        seed(db, identity, [1, 0], project=project)
    db.commit()
    index = VectorSearch(db, max_cache_bytes=45)
    for project in ("a", "b", "c"):
        index.search([1, 0], VectorScope(2, project=project), candidates=1, limit=1, exact_limit=10)
        assert index.cache_bytes <= 45
    assert len(index.pools) == 2
    assert [key[0].project for key in index.pools] == ["b", "c"]


def test_newly_inserted_best_vector_is_visible(db):
    seed(db, 1, [0, 1]); db.commit()
    index = VectorSearch(db)
    assert search(index)[0][0] == 1
    seed(db, 2, [1, 0]); db.commit()
    assert search(index)[0][0] == 2


@pytest.mark.parametrize("query", [[float("nan"), 0], [float("inf"), 0], [1], []])
def test_invalid_query_is_rejected(db, query):
    with pytest.raises(ValueError, match="Query"):
        VectorSearch(db).search(query, VectorScope(2), candidates=1, limit=1)


def test_migration_preserves_records_and_is_idempotent(db):
    seed(db, 1, [1, 0]); db.commit()
    before = [tuple(row) for row in db.execute("SELECT * FROM embeddings")]
    revision = db.execute("SELECT revision FROM vector_index_revision").fetchone()[0]
    db.executescript(MIGRATION.read_text())
    assert [tuple(row) for row in db.execute("SELECT * FROM embeddings")] == before
    assert db.execute("SELECT revision FROM vector_index_revision").fetchone()[0] == revision
    db.execute("UPDATE knowledge SET project='b'")
    db.rollback()
    assert db.execute("SELECT revision FROM vector_index_revision").fetchone()[0] == revision


def test_record_hydration_batches_ids_and_excludes_deleted(db):
    for identity in range(1, 902):
        db.execute("INSERT INTO knowledge(id,content) VALUES (?,?)", (identity, "body"))
    db.execute("UPDATE knowledge SET status='deleted' WHERE id=2"); db.commit()
    sql = []
    db.set_trace_callback(sql.append)
    records = fetch_active_records(db, list(range(1, 903)) + [1])
    assert len(records) == 900
    assert 2 not in records and 902 not in records
    assert len(sql) == 3


def test_no_records_means_no_hydration_sql(db):
    sql = []
    db.set_trace_callback(sql.append)
    assert fetch_active_records(db, []) == {}
    assert not sql


def test_group_cache_observes_model_and_project_changes(db):
    seed(db, 1, [1, 0]); db.commit()
    index = VectorSearch(db)
    assert index.groups("p") == (("text", "m", 2),)
    assert index.groups("p") == (("text", "m", 2),)
    db.execute("UPDATE embeddings SET embed_model='replacement'"); db.commit()
    assert index.groups("p") == (("text", "replacement", 2),)
    db.execute("UPDATE knowledge SET project='q'"); db.commit()
    assert index.groups("p") == ()
    assert index.groups("q") == (("text", "replacement", 2),)


def test_empty_scope_caches_are_bounded(db, monkeypatch):
    import memory_core.vector_search as vectors

    monkeypatch.setattr(vectors, "MAX_VECTOR_CACHE_ENTRIES", 2)
    index = VectorSearch(db)
    for project in ("a", "b", "c", "d"):
        assert index.groups(project) == ()
        assert index.search([1, 0], VectorScope(2, project=project), candidates=1, limit=1) == []
    assert len(index.pools) == len(index.group_cache) == 2


def test_concurrent_growth_between_count_and_load_respects_budget(db):
    seed(db, 1, [0, 1]); db.commit()
    path = db.execute("PRAGMA database_list").fetchone()[2]
    writer = sqlite3.connect(path)
    inserted = False

    def insert_before_load(sql):
        nonlocal inserted
        if not inserted and sql.startswith("SELECT e.knowledge_id,e.float32_vector"):
            inserted = True
            seed(writer, 2, [1, 0])
            writer.commit()

    index = VectorSearch(db, max_cache_bytes=20)
    db.set_trace_callback(insert_before_load)
    try:
        assert [identity for identity, _ in search(index)] == [2, 1]
        assert inserted
        assert index.cache_bytes == 0
    finally:
        db.set_trace_callback(None)
        writer.close()
