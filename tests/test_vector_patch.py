"""Vector pools follow writes through the migration 036 change log."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pytest

from memory_core.telemetry import counters
from memory_core.vector_search import VectorScope, VectorSearch

MIGRATIONS = Path(__file__).parents[1] / "migrations"


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
    connection.executescript((MIGRATIONS / "032_vector_index_revision.sql").read_text())
    connection.executescript((MIGRATIONS / "036_vector_change_log.sql").read_text())
    yield connection
    connection.close()


def seed(db, identity, vector, project="p", model="m"):
    vector = np.asarray(vector, dtype=np.float32)
    db.execute("INSERT INTO knowledge(id,content,project) VALUES (?,?,?)", (identity, f"record {identity}", project))
    db.execute("INSERT INTO embeddings VALUES (?,?,?,?,?,?)", (
        identity, np.packbits(vector > 0).tobytes(), vector.tobytes(), len(vector), model, "text",
    ))
    db.commit()


QUERY = [1.0, 0.2, 0.0, 0.0]


def search(index, exact_limit=10, project=None):
    return index.search(QUERY, VectorScope(4, project=project), candidates=50, limit=5, exact_limit=exact_limit)


def loads(db, index, **kwargs):
    statements = []
    db.set_trace_callback(statements.append)
    result = search(index, **kwargs)
    db.set_trace_callback(None)
    return result, [s for s in statements if s.startswith("SELECT COUNT(*)")]


@pytest.mark.parametrize("exact_limit", [10, 0])
def test_insert_patches_pool_without_reload(db, exact_limit):
    rng = np.random.default_rng(7)
    for identity in range(1, 6):
        seed(db, identity, rng.normal(size=4))
    index = VectorSearch(db)
    search(index, exact_limit=exact_limit)
    before = counters.get("vector_pool_patches")
    seed(db, 6, [1.0, 0.2, 0.0, 0.0])

    result, reloads = loads(db, index, exact_limit=exact_limit)

    assert reloads == []
    assert counters.get("vector_pool_patches") == before + 1
    assert result == search(VectorSearch(db), exact_limit=exact_limit)
    assert result[0][0] == 6


def test_status_change_removes_record(db):
    seed(db, 1, [1, 0.2, 0, 0]); seed(db, 2, [0, 1, 0, 0])
    index = VectorSearch(db)
    assert search(index)[0][0] == 1
    db.execute("UPDATE knowledge SET status='superseded' WHERE id=1"); db.commit()

    result, reloads = loads(db, index)

    assert reloads == []
    assert [identity for identity, _ in result] == [2]


def test_truncated_log_reloads(db):
    seed(db, 1, [1, 0, 0, 0])
    index = VectorSearch(db)
    search(index)
    seed(db, 2, [1, 0.2, 0, 0])
    db.execute("DELETE FROM vector_changes"); db.commit()

    result, reloads = loads(db, index)

    assert reloads
    assert result[0][0] == 2


def test_crossing_exact_limit_reloads(db):
    seed(db, 1, [1, 0, 0, 0]); seed(db, 2, [0, 1, 0, 0])
    index = VectorSearch(db)
    search(index, exact_limit=2)
    seed(db, 3, [1, 0.2, 0, 0])

    result, reloads = loads(db, index, exact_limit=2)

    assert reloads
    assert result == search(VectorSearch(db), exact_limit=2)


def test_new_group_drops_cached_groups(db):
    seed(db, 1, [1, 0, 0, 0])
    index = VectorSearch(db)
    assert index.groups("p") == (("text", "m", 4),)
    search(index)
    seed(db, 2, [0, 1, 0, 0], model="other")
    search(index)
    assert set(index.groups("p")) == {("text", "m", 4), ("text", "other", 4)}


def test_many_appended_records_stay_sorted(db):
    seed(db, 1, [0, 1, 0, 0])
    index = VectorSearch(db)
    search(index, exact_limit=0)
    for identity in (5, 3, 4, 2):
        seed(db, identity, [1.0, 0.1 * identity, 0, 0])
    result, reloads = loads(db, index, exact_limit=0)
    assert reloads == []
    pool = next(iter(index.pools.values()))
    assert list(pool.ids) == sorted(pool.ids)
    assert result == search(VectorSearch(db), exact_limit=0)


def test_write_patches_only_the_pool_a_search_uses(db):
    """A save must not re-read every cached pool: 14.4.0 candidates did, +4 ms per recall."""
    rng = np.random.default_rng(3)
    projects = [f"t{n}" for n in range(20)]
    for offset, project in enumerate(projects):
        seed(db, offset * 10 + 1, rng.normal(size=4), project=project)
    index = VectorSearch(db)
    for project in projects:
        search(index, project=project)
    stale = [key for key in index.pools if key[0].project == "t1"]
    seed(db, 1000, [1.0, 0.2, 0.0, 0.0], project="t0")

    statements = []
    db.set_trace_callback(statements.append)
    result = search(index, project="t0")
    db.set_trace_callback(None)

    assert result[0][0] == 1000
    assert len([s for s in statements if "e.knowledge_id IN (" in s and "ORDER BY e.knowledge_id" in s]) == 1
    assert all(index.pool_revisions[key] < index.revision for key in stale)
    assert search(index, project="t1") == search(VectorSearch(db), project="t1")


def test_pool_behind_several_writes_catches_up_once(db):
    seed(db, 1, [0, 1, 0, 0], project="a")
    seed(db, 2, [0, 1, 0, 0], project="b")
    index = VectorSearch(db)
    search(index, project="a")
    for identity in (3, 4, 5):
        seed(db, identity, [1.0, 0.1 * identity, 0, 0], project="a")
    search(index, project="b")
    db.execute("UPDATE knowledge SET status='superseded' WHERE id=4"); db.commit()
    before = counters.get("vector_pool_patches")

    result, reloads = loads(db, index, project="a")

    assert reloads == []
    assert counters.get("vector_pool_patches") == before + 1
    assert result == search(VectorSearch(db), project="a")
    assert 4 not in [identity for identity, _ in result]


def test_revision_range_reads_use_the_index(db):
    plan = db.execute(
        "EXPLAIN QUERY PLAN SELECT revision, knowledge_id FROM vector_changes WHERE revision > 1 AND revision <= 2"
    ).fetchall()
    assert any("idx_vector_changes_revision" in str(tuple(row)) for row in plan)
