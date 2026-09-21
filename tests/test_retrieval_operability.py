import logging

import pytest

import server
from memory_core.telemetry import counters


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "MEMORY_DIR", tmp_path)
    instance = server.Store()
    instance._embed_mode = "none"
    monkeypatch.setattr(instance, "bump_recall", lambda ids: None)
    for identity, content in enumerate(
        ("database owner database", "database owner", "other topic"), 1
    ):
        instance.db.execute(
            "INSERT INTO knowledge(id,content,session_id,project,type,status,created_at,last_confirmed) "
            "VALUES (?,?,'s','p','fact','active','2026-01-01','2026-01-01')",
            (identity, content),
        )
    instance.db.commit()
    yield instance
    instance.db.close()


def ids(result):
    return [hit["id"] for group in result["results"].values() for hit in group]


def test_atomic_index_cannot_displace_primary_results(store):
    recall = server.Recall(store)
    before = recall.search(
        "database owner", project="p", limit=2, detail="full", _explain=True
    )
    store.db.execute(
        "INSERT INTO atomic_facts(knowledge_id,subject,predicate,object,content) "
        "VALUES (3,'database','owner','database','database owner database owner')"
    )
    store.db.execute("INSERT INTO atomic_fact_sources VALUES (1,3,'other topic')")
    store.db.commit()
    after = recall.search(
        "database owner", project="p", limit=2, detail="full", _explain=True
    )
    assert ids(before) == ids(after)
    assert 3 not in ids(after)


def test_missing_fts_is_visible_and_does_not_crash_other_tiers(store, caplog):
    store.db.execute("DROP TABLE knowledge_fts")
    before = counters.get("retrieval_fts_errors")
    with caplog.at_level(logging.ERROR):
        result = server.Recall(store).search(
            "database owner", project="p", limit=2, _explain=True
        )
    assert result["query"] == "database owner"
    assert counters.get("retrieval_fts_errors") == before + 1
    assert any(getattr(record, "tier", "") == "fts" for record in caplog.records)


@pytest.mark.parametrize("project", ["p", "other", "absent", None])
def test_fts_bounds_preserve_scope_with_interleaved_and_deleted_records(store, project):
    store.db.execute("UPDATE knowledge SET project='other' WHERE id=2")
    store.db.execute(
        "INSERT INTO knowledge(id,content,session_id,project,type,status,created_at,last_confirmed) "
        "VALUES (4,'database owner','s','p','fact','deleted','2026-01-01','2026-01-01')"
    )
    store.db.execute("UPDATE knowledge SET content='database owner' WHERE id=3")
    store.db.commit()
    statements = []
    store.db.set_trace_callback(statements.append)
    server.Recall(store).search("database owner", project=project, limit=10, _explain=True)
    store.db.set_trace_callback(None)
    sql = next(statement for statement in statements if "AS _bm25" in statement and "knowledge_fts MATCH" in statement)
    actual = [(row["id"], row["_bm25"]) for row in store.db.execute(sql)]
    predicate = " AND k.project=?" if project else ""
    expected = store.db.execute(
        "SELECT k.id, bm25(knowledge_fts) FROM knowledge_fts f "
        "JOIN knowledge k ON k.id=f.rowid WHERE knowledge_fts MATCH ? "
        "AND k.status='active'" + predicate + " ORDER BY bm25(knowledge_fts)" + (", k.id" if project else "") + " LIMIT 30",
        ('"database" OR "owner"', project) if project else ('"database" OR "owner"',),
    ).fetchall()
    assert actual == [tuple(row) for row in expected]
    # A project filter must not make SQLite re-run MATCH per project row.
    assert ("AS MATERIALIZED" in sql) == bool(project)


@pytest.mark.parametrize("savepoint", [False, True])
def test_query_cache_does_not_preserve_rolled_back_results(store, savepoint):
    recall = server.Recall(store)

    def search():
        return ids(recall.search("database owner", project="p", limit=2, record_usage=False))

    assert 1 in search()
    store.db.execute("SAVEPOINT caller")
    store.db.execute("UPDATE knowledge SET status='deleted' WHERE id=1")
    assert 1 not in search()
    assert store.db.in_transaction
    if savepoint:
        store.db.execute("ROLLBACK TO caller")
    else:
        store.db.rollback()
    assert 1 in search()
    if savepoint:
        store.db.execute("RELEASE caller")
    assert 1 in search()
