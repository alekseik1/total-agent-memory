import asyncio
import json
import sqlite3
from types import SimpleNamespace

import pytest

from memory_core.evidence_window import EvidenceWindow
from memory_core.retrieval import SearchScope


@pytest.fixture
def db():
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.execute(
        "CREATE TABLE knowledge (id INTEGER PRIMARY KEY, content TEXT, project TEXT, "
        "session_id TEXT, created_at TEXT, status TEXT, type TEXT, branch TEXT)"
    )
    connection.execute(
        "CREATE TABLE embeddings (knowledge_id INTEGER, embedding_space TEXT)"
    )
    rows = [
        (
            1,
            "The service owner is Morgan.",
            "app",
            "s",
            "2026-01-01",
            "active",
            "fact",
            "",
        ),
        (2, "Morgan uses PostgreSQL.", "app", "s", "2026-01-01", "active", "fact", ""),
        (3, "secret", "other", "s", "2026-01-01", "active", "fact", ""),
        (4, "deleted", "app", "s", "2026-01-01", "deleted", "fact", ""),
        (5, "other session", "app", "other", "2026-01-01", "active", "fact", ""),
        (6, "branch private", "app", "s", "2026-01-01", "active", "fact", "other"),
        (7, "source code", "app", "s", "2026-01-01", "active", "fact", ""),
    ]
    connection.executemany("INSERT INTO knowledge VALUES (?,?,?,?,?,?,?,?)", rows)
    connection.executemany(
        "INSERT INTO embeddings VALUES (?,?)", [(1, "text"), (2, "text"), (7, "code")]
    )
    connection.execute("ALTER TABLE knowledge ADD COLUMN tags TEXT DEFAULT '[]'")
    yield connection
    connection.close()


def test_recover_bridge_even_when_timestamps_are_equal(db):
    hits = [{"id": 1, "content": "The service owner is Morgan."}]
    result = EvidenceWindow(db).expand(hits, scope=SearchScope(project="app"), radius=1)
    assert [hit["id"] for hit in result] == [1, 2]
    assert result[1]["content"] == "Morgan uses PostgreSQL."
    assert result[1]["source_ref"] == "knowledge:2"


def test_anchor_is_refreshed_even_without_neighbors(db):
    result = EvidenceWindow(db).expand(
        [{"id": 1, "content": "stale or truncated", "score": 0.8}],
        scope=SearchScope(project="app"), radius=0,
    )
    assert result[0]["content"] == "The service owner is Morgan."
    assert result[0]["source_ref"] == "knowledge:1"
    assert result[0]["score"] == 0.8


def test_neighbor_budget_serves_later_anchor_before_distant_context(db):
    db.execute("UPDATE knowledge SET session_id='s',project='app',status='active',branch='' WHERE id<=7")
    result = EvidenceWindow(db).expand(
        [{"id": 1}, {"id": 7}], scope=SearchScope(project="app"), radius=3, max_neighbors=2,
    )
    assert [hit["id"] for hit in result] == [1, 7, 2, 6]


def test_batch_window_keeps_order_with_bounded_sql_count(db):
    statements = []
    db.set_trace_callback(statements.append)
    result = EvidenceWindow(db).expand(
        [{"id": 2}, {"id": 7}],
        scope=SearchScope(project="app", branch="main"),
        radius=1,
    )
    db.set_trace_callback(None)
    assert [hit["id"] for hit in result] == [2, 7, 1]
    assert len([sql for sql in statements if sql.startswith(("SELECT", "WITH"))]) == 2


def test_scope_is_enforced_before_neighbor_limit(db):
    result = EvidenceWindow(db).expand(
        [{"id": 2}],
        scope=SearchScope(project="app", branch="main", spaces="text"),
        radius=3,
    )
    assert [hit["id"] for hit in result] == [2, 1]
    assert (
        EvidenceWindow(db).expand([{"id": 3}], scope=SearchScope(project="app")) == []
    )


def test_no_global_fallback_and_no_duplicates(db):
    result = EvidenceWindow(db).expand(
        [{"id": 1}, {"id": 2}], scope=SearchScope(project="app"), radius=1
    )
    assert len({hit["id"] for hit in result}) == len(result)
    assert not {3, 4, 5}.intersection(hit["id"] for hit in result)
    assert [
        hit["id"]
        for hit in EvidenceWindow(db).expand(
            [{"id": 5}], scope=SearchScope(project="app")
        )
    ] == [5]


def test_budget_and_invalid_radius(db):
    assert (
        len(
            EvidenceWindow(db).expand(
                [{"id": 2}], scope=SearchScope(project="app"), max_neighbors=1
            )
        )
        == 2
    )
    with pytest.raises(ValueError):
        EvidenceWindow(db).expand([{"id": 2}], scope=SearchScope(), radius=-1)


def test_public_context_mode_returns_full_scoped_evidence(db, monkeypatch):
    import server

    monkeypatch.setattr(server, "store", SimpleNamespace(db=db))
    monkeypatch.setattr(
        server,
        "recall",
        SimpleNamespace(
            search=lambda *args: {
                "query": "service owner database",
                "results": {"fact": [{"id": 1, "content": "owner is Morgan"}]},
            }
        ),
    )
    result = json.loads(
        asyncio.run(
            server._do(
                "memory_recall",
                {
                    "query": "service owner database",
                    "project": "app",
                    "mode": "context",
                    "branch": "main",
                    "neighbors": 1,
                },
            )
        )
    )
    assert result["mode"] == "context"
    assert result["total_tokens"] > 0
    assert "qualified inference" in result["answer_guidance"]
    assert [hit["id"] for hit in result["results"]] == [1, 2]
    assert result["results"][1]["content"] == "Morgan uses PostgreSQL."


def test_public_iterative_mode_expands_full_scoped_evidence(db, monkeypatch):
    import server
    from ai_layer import iterative_retriever as ir

    monkeypatch.setattr(server, "store", SimpleNamespace(db=db))

    def search(*args, **kwargs):
        assert kwargs["detail"] == "full"
        return {"results": {"fact": [{"id": 1, "content": "The owner is Morgan"}]}}

    monkeypatch.setattr(server, "recall", SimpleNamespace(search=search))

    def retrieve(query, **kwargs):
        evidence = kwargs["search_fn"](query, k=5, project="app")
        assert 2 in {hit["id"] for hit in evidence}
        assert 3 not in {hit["id"] for hit in evidence}
        return ir.IterativeResult(evidence * 12, [query], [], 1, "converged")

    monkeypatch.setattr(ir, "iterative_retrieve", retrieve)
    result = json.loads(
        asyncio.run(
            server._do(
                "memory_recall_iterative", {"query": "owner database", "project": "app"}
            )
        )
    )
    assert len(result["evidence"]) == result["evidence_count"]
    assert result["evidence_count"] > 20


def test_public_context_mode_selects_middle_excerpt_with_source(db, monkeypatch):
    import server

    content = "Date: March 1\n" + "Unrelated conversation.\n" * 200
    content += "The service owner is Morgan. Morgan uses PostgreSQL.\n"
    content += "Unrelated conversation.\n" * 200
    db.execute("UPDATE knowledge SET content=? WHERE id=1", (content,))
    monkeypatch.setattr(server, "store", SimpleNamespace(db=db))
    monkeypatch.setattr(server, "recall", SimpleNamespace(search=lambda *args: {
        "results": {"fact": [{"id": 1, "content": content}]},
    }))
    result = json.loads(asyncio.run(server._do("memory_recall", {
        "query": "service owner database", "project": "app", "mode": "context",
        "neighbors": 0, "context_max_chars": 1200,
    })))
    assert "Morgan uses PostgreSQL" in result["results"][0]["content"]
    assert result["results"][0]["source_ref"] == "knowledge:1"
    assert len(result["results"][0]["content"]) < 1200
    assert db.execute("SELECT content FROM knowledge WHERE id=1").fetchone()[0] == content


def test_excluded_operational_records_do_not_reenter_context(db):
    db.execute("UPDATE knowledge SET tags=? WHERE id=2", ('["recovery"]',))
    result = EvidenceWindow(db, ("recovery",)).expand(
        [{"id": 1}],
        scope=SearchScope(project="app", branch="main"),
        radius=1,
    )
    assert [hit["id"] for hit in result] == [1, 7]


def test_public_timeline_does_not_leak_other_project(db, monkeypatch):
    import server

    db.execute("UPDATE knowledge SET created_at='2026-01-02' WHERE id=3")
    monkeypatch.setattr(server, "store", SimpleNamespace(db=db))
    monkeypatch.setattr(
        server,
        "recall",
        SimpleNamespace(
            search=lambda *args: {
                "results": {
                    "fact": [
                        {
                            "id": 2,
                            "content": "Morgan uses PostgreSQL.",
                            "project": "app",
                            "session_id": "s",
                            "created_at": "2026-01-01",
                        }
                    ]
                }
            }
        ),
    )
    result = json.loads(
        asyncio.run(
            server._do(
                "memory_recall",
                {
                    "query": "database",
                    "project": "app",
                    "mode": "timeline",
                    "neighbors": 1,
                },
            )
        )
    )
    assert [hit["id"] for hit in result["results"]] == [2]
