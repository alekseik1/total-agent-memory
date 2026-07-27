"""Tests for dashboard_v6 API endpoints."""

from __future__ import annotations

import sqlite3

import pytest

from base_schema import apply_full_schema


@pytest.fixture
def dash_db():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    apply_full_schema(conn)
    yield conn
    conn.close()


def test_savings_aggregates_empty(dash_db):
    from dashboard_v6 import api_v6_savings

    res = api_v6_savings(dash_db)
    assert res["applied_count"] == 0
    assert res["tokens_saved_estimate"] == 0
    assert res["by_filter"] == []


def test_savings_with_data(dash_db):
    from dashboard_v6 import api_v6_savings

    dash_db.executemany(
        "INSERT INTO filter_savings "
        "(knowledge_id, filter_name, input_chars, output_chars, reduction_pct, safety, created_at) "
        "VALUES (?,?,?,?,?,?,?)",
        [
            (1, "pytest", 1000, 200, 80.0, "strict", "2026-04-14T00:00:00Z"),
            (2, "pytest", 2000, 500, 75.0, "strict", "2026-04-14T00:00:00Z"),
            (3, "cargo",  800,  300, 62.5, "strict", "2026-04-14T00:00:00Z"),
        ],
    )
    dash_db.commit()

    res = api_v6_savings(dash_db)
    assert res["applied_count"] == 3
    assert res["chars_saved"] == (1000+2000+800) - (200+500+300)
    assert res["tokens_saved_estimate"] == res["chars_saved"] // 4
    names = {b["name"] for b in res["by_filter"]}
    assert names == {"pytest", "cargo"}


def test_queues_all_empty_tables_return_zeros(dash_db):
    from dashboard_v6 import api_v6_queues

    res = api_v6_queues(dash_db)
    assert set(res.keys()) == {
        "triple_extraction_queue", "deep_enrichment_queue", "representations_queue"
    }
    for counts in res.values():
        assert counts.get("pending") == 0
        assert counts.get("done") == 0


def test_queues_report_status_breakdown(dash_db):
    from dashboard_v6 import api_v6_queues

    dash_db.executemany(
        "INSERT INTO triple_extraction_queue (knowledge_id, status, created_at) VALUES (?,?,?)",
        [(1, "pending", "t"), (2, "pending", "t"), (3, "done", "t"), (4, "failed", "t")],
    )
    dash_db.commit()
    res = api_v6_queues(dash_db)
    assert res["triple_extraction_queue"]["pending"] == 2
    assert res["triple_extraction_queue"]["done"] == 1
    assert res["triple_extraction_queue"]["failed"] == 1


def test_coverage_no_knowledge(dash_db):
    from dashboard_v6 import api_v6_coverage

    res = api_v6_coverage(dash_db)
    assert res["active_knowledge"] == 0
    assert res["representations_pct"] == 0


def test_coverage_computes_percentages(dash_db):
    from dashboard_v6 import api_v6_coverage

    for i in range(4):
        dash_db.execute(
            "INSERT INTO knowledge (session_id, type, content, project, status, created_at) "
            "VALUES ('s1', 'fact', 'c','demo','active','2026-04-14T00:00:00Z')"
        )
    dash_db.executemany(
        "INSERT INTO knowledge_representations "
        "(knowledge_id, representation, content, binary_vector, float32_vector, embed_model, embed_dim, created_at) "
        "VALUES (?, 'raw', ?, ?, ?, 'fake', 4, ?)",
        [(1, "c", b"\0" * 4, b"\0" * 16, "t"), (2, "c", b"\0" * 4, b"\0" * 16, "t")],
    )
    dash_db.execute(
        "INSERT INTO knowledge_enrichment (knowledge_id, entities, intent, topics, updated_at) "
        "VALUES (1, '[]', 'fact', '[]', 't')"
    )
    dash_db.commit()

    res = api_v6_coverage(dash_db)
    assert res["active_knowledge"] == 4
    assert res["representations_pct"] == 50.0   # 2/4
    assert res["enrichment_pct"] == 25.0        # 1/4


def test_v10_enrichment_queue_returns_zeros_on_empty_db(dash_db):
    from dashboard_v6 import api_v10_enrichment_queue
    out = api_v10_enrichment_queue(dash_db)
    assert out["status_counts"] == {
        "pending": 0, "processing": 0, "done": 0, "failed": 0,
    }
    assert out["throughput_per_min"] == 0.0
    assert out["p50_ms_per_task"] is None
    assert out["oldest_pending_age_sec"] is None
    assert out["recent_failures"] == []


def test_v10_enrichment_queue_reports_status_breakdown(dash_db):
    from dashboard_v6 import api_v10_enrichment_queue
    now = "2026-04-27T08:00:00Z"
    rows = [
        ("pending", None, None, None, None),
        ("pending", None, None, None, None),
        ("processing", "2026-04-27T07:59:50Z", None, None, None),
        ("done", "2026-04-27T07:59:00Z", "2026-04-27T07:59:00.250Z", None, None),
        ("done", "2026-04-27T07:58:00Z", "2026-04-27T07:58:00.450Z", None, None),
        ("failed", "2026-04-27T07:55:00Z", "2026-04-27T07:55:00.500Z", 3,
         "stage 'contradiction' failed: provider down"),
    ]
    for status, started, finished, attempts, err in rows:
        dash_db.execute(
            "INSERT INTO enrichment_queue "
            "(knowledge_id, project, ktype, content_snapshot, status, "
            " attempts, started_at, finished_at, last_error, enqueued_at) "
            "VALUES (1, 'p', 'fact', 'x', ?, ?, ?, ?, ?, ?)",
            (status, attempts or 0, started, finished, err, now),
        )
    dash_db.commit()
    out = api_v10_enrichment_queue(dash_db)
    assert out["status_counts"]["pending"] == 2
    assert out["status_counts"]["processing"] == 1
    assert out["status_counts"]["done"] == 2
    assert out["status_counts"]["failed"] == 1
    # p50 from durations [250ms, 450ms] → middle → 450ms (index 1 of len 2).
    assert out["p50_ms_per_task"] in (250, 450)
    assert len(out["recent_failures"]) == 1
    assert "provider down" in out["recent_failures"][0]["last_error"]


def test_graph_delta_returns_recent_nodes_and_edges(dash_db):
    from dashboard_v6 import api_graph_delta

    dash_db.executemany(
        "INSERT INTO graph_nodes (id, type, name, first_seen_at, last_seen_at) "
        "VALUES (?, 'concept', ?, ?, ?)",
        [
            ("n1", "alpha", "2026-04-14T00:00:00Z", "2026-04-14T00:00:00Z"),
            ("n2", "beta",  "2026-04-14T01:00:00Z", "2026-04-14T01:00:00Z"),
        ],
    )
    dash_db.execute(
        "INSERT INTO graph_edges (id, source_id, target_id, relation_type, weight, created_at) "
        "VALUES ('e1', 'n1', 'n2', 'uses', 1.0, '2026-04-14T01:30:00Z')"
    )
    dash_db.commit()

    res = api_graph_delta(dash_db, since="2026-04-13T00:00:00Z")
    assert len(res["nodes"]) == 2
    assert len(res["edges"]) == 1
    assert res["max_ts"] >= "2026-04-14T01:30:00Z"

    res2 = api_graph_delta(dash_db, since="2026-04-14T00:30:00Z")
    names = {n["name"] for n in res2["nodes"]}
    assert "beta" in names
    # alpha still comes back as a filled endpoint so the edge is drawable
    assert "alpha" in names
    assert res2["stats"]["endpoint_nodes_filled"] >= 1


def test_graph_delta_fills_missing_edge_endpoints(dash_db):
    """Edges pointing to nodes older than `since` must have their endpoints added."""
    from dashboard_v6 import api_graph_delta

    # Three nodes: two old, one new. Edges bind old→old, old→new, new→old.
    dash_db.executemany(
        "INSERT INTO graph_nodes (id, type, name, first_seen_at, last_seen_at) "
        "VALUES (?, 'concept', ?, ?, ?)",
        [
            ("old1", "old_a", "2026-04-10T00:00:00Z", "2026-04-10T00:00:00Z"),
            ("old2", "old_b", "2026-04-10T00:00:00Z", "2026-04-10T00:00:00Z"),
            ("new1", "new_x", "2026-04-14T00:00:00Z", "2026-04-14T00:00:00Z"),
        ],
    )
    # A fresh edge that links old→old (both endpoints are older than `since`)
    dash_db.execute(
        "INSERT INTO graph_edges (id, source_id, target_id, relation_type, weight, created_at) "
        "VALUES ('e_old_old', 'old1', 'old2', 'ref', 1.0, '2026-04-14T00:30:00Z')"
    )
    dash_db.commit()

    res = api_graph_delta(dash_db, since="2026-04-13T00:00:00Z")
    node_ids = {n["id"] for n in res["nodes"]}
    # All endpoints present even though old1/old2 predate `since`
    assert {"old1", "old2", "new1"}.issubset(node_ids)
    assert res["stats"]["endpoint_nodes_filled"] >= 2
