"""HTTP-level tests for citation endpoints: /api/knowledge/{id}, /api/session/{id}."""

from __future__ import annotations

import json
import socket
import sqlite3
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from http.server import HTTPServer
from pathlib import Path

import pytest

from base_schema import apply_full_schema


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def citation_server():
    """Spin up a dashboard instance with seeded knowledge + graph edges."""
    tmp = Path(tempfile.mkdtemp(prefix="dash_cite_"))
    db_path = tmp / "memory.db"
    root = Path(__file__).parent.parent

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    apply_full_schema(conn)

    # Seed 3 knowledge records; link 1 ↔ 2 via graph_edges (k-1 to k-2).
    # Record 3 is unrelated to test negatives.
    conn.executemany(
        "INSERT INTO knowledge "
        "(id, session_id, type, content, context, project, tags, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (1, "s-citation", "solution", "Primary knowledge for citation.",
             "First record context", "proj-cite", '["reusable"]', "2026-04-18T10:00:00Z"),
            (2, "s-citation", "fact", "Related peer record used as edge.",
             "", "proj-cite", '[]', "2026-04-18T10:05:00Z"),
            (3, "s-other", "fact", "Unrelated record in another session.",
             "", "proj-other", '[]', "2026-04-18T10:10:00Z"),
        ],
    )

    # Graph nodes + edge — k-1 is linked to k-2 via graph_edges
    now = "2026-04-18T10:05:00Z"
    for nid, name in (("k-1", "primary"), ("k-2", "peer")):
        conn.execute(
            "INSERT INTO graph_nodes (id, type, name, first_seen_at, last_seen_at, mention_count) "
            "VALUES (?, 'knowledge', ?, ?, ?, 1)",
            (nid, name, now, now),
        )
    conn.execute(
        "INSERT INTO graph_edges (id, source_id, target_id, relation_type, weight, created_at) "
        "VALUES ('e-1-2', 'k-1', 'k-2', 'related_to', 1.0, ?)",
        (now,),
    )

    # Session row + session_summaries row (continuity blob).
    conn.execute(
        "INSERT INTO sessions (id, started_at, ended_at, project, status, summary) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("s-citation", "2026-04-18T09:00:00Z", "2026-04-18T10:15:00Z",
         "proj-cite", "closed", "Implemented citation endpoints."),
    )
    conn.execute(
        "INSERT INTO session_summaries "
        "(id, session_id, project, summary, next_steps, pitfalls, ended_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("ss-1", "s-citation", "proj-cite",
         "Closed out citation work",
         '["Test on live DB", "Update docs"]',
         '["Remember shell-escaping in hooks"]',
         "2026-04-18T10:15:00Z"),
    )

    conn.execute("PRAGMA journal_mode=WAL")
    conn.commit()
    conn.close()

    sys.path.insert(0, str(root / "src"))
    import dashboard  # noqa: E402
    dashboard.DB_PATH = db_path

    class _ThreadedHTTP(HTTPServer):
        daemon_threads = True

    port = _free_port()
    server = _ThreadedHTTP(("127.0.0.1", port), dashboard.DashboardHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    time.sleep(0.1)

    yield f"http://127.0.0.1:{port}"

    server.shutdown()
    server.server_close()
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)


def _get(url: str) -> tuple[int, dict]:
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8")
        try:
            return e.code, json.loads(body)
        except Exception:
            return e.code, {"error": body}


def test_api_knowledge_by_id_returns_json(citation_server: str) -> None:
    status, data = _get(f"{citation_server}/api/knowledge/1")
    assert status == 200
    # Stable citation shape — every field present:
    for key in ("id", "type", "content", "context", "project",
                "tags", "created_at", "session_id", "related"):
        assert key in data, f"missing field: {key}"
    assert data["id"] == 1
    assert data["type"] == "solution"
    assert data["project"] == "proj-cite"
    assert data["session_id"] == "s-citation"
    assert data["tags"] == ["reusable"]
    assert isinstance(data["related"], list)


def test_api_knowledge_not_found_returns_404(citation_server: str) -> None:
    status, data = _get(f"{citation_server}/api/knowledge/99999")
    assert status == 404
    assert data["error"] == "not_found"
    assert "message" in data
    assert "99999" in data["message"]


def test_api_knowledge_includes_related_edges(citation_server: str) -> None:
    status, data = _get(f"{citation_server}/api/knowledge/1")
    assert status == 200
    # Record 2 was linked via graph_edges
    peer_ids = [r["id"] for r in data["related"]]
    assert 2 in peer_ids
    via = {r["id"]: r["via"] for r in data["related"]}
    assert via[2] == "graph_edge"
    # Record 3 is unrelated
    assert 3 not in peer_ids


def test_api_session_by_id_returns_summary(citation_server: str) -> None:
    status, data = _get(f"{citation_server}/api/session/s-citation")
    assert status == 200
    for key in ("session_id", "summary", "next_steps",
                "pitfalls", "knowledge", "created_at"):
        assert key in data, f"missing field: {key}"
    assert data["session_id"] == "s-citation"
    assert "citation work" in data["summary"]
    assert "Test on live DB" in data["next_steps"]
    assert "Remember shell-escaping in hooks" in data["pitfalls"]
    # Knowledge records in this session are attached
    k_ids = [k["id"] for k in data["knowledge"]]
    assert 1 in k_ids and 2 in k_ids and 3 not in k_ids


def test_api_session_not_found_returns_404(citation_server: str) -> None:
    status, data = _get(f"{citation_server}/api/session/does-not-exist")
    assert status == 404
    assert data["error"] == "not_found"
    assert "message" in data


def test_knowledge_html_view_renders(citation_server: str) -> None:
    with urllib.request.urlopen(f"{citation_server}/knowledge/1", timeout=5) as r:
        assert r.status == 200
        body = r.read().decode("utf-8")
    assert "Knowledge" in body
    # The template embeds the ID into the page.
    assert 'const KID = "1"' in body
    # And links back to the JSON endpoint.
    assert "/api/knowledge/" in body


def test_session_html_view_renders(citation_server: str) -> None:
    with urllib.request.urlopen(f"{citation_server}/session/s-citation", timeout=5) as r:
        assert r.status == 200
        body = r.read().decode("utf-8")
    assert "Session" in body
    assert 'const SID = "s-citation"' in body
