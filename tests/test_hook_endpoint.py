"""POST /internal/hook/{cmd}: the server side of remote hooks mode (slice S1).

Covers the request envelope (404/400/413/415/500/202), each command's writes
against a real schema, the session-end rule "never read laptop paths", the
X-TAM-Session -> _REQUEST_SESSION -> _current_sid chain, and one end-to-end
run against a real HTTP server subprocess.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import hook_endpoint
from ambient.hooks import HookCall
from base_schema import apply_full_schema

httpx = pytest.importorskip("httpx")

SESSION = "hook-sess-1"


def make_call(session: str = SESSION, project: str = "proj", **payload) -> HookCall:
    return HookCall(payload=payload, project=project, session_id=session, cwd="/work/proj")


@pytest.fixture
def mem(tmp_path: Path) -> Path:
    memory_dir = tmp_path / "mem"
    memory_dir.mkdir()
    conn = sqlite3.connect(str(memory_dir / "memory.db"))
    try:
        apply_full_schema(conn)
        conn.commit()
    finally:
        conn.close()
    return memory_dir


def dispatch(command: str, call: HookCall, payload: dict, memory_dir: Path):
    return hook_endpoint.dispatch(command, call, payload, memory_dir)


# ── commands ─────────────────────────────────────────────────────────


def test_save_intent_writes_prompt_row(mem):
    call = make_call(prompt="fix the flaky test")
    status, _ = dispatch("save-intent", call, {}, mem)
    assert status == 200
    db = sqlite3.connect(str(mem / "memory.db"))
    try:
        row = db.execute("SELECT session_id, project, prompt FROM intents").fetchone()
    finally:
        db.close()
    assert row == (SESSION, "proj", "fix the flaky test")


def test_actions_batch_inserts_in_payload_order(mem):
    actions = [
        {"ts": "2026-11-08T10:00:00.000000Z", "tool": "Bash", "summary": "pytest -q", "status": "ok"},
        {"ts": "2026-11-08T10:00:01.000000Z", "tool": "Edit", "summary": "src/x.py :: new", "status": "failed",
         "exit_code": "2"},
        {"ts": "2026-11-08T10:00:02.000000Z", "tool": "Read", "summary": "skipped"},  # UNLOGGED_TOOLS
    ]
    status, output = dispatch("actions-batch", make_call(), {"actions": actions}, mem)
    assert (status, output) == (200, "")
    db = sqlite3.connect(str(mem / "memory.db"))
    try:
        rows = db.execute("SELECT tool, summary, status, exit_code FROM agent_actions "
                          "ORDER BY id").fetchall()
    finally:
        db.close()
    assert [r[0] for r in rows] == ["Bash", "Edit"]
    assert rows[1][1].startswith("src/x.py") and rows[1][2] == "failed" and rows[1][3] == 2


def test_error_capture_logs_error(mem):
    payload = {"action": "error", "description": "boom", "category": "code_error",
               "severity": "high", "project": "proj"}
    status, output = dispatch("error-capture", make_call(), payload, mem)
    assert (status, output) == (200, "")
    db = sqlite3.connect(str(mem / "memory.db"))
    try:
        row = db.execute("SELECT category, severity, status FROM errors").fetchone()
    finally:
        db.close()
    assert row == ("code_error", "high", "open")


def test_error_capture_unknown_action_raises(mem):
    with pytest.raises(Exception, match="unknown error-capture action"):
        dispatch("error-capture", make_call(), {"action": "nope"}, mem)


def test_session_start_returns_text_or_empty(mem):
    status, output = dispatch("session-start", make_call(source="startup"), {}, mem)
    assert status == 200 and isinstance(output, str)


def test_session_end_uses_messages_and_never_reads_files(mem, monkeypatch):
    """session-end gets pre-extracted text; a bogus transcript_path must be ignored."""
    monkeypatch.setenv("MEMORY_LLM_ENABLED", "false")
    call = make_call("e2e-sess", transcript_path="/nonexistent/laptop/transcript.jsonl")
    payload = {
        "user_texts": ["please fix the bug"],
        "assistant_text": "fixed it",
        "episode_messages": [
            {"role": "user", "text": "please fix the bug"},
            {"role": "assistant", "text": "on it"},
            {"role": "user", "text": "thanks"},
            {"role": "assistant", "text": "done"},
        ],
        "branch": "feature/x",
        "project": "proj",
        "reason": "clear",
    }
    status, output = dispatch("session-end", call, payload, mem)
    assert (status, output) == (202, "")
    hook_endpoint.join_background_threads(timeout=60)

    db = sqlite3.connect(str(mem / "memory.db"))
    try:
        episode = db.execute("SELECT session_id FROM episodes WHERE session_id='e2e-sess'").fetchone()
        session = db.execute("SELECT ended_at, branch FROM sessions WHERE id='e2e-sess'").fetchone()
        autosave = db.execute("SELECT COUNT(*) FROM knowledge WHERE tags LIKE '%session-autosave%'").fetchone()
    finally:
        db.close()
    # The episode exists only because episode_messages was used: the transcript
    # path does not exist and find_transcript would have found nothing.
    assert episode is not None
    assert session is not None and session[0] is not None and session[1] == "feature/x"
    assert autosave[0] >= 1


# ── HTTP envelope (in-process Starlette) ─────────────────────────────


@pytest.fixture
def client(mem, monkeypatch):
    monkeypatch.setenv("TAM_MEMORY_DIR", str(mem))
    monkeypatch.setenv("MEMORY_LLM_ENABLED", "false")
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient

    app = Starlette(routes=[Route("/internal/hook/{cmd}", hook_endpoint.handle, methods=["POST"])])
    return TestClient(app)


def post(client, cmd, **kwargs):
    return client.post(f"/internal/hook/{cmd}", **kwargs)


def test_unknown_command_is_404(client):
    response = post(client, "bogus", json={"call": {"payload": {}}, "payload": {}})
    assert response.status_code == 404 and response.json()["ok"] is False


def test_malformed_json_is_400(client):
    response = post(client, "save-intent", content=b"{nope",
                   headers={"Content-Type": "application/json"})
    assert response.status_code == 400 and response.json()["ok"] is False


def test_missing_call_is_400(client):
    response = post(client, "save-intent", json={"payload": {}})
    assert response.status_code == 400


def test_non_json_content_type_is_415(client):
    response = post(client, "save-intent", data="{}", headers={"Content-Type": "text/plain"})
    assert response.status_code == 415


def test_oversize_body_is_413(client):
    big = json.dumps({"call": {"payload": {"pad": "x" * (hook_endpoint.BODY_LIMIT_BYTES + 1)}},
                      "payload": {}}).encode()
    response = post(client, "save-intent", content=big,
                    headers={"Content-Type": "application/json"})
    assert response.status_code == 413


def test_handler_error_is_500_json(client, monkeypatch):
    def boom(*args):
        raise RuntimeError("kaput")

    monkeypatch.setattr(hook_endpoint, "dispatch", boom)
    response = post(client, "save-intent", json={"call": {"payload": {}}, "payload": {}})
    assert response.status_code == 500
    assert response.json() == {"ok": False, "error": "RuntimeError: kaput"}


def test_timeout_is_500_json(client, monkeypatch):
    monkeypatch.setattr(hook_endpoint, "dispatch", lambda *a: time.sleep(30))
    monkeypatch.setenv(hook_endpoint.TIMEOUT_ENV, "0.2")
    response = post(client, "save-intent", json={"call": {"payload": {}}, "payload": {}})
    assert response.status_code == 500 and "TimeoutError" in response.json()["error"]


def test_save_intent_roundtrip_over_http(client):
    body = {"call": {"payload": {"prompt": "ship it"}, "project": "proj",
                     "session_id": SESSION, "cwd": "/work/proj"},
            "payload": {}}
    response = post(client, "save-intent", json=body)
    assert response.status_code == 200 and response.json() == {"ok": True, "output": ""}
    db = sqlite3.connect(str(Path(os.environ["TAM_MEMORY_DIR"]) / "memory.db"))
    try:
        row = db.execute("SELECT session_id, prompt FROM intents").fetchone()
    finally:
        db.close()
    assert row == (SESSION, "ship it")


# ── X-TAM-Session -> _REQUEST_SESSION -> _current_sid ────────────────


def test_request_session_scope_sets_and_resets():
    import server

    seen = {}

    async def inner(scope, receive, send):
        seen["sid"] = server._REQUEST_SESSION.get()
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    sent = []

    async def send(message):
        sent.append(message)

    scope = {"type": "http", "path": "/mcp", "headers": [(b"x-tam-session", b"client-session-1")]}
    asyncio.run(server._RequestSessionScope(inner)(scope, receive, send))
    assert seen["sid"] == "client-session-1"
    assert server._REQUEST_SESSION.get() is None

    asyncio.run(server._RequestSessionScope(inner)(
        {"type": "http", "path": "/mcp", "headers": []}, receive, send))
    assert seen["sid"] is None


def test_current_sid_prefers_request_session(monkeypatch):
    import server

    monkeypatch.setattr(server, "SID", "bootstrap-sid")
    monkeypatch.delenv("CLAUDE_CODE_SESSION_ID", raising=False)
    assert server._current_sid() == "bootstrap-sid"
    token = server._REQUEST_SESSION.set("header-sid")
    try:
        assert server._current_sid() == "header-sid"
        assert server._session_id() == "header-sid"
    finally:
        server._REQUEST_SESSION.reset(token)


def test_tool_call_records_request_session(tmp_path, monkeypatch):
    """A tool call inside a request scope writes rows under the header's session id."""
    import server

    monkeypatch.setattr(server, "MEMORY_DIR", tmp_path)
    store = server.Store()
    monkeypatch.setattr(server, "store", store)
    monkeypatch.setattr(server, "SID", "bootstrap-sid")
    token = server._REQUEST_SESSION.set("hook-session-9")
    try:
        result = json.loads(asyncio.run(
            server.call_tool("save_intent", {"prompt": "remember this", "project": "p"})
        )[0].text)
    finally:
        server._REQUEST_SESSION.reset(token)
        store.db.close()
    assert result["saved"] is True
    db = sqlite3.connect(str(tmp_path / "memory.db"))
    try:
        row = db.execute("SELECT session_id FROM intents ORDER BY id DESC LIMIT 1").fetchone()
    finally:
        db.close()
    assert row[0] == "hook-session-9"


# ── end-to-end against a real server subprocess ──────────────────────

READY_TIMEOUT_S = 120
STOP_TIMEOUT_S = 30


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def http_server(tmp_path_factory):
    mem = tmp_path_factory.mktemp("hook_e2e_mem")
    port = free_port()
    env = dict(os.environ)
    env.update(TAM_MEMORY_DIR=str(mem), MEMORY_MODE="fast", MEMORY_LLM_ENABLED="false",
               MCP_TRANSPORT="http", MCP_HTTP_HOST="127.0.0.1", MCP_HTTP_PORT=str(port),
               MCP_HTTP_WORKERS="1")
    proc = subprocess.Popen([sys.executable, str(ROOT / "src" / "server.py")], env=env,
                            cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.time() + READY_TIMEOUT_S
    while True:
        assert proc.poll() is None, "server exited during startup"
        assert time.time() < deadline, "server not ready"
        try:
            httpx.get(f"http://127.0.0.1:{port}/healthz", timeout=2).raise_for_status()
            break
        except httpx.HTTPError:
            time.sleep(0.5)
    yield mem, port
    if proc.poll() is None:
        proc.terminate()
        proc.wait(timeout=STOP_TIMEOUT_S)


def test_session_end_over_http_e2e(http_server):
    mem, port = http_server
    body = {
        "call": {"payload": {"transcript_path": "/nonexistent/laptop/t.jsonl",
                             "session_id": "e2e-sess", "cwd": "/tmp"},
                 "project": "e2eproj", "session_id": "e2e-sess", "cwd": "/tmp"},
        "payload": {
            "user_texts": ["do the thing"], "assistant_text": "done",
            "episode_messages": [
                {"role": "user", "text": "do the thing"},
                {"role": "assistant", "text": "working"},
                {"role": "user", "text": "ok"},
                {"role": "assistant", "text": "done"},
            ],
            "branch": "main", "project": "e2eproj", "reason": "clear",
        },
    }
    response = httpx.post(f"http://127.0.0.1:{port}/internal/hook/session-end",
                          json=body, timeout=30)
    assert response.status_code == 202 and response.json() == {"ok": True, "output": ""}

    db_path = mem / "memory.db"
    deadline = time.time() + 60
    episode = session_row = None
    while time.time() < deadline:
        db = sqlite3.connect(str(db_path))
        try:
            episode = db.execute(
                "SELECT id FROM episodes WHERE session_id='e2e-sess'").fetchone()
            session_row = db.execute(
                "SELECT ended_at FROM sessions WHERE id='e2e-sess'").fetchone()
        except sqlite3.OperationalError:
            pass
        finally:
            db.close()
        if episode and session_row and session_row[0]:
            break
        time.sleep(0.5)
    assert episode is not None, "episode row missing after session-end"
    assert session_row is not None and session_row[0] is not None


def test_unknown_hook_command_over_http(http_server):
    _mem, port = http_server
    response = httpx.post(f"http://127.0.0.1:{port}/internal/hook/nope",
                          json={"call": {"payload": {}}, "payload": {}}, timeout=10)
    assert response.status_code == 404
