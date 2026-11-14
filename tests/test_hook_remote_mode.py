"""Remote hooks mode (TAM_REMOTE_URL): the client side of POST /internal/hook/{cmd}.

Covers the slice-S2 contract: hooks never open memory.db and never import
server/memory_core on the remote path, PostToolUse only appends to the local
spool, Stop/UserPromptSubmit/SessionEnd flush it first, every failure exits 0,
and hooks/tam-headers prints the MCP headersHelper JSON.
"""
from __future__ import annotations

import importlib.util
import io
import json
import os
import re
import socket
import sqlite3
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "hooks" / "lib" / "tam_hook.py"
TAM_HEADERS = ROOT / "hooks" / "tam-headers"

httpx = pytest.importorskip("httpx")


def load_helper() -> ModuleType:
    spec = importlib.util.spec_from_file_location("tam_hook_remote_test", HELPER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tam_hook = load_helper()

SESSION = "s2-sess"
CWD = "workdir"

# Modules a remote-mode hook process must never load: anything that can open
# memory.db or pull in the server runtime.
FORBIDDEN_MODULES = {
    "server", "memory_core", "ambient", "hook_endpoint", "intents",
    "auto_episode_capture", "auto_session_end", "auto_session_save",
    "auto_self_improve", "auto_extract_active", "session_continuity",
    "memory_systems", "llm_provider", "rule_surfaces", "embedding_backfill",
}


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def client_env(tmp_path: Path, remote_url: str, **extra: str) -> dict[str, str]:
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    (tmp_path / "workdir").mkdir(exist_ok=True)
    env = dict(os.environ)
    env.update({
        "HOME": str(home),
        "TAM_REMOTE_URL": remote_url,
        "TAM_REMOTE_TOKEN": "test-token",
        "XDG_CACHE_HOME": str(tmp_path / "cache"),
        "TAM_MEMORY_DIR": str(tmp_path / "client-mem"),
        "CLAUDE_MEMORY_INSTALL_DIR": str(ROOT),
        "MEMORY_PROJECT": "e2eproj",
    })
    env.update(extra)
    return env


def spool_file(tmp_path: Path, session: str = SESSION) -> Path:
    return tmp_path / "cache" / "tam" / "spool" / f"{session}.jsonl"


def run_hook(args: list[str], env: dict[str, str], payload: dict | None = None,
             stdin_text: str | None = None, timeout: float = 30) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(HELPER), *args],
        input=stdin_text if stdin_text is not None else json.dumps(payload or {}),
        capture_output=True, text=True, env=env, timeout=timeout, check=False)


def post_tool_payload(tool: str, tool_input: dict, tool_use_id: str, cwd: Path,
                      event: str = "PostToolUse", **extra) -> dict:
    payload = {"hook_event_name": event, "tool_name": tool, "tool_input": tool_input,
               "session_id": SESSION, "cwd": str(cwd), "tool_use_id": tool_use_id}
    payload.update(extra)
    return payload


def hook_payload(session: str, cwd: Path, **extra) -> dict:
    payload = {"session_id": session, "cwd": str(cwd)}
    payload.update(extra)
    return payload


# ── import isolation ─────────────────────────────────────────────────


def test_remote_record_action_imports_no_server_modules(tmp_path):
    """The remote PostToolUse path must not pull in anything that opens memory.db."""
    env = client_env(tmp_path, "http://127.0.0.1:9")
    payload_file = tmp_path / "payload.json"
    payload_file.write_text(json.dumps(post_tool_payload(
        "Bash", {"command": "ls"}, "toolu_iso", tmp_path / "workdir")))
    code = (
        "import sys, json; sys.path.insert(0, {helper!r}); import tam_hook; "
        "sys.stdin = open({payload!r}); "
        "rc = tam_hook.main({args!r}); "
        "print('TAM_RC=' + str(rc)); "
        "print('TAM_MODULES=' + json.dumps(sorted(sys.modules)))"
    ).format(helper=str(HELPER.parent), payload=str(payload_file),
             args=["record-action", "--src", str(ROOT / "src"),
                   "--memory-dir", str(tmp_path / "client-mem")])
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          env=env, timeout=30, check=False)
    assert proc.returncode == 0, proc.stderr
    out = dict(line.split("=", 1) for line in proc.stdout.splitlines() if "=" in line)
    assert out.get("TAM_RC") == "0"
    modules = set(json.loads(out["TAM_MODULES"]))
    loaded = {name.split(".")[0] for name in modules} | modules
    assert not (loaded & FORBIDDEN_MODULES), sorted(loaded & FORBIDDEN_MODULES)
    # record-action only spools; urllib.request stays lazy inside remote_post.
    assert "urllib.request" not in modules


# ── spool: PostToolUse is local-only ─────────────────────────────────


def test_remote_record_action_spools_without_network(monkeypatch, tmp_path):
    """PostToolUse appends one JSONL line and does no network I/O at all."""
    monkeypatch.setenv("TAM_REMOTE_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    calls = []

    def no_network(*args, **kwargs):
        calls.append(args)
        raise AssertionError("PostToolUse must not do network I/O")

    monkeypatch.setattr(urllib.request, "urlopen", no_network)
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(
        post_tool_payload("Bash", {"command": "pytest -q"}, "toolu_net",
                          tmp_path / "workdir"))))
    rc = tam_hook.main(["record-action", "--src", str(ROOT / "src"),
                        "--memory-dir", str(tmp_path / "client-mem")])
    assert rc == 0
    assert calls == []
    lines = spool_file(tmp_path).read_text().splitlines()
    assert len(lines) == 1
    action = json.loads(lines[0])
    assert action["tool"] == "Bash" and action["tool_use_id"] == "toolu_net"
    assert "pytest -q" in action["summary"] and action["status"] == "ok"


def test_remote_record_action_skips_unlogged_tools(tmp_path):
    env = client_env(tmp_path, "http://127.0.0.1:9")
    proc = run_hook(["record-action", "--src", str(ROOT / "src"),
                     "--memory-dir", str(tmp_path / "client-mem")],
                    env=env, payload=post_tool_payload("Read", {"file_path": "/x"},
                                                       "toolu_r", tmp_path / "workdir"))
    assert proc.returncode == 0
    assert not spool_file(tmp_path).exists()


# ── server down: every hook exits 0, spool survives ──────────────────


def test_remote_server_down_keeps_spool_and_recovers(http_server, tmp_path):
    mem, port = http_server
    dead_port = free_port()
    env = client_env(tmp_path, f"http://127.0.0.1:{dead_port}")
    cwd = tmp_path / "workdir"
    session = "s2-down"

    for tuid in ("tu_a", "tu_b"):
        payload = hook_payload(session, cwd, hook_event_name="PostToolUse",
                               tool_name="Bash", tool_input={"command": f"echo {tuid}"},
                               tool_use_id=tuid)
        proc = run_hook(["record-action", "--src", str(ROOT / "src"),
                         "--memory-dir", str(tmp_path / "client-mem")], env=env, payload=payload)
        assert proc.returncode == 0
    assert len(spool_file(tmp_path, session).read_text().splitlines()) == 2

    started = time.monotonic()
    for args, payload in (
        (["stop", "--memory-dir", str(tmp_path / "client-mem"),
          "--recovery-dir", str(tmp_path / "client-mem" / "recovery"),
          "--src", str(ROOT / "src")],
         hook_payload(session, cwd, hook_event_name="Stop")),
        (["prompt-context", "--src", str(ROOT / "src"),
          "--memory-dir", str(tmp_path / "client-mem"), "--emit", "UserPromptSubmit"],
         hook_payload(session, cwd, hook_event_name="UserPromptSubmit", prompt="hi")),
        (["session-start-context", "--db", str(tmp_path / "client-mem" / "memory.db"),
          "--project", "e2eproj", "--cwd", str(cwd), "--session", session,
          "--source", "startup", "--recovery-dir", str(tmp_path / "client-mem" / "recovery")],
         hook_payload(session, cwd, hook_event_name="SessionStart", source="startup")),
        (["session-end", "--src", str(ROOT / "src"),
          "--recovery-dir", str(tmp_path / "client-mem" / "recovery"),
          "--reason-text", "Session ended (test)"],
         hook_payload(session, cwd, hook_event_name="SessionEnd", reason="test")),
    ):
        proc = run_hook(args, env=env, payload=payload)
        assert proc.returncode == 0, (args[0], proc.stderr)
        assert time.monotonic() - started < 5.0, f"{args[0]} took too long with the server down"
        started = time.monotonic()
    # Nothing was delivered: the spool must still hold both actions.
    pending = pending_spool_actions(tmp_path)
    assert sorted(a["tool_use_id"] for a in pending) == ["tu_a", "tu_b"]

    # The server comes up: the next Stop flushes the spool exactly once.
    live_env = client_env(tmp_path, f"http://127.0.0.1:{port}")
    proc = run_hook(["stop", "--memory-dir", str(tmp_path / "client-mem"),
                     "--recovery-dir", str(tmp_path / "client-mem" / "recovery"),
                     "--src", str(ROOT / "src")],
                    env=live_env, payload=hook_payload(session, cwd, hook_event_name="Stop"))
    assert proc.returncode == 0
    assert pending_spool_actions(tmp_path) == []

    deadline = time.time() + 30
    rows = []
    while time.time() < deadline:
        db = sqlite3.connect(str(mem / "memory.db"))
        try:
            rows = db.execute(
                "SELECT tool, summary FROM agent_actions WHERE session_id=? ORDER BY id",
                (session,)).fetchall()
        finally:
            db.close()
        if len(rows) == 2:
            break
        time.sleep(0.5)
    assert [r[0] for r in rows] == ["Bash", "Bash"]
    assert "echo tu_a" in rows[0][1] and "echo tu_b" in rows[1][1]

    # A second Stop sends nothing again: the flush deleted the spool.
    proc = run_hook(["stop", "--memory-dir", str(tmp_path / "client-mem"),
                     "--recovery-dir", str(tmp_path / "client-mem" / "recovery"),
                     "--src", str(ROOT / "src")],
                    env=live_env, payload=hook_payload(session, cwd, hook_event_name="Stop"))
    assert proc.returncode == 0
    db = sqlite3.connect(str(mem / "memory.db"))
    try:
        count = db.execute("SELECT COUNT(*) FROM agent_actions WHERE session_id=?",
                           (session,)).fetchone()[0]
    finally:
        db.close()
    assert count == 2


# ── end-to-end against a real S1 server subprocess ───────────────────

READY_TIMEOUT_S = 120
STOP_TIMEOUT_S = 30


@pytest.fixture(scope="module")
def http_server(tmp_path_factory):
    mem = tmp_path_factory.mktemp("hook_remote_mem")
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


def write_transcript(path: Path) -> None:
    lines = [
        {"type": "user", "message": {"role": "user", "content": "fix the flaky test"}},
        {"type": "assistant", "message": {"role": "assistant",
                                          "content": [{"type": "text", "text": "working on it"}]}},
        {"type": "user", "message": {"role": "user", "content": "thanks"}},
        {"type": "assistant", "message": {"role": "assistant",
                                          "content": [{"type": "text", "text": "done"}]}},
    ]
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n")


def test_remote_e2e_session_flow(http_server, tmp_path):
    """session-start -> prompt -> 3 tool calls -> stop -> session-end over HTTP."""
    mem, port = http_server
    env = client_env(tmp_path, f"http://127.0.0.1:{port}")
    cwd = tmp_path / "workdir"
    client_mem = tmp_path / "client-mem"
    session = "s2-e2e"

    proc = run_hook(
        ["session-start-context", "--db", str(client_mem / "memory.db"),
         "--project", "e2eproj", "--cwd", str(cwd), "--session", session,
         "--source", "startup", "--recovery-dir", str(client_mem / "recovery")],
        env=env, payload=hook_payload(session, cwd, hook_event_name="SessionStart", source="startup"))
    assert proc.returncode == 0, proc.stderr

    prompt = {"hook_event_name": "UserPromptSubmit", "prompt": "fix the flaky test",
              "session_id": session, "cwd": str(cwd)}
    proc = run_hook(["prompt-context", "--src", str(ROOT / "src"),
                     "--memory-dir", str(client_mem), "--emit", "UserPromptSubmit"],
                    env=env, payload=prompt)
    assert proc.returncode == 0, proc.stderr
    intent_file = tmp_path / "intent.json"
    intent_file.write_text(json.dumps(prompt))
    proc = run_hook(["save-intent", "--src", str(ROOT / "src"),
                     "--db", str(client_mem / "memory.db"), "--input", str(intent_file)],
                    env=env, stdin_text="")
    assert proc.returncode == 0, proc.stderr

    for tool, tool_input, tuid, event, response in (
        ("Bash", {"command": "pytest -q tests/test_x.py"}, "toolu_1", "PostToolUse", {}),
        ("Edit", {"file_path": "/tmp/s2proj/src/x.py", "new_string": "fixed code"},
         "toolu_2", "PostToolUse", {}),
        ("Bash", {"command": "make deploy"}, "toolu_3", "PostToolUse",
         {"exit_code": "2", "stderr": "boom"}),
    ):
        payload = {"hook_event_name": event, "tool_name": tool, "tool_input": tool_input,
                   "tool_response": response, "session_id": session, "cwd": str(cwd),
                   "tool_use_id": tuid}
        proc = run_hook(["record-action", "--src", str(ROOT / "src"),
                         "--memory-dir", str(client_mem)], env=env, payload=payload)
        assert proc.returncode == 0, proc.stderr
    assert len(spool_file(tmp_path, session).read_text().splitlines()) == 3

    proc = run_hook(["stop", "--memory-dir", str(client_mem),
                     "--recovery-dir", str(client_mem / "recovery"),
                     "--src", str(ROOT / "src")],
                    env=env, payload=hook_payload(session, cwd, hook_event_name="Stop"))
    assert proc.returncode == 0, proc.stderr
    assert not spool_file(tmp_path, session).exists()

    transcript = tmp_path / "transcript.jsonl"
    write_transcript(transcript)
    proc = run_hook(["session-end", "--src", str(ROOT / "src"),
                     "--recovery-dir", str(client_mem / "recovery"),
                     "--reason-text", "User ran /clear"],
                    env=env, payload=hook_payload(
                        session, cwd, hook_event_name="SessionEnd", reason="clear",
                        transcript_path=str(transcript)))
    assert proc.returncode == 0, proc.stderr

    db_path = mem / "memory.db"
    deadline = time.time() + 60
    actions = intents = turn = summary = episode = None
    while time.time() < deadline:
        db = sqlite3.connect(str(db_path))
        try:
            actions = db.execute(
                "SELECT tool, summary, status, exit_code FROM agent_actions "
                "WHERE session_id=? ORDER BY id", (session,)).fetchall()
            intents = db.execute(
                "SELECT prompt FROM intents WHERE session_id=?", (session,)).fetchall()
            turn = db.execute(
                "SELECT content FROM knowledge WHERE session_id=? "
                "AND tags LIKE '%turn-autosave%'", (session,)).fetchall()
            summary = db.execute(
                "SELECT summary FROM session_summaries WHERE session_id=?",
                (session,)).fetchall()
            episode = db.execute(
                "SELECT id FROM episodes WHERE session_id=?", (session,)).fetchall()
        except sqlite3.OperationalError:
            pass
        finally:
            db.close()
        if actions and intents and turn and summary and episode:
            break
        time.sleep(0.5)

    assert [a[0] for a in actions] == ["Bash", "Edit", "Bash"]
    assert "pytest -q tests/test_x.py" in actions[0][1]
    assert actions[0][2] == "ok"
    assert "x.py" in actions[1][1] and "fixed code" in actions[1][1]
    assert "make deploy" in actions[2][1] and actions[2][2] == "failed" and actions[2][3] == 2
    assert [i[0] for i in intents] == ["fix the flaky test"]
    assert len(turn) == 1 and turn[0][0].startswith("Turn summary")
    assert len(summary) == 1 and "Session ended" in summary[0][0]
    assert len(episode) == 1


# ── tam-headers (MCP headersHelper) ──────────────────────────────────


def test_tam_headers_prints_json_headers(tmp_path):
    env = dict(os.environ, TAM_REMOTE_TOKEN="tok-123", CLAUDE_CODE_SESSION_ID="sess-9",
               HOME=str(tmp_path / "home"))
    (tmp_path / "home").mkdir(exist_ok=True)
    proc = subprocess.run([str(TAM_HEADERS)], capture_output=True, text=True,
                          env=env, timeout=15, check=False)
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout) == {"Authorization": "Bearer tok-123",
                                       "X-TAM-Session": "sess-9"}


# ── flush chunking: UTF-8 byte budget, atomic take, sweep ────────────


def spool_env(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("TAM_REMOTE_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("TAM_REMOTE_TOKEN", "test-token")
    monkeypatch.setenv("MEMORY_PROJECT", "e2eproj")


def spool_action(tuid: str, summary: str = "cmd", session: str = SESSION,
                 project: str = "e2eproj", cwd: str = "/w") -> dict:
    return {"tool": "Bash", "summary": summary, "status": "ok", "exit_code": None,
            "session_id": session, "project": project, "cwd": cwd, "tool_use_id": tuid}


def batch_call(session: str = SESSION) -> dict:
    return {"payload": {"prompt": "ignored"}, "project": "e2eproj",
            "session_id": session, "cwd": "/w"}


def pending_spool_actions(tmp_path: Path) -> list[dict]:
    """Every action still waiting in the spool dir (.jsonl and .sending.*)."""
    spool_dir = tmp_path / "cache" / "tam" / "spool"
    actions = []
    for pattern in ("*.jsonl", "*.sending.*"):
        for path in sorted(spool_dir.glob(pattern)):
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    actions.append(json.loads(line))
    return actions


def test_remote_flush_actions_chunks_and_strips_call_payload(monkeypatch, tmp_path):
    """401 actions flush as [400, 1] and the batch call carries no hook payload."""
    spool_env(monkeypatch, tmp_path)
    posts = []

    def fake_post(command, call, payload, timeout):
        posts.append((command, call, payload))
        return True, ""

    monkeypatch.setattr(tam_hook, "remote_post", fake_post)
    for i in range(401):
        tam_hook.spool_append(SESSION, spool_action(f"tu_{i}", f"cmd {i}"))
    assert tam_hook.remote_flush_own(batch_call(), time.monotonic() + 30) is True
    assert [len(p["actions"]) for _c, _call, p in posts] == [400, 1]
    assert all(command == "actions-batch" for command, _c, _p in posts)
    assert all(call["payload"] == {} for _c, call, _p in posts)
    assert pending_spool_actions(tmp_path) == []


def test_remote_flush_failed_chunk_keeps_the_tail(monkeypatch, tmp_path):
    """A failed second chunk leaves exactly the unacknowledged tail in the spool."""
    spool_env(monkeypatch, tmp_path)
    attempts = []

    def flaky_post(command, call, payload, timeout):
        attempts.append(len(payload["actions"]))
        return len(attempts) == 1, ""

    monkeypatch.setattr(tam_hook, "remote_post", flaky_post)
    for i in range(401):
        tam_hook.spool_append(SESSION, spool_action(f"tu_{i}", f"cmd {i}"))
    assert tam_hook.remote_flush_own(batch_call(), time.monotonic() + 30) is False
    assert attempts == [400, 1]
    pending = pending_spool_actions(tmp_path)
    assert [a["tool_use_id"] for a in pending] == ["tu_400"]


def test_remote_flush_actions_budgets_utf8_bytes(monkeypatch, tmp_path):
    """Cyrillic summaries count as UTF-8 bytes: every POST body stays under 256 KB."""
    spool_env(monkeypatch, tmp_path)
    bodies = []

    def fake_post(command, call, payload, timeout):
        bodies.append(len(json.dumps({"call": call, "payload": payload},
                                     ensure_ascii=False).encode("utf-8")))
        return True, ""

    monkeypatch.setattr(tam_hook, "remote_post", fake_post)
    for i in range(400):
        tam_hook.spool_append(SESSION, spool_action(f"tu_{i}", "привет мир " * 30))
    assert tam_hook.remote_flush_own(batch_call(), time.monotonic() + 30) is True
    assert len(bodies) >= 2, "400 multibyte actions must not fit one 256 KB body"
    assert all(size <= tam_hook.REMOTE_BODY_LIMIT_BYTES for size in bodies), bodies


def test_remote_flush_append_during_post_survives(monkeypatch, tmp_path):
    """A PostToolUse landing while the flush POSTs starts a fresh spool; the
    own-session re-list picks it up before the flush returns."""
    spool_env(monkeypatch, tmp_path)
    late = spool_action("tu_late", "late cmd")
    posts = []
    appended = False

    def fake_post(command, call, payload, timeout):
        nonlocal appended
        posts.append(payload["actions"])
        if not appended:
            appended = True
            tam_hook.spool_append(SESSION, late)
        return True, ""

    monkeypatch.setattr(tam_hook, "remote_post", fake_post)
    tam_hook.spool_append(SESSION, spool_action("tu_first", "first cmd"))
    assert tam_hook.remote_flush_own(batch_call(), time.monotonic() + 30) is True
    sent = [a["tool_use_id"] for actions in posts for a in actions]
    assert sent == ["tu_first", "tu_late"]
    assert pending_spool_actions(tmp_path) == []


def test_remote_flush_sweeps_stranded_spools(http_server, tmp_path):
    """Stale foreign spools flush under their own session AND project, after the
    follow-up POST; .sending leftovers first, then .jsonl oldest first."""
    mem, port = http_server
    env = client_env(tmp_path, f"http://127.0.0.1:{port}")
    cwd = tmp_path / "workdir"
    current = "s2-sweep"
    spool_dir = tmp_path / "cache" / "tam" / "spool"
    spool_dir.mkdir(parents=True)
    (spool_dir / "deadbeef.sending.31337").write_text(
        json.dumps(spool_action("tu_dead", "stranded cmd", session="",
                                project="deadproj")) + "\n",
        encoding="utf-8")
    other_lines = [spool_action(f"tu_o{i}", f"other {i}", session="othersess",
                                project="othersproj")
                   for i in range(2)]
    (spool_dir / "othersess.jsonl").write_text(
        "".join(json.dumps(a) + "\n" for a in other_lines), encoding="utf-8")
    stale = time.time() - tam_hook.SPOOL_STALE_SEC - 60
    for path in spool_dir.iterdir():
        os.utime(path, (stale, stale))

    proc = run_hook(["record-action", "--src", str(ROOT / "src"),
                     "--memory-dir", str(tmp_path / "client-mem")],
                    env=env, payload=post_tool_payload(
                        "Bash", {"command": "echo curr"}, "tu_curr", cwd,
                        session_id=current))
    assert proc.returncode == 0, proc.stderr
    proc = run_hook(["prompt-context", "--src", str(ROOT / "src"),
                     "--memory-dir", str(tmp_path / "client-mem"), "--emit", "UserPromptSubmit"],
                    env=env, payload=hook_payload(
                        current, cwd, hook_event_name="UserPromptSubmit", prompt="hi"))
    assert proc.returncode == 0, proc.stderr

    deadline = time.time() + 30
    rows = []
    while time.time() < deadline:
        db = sqlite3.connect(str(mem / "memory.db"))
        try:
            rows = db.execute(
                "SELECT session_id, tool, project FROM agent_actions "
                "WHERE session_id IN ('deadbeef','othersess',?) ORDER BY id",
                (current,)).fetchall()
        finally:
            db.close()
        if len(rows) == 4:
            break
        time.sleep(0.5)
    # Own session first (it gates the follow-up POST), then the stale sweep:
    # .sending leftover (session and project from the spooled line/filename),
    # then the older .jsonl - all under their own session ids and projects.
    assert rows == [(current, "Bash", "e2eproj"), ("deadbeef", "Bash", "deadproj"),
                    ("othersess", "Bash", "othersproj"),
                    ("othersess", "Bash", "othersproj")]
    assert not list(spool_dir.iterdir())


def test_remote_session_end_skips_post_and_keeps_spool_on_flush_failure(
        monkeypatch, tmp_path):
    """remote_session_end returns False without POSTing session-end when the
    flush failed; the spool keeps every pending action."""
    spool_env(monkeypatch, tmp_path)
    posts = []

    def failing_post(command, call, payload, timeout):
        posts.append(command)
        return False, ""

    monkeypatch.setattr(tam_hook, "remote_post", failing_post)
    tam_hook.spool_append("sess-end", spool_action("tu_se", "end cmd", session="sess-end"))
    transcript = tmp_path / "t.jsonl"
    write_transcript(transcript)
    recovery_dir = tmp_path / "recovery"
    payload = {"session_id": "sess-end", "cwd": str(tmp_path / "workdir"),
               "hook_event_name": "SessionEnd", "transcript_path": str(transcript)}
    ok = tam_hook.remote_session_end(payload, "e2eproj", ROOT / "src",
                                     "test reason", time.monotonic() + 30,
                                     recovery_dir)
    assert ok is False
    assert posts == ["actions-batch"], "session-end must not close before its actions land"
    assert (recovery_dir / "pending-sess-end.md").is_file(), \
        "the laptop recovery file is written before the flush, even when it fails"
    pending = pending_spool_actions(tmp_path)
    assert [a["tool_use_id"] for a in pending] == ["tu_se"]


def test_remote_sweep_sends_spooled_project_and_cwd(monkeypatch, tmp_path):
    """A stale foreign spool is sent under ITS session, project and cwd - never
    the sweeping hook's, which would mislabel the rows in another repo."""
    spool_env(monkeypatch, tmp_path)
    posts = []

    def fake_post(command, call, payload, timeout):
        posts.append((command, call))
        return True, ""

    monkeypatch.setattr(tam_hook, "remote_post", fake_post)
    spool_dir = tmp_path / "cache" / "tam" / "spool"
    spool_dir.mkdir(parents=True)
    foreign = spool_dir / "foreign1.jsonl"
    foreign.write_text(json.dumps(spool_action(
        "tu_f1", "foreign cmd", session="foreign1",
        project="p1proj", cwd="/p1cwd")) + "\n", encoding="utf-8")
    stale = time.time() - 700
    os.utime(foreign, (stale, stale))
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(hook_payload(
        "own-sess", tmp_path / "workdir",
        hook_event_name="UserPromptSubmit", prompt="hi"))))
    rc = tam_hook.main(["prompt-context", "--src", str(ROOT / "src"),
                        "--memory-dir", str(tmp_path / "client-mem"),
                        "--emit", "UserPromptSubmit"])
    assert rc == 0
    batches = [call for command, call in posts if command == "actions-batch"]
    assert len(batches) == 1
    assert batches[0]["session_id"] == "foreign1"
    assert batches[0]["project"] == "p1proj"
    assert batches[0]["cwd"] == "/p1cwd"


def test_remote_stop_flushes_own_sending_left_by_concurrent_hook(monkeypatch, tmp_path):
    """A <sid>.sending.<pid> of THIS session appearing mid-flush is delivered
    before turn-end: the turn boundary must not pass undelivered actions."""
    spool_env(monkeypatch, tmp_path)
    spool_dir = tmp_path / "cache" / "tam" / "spool"
    spool_dir.mkdir(parents=True)
    tam_hook.spool_append(SESSION, spool_action("tu_first", "first cmd"))
    posts = []
    injected = False

    def fake_post(command, call, payload, timeout):
        nonlocal injected
        posts.append(command)
        if command == "actions-batch" and not injected:
            injected = True
            # A concurrent hook of the same session took a fresh .jsonl aside.
            (spool_dir / f"{SESSION}.sending.4242").write_text(
                json.dumps(spool_action("tu_late", "late cmd")) + "\n",
                encoding="utf-8")
        return True, ""

    monkeypatch.setattr(tam_hook, "remote_post", fake_post)
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(hook_payload(
        SESSION, tmp_path / "workdir", hook_event_name="Stop"))))
    rc = tam_hook.main(["stop", "--memory-dir", str(tmp_path / "client-mem"),
                        "--recovery-dir", str(tmp_path / "client-mem" / "recovery"),
                        "--src", str(ROOT / "src")])
    assert rc == 0
    assert posts == ["actions-batch", "actions-batch", "turn-end"]
    assert pending_spool_actions(tmp_path) == []


def test_remote_session_end_own_spool_beats_foreign_backlog(monkeypatch, tmp_path):
    """Own session-end is POSTed within the deadline even when a large stale
    foreign backlog would eat the whole budget under a global sweep."""
    spool_env(monkeypatch, tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir(exist_ok=True)
    spool_dir = tmp_path / "cache" / "tam" / "spool"
    spool_dir.mkdir(parents=True)
    stale = time.time() - 700
    for i in range(30):
        path = spool_dir / f"foreign{i}.jsonl"
        path.write_text("".join(
            json.dumps(spool_action(f"tu_f{i}_{j}", f"cmd {j}",
                                    session=f"foreign{i}", project=f"fp{i}",
                                    cwd="/f")) + "\n"
            for j in range(tam_hook.SPOOL_ACTION_LIMIT)), encoding="utf-8")
        os.utime(path, (stale, stale))
    tam_hook.spool_append("own-end", spool_action("tu_own", "own cmd", session="own-end"))
    posts = []

    def slow_post(command, call, payload, timeout):
        posts.append(command)
        time.sleep(0.05)
        return True, ""

    monkeypatch.setattr(tam_hook, "remote_post", slow_post)
    payload = {"session_id": "own-end", "cwd": str(tmp_path / "workdir"),
               "hook_event_name": "SessionEnd"}
    started = time.monotonic()
    ok = tam_hook.remote_session_end(payload, "e2eproj", ROOT / "src",
                                     "test reason", started + 1.0,
                                     tmp_path / "recovery")
    elapsed = time.monotonic() - started
    assert ok is True
    assert "session-end" in posts
    assert elapsed < 2.0


def test_remote_sweep_never_takes_a_fresh_foreign_spool(monkeypatch, tmp_path):
    """A foreign .jsonl younger than the stale threshold is left alone: its live
    session may still be appending through an fd opened before a take-aside."""
    spool_env(monkeypatch, tmp_path)
    posts = []

    def fake_post(command, call, payload, timeout):
        posts.append(command)
        return True, ""

    monkeypatch.setattr(tam_hook, "remote_post", fake_post)
    spool_dir = tmp_path / "cache" / "tam" / "spool"
    spool_dir.mkdir(parents=True)
    foreign = spool_dir / "live1.jsonl"
    foreign.write_text(json.dumps(spool_action(
        "tu_live", "live cmd", session="live1", project="lp", cwd="/l")) + "\n",
        encoding="utf-8")
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(hook_payload(
        "own-sess", tmp_path / "workdir",
        hook_event_name="UserPromptSubmit", prompt="hi"))))
    rc = tam_hook.main(["prompt-context", "--src", str(ROOT / "src"),
                        "--memory-dir", str(tmp_path / "client-mem"),
                        "--emit", "UserPromptSubmit"])
    assert rc == 0
    assert posts == ["prompt-context"]
    assert foreign.exists()
    assert [a["tool_use_id"] for a in pending_spool_actions(tmp_path)] == ["tu_live"]


def test_remote_flush_batches_per_project_run(http_server, tmp_path):
    """A session that changed project mid-spool gets one actions-batch call per
    consecutive (session, project, cwd) run: the server stores every action
    under the project its own line carried, not the file's first line's."""
    mem, port = http_server
    env = client_env(tmp_path, f"http://127.0.0.1:{port}")
    cwd = tmp_path / "workdir"
    session = "s2-runs"
    spool_dir = tmp_path / "cache" / "tam" / "spool"
    spool_dir.mkdir(parents=True)
    lines = [spool_action("tu_p1a", "in p1", session=session, project="p1proj", cwd="/p1"),
             spool_action("tu_p1b", "still p1", session=session, project="p1proj", cwd="/p1"),
             spool_action("tu_p2a", "cd p2", session=session, project="p2proj", cwd="/p2"),
             spool_action("tu_p2b", "still p2", session=session, project="p2proj", cwd="/p2")]
    (spool_dir / f"{session}.jsonl").write_text(
        "".join(json.dumps(a) + "\n" for a in lines), encoding="utf-8")

    proc = run_hook(["prompt-context", "--src", str(ROOT / "src"),
                     "--memory-dir", str(tmp_path / "client-mem"),
                     "--emit", "UserPromptSubmit"],
                    env=env, payload=hook_payload(
                        session, cwd, hook_event_name="UserPromptSubmit", prompt="hi"))
    assert proc.returncode == 0, proc.stderr

    deadline = time.time() + 30
    rows = []
    while time.time() < deadline:
        db = sqlite3.connect(str(mem / "memory.db"))
        try:
            rows = db.execute(
                "SELECT project FROM agent_actions WHERE session_id=? ORDER BY id",
                (session,)).fetchall()
        finally:
            db.close()
        if len(rows) == 4:
            break
        time.sleep(0.5)
    assert [r[0] for r in rows] == ["p1proj", "p1proj", "p2proj", "p2proj"]


def test_remote_flush_own_stops_when_flushed_file_cannot_be_removed(monkeypatch, tmp_path):
    """An undeletable flushed spool must not spin the re-list loop until the
    deadline: once a re-list yields only paths already flushed in this call,
    remote_flush_own returns True so the follow-up POST still goes out."""
    spool_env(monkeypatch, tmp_path)
    posts = []

    def fake_post(command, call, payload, timeout):
        posts.append(command)
        return True, ""

    def no_unlink(path, why):
        raise OSError("read-only file system")

    monkeypatch.setattr(tam_hook, "remote_post", fake_post)
    monkeypatch.setattr(tam_hook, "unlink", no_unlink)
    tam_hook.spool_append(SESSION, spool_action("tu_stuck", "stuck cmd"))
    started = time.monotonic()
    ok = tam_hook.remote_flush_own(batch_call(), started + 2)
    assert ok is True
    assert time.monotonic() - started < 2
    assert posts == ["actions-batch"]
    assert [a["tool_use_id"] for a in pending_spool_actions(tmp_path)] == ["tu_stuck"]


def test_remote_spool_take_refreshes_the_taken_file_mtime(monkeypatch, tmp_path):
    """os.replace keeps the source mtime; the taken-aside .sending must look
    fresh or another session's stale-sweep could grab a live flush."""
    spool_env(monkeypatch, tmp_path)
    path = spool_file(tmp_path, SESSION)
    tam_hook.spool_append(SESSION, spool_action("tu_mt", "mtime cmd"))
    stale = time.time() - tam_hook.SPOOL_STALE_SEC - 60
    os.utime(path, (stale, stale))
    taken = tam_hook._spool_take(path)
    assert time.time() - taken.stat().st_mtime < 60


# ── one deadline per hook: a blackholed server cannot hang the hook ──


def test_remote_blackholed_server_exits_within_deadline(tmp_path):
    """A server that accepts TCP but never answers must not push the hook past
    its deadline; the follow-up POST is skipped after the first failure."""
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(8)
    port = listener.getsockname()[1]
    accepted: list[socket.socket] = []
    stop_accept = threading.Event()

    def accept_loop() -> None:
        listener.settimeout(0.2)
        while not stop_accept.is_set():
            try:
                conn, _ = listener.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            accepted.append(conn)

    thread = threading.Thread(target=accept_loop, daemon=True)
    thread.start()
    try:
        env = client_env(tmp_path, f"http://127.0.0.1:{port}")
        cwd = tmp_path / "workdir"
        proc = run_hook(["record-action", "--src", str(ROOT / "src"),
                         "--memory-dir", str(tmp_path / "client-mem")],
                        env=env, payload=post_tool_payload(
                            "Bash", {"command": "echo hi"}, "tu_bh", cwd))
        assert proc.returncode == 0, proc.stderr

        started = time.monotonic()
        proc = run_hook(["prompt-context", "--src", str(ROOT / "src"),
                         "--memory-dir", str(tmp_path / "client-mem"),
                         "--emit", "UserPromptSubmit"],
                        env=env, payload=hook_payload(
                            SESSION, cwd, hook_event_name="UserPromptSubmit", prompt="hi"),
                        timeout=30)
        elapsed = time.monotonic() - started
        assert proc.returncode == 0, proc.stderr
        # hooks.json gives UserPromptSubmit 5 s; the flush deadline is 3 s.
        assert elapsed < 4.5, f"prompt-context took {elapsed:.1f}s against a blackhole"
        time.sleep(0.3)
        # Exactly one POST was attempted: the failed actions-batch. The
        # prompt-context follow-up is skipped once the transport fails.
        assert len(accepted) == 1
        # The action is still spooled for the next flush.
        assert len(pending_spool_actions(tmp_path)) == 1
    finally:
        stop_accept.set()
        listener.close()
        for conn in accepted:
            conn.close()


# ── tool summary: local parity and redact-before-clip ────────────────


def local_summarize(tool: str, tool_input) -> str:
    sys.path.insert(0, str(ROOT / "src"))
    from ambient.actions import summarize_tool_input

    return summarize_tool_input(tool, tool_input)


@pytest.mark.parametrize("tool,tool_input", [
    ("Bash", {"command": "pytest -q tests/test_x.py"}),
    ("Bash", {"command": "x" * 500}),
    ("Edit", {"file_path": "/w/src/x.py", "new_string": "fixed code"}),
    ("Edit", {"file_path": "/w/src/x.py", "new_string": "y" * 300}),
    ("Write", {"file_path": "/w/src/new.py", "content": "print(1)"}),
    ("MultiEdit", {"file_path": "/w/src/x.py",
                   "edits": [{"new_string": "one"}, {"new_string": "two"}]}),
    ("NotebookEdit", {"notebook_path": "/w/n.ipynb", "new_source": "cell"}),
    ("mcp__memory__memory_save", {"content": "whole record", "type": "fact"}),
    ("Grep", {"pattern": "TODO", "path": "/w/src"}),
    ("CustomTool", {"alpha": 1, "beta": "two"}),
    ("Bash", None),
    ("Edit", {"file_path": "/w/x.py"}),
])
def test_remote_tool_summary_matches_local(tool, tool_input, tmp_path):
    """remote_tool_summary must equal ambient.actions.summarize_tool_input row for row."""
    assert (tam_hook.remote_tool_summary(tool, tool_input, ROOT / "src")
            == local_summarize(tool, tool_input))


def test_remote_tool_summary_covers_devin_tools(tmp_path):
    """Devin CLI tool names summarise exactly like their Claude counterparts."""
    assert (tam_hook.remote_tool_summary("exec", {"command": "pytest -q"}, ROOT / "src")
            == "pytest -q")
    assert (tam_hook.remote_tool_summary("edit", {"file_path": "/w/a.py", "new_string": "x = 1"},
                                         ROOT / "src") == "/w/a.py :: x = 1")
    assert (tam_hook.remote_tool_summary("write", {"file_path": "/w/b.py", "content": "print(1)"},
                                         ROOT / "src") == "/w/b.py :: print(1)")
    assert (tam_hook.remote_tool_summary("notebook_edit",
                                         {"notebook_path": "/w/n.ipynb", "new_source": "cell"},
                                         ROOT / "src") == "/w/n.ipynb :: cell")


def test_remote_record_action_marks_devin_tool_response_failure(tmp_path, monkeypatch):
    """Devin tool_response {success: false} spools a failed action; there is no exit code."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    tam_hook.remote_record_action(
        {"tool_name": "exec", "tool_input": {"command": "false"},
         "tool_response": {"success": False, "output": "", "error": "boom"},
         "session_id": "s-dev", "cwd": str(tmp_path / "workdir")}, ROOT / "src")
    lines = spool_file(tmp_path, "s-dev").read_text().splitlines()
    assert len(lines) == 1
    action = json.loads(lines[0])
    assert action["tool"] == "exec" and action["status"] == "failed"
    assert action["summary"] == "false" and action["exit_code"] is None


def test_remote_tool_summary_redacts_secret_straddling_the_clip(tmp_path):
    """A credential cut by the 400-char clip would survive redaction; redact first."""
    secret = "sk-" + "q9xQ9xQ9zZ8yY7xX6wW5vV4uU3tT2sS1rR0"
    command = "deploy " + "a" * 380 + " " + secret + " " + "b" * 60
    summary = tam_hook.remote_tool_summary("Bash", {"command": command}, ROOT / "src")
    assert "q9xQ9xQ9" not in summary and secret not in summary
    assert "[REDACTED]" in summary


# ── memory-trigger stays network-free in remote mode ─────────────────


def test_remote_memory_trigger_never_posts_in_process(monkeypatch, tmp_path):
    """PostToolUse must not do network I/O: error/fix/check-patterns go to a
    detached remote-post worker, not a synchronous remote_post."""
    spool_env(monkeypatch, tmp_path)
    calls = []

    def no_network(*args, **kwargs):
        calls.append(args)
        raise AssertionError("PostToolUse must not do network I/O")

    monkeypatch.setattr(urllib.request, "urlopen", no_network)
    payload = {"hook_event_name": "PostToolUseFailure", "tool_name": "Bash",
               "tool_input": {"command": "false"},
               "tool_response": {"exit_code": "2", "stderr": "boom"},
               "session_id": "s2-mt", "cwd": str(tmp_path / "workdir")}
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(payload)))
    rc = tam_hook.main(["memory-trigger", "--state-dir", str(tmp_path / "state"),
                        "--src", str(ROOT / "src"), "--emit", "PostToolUse"])
    assert rc == 0
    assert calls == []

    # The runner itself hands every supported action to the detached worker.
    runner = tam_hook.remote_error_capture_runner(
        {"payload": {}, "project": "e2eproj", "session_id": "s2-mt", "cwd": ""})
    assert runner(tmp_path, "auto_self_improve.py",
                  ["error", "--project", "e2eproj"], stdin_text="d") is True
    assert runner(tmp_path, "auto_self_improve.py",
                  ["fix", "--project", "e2eproj", "--fix", "did it"],
                  stdin_text="d") is True
    assert runner(tmp_path, "auto_self_improve.py",
                  ["check-patterns", "--project", "e2eproj", "--days", "7"]) is True
    assert runner(tmp_path, "auto_self_improve.py", ["other"], stdin_text=None) is False
    assert runner(tmp_path, "other.py", ["error"]) is False
    assert calls == []


def test_remote_memory_trigger_requests_use_one_ordered_worker(monkeypatch, tmp_path):
    """error + check-patterns from one memory-trigger invocation reach the server
    in order through a SINGLE detached worker (an ordered list on stdin)."""
    received = []

    class Recorder(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
            received.append((self.path, body))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"ok": true, "output": ""}')

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Recorder)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        spool_env(monkeypatch, tmp_path)
        monkeypatch.setenv("TAM_REMOTE_URL",
                           f"http://127.0.0.1:{server.server_address[1]}")
        category, _severity = tam_hook.categorize("boom")
        counts = tmp_path / "state" / "error-counts"
        counts.mkdir(parents=True)
        (counts / tam_hook.safe_name(f"{category}-e2eproj")).write_text(
            str(tam_hook.ERROR_PATTERN_THRESHOLD - 1), encoding="utf-8")
        real_popen = subprocess.Popen
        spawns = []

        def counting_popen(args, *a, **k):
            if "remote-post" in args:
                spawns.append(args)
            return real_popen(args, *a, **k)

        monkeypatch.setattr(tam_hook.subprocess, "Popen", counting_popen)
        payload = {"hook_event_name": "PostToolUseFailure", "tool_name": "Bash",
                   "tool_input": {"command": "false"},
                   "tool_response": {"exit_code": "2", "stderr": "boom"},
                   "session_id": "s2-mt-order", "cwd": str(tmp_path / "workdir")}
        monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(payload)))
        rc = tam_hook.main(["memory-trigger", "--state-dir", str(tmp_path / "state"),
                            "--src", str(ROOT / "src"), "--emit", "PostToolUse"])
        assert rc == 0
        deadline = time.time() + 15
        while time.time() < deadline and len(received) < 2:
            time.sleep(0.1)
    finally:
        server.shutdown()
        server.server_close()
    actions = [body["payload"]["action"] for path, body in received
               if path.endswith("/error-capture")]
    assert actions == ["error", "check-patterns"]
    assert len(spawns) == 1, "one memory-trigger invocation must spawn one worker"


def test_remote_memory_trigger_flushes_queued_requests_on_failure(monkeypatch, tmp_path):
    """If memory_trigger raises after queueing error-capture requests (a state
    file write fails), they are still sent: capture.flush() runs in finally."""
    spool_env(monkeypatch, tmp_path)
    flushed = []

    def spy_flush(self):
        flushed.append([r["payload"]["action"] for r in self.requests])

    def failing_trigger(*args, **kwargs):
        kwargs["runner"](tmp_path, "auto_self_improve.py",
                         ["error", "--project", "e2eproj"], stdin_text="d")
        raise OSError("state file write failed")

    monkeypatch.setattr(tam_hook.RemoteErrorCaptureRunner, "flush", spy_flush)
    monkeypatch.setattr(tam_hook, "memory_trigger", failing_trigger)
    payload = {"hook_event_name": "PostToolUseFailure", "tool_name": "Bash",
               "tool_input": {"command": "false"},
               "tool_response": {"exit_code": "2", "stderr": "boom"},
               "session_id": "s2-mt-fail", "cwd": str(tmp_path / "workdir")}
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(payload)))
    rc = tam_hook.main(["memory-trigger", "--state-dir", str(tmp_path / "state"),
                        "--src", str(ROOT / "src"), "--emit", "PostToolUse"])
    assert rc == 1
    assert flushed == [["error"]]


# ── local-mode stop keeps its early return ───────────────────────────


def test_local_stop_hook_active_returns_before_project_lookup(monkeypatch, tmp_path):
    """Local stop with stop_hook_active must not even resolve the project (git spawn)."""
    monkeypatch.delenv("TAM_REMOTE_URL", raising=False)
    looked_up = []
    monkeypatch.setattr(tam_hook, "project_of",
                        lambda payload: looked_up.append(payload) or "p")
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(
        {"session_id": "s", "cwd": str(tmp_path / "workdir"),
         "hook_event_name": "Stop", "stop_hook_active": True})))
    rc = tam_hook.main(["stop", "--memory-dir", str(tmp_path / "mem"),
                        "--recovery-dir", str(tmp_path / "mem" / "recovery")])
    assert rc == 0
    assert looked_up == []


# ── slim call payloads (review ride-along) ───────────────────────────


def test_remote_slim_call_keeps_only_the_fields_the_server_reads(tmp_path):
    """Only the fields the server reads survive, under one shared budget measured
    on the JSON-encoded size: escapes count, not just raw UTF-8 length."""
    big = "x" * (tam_hook.REMOTE_CALL_FIELD_MAX_BYTES // 4)
    call = {"payload": {"prompt": big, "last_assistant_message": "done",
                        "user_message": {"role": "user", "content": big},
                        "transcript_path": "/secret/path", "cwd": "/w",
                        "agent_id": "a1"},
            "project": "e2eproj", "session_id": "s", "cwd": "/w"}
    slim = tam_hook._remote_slim_call(call, "prompt", "user_message")
    assert set(slim["payload"]) == {"prompt", "user_message"}
    assert slim["payload"]["user_message"]["role"] == "user"
    size = sum(len(json.dumps(value, ensure_ascii=False).encode("utf-8"))
               for value in slim["payload"].values())
    assert size <= tam_hook.REMOTE_CALL_FIELD_MAX_BYTES
    # The raw call is not mutated.
    assert len(call["payload"]["prompt"]) == len(big)

    # Escape-heavy text is clipped by encoded size; one field can crowd the
    # rest out of the shared budget entirely.
    escapes = '"' * (tam_hook.REMOTE_CALL_FIELD_MAX_BYTES // 2)
    call = {"payload": {"prompt": escapes,
                        "user_message": {"role": "user", "content": "hi"}},
            "project": "e2eproj", "session_id": "s", "cwd": "/w"}
    slim = tam_hook._remote_slim_call(call, "prompt", "user_message")
    assert set(slim["payload"]) == {"prompt"}
    assert len(json.dumps(slim["payload"]["prompt"], ensure_ascii=False)
               .encode("utf-8")) <= tam_hook.REMOTE_CALL_FIELD_MAX_BYTES


# ── end-to-end parity: same session local vs remote ──────────────────


def make_local_db(mem: Path) -> None:
    """A fully migrated memory.db for the local-mode side (Store, no embedder)."""
    mem.mkdir(parents=True, exist_ok=True)
    code = (
        "import sys; sys.path.insert(0, {src!r}); import server;"
        "server.HAS_CHROMA = False;"
        "server.Store._init_embed_mode = lambda self: 'none';"
        "server.Store._check_embed_dim_compat = lambda self: None;"
        "server.Store.embed = lambda self, texts: [];"
        "store = server.Store(); store.db.close();"
        "worker = getattr(store, '_enrich_worker', None);"
        "worker.stop() if worker is not None else None"
    ).format(src=str(ROOT / "src"))
    env = dict(os.environ, TAM_MEMORY_DIR=str(mem), CLAUDE_MEMORY_DIR=str(mem),
               MEMORY_ASYNC_ENRICHMENT="false", MEMORY_LLM_ENABLED="false",
               MEMORY_MODE="fast")
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          env=env, cwd=str(ROOT), timeout=120, check=False)
    assert proc.returncode == 0, proc.stderr


def local_env(tmp_path: Path, mem: Path) -> dict[str, str]:
    """Hook env for local mode: a real memory.db, no TAM_REMOTE_URL."""
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("TAM_", "CLAUDE_MEMORY", "MEMORY_", "HOOK_"))}
    home = tmp_path / "local-home"
    home.mkdir(exist_ok=True)
    env.update({"HOME": str(home), "TAM_MEMORY_DIR": str(mem),
                "CLAUDE_MEMORY_DIR": str(mem),
                "CLAUDE_MEMORY_INSTALL_DIR": str(ROOT), "MEMORY_PROJECT": "e2eproj",
                "MEMORY_LLM_ENABLED": "false", "MEMORY_ASYNC_ENRICHMENT": "false",
                "TAM_HOOK_NOTIFY": "0",
                "XDG_CACHE_HOME": str(tmp_path / "local-cache")})
    return env


def collect_rows(db_path: Path, session: str, cwd: Path):
    db = sqlite3.connect(str(db_path))
    try:
        actions = db.execute(
            "SELECT tool, summary, status, exit_code, project FROM agent_actions "
            "WHERE session_id=? ORDER BY id", (session,)).fetchall()
        intents = db.execute(
            "SELECT prompt FROM intents WHERE session_id=? ORDER BY id",
            (session,)).fetchall()
        # session-autosave rows carry an auto_* session id; the cwd in context
        # pins them to this run (the module server DB sees other tests too).
        knowledge = db.execute(
            "SELECT content FROM knowledge WHERE tags LIKE '%autosave%' "
            "AND context LIKE ? ORDER BY id", (f"%{cwd}%",)).fetchall()
        summaries = db.execute(
            "SELECT summary FROM session_summaries WHERE session_id=? ORDER BY id",
            (session,)).fetchall()
        episodes = db.execute(
            "SELECT narrative, outcome, impact_score, key_insight, approaches_tried, "
            "concepts, tools_used, user_corrections, frustration_signals "
            "FROM episodes WHERE session_id=? ORDER BY id", (session,)).fetchall()
    except sqlite3.OperationalError:
        return None
    finally:
        db.close()
    if not (len(actions) >= 3 and intents and len(knowledge) >= 2
            and summaries and episodes):
        return None
    return {"actions": actions, "intents": intents, "knowledge": knowledge,
            "summaries": summaries, "episodes": episodes}


def episode_key(row) -> tuple:
    narrative, outcome, impact, insight, approaches, concepts, tools, corrections, frustration = row
    return (narrative, outcome, impact, insight, json.loads(approaches),
            sorted(json.loads(concepts)), sorted(json.loads(tools)),
            json.loads(corrections), frustration)


def test_remote_e2e_local_remote_row_parity(http_server, tmp_path):
    """The same scripted session through local hooks (memory.db A) and remote
    hooks (S1 server, memory.db B) must leave equal rows - ids, timestamps and
    session bookkeeping aside."""
    mem_remote, port = http_server
    session = "e2e-parity"
    PROJECT = "parityproj"
    cwd = tmp_path / "workdir"
    cwd.mkdir(exist_ok=True)
    transcript = tmp_path / f"{session}.jsonl"
    write_transcript(transcript)

    ups_payload = {"hook_event_name": "UserPromptSubmit", "prompt": "fix the flaky test",
                   "session_id": session, "cwd": str(cwd)}
    tool_calls = [
        ("Bash", {"command": "pytest -q tests/test_x.py"}, "toolu_p1", {}),
        ("Edit", {"file_path": str(cwd / "src" / "x.py"), "new_string": "fixed code"},
         "toolu_p2", {}),
        ("Bash", {"command": "make deploy"}, "toolu_p3",
         {"exit_code": "2", "stderr": "boom"}),
    ]
    stop_payload = {"hook_event_name": "Stop", "session_id": session, "cwd": str(cwd),
                    "last_assistant_message": "Fixed the flaky test."}
    end_payload = {"hook_event_name": "SessionEnd", "session_id": session, "cwd": str(cwd),
                   "reason": "clear", "transcript_path": str(transcript)}
    reason_text = "User ran /clear"

    def scripted_session(env: dict, mem: Path, tag: str) -> None:
        def hook(args: list, payload: dict | None = None,
                 stdin_text: str | None = None) -> None:
            proc = run_hook(args, env=env, payload=payload, stdin_text=stdin_text)
            assert proc.returncode == 0, (tag, args[0], proc.stderr)

        hook(["session-start-context", "--db", str(mem / "memory.db"),
              "--project", PROJECT, "--cwd", str(cwd), "--session", session,
              "--source", "startup", "--recovery-dir", str(mem / "recovery")],
             hook_payload(session, cwd, hook_event_name="SessionStart", source="startup"))
        hook(["prompt-context", "--src", str(ROOT / "src"),
              "--memory-dir", str(mem), "--emit", "UserPromptSubmit"], ups_payload)
        intent_file = tmp_path / f"intent-{tag}.json"
        intent_file.write_text(json.dumps(ups_payload))
        hook(["save-intent", "--src", str(ROOT / "src"),
              "--db", str(mem / "memory.db"), "--input", str(intent_file)], stdin_text="")
        for tool, tool_input, tuid, response in tool_calls:
            hook(["record-action", "--src", str(ROOT / "src"),
                  "--memory-dir", str(mem)],
                 {"hook_event_name": "PostToolUse", "tool_name": tool,
                  "tool_input": tool_input, "tool_response": response,
                  "session_id": session, "cwd": str(cwd), "tool_use_id": tuid})
        hook(["stop", "--memory-dir", str(mem),
              "--recovery-dir", str(mem / "recovery"), "--src", str(ROOT / "src")],
             stop_payload)
        hook(["session-end", "--src", str(ROOT / "src"),
              "--recovery-dir", str(mem / "recovery"), "--reason-text", reason_text],
             end_payload)

    # Local mode against its own migrated store.
    mem_local = tmp_path / "mem-local"
    make_local_db(mem_local)
    env_local = local_env(tmp_path, mem_local)
    env_local["MEMORY_PROJECT"] = PROJECT
    scripted_session(env_local, mem_local, "local")

    # Remote mode against the shared S1 server; the "memory dir" args are the
    # client's own state dir and are ignored by the remote commands.
    client_mem = tmp_path / "client-mem"
    scripted_session(client_env(tmp_path, f"http://127.0.0.1:{port}",
                                MEMORY_PROJECT=PROJECT), client_mem, "remote")

    deadline = time.time() + 60
    local_rows = remote_rows = None
    while time.time() < deadline:
        local_rows = collect_rows(mem_local / "memory.db", session, cwd)
        remote_rows = collect_rows(mem_remote / "memory.db", session, cwd)
        if local_rows and remote_rows:
            break
        time.sleep(0.5)
    assert local_rows, "local side did not produce the expected rows"
    assert remote_rows, "remote side did not produce the expected rows"

    assert local_rows["actions"] == remote_rows["actions"]
    assert local_rows["intents"] == remote_rows["intents"]
    assert local_rows["knowledge"] == remote_rows["knowledge"]
    assert local_rows["summaries"] == remote_rows["summaries"]
    assert ([episode_key(r) for r in local_rows["episodes"]]
            == [episode_key(r) for r in remote_rows["episodes"]])


# ── S2b: laptop-local recovery in remote mode ────────────────────────


def _remote_env(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TAM_REMOTE_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("TAM_REMOTE_TOKEN", "test-token")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("MEMORY_PROJECT", "e2eproj")


def test_remote_stop_writes_recovery_like_local(monkeypatch, tmp_path):
    """Remote stop must leave the same pending-<sid>.md recovery file local stop does."""
    _remote_env(monkeypatch, tmp_path)
    monkeypatch.setattr(tam_hook, "remote_post", lambda *a, **k: (True, ""))
    cwd = tmp_path / "workdir"
    cwd.mkdir(exist_ok=True)
    recovery_dir = tmp_path / "recovery"
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(
        {"session_id": "s-rec", "cwd": str(cwd), "hook_event_name": "Stop"})))
    rc = tam_hook.main(["stop", "--memory-dir", str(tmp_path / "mem"),
                        "--recovery-dir", str(recovery_dir),
                        "--src", str(ROOT / "src")])
    assert rc == 0
    path = recovery_dir / "pending-s-rec.md"
    assert path.is_file()
    body = path.read_text(encoding="utf-8")
    assert '"project": "e2eproj"' in body and str(cwd) in body


def test_remote_stop_hook_active_skips_recovery(monkeypatch, tmp_path):
    """stop_hook_active: remote flushes the spool but writes no recovery, like local."""
    _remote_env(monkeypatch, tmp_path)
    monkeypatch.setattr(tam_hook, "remote_post", lambda *a, **k: (True, ""))
    cwd = tmp_path / "workdir"
    cwd.mkdir(exist_ok=True)
    recovery_dir = tmp_path / "recovery"
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(
        {"session_id": "s-rec2", "cwd": str(cwd), "hook_event_name": "Stop",
         "stop_hook_active": True})))
    rc = tam_hook.main(["stop", "--memory-dir", str(tmp_path / "mem"),
                        "--recovery-dir", str(recovery_dir)])
    assert rc == 0
    assert not (recovery_dir / "pending-s-rec2.md").exists()


def test_remote_session_end_writes_recovery_with_transcript(monkeypatch, tmp_path):
    """Remote session-end writes the recovery file locally before flushing/posting."""
    _remote_env(monkeypatch, tmp_path)
    posts = []
    monkeypatch.setattr(tam_hook, "remote_post",
                        lambda command, call, payload, timeout: posts.append(command) or (True, ""))
    cwd = tmp_path / "workdir"
    cwd.mkdir(exist_ok=True)
    transcript = tmp_path / "t.jsonl"
    write_transcript(transcript)
    recovery_dir = tmp_path / "recovery"
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(
        {"session_id": "s-end", "cwd": str(cwd), "hook_event_name": "SessionEnd",
         "transcript_path": str(transcript)})))
    rc = tam_hook.main(["session-end", "--src", str(ROOT / "src"),
                        "--recovery-dir", str(recovery_dir), "--reason-text", "clear"])
    assert rc == 0
    path = recovery_dir / "pending-s-end.md"
    assert path.is_file()
    body = path.read_text(encoding="utf-8")
    assert "fix the flaky test" in body and "working on it" in body
    assert "session-end" in posts


def test_remote_session_end_devin_payload_reads_no_transcript(monkeypatch, tmp_path):
    """A Devin SessionEnd payload has prompt_id and no transcript_path; the
    ~/.claude/projects cwd fallback must not adopt a Claude session's transcript,
    so session-end goes out with empty texts and no recovery file is written."""
    _remote_env(monkeypatch, tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir(exist_ok=True)
    cwd = tmp_path / "workdir"
    cwd.mkdir(exist_ok=True)
    project_dir = (tmp_path / "home" / ".claude" / "projects"
                   / re.sub(r"[^A-Za-z0-9]", "-", str(cwd)))
    project_dir.mkdir(parents=True)
    (project_dir / "claude-session.jsonl").write_text(
        "\n".join(json.dumps(e) for e in [
            {"type": "user", "message": {"role": "user", "content": "claude decoy"}},
            {"type": "assistant", "message": {"role": "assistant",
                                              "content": [{"type": "text", "text": "decoy"}]}},
        ]) + "\n", encoding="utf-8")
    posts = []
    monkeypatch.setattr(tam_hook, "remote_post",
                        lambda command, call, payload, timeout: posts.append((command, payload))
                        or (True, ""))
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(
        {"session_id": "truth-patella", "cwd": str(cwd), "hook_event_name": "SessionEnd",
         "prompt_id": "p-1", "reason": "logout"})))
    rc = tam_hook.main(["session-end", "--src", str(ROOT / "src"),
                        "--recovery-dir", str(tmp_path / "recovery"), "--reason-text", "logout"])
    assert rc == 0
    end = [payload for command, payload in posts if command == "session-end"]
    assert len(end) == 1
    assert end[0]["user_texts"] == [] and end[0]["assistant_text"] == ""
    assert end[0]["episode_messages"] == []
    recovery = tmp_path / "recovery"
    assert not recovery.exists() or not list(recovery.glob("pending-*.md"))


def test_remote_session_start_context_appends_recovery(monkeypatch, tmp_path, capsys):
    """The laptop recovery block is appended after the server's session-start text."""
    _remote_env(monkeypatch, tmp_path)
    monkeypatch.setattr(tam_hook, "remote_post", lambda *a, **k: (True, "SERVER TEXT"))
    cwd = tmp_path / "workdir"
    cwd.mkdir(exist_ok=True)
    recovery_dir = tmp_path / "recovery"
    tam_hook.write_recovery(recovery_dir, "old-sess", "e2eproj", str(cwd),
                            "interrupted", "user asked X", "assistant replied Y")
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(
        {"session_id": "s-new", "cwd": str(cwd), "hook_event_name": "SessionStart",
         "source": "startup"})))
    rc = tam_hook.main(["session-start-context", "--db", str(tmp_path / "mem" / "memory.db"),
                        "--project", "e2eproj", "--cwd", str(cwd), "--session", "s-new",
                        "--source", "startup", "--recovery-dir", str(recovery_dir)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "SERVER TEXT" in out and "RECOVERY:" in out
    assert out.index("SERVER TEXT") < out.index("RECOVERY:")
    assert "user asked X" in out


def test_remote_session_start_context_gates_recovery_by_source(monkeypatch, tmp_path, capsys):
    """Same gate as local session_start_context: only startup/resume sources
    print the recovery block, and clear/compact must not consume the file."""
    _remote_env(monkeypatch, tmp_path)
    monkeypatch.setattr(tam_hook, "remote_post", lambda *a, **k: (True, "SERVER TEXT"))
    cwd = tmp_path / "workdir"
    cwd.mkdir(exist_ok=True)
    recovery_dir = tmp_path / "recovery"
    recovery = recovery_dir / "pending-old-sess.md"

    def run(source: str, session: str) -> str:
        tam_hook.write_recovery(recovery_dir, "old-sess", "e2eproj", str(cwd),
                                "interrupted", "user asked X", "assistant replied Y")
        monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(
            {"session_id": session, "cwd": str(cwd), "hook_event_name": "SessionStart",
             "source": source})))
        capsys.readouterr()
        rc = tam_hook.main(["session-start-context", "--db", str(tmp_path / "mem" / "memory.db"),
                            "--project", "e2eproj", "--cwd", str(cwd), "--session", session,
                            "--source", source, "--recovery-dir", str(recovery_dir)])
        assert rc == 0
        return capsys.readouterr().out

    for source in ("clear", "compact"):
        out = run(source, f"s-{source}")
        assert "SERVER TEXT" in out and "RECOVERY:" not in out
        assert recovery.is_file(), f"source={source} must not consume the recovery file"

    out = run("startup", "s-start")
    assert "RECOVERY:" in out and "user asked X" in out
    assert not recovery.exists(), "startup must deliver (consume) the recovery file"
