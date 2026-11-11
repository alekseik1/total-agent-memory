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
import socket
import sqlite3
import subprocess
import sys
import time
import urllib.request
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
    assert len(spool_file(tmp_path, session).read_text().splitlines()) == 2

    # The server comes up: the next Stop flushes the spool exactly once.
    live_env = client_env(tmp_path, f"http://127.0.0.1:{port}")
    proc = run_hook(["stop", "--memory-dir", str(tmp_path / "client-mem"),
                     "--recovery-dir", str(tmp_path / "client-mem" / "recovery"),
                     "--src", str(ROOT / "src")],
                    env=live_env, payload=hook_payload(session, cwd, hook_event_name="Stop"))
    assert proc.returncode == 0
    assert not spool_file(tmp_path, session).exists()

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
