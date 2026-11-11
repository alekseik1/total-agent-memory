"""memory_reflect_now runs out of process over HTTP and in-process over stdio."""

from __future__ import annotations

import asyncio
import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import server  # noqa: E402

RUNNER = Path(__file__).resolve().parent.parent / "src" / "tools" / "run_reflection.py"
SCOPES = [(None, "full"), ("quick", "quick"), ("weekly", "weekly"), ("bogus", "full")]


class _Agent:
    def __init__(self):
        self.calls = []

    async def run_quick(self):
        self.calls.append("quick")
        return {"ran": "quick"}

    async def run_full(self):
        self.calls.append("full")
        return {"ran": "full"}

    async def run_weekly(self):
        self.calls.append("weekly")
        return {"ran": "weekly"}


class _Proc:
    pid = 4242

    def __init__(self):
        self.waited = threading.Event()

    def wait(self):
        self.waited.set()
        return 0


@pytest.fixture
def harness(monkeypatch):
    agent = _Agent()
    popen_calls = []
    v5_calls = []

    def fake_popen(argv, **kwargs):
        proc = _Proc()
        popen_calls.append((argv, kwargs, proc))
        return proc

    def fake_get_v5(*args, **kwargs):
        v5_calls.append(args)
        return agent

    monkeypatch.setattr(server, "store", SimpleNamespace(db=None), raising=False)
    monkeypatch.setattr(server, "_get_v5", fake_get_v5)
    monkeypatch.setattr(server.subprocess, "Popen", fake_popen)
    return agent, popen_calls, v5_calls


def _call(args):
    return json.loads(asyncio.run(server._do("memory_reflect_now", args)))


@pytest.mark.parametrize("transport", server.HTTP_TRANSPORTS)
@pytest.mark.parametrize(("scope", "expected"), SCOPES)
def test_http_spawns_runner_and_skips_in_process_run(harness, monkeypatch, transport, scope, expected):
    agent, popen_calls, v5_calls = harness
    monkeypatch.setenv("MCP_TRANSPORT", transport)

    result = _call({} if scope is None else {"scope": scope})

    assert result == {"queued": True, "scope": expected, "pid": 4242, "mode": "subprocess",
                      "note": "runs out of process; result in server log / reflection report"}
    assert agent.calls == []
    assert v5_calls == []
    [(argv, kwargs, proc)] = popen_calls
    assert RUNNER.exists()
    assert argv == [sys.executable, str(RUNNER), f"--scope={expected}", "--debounce=0"]
    assert kwargs == {"start_new_session": True}
    assert proc.waited.wait(timeout=2)


def test_http_spawn_failure_returns_error_without_in_process_run(harness, monkeypatch):
    agent, _, v5_calls = harness
    monkeypatch.setenv("MCP_TRANSPORT", "http")

    def broken_popen(argv, **kwargs):
        raise OSError("no exec")

    monkeypatch.setattr(server.subprocess, "Popen", broken_popen)

    result = _call({"scope": "full"})

    assert result == {"error": "failed to start reflection: no exec"}
    assert agent.calls == []
    assert v5_calls == []


@pytest.mark.parametrize(("scope", "expected"), SCOPES)
def test_stdio_runs_in_process(harness, monkeypatch, scope, expected):
    agent, popen_calls, v5_calls = harness
    monkeypatch.setenv("MCP_TRANSPORT", "stdio")

    result = _call({} if scope is None else {"scope": scope})

    assert result == {"ran": expected}
    assert agent.calls == [expected]
    assert len(v5_calls) == 1
    assert popen_calls == []
