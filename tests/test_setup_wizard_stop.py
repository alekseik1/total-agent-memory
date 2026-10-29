"""The verification server is stopped completely, so its throwaway directory can be removed."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from setup_wizard import personal  # noqa: E402

POLL_SECONDS = 0.05
SPAWN_TIMEOUT_SECONDS = 30


def _spawn(code: str) -> subprocess.Popen:
    return subprocess.Popen([sys.executable, "-c", code], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding="utf-8")


def _alive(pid: int) -> bool:
    if os.name == "nt":
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True, encoding="utf-8",
                             errors="replace", check=False, timeout=SPAWN_TIMEOUT_SECONDS).stdout
        return str(pid) in out
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_gone(pid: int) -> bool:
    deadline = time.monotonic() + SPAWN_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if not _alive(pid):
            return True
        time.sleep(POLL_SECONDS)
    return False


def test_a_server_that_leaves_on_end_of_input_is_not_terminated(monkeypatch):
    monkeypatch.setattr(personal, "_terminate_tree", lambda process: pytest.fail("terminated a server that was exiting"))
    process = _spawn("import sys; sys.stdin.read()")
    personal._stop(process)
    assert process.returncode == 0


def test_a_server_that_ignores_end_of_input_is_stopped(monkeypatch):
    monkeypatch.setattr(personal, "GRACEFUL_EXIT_SECONDS", 0.2)
    process = _spawn("import time; print('up', flush=True); time.sleep(600)")
    assert process.stdout.readline().strip() == "up"
    personal._stop(process)
    assert process.poll() is not None


CHILD_HOLDS_FILE = textwrap.dedent(
    """
    import subprocess, sys, time
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
    print(child.pid, flush=True)
    time.sleep(600)
    """
)


@pytest.mark.skipif(os.name != "nt", reason="the venv launcher that leaves a child behind exists on Windows only")
def test_the_children_of_a_stuck_server_are_stopped_with_it(monkeypatch):
    monkeypatch.setattr(personal, "GRACEFUL_EXIT_SECONDS", 0.2)
    process = _spawn(CHILD_HOLDS_FILE)
    child_pid = int(process.stdout.readline())
    personal._stop(process)
    assert _wait_gone(child_pid), "the child of the verification server kept running"
