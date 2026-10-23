"""Tests for PostToolUse capture — enqueue function + opt-in shell hook."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from auto_extract_active import capture_tool_observation


HOOK_PATH = Path(__file__).parent.parent / "hooks" / "post-tool-use.sh"
REPO_ROOT = Path(__file__).parent.parent


def test_capture_tool_observation_enqueues(tmp_path: Path) -> None:
    queue = tmp_path / "queue"
    path = capture_tool_observation(
        tool_name="Bash",
        output="hello world",
        session_id="sess-42",
        project="proj-42",
        queue_dir=queue,
    )
    assert path is not None
    assert path.exists()
    payload = json.loads(path.read_text())
    assert payload["kind"] == "tool_observation"
    assert payload["tool_name"] == "Bash"
    assert payload["session_id"] == "sess-42"
    assert payload["project"] == "proj-42"
    assert payload["output"] == "hello world"
    assert "post-tool-use" in payload["tags"]
    assert "Bash" in payload["tags"]
    assert payload["error_candidate"] is False


def test_capture_tool_observation_flags_errors(tmp_path: Path) -> None:
    path = capture_tool_observation(
        tool_name="Bash",
        output="Traceback (most recent call last):\n  ...\nException: oops",
        session_id="s",
        project="p",
        queue_dir=tmp_path,
    )
    assert path is not None
    payload = json.loads(path.read_text())
    assert payload["error_candidate"] is True
    assert "error-candidate" in payload["tags"]


def test_capture_tool_observation_skips_empty(tmp_path: Path) -> None:
    assert capture_tool_observation("Bash", "", "s", "p", queue_dir=tmp_path) is None
    assert capture_tool_observation("Bash", "   \n  ", "s", "p", queue_dir=tmp_path) is None
    # No tool name is also a skip.
    assert capture_tool_observation("", "something", "s", "p", queue_dir=tmp_path) is None
    # No files written
    assert list(tmp_path.glob("*.json")) == []


def test_capture_tool_observation_truncates_huge_output(tmp_path: Path) -> None:
    big = "A" * 50_000
    path = capture_tool_observation(
        "Bash", big, "s", "p", queue_dir=tmp_path,
    )
    assert path is not None
    payload = json.loads(path.read_text())
    # 10_000 cap + truncation marker
    assert len(payload["output"]) < 50_000
    assert "truncated" in payload["output"]


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
@pytest.mark.skipif(sys.platform == "win32", reason="bare bash resolves to the WSL stub on Windows; the .sh hooks are POSIX-only")
def test_post_tool_hook_respects_env_flag(tmp_path: Path) -> None:
    """Without MEMORY_POST_TOOL_CAPTURE=1 the hook is a no-op."""
    mem_dir = tmp_path / "memdir"
    mem_dir.mkdir()
    queue_dir = mem_dir / "extract-queue"

    env = os.environ.copy()
    env["CLAUDE_MEMORY_DIR"] = str(mem_dir)
    env["CLAUDE_MEMORY_INSTALL_DIR"] = str(REPO_ROOT)
    env.pop("MEMORY_POST_TOOL_CAPTURE", None)

    payload = {
        "tool_name": "Bash",
        "session_id": "s-off",
        "cwd": "/tmp/off-proj",
        "tool_response": {"stdout": "hello"},
    }
    r = subprocess.run(
        ["bash", str(HOOK_PATH)],
        input=json.dumps(payload).encode(),
        env=env,
        capture_output=True,
        timeout=10,
    )
    assert r.returncode == 0
    time.sleep(0.8)

    # No queue file should exist since capture is disabled.
    queued = list(queue_dir.glob("*.json")) if queue_dir.exists() else []
    assert queued == []


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
@pytest.mark.skipif(sys.platform == "win32", reason="bare bash resolves to the WSL stub on Windows; the .sh hooks are POSIX-only")
def test_post_tool_hook_enqueues_when_enabled(tmp_path: Path) -> None:
    """With MEMORY_POST_TOOL_CAPTURE=1 the hook writes to the extract queue."""
    mem_dir = tmp_path / "memdir"
    mem_dir.mkdir()
    queue_dir = mem_dir / "extract-queue"

    env = os.environ.copy()
    env["CLAUDE_MEMORY_DIR"] = str(mem_dir)
    env["CLAUDE_MEMORY_INSTALL_DIR"] = str(REPO_ROOT)
    env["MEMORY_POST_TOOL_CAPTURE"] = "1"

    payload = {
        "tool_name": "Bash",
        "session_id": "s-on",
        "cwd": "/tmp/on-proj",
        "tool_response": {"stdout": "it worked"},
    }
    r = subprocess.run(
        ["bash", str(HOOK_PATH)],
        input=json.dumps(payload).encode(),
        env=env,
        capture_output=True,
        timeout=10,
    )
    assert r.returncode == 0

    # Poll for a file to appear.
    deadline = time.time() + 5.0
    queued: list[Path] = []
    while time.time() < deadline:
        if queue_dir.exists():
            queued = list(queue_dir.glob("*.json"))
            if queued:
                break
        time.sleep(0.05)

    assert queued, "expected at least one queued observation file"
    payload_out = json.loads(queued[0].read_text())
    assert payload_out["tool_name"] == "Bash"
    assert payload_out["project"] == "on-proj"
    assert "it worked" in payload_out["output"]
