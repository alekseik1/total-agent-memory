"""Subprocess-level tests for hooks/user-prompt-submit.sh.

The hook runs its Python body asynchronously via `python3 ... &`. We wait
briefly for the background process to finish before asserting state.
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="bare bash resolves to the WSL stub on Windows; the .sh hooks are POSIX-only")

HOOK_PATH = Path(__file__).parent.parent / "hooks" / "user-prompt-submit.sh"
REPO_ROOT = Path(__file__).parent.parent


@pytest.fixture
def hook_env(tmp_path: Path) -> dict[str, str]:
    """Isolated memory dir + seeded DB for the hook to write into."""
    mem_dir = tmp_path / "claude-memory"
    mem_dir.mkdir()
    db_path = mem_dir / "memory.db"

    conn = sqlite3.connect(str(db_path))
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS migrations (
            version TEXT PRIMARY KEY, description TEXT, applied_at TEXT
        );
        """
    )
    mig = REPO_ROOT / "migrations" / "013_intents.sql"
    conn.executescript(mig.read_text())
    conn.commit()
    conn.close()

    env = os.environ.copy()
    env["CLAUDE_MEMORY_DIR"] = str(mem_dir)
    env["CLAUDE_MEMORY_INSTALL_DIR"] = str(REPO_ROOT)
    return env


def _run_hook(env: dict[str, str], payload: dict) -> subprocess.CompletedProcess:
    """Run the shell hook, feeding payload as stdin JSON."""
    return subprocess.run(
        ["bash", str(HOOK_PATH)],
        input=json.dumps(payload).encode("utf-8"),
        env=env,
        capture_output=True,
        timeout=15,
    )


def _wait_for_insert(db_path: Path, expected: int, timeout: float = 5.0) -> int:
    """Poll the DB until at least `expected` rows exist or timeout expires."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            conn = sqlite3.connect(str(db_path))
            n = conn.execute("SELECT COUNT(*) FROM intents").fetchone()[0]
            conn.close()
            if n >= expected:
                return n
        except sqlite3.Error:
            pass
        time.sleep(0.05)
    # Final read
    conn = sqlite3.connect(str(db_path))
    n = conn.execute("SELECT COUNT(*) FROM intents").fetchone()[0]
    conn.close()
    return n


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
def test_hook_saves_intent(hook_env: dict[str, str]) -> None:
    payload = {
        "session_id": "hook-sess-1",
        "cwd": "/tmp/proj-hooktest",
        "prompt": "please fix auth bug",
    }
    r = _run_hook(hook_env, payload)
    assert r.returncode == 0

    db_path = Path(hook_env["CLAUDE_MEMORY_DIR"]) / "memory.db"
    n = _wait_for_insert(db_path, 1)
    assert n == 1

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM intents LIMIT 1").fetchone()
    conn.close()
    assert row["prompt"] == "please fix auth bug"
    assert row["session_id"] == "hook-sess-1"
    assert row["project"] == "proj-hooktest"


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
def test_hook_empty_prompt_is_noop(hook_env: dict[str, str]) -> None:
    payload = {"session_id": "s", "cwd": "/tmp/p", "prompt": "   "}
    r = _run_hook(hook_env, payload)
    assert r.returncode == 0

    db_path = Path(hook_env["CLAUDE_MEMORY_DIR"]) / "memory.db"
    # Give any stray background work time to (fail to) write.
    time.sleep(0.8)
    conn = sqlite3.connect(str(db_path))
    n = conn.execute("SELECT COUNT(*) FROM intents").fetchone()[0]
    conn.close()
    assert n == 0


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
def test_hook_handles_quotes_in_prompt(hook_env: dict[str, str]) -> None:
    """Prompts with single/double quotes and newlines must not break the hook."""
    tricky = "let's 'fix' \"auth\" & add\nmulti-line\n`backticks` too"
    payload = {
        "session_id": "hook-sess-quote",
        "cwd": "/tmp/quote-proj",
        "prompt": tricky,
    }
    r = _run_hook(hook_env, payload)
    assert r.returncode == 0

    db_path = Path(hook_env["CLAUDE_MEMORY_DIR"]) / "memory.db"
    _wait_for_insert(db_path, 1)

    conn = sqlite3.connect(str(db_path))
    row = conn.execute("SELECT prompt FROM intents ORDER BY id DESC LIMIT 1").fetchone()
    conn.close()
    assert row is not None
    assert row[0] == tricky


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
def test_hook_falls_back_to_user_message_shape(hook_env: dict[str, str]) -> None:
    """Older Claude Code builds used user_message.content — hook must still save."""
    payload = {
        "session_id": "hook-sess-legacy",
        "cwd": "/tmp/legacy",
        "user_message": {"content": "legacy style prompt"},
    }
    r = _run_hook(hook_env, payload)
    assert r.returncode == 0

    db_path = Path(hook_env["CLAUDE_MEMORY_DIR"]) / "memory.db"
    _wait_for_insert(db_path, 1)

    conn = sqlite3.connect(str(db_path))
    row = conn.execute("SELECT prompt FROM intents LIMIT 1").fetchone()
    conn.close()
    assert row[0] == "legacy style prompt"
