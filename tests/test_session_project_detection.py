"""The session row must name the project the server was started in.

`_bootstrap_session` used to call `store.session_start(SID, branch=BRANCH)`
without a project, so `Store.session_start`'s 'general' default landed on every
row - one bucket for every repository, and `memory_timeline` could not tell
them apart. The cwd that already gives `_detect_git_branch` its answer is the
same cwd that names the project.
"""

import os
import subprocess
import sqlite3

import pytest

import server
from base_schema import apply_full_schema


@pytest.fixture
def repo(tmp_path):
    """A real git repository - `_detect_project` shells out to git."""
    root = tmp_path / "MyRepo"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    return root


def _in(path, fn):
    prev = os.getcwd()
    os.chdir(path)
    try:
        return fn()
    finally:
        os.chdir(prev)


def test_project_comes_from_the_repository_name(repo, monkeypatch):
    monkeypatch.delenv("MEMORY_PROJECT", raising=False)
    assert _in(repo, server._detect_project) == "myrepo"


def test_project_of_a_worktree_is_the_main_repository(repo, tmp_path, monkeypatch):
    """A worktree directory is named after its branch, not the project.

    Resolving via the *common* git dir is what keeps every worktree of one
    repository in a single bucket - the same rule hooks/lib/common.sh applies.
    """
    monkeypatch.delenv("MEMORY_PROJECT", raising=False)
    subprocess.run(["git", "commit", "-q", "--allow-empty", "-m", "init"], cwd=repo, check=True)
    wt = tmp_path / "AG-1234-some-branch"
    subprocess.run(
        ["git", "worktree", "add", "-q", "-b", "AG-1234", str(wt)], cwd=repo, check=True
    )

    assert _in(wt, server._detect_project) == "myrepo"


def test_memory_project_env_wins(repo, monkeypatch):
    monkeypatch.setenv("MEMORY_PROJECT", "explicit-name")
    assert _in(repo, server._detect_project) == "explicit-name"


def test_outside_a_repository_the_directory_names_the_project(tmp_path, monkeypatch):
    monkeypatch.delenv("MEMORY_PROJECT", raising=False)
    plain = tmp_path / "Loose-Dir"
    plain.mkdir()
    # `git rev-parse` may still resolve an enclosing repository; only assert the
    # fallback shape when the temp dir really is outside one.
    inside = _in(plain, lambda: subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        capture_output=True, text=True,
    ).stdout.strip())
    if inside == "true":
        pytest.skip("temp dir sits inside a git repository on this machine")
    assert _in(plain, server._detect_project) == "loose-dir"


def test_session_start_records_the_resolved_project():
    """The value must reach the row, not just the resolver."""
    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    apply_full_schema(db)
    store = server.Store.__new__(server.Store)
    store.db = db

    store.session_start("sess_x", project="claude-memory-server", branch="main")

    row = db.execute("SELECT project, branch FROM sessions WHERE id='sess_x'").fetchone()
    assert row["project"] == "claude-memory-server"
    assert row["branch"] == "main"
    db.close()


def test_bootstrap_passes_the_resolved_project_to_session_start():
    """Pin the wiring, not just the resolver.

    The bug was exactly here: `_bootstrap_session` called `session_start` with
    a branch and no project, so the resolver's answer never reached the row.
    """
    import inspect

    src = inspect.getsource(server._bootstrap_session)
    assert "session_start(SID, project=_detect_project()" in src, src
