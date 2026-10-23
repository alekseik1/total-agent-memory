"""S5: the SessionEnd helper's project derivation must resolve a git worktree
to the main repo's own name - the same value a tool-side session_end call
uses (memory protocol: git-root basename, lowercased by _detect_project), so
the two producers' dedup keys for the same close actually match.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import pytest

from tests.test_hook_runtime import tam_hook

REPO_ROOT = Path(__file__).resolve().parent.parent
COMMON_SH = REPO_ROOT / "hooks" / "lib" / "common.sh"


@pytest.fixture
def git_worktree(tmp_path: Path) -> tuple[Path, Path]:
    """A real repo plus a worktree checked out under an unrelated directory
    name - exactly the scenario S5 is about: cwd basename != the repo name.
    The repo name is mixed-case: the tool side lowercases it."""
    main = tmp_path / "Sample-Project"
    main.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=main, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.t"], cwd=main, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=main, check=True)
    (main / "f.txt").write_text("x")
    subprocess.run(["git", "add", "."], cwd=main, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=main, check=True)
    worktree = tmp_path / "wt-unrelated-dirname"
    subprocess.run(
        ["git", "worktree", "add", "-q", "-b", "feature/x", str(worktree)],
        cwd=main, check=True,
    )
    return main, worktree


@pytest.mark.skipif(shutil.which("git") is None, reason="git not available")
def test_session_end_helper_names_the_main_repo_from_a_worktree(git_worktree, tmp_path, monkeypatch):
    """`tam_hook.py session-end` is what hooks/session-end.sh runs; the project
    it hands to the session-end writers must be the tool side's value, not the
    worktree's own directory name."""
    main, worktree = git_worktree
    monkeypatch.delenv("MEMORY_PROJECT", raising=False)
    monkeypatch.setattr(tam_hook, "read_payload", lambda: {"cwd": str(worktree), "session_id": "s"})
    seen: list[str] = []
    monkeypatch.setattr(tam_hook, "session_end",
                        lambda _payload, _src, _rec, project, _reason: seen.append(project) or [])

    tam_hook.cmd_session_end(argparse.Namespace(
        project=None, src=str(tmp_path), recovery_dir=str(tmp_path), reason_text="User exited"))

    assert seen == [main.name.lower()]
    assert seen[0] != worktree.name


@pytest.mark.skipif(shutil.which("git") is None, reason="git not available")
def test_memory_project_env_still_wins(git_worktree, monkeypatch):
    _main, worktree = git_worktree
    monkeypatch.setenv("MEMORY_PROJECT", "Explicit-Name")
    assert tam_hook.project_of({"cwd": str(worktree)}) == "Explicit-Name"


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
@pytest.mark.skipif(shutil.which("git") is None, reason="git not available")
def test_hook_project_name_agrees_with_the_helper(git_worktree, monkeypatch):
    """session-start.sh still names the project in bash (hook_project_name);
    it must agree with tam_hook.project_of, or the MEMORY_HINT it prints and
    the session-end key differ."""
    main, worktree = git_worktree
    monkeypatch.delenv("MEMORY_PROJECT", raising=False)

    r = subprocess.run(
        ["bash", "-c", f'source "{COMMON_SH}"; hook_project_name'],
        input=("{\"cwd\": \"%s\"}" % str(worktree)).encode(),
        capture_output=True,
        timeout=10,
    )
    assert r.returncode == 0, r.stderr
    hook_side_project = r.stdout.decode().strip()

    assert hook_side_project == tam_hook.project_of({"cwd": str(worktree)}) == main.name.lower()


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
@pytest.mark.skipif(shutil.which("git") is None, reason="git not available")
def test_a_bare_clone_worktree_is_named_like_the_server_names_it(git_worktree, tmp_path, monkeypatch):
    """A bare clone's common dir is not called .git (`Repo.git`, `.bare`): the
    server's _detect_project names it by that dir, and both hook sides must agree."""
    import server

    main, _worktree = git_worktree
    bare = tmp_path / "Bare-Repo.git"
    subprocess.run(["git", "clone", "-q", "--bare", str(main), str(bare)], check=True)
    wt = tmp_path / "wt-of-bare"
    subprocess.run(["git", "worktree", "add", "-q", "-b", "feature/y", str(wt)], cwd=bare, check=True)
    monkeypatch.delenv("MEMORY_PROJECT", raising=False)

    r = subprocess.run(
        ["bash", "-c", f'source "{COMMON_SH}"; hook_project_name'],
        input=("{\"cwd\": \"%s\"}" % str(wt)).encode(),
        capture_output=True,
        timeout=10,
    )
    assert r.returncode == 0, r.stderr
    monkeypatch.chdir(wt)
    tool_side_project = server._detect_project()

    assert tool_side_project == "bare-repo.git"
    assert r.stdout.decode().strip() == tam_hook.project_of({"cwd": str(wt)}) == tool_side_project
