"""S5: hooks/session-end.sh's project derivation must resolve a git worktree
to the main repo's own name - the same value a tool-side session_end call
uses (memory protocol: git-root basename), so the two producers' dedup keys
for the same close actually match.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
COMMON_SH = REPO_ROOT / "hooks" / "lib" / "common.sh"
SESSION_END_SH = REPO_ROOT / "hooks" / "session-end.sh"


@pytest.fixture
def git_worktree(tmp_path: Path) -> tuple[Path, Path]:
    """A real repo plus a worktree checked out under an unrelated directory
    name - exactly the scenario S5 is about: cwd basename != the repo name."""
    main = tmp_path / "sample-project"
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


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
@pytest.mark.skipif(shutil.which("git") is None, reason="git not available")
def test_hook_project_name_resolves_a_worktree_to_the_main_repo(git_worktree):
    """The tool side derives `project` from the git-root basename (memory
    protocol convention); the hook must resolve to the same value, not the
    worktree's own directory name."""
    main, worktree = git_worktree
    tool_side_project = main.name  # what an agent would pass as `project`

    r = subprocess.run(
        ["bash", "-c", f'source "{COMMON_SH}"; hook_project_name'],
        input=("{\"cwd\": \"%s\"}" % str(worktree)).encode(),
        capture_output=True,
        timeout=10,
    )
    assert r.returncode == 0, r.stderr
    hook_side_project = r.stdout.decode().strip()

    assert hook_side_project == tool_side_project
    assert hook_side_project != worktree.name


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
def test_session_end_hook_uses_the_worktree_aware_helper():
    """Regression pin: the hook must derive PROJECT via hook_project_name,
    not a bare `basename "$CWD"` that names the worktree dir instead of the
    repo (S5)."""
    text = SESSION_END_SH.read_text()
    assert 'PROJECT=$(hook_project_name)' in text
    assert 'PROJECT=$(basename "$CWD")' not in text
