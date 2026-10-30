"""The working directory is an implicit path root only when it is a project directory."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from path_roots import cwd_root  # noqa: E402


def test_project_directory_is_a_root(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    assert cwd_root(project, home=tmp_path / "home") == project.resolve()


def test_filesystem_root_is_not_a_root(tmp_path):
    assert cwd_root(Path(tmp_path.anchor), home=tmp_path) is None


def test_home_directory_is_not_a_root(tmp_path):
    assert cwd_root(tmp_path, home=tmp_path) is None


def test_server_allowed_path_does_not_trust_a_filesystem_root_cwd(monkeypatch, tmp_path):
    import server
    monkeypatch.setattr(server, "MEMORY_DIR", tmp_path)
    monkeypatch.chdir("/")
    monkeypatch.delenv(server.ALLOWED_ROOTS_ENV, raising=False)
    try:
        server.allowed_path("/etc/passwd")
    except ValueError:
        return
    raise AssertionError("/etc/passwd accepted with cwd=/")


def test_open_private_text_creates_owner_only_files_and_tightens_existing_ones(tmp_path):
    import os

    from private_file import open_private_text
    target = tmp_path / "export.json"
    with open_private_text(target) as handle:
        handle.write("{}")
    target.chmod(0o644)
    with open_private_text(target) as handle:
        handle.write("[]")
    assert target.read_text(encoding="utf-8") == "[]"
    if os.name != "nt":
        assert target.stat().st_mode & 0o777 == 0o600
