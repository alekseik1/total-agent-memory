"""User-facing tools must resolve the memory dir like src/paths.py: TAM_MEMORY_DIR, legacy
CLAUDE_MEMORY_DIR, ~/.tam, ~/.claude-memory, else ~/.tam. A fresh install keeps its data in
~/.tam, so a tool that only knows ~/.claude-memory reports a missing database."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
posix_only = pytest.mark.skipif(sys.platform == "win32", reason="bash scripts")

RESOLVE_AND_PRINT = (
    "import runpy, sys\n"
    "ns = runpy.run_path(sys.argv[1], run_name='tool_under_test')\n"
    "print(ns[sys.argv[2]])\n"
)


def _env(home: Path, **extra: str) -> dict[str, str]:
    base = {k: v for k, v in os.environ.items()
            if k not in ("TAM_MEMORY_DIR", "CLAUDE_MEMORY_DIR", "MEMORY_DB_PATH")}
    base.update({"HOME": str(home), "USERPROFILE": str(home), **extra})
    return base


def _db(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    db = directory / "memory.db"
    db.write_bytes(b"")
    return db


def _diagnose_sh(home: Path, **extra: str) -> str:
    # INSTALL_DIR derives from the script location: run a copy so the repo .venv cannot interfere.
    (home / "install" / "scripts").mkdir(parents=True)
    copy = home / "install" / "scripts" / "diagnose.sh"
    shutil.copy(SCRIPTS / "diagnose.sh", copy)
    proc = subprocess.run(["bash", str(copy)], env=_env(home, DASHBOARD_PORT="1", **extra),
                          capture_output=True, text=True, timeout=120, check=False)
    return proc.stdout + proc.stderr


@posix_only
def test_diagnose_sh_finds_db_in_fresh_tam_dir(tmp_path):
    _db(tmp_path / ".tam")
    out = _diagnose_sh(tmp_path)
    assert "DB missing" not in out
    assert str(tmp_path / ".tam" / "memory.db") in out


@posix_only
def test_diagnose_sh_honours_tam_memory_dir(tmp_path):
    custom = tmp_path / "custom"
    _db(custom)
    out = _diagnose_sh(tmp_path, TAM_MEMORY_DIR=str(custom))
    assert "DB missing" not in out
    assert str(custom / "memory.db") in out


@posix_only
def test_diagnose_sh_tam_env_wins_over_legacy_env(tmp_path):
    new, old = tmp_path / "new", tmp_path / "old"
    _db(new)
    out = _diagnose_sh(tmp_path, TAM_MEMORY_DIR=str(new), CLAUDE_MEMORY_DIR=str(old))
    assert "DB missing" not in out


@posix_only
def test_diagnose_sh_still_reads_legacy_env(tmp_path):
    legacy = tmp_path / "legacy-env"
    _db(legacy)
    assert "DB missing" not in _diagnose_sh(tmp_path, CLAUDE_MEMORY_DIR=str(legacy))


@posix_only
def test_diagnose_sh_still_reads_legacy_claude_memory_dir(tmp_path):
    _db(tmp_path / ".claude-memory")
    out = _diagnose_sh(tmp_path)
    assert "DB missing" not in out
    assert str(tmp_path / ".claude-memory" / "memory.db") in out


@posix_only
def test_diagnose_sh_reports_fresh_tam_path_when_nothing_exists(tmp_path):
    out = _diagnose_sh(tmp_path)
    assert f"DB missing: {tmp_path / '.tam' / 'memory.db'}" in out


@posix_only
def test_diagnose_sh_warns_when_the_installed_dir_has_no_db_yet(tmp_path):
    (tmp_path / ".tam").mkdir()
    out = _diagnose_sh(tmp_path)
    assert f"WARN DB not created yet: {tmp_path / '.tam' / 'memory.db'}" in out
    assert "DB missing" not in out


def _resolved(tool: Path, attr: str, home: Path, **extra: str) -> Path:
    proc = subprocess.run([sys.executable, "-c", RESOLVE_AND_PRINT, str(tool), attr],
                          env=_env(home, PYTHONPATH=str(ROOT / "src"), **extra),
                          capture_output=True, text=True, timeout=120, check=False, cwd=str(home))
    assert proc.returncode == 0, proc.stderr
    return Path(proc.stdout.strip().splitlines()[-1])


PY_TOOLS = [
    pytest.param(SCRIPTS / "cm-import", "DB_PATH", id="cm-import"),
    pytest.param(SCRIPTS / "consolidation-daemon", "_DEFAULT_DB", id="consolidation-daemon"),
    pytest.param(SCRIPTS / "reembed.py", "DEFAULT_DB", id="reembed"),
]


@pytest.mark.parametrize(("tool", "attr"), PY_TOOLS)
def test_python_tool_defaults_to_fresh_tam_dir(tool, attr, tmp_path):
    assert _resolved(tool, attr, tmp_path).resolve() == (tmp_path / ".tam" / "memory.db").resolve()


@pytest.mark.parametrize(("tool", "attr"), PY_TOOLS)
def test_python_tool_honours_tam_memory_dir(tool, attr, tmp_path):
    custom = tmp_path / "custom"
    assert _resolved(tool, attr, tmp_path, TAM_MEMORY_DIR=str(custom)) == custom / "memory.db"


@pytest.mark.parametrize(("tool", "attr"), PY_TOOLS)
def test_python_tool_honours_legacy_env(tool, attr, tmp_path):
    custom = tmp_path / "legacy"
    assert _resolved(tool, attr, tmp_path, CLAUDE_MEMORY_DIR=str(custom)) == custom / "memory.db"


@pytest.mark.parametrize(("tool", "attr"), PY_TOOLS)
def test_python_tool_still_finds_legacy_claude_memory_dir(tool, attr, tmp_path):
    _db(tmp_path / ".claude-memory")
    assert _resolved(tool, attr, tmp_path).resolve().read_bytes() == b""


def test_consolidation_daemon_keeps_memory_db_path_override(tmp_path):
    override = tmp_path / "x" / "other.db"
    got = _resolved(SCRIPTS / "consolidation-daemon", "_DEFAULT_DB", tmp_path, MEMORY_DB_PATH=str(override))
    assert got == override


def test_calibrate_nli_default_output_uses_resolved_memory_dir():
    source = (SCRIPTS / "calibrate_nli.py").read_text(encoding="utf-8")
    assert "~/.claude-memory" not in source
    assert "memory_dir()" in source


@posix_only
def test_dashboard_service_management_output_shows_resolved_dir(tmp_path):
    _db(tmp_path / ".tam")
    env = _env(tmp_path, CTM_UNAME="Darwin")
    out = subprocess.run(["bash", str(SCRIPTS / "dashboard-service.sh"), "print-management"],
                         env=env, capture_output=True, text=True, timeout=60, check=False).stdout
    assert f"tail -f {tmp_path / '.tam'}/logs/dashboard.log" in out
    assert ".claude-memory" not in out
    explicit = subprocess.run(["bash", str(SCRIPTS / "dashboard-service.sh"), "print-management", "/data/mem"],
                              env=env, capture_output=True, text=True, timeout=60, check=False).stdout
    assert "tail -f /data/mem/logs/dashboard.log" in explicit


@pytest.mark.parametrize("name", ["diagnose.ps1", "install-codex.ps1"])
def test_powershell_resolver_follows_paths_order(name):
    path = SCRIPTS / name if name == "diagnose.ps1" else ROOT / name
    text = path.read_text(encoding="utf-8")
    block = text[text.index("$MemoryDir"):]
    positions = [block.index(token) for token in
                 ("$env:TAM_MEMORY_DIR", "$env:CLAUDE_MEMORY_DIR", '".tam"', '".claude-memory"')]
    assert positions == sorted(positions)
    assert 'Join-Path $env:USERPROFILE ".claude-memory" }' not in text


def test_install_codex_passes_both_env_names():
    text = (ROOT / "install-codex.ps1").read_text(encoding="utf-8")
    assert '--env "TAM_MEMORY_DIR=$MemoryDir"' in text
    assert '--env "CLAUDE_MEMORY_DIR=$MemoryDir"' in text
