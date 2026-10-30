"""Regression tests for the 14.10.0 integration review (hooks, installers, docker, update.sh)."""
from __future__ import annotations

import importlib.util
import re
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
HELPER = ROOT / "hooks" / "lib" / "tam_hook.py"

FAKE_KEY = "sk-ant-api03-" + "A1b2C3d4E5f6G7h8I9j0"
FAKE_PG = "hunter2SuperSecret"


def _load_hook():
    spec = importlib.util.spec_from_file_location("tam_hook_hardening", HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tam_hook = _load_hook()


def _failing_bash(command: str, stderr: str) -> dict:
    return {"tool_name": "Bash", "tool_input": {"command": command},
            "tool_response": {"stderr": stderr, "exit_code": 1}, "session_id": "s1", "cwd": "/w/proj"}


class Calls(list):
    def __call__(self, src, script, args, stdin_text=None):
        self.append((script, list(args), stdin_text))
        return True


# ── H1: Bash failure ─────────────────────────────────────────────

@pytest.mark.parametrize("src", [SRC, Path("/nonexistent-src")], ids=["shared-redactor", "fallback"])
def test_h1_failed_command_is_redacted_everywhere(tmp_path, src):
    calls = Calls()
    cmd = f'curl -H "Authorization: Bearer {FAKE_KEY}" https://api.example.test'
    err = f"401 Unauthorized for token={FAKE_KEY}"
    messages = tam_hook.memory_trigger(_failing_bash(cmd, err), tmp_path, src, "proj", runner=calls)
    state = (tmp_path / "last-error-proj.json").read_text(encoding="utf-8")
    argv_blob = json.dumps([c[1] for c in calls])
    for blob in (state, argv_blob, "\n".join(messages)):
        assert FAKE_KEY not in blob
    assert calls and calls[0][0] == "auto_self_improve.py"
    assert "--description" not in calls[0][1], "description must not travel in argv"
    assert FAKE_KEY not in calls[0][2]
    assert "Command failed" in calls[0][2]


def test_h1_fix_reinjection_redacts_old_state_files(tmp_path):
    (tmp_path / "last-error-proj.json").write_text(json.dumps(
        {"cmd": f"curl -u admin:pw --token {FAKE_KEY}", "msg": f"bad PGPASSWORD={FAKE_PG}",
         "category": "code_error", "time": int(__import__("time").time())}), encoding="utf-8")
    calls = Calls()
    ok = {"tool_name": "Bash", "tool_input": {"command": "curl https://x.test"},
          "tool_response": {"exit_code": 0}, "session_id": "s1"}
    messages = tam_hook.memory_trigger(ok, tmp_path, SRC, "proj", runner=calls)
    joined = "\n".join(messages) + json.dumps(calls)
    assert FAKE_KEY not in joined and FAKE_PG not in joined
    assert any(m.startswith("ERROR_FIXED") for m in messages)


def test_h1_auto_self_improve_reads_stdin_and_redacts(tmp_path):
    db = _store(tmp_path)
    env = {**os.environ, "TAM_MEMORY_DIR": str(tmp_path), "CLAUDE_MEMORY_DIR": str(tmp_path),
           "HOME": str(tmp_path)}
    proc = subprocess.run(
        [sys.executable, str(SRC / "auto_self_improve.py"), "error", "--description-stdin",
         "--category", "code_error", "--project", "proj"],
        input=f"Command failed: curl --token {FAKE_KEY}", env=env, capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    with sqlite3.connect(db) as conn:
        rows = conn.execute("SELECT description FROM errors").fetchall()
    assert rows and FAKE_KEY not in rows[0][0]


# ── H2: SessionEnd ───────────────────────────────────────────────

def _transcript(path: Path, text: str) -> None:
    entries = [{"type": "user", "message": {"role": "user", "content": text}},
               {"type": "assistant", "message": {"role": "assistant", "content": [
                   {"type": "text", "text": f"ok, using {FAKE_KEY}"}]}}]
    path.write_text("\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8")


def test_h2_recovery_file_and_stdin_payload_are_redacted(tmp_path):
    transcript = tmp_path / "t.jsonl"
    _transcript(transcript, f"my db password is PGPASSWORD={FAKE_PG} and key {FAKE_KEY}")
    calls = Calls()
    recovery = tmp_path / "recovery"
    tam_hook.session_end({"session_id": "s9", "cwd": str(tmp_path), "transcript_path": str(transcript)},
                         SRC, recovery, "proj", "exit", runner=calls)
    body = next(recovery.glob("pending-*.md")).read_text(encoding="utf-8")
    assert FAKE_KEY not in body and FAKE_PG not in body
    save = next(c for c in calls if c[0] == "auto_session_save.py")
    assert FAKE_KEY not in save[2] and FAKE_PG not in save[2]
    assert "--user-context" not in save[1]
    mode = next(recovery.glob("pending-*.md")).stat().st_mode & 0o777
    assert mode == 0o600 or os.name == "nt"


def test_h2_auto_session_save_stores_no_prompt_text(tmp_path):
    db = _store(tmp_path)
    env = {**os.environ, "TAM_MEMORY_DIR": str(tmp_path), "CLAUDE_MEMORY_DIR": str(tmp_path),
           "HOME": str(tmp_path)}
    payload = json.dumps({"user": f"deploy with PGPASSWORD={FAKE_PG} please", "assistant": "done"})
    proc = subprocess.run(
        [sys.executable, str(SRC / "auto_session_save.py"), "--project", "proj", "--cwd", str(tmp_path),
         "--reason", "exit", "--context-stdin"], input=payload, env=env, capture_output=True,
        text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    with sqlite3.connect(db) as conn:
        rows = conn.execute("SELECT content, context FROM knowledge").fetchall()
    assert len(rows) == 1
    assert FAKE_PG not in rows[0][0] and "deploy with" not in rows[0][0]


def _store(tmp_path: Path) -> Path:
    """Minimal store with the columns the hook-side writers use."""
    db = tmp_path / "memory.db"
    with sqlite3.connect(db) as conn:
        conn.executescript("""
            CREATE TABLE knowledge (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, type TEXT,
                content TEXT, context TEXT, project TEXT, tags TEXT, status TEXT, confidence REAL,
                source TEXT, created_at TEXT, last_confirmed TEXT, recall_count INTEGER);
            CREATE TABLE errors (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, category TEXT,
                severity TEXT, description TEXT, context TEXT, fix TEXT, project TEXT, tags TEXT,
                status TEXT, resolved_at TEXT, created_at TEXT);
        """)
    return db


# ── H4: install-docker.sh ────────────────────────────────────────

BASH = __import__("shutil").which("bash")


def _docker_checkout(tmp_path: Path) -> tuple[Path, Path, dict]:
    import shutil

    root = tmp_path / "checkout"
    shutil.copytree(ROOT / "hooks", root / "hooks", ignore=shutil.ignore_patterns("_docker-launchers", "__pycache__"))
    shutil.copytree(ROOT / "src" / "setup_wizard", root / "src" / "setup_wizard",
                    ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(ROOT / "install-docker.sh", root / "install-docker.sh")
    home = tmp_path / "home"
    (home / ".claude").mkdir(parents=True)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    docker = bin_dir / "docker"
    docker.write_text('#!/usr/bin/env bash\n[ "$1" = exec ] && { printf "tok-from-container"; exit 0; }\nexit 0\n', encoding="utf-8")
    curl = bin_dir / "curl"
    curl.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    for tool in (docker, curl):
        tool.chmod(0o755)
    env = {**os.environ, "HOME": str(home), "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"}
    env.pop("TAM_HTTP_TOKEN", None)
    return root, home, env


def _run_docker_install(root: Path, env: dict) -> subprocess.CompletedProcess:
    return subprocess.run([BASH, str(root / "install-docker.sh")], env=env, capture_output=True, text=True,
                          timeout=120, check=False)


@pytest.mark.skipif(not BASH or sys.platform == "win32", reason="POSIX installer")
class TestInstallDocker:
    def test_user_hooks_survive_and_backup_is_written(self, tmp_path):
        root, home, env = _docker_checkout(tmp_path)
        settings = home / ".claude" / "settings.json"
        mine = {"hooks": {"SessionStart": [{"matcher": "", "hooks": [{"type": "command", "command": "/u/a2a.sh"}]}],
                          "Notification": [{"matcher": "", "hooks": [{"type": "command", "command": "/u/n.sh"}]}]},
                "model": "opus"}
        settings.write_text(json.dumps(mine), encoding="utf-8")
        proc = _run_docker_install(root, env)
        assert proc.returncode == 0, proc.stderr + proc.stdout
        after = json.loads(settings.read_text(encoding="utf-8"))
        cmds = [h["command"] for b in after["hooks"]["SessionStart"] for h in b["hooks"]]
        assert "/u/a2a.sh" in cmds and any("_docker-launchers" in c for c in cmds)
        assert after["hooks"]["Notification"] == mine["hooks"]["Notification"] and after["model"] == "opus"
        assert list((home / ".claude").glob("settings.json.bak-*")), "settings.json must be backed up"
        again = _run_docker_install(root, env)
        assert again.returncode == 0
        rerun = json.loads(settings.read_text(encoding="utf-8"))
        assert len([h for b in rerun["hooks"]["SessionStart"] for h in b["hooks"]]) == len(cmds)

    def test_invalid_json_is_refused_and_left_untouched(self, tmp_path):
        root, home, env = _docker_checkout(tmp_path)
        settings = home / ".claude" / "settings.json"
        settings.write_text('{"permissions": {"allow": ["Bash"],},}', encoding="utf-8")
        proc = _run_docker_install(root, env)
        assert proc.returncode != 0
        assert settings.read_text(encoding="utf-8") == '{"permissions": {"allow": ["Bash"],},}'
        assert not (home / ".claude.json").exists()

    def test_mcp_entry_goes_to_claude_json_with_token(self, tmp_path):
        root, home, env = _docker_checkout(tmp_path)
        proc = _run_docker_install(root, env)
        assert proc.returncode == 0, proc.stderr
        entry = json.loads((home / ".claude.json").read_text(encoding="utf-8"))["mcpServers"]["memory"]
        assert entry["type"] == "http" and entry["headers"]["Authorization"] == "Bearer tok-from-container"
        assert "mcpServers" not in json.loads((home / ".claude" / "settings.json").read_text(encoding="utf-8"))
        if os.name != "nt":
            assert (home / ".claude.json").stat().st_mode & 0o777 == 0o600

    def test_legacy_mcp_entry_in_settings_is_removed(self, tmp_path):
        root, home, env = _docker_checkout(tmp_path)
        settings = home / ".claude" / "settings.json"
        settings.write_text(json.dumps({"mcpServers": {"memory": {"type": "http", "url": "http://127.0.0.1:3737/mcp"},
                                                       "other": {"command": "x"}}}), encoding="utf-8")
        assert _run_docker_install(root, env).returncode == 0
        assert list(json.loads(settings.read_text(encoding="utf-8"))["mcpServers"]) == ["other"]


# ── M1: post-tool-use matcher ────────────────────────────────────

def test_m1_post_tool_use_matcher_is_limited_to_logged_tools():
    sys.path.insert(0, str(SRC))
    from setup_wizard.extras import HOOK_GROUPS

    groups = [(event, matcher) for event, matcher, names in HOOK_GROUPS if "post-tool-use" in names]
    assert groups and all(matcher for _event, matcher in groups), "an empty matcher runs Python for Read/Grep/Glob"
    for _event, matcher in groups:
        assert not {"Read", "Grep", "Glob"} & set(matcher.split("|"))
        assert {"Edit", "Write", "Bash"} <= set(matcher.split("|"))


def test_post_tool_use_matcher_reaches_memory_saves_for_the_nudge():
    sys.path.insert(0, str(SRC))
    from setup_wizard.extras import HOOK_GROUPS

    plugin = json.loads((ROOT / "hooks" / "hooks.json").read_text(encoding="utf-8"))["hooks"]
    matchers = [m for _e, m, names in HOOK_GROUPS if "post-tool-use" in names]
    matchers += [b["matcher"] for event in ("PostToolUse", "PostToolUseFailure") for b in plugin[event]
                 if any(h["command"].endswith('post-tool-use.sh"') for h in b["hooks"])]
    assert len(matchers) == 4
    for matcher in matchers:
        for tool in ("Edit", "mcp__memory__memory_save", "mcp__memory__save_decision"):
            assert re.fullmatch(matcher, tool), (matcher, tool)
        assert not re.search(matcher, "Read")


# ── M2: interpreter discovery ────────────────────────────────────

def _resolved_python(tmp_path: Path, layout: str, where: str) -> str:
    home = tmp_path / "home"
    mem = tmp_path / "mem"
    install = tmp_path / "install"
    for d in (home, mem, install):
        d.mkdir(exist_ok=True)
    base = {"mem": mem, "tam": home / ".tam", "install": install}[where]
    exe = base / ".venv" / layout
    exe.parent.mkdir(parents=True)
    exe.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    exe.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("TAM_", "CLAUDE_MEMORY", "HOOK_"))}
    env.update({"HOME": str(home), "TAM_MEMORY_DIR": str(mem), "CLAUDE_MEMORY_INSTALL_DIR": str(install)})
    proc = subprocess.run(
        [BASH, "-c", f'HOOK_PARSE_FIELDS=0; HOOK_SKIP_STDIN=1; . "{ROOT}/hooks/lib/common.sh"; printf %s "$HOOK_PYTHON"'],
        env=env, capture_output=True, text=True, timeout=30, check=False)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


@pytest.mark.skipif(not BASH, reason="bash required")
@pytest.mark.parametrize("where", ["install", "mem", "tam"])
@pytest.mark.parametrize("layout", ["bin/python", "Scripts/python.exe"])
def test_m2_hooks_find_the_venv_where_the_plugin_and_windows_put_it(tmp_path, where, layout):
    found = _resolved_python(tmp_path, layout, where)
    assert found.endswith(layout) and ".venv" in found


def test_m2_ps1_probes_memory_dir_and_home_venvs():
    text = (ROOT / "hooks" / "lib" / "common.ps1").read_text(encoding="utf-8")
    assert "$MemoryDir" in text.split("$HookPython = $env:HOOK_PYTHON", 1)[1].split("function Write-HookLog", 1)[0]
    assert '(Join-Path $HOME ".tam")' in text


# ── PowerShell 5.1 compatibility (static: pwsh/powershell.exe are absent on this runner) ──

PS1_FILES = sorted(p for pattern in ("*.ps1", "hooks/*.ps1", "hooks/lib/*.ps1", "scripts/*.ps1")
                   for p in ROOT.glob(pattern))
PS51_FORBIDDEN = {
    "Join-Path with more than two positional arguments (PowerShell 6+)":
        re.compile(r"Join-Path\s+(?:\(.*?\)|\"[^\"]*\"|'[^']*'|\$\w+)\s+(?:\"[^\"]*\"|'[^']*'|\$\w+)\s+(?:\"[^\"]*\"|'[^']*'|\$\w+)"),
    "null-coalescing / null-conditional operators (7+)": re.compile(r"\?\?|\?\.\w"),
    "pipeline chain operators (7+)": re.compile(r"(?<![|&\"'])(?:&&|\|\|)(?![|&\"'])"),
    "$IsWindows/$IsMacOS/$IsLinux (absent in 5.1, an error under StrictMode)":
        re.compile(r"\$Is(?:Windows|MacOS|Linux)\b"),
    "ConvertFrom-Json -AsHashtable (6+)": re.compile(r"-AsHashtable"),
    "Get-Content -AsByteStream (6+)": re.compile(r"-AsByteStream"),
}


@pytest.mark.parametrize("name", sorted(PS51_FORBIDDEN))
def test_ps1_scripts_avoid_constructs_missing_from_windows_powershell_5_1(name):
    offenders = []
    for path in PS1_FILES:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            if PS51_FORBIDDEN[name].search(line):
                offenders.append(f"{path.relative_to(ROOT)}:{number}: {line.strip()}")
    assert not offenders, name + "\n" + "\n".join(offenders)


def test_m3_codex_notify_ps1_pipes_the_payload_instead_of_passing_it_as_an_argument():
    text = (ROOT / "hooks" / "codex-notify.ps1").read_text(encoding="utf-8")
    code = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    assert "--payload" not in code
    assert re.search(r"\$Payload\s*\|\s*&\s*\$HookPython\s+\$HookHelper\s+codex-notify", code)


def test_l7_codex_wrapper_quotes_cmd_set_values():
    text = (ROOT / "install-codex.ps1").read_text(encoding="utf-8")
    assert 'set "CLAUDE_MEMORY_DIR=$MemoryDir"' in text


# ── update.sh: rollback after a red gate, online SQLite backup, private log ──

def _git(cwd: Path, *args: str) -> str:
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.test",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.test"}
    return subprocess.run(["git", "-C", str(cwd), *args], env=env, capture_output=True, text=True, check=True).stdout.strip()


def _update_world(tmp_path: Path, pytest_exit: int) -> dict:
    remote = tmp_path / "remote.git"
    seed = tmp_path / "seed"
    subprocess.run(["git", "init", "-q", "--bare", str(remote)], check=True)
    subprocess.run(["git", "clone", "-q", str(remote), str(seed)], check=True, capture_output=True)
    (seed / "requirements.txt").write_text("# none\n", encoding="utf-8")
    (seed / "marker.txt").write_text("old\n", encoding="utf-8")
    _git(seed, "add", ".")
    _git(seed, "commit", "-q", "-m", "old")
    _git(seed, "push", "-q", "origin", "HEAD")
    root = tmp_path / "root"
    subprocess.run(["git", "clone", "-q", str(remote), str(root)], check=True, capture_output=True)
    old_sha = _git(root, "rev-parse", "HEAD")
    (seed / "marker.txt").write_text("new\n", encoding="utf-8")
    _git(seed, "commit", "-q", "-am", "new")
    _git(seed, "push", "-q", "origin", "HEAD")

    (root / ".venv" / "bin").mkdir(parents=True)
    pip_log = tmp_path / "pip.log"
    pip = root / ".venv" / "bin" / "pip"
    pip.write_text(f'#!/bin/sh\necho "$@" >> "{pip_log}"\n', encoding="utf-8")
    py = tmp_path / "fakepy"
    py.write_text(f'#!/bin/sh\ncase "$1 $2" in\n  "-m pytest") exit {pytest_exit} ;;\n  "-c "*) echo 9.9.9 ;;\nesac\n'
                  'case "$1" in -) cat >/dev/null ;; esac\nexit 0\n', encoding="utf-8")
    for tool in (pip, py):
        tool.chmod(0o755)
    home = tmp_path / "home"
    mem = tmp_path / "mem"
    home.mkdir()
    mem.mkdir()
    env = {**os.environ, "HOME": str(home), "TAM_MEMORY_DIR": str(mem), "ROOT": str(root), "PY": str(py)}
    env.pop("CLAUDE_MEMORY_DIR", None)
    return {"root": root, "old_sha": old_sha, "env": env, "mem": mem, "pip_log": pip_log}


def _run_update(world: dict) -> subprocess.CompletedProcess:
    return subprocess.run([BASH, str(ROOT / "update.sh")], env=world["env"], capture_output=True, text=True,
                          timeout=180, check=False)


@pytest.mark.skipif(not BASH or not __import__("shutil").which("git"), reason="bash and git required")
class TestUpdateScript:
    def test_m5_red_gate_restores_the_previous_source(self, tmp_path):
        world = _update_world(tmp_path, pytest_exit=1)
        proc = _run_update(world)
        assert proc.returncode != 0
        assert "rolled back" in proc.stdout + proc.stderr
        assert _git(world["root"], "rev-parse", "HEAD") == world["old_sha"]
        assert (world["root"] / "marker.txt").read_text(encoding="utf-8") == "old\n"
        assert "-r" in world["pip_log"].read_text(encoding="utf-8"), "previous requirements are reinstalled"

    def test_m5_rollback_keeps_uncommitted_local_edits(self, tmp_path):
        world = _update_world(tmp_path, pytest_exit=1)
        (world["root"] / "requirements.txt").write_text("# local edit\n", encoding="utf-8")
        proc = _run_update(world)
        assert proc.returncode != 0
        assert (world["root"] / "requirements.txt").read_text(encoding="utf-8") == "# local edit\n"

    def test_l4_log_lives_in_the_private_memory_dir_not_tmp(self, tmp_path):
        world = _update_world(tmp_path, pytest_exit=1)
        _run_update(world)
        assert (world["mem"] / "logs" / "update.log").is_file()
        text = (ROOT / "update.sh").read_text(encoding="utf-8")
        assert "/tmp/" not in "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))

    def test_m6_snapshot_contains_uncheckpointed_wal_rows(self, tmp_path):
        world = _update_world(tmp_path, pytest_exit=1)
        db = world["mem"] / "memory.db"
        writer = sqlite3.connect(db)
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("PRAGMA wal_autocheckpoint=0")
        writer.execute("CREATE TABLE t (v TEXT)")
        writer.executemany("INSERT INTO t VALUES (?)", [("row-%d" % i,) for i in range(50)])
        writer.commit()
        try:
            _run_update(world)
        finally:
            writer.close()
        import gzip
        import shutil

        snaps = sorted((world["mem"] / "backups").glob("memory.db.*.gz"))
        assert snaps, "a snapshot is written before the update"
        restored = tmp_path / "restored.db"
        with gzip.open(snaps[-1], "rb") as src, open(restored, "wb") as dst:
            shutil.copyfileobj(src, dst)
        with sqlite3.connect(restored) as conn:
            assert conn.execute("SELECT COUNT(*) FROM t").fetchone()[0] == 50

    def test_m7_update_refreshes_installed_hooks(self):
        text = (ROOT / "update.sh").read_text(encoding="utf-8")
        assert "setup_wizard.refresh_hooks" in text


# ── M7 / M8 ──────────────────────────────────────────────────────

def _host(home: Path):
    sys.path.insert(0, str(SRC))
    from setup_wizard.clients import Host

    return Host(home, "Linux", {}, lambda _name: None)


def test_m7_refresh_updates_installed_runtime_and_scripts_with_backup(tmp_path):
    from setup_wizard import extras

    home = tmp_path / "home"
    host = _host(home)
    installed = home / ".claude" / "hooks"
    (installed / "lib").mkdir(parents=True)
    (installed / "lib" / "tam_hook.py").write_text("OLD RUNTIME", encoding="utf-8")
    (installed / "pre-edit.sh").write_text("OLD SCRIPT", encoding="utf-8")
    written = extras.refresh_installed_hooks(ROOT, host, tmp_path / "mem")
    assert (installed / "lib" / "tam_hook.py").read_text(encoding="utf-8") == (ROOT / "hooks" / "lib" / "tam_hook.py").read_text(encoding="utf-8")
    assert (installed / "pre-edit.sh").read_text(encoding="utf-8") == (ROOT / "hooks" / "pre-edit.sh").read_text(encoding="utf-8")
    assert (installed / "pre-edit.sh.bak").read_text(encoding="utf-8") == "OLD SCRIPT"
    assert not (installed / "on-stop.sh").exists(), "scripts the user never installed are not added"
    assert not (home / ".claude" / "settings.json").exists()
    assert installed / "lib" / "tam_hook.py" in written


def test_m7_refresh_does_nothing_without_installed_hooks(tmp_path):
    from setup_wizard import extras

    assert extras.refresh_installed_hooks(ROOT, _host(tmp_path / "home")) == []


def test_m8_wizard_skips_settings_hooks_when_the_plugin_is_enabled(tmp_path):
    from setup_wizard import extras

    home = tmp_path / "home"
    (home / ".claude").mkdir(parents=True)
    (home / ".claude" / "settings.json").write_text(
        json.dumps({"enabledPlugins": {"total-agent-memory@vbcherepanov": True}}), encoding="utf-8")
    plan = extras.plan_hooks(ROOT, _host(home))
    assert plan.plugin_active and plan.added == ()
    assert "SessionStart" not in json.loads(plan.text).get("hooks", {})
    (home / ".claude" / "settings.json").write_text(
        json.dumps({"enabledPlugins": {"total-agent-memory@vbcherepanov": False}}), encoding="utf-8")
    assert not extras.plan_hooks(ROOT, _host(home)).plugin_active


def test_m8_record_action_is_idempotent_per_tool_use_id(tmp_path):
    sys.path.insert(0, str(SRC))
    from ambient.state import claim_tool_use

    assert claim_tool_use(tmp_path, "s1", "toolu_01") is True
    assert claim_tool_use(tmp_path, "s1", "toolu_01") is False
    assert claim_tool_use(tmp_path, "s2", "toolu_01") is True
    assert claim_tool_use(tmp_path, "s1", "toolu_02") is True


def test_m8_duplicate_hook_fire_logs_one_action(tmp_path):
    sys.path.insert(0, str(SRC))
    from ambient import hooks as ambient_hooks

    db = _store(tmp_path)
    with sqlite3.connect(db) as conn:
        conn.executescript(
            "CREATE TABLE agent_actions (id INTEGER PRIMARY KEY, ts TEXT NOT NULL, client TEXT NOT NULL DEFAULT 'claude-code',"
            " session_id TEXT NOT NULL DEFAULT '', project TEXT NOT NULL DEFAULT 'general', tool TEXT NOT NULL,"
            " summary TEXT NOT NULL DEFAULT '', status TEXT NOT NULL DEFAULT 'ok', exit_code INTEGER);")
    call = ambient_hooks.HookCall(payload={"tool_name": "Bash", "tool_use_id": "toolu_9",
                                           "tool_input": {"command": "ls"}}, project="p", session_id="s1")
    ambient_hooks.record_action(call, tmp_path)
    ambient_hooks.record_action(call, tmp_path)
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM agent_actions").fetchone()[0] == 1


# ── L1: dashboard escaping ───────────────────────────────────────

def _dashboard_source() -> str:
    return (SRC / "dashboard.py").read_text(encoding="utf-8")


def test_l1_escape_html_escapes_quotes_and_badge_classes_are_whitelisted():
    text = _dashboard_source()
    body = text.split("function escapeHtml(str) {", 1)[1].split("\n}\n", 1)[0]
    assert "textContent" not in body and "&quot;" in text and "&#39;" in text
    assert "function cssToken(" in text
    for raw in ("'badge badge-' + type", "'badge badge-' + (n.type || 'fact')", "'badge badge-' + (type || 'fact')",
                "'badge badge-' + s +"):
        assert raw not in text, raw


@pytest.mark.skipif(not __import__("shutil").which("node"), reason="node not installed")
def test_l1_escape_html_in_node():
    text = _dashboard_source()
    start = text.index("const HTML_ESCAPES")
    end = text.index("function truncate(")
    script = text[start:end] + (
        "\nconsole.log(JSON.stringify([escapeHtml('a\"b\\'c<d>&'), cssToken('x\" onmouseover=\"y', 'fact'), cssToken('Decision', 'fact')]));")
    out = subprocess.run(["node", "-e", script], capture_output=True, text=True, timeout=30, check=True).stdout
    assert json.loads(out) == ["a&quot;b&#39;c&lt;d&gt;&amp;", "fact", "decision"]


def test_l1_graph_pages_escape_node_names():
    text = (SRC / "dashboard_v6.py").read_text(encoding="utf-8")
    for raw in ("${f.name}", "${f.last_error}", "${n.label}</b>", "${nodes[si].name}"):
        assert raw not in text, raw
    assert text.count("const esc = ") == 3


# ── L2 / L3: workflows ───────────────────────────────────────────

WORKFLOWS = ROOT / ".github" / "workflows"
# The source distribution carries no .github/; these checks belong to the repository checkout.
needs_workflows = pytest.mark.skipif(not WORKFLOWS.is_dir(), reason=".github/ is not shipped in the source distribution")
PINNED_WORKFLOWS = ("docker.yml", "joss-paper.yml", "smoke.yml", "postgres.yml", "codeql.yml")


@needs_workflows
@pytest.mark.parametrize("name", PINNED_WORKFLOWS)
def test_l3_third_party_actions_are_pinned_to_commit_shas(name):
    for number, line in enumerate((WORKFLOWS / name).read_text(encoding="utf-8").splitlines(), 1):
        match = re.search(r"uses:\s*([^\s#]+)", line)
        if match and not match.group(1).startswith("./"):
            assert re.search(r"@[0-9a-f]{40}$", match.group(1)), f"{name}:{number}: {line.strip()}"


def _resolve_script() -> str:
    import yaml

    workflow = yaml.safe_load((WORKFLOWS / "docker.yml").read_text(encoding="utf-8"))
    step = next(s for s in workflow["jobs"]["build"]["steps"] if s.get("id") == "version")
    assert "${{ inputs" not in step["run"], "workflow inputs must reach the shell through env:"
    return step["run"]


def _resolve(tmp_path: Path, version: str, event: str, existing: list[str]) -> dict[str, str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    listing = "\\n".join(f"abc\\trefs/tags/v{tag}" for tag in existing)
    git = bin_dir / "git"
    git.write_text(f"#!/bin/sh\nprintf '{listing}\\n'\n", encoding="utf-8")
    git.chmod(0o755)
    out = tmp_path / "out"
    out.write_text("", encoding="utf-8")
    env = {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}", "INPUT_TAG": version,
           "EVENT_NAME": event, "REGISTRY_IMAGE": "ghcr.io/o/img", "GITHUB_OUTPUT": str(out),
           "GITHUB_REPOSITORY": "o/r", "GITHUB_REF_NAME": f"v{version}"}
    proc = subprocess.run([BASH, "-eo", "pipefail", "-c", _resolve_script()], env=env, capture_output=True,
                          text=True, timeout=30, check=False)
    assert proc.returncode == 0, proc.stderr
    text = out.read_text(encoding="utf-8")
    return {"tags": text.split("tags<<EOF\n", 1)[1].split("EOF", 1)[0], "text": text}


@needs_workflows
@pytest.mark.skipif(not BASH, reason="bash required")
def test_l2_backfilling_an_old_version_does_not_move_latest_or_major(tmp_path):
    pytest.importorskip("yaml")
    tags = _resolve(tmp_path, "12.1.0", "workflow_dispatch", ["12.1.0", "12.3.0", "14.9.0"])["tags"].split()
    assert tags == ["ghcr.io/o/img:12.1.0"]


@needs_workflows
@pytest.mark.skipif(not BASH, reason="bash required")
def test_l2_new_highest_release_moves_latest_major_and_minor(tmp_path):
    pytest.importorskip("yaml")
    tags = _resolve(tmp_path, "14.10.0", "push", ["14.9.0", "14.10.0"])["tags"].split()
    assert tags == ["ghcr.io/o/img:14.10.0", "ghcr.io/o/img:14.10", "ghcr.io/o/img:14", "ghcr.io/o/img:latest"]


@needs_workflows
@pytest.mark.skipif(not BASH, reason="bash required")
def test_l2_manual_dispatch_never_publishes_latest(tmp_path):
    pytest.importorskip("yaml")
    tags = _resolve(tmp_path, "14.10.0", "workflow_dispatch", ["14.9.0", "14.10.0"])["tags"].split()
    assert "ghcr.io/o/img:latest" not in tags and "ghcr.io/o/img:14" in tags


@needs_workflows
def test_l2_docker_smoke_job_sends_the_access_token():
    text = (WORKFLOWS / "docker.yml").read_text(encoding="utf-8")
    assert "-p 127.0.0.1:37737:37737" in text and "http-token" in text


# ── L7: placeholder substitution and service-file quoting ────────

def _render(mode: str, template: str, **variables: str) -> str:
    args = [sys.executable, str(ROOT / "scripts" / "render_template.py"), "--mode", mode]
    for name, value in variables.items():
        args += ["--var", f"@{name}@={value}"]
    return subprocess.run(args, input=template, capture_output=True, text=True, timeout=30, check=True).stdout


def test_l7_renderer_inserts_values_literally():
    out = _render("systemd", "A=@INSTALL_DIR@|@INSTALL_DIR@", INSTALL_DIR="/opt/a&b|c %h \"q\"")
    assert out == 'A=/opt/a&b|c %%h \\"q\\"|/opt/a&b|c %%h \\"q\\"'
    assert _render("xml", "<s>@HOME@</s>", HOME="/h/a&b<c>") == "<s>/h/a&amp;b&lt;c&gt;</s>"


def test_l7_systemd_templates_quote_exec_and_environment():
    for unit in sorted((ROOT / "systemd").glob("*.service")):
        for line in unit.read_text(encoding="utf-8").splitlines():
            if line.startswith("ExecStart="):
                words = re.findall(r'"[^"]*"|\S+', line[len("ExecStart="):])
                assert all(w.startswith('"') for w in words if "@" in w), f"{unit.name}: {line}"
            if line.startswith("Environment=") and "@" in line:
                assert line.startswith('Environment="'), f"{unit.name}: {line}"


@pytest.mark.skipif(not BASH or sys.platform == "win32", reason="systemd units are installed on Linux only")
def test_l7_installed_units_survive_spaces_and_ampersands(tmp_path):
    install = tmp_path / "my dir & more"
    (install / "scripts").mkdir(parents=True)
    (install / "systemd").mkdir()
    __import__("shutil").copy2(ROOT / "scripts" / "render_template.py", install / "scripts")
    for unit in (ROOT / "systemd").glob("*.service"):
        __import__("shutil").copy2(unit, install / "systemd")
    out = tmp_path / "out"
    out.mkdir()
    script = (f'INSTALL_DIR="{install}"; MEMORY_DIR="{tmp_path}/mem & data"; HOME="{tmp_path}"; '
              'for t in "$INSTALL_DIR"/systemd/*.service; do python3 "$INSTALL_DIR/scripts/render_template.py" --mode systemd '
              '--var "@INSTALL_DIR@=$INSTALL_DIR" --var "@MEMORY_DIR@=$MEMORY_DIR" --var "@HOME@=$HOME" '
              f'< "$t" > "{out}/$(basename "$t")"; done')
    subprocess.run([BASH, "-c", script], check=True, timeout=60)
    text = (out / "claude-memory-dashboard.service").read_text(encoding="utf-8")
    assert f'ExecStart="{install}/.venv/bin/python" "{install}/src/dashboard.py"' in text
    assert "@INSTALL_DIR@" not in text


@pytest.mark.skipif(not BASH, reason="bash required")
def test_l7_dashboard_service_helpers_escape_values(tmp_path):
    text = (ROOT / "scripts" / "dashboard-service.sh").read_text(encoding="utf-8")
    helpers = tmp_path / "helpers.sh"
    helpers.write_text(text[text.index("xml_escape() {"):text.index("install_service() {")], encoding="utf-8")
    script = f'. "{helpers}"; xml_escape "a&b<c>"; echo; systemd_arg "/p a/b"; echo; systemd_arg "/plain/path"'
    out = subprocess.run([BASH, "-c", script], capture_output=True, text=True, timeout=30, check=True).stdout.splitlines()
    assert out == ["a&amp;b&lt;c&gt;", '"/p a/b"', "/plain/path"]


# ── M7 version stamp, M2 scenarios ───────────────────────────────

def test_m7_install_env_is_stamped_and_mismatch_is_logged_not_printed(tmp_path, capsys):
    from setup_wizard import extras

    text = extras.install_env_text(_host(tmp_path), ROOT, None)
    version = extras.checkout_version(ROOT)
    assert version and f"TAM_HOOKS_VERSION:=" in text and version in text
    checkout = tmp_path / "co" / "src"
    checkout.mkdir(parents=True)
    (checkout / "version.py").write_text('VERSION = "99.0.0"\n', encoding="utf-8")
    env = {"TAM_HOOKS_VERSION": version, "CLAUDE_MEMORY_INSTALL_DIR": str(tmp_path / "co")}
    assert tam_hook.check_runtime_version(env) is False
    captured = capsys.readouterr()
    assert captured.out == "" and "hook_runtime_version_mismatch" in captured.err and "99.0.0" in captured.err
    assert tam_hook.check_runtime_version({**env, "TAM_HOOKS_VERSION": "99.0.0"}) is True
    assert tam_hook.check_runtime_version({}) is True


@pytest.mark.skipif(not BASH, reason="bash required")
def test_m2_plugin_install_resolves_the_server_venv_in_the_memory_dir(tmp_path):
    # Plugin root has no .venv; the bootstrap venv lives in the memory dir.
    assert _resolved_python(tmp_path, "bin/python", "mem").endswith("mem/.venv/bin/python")


@pytest.mark.skipif(not BASH, reason="bash required")
def test_m2_git_bash_on_windows_resolves_scripts_python_exe(tmp_path):
    assert _resolved_python(tmp_path, "Scripts/python.exe", "install").endswith(".venv/Scripts/python.exe")
