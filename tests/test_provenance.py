"""Provenance of memory writes: `_ctx` from the PreToolUse hook, request _meta, migration 046."""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import provenance  # noqa: E402
from setup_wizard.clients import server_entry  # noqa: E402

MIGRATION = ROOT / "migrations" / "046_memory_provenance.sql"
HELPER = ROOT / "hooks" / "lib" / "tam_hook.py"
BASH = shutil.which("bash")


def _load_hook():
    spec = importlib.util.spec_from_file_location("tam_hook_provenance", HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tam_hook = _load_hook()


# ── parsing ──────────────────────────────────────────────────────

def test_split_call_strips_ctx_without_mutating_the_input():
    original = {"content": "x", "_ctx": {"agent": "Claude-Code", "session_id": "s1", "workspace": "src/app",
                                          "repo": "proj", "tool_use_id": "toolu_1"}}
    arguments, ctx = provenance.split_call(original)
    assert arguments == {"content": "x"} and "_ctx" in original
    assert ctx == provenance.CallProvenance(agent="claude-code", session_id="s1", workspace="src/app", repo="proj",
                                            tool_use_id="toolu_1", source="hook")


def test_arguments_without_ctx_or_meta_have_no_provenance():
    arguments = {"query": "q"}
    out, ctx = provenance.split_call(arguments)
    assert out is arguments and ctx is None
    assert provenance.split_call(None) == (None, None)
    assert provenance.split_call("not an object") == ("not an object", None)


@pytest.mark.parametrize("workspace", ["/Users/someone/proj", "C:\\Users\\x\\proj", "../escape", "a/../../b"])
def test_absolute_or_escaping_workspaces_are_never_stored(workspace):
    _, ctx = provenance.split_call({"_ctx": {"agent": "claude-code", "workspace": workspace, "repo": "p"}})
    assert ctx is not None and ctx.workspace == ""


def test_ctx_fields_are_cleaned_and_bounded():
    _, ctx = provenance.split_call({"_ctx": {"agent": "Evil Agent!\n", "session_id": "s\x00\x1b[31m" + "x" * 500,
                                              "repo": 7, "workspace": "a\\b\\c"}})
    assert ctx.agent == "evilagent" and len(ctx.session_id) == provenance.MAX_FIELD_CHARS
    assert "\x00" not in ctx.session_id and ctx.repo == "" and ctx.workspace == "a/b/c"


def test_a_ctx_that_is_not_an_object_or_empty_is_dropped():
    for junk in ("text", 5, [1], {}, {"workspace": "x/y"}):
        out, ctx = provenance.split_call({"k": 1, "_ctx": junk})
        assert out == {"k": 1} and ctx is None


def test_claude_code_request_meta():
    ctx = provenance.from_request_meta({"claudecode/toolUseId": "toolu_9", "progressToken": 3})
    assert ctx == provenance.CallProvenance(agent="claude-code", tool_use_id="toolu_9", source="meta")


@pytest.mark.parametrize("codex_meta", [
    {"session_id": "S1", "turn_id": "T1", "workspaces": {"/abs/path/to/repo": {"latest_git_commit_hash": "abc"}}},
    json.dumps({"session_id": "S1", "turn_id": "T1", "workspaces": {"/abs/path/to/repo": {}}}),
])
def test_codex_turn_metadata_dict_or_json_string(codex_meta):
    ctx = provenance.from_request_meta({"x-codex-turn-metadata": codex_meta})
    assert (ctx.agent, ctx.session_id, ctx.turn_id, ctx.repo) == ("codex", "S1", "T1", "repo")
    assert "/abs" not in json.dumps(ctx.as_dict())


def test_conversation_ids_from_other_clients_and_pydantic_meta():
    assert provenance.from_request_meta({"conversationId": "c-1"}).session_id == "c-1"
    assert provenance.from_request_meta({"threadId": "th"}).session_id == "th"
    assert provenance.from_request_meta({"unrelated": 1}) is None
    assert provenance.from_request_meta(None) is None

    import mcp.types as mcp_types

    params = mcp_types.CallToolRequestParams(name="memory_save", arguments={}, **{"_meta": {"claudecode/toolUseId": "t1"}})
    assert provenance.from_request_meta(params.meta).tool_use_id == "t1"


def test_hook_and_meta_merge_with_the_request_authoritative_for_the_call_id():
    _, ctx = provenance.split_call(
        {"_ctx": {"agent": "claude-code", "session_id": "hook-s", "workspace": "pkg", "repo": "r", "tool_use_id": "from-hook"}},
        {"claudecode/toolUseId": "from-meta", "sessionId": "meta-s"})
    assert (ctx.session_id, ctx.workspace, ctx.repo, ctx.tool_use_id, ctx.source) == \
        ("hook-s", "pkg", "r", "from-meta", "hook+meta")


# ── storage (migration 046) ──────────────────────────────────────

@pytest.fixture
def db_path(tmp_path):
    path = tmp_path / "memory.db"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE knowledge (id INTEGER PRIMARY KEY AUTOINCREMENT, content TEXT)")
        db.executescript(MIGRATION.read_text(encoding="utf-8"))
        db.executemany("INSERT INTO knowledge (content) VALUES (?)", [("a",), ("b",), ("c",)])
    return path


def _ctx() -> provenance.CallProvenance:
    return provenance.CallProvenance(agent="claude-code", session_id="s1", workspace="pkg", repo="r",
                                     tool_use_id="toolu_1", source="hook+meta")


def test_one_row_per_written_record(db_path):
    assert provenance.record(db_path, "memory_save", _ctx(), [1, 2, 2]) == 2
    with sqlite3.connect(db_path) as db:
        rows = db.execute("SELECT knowledge_id, tool, agent, session_id, workspace FROM memory_provenance ORDER BY 1").fetchall()
        assert rows == [(1, "memory_save", "claude-code", "s1", "pkg"), (2, "memory_save", "claude-code", "s1", "pkg")]
        assert provenance.provenance_for(db, 1)[0]["tool_use_id"] == "toolu_1"


def test_nothing_is_stored_without_provenance_or_written_records(db_path):
    assert provenance.record(db_path, "memory_save", None, [1]) == 0
    assert provenance.record(db_path, "memory_recall", _ctx(), []) == 0
    assert provenance.record(db_path, "memory_recall", _ctx(), None) == 0
    with sqlite3.connect(db_path) as db:
        assert db.execute("SELECT COUNT(*) FROM memory_provenance").fetchone()[0] == 0


def test_deleting_a_record_removes_its_provenance(db_path):
    provenance.record(db_path, "memory_save", _ctx(), [1, 2])
    with sqlite3.connect(db_path) as db:
        db.execute("DELETE FROM knowledge WHERE id = 1")
        assert [r[0] for r in db.execute("SELECT knowledge_id FROM memory_provenance")] == [2]


def test_recording_before_the_migration_is_logged_not_raised(tmp_path, caplog):
    path = tmp_path / "old.db"
    sqlite3.connect(path).close()
    with caplog.at_level("WARNING"):
        assert provenance.record(path, "memory_save", _ctx(), [1]) == 0
    assert "migration 046 not applied" in caplog.text


def test_migration_is_idempotent_and_additive(db_path):
    with sqlite3.connect(db_path) as db:
        db.executescript(MIGRATION.read_text(encoding="utf-8"))
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert tables >= {"knowledge", "memory_provenance"}
    text = MIGRATION.read_text(encoding="utf-8")
    assert "ALTER TABLE" not in text and "DROP" not in text


# ── the PreToolUse hook ──────────────────────────────────────────

def _git_repo(tmp_path: Path) -> Path:
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    repo = tmp_path / "proj"
    (repo / "pkg" / "deep").mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    return repo


_LAUNCH = server_entry({})
# The launch the installer itself registers on this machine, plus a secret that must never reach the marker.
TAM_ENTRY = {"command": _LAUNCH.command, "args": list(_LAUNCH.args), "env": {"TOKEN": "s3cret"}}
TAM_FINGERPRINT = {key: value for key, value in (("command", _LAUNCH.command), ("args", list(_LAUNCH.args))) if value}


def _register(home: Path, *, project_servers: dict | None = None, cwd: Path | None = None) -> Path:
    """A home where the installer registered the TAM server as "memory" and wrote the provenance marker."""
    sys.path.insert(0, str(ROOT / "src"))
    from setup_wizard.extras import write_provenance_marker

    config = {"mcpServers": {"memory": TAM_ENTRY}}
    if project_servers is not None:
        config["projects"] = {str(cwd): {"mcpServers": project_servers}}
    (home / ".claude.json").write_text(json.dumps(config), encoding="utf-8")
    (home / ".claude").mkdir(exist_ok=True)
    write_provenance_marker(home / ".claude")
    return home


@pytest.fixture
def tam_home(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    return _register(home)


def test_hook_input_adds_ctx_with_a_repo_relative_workspace(tmp_path, tam_home):
    repo = _git_repo(tmp_path)
    payload = {"tool_name": "mcp__memory__memory_save", "session_id": "sess-1", "tool_use_id": "toolu_7",
               "cwd": str(repo / "pkg" / "deep"), "tool_input": {"content": "hello", "type": "fact"}}
    updated = tam_hook.provenance_input(payload, tam_home)
    assert updated["content"] == "hello" and updated["type"] == "fact"
    assert updated["_ctx"] == {"agent": "claude-code", "session_id": "sess-1", "workspace": "pkg/deep",
                               "repo": "proj", "tool_use_id": "toolu_7"}
    assert str(tmp_path) not in json.dumps(updated)
    payload["cwd"] = str(repo)
    assert tam_hook.provenance_input(payload, tam_home)["_ctx"]["workspace"] == ""


def test_workspace_is_repo_relative_when_git_reports_the_root_with_forward_slashes(tmp_path, monkeypatch):
    # git prints C:/Users/... on Windows; the workspace must still come out as "pkg/deep".
    repo = tmp_path / "proj"
    deep = repo / "pkg" / "deep"
    deep.mkdir(parents=True)
    monkeypatch.setattr(tam_hook, "_git", lambda cwd, *args: repo.as_posix())
    assert tam_hook.workspace_of(str(deep)) == ("proj", "pkg/deep")
    assert tam_hook.workspace_of(str(repo)) == ("proj", "")


def test_hook_input_outside_a_repository_has_no_path_at_all(tmp_path, tam_home):
    plain = tmp_path / "scratch"
    plain.mkdir()
    updated = tam_hook.provenance_input({"tool_name": "mcp__memory__memory_recall", "session_id": "s", "cwd": str(plain),
                                         "tool_input": {"query": "q"}}, tam_home)
    assert updated["_ctx"]["repo"] == "scratch" and updated["_ctx"]["workspace"] == ""
    assert str(tmp_path) not in json.dumps(updated)


@pytest.mark.parametrize("payload", [
    {"tool_name": "Bash", "tool_input": {"command": "ls"}},
    {"tool_name": "mcp__other__memory_save", "tool_input": {"content": "x"}},
    {"tool_name": "mcp__memory__memory_save", "tool_input": {"content": "x", "_ctx": {"agent": "already"}}},
    {"tool_name": "mcp__memory__memory_save", "tool_input": "not an object"},
    {"tool_name": "mcp__memory__memory_save"},
])
def test_hook_leaves_other_calls_alone(payload, tam_home):
    assert tam_hook.provenance_input({**payload, "session_id": "s", "cwd": "/"}, tam_home) is None


def test_plugin_namespaced_memory_tools_match(tmp_path):
    payload = {"tool_name": "mcp__plugin_total-agent-memory_memory__memory_save", "session_id": "s", "cwd": "",
               "tool_input": {"content": "x"}}
    assert tam_hook.provenance_input(payload, tmp_path)["_ctx"]["agent"] == "claude-code"


@pytest.mark.skipif(not BASH, reason="bash required")
def test_hook_script_prints_updated_input_without_a_permission_decision(tmp_path):
    repo = _git_repo(tmp_path)
    _register(tmp_path)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("TAM_", "CLAUDE_MEMORY", "HOOK_"))}
    env.update({"HOME": str(tmp_path), "USERPROFILE": str(tmp_path), "HOOK_PYTHON": sys.executable, "TAM_MEMORY_DIR": str(tmp_path / "mem")})
    payload = {"hook_event_name": "PreToolUse", "tool_name": "mcp__memory__memory_save", "session_id": "s",
               "tool_use_id": "toolu_3", "cwd": str(repo), "tool_input": {"content": "c"}}
    proc = subprocess.run([BASH, str(ROOT / "hooks" / "inject-provenance.sh")], input=json.dumps(payload).encode(),
                          env=env, capture_output=True, timeout=30, check=False)
    assert proc.returncode == 0, proc.stderr.decode()
    out = json.loads(proc.stdout.decode())["hookSpecificOutput"]
    assert out["hookEventName"] == "PreToolUse" and "permissionDecision" not in out
    assert out["updatedInput"]["content"] == "c" and out["updatedInput"]["_ctx"]["tool_use_id"] == "toolu_3"
    silent = subprocess.run([BASH, str(ROOT / "hooks" / "inject-provenance.sh")],
                            input=json.dumps({"tool_name": "Read", "tool_input": {}}).encode(), env=env,
                            capture_output=True, timeout=30, check=False)
    assert silent.returncode == 0 and silent.stdout == b""


def test_installers_register_the_provenance_hook_and_ship_both_shells():
    sys.path.insert(0, str(ROOT / "src"))
    from setup_wizard.docker_install import DOCKER_HOOKS
    from setup_wizard.extras import HOOK_GROUPS, PROVENANCE_MATCHER

    assert ("PreToolUse", PROVENANCE_MATCHER, ("inject-provenance",)) in HOOK_GROUPS
    assert ("PreToolUse", PROVENANCE_MATCHER, "inject-provenance.sh") in DOCKER_HOOKS
    for suffix in (".sh", ".ps1"):
        assert (ROOT / "hooks" / f"inject-provenance{suffix}").is_file()
    plugin_hooks = json.loads((ROOT / "hooks" / "hooks.json").read_text(encoding="utf-8"))
    assert "inject-provenance" not in json.dumps(plugin_hooks), "provenance is installer-managed, not plugin-managed"


def test_ps1_twin_calls_the_same_helper_command():
    text = (ROOT / "hooks" / "inject-provenance.ps1").read_text(encoding="utf-8")
    assert "Invoke-HookHelper inject-provenance" in text and "common.ps1" in text


def test_ctx_and_meta_values_are_credential_redacted():
    secret = "ghp_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8"
    ctx = provenance.sanitize_ctx({"agent": "claude-code", "session_id": f"s {secret}", "repo": "password=hunter22"})
    assert secret not in ctx.session_id and "hunter22" not in ctx.repo
    meta = provenance.from_request_meta({"claudecode/toolUseId": f"tool {secret}"})
    assert secret not in meta.tool_use_id


# ── only the TAM server receives `_ctx` ──────────────────────────

def _call(cwd: Path, tool: str = "mcp__memory__memory_save") -> dict:
    return {"tool_name": tool, "session_id": "s", "cwd": str(cwd), "tool_input": {"content": "x"}}


def test_a_foreign_server_named_memory_gets_no_ctx(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    (home / ".claude.json").write_text(json.dumps({"mcpServers": {"memory": {"command": "npx", "args": ["other"]}}}), encoding="utf-8")
    assert tam_hook.provenance_input(_call(tmp_path), home) is None


def test_a_project_scoped_foreign_memory_server_overrides_the_installed_one(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    work = tmp_path / "work"
    work.mkdir()
    _register(home, project_servers={"memory": {"command": "npx", "args": ["other"]}}, cwd=work)
    assert tam_hook.provenance_input(_call(work), home) is None
    assert tam_hook.provenance_input(_call(tmp_path), home) is not None


def test_a_project_mcp_json_foreign_memory_server_overrides_the_installed_one(tmp_path, tam_home):
    work = tmp_path / "work"
    (work / "deep").mkdir(parents=True)
    (work / ".mcp.json").write_text(json.dumps({"mcpServers": {"memory": {"url": "https://example.test/mcp"}}}), encoding="utf-8")
    assert tam_hook.provenance_input(_call(work / "deep"), tam_home) is None


def test_no_marker_or_unregistered_server_means_no_injection(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    assert tam_hook.provenance_input(_call(tmp_path), home) is None
    (home / ".claude.json").write_text(json.dumps({"mcpServers": {"memory": TAM_ENTRY}}), encoding="utf-8")
    assert tam_hook.provenance_input(_call(tmp_path), home) is None, "registered but never recorded by the installer"


def test_another_plugin_s_memory_server_gets_no_ctx(tmp_path, tam_home):
    assert tam_hook.provenance_input(_call(tmp_path, "mcp__plugin_acme_memory__memory_save"), tam_home) is None


def test_marker_records_the_launch_identity_without_secrets_and_is_cleared_with_the_server(tmp_path):
    home = _register(tmp_path)
    marker = json.loads((home / ".claude" / "tam-provenance.json").read_text(encoding="utf-8"))
    assert marker == {"servers": {"memory": TAM_FINGERPRINT}}
    assert "s3cret" not in json.dumps(marker)
    (home / ".claude.json").write_text("{}", encoding="utf-8")
    from setup_wizard.extras import write_provenance_marker

    write_provenance_marker(home / ".claude")
    assert not (home / ".claude" / "tam-provenance.json").exists()


def test_docker_install_records_the_http_registration(tmp_path):
    sys.path.insert(0, str(ROOT / "src"))
    from setup_wizard.docker_install import install

    install(tmp_path, tmp_path / "launchers", "http://127.0.0.1:3737/mcp", "tok")
    marker = json.loads((tmp_path / ".claude" / "tam-provenance.json").read_text(encoding="utf-8"))
    assert marker == {"servers": {"memory": {"url": "http://127.0.0.1:3737/mcp"}}}
    assert "tok" not in json.dumps(marker)
    payload = _call(tmp_path)
    assert tam_hook.provenance_input(payload, tmp_path)["_ctx"]["agent"] == "claude-code"


def test_a_user_scope_memory_server_the_installer_did_not_register_is_never_recorded(tmp_path):
    from setup_wizard.extras import write_provenance_marker

    home = _register(tmp_path)
    marker = home / ".claude" / "tam-provenance.json"
    assert marker.exists()
    foreign = {"command": "npx", "args": ["other-memory"]}
    (home / ".claude.json").write_text(json.dumps({"mcpServers": {"memory": foreign}}), encoding="utf-8")
    write_provenance_marker(home / ".claude")
    assert not marker.exists(), "a refresh must not adopt a foreign server"
    write_provenance_marker(home / ".claude")
    assert not marker.exists()
    assert tam_hook.provenance_input(_call(tmp_path), home) is None


def test_a_hook_refresh_keeps_the_marker_of_an_http_install(tmp_path):
    from setup_wizard.extras import write_provenance_marker

    (tmp_path / ".claude").mkdir()
    (tmp_path / ".claude.json").write_text(json.dumps({"mcpServers": {"memory": {"type": "http", "url": "http://h/mcp"}}}), encoding="utf-8")
    marker = write_provenance_marker(tmp_path / ".claude", "http://h/mcp")
    before = marker.read_text(encoding="utf-8")
    write_provenance_marker(tmp_path / ".claude")
    assert marker.read_text(encoding="utf-8") == before
    (tmp_path / ".claude.json").write_text(json.dumps({"mcpServers": {"memory": {"type": "http", "url": "http://evil/mcp"}}}), encoding="utf-8")
    write_provenance_marker(tmp_path / ".claude")
    assert not marker.exists()


def test_a_parent_directory_local_scope_does_not_outrank_the_project_mcp_json(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    parent = tmp_path / "parent"
    work = parent / "repo"
    work.mkdir(parents=True)
    _register(home, project_servers={"memory": TAM_ENTRY}, cwd=parent)
    (work / ".mcp.json").write_text(json.dumps({"mcpServers": {"memory": {"url": "https://example.test/mcp"}}}), encoding="utf-8")
    assert tam_hook.provenance_input(_call(work), home) is None
    # The exact project path still has local scope.
    _register(home, project_servers={"memory": {"command": "npx", "args": ["other"]}}, cwd=work)
    assert tam_hook.provenance_input(_call(work), home) is None
