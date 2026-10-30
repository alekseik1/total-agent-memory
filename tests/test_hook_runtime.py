"""Regression tests for the Claude Code hook runtime (hooks/*.sh + hooks/lib).

Covers the 14.9 hook fixes:
  * PreToolUse/PostToolUse/PostToolUseFailure text reaches the model only as
    one ``hookSpecificOutput.additionalContext`` JSON object per invocation;
  * Stop never prints model-directed text, writes one recovery file per
    session and shows a user-only systemMessage at most once;
  * SessionEnd parses the nested transcript format (and the legacy flat one);
  * SessionStart restores session_continuity after compact/resume and shows
    only a recovery of another session in the same directory;
  * payload fields are parsed once per hook (no python spawn per field);
  * memory dir resolution is shared by every hook.

Each hook runs as a subprocess with a crafted payload, a temp HOME and a temp
memory dir. Hooks run under every bash available (macOS /bin/bash is 3.2).
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import shlex
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parent.parent
HOOKS = ROOT / "hooks"
HELPER = HOOKS / "lib" / "tam_hook.py"

BASHES = sorted({b for b in (shutil.which("bash"), "/bin/bash") if b and Path(b).exists()})
pytestmark = pytest.mark.skipif(not BASHES, reason="bash not available")


def load_helper() -> ModuleType:
    spec = importlib.util.spec_from_file_location("tam_hook_under_test", HELPER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tam_hook = load_helper()


@pytest.fixture(params=BASHES, ids=lambda b: "bash-" + b.replace("/", "_"))
def bash(request) -> str:
    return request.param


@pytest.fixture
def world(tmp_path: Path) -> dict[str, Path]:
    home = tmp_path / "home"
    mem = tmp_path / "mem"
    install = tmp_path / "install-without-src"
    home.mkdir()
    mem.mkdir()
    install.mkdir()
    return {"home": home, "mem": mem, "install": install, "tmp": tmp_path}


def hook_env(world: dict[str, Path], install_dir: Path | None = None, **extra: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("TAM_", "CLAUDE_MEMORY", "MEMORY_NUDGE", "MEMORY_PROJECT", "HOOK_"))}
    env.update({
        "HOME": str(world["home"]),
        "CLAUDE_MEMORY_DIR": str(world["mem"]),
        "CLAUDE_MEMORY_INSTALL_DIR": str(install_dir or world["install"]),
        "HOOK_PYTHON": sys.executable,
        "TAM_HOOK_NOTIFY": "0",
    })
    env.update(extra)
    return env


def run_hook(bash: str, name: str, payload: dict | str, env: dict[str, str], hooks_dir: Path = HOOKS
             ) -> subprocess.CompletedProcess:
    data = payload if isinstance(payload, str) else json.dumps(payload)
    return subprocess.run([bash, str(hooks_dir / name)], input=data.encode(), env=env,
                          capture_output=True, timeout=30, check=False)


def single_json(proc: subprocess.CompletedProcess) -> dict:
    """The hook printed exactly one JSON object (and nothing else) on stdout."""
    assert proc.returncode == 0, proc.stderr.decode()
    lines = [line for line in proc.stdout.decode().splitlines() if line.strip()]
    assert len(lines) == 1, f"expected one JSON line, got: {proc.stdout.decode()!r}"
    return json.loads(lines[0])


def context_of(proc: subprocess.CompletedProcess, event: str) -> str:
    out = single_json(proc)["hookSpecificOutput"]
    assert out["hookEventName"] == event
    assert set(out) == {"hookEventName", "additionalContext"}, "no permission/blocking decision"
    return out["additionalContext"]


# ──────────────────────────────────────────────
# PreToolUse / PostToolUse JSON envelopes
# ──────────────────────────────────────────────

class TestToolEventJson:
    def test_pre_edit_emits_additional_context_json_with_escaping(self, bash, world):
        path = '/work/proj/we"ird $(touch PWNED) `x`\\n.py'
        proc = run_hook(bash, "pre-edit.sh", {
            "hook_event_name": "PreToolUse", "tool_name": "Edit", "cwd": str(world["tmp"]),
            "tool_input": {"file_path": path}}, hook_env(world))
        text = context_of(proc, "PreToolUse")
        assert f'file_context(path="{path}")' in text
        assert "<system-reminder>" not in text
        assert not (world["tmp"] / "PWNED").exists()

    def test_pre_edit_defaults_event_name_when_payload_lacks_it(self, bash, world):
        proc = run_hook(bash, "pre-edit.sh", {"tool_name": "Write", "tool_input": {"file_path": "/w/a.py"}},
                        hook_env(world))
        assert "file_context" in context_of(proc, "PreToolUse")

    @pytest.mark.parametrize("payload", [
        {"tool_name": "NotebookEdit", "tool_input": {"file_path": "/w/a.ipynb"}},
        {"tool_name": "Edit", "tool_input": {"file_path": "/w/node_modules/x.js"}},
        {"tool_name": "Edit", "tool_input": {}},
    ])
    def test_pre_edit_is_silent_when_skipped(self, bash, world, payload):
        proc = run_hook(bash, "pre-edit.sh", payload, hook_env(world))
        assert proc.returncode == 0 and proc.stdout == b""

    def test_on_bash_error_fires_on_post_tool_use_failure(self, bash, world):
        proc = run_hook(bash, "on-bash-error.sh", {
            "hook_event_name": "PostToolUseFailure", "tool_name": "Bash",
            "tool_input": {"command": "ls /nope"},
            "error": "Exit code 2\nls: /nope: No such file or directory"}, hook_env(world))
        text = context_of(proc, "PostToolUseFailure")
        assert "bash exited 2" in text and "learn_error(" in text

    def test_on_bash_error_reads_tool_response_exit_code(self, bash, world):
        proc = run_hook(bash, "on-bash-error.sh", {
            "hook_event_name": "PostToolUse", "tool_name": "Bash",
            "tool_response": {"exit_code": 1, "stderr": "boom"}}, hook_env(world))
        assert "bash exited 1" in context_of(proc, "PostToolUse")

    @pytest.mark.parametrize("payload", [
        {"hook_event_name": "PostToolUse", "tool_name": "Bash", "tool_response": {"stdout": "ok", "stderr": ""}},
        {"hook_event_name": "PostToolUseFailure", "tool_name": "Bash", "is_interrupt": True,
         "error": "Exit code 130\ninterrupted"},
        {"hook_event_name": "PostToolUseFailure", "tool_name": "Bash", "error": "User denied this command"},
    ])
    def test_on_bash_error_silent_for_success_and_user_aborts(self, bash, world, payload):
        proc = run_hook(bash, "on-bash-error.sh", payload, hook_env(world))
        assert proc.returncode == 0 and proc.stdout == b""

    def test_auto_capture_json_for_code_and_silence_for_docs(self, bash, world):
        env = hook_env(world)
        proc = run_hook(bash, "auto-capture.sh", {"hook_event_name": "PostToolUse", "tool_name": "Write",
                                                  "tool_input": {"file_path": "/w/app/main.go"}}, env)
        assert "memory_observe(" in context_of(proc, "PostToolUse")
        proc = run_hook(bash, "auto-capture.sh", {"hook_event_name": "PostToolUse", "tool_name": "Write",
                                                  "tool_input": {"file_path": "/w/README.md"}}, env)
        assert proc.stdout == b""


class TestMemoryTrigger:
    def payload(self, cwd: str, command: str, failed: bool, error: str = "") -> dict:
        if failed:
            return {"hook_event_name": "PostToolUseFailure", "tool_name": "Bash", "cwd": cwd,
                    "session_id": "s1", "tool_input": {"command": command}, "error": error}
        return {"hook_event_name": "PostToolUse", "tool_name": "Bash", "cwd": cwd, "session_id": "s1",
                "tool_input": {"command": command}, "tool_response": {"stdout": "ok", "stderr": ""}}

    def test_failure_then_fix_reach_the_model_and_state_is_not_executable(self, bash, world):
        env = hook_env(world)
        cwd = str(world["tmp"] / "proj")
        marker = world["tmp"] / "PWNED"
        command = f"pytest -x; touch {marker}; echo $(touch {marker})"
        proc = run_hook(bash, "memory-trigger.sh",
                        self.payload(cwd, command, True, "Exit code 1\nconnection refused"), env)
        text = context_of(proc, "PostToolUseFailure")
        assert "SELF_LEARNING_ERROR" in text and "exit=1" in text and "category='api_error'" in text
        state = json.loads((world["mem"] / "state" / "last-error-proj.json").read_text(encoding="utf-8"))
        assert state["cmd"].startswith("pytest -x")

        proc = run_hook(bash, "memory-trigger.sh", self.payload(cwd, "pytest -x", False), env)
        text = context_of(proc, "PostToolUse")
        assert "ERROR_FIXED" in text and "SAVE THE FIX" in text
        assert "SELF_LEARNING_SUCCESS: Tests passed" in text
        assert not (world["mem"] / "state" / "last-error-proj.json").exists()
        assert not marker.exists()
        # src/ is absent in this install dir: the skipped background script is logged, not silent.
        assert "auto_self_improve.py not found" in (world["mem"] / "hooks.log").read_text(encoding="utf-8")

    def test_significant_command_and_quiet_command(self, bash, world):
        env = hook_env(world)
        cwd = str(world["tmp"] / "proj")
        proc = run_hook(bash, "memory-trigger.sh", self.payload(cwd, "git commit -m x", False), env)
        assert "Git commit completed" in context_of(proc, "PostToolUse")
        proc = run_hook(bash, "memory-trigger.sh", self.payload(cwd, "ls -la", False), env)
        assert proc.returncode == 0 and proc.stdout == b""

    def test_repeated_category_raises_pattern_alert(self, bash, world):
        env = hook_env(world)
        cwd = str(world["tmp"] / "proj")
        texts = [context_of(run_hook(bash, "memory-trigger.sh",
                                     self.payload(cwd, f"curl x{i}", True, "Exit code 7\ntimed out"), env),
                            "PostToolUseFailure") for i in range(3)]
        assert "SELF_LEARNING_PATTERN" not in texts[1]
        assert "SELF_LEARNING_PATTERN: 3x errors in category 'timeout'" in texts[2]


class TestNudgeJson:
    def test_nudge_is_additional_context_json(self, bash, world):
        env = hook_env(world)
        outputs = [run_hook(bash, "post-tool-use.sh", {"hook_event_name": "PostToolUse", "tool_name": "Edit",
                                                       "session_id": "nj", "cwd": "/w/proj"}, env)
                   for _ in range(3)]
        assert outputs[0].stdout == b"" and outputs[1].stdout == b""
        assert "MEMORY_NUDGE [soft]" in context_of(outputs[2], "PostToolUse")


SUBAGENT = {"agent_id": "a1", "agent_type": "general-purpose"}
EDIT = {"hook_event_name": "PreToolUse", "tool_name": "Edit", "tool_input": {"file_path": "/w/a.py"}}
WRITE = {"hook_event_name": "PostToolUse", "tool_name": "Write", "tool_input": {"file_path": "/w/app/main.go"}}


class TestQuietHooks:
    @pytest.mark.parametrize("hook,payload", [("pre-edit.sh", EDIT), ("auto-capture.sh", WRITE)])
    def test_hint_hooks_are_silent_in_subagents(self, bash, world, hook, payload):
        proc = run_hook(bash, hook, {**payload, **SUBAGENT}, hook_env(world))
        assert proc.returncode == 0 and proc.stdout == b""

    @pytest.mark.parametrize("hook,payload,text", [
        ("pre-edit.sh", EDIT, "file_context("), ("auto-capture.sh", WRITE, "MEMORY_HINT")])
    def test_override_restores_hint_hooks_in_subagents(self, bash, world, hook, payload, text):
        proc = run_hook(bash, hook, {**payload, **SUBAGENT}, hook_env(world, TAM_HOOKS_IN_SUBAGENTS="1"))
        assert text in context_of(proc, payload["hook_event_name"])

    @pytest.mark.parametrize("value", ["0", "false"])
    @pytest.mark.parametrize("hook,payload,switch", [
        ("pre-edit.sh", EDIT, "TAM_PRE_EDIT_GUARD"), ("auto-capture.sh", WRITE, "TAM_AUTO_CAPTURE_HINT")])
    def test_env_switch_silences_hint_hook(self, bash, world, hook, payload, switch, value):
        proc = run_hook(bash, hook, payload, hook_env(world, **{switch: value}))
        assert proc.returncode == 0 and proc.stdout == b""

    def test_subagent_edits_neither_nudge_nor_count(self, bash, world):
        env = hook_env(world)
        payload = {"hook_event_name": "PostToolUse", "tool_name": "Edit", "session_id": "sa", "cwd": "/w/proj",
                   **SUBAGENT}
        outputs = [run_hook(bash, "post-tool-use.sh", payload, env) for _ in range(3)]
        assert all(proc.returncode == 0 and proc.stdout == b"" for proc in outputs)
        assert not (world["mem"] / "state" / "nudge-sa.json").exists()

    def test_subagent_save_counts_for_parent(self, bash, world):
        env = hook_env(world)
        base = {"hook_event_name": "PostToolUse", "session_id": "sd", "cwd": "/w/proj"}
        saved = run_hook(bash, "post-tool-use.sh", {**base, "tool_name": "mcp__memory__memory_save", **SUBAGENT}, env)
        outputs = [run_hook(bash, "post-tool-use.sh", {**base, "tool_name": "Edit"}, env) for _ in range(3)]
        assert saved.stdout == b"" and all(proc.stdout == b"" for proc in outputs)
        state = json.loads((world["mem"] / "state" / "nudge-sd.json").read_text(encoding="utf-8"))
        assert state["memory_saves"] == 1 and state["edits"] == 3

    def test_override_restores_nudge_in_subagents(self, bash, world):
        env = hook_env(world, TAM_HOOKS_IN_SUBAGENTS="1")
        payload = {"hook_event_name": "PostToolUse", "tool_name": "Edit", "session_id": "sb", "cwd": "/w/proj",
                   **SUBAGENT}
        outputs = [run_hook(bash, "post-tool-use.sh", payload, env) for _ in range(3)]
        assert "MEMORY_NUDGE [soft]" in context_of(outputs[2], "PostToolUse")

    def test_error_learning_still_fires_in_subagents(self, bash, world):
        proc = run_hook(bash, "memory-trigger.sh", {
            "hook_event_name": "PostToolUseFailure", "tool_name": "Bash", "cwd": str(world["tmp"] / "proj"),
            "session_id": "s1", "tool_input": {"command": "pytest -x"}, "error": "Exit code 1\nboom",
            **SUBAGENT}, hook_env(world))
        assert "SELF_LEARNING_ERROR" in context_of(proc, "PostToolUseFailure")


# ──────────────────────────────────────────────
# Stop
# ──────────────────────────────────────────────

class TestStop:
    def stop(self, bash, world, session="stop-1", **extra):
        payload = {"hook_event_name": "Stop", "session_id": session, "cwd": str(world["tmp"]),
                   "stop_hook_active": False, **extra}
        return run_hook(bash, "on-stop.sh", payload, hook_env(world))

    def test_stop_is_silent_and_keeps_one_file_per_session(self, bash, world):
        for turn in range(4):
            proc = self.stop(bash, world)
            assert proc.returncode == 0
            if turn == 0:
                # The fixture has no src/: the user hears once that ambient memory is off.
                assert tam_hook.AMBIENT_UNAVAILABLE_NOTICE in json.loads(proc.stdout)["systemMessage"]
            else:
                assert proc.stdout == b""
        files = list((world["mem"] / "recovery").glob("pending-*.md"))
        assert [f.name for f in files] == ["pending-stop-1.md"]
        meta = tam_hook.recovery_meta(files[0])
        assert meta["session_id"] == "stop-1" and meta["cwd"] == str(world["tmp"])

    def test_snapshot_is_rate_limited(self, bash, world):
        self.stop(bash, world)
        path = world["mem"] / "recovery" / "pending-stop-1.md"
        old = time.time() - 30
        os.utime(path, (old, old))
        self.stop(bash, world)
        assert abs(path.stat().st_mtime - old) < 1, "rewritten inside STOP_SNAPSHOT_MIN_INTERVAL_SEC"
        stale = time.time() - tam_hook.STOP_SNAPSHOT_MIN_INTERVAL_SEC - 5
        os.utime(path, (stale, stale))
        self.stop(bash, world)
        assert path.stat().st_mtime > stale + 1

    def test_stop_hook_active_does_nothing(self, bash, world):
        proc = self.stop(bash, world, stop_hook_active=True)
        assert proc.stdout == b"" and not (world["mem"] / "recovery").exists()

    def test_user_notice_is_system_message_once_per_session(self, bash, world):
        env = hook_env(world)
        for _ in range(3):
            run_hook(bash, "post-tool-use.sh", {"hook_event_name": "PostToolUse", "tool_name": "Edit",
                                                "session_id": "stop-1", "cwd": "/w/proj"}, env)
        first = single_json(self.stop(bash, world))
        assert set(first) == {"systemMessage"} and "no memory_save" in first["systemMessage"]
        assert "decision" not in first
        assert self.stop(bash, world).stdout == b""


# ──────────────────────────────────────────────
# SessionEnd transcript parsing
# ──────────────────────────────────────────────

NESTED_TRANSCRIPT = [
    {"type": "user", "message": {"role": "user", "content": "<command-name>/model</command-name>"}},
    {"type": "user", "message": {"role": "user", "content": "please fix the flaky auth test"}},
    {"type": "assistant", "message": {"role": "assistant", "content": [
        {"type": "text", "text": "Root cause: shared fixture state."},
        {"type": "tool_use", "name": "Edit", "input": {}}]}},
    {"type": "user", "message": {"role": "user", "content": [
        {"type": "tool_result", "content": "ok", "tool_use_id": "t1"}]}},
    {"type": "user", "message": {"role": "user", "content": [{"type": "text", "text": "now add a regression test"}]}},
]
FLAT_TRANSCRIPT = [
    {"type": "human", "role": "user", "content": "legacy flat request"},
    {"role": "assistant", "content": "legacy flat answer"},
]


def _cli_opts(args: list[str]) -> dict[str, str]:
    """`--key value` and `--key=value` argv pairs as a dict."""
    opts, rest = {}, iter(args)
    for arg in rest:
        key, sep, value = arg.partition("=")
        opts[key] = value if sep else next(rest)
    return opts


class TestSessionEnd:
    def write_transcript(self, world, entries) -> Path:
        path = world["tmp"] / "abc-123.jsonl"
        path.write_text("\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8")
        return path

    def test_nested_transcript_lands_in_recovery_file(self, bash, world):
        transcript = self.write_transcript(world, NESTED_TRANSCRIPT)
        proc = run_hook(bash, "session-end.sh", {
            "hook_event_name": "SessionEnd", "session_id": "end-1", "cwd": str(world["tmp"]),
            "reason": "prompt_input_exit", "transcript_path": str(transcript)}, hook_env(world, ROOT))
        assert proc.returncode == 0 and proc.stdout == b""
        body = (world["mem"] / "recovery" / "pending-end-1.md").read_text(encoding="utf-8")
        assert "please fix the flaky auth test" in body and "now add a regression test" in body
        assert "Root cause: shared fixture state." in body
        assert "<command-name>" not in body
        assert "User exited" in body

    def test_session_end_overwrites_the_stop_snapshot(self, bash, world):
        env = hook_env(world, ROOT)
        run_hook(bash, "on-stop.sh", {"session_id": "end-2", "cwd": str(world["tmp"])}, env)
        transcript = self.write_transcript(world, NESTED_TRANSCRIPT)
        run_hook(bash, "session-end.sh", {"session_id": "end-2", "cwd": str(world["tmp"]), "reason": "logout",
                                          "transcript_path": str(transcript)}, env)
        files = list((world["mem"] / "recovery").glob("pending-*.md"))
        assert [f.name for f in files] == ["pending-end-2.md"]
        assert "please fix the flaky auth test" in files[0].read_text(encoding="utf-8")

    @pytest.mark.parametrize("entries,user,assistant", [
        (NESTED_TRANSCRIPT, "now add a regression test", "Root cause"),
        (FLAT_TRANSCRIPT, "legacy flat request", "legacy flat answer"),
    ])
    def test_session_end_starts_auto_save_with_parsed_context(self, world, entries, user, assistant):
        transcript = self.write_transcript(world, entries)
        calls: list[tuple[str, list[str]]] = []
        actions = tam_hook.session_end(
            {"session_id": "s", "cwd": str(world["tmp"]), "transcript_path": str(transcript)},
            ROOT / "src", world["mem"] / "recovery", "proj", "User exited",
            runner=lambda _src, script, args, stdin_text=None: calls.append((script, list(args), stdin_text)) or True)
        scripts = [c[0] for c in calls]
        assert scripts == ["auto_session_save.py", "auto_episode_capture.py", "auto_session_end.py"]
        save_args = calls[0][1]
        assert "--user-context" not in save_args, "prompts must not travel in argv"
        context = json.loads(calls[0][2])
        assert user in context["user"]
        assert assistant in context["assistant"]
        episode_args = calls[1][1]
        assert episode_args[episode_args.index("--transcript") + 1] == str(transcript)
        assert any("auto_session_save started" in a for a in actions)

    @pytest.mark.skipif(shutil.which("git") is None, reason="git not available")
    def test_session_end_starts_auto_session_end_with_its_args(self, world):
        """auto_session_end.py writes the session_summaries row session_init
        resumes from; only the hook knows the Claude Code session id."""
        repo = world["tmp"] / "repo"
        repo.mkdir()
        subprocess.run(["git", "init", "-q", "-b", "feature/x"], cwd=repo, check=True)
        transcript = self.write_transcript(world, NESTED_TRANSCRIPT)
        calls: dict[str, tuple[list[str], str | None]] = {}
        actions = tam_hook.session_end(
            {"session_id": "s", "cwd": str(repo), "transcript_path": str(transcript)},
            ROOT / "src", world["mem"] / "recovery", "proj", "User exited",
            runner=lambda _src, script, args, stdin_text=None: calls.update({script: (list(args), stdin_text)}) or True)

        end_args, end_stdin = calls["auto_session_end.py"]
        assert _cli_opts(end_args) == {"--session-id": "s", "--project": "proj", "--cwd": str(repo),
                                       "--branch": "feature/x", "--reason": "User exited"}
        assert not any("now add a regression test" in a or "Root cause" in a for a in end_args), \
            "prompts must not travel in argv"
        context = json.loads(end_stdin)
        assert "now add a regression test" in context["user"]
        assert "Root cause" in context["assistant"]
        assert any("auto_session_end started for s" in a for a in actions)

    def test_auto_session_end_takes_the_payload_session_not_a_fallback_transcript(self, world, monkeypatch):
        """Without transcript_path the transcript is the newest one in the cwd's
        project dir - possibly another session's; the id must come from the payload."""
        monkeypatch.setenv("HOME", str(world["home"]))
        cwd = world["tmp"] / "work"
        cwd.mkdir()
        project_dir = world["home"] / ".claude" / "projects" / re.sub(r"[^A-Za-z0-9]", "-", str(cwd))
        project_dir.mkdir(parents=True)
        (project_dir / "other-session.jsonl").write_text(
            "\n".join(json.dumps(e) for e in NESTED_TRANSCRIPT) + "\n", encoding="utf-8")
        calls: list[tuple[str, list[str]]] = []
        tam_hook.session_end(
            {"session_id": "live-session", "cwd": str(cwd)},
            ROOT / "src", world["mem"] / "recovery", "proj", "User exited",
            runner=lambda _src, script, args, stdin_text=None: calls.append((script, list(args))) or True)

        assert _cli_opts(dict(calls)["auto_session_end.py"])["--session-id"] == "live-session"


# ──────────────────────────────────────────────
# SessionStart: restore + recovery
# ──────────────────────────────────────────────

def seed_db(mem: Path, project: str, summary: str = "Refactored hooks to JSON output") -> None:
    db = sqlite3.connect(mem / "memory.db")
    db.executescript((ROOT / "migrations" / "010_session_continuity.sql").read_text(encoding="utf-8"))
    db.execute("CREATE TABLE IF NOT EXISTS rules (id INTEGER PRIMARY KEY, content TEXT, priority INTEGER, "
               "status TEXT, scope TEXT, project TEXT, success_rate REAL DEFAULT 0)")
    db.execute("INSERT INTO session_summaries (id, session_id, project, summary, next_steps, pitfalls, ended_at, "
               "consumed) VALUES ('a', 'old', ?, 'older summary', '[]', '[]', '2026-01-01T00:00:00Z', 1)",
               (project,))
    db.execute("INSERT INTO session_summaries (id, session_id, project, summary, next_steps, pitfalls, ended_at, "
               "consumed) VALUES ('b', 'prev', ?, ?, ?, ?, '2026-10-01T00:00:00Z', 1)",
               (project, summary, json.dumps(["wire PostToolUseFailure"]), json.dumps(["bash 3.2 lacks assoc arrays"])))
    db.execute("INSERT INTO rules (content, priority, status, scope, project) VALUES "
               "('Always run the full suite', 9, 'active', 'global', 'general'), "
               "('Inactive rule', 10, 'archived', 'global', 'general')")
    db.commit()
    db.close()


class TestSessionStart:
    def start(self, bash, world, source, session="new-1", cwd=None):
        payload = {"hook_event_name": "SessionStart", "session_id": session, "source": source,
                   "cwd": str(cwd or world["tmp"] / "proj")}
        return run_hook(bash, "session-start.sh", payload, hook_env(world))

    @pytest.mark.parametrize("source", ["compact", "resume"])
    def test_compact_and_resume_inject_latest_continuity(self, bash, world, source):
        seed_db(world["mem"], "proj")
        proc = self.start(bash, world, source)
        out = proc.stdout.decode()
        assert proc.returncode == 0
        assert f"SESSION_RESTORE ({source})" in out
        assert "Refactored hooks to JSON output" in out and "older summary" not in out
        assert "wire PostToolUseFailure" in out and "bash 3.2 lacks assoc arrays" in out
        assert "[P9] Always run the full suite" in out and "Inactive rule" not in out

    def test_restore_is_bounded(self, bash, world):
        seed_db(world["mem"], "proj", summary="x" * 20_000)
        out = self.start(bash, world, "compact").stdout.decode()
        block = out[out.index("SESSION_RESTORE"):]
        assert len(block.split("\n\n")[0]) <= tam_hook.RESTORE_MAX_CHARS

    def test_startup_keeps_rules_summary_without_restore(self, bash, world):
        seed_db(world["mem"], "proj")
        out = self.start(bash, world, "startup").stdout.decode()
        assert "SESSION_RESTORE" not in out
        assert "SOUL_RULES: 1 active behavioral rule(s) loaded." in out
        assert "MEMORY_HINT: Project: proj" in out

    def test_compact_without_summary_points_to_session_init(self, bash, world):
        out = self.start(bash, world, "compact").stdout.decode()
        assert 'session_init(project="proj")' in out

    def write_recovery(self, world, session, cwd, age_hours=0.0):
        path = tam_hook.write_recovery(world["mem"] / "recovery", session, Path(cwd).name, str(cwd),
                                       "test", user_context=f"work of {session}")
        when = time.time() - age_hours * 3600
        os.utime(path, (when, when))
        return path

    def test_recovery_shown_once_for_same_directory_only(self, bash, world):
        proj, other = world["tmp"] / "proj", world["tmp"] / "other"
        recovery = world["mem"] / "recovery"
        self.write_recovery(world, "old-same", proj, age_hours=2)
        self.write_recovery(world, "newer-same", proj, age_hours=1)
        other_file = self.write_recovery(world, "other-dir", other, age_hours=1)
        own = self.write_recovery(world, "new-1", proj)
        stale = self.write_recovery(world, "stale", proj, age_hours=tam_hook.RECOVERY_MAX_AGE_HOURS + 1)
        legacy = recovery / "pending-20260101-000000.md"
        legacy.write_text("# Session Recovery\nper-turn snapshot", encoding="utf-8")

        out = self.start(bash, world, "startup").stdout.decode()
        assert "work of newer-same" in out
        assert "work of old-same" not in out and "work of other-dir" not in out and "work of new-1" not in out
        assert "rm " not in out
        remaining = sorted(p.name for p in recovery.glob("pending-*.md"))
        assert remaining == sorted([other_file.name, own.name])
        assert not stale.exists() and not legacy.exists()

        second = self.start(bash, world, "startup").stdout.decode()
        assert "RECOVERY" not in second

    def test_clear_and_compact_do_not_inject_recovery(self, bash, world):
        proj = world["tmp"] / "proj"
        path = self.write_recovery(world, "prev", proj)
        for source in ("clear", "compact"):
            assert "work of prev" not in self.start(bash, world, source).stdout.decode()
        assert path.exists()


# ──────────────────────────────────────────────
# common.sh: field parsing, spawn count, memory dir resolution
# ──────────────────────────────────────────────

class TestCommonLib:
    def run_snippet(self, bash, world, snippet: str, payload: dict | str, **extra: str):
        script = world["tmp"] / "probe.sh"
        script.write_text(f'source "{HOOKS}/lib/common.sh"\n{snippet}\n', encoding="utf-8")
        data = payload if isinstance(payload, str) else json.dumps(payload)
        return subprocess.run([bash, str(script)], input=data.encode(), env=hook_env(world, **extra),
                              capture_output=True, timeout=30, check=False)

    def test_hook_get_returns_values_verbatim_without_evaluating_them(self, bash, world):
        marker = world["tmp"] / "PWNED"
        tricky = f"a 'b' \"c\" $(touch {marker}) `touch {marker}` \\ ; line1\nline2 ünï"
        proc = self.run_snippet(bash, world, 'hook_get ".tool_input.command"; hook_get "tool_response.exit_code";'
                                'hook_get "missing.key"; echo "[$HOOK_EVENT]"',
                                {"hook_event_name": "PostToolUse", "tool_input": {"command": tricky},
                                 "tool_response": {"exit_code": 3}})
        assert proc.returncode == 0, proc.stderr.decode()
        assert proc.stdout.decode() == f"{tricky}\n3\n[PostToolUse]\n"
        assert not marker.exists()

    def test_fields_are_parsed_with_one_python_spawn(self, bash, world):
        counter = world["tmp"] / "spawns"
        wrapper = world["tmp"] / "counting-python"
        wrapper.write_text(f'#!/bin/sh\necho x >> "{counter}"\nexec "{sys.executable}" "$@"\n', encoding="utf-8")
        wrapper.chmod(0o755)
        proc = self.run_snippet(bash, world, "for f in cwd session_id source model tool_name transcript_path "
                                "reason tool_input.command; do hook_get $f >/dev/null; done",
                                {"cwd": "/w/p", "session_id": "s", "source": "startup", "model": "opus"},
                                HOOK_PYTHON=str(wrapper))
        assert proc.returncode == 0
        assert counter.read_text(encoding="utf-8").count("x") == 1

    def test_invalid_payload_does_not_break_the_hook(self, bash, world):
        proc = self.run_snippet(bash, world, 'echo "[$(hook_get cwd)]"', "not json")
        assert proc.returncode == 0 and proc.stdout.decode() == "[]\n"

    def test_context_buffer_merges_messages_into_one_object(self, bash, world):
        proc = self.run_snippet(bash, world, 'hook_add_context "one"; hook_add_context "two \\"quoted\\""',
                                {"hook_event_name": "PostToolUse"})
        assert context_of(proc, "PostToolUse") == 'one\n\ntwo "quoted"'

    def test_events_without_context_support_are_refused(self, bash, world):
        proc = self.run_snippet(bash, world, 'hook_add_context "never shown"', {"hook_event_name": "Stop"})
        assert proc.stdout == b"" and b"cannot carry additionalContext" in proc.stderr

    def test_memory_dir_resolution_matches_for_every_hook(self, bash, world):
        env = hook_env(world)
        env.pop("CLAUDE_MEMORY_DIR")
        env["TAM_MEMORY_DIR"] = str(world["tmp"] / "tam-only")
        run_hook(bash, "post-tool-use.sh", {"tool_name": "Edit", "session_id": "md", "cwd": "/w/p"}, env)
        assert (world["tmp"] / "tam-only" / "state" / "nudge-md.json").is_file()

        env.pop("TAM_MEMORY_DIR")
        run_hook(bash, "post-tool-use.sh", {"tool_name": "Edit", "session_id": "md2", "cwd": "/w/p"}, env)
        assert (world["home"] / ".tam" / "state" / "nudge-md2.json").is_file()
        assert not (world["home"] / ".claude-memory").exists()

    def test_installed_copy_uses_generated_install_env(self, bash, world):
        from setup_wizard import clients, extras

        host = clients.Host(world["home"], "Darwin", {}, lambda _b: None)
        plan = extras.plan_hooks(ROOT, host, memory_dir=world["tmp"] / "installed-mem")
        extras.apply_hooks(plan)
        hooks_dir = world["home"] / ".claude" / "hooks"
        assert (hooks_dir / "lib" / "tam_hook.py").is_file()
        env = hook_env(world)
        for key in ("CLAUDE_MEMORY_DIR", "CLAUDE_MEMORY_INSTALL_DIR"):
            env.pop(key)
        transcript = world["tmp"] / "t.jsonl"
        transcript.write_text("\n".join(json.dumps(e) for e in NESTED_TRANSCRIPT) + "\n", encoding="utf-8")
        proc = run_hook(bash, "session-end.sh", {"session_id": "inst", "cwd": str(world["tmp"]),
                                                 "transcript_path": str(transcript), "reason": "logout"},
                        env, hooks_dir=hooks_dir)
        assert proc.returncode == 0, proc.stderr.decode()
        recovery = world["tmp"] / "installed-mem" / "recovery" / "pending-inst.md"
        assert "please fix the flaky auth test" in recovery.read_text(encoding="utf-8")


# ──────────────────────────────────────────────
# Registration
# ──────────────────────────────────────────────

def test_plugin_session_start_matcher_includes_compact():
    cfg = json.loads((HOOKS / "hooks.json").read_text(encoding="utf-8"))
    matchers = [block.get("matcher", "") for block in cfg["hooks"]["SessionStart"]]
    assert any("compact" in m.split("|") for m in matchers)


def test_installer_registers_failure_event_and_ships_runtime(tmp_path):
    from setup_wizard import clients, extras

    host = clients.Host(tmp_path, "Darwin", {}, lambda _b: None)
    plan = extras.plan_hooks(ROOT, host, memory_dir=tmp_path / "m")
    settings = json.loads(plan.text)["hooks"]
    failure = [h["command"] for b in settings["PostToolUseFailure"] for h in b["hooks"]]
    scripts = [shlex.split(c) for c in failure]
    assert all(len(words) == 1 for words in scripts), "each command is one shell word"
    assert any(words[0].endswith("memory-trigger.sh") for words in scripts)
    assert any(words[0].endswith("on-bash-error.sh") for words in scripts)
    assert all(b.get("matcher", "") == "" for b in settings["SessionStart"]), "empty matcher covers compact"
    runtime = {dest.name for _src, dest in plan.runtime}
    assert {"common.sh", "memory-nudge.sh", "tam_hook.py"} <= runtime
    (env_path, env_text), = plan.generated
    assert env_path.name == "install-env.sh"
    escaped_root = str(ROOT).replace("\\", "\\\\")
    assert f'"${{CLAUDE_MEMORY_INSTALL_DIR:={escaped_root}}}"' in env_text


def test_windows_install_env_and_runtime(tmp_path):
    from setup_wizard import clients, extras

    host = clients.Host(tmp_path, "Windows", {"APPDATA": str(tmp_path)}, lambda _b: None)
    plan = extras.plan_hooks(ROOT, host, memory_dir=PureWindowsPath("C:/Users/o'brien/.tam"))
    runtime = {dest.name for _src, dest in plan.runtime}
    assert {"common.ps1", "tam_hook.py"} <= runtime and "common.sh" not in runtime
    (env_path, env_text), = plan.generated
    assert env_path.name == "install-env.ps1"
    assert "$env:TAM_MEMORY_DIR = 'C:\\Users\\o''brien\\.tam'" in env_text


def test_install_env_quotes_shell_metacharacters():
    from setup_wizard import clients, extras

    host = clients.Host(Path("/h"), "Linux", {}, lambda _b: None)
    text = extras.install_env_text(host, PurePosixPath('/opt/a"b$c`d'), None)
    assert ': "${CLAUDE_MEMORY_INSTALL_DIR:=/opt/a\\"b\\$c\\`d}"' in text
    assert "TAM_MEMORY_DIR" not in text


# ──────────────────────────────────────────────
# PowerShell parity (static: pwsh is not required to run the suite)
# ──────────────────────────────────────────────

PS1_HOOKS = ["auto-capture", "memory-trigger", "on-bash-error", "on-stop", "post-tool-use", "pre-edit",
             "session-end", "session-start", "user-prompt-submit"]


@pytest.mark.parametrize("name", PS1_HOOKS)
def test_ps1_hooks_share_the_common_runtime(name):
    text = (HOOKS / f"{name}.ps1").read_text(encoding="utf-8")
    assert '. (Join-Path $PSScriptRoot "lib\\common.ps1")' in text
    assert ".claude-memory" not in text and "USERPROFILE" not in text
    assert "<system-reminder>" not in text
    assert "TODO" not in text


@pytest.mark.parametrize("name,command", [
    ("memory-trigger", "memory-trigger"), ("on-stop", "stop"), ("post-tool-use", "nudge"),
    ("session-end", "session-end"), ("session-start", "session-start-context"),
    ("user-prompt-submit", "save-intent"),
])
def test_ps1_and_sh_call_the_same_helper_command(name, command):
    assert command in (HOOKS / f"{name}.ps1").read_text(encoding="utf-8")
    assert command in (HOOKS / f"{name}.sh").read_text(encoding="utf-8")


@pytest.mark.skipif(shutil.which("pwsh") is None, reason="pwsh not installed")
def test_ps1_pre_edit_emits_json(tmp_path):
    env = {**os.environ, "CLAUDE_MEMORY_DIR": str(tmp_path), "HOOK_PYTHON": sys.executable}
    proc = subprocess.run(["pwsh", "-NoProfile", "-File", str(HOOKS / "pre-edit.ps1")],
                          input=json.dumps({"hook_event_name": "PreToolUse", "tool_name": "Edit",
                                            "tool_input": {"file_path": "/w/a.py"}}).encode(),
                          env=env, capture_output=True, timeout=60, check=False)
    assert "file_context" in context_of(proc, "PreToolUse")
