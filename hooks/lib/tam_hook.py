#!/usr/bin/env python3
"""Shared runtime for the TAM Claude Code hooks (bash and PowerShell twins).

Stdlib only. The ``session-end`` command additionally imports the transcript
parser from the install's ``src/`` (passed with ``--src``) so transcript
parsing lives in exactly one place (``auto_episode_capture``).

Commands (all read the hook payload JSON from stdin unless noted):

  fields                     shell assignments ``HOOK_F_<path>=...`` plus the
                             normalized ``HOOK_EVENT``/``HOOK_EXIT_CODE``/
                             ``HOOK_ERROR_TEXT``/``HOOK_INTERRUPTED``/
                             ``HOOK_QUIET_SUBAGENT``; meant for ``eval`` in
                             bash (values are shlex-quoted)
  emit EVENT                 stdin is plain text; prints one
                             ``hookSpecificOutput.additionalContext`` object
  system-message             stdin is plain text; prints ``{"systemMessage"}``
  nudge                      PostToolUse write/save counters + nudge text
  nudge-summary              end-of-session counters line (no stdin)
  memory-trigger             Bash error/fix/significant-command reminders
  session-start-context      SessionStart text (restore / rules / recovery)
  stop                       Stop: per-session recovery snapshot + notice
  session-end                SessionEnd: recovery file + auto-save scripts
  save-intent                detached worker: cached payload -> intents
  capture-observation        detached worker: cached payload -> extract queue
  record-action              PostToolUse(Failure): one agent_actions row (src/ambient)
  prompt-context             UserPromptSubmit: applicable rules + background recall
  rule-reminder              PreToolUse: rules matching the command / edit
  codex-notify               Codex notify payload -> action log + turn summary
  inject-provenance          PreToolUse(mcp__memory__*): tool input + ``_ctx`` (agent, session, workspace)
                             as hookSpecificOutput.updatedInput; the server strips and records it

The ambient commands (and ``stop --src``) run ``src/ambient/hooks.py``: in-process,
or with TAM_DOCKER=1 inside the TAM container (``docker exec -i ... python -m
ambient``), whose volume holds the store. When it cannot run (no checkout, a
python without TAM's packages, no container) they log and do nothing; any
failure inside them is logged, never raised into the hook.

Model-visible output must only ever be produced through ``emit`` (or the
``--emit`` flag of producer commands) for PreToolUse/PostToolUse, because
Claude Code ignores plain stdout of those events.
"""
from __future__ import annotations

import argparse
import functools
import json
import logging
import os
import re
import shlex
import sqlite3
import subprocess
import sys
import time
import urllib.request
from collections.abc import Callable, Iterable, Sequence
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PRIVATE_FILE_MODE = 0o600
FIELD_PREFIX = "HOOK_F_"
FIELD_MAX_CHARS = 16_384
FIELD_MAX_DEPTH = 3
FIELD_JSON_MAX_DEPTH = 1
CONTEXT_MAX_CHARS = 9_500
TRUNCATION_MARK = "\n[... truncated]"
CONTEXT_EVENTS = frozenset(
    {"PreToolUse", "PostToolUse", "PostToolUseFailure", "UserPromptSubmit", "SessionStart"}
)

RESTORE_MAX_CHARS = 3_000
RESTORE_SUMMARY_MAX_CHARS = 1_500
RESTORE_ITEM_MAX_CHARS = 240
RESTORE_LIST_LIMIT = 6
RULES_LIMIT = 5
RULE_PREVIEW_CHARS = 160
STARTUP_RULE_PREVIEW_CHARS = 80
PATTERN_MIN_COUNT = 3
PATTERN_WINDOW_DAYS = 30
PATTERN_LIST_LIMIT = 5

RECOVERY_GLOB = "pending-*.md"
RECOVERY_MAX_AGE_HOURS = 72
RECOVERY_KEEP_MAX = 5
RECOVERY_DISPLAY_MAX_CHARS = 3_000
RECOVERY_HEADER_PREFIX = "<!-- tam-recovery "
RECOVERY_HEADER_SUFFIX = " -->"
STOP_SNAPSHOT_MIN_INTERVAL_SEC = 300
# A Stop snapshot refreshed this recently belongs to a session that is still running.
ACTIVE_SNAPSHOT_WINDOW_SEC = 2 * STOP_SNAPSHOT_MIN_INTERVAL_SEC
RECOVERY_KIND_STOP = "stop"
RECOVERY_KIND_END = "end"
GIT_TIMEOUT_SEC = 3
GIT_STATUS_MAX_LINES = 20
GIT_LOG_ENTRIES = 5

SQLITE_TIMEOUT_SEC = 2.0
NUDGE_STATE_MAX_AGE_DAYS = 7
# nudge counters and src/ambient/state.py files share the retention.
SESSION_STATE_GLOBS = ("nudge-*.json", "ambient-*.json", "ambient-unavailable-*", "ambient-tooluse-*.claim")
NUDGE_DEFAULT_SOFT = 3
NUDGE_DEFAULT_HARD = 7
NUDGE_DEFAULT_STEP = 3
FINAL_WARNING_MIN_WRITES = 3
FINAL_NOTE_MIN_WRITES = 5
FINAL_NOTE_RATIO = 5

ERROR_FIX_WINDOW_SEC = 600
ERROR_PATTERN_THRESHOLD = 3
ERROR_CMD_MAX_CHARS = 200
ERROR_MSG_MAX_CHARS = 500
ERROR_MSG_MAX_LINES = 5
FIXED_MSG_PREVIEW_CHARS = 200

TRANSCRIPT_USER_MSGS = 5
TRANSCRIPT_USER_MSG_CHARS = 500
TRANSCRIPT_ASSISTANT_MSGS = 3
TRANSCRIPT_ASSISTANT_CHARS = 2_000
AUTOSAVE_ASSISTANT_CHARS = 1_000
TRANSCRIPT_NOISE_PREFIXES = (
    "<command-name>",
    "<command-message>",
    "<command-args>",
    "<local-command-stdout>",
    "<local-command-stderr>",
    "<local-command-caveat>",
    "<system-reminder>",
    "Caveat: The messages below were generated by the user while running local commands",
)

DOCKER_DEFAULT_CONTAINER = "claude-memory-mcp"
DOCKER_PS_TIMEOUT_SEC = 3

JsonDict = dict[str, Any]

# Used only when src/secret_redaction.py cannot be imported (no checkout next to the hook,
# e.g. Docker installs): the shapes that most often end up on a command line.
REDACTED_MARK = "[REDACTED]"
FALLBACK_SECRET_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(?i)\b([A-Za-z0-9_-]*(?:authorization|token|password|passwd|secret|api[_-]?key)[A-Za-z0-9_-]*)"
                r"(\s*[:=]\s*)(?:(?:basic|bearer)\s+)?[^\s'\"&;]+"), r"\1\2" + REDACTED_MARK),
    (re.compile(r"(?i)\b(bearer|basic)(\s+)[A-Za-z0-9._~+/=-]{8,}"), r"\1\2" + REDACTED_MARK),
    (re.compile(r"\b(?:sk-[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|xox[abp]-[A-Za-z0-9-]{10,}"
                r"|eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{5,})"), REDACTED_MARK),
    (re.compile(r"(?i)(://[^/\s:@]+:)[^@\s/]+(@)"), r"\1" + REDACTED_MARK + r"\2"),
)


# ──────────────────────────────────────────────
# Small utilities
# ──────────────────────────────────────────────

def log(message: str) -> None:
    """Report a problem without ever failing the hook: stderr + optional hooks.log."""
    line = f"[tam-hook] {message}"
    sys.stderr.write(line + "\n")
    log_file = os.environ.get("TAM_HOOK_LOG_FILE")
    if not log_file:
        return
    try:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "a", encoding="utf-8") as fh:
            fh.write(f"[{datetime.now().astimezone():%Y-%m-%d %H:%M:%S}] {line}\n")
    except OSError as exc:
        sys.stderr.write(f"[tam-hook] cannot write {log_file}: {exc}\n")


def truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - len(TRUNCATION_MARK))] + TRUNCATION_MARK


def safe_name(value: str) -> str:
    """Filename-safe form; identical to the historical nudge-state naming."""
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in value) or "unknown"


def parse_payload(raw: str) -> JsonDict:
    if not raw.strip():
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        log(f"hook payload is not valid JSON: {exc}")
        return {}
    if not isinstance(data, dict):
        log("hook payload is not a JSON object")
        return {}
    return data


def read_payload() -> JsonDict:
    return parse_payload(sys.stdin.read())


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    # State and recovery files hold session text: owner-only from the first byte.
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, PRIVATE_FILE_MODE)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(text)
    os.replace(tmp, path)


def load_json_file(path: Path) -> JsonDict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as exc:
        log(f"ignoring unreadable state file {path}: {exc}")
        return None
    return data if isinstance(data, dict) else None


def unlink(path: Path, why: str) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return
    except OSError as exc:
        log(f"cannot remove {path} ({why}): {exc}")


def connect_ro(db_path: Path) -> sqlite3.Connection | None:
    if not db_path.is_file():
        return None
    try:
        return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=SQLITE_TIMEOUT_SEC)
    except sqlite3.Error as exc:
        log(f"cannot open {db_path} read-only: {exc}")
        return None


def table_names(db: sqlite3.Connection) -> set[str]:
    return {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}


# ──────────────────────────────────────────────
# Payload normalization
# ──────────────────────────────────────────────

def field_key(segment: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", segment)


def flatten(payload: JsonDict) -> dict[str, str]:
    """Scalars become strings, nested objects are walked (dicts up to depth 1 also as JSON)."""
    out: dict[str, str] = {}

    def walk(name: str, value: Any, depth: int) -> None:
        if value is None:
            return
        if isinstance(value, dict):
            if name and depth <= FIELD_JSON_MAX_DEPTH:
                out[name] = truncate(json.dumps(value, ensure_ascii=False), FIELD_MAX_CHARS)
            if depth < FIELD_MAX_DEPTH:
                for key, item in value.items():
                    child = field_key(str(key))
                    walk(f"{name}__{child}" if name else child, item, depth + 1)
            return
        if isinstance(value, str):
            text = value
        else:
            text = json.dumps(value, ensure_ascii=False)
        text = text.replace("\x00", "")
        if text != "":
            out[name] = truncate(text, FIELD_MAX_CHARS)

    walk("", payload, 0)
    return out


def _containers(payload: JsonDict) -> Iterable[JsonDict]:
    for key in ("tool_response", "tool_output"):
        value = payload.get(key)
        if isinstance(value, dict):
            yield value


def exit_code(payload: JsonDict) -> str:
    for container in _containers(payload):
        for key in ("exit_code", "exitCode", "returncode"):
            value = container.get(key)
            if value is not None and value != "":
                return str(value)
    if payload.get("hook_event_name") == "PostToolUseFailure":
        match = re.search(r"exit code:?\s*(-?\d+)", str(payload.get("error") or ""), re.IGNORECASE)
        return match.group(1) if match else "1"
    return ""


def error_text(payload: JsonDict) -> str:
    for container in _containers(payload):
        for key in ("stderr", "error"):
            value = container.get(key)
            if isinstance(value, str) and value.strip():
                return value
    value = payload.get("error")
    return value if isinstance(value, str) else ""


def interrupted(payload: JsonDict) -> bool:
    if payload.get("is_interrupt") is True:
        return True
    return any(container.get("interrupted") is True for container in _containers(payload))


def is_failure(code: str) -> bool:
    return code not in ("", "0", "null")


def quiet_in_subagent(payload: JsonDict) -> bool:
    """Only subagent calls carry ``agent_id``; the hint hooks keep out of them unless TAM_HOOKS_IN_SUBAGENTS=1."""
    return bool(payload.get("agent_id")) and os.environ.get("TAM_HOOKS_IN_SUBAGENTS") != "1"


def derived_fields(payload: JsonDict) -> dict[str, str]:
    return {
        "HOOK_EVENT": str(payload.get("hook_event_name") or ""),
        "HOOK_EXIT_CODE": exit_code(payload),
        "HOOK_ERROR_TEXT": truncate(error_text(payload), FIELD_MAX_CHARS),
        "HOOK_INTERRUPTED": "1" if interrupted(payload) else "0",
        "HOOK_QUIET_SUBAGENT": "1" if quiet_in_subagent(payload) else "0",
    }


def project_of(payload: JsonDict) -> str:
    override = os.environ.get("MEMORY_PROJECT")
    if override:
        return override
    return _project_of_cwd(str(payload.get("cwd") or os.getcwd()))


@functools.lru_cache(maxsize=None)
def _project_of_cwd(cwd: str) -> str:
    # A worktree's directory is named after its branch: name the main repository
    # via the common git dir, as the server's _detect_project does (lowercased; a
    # common dir not named .git, as in a bare clone, names itself), so a hook and
    # a tool call from the same session mint the same project key.
    common_dir = _git(cwd, "rev-parse", "--path-format=absolute", "--git-common-dir")
    if common_dir:
        root = os.path.dirname(common_dir) if os.path.basename(common_dir) == ".git" else common_dir
        if os.path.basename(root):
            return os.path.basename(root).lower()
    return (os.path.basename(cwd.rstrip("/\\")) or "unknown").lower()


def session_of(payload: JsonDict) -> str:
    return str(payload.get("session_id") or os.environ.get("CLAUDE_SESSION_ID") or "unknown")


# ──────────────────────────────────────────────
# Output envelopes
# ──────────────────────────────────────────────

def context_json(event: str, text: str) -> str | None:
    """The only shape Claude Code feeds back to the model for tool events."""
    body = text.strip()
    if not body:
        return None
    if event not in CONTEXT_EVENTS:
        raise ValueError(f"event {event!r} cannot carry additionalContext")
    return json.dumps(
        {"hookSpecificOutput": {"hookEventName": event,
                                "additionalContext": truncate(body, CONTEXT_MAX_CHARS)}},
        ensure_ascii=False,
    )


def system_message_json(text: str) -> str | None:
    body = text.strip()
    return json.dumps({"systemMessage": body}, ensure_ascii=False) if body else None


def print_context(event: str, parts: Sequence[str]) -> int:
    try:
        rendered = context_json(event, "\n\n".join(p.strip() for p in parts if p.strip()))
    except ValueError as exc:
        log(str(exc))
        return 1
    if rendered:
        print(rendered)
    return 0


def event_for(payload: JsonDict, default: str) -> str:
    event = str(payload.get("hook_event_name") or default)
    return event if event in CONTEXT_EVENTS else default


# ──────────────────────────────────────────────
# Background script runner (mirrors the old hook_run_script)
# ──────────────────────────────────────────────

def _detached_kwargs() -> JsonDict:
    if os.name == "nt":
        flags = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        return {"creationflags": flags}
    return {"start_new_session": True}


def docker_mode() -> bool:
    return (os.environ.get("CLAUDE_MEMORY_DOCKER") or os.environ.get("TAM_DOCKER") or "0") == "1"


def docker_container() -> str:
    return (os.environ.get("CLAUDE_MEMORY_CONTAINER") or os.environ.get("TAM_CONTAINER")
            or DOCKER_DEFAULT_CONTAINER)


def redact_text(text: str, src_dir: Path) -> str:
    """Credential-free copy of ``text``: the shared redactor from src/, or the built-in fallback patterns."""
    if not text:
        return text
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    try:
        from secret_redaction import redact_secrets
    except ImportError as exc:
        log(f"secret_redaction unavailable from {src_dir}, using fallback patterns only: {exc}")
        cleaned = text
    else:
        cleaned = redact_secrets(text)[0]
    # The built-in patterns also run after the shared redactor: they cover KEY=value shapes it may miss.
    for pattern, replacement in FALLBACK_SECRET_PATTERNS:
        cleaned = pattern.sub(replacement, cleaned)
    return cleaned


def run_script(src_dir: Path, script: str, args: Sequence[str], stdin_text: str | None = None) -> bool:
    """Start ``src/<script>`` detached, output appended to hooks.log. Returns False if it cannot start.

    ``stdin_text`` carries payloads that must not appear in argv (visible to every local user in ``ps``).
    """
    log_file = os.environ.get("TAM_HOOK_LOG_FILE")
    if docker_mode():
        container = docker_container()
        try:
            names = subprocess.run(["docker", "ps", "--format", "{{.Names}}"], capture_output=True,
                                   text=True, encoding="utf-8", errors="replace", timeout=DOCKER_PS_TIMEOUT_SEC, check=False).stdout.split()
        except (OSError, subprocess.TimeoutExpired) as exc:
            log(f"docker unavailable, skipped {script}: {exc}")
            return False
        if container not in names:
            log(f"container {container} is not running, skipped {script}")
            return False
        command = ["docker", "exec", "-i", container, "python", f"src/{script}", *args]
    else:
        path = src_dir / script
        if not path.is_file():
            log(f"{path} not found, skipped (set CLAUDE_MEMORY_INSTALL_DIR to the TAM checkout)")
            return False
        command = [sys.executable, str(path), *args]
    with ExitStack() as stack:
        sink: Any = subprocess.DEVNULL
        if log_file:
            try:
                sink = stack.enter_context(open(log_file, "a", encoding="utf-8"))
            except OSError as exc:
                log(f"cannot open {log_file}, {script} output discarded: {exc}")
        try:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL if stdin_text is None else subprocess.PIPE,
                                     stdout=sink, stderr=sink, **_detached_kwargs())
        except OSError as exc:
            log(f"cannot start {script}: {exc}")
            return False
        if stdin_text is not None and child.stdin is not None:
            try:
                child.stdin.write(stdin_text.encode("utf-8"))
                child.stdin.close()
            except OSError as exc:
                log(f"cannot pass stdin to {script}: {exc}")
                return False
    return True


# ──────────────────────────────────────────────
# Memory-save nudges
# ──────────────────────────────────────────────

def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        log(f"{name}={raw!r} is not an integer, using {default}")
        return default


def nudge_state_path(memory_dir: Path, session: str) -> Path:
    return memory_dir / "state" / f"nudge-{safe_name(session)}.json"


def classify_tool(tool: str) -> str | None:
    lower = tool.lower()
    if "memory_save" in lower or lower.endswith(("save_decision", "save_intent")):
        return "memory_saves"
    if tool in ("Edit", "MultiEdit") or lower.endswith("__edit"):
        return "edits"
    if tool == "Write" or lower.endswith("__write"):
        return "writes"
    if tool == "Bash" or lower.endswith("__bash"):
        return "bashes"
    return None


def nudge(payload: JsonDict, memory_dir: Path) -> str | None:
    """Bump the per-session counters; return a nudge line when thresholds are crossed."""
    tool = str(payload.get("tool_name") or "").strip()
    # A failed call changed nothing: it is logged as an action, not counted as an edit.
    if not tool or payload.get("hook_event_name") == "PostToolUseFailure":
        return None
    session = session_of(payload)
    project = project_of(payload)
    path = nudge_state_path(memory_dir, session)
    state = load_json_file(path) or {
        "session_id": session,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "writes": 0, "edits": 0, "bashes": 0, "memory_saves": 0,
        "last_nudge_writes": 0, "nudge_count": 0,
    }
    field = classify_tool(tool)
    if field:
        state[field] = int(state.get(field, 0)) + 1
        atomic_write(path, json.dumps(state))
    if field not in ("edits", "writes") or os.environ.get("MEMORY_NUDGE_DISABLE") == "1":
        return None

    writes = int(state.get("writes", 0)) + int(state.get("edits", 0))
    saves = int(state.get("memory_saves", 0))
    last = int(state.get("last_nudge_writes", 0))
    soft = _env_int("MEMORY_NUDGE_SOFT", NUDGE_DEFAULT_SOFT)
    hard = _env_int("MEMORY_NUDGE_HARD", NUDGE_DEFAULT_HARD)
    step = _env_int("MEMORY_NUDGE_STEP", NUDGE_DEFAULT_STEP)

    if saves > 0 and (writes - last) < step * 2:
        return None
    if writes < soft:
        return None
    # The hard escalation always fires once; the throttle never hides it.
    escalating = writes >= hard and saves == 0 and last < hard
    if not escalating and writes - last < step:
        return None

    if writes >= hard and saves == 0:
        msg = (f"MEMORY_NUDGE [hard]: {writes} significant edits this session, 0 memory_save calls. "
               f"Save decisions/solutions NOW while context is fresh: memory_save(content=..., "
               f"type='decision'|'solution', project='{project}', tags=['reusable', ...]). "
               f"Skipping saves is the #1 cause of session amnesia.")
    elif saves == 0:
        msg = (f"MEMORY_NUDGE [soft]: {writes} edits without memory_save. When the next decision/fix "
               f"is finalized, call memory_save(project='{project}'). Don't batch for end of session.")
    else:
        msg = (f"MEMORY_NUDGE: {writes} writes vs {saves} saves. "
               f"If a non-trivial new fact landed, memory_save now while it's hot.")
    state["last_nudge_writes"] = writes
    state["nudge_count"] = int(state.get("nudge_count", 0)) + 1
    atomic_write(path, json.dumps(state))
    return msg


def nudge_summary(memory_dir: Path, session: str, project: str) -> tuple[str, str] | None:
    """(level, line) for the session counters; None when nothing relevant happened."""
    state = load_json_file(nudge_state_path(memory_dir, session))
    if state is None:
        return None
    writes = int(state.get("writes", 0)) + int(state.get("edits", 0))
    saves = int(state.get("memory_saves", 0))
    nudges = int(state.get("nudge_count", 0))
    if writes == 0 and saves == 0:
        return None
    if saves == 0 and writes >= FINAL_WARNING_MIN_WRITES:
        return "warning", (f"MEMORY_FINAL_WARNING: session ending with {writes} significant edits and "
                           f"0 memory_save calls (received {nudges} nudges). Before stop, save the most "
                           f"important decision/fix from this session: memory_save(project='{project}').")
    if writes >= FINAL_NOTE_MIN_WRITES and saves < writes // FINAL_NOTE_RATIO:
        return "note", (f"MEMORY_FINAL_NOTE: {writes} edits vs {saves} saves. Coverage ratio low — "
                        f"consider one more memory_save if anything reusable wasn't captured.")
    return "ok", f"MEMORY_FINAL_OK: {writes} edits, {saves} saves recorded."


def stop_notice(memory_dir: Path, session: str) -> str | None:
    """User-visible (systemMessage) reminder, at most once per session."""
    path = nudge_state_path(memory_dir, session)
    state = load_json_file(path)
    if state is None or state.get("stop_notice_shown"):
        return None
    writes = int(state.get("writes", 0)) + int(state.get("edits", 0))
    saves = int(state.get("memory_saves", 0))
    if saves > 0 or writes < FINAL_WARNING_MIN_WRITES:
        return None
    state["stop_notice_shown"] = True
    atomic_write(path, json.dumps(state))
    return (f"TAM memory: {writes} file edits this session and no memory_save yet. "
            f"Ask Claude to save the key decision or fix before you close the session.")


def prune_nudge_state(memory_dir: Path) -> None:
    """Remove per-session nudge and ambient state untouched for NUDGE_STATE_MAX_AGE_DAYS."""
    cutoff = time.time() - NUDGE_STATE_MAX_AGE_DAYS * 86_400
    for pattern in SESSION_STATE_GLOBS:
        for path in (memory_dir / "state").glob(pattern):
            try:
                if path.stat().st_mtime < cutoff:
                    unlink(path, "session state expired")
            except FileNotFoundError:
                continue


# ──────────────────────────────────────────────
# Bash error / fix / significant-command reminders
# ──────────────────────────────────────────────

ERROR_CATEGORIES: tuple[tuple[str, str, str], ...] = (
    (r"timeout|timed out|deadline exceeded", "timeout", "high"),
    (r"connection refused|ECONNREFUSED|network|dns|resolve", "api_error", "high"),
    (r"permission denied|EACCES|forbidden|401|403", "config_error", "medium"),
    (r"not found|No such file|ENOENT|404|unknown command", "wrong_assumption", "low"),
    (r"syntax error|parse error|unexpected token|SyntaxError", "code_error", "medium"),
    (r"out of memory|OOM|memory limit|Cannot allocate", "config_error", "critical"),
)
TEST_CMD = re.compile(r"(go test|phpunit|npm test|npm run test|jest|pytest|vendor/bin/phpunit)")
FIX_CONFIRM_CMD = re.compile(
    r"(go test|phpunit|npm test|npm run test|jest|pytest|go build|npm run build|make build|composer check)")
BUILD_CMD = re.compile(r"(go build|npm run build|composer check|make build)")
FULL_SUITE_CMD = re.compile(r"^(go test|phpunit|npm test).*(\.\.\.|--all|-A)")
LINT_CMD = re.compile(r"(golangci-lint|phpstan|eslint|npm run lint)")
SIGNIFICANT: tuple[tuple[tuple[str, ...], str], ...] = (
    (("docker compose up", "docker-compose up", "docker build"),
     "MEMORY_TRIGGER: Docker operation. If setup was non-trivial, save config details."),
    (("make setup", "make init"), "MEMORY_TRIGGER: Project setup completed. Save infrastructure facts."),
    (("npm install", "npm ci", "yarn add", "pnpm add"),
     "MEMORY_TRIGGER: Package install detected. Save dependency changes."),
    (("pip install", "poetry add", "pipenv install"),
     "MEMORY_TRIGGER: Python package install detected. Save dependency changes."),
    (("go mod", "go get"), "MEMORY_TRIGGER: Go module change detected. Save dependency changes."),
    (("composer require", "composer install"),
     "MEMORY_TRIGGER: Composer operation detected. Save dependency changes."),
)


def categorize(stderr: str) -> tuple[str, str]:
    for pattern, category, severity in ERROR_CATEGORIES:
        if re.search(pattern, stderr, re.IGNORECASE):
            return category, severity
    return "code_error", "medium"


def _significant(command: str, project: str) -> list[str]:
    if "git commit" in command:
        return [("MEMORY_TRIGGER: Git commit completed. Save commit summary:\n"
                 f"  memory_save(type='fact', content='Committed: <scope and changes>', "
                 f"project='{project}', tags=['commit'])")]
    for needles, message in SIGNIFICANT[:1]:
        if any(n in command for n in needles):
            return [message]
    if "migrate" in command:
        return [] if "status" in command else ["MEMORY_TRIGGER: Migration executed. Save schema changes as fact."]
    for needles, message in SIGNIFICANT[1:]:
        if any(n in command for n in needles):
            return [message]
    return []


def memory_trigger(payload: JsonDict, state_dir: Path, src_dir: Path, project: str,
                   runner: Callable[[Path, str, Sequence[str]], bool] = run_script) -> list[str]:
    """Port of the historical memory-trigger.sh decision table; returns model-facing messages."""
    tool_input = payload.get("tool_input") if isinstance(payload.get("tool_input"), dict) else {}
    command = str(tool_input.get("command") or "")
    if not command:
        return []
    code = exit_code(payload)
    stderr = error_text(payload)
    state_dir.mkdir(parents=True, exist_ok=True)
    error_file = state_dir / f"last-error-{safe_name(project)}.json"
    messages: list[str] = []

    if is_failure(code):
        category, severity = categorize(stderr)
        err_short = truncate("\n".join(redact_text(stderr, src_dir).splitlines()[:ERROR_MSG_MAX_LINES]),
                             ERROR_MSG_MAX_CHARS)
        cmd_short = redact_text(command, src_dir)[:ERROR_CMD_MAX_CHARS]
        description = f"Command failed: {cmd_short}. Error: {err_short}"
        messages.append(
            f"SELF_LEARNING_ERROR: Bash command failed (exit={code}). You MUST call self_error_log NOW:\n"
            f"  self_error_log(\n    description={description!r},\n    category='{category}',\n"
            f"    severity='{severity}',\n    context='project: {project}',\n    project='{project}'\n  )")
        runner(src_dir, "auto_self_improve.py",
               ["error", "--description-stdin", "--category", category, "--severity", severity,
                "--project", project], stdin_text=description)
        atomic_write(error_file, json.dumps({"cmd": cmd_short, "msg": err_short, "category": category,
                                             "time": int(time.time())}))
        counts = state_dir / "error-counts"
        counts.mkdir(parents=True, exist_ok=True)
        count_file = counts / safe_name(f"{category}-{project}")
        try:
            count = int(count_file.read_text(encoding="utf-8").strip() or 0) + 1
        except FileNotFoundError:
            count = 1
        except (OSError, ValueError) as exc:
            log(f"resetting unreadable error counter {count_file}: {exc}")
            count = 1
        if count >= ERROR_PATTERN_THRESHOLD:
            messages.append(
                f"SELF_LEARNING_PATTERN: {count}x errors in category '{category}' for project '{project}'!\n"
                f"  Consider: self_insight(action='add', content='...', category='{category}', "
                f"project='{project}')")
            count = 0
            runner(src_dir, "auto_self_improve.py", ["check-patterns", "--project", project])
        count_file.write_text(str(count), encoding="utf-8")
        return messages

    previous = load_json_file(error_file)
    if previous is not None:
        if time.time() - int(previous.get("time", 0)) < ERROR_FIX_WINDOW_SEC:
            # State files written before redaction existed may still hold a credential.
            prev_cmd = redact_text(str(previous.get("cmd", "")), src_dir)
            prev_msg = redact_text(str(previous.get("msg", "")), src_dir)
            same_base = command.split()[:1] == prev_cmd.split()[:1]
            if same_base or FIX_CONFIRM_CMD.search(command):
                messages.append(
                    "ERROR_FIXED: Previous error appears resolved!\n"
                    f"  Error was: {prev_cmd}\n  Error msg: {prev_msg[:FIXED_MSG_PREVIEW_CHARS]}\n\n"
                    "  SAVE THE FIX to memory NOW:\n"
                    f"  memory_save(type='solution', content='PROBLEM: {prev_msg}\\nFIX: <describe what "
                    f"fixed it>\\nPROJECT: {project}', project='{project}', tags=['reusable', 'bugfix'])")
                runner(src_dir, "auto_self_improve.py",
                       ["fix", "--description-stdin", "--category",
                        str(previous.get("category") or "code_error"), "--fix",
                        "Command succeeded after previous failure", "--project", project],
                       stdin_text=f"Fixed: {prev_cmd}")
                unlink(error_file, "error resolved")
        else:
            unlink(error_file, "error expired")

    if TEST_CMD.search(command):
        messages.append("SELF_LEARNING_SUCCESS: Tests passed. If SOUL rules were relevant, rate them:\n"
                        "  self_rules(action='rate', id=<rule_id>, success=true)")
    if BUILD_CMD.search(command):
        messages.append("SELF_LEARNING_SUCCESS: Build succeeded. Rate relevant SOUL rules if any.")
    messages.extend(_significant(command, project))
    if FULL_SUITE_CMD.search(command):
        messages.append(
            "MEMORY_TRIGGER: Full test suite passed. If you completed a significant feature, save it:\n"
            f"  memory_save(type='solution', content='WHAT: ..., PROJECT: {project}, FILES: ..., "
            f"APPROACH: ...', project='{project}', tags=['reusable'])")
    if LINT_CMD.search(command):
        messages.append("SELF_LEARNING_SUCCESS: Linter passed. If conventions were established, save:\n"
                        f"  memory_save(type='convention', content='<convention>', project='{project}', "
                        f"tags=['convention'])")
    return messages


# ──────────────────────────────────────────────
# Recovery files (one per session)
# ──────────────────────────────────────────────

def recovery_path(recovery_dir: Path, session: str) -> Path:
    return recovery_dir / f"pending-{safe_name(session)}.md"


def recovery_meta(path: Path) -> JsonDict | None:
    try:
        with open(path, encoding="utf-8") as fh:
            first = fh.readline().strip()
    except OSError as exc:
        log(f"cannot read {path}: {exc}")
        return None
    if not (first.startswith(RECOVERY_HEADER_PREFIX) and first.endswith(RECOVERY_HEADER_SUFFIX)):
        return None
    try:
        meta = json.loads(first[len(RECOVERY_HEADER_PREFIX):-len(RECOVERY_HEADER_SUFFIX)])
    except json.JSONDecodeError:
        return None
    return meta if isinstance(meta, dict) else None


def _git(cwd: str, *args: str) -> str:
    try:
        result = subprocess.run(["git", "-C", cwd, *args], capture_output=True, text=True, encoding="utf-8", errors="replace",
                                timeout=GIT_TIMEOUT_SEC, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        log(f"git {' '.join(args)} failed in {cwd}: {exc}")
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def git_state(cwd: str) -> tuple[str, str, str]:
    if not cwd or not os.path.isdir(cwd) or not _git(cwd, "rev-parse", "--git-dir"):
        return "", "", ""
    status = "\n".join(_git(cwd, "status", "--short").splitlines()[:GIT_STATUS_MAX_LINES])
    return (_git(cwd, "rev-parse", "--abbrev-ref", "HEAD"), status,
            _git(cwd, "log", "--oneline", f"-{GIT_LOG_ENTRIES}"))


def write_recovery(recovery_dir: Path, session: str, project: str, cwd: str, reason: str,
                   user_context: str = "", assistant_context: str = "", transcript_events: int = 0,
                   min_interval_sec: int = 0) -> Path | None:
    path = recovery_path(recovery_dir, session)
    # A missing file has mtime 0.0, so it is always (re)written.
    if min_interval_sec > 0 and time.time() - _mtime(path) < min_interval_sec:
        return None
    branch, status, recent = git_state(cwd)
    now = datetime.now(timezone.utc)  # noqa: UP017 — must also run on a python3.9 fallback interpreter
    meta = {"session_id": session, "project": project, "cwd": cwd,
            "kind": RECOVERY_KIND_STOP if min_interval_sec > 0 else RECOVERY_KIND_END,
            "written_at": now.strftime("%Y-%m-%dT%H:%M:%SZ")}
    lines = [
        RECOVERY_HEADER_PREFIX + json.dumps(meta, ensure_ascii=False) + RECOVERY_HEADER_SUFFIX,
        f"# Session Recovery - {now.astimezone():%Y-%m-%d %H:%M:%S}",
        "",
        "## Context",
        f"- **Project**: {project}",
        f"- **Path**: {cwd}",
        f"- **Branch**: {branch or 'N/A'}",
        f"- **Reason**: {reason}",
    ]
    if transcript_events:
        lines.append(f"- **Transcript**: ~{transcript_events} transcript events")
    if user_context:
        lines += ["", "## Last User Requests", user_context]
    if assistant_context:
        lines += ["", "## Last Assistant Context", assistant_context]
    lines += ["", "## Git State", "### Modified files:", "```", status or "No git changes detected", "```",
              "", "### Recent commits:", "```", recent or "No recent commits", "```", ""]
    atomic_write(path, "\n".join(lines))
    files = sorted(recovery_dir.glob(RECOVERY_GLOB), key=_mtime, reverse=True)
    for extra in files[RECOVERY_KEEP_MAX:]:
        unlink(extra, "recovery cap reached")
    return path


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except FileNotFoundError:
        return 0.0


def pick_recovery(recovery_dir: Path, cwd: str, session: str) -> tuple[JsonDict, str] | None:
    """Newest recovery of another session in the same directory; consumed files and stale ones are removed."""
    if not recovery_dir.is_dir():
        return None
    cutoff = time.time() - RECOVERY_MAX_AGE_HOURS * 3_600
    chosen: tuple[Path, JsonDict] | None = None
    for path in sorted(recovery_dir.glob(RECOVERY_GLOB), key=_mtime, reverse=True):
        if _mtime(path) < cutoff:
            unlink(path, "recovery older than the retention window")
            continue
        meta = recovery_meta(path)
        if meta is None:
            # Pre-14.9 per-turn snapshots carry no owner; they cannot be attributed to a project.
            unlink(path, "legacy recovery snapshot without metadata")
            continue
        if meta.get("session_id") == session or os.path.normpath(str(meta.get("cwd") or "")) != os.path.normpath(cwd):
            continue
        if meta.get("kind") == RECOVERY_KIND_STOP and time.time() - _mtime(path) < ACTIVE_SNAPSHOT_WINDOW_SEC:
            # A parallel session in this directory is still running; leave its snapshot for a crash.
            continue
        if chosen is None:
            chosen = (path, meta)
        else:
            unlink(path, "superseded by a newer recovery for the same directory")
    if chosen is None:
        return None
    path, meta = chosen
    try:
        body = path.read_text(encoding="utf-8").split("\n", 1)[1]
    except (OSError, IndexError) as exc:
        log(f"cannot read recovery {path}: {exc}")
        return None
    unlink(path, "recovery delivered to the new session")
    return meta, body.strip()


# ──────────────────────────────────────────────
# SessionStart context
# ──────────────────────────────────────────────

def _json_list(raw: Any) -> list[str]:
    if not raw:
        return []
    if isinstance(raw, list):
        items = raw
    else:
        try:
            items = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return [str(raw)]
        if not isinstance(items, list):
            return [str(items)]
    return [str(item) for item in items if str(item).strip()]


def active_rules(db: sqlite3.Connection, project: str, preview: int) -> list[tuple[int, str]]:
    queries = (
        (("SELECT priority, content FROM rules WHERE status='active' "
          "AND (scope='global' OR project=? OR project='general') "
          "ORDER BY priority DESC, success_rate DESC LIMIT ?"), (project, RULES_LIMIT)),
        ("SELECT priority, content FROM rules WHERE status='active' ORDER BY priority DESC LIMIT ?",
         (RULES_LIMIT,)),
    )
    for sql, params in queries:
        try:
            rows = db.execute(sql, params).fetchall()
        except sqlite3.OperationalError:
            continue
        return [(int(p or 0), " ".join(str(c).split())[:preview]) for p, c in rows]
    return []


def restore_block(db_path: Path, project: str, source: str) -> str:
    """Latest session_continuity summary + active rules for compact/resume, bounded to RESTORE_MAX_CHARS."""
    lines = [f"SESSION_RESTORE ({source}) for project \"{project}\":"]
    found = False
    db = connect_ro(db_path)
    if db is not None:
        try:
            tables = table_names(db)
            if "session_summaries" in tables:
                row = db.execute(
                    "SELECT summary, next_steps, pitfalls, ended_at, branch FROM session_summaries "
                    "WHERE project = ? ORDER BY ended_at DESC, rowid DESC LIMIT 1", (project,)).fetchone()
                if row:
                    found = True
                    summary, next_steps, pitfalls, ended_at, branch = row
                    where = f" on branch {branch}" if branch else ""
                    lines.append(f"Last session summary (ended {ended_at}{where}):")
                    lines.append(truncate(str(summary or "").strip(), RESTORE_SUMMARY_MAX_CHARS))
                    for title, items in (("Next steps:", next_steps), ("Pitfalls:", pitfalls)):
                        values = _json_list(items)[:RESTORE_LIST_LIMIT]
                        if values:
                            lines.append(title)
                            lines.extend(f"  - {truncate(v, RESTORE_ITEM_MAX_CHARS)}" for v in values)
            if "rules" in tables:
                rules = active_rules(db, project, RULE_PREVIEW_CHARS)
                if rules:
                    lines.append("Active rules:")
                    lines.extend(f"  - [P{p}] {c}" for p, c in rules)
        except sqlite3.Error as exc:
            log(f"restore query failed on {db_path}: {exc}")
        finally:
            db.close()
    if not found:
        lines.append(f"No stored session summary. Call session_init(project=\"{project}\") and "
                     f"memory_recall(query=\"<current task>\", project=\"{project}\").")
    return truncate("\n".join(lines), RESTORE_MAX_CHARS)


def startup_block(db_path: Path, project: str) -> str:
    db = connect_ro(db_path)
    if db is None:
        return ""
    out: list[str] = []
    try:
        tables = table_names(db)
        if "rules" in tables:
            active = db.execute("SELECT COUNT(*) FROM rules WHERE status='active'").fetchone()[0]
            if active:
                out.append(f"SOUL_RULES: {active} active behavioral rule(s) loaded.")
                out.extend(f"  - [P{p}] {c}" for p, c in active_rules(db, project, STARTUP_RULE_PREVIEW_CHARS))
                out.append(f"  -> Call self_rules_context(project=\"{project}\") at session start")
        if "errors" in tables:
            patterns = db.execute(
                "SELECT category, COUNT(*) AS cnt FROM errors WHERE status != 'insight_extracted' "
                "AND created_at > datetime('now', ?) GROUP BY category HAVING cnt >= ?",
                (f"-{PATTERN_WINDOW_DAYS} days", PATTERN_MIN_COUNT)).fetchall()
            if patterns:
                out.append(f"PATTERN_ALERT: {len(patterns)} error pattern(s) detected "
                           f"({PATTERN_MIN_COUNT}+ in {PATTERN_WINDOW_DAYS} days).")
                out.extend(f"  - {cat} ({cnt}x)" for cat, cnt in patterns[:PATTERN_LIST_LIMIT])
                out.append("  -> Call self_patterns(view=\"full_report\") to analyze")
        if "knowledge" in tables and project and project != os.path.basename(os.path.expanduser("~")):
            kcount, scount = db.execute(
                "SELECT COUNT(*), COALESCE(SUM(type='solution'), 0) FROM knowledge "
                "WHERE project=? AND status='active'", (project,)).fetchone()
            if kcount:
                out.append(f"PROJECT_MEMORY: \"{project}\" has {kcount} knowledge records ({scount} solutions)")
                out.append(f"  -> memory_recall(query=\"<task>\", project=\"{project}\") before starting work")
            else:
                out.append(f"PROJECT_MEMORY: No prior knowledge for \"{project}\". Start fresh.")
    except sqlite3.Error as exc:
        log(f"session-start query failed on {db_path}: {exc}")
    finally:
        db.close()
    return "\n".join(out)


def recovery_block(recovery_dir: Path, cwd: str, session: str, project: str) -> str:
    picked = pick_recovery(recovery_dir, cwd, session)
    if picked is None:
        return ""
    meta, body = picked
    return (f"RECOVERY: context of the previous session in this directory "
            f"(written {meta.get('written_at', 'unknown')}). If something here was not saved yet, "
            f"save it with memory_save(project=\"{project}\").\n--- RECOVERY ---\n"
            f"{truncate(body, RECOVERY_DISPLAY_MAX_CHARS)}\n--- END RECOVERY ---")


def session_start_context(db_path: Path, project: str, cwd: str, session: str, source: str,
                          recovery_dir: Path) -> str:
    parts: list[str] = []
    if source in ("compact", "resume"):
        parts.append(restore_block(db_path, project, source))
    else:
        parts.append(startup_block(db_path, project))
    if source in ("startup", "resume", ""):
        parts.append(recovery_block(recovery_dir, cwd, session, project))
    return "\n\n".join(p for p in parts if p)


# ──────────────────────────────────────────────
# SessionEnd
# ──────────────────────────────────────────────

def transcript_messages(transcript: Path, src_dir: Path) -> tuple[str, str]:
    """(user, assistant) excerpts parsed by auto_episode_capture (nested and legacy flat formats)."""
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    try:
        from auto_episode_capture import extract_messages_from_transcript
    except ImportError as exc:
        log(f"transcript parser unavailable from {src_dir}: {exc}")
        return "", ""
    messages = extract_messages_from_transcript(transcript)
    users = [m["text"] for m in messages
             if m.get("role") == "user" and not m["text"].lstrip().startswith(TRANSCRIPT_NOISE_PREFIXES)]
    assistants = [m["text"] for m in messages if m.get("role") == "assistant"]
    user = "\n".join(u[:TRANSCRIPT_USER_MSG_CHARS] for u in users[-TRANSCRIPT_USER_MSGS:])
    assistant = "\n".join(assistants[-TRANSCRIPT_ASSISTANT_MSGS:])[:TRANSCRIPT_ASSISTANT_CHARS]
    # Redacted after truncation could leave half a key; redact the joined text, then bound it again.
    return (redact_text(user, src_dir)[:TRANSCRIPT_USER_MSGS * TRANSCRIPT_USER_MSG_CHARS],
            redact_text(assistant, src_dir)[:TRANSCRIPT_ASSISTANT_CHARS])


def _line_count(path: Path) -> int:
    try:
        with open(path, "rb") as fh:
            return sum(1 for _ in fh)
    except OSError as exc:
        log(f"cannot count lines of {path}: {exc}")
        return 0


def find_transcript(transcript_path: str, cwd: str, session: str = "") -> Path | None:
    if transcript_path and Path(transcript_path).is_file():
        return Path(transcript_path)
    if not cwd:
        return None
    escaped = re.sub(r"[^A-Za-z0-9]", "-", cwd)
    project_dir = Path.home() / ".claude" / "projects" / escaped
    if session:
        # The session's own transcript; the newest file may belong to a concurrent session.
        own = project_dir / f"{safe_name(session)}.jsonl"
        if own.is_file():
            return own
    candidates = sorted(project_dir.glob("*.jsonl"), key=_mtime, reverse=True)
    return candidates[0] if candidates else None


def session_end(payload: JsonDict, src_dir: Path, recovery_dir: Path, project: str, reason_text: str,
                runner: Callable[[Path, str, Sequence[str]], bool] = run_script) -> list[str]:
    """Write the session's recovery file and start auto_session_save / auto_episode_capture /
    auto_session_end."""
    cwd = str(payload.get("cwd") or os.getcwd())
    session = session_of(payload)
    actions: list[str] = []
    transcript = find_transcript(str(payload.get("transcript_path") or ""), cwd, session)
    user = assistant = ""
    events = 0
    if transcript is not None:
        user, assistant = transcript_messages(transcript, src_dir)
        events = _line_count(transcript)
    if user or assistant:
        path = write_recovery(recovery_dir, session, project, cwd, reason_text, user, assistant, events)
        actions.append(f"recovery saved to {path}")
    tail = assistant[:AUTOSAVE_ASSISTANT_CHARS]
    context = json.dumps({"user": user, "assistant": tail})
    if user and runner(src_dir, "auto_session_save.py",
                       ["--project", project, "--cwd", cwd, "--reason", reason_text, "--context-stdin"],
                       stdin_text=context):
        actions.append("auto_session_save started")
    if transcript is not None and runner(src_dir, "auto_episode_capture.py",
                                         ["--session-id", transcript.stem, "--project", project,
                                          "--transcript", str(transcript)]):
        actions.append(f"auto_episode_capture started for {transcript.stem}")
    # Writes the session_summaries row (producer="hook") the next session_init resumes from.
    # The payload's id first: a transcript found by the cwd fallback may be another session's.
    if transcript is not None:
        end_session = str(payload.get("session_id") or transcript.stem)
        if runner(src_dir, "auto_session_end.py",
                  ["--session-id", end_session, "--project", project, "--cwd", cwd,
                   "--branch", _git(cwd, "branch", "--show-current"), "--reason", reason_text],
                  stdin_text=json.dumps({"assistant": tail})):
            actions.append(f"auto_session_end started for {end_session}")
    return actions


# ──────────────────────────────────────────────
# Background writers (UserPromptSubmit intents, PostToolUse observations)
# ──────────────────────────────────────────────

def consume_input_file(path: Path) -> JsonDict:
    """Read and delete the payload cached by the hook for a detached worker."""
    try:
        raw = path.read_text(encoding="utf-8")
    finally:
        unlink(path, "payload consumed")
    return parse_payload(raw)


def prompt_of(payload: JsonDict) -> str:
    # Claude Code sends {"prompt": ...}; older builds used user_message.content.
    prompt = payload.get("prompt")
    if not prompt:
        user_msg = payload.get("user_message")
        prompt = user_msg.get("content") if isinstance(user_msg, dict) else ""
    return str(prompt or "").strip()


def save_intent_from(payload: JsonDict, src_dir: Path, db_path: Path) -> bool:
    prompt = prompt_of(payload)
    if not prompt or not db_path.is_file():
        return False
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from intents import save_intent
    save_intent(str(db_path), prompt, session_of(payload), project_of(payload))
    return True


def tool_output_text(payload: JsonDict) -> str:
    response = payload.get("tool_response") or {}
    if isinstance(response, str):
        return response.strip()
    parts: list[str] = []
    if isinstance(response, dict):
        for key in ("stdout", "stderr", "output", "content"):
            value = response.get(key)
            if value:
                parts.append(value if isinstance(value, str) else json.dumps(value))
    return "\n".join(parts).strip()


def capture_observation_from(payload: JsonDict, src_dir: Path, memory_dir: Path) -> bool:
    tool_name = str(payload.get("tool_name") or "")
    combined = tool_output_text(payload)
    if not tool_name or not combined:
        return False
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from auto_extract_active import capture_tool_observation
    capture_tool_observation(tool_name, combined, session_of(payload), project_of(payload),
                             queue_dir=memory_dir / "extract-queue")
    return True


# ──────────────────────────────────────────────
# Ambient memory (src/ambient): action log, prompt context, rule reminders,
# turn end, Codex capture
# ──────────────────────────────────────────────

# Docker installs: where the image keeps src/, and how long one command may take
# (inside every client hook timeout of 5 s: the field parse and the nudge spawn use the rest).
DOCKER_SRC_DIR = "/app/src"
DOCKER_AMBIENT_TIMEOUT_SEC = 3.0
# Ambient settings forwarded into the container (src/ambient/settings.py SETTING_KEYS).
AMBIENT_ENV_KEYS = ("TAM_ACTION_LOG", "TAM_ACTION_LOG_RETENTION_DAYS", "TAM_AMBIENT_RECALL",
                    "TAM_AMBIENT_TOKEN_BUDGET", "TAM_AMBIENT_LATENCY_MS", "TAM_RULE_REMINDERS",
                    "TAM_RULE_COMPLIANCE", "TAM_TURN_AUTOSAVE", "TAM_CORRECTION_CAPTURE")
SRC_LOG_FORMAT = "[%(asctime)s] [tam-hook] %(name)s %(levelname)s %(message)s"
SRC_LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class _HookLogHandler(logging.FileHandler):
    """Marks the handler this runtime installed, so it is added once per process."""


def configure_src_logging() -> None:
    """Route src/ logging to hooks.log (INFO); without a log file only warnings reach stderr."""
    root = logging.getLogger()
    if any(isinstance(handler, _HookLogHandler) for handler in root.handlers):
        return
    log_file = os.environ.get("TAM_HOOK_LOG_FILE")
    if not log_file:
        logging.basicConfig(level=logging.WARNING, format=SRC_LOG_FORMAT, datefmt=SRC_LOG_DATE_FORMAT)
        return
    try:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handler = _HookLogHandler(log_file, encoding="utf-8")
    except OSError as exc:
        log(f"cannot open {log_file} for src logging: {exc}")
        return
    handler.setFormatter(logging.Formatter(SRC_LOG_FORMAT, SRC_LOG_DATE_FORMAT))
    root.addHandler(handler)
    root.setLevel(logging.INFO)


def load_ambient(src_dir: Path) -> bool:
    """Make src/ importable; False (logged) when the ambient package cannot be imported."""
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    configure_src_logging()
    try:
        import ambient.settings  # noqa: F401 — import probe for the whole package's dependencies
    except ImportError as exc:
        log(f"ambient memory unavailable from {src_dir}: {exc}")
        return False
    return True


def run_ambient(name: str, job: Callable[[], Any], default: Any = None) -> Any:
    """Hook boundary of an ambient job: any failure is logged with its type, never raised."""
    try:
        return job()
    except Exception as exc:  # noqa: BLE001 — a hook must never fail the client; logged with type
        log(f"{name} failed: {type(exc).__name__}: {exc}")
        return default


def hook_call(payload: JsonDict, project: str) -> JsonDict:
    """The payload plus what this runtime derives from it: src/ambient/hooks.py HookCall."""
    code = exit_code(payload)
    return {"payload": payload, "project": project, "session_id": session_of(payload),
            "cwd": str(payload.get("cwd") or os.getcwd()), "exit_code": code,
            "failed": is_failure(code) or payload.get("hook_event_name") == "PostToolUseFailure",
            "interrupted": interrupted(payload)}


def ambient_call(command: str, call: JsonDict, memory_dir: Path, src_dir: Path) -> str:
    """Run one src/ambient hook command; its text, or '' when it cannot run (always logged)."""
    if docker_mode():
        return docker_ambient_call(command, call)
    if not load_ambient(src_dir):
        mark_ambient_unavailable(memory_dir, str(call.get("session_id") or "unknown"))
        return ""

    def job() -> str:
        from ambient.hooks import HookCall, run

        return run(command, HookCall.from_json(json.dumps(call)), memory_dir)

    return str(run_ambient(command, job, "") or "")


AMBIENT_UNAVAILABLE_NOTICE = (
    "TAM ambient memory is off for this session: the hook interpreter cannot import TAM's packages "
    "(action log, rule reminders and prompt recall are disabled). Install into the TAM venv "
    "(~/.tam/.venv or <checkout>/.venv) or set HOOK_PYTHON; details are in hooks.log.")


def _ambient_flag(memory_dir: Path, session: str, suffix: str) -> Path:
    return memory_dir / "state" / f"ambient-unavailable-{safe_name(session)}.{suffix}"


def mark_ambient_unavailable(memory_dir: Path, session: str) -> None:
    """Remember that ambient memory could not run, so the Stop hook can tell the user once."""
    flag = _ambient_flag(memory_dir, session, "flag")
    if flag.exists() or _ambient_flag(memory_dir, session, "shown").exists():
        return
    atomic_write(flag, "")


def ambient_unavailable_notice(memory_dir: Path, session: str) -> str:
    """The user-facing notice the first time it is asked for after a failure, '' afterwards."""
    flag = _ambient_flag(memory_dir, session, "flag")
    if not flag.exists():
        return ""
    try:
        os.replace(flag, _ambient_flag(memory_dir, session, "shown"))
    except OSError as exc:
        log(f"cannot update {flag}: {exc}")
        return ""
    return AMBIENT_UNAVAILABLE_NOTICE


def docker_ambient_call(command: str, call: JsonDict) -> str:
    """The same command inside the TAM container, whose volume holds the store (Docker installs)."""
    forwarded = [arg for key in AMBIENT_ENV_KEYS if os.environ.get(key)
                 for arg in ("-e", f"{key}={os.environ[key]}")]
    command_line = ["docker", "exec", "-i", "-w", DOCKER_SRC_DIR, *forwarded, docker_container(),
                    "python", "-m", "ambient", command]
    try:
        result = subprocess.run(command_line, input=json.dumps(call), capture_output=True, text=True, encoding="utf-8", errors="replace",
                                timeout=DOCKER_AMBIENT_TIMEOUT_SEC, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        log(f"{command} in container {docker_container()} did not run: {exc}")
        return ""
    if result.returncode != 0:
        log(f"{command} in container {docker_container()} failed (exit {result.returncode}): "
            f"{result.stderr.strip()[-ERROR_MSG_MAX_CHARS:]}")
        return ""
    return result.stdout.strip()


# ──────────────────────────────────────────────
# Remote mode (TAM_REMOTE_URL)
# ──────────────────────────────────────────────
# With TAM_REMOTE_URL set, every hook POSTs to the server's /internal/hook/{cmd}
# instead of opening memory.db; nothing on this path may import src/ modules
# that touch sqlite. PostToolUse only appends to a local spool; Stop,
# UserPromptSubmit and SessionEnd flush it first, in order, chunked under the
# server's 256 KB body limit. Every transport failure keeps the spool and the
# hook still exits 0.

REMOTE_URL_ENV = "TAM_REMOTE_URL"
REMOTE_TOKEN_ENV = "TAM_REMOTE_TOKEN"
REMOTE_WRITE_TIMEOUT_SEC = 10.0
REMOTE_BODY_LIMIT_BYTES = 256 * 1024
# One deadline per hook process, under the hooks.json timeouts (5/15/20 s):
# sync hooks stay fast, Stop and SessionEnd may spend most of their budget
# flushing the spool on a slow uplink.
REMOTE_SYNC_BUDGET_SEC = 3.0
REMOTE_STOP_BUDGET_SEC = 13.0
REMOTE_END_BUDGET_SEC = 18.0
REMOTE_CALL_FIELD_MAX_BYTES = 200 * 1024
SPOOL_ACTION_LIMIT = 400
SPOOL_CHUNK_BYTES = 200 * 1024
# A foreign spool younger than this may still be appended to by its live
# session; only stale files are safe to take aside and sweep.
SPOOL_STALE_SEC = 600.0
REMOTE_EPISODE_MESSAGES_MAX_BYTES = 64 * 1024
REMOTE_EPISODE_MESSAGE_LIMIT = 200

REMOTE_SUMMARY_MAX_CHARS = 400
REMOTE_DETAIL_SNIPPET_CHARS = 240
REMOTE_EDIT_DETAIL_SEPARATOR = " :: "
REMOTE_MAX_TOOL_CHARS = 120
REMOTE_UNLOGGED_TOOLS = frozenset({"Read", "Glob", "Grep", "LS", "TodoWrite", "TodoRead"})
REMOTE_EDIT_TOOLS = frozenset({"Edit", "Write", "MultiEdit", "NotebookEdit"})
REMOTE_SHELL_TOOLS = frozenset({"Bash"})
REMOTE_LOCATOR_KEYS = ("file_path", "notebook_path", "path", "pattern", "url", "query", "description")
REMOTE_MCP_PREFIX = "mcp__"

_REMOTE_TOKEN: str | None = None


def remote_url() -> str:
    return os.environ.get(REMOTE_URL_ENV, "").strip().rstrip("/")


def remote_mode() -> bool:
    return bool(remote_url())


def remote_token(timeout: float = 5.0) -> str:
    """Bearer token for /internal/hook: env first, then the OS secret store; cached per process."""
    global _REMOTE_TOKEN
    if _REMOTE_TOKEN is None:
        _REMOTE_TOKEN = _remote_token_lookup(timeout)
    return _REMOTE_TOKEN


def _remote_token_lookup(timeout: float) -> str:
    env = os.environ.get(REMOTE_TOKEN_ENV, "").strip()
    if env:
        return env
    if sys.platform == "darwin":
        command = ["security", "find-generic-password", "-s", "tam-remote-token", "-w"]
    else:
        command = ["secret-tool", "lookup", "service", "tam-remote-token"]
    try:
        result = subprocess.run(command, capture_output=True, text=True,
                                timeout=max(0.1, timeout), check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        log(f"remote token lookup failed: {exc}")
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def remote_post(command: str, call: JsonDict, payload: JsonDict, timeout: float) -> tuple[bool, str]:
    """POST ``{"call", "payload"}`` to /internal/hook/<command>; (False, "") on any failure.

    ``timeout`` bounds the whole call, token lookup included: callers pass the
    time left before their hook deadline, so a hung server cannot outlive it.
    """
    if timeout <= 0:
        return False, ""
    body = json.dumps({"call": call, "payload": payload}, ensure_ascii=False).encode("utf-8")
    if len(body) > REMOTE_BODY_LIMIT_BYTES:
        log(f"remote {command} dropped: body of {len(body)} bytes exceeds {REMOTE_BODY_LIMIT_BYTES}")
        return False, ""
    started = time.monotonic()
    headers = {"Content-Type": "application/json",
               "X-TAM-Session": str(call.get("session_id") or "")}
    token = remote_token(min(5.0, timeout))
    if token:
        headers["Authorization"] = f"Bearer {token}"
    remaining = timeout - (time.monotonic() - started)
    if remaining <= 0:
        return False, ""
    request = urllib.request.Request(f"{remote_url()}/internal/hook/{command}", data=body,
                                     headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=remaining) as response:
            data = json.loads(response.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 - a hook must never fail the client
        log(f"remote {command} failed: {type(exc).__name__}: {exc}")
        return False, ""
    if not isinstance(data, dict) or data.get("ok") is not True:
        error = data.get("error") if isinstance(data, dict) else data
        log(f"remote {command} rejected: {str(error)[:300]}")
        return False, ""
    return True, str(data.get("output") or "")


# ── action spool ─────────────────────────────────────────────────────

def _spool_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "tam" / "spool"


def spool_path(session: str) -> Path:
    return _spool_dir() / f"{safe_name(session)}.jsonl"


def spool_append(session: str, action: JsonDict) -> None:
    """One JSONL line per tool call; flushed by the next Stop/Submit/SessionEnd."""
    path = spool_path(session)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, PRIVATE_FILE_MODE)
    with os.fdopen(fd, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(action, ensure_ascii=False) + "\n")


def _spool_entries(path: Path) -> list[tuple[str, JsonDict]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return []
    except OSError as exc:
        log(f"cannot read spool {path}: {exc}")
        return []
    entries = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            action = json.loads(line)
        except ValueError:
            log(f"dropping corrupt spool line in {path}")
            continue
        if isinstance(action, dict):
            entries.append((line, action))
    return entries


def _spool_keep(path: Path, entries: list[tuple[str, JsonDict]]) -> None:
    if not entries:
        unlink(path, "spool flushed")
        return
    atomic_write(path, "".join(f"{line}\n" for line, _ in entries))


def _spool_pending() -> list[Path]:
    """Spool files awaiting delivery: in-flight .sending.* first, then .jsonl,
    each oldest first, so a flush interrupted mid-POST is resumed before newer work."""
    try:
        files = [path for path in _spool_dir().iterdir() if path.is_file()]
    except OSError:
        return []
    sending = sorted((p for p in files if ".sending." in p.name), key=_mtime)
    pending = sorted((p for p in files if p.name.endswith(".jsonl")), key=_mtime)
    return sending + pending


def _spool_session(path: Path) -> str:
    """The session a spool file belongs to: the filename stem before .jsonl
    or .sending.<pid>."""
    name = path.name.split(".sending.", 1)[0]
    if name.endswith(".jsonl"):
        name = name[: -len(".jsonl")]
    return name


def _spool_take(path: Path) -> Path:
    """Rename a pending spool aside; appends during the flush start a fresh file."""
    sending = path.with_name(f"{path.stem}.sending.{os.getpid()}")
    os.replace(path, sending)
    return sending


def _spool_call(call: JsonDict, path: Path, entries: list[tuple[str, JsonDict]]) -> JsonDict:
    """The batch call for one spool file: session, project and cwd come from the
    spooled lines - a swept file belongs to another session working in another
    project, so the flushing hook's own fields are only the fallback for lines
    written before they carried them."""
    batch = {**call, "payload": {}}
    sourced: set[str] = set()
    for _line, action in entries:
        for key in ("session_id", "project", "cwd"):
            if key in sourced:
                continue
            value = action.get(key)
            if value:
                batch[key] = str(value)
                sourced.add(key)
    if "session_id" not in sourced:
        name = _spool_session(path)
        if name:
            batch["session_id"] = name
    return batch


def _flush_entries(path: Path, call: JsonDict, entries: list[tuple[str, JsonDict]],
                   deadline: float) -> bool:
    """POST one spool file's actions as actions-batch chunks, oldest first.

    Sent chunks leave the file as they succeed, so a crash or a failed chunk
    replays at most the unacknowledged tail; the server dedups by tool_use_id.
    The batch call carries an empty payload: the stored hook payload can be
    large and the 256 KB body limit must fit the actions themselves.
    """
    index = 0
    while index < len(entries):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        chunk: list[JsonDict] = []
        size = 0
        while index + len(chunk) < len(entries) and len(chunk) < SPOOL_ACTION_LIMIT:
            line, action = entries[index + len(chunk)]
            line_bytes = len(line.encode("utf-8"))
            if chunk and size + line_bytes > SPOOL_CHUNK_BYTES:
                break
            chunk.append(action)
            size += line_bytes
        ok, _ = remote_post("actions-batch", call, {"actions": chunk}, remaining)
        if not ok:
            return False
        index += len(chunk)
        try:
            _spool_keep(path, entries[index:])
        except OSError as exc:
            log(f"cannot update spool {path}: {exc}")
    return True


def _flush_one(path: Path, call: JsonDict, deadline: float) -> bool:
    """Flush a single spool file: a .jsonl is taken aside first (appends during
    the POSTs land in a fresh file), then its actions go out as actions-batch
    chunks under the session/project/cwd carried by the spooled lines. False on
    a transport failure or the deadline; whatever is left stays on disk."""
    if path.name.endswith(".jsonl"):
        try:
            path = _spool_take(path)
        except FileNotFoundError:
            return True
        except OSError as exc:
            log(f"cannot take spool {path}: {exc}")
            return False
    entries = _spool_entries(path)
    if not entries:
        unlink(path, "spool flushed")
        return True
    return _flush_entries(path, _spool_call(call, path, entries), entries, deadline)


def remote_flush_own(call: JsonDict, deadline: float) -> bool:
    """Flush THIS session's spool files (.sending.* of any pid, then .jsonl)
    before ``deadline``; the follow-up POST is gated on this result so the turn
    boundary never closes ahead of the turn's actions.

    Own files are re-listed until none remain: a concurrent hook of the same
    session can leave a .sending.<pid> behind mid-flush, and resending is safe
    because the server dedups by tool_use_id. No lock is ever held across a POST.
    """
    own_name = safe_name(str(call.get("session_id") or ""))
    while True:
        own = [path for path in _spool_pending() if _spool_session(path) == own_name]
        if not own:
            return True
        if time.monotonic() >= deadline:
            return False
        for path in own:
            if not _flush_one(path, call, deadline):
                return False


def remote_sweep_stale(call: JsonDict, deadline: float) -> None:
    """Best-effort flush of OTHER sessions' spools, only once they are stale
    (mtime older than SPOOL_STALE_SEC - a live session may still append through
    an fd opened before the take-aside rename). Runs after the follow-up POST
    with whatever deadline is left; failures are logged, never reported."""
    own_name = safe_name(str(call.get("session_id") or ""))
    cutoff = time.time() - SPOOL_STALE_SEC
    for path in _spool_pending():
        if time.monotonic() >= deadline:
            return
        if _spool_session(path) == own_name or _mtime(path) >= cutoff:
            continue
        entries = _spool_entries(path)
        if entries and not all(action.get("project") for _line, action in entries):
            # Lines without project are old format: only their own session can
            # send them under the right project.
            continue
        if not _flush_one(path, call, deadline):
            return


# ── tool summary (same selection as src/ambient/actions.summarize_tool_input) ──

def _remote_clip(text: str, limit: int) -> str:
    # Same clip as src/ambient/actions._clip: limit-1 chars plus the ellipsis.
    return text if len(text) <= limit else text[: max(0, limit - 1)].rstrip() + "…"


def _remote_one_line(text: Any) -> str:
    return " ".join(str(text).split())


def _remote_snippet(value: Any) -> str:
    return (_remote_clip(_remote_one_line(value), REMOTE_DETAIL_SNIPPET_CHARS)
            if isinstance(value, str) and value.strip() else "")


def remote_tool_summary(tool: str, tool_input: Any, src_dir: Path) -> str:
    """Short description of one tool call, identical to the local
    summarize_tool_input: redact first, then clip, so a credential straddling
    the boundary is fully redacted instead of stored half-cut."""
    data = tool_input if isinstance(tool_input, dict) else {}
    if tool in REMOTE_SHELL_TOOLS:
        text = _remote_one_line(data.get("command") or "")
    elif tool in REMOTE_EDIT_TOOLS:
        target = str(data.get("file_path") or data.get("notebook_path") or "")
        if tool == "MultiEdit" and isinstance(data.get("edits"), list):
            pieces = [edit.get("new_string") for edit in data["edits"] if isinstance(edit, dict)]
            detail = _remote_snippet(" ".join(piece for piece in pieces if isinstance(piece, str)))
        else:
            detail = _remote_snippet(data.get("new_string") or data.get("content") or data.get("new_source") or "")
        text = f"{target}{REMOTE_EDIT_DETAIL_SEPARATOR}{detail}" if detail else target
    else:
        # MCP tools keep only argument names: their values can be whole memory records.
        locator = "" if tool.startswith(REMOTE_MCP_PREFIX) else next(
            (str(data[key]) for key in REMOTE_LOCATOR_KEYS
             if isinstance(data.get(key), str) and data[key]), "")
        text = _remote_one_line(locator) if locator else (
            "keys: " + ", ".join(sorted(map(str, data))) if data else "")
    return _remote_clip(redact_text(text, src_dir), REMOTE_SUMMARY_MAX_CHARS)


def _remote_exit_code(raw: Any) -> int | None:
    if raw is None or isinstance(raw, bool):
        return None
    try:
        return int(str(raw).strip())
    except ValueError:
        return None


def remote_record_action(payload: JsonDict, src_dir: Path) -> None:
    """Spool one PostToolUse action; no network on this path."""
    tool = _remote_one_line(payload.get("tool_name") or "")[:REMOTE_MAX_TOOL_CHARS]
    if not tool or tool in REMOTE_UNLOGGED_TOOLS:
        return
    code = exit_code(payload)
    if interrupted(payload):
        status = "interrupted"
    elif is_failure(code) or payload.get("hook_event_name") == "PostToolUseFailure":
        status = "failed"
    else:
        status = "ok"
    action: JsonDict = {"tool": tool,
                        "summary": remote_tool_summary(tool, payload.get("tool_input"), src_dir),
                        "status": status,
                        "exit_code": _remote_exit_code(code),
                        # The sweep of another session's spool needs its own id,
                        # project and cwd; the server ignores these keys on
                        # actions-batch but the batch call is built from them.
                        "session_id": session_of(payload),
                        "project": project_of(payload),
                        "cwd": str(payload.get("cwd") or "")}
    tool_use_id = str(payload.get("tool_use_id") or "")
    if tool_use_id:
        action["tool_use_id"] = tool_use_id
    spool_append(session_of(payload), action)


# ── session-end transcript extraction (same rules, stdlib-only) ──────

def _remote_transcript_messages(path: Path) -> list[JsonDict]:
    """user/assistant/tool messages of a Claude Code JSONL transcript; the same
    selection rules as auto_episode_capture.extract_messages_from_transcript."""
    messages: list[JsonDict] = []

    def text_of(content: Any) -> str:
        if isinstance(content, list):
            return " ".join(item.get("text", "") for item in content
                            if isinstance(item, dict) and item.get("type") == "text")
        return content if isinstance(content, str) else ""

    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(entry, dict):
                    continue
                entry_type = str(entry.get("type") or "")
                message = entry.get("message")
                message = message if isinstance(message, dict) else {}
                role = str(message.get("role") or entry.get("role") or "")
                content = message.get("content") or entry.get("content") or ""
                if entry_type == "user" or (role == "user" and entry_type in ("human", "")):
                    text = text_of(content).strip()
                    if text:
                        messages.append({"role": "user", "text": text})
                elif entry_type == "assistant" or role == "assistant":
                    text = text_of(content).strip()
                    if text:
                        messages.append({"role": "assistant", "text": text[:500]})
                elif entry_type == "tool_use":
                    name = str(entry.get("name") or entry.get("tool") or "")
                    if name:
                        messages.append({"role": "tool", "text": name})
    except OSError as exc:
        log(f"cannot read transcript {path}: {exc}")
    return messages[-REMOTE_EPISODE_MESSAGE_LIMIT:]


def remote_session_texts(transcript: Path | None, src_dir: Path
                         ) -> tuple[list[str], str, list[JsonDict]]:
    """(user_texts, assistant_text, episode_messages) for the session-end payload,
    with the same bounds the local session_end applies."""
    if transcript is None:
        return [], "", []
    messages = _remote_transcript_messages(transcript)
    users = [str(m["text"]) for m in messages
             if m.get("role") == "user"
             and not str(m["text"]).lstrip().startswith(TRANSCRIPT_NOISE_PREFIXES)]
    user_texts = [redact_text(u[:TRANSCRIPT_USER_MSG_CHARS], src_dir)
                  for u in users[-TRANSCRIPT_USER_MSGS:]]
    assistants = [str(m["text"]) for m in messages if m.get("role") == "assistant"]
    assistant_text = redact_text("\n".join(assistants[-TRANSCRIPT_ASSISTANT_MSGS:]),
                                 src_dir)[:TRANSCRIPT_ASSISTANT_CHARS]
    episode: list[JsonDict] = []
    size = 0
    for message in reversed(messages):
        entry = {"role": str(message.get("role") or ""), "text": str(message.get("text") or "")}
        if entry["role"] not in ("user", "assistant", "tool") or not entry["text"].strip():
            continue
        cost = len(json.dumps(entry, ensure_ascii=False).encode("utf-8"))
        if size + cost > REMOTE_EPISODE_MESSAGES_MAX_BYTES:
            break
        size += cost
        episode.append(entry)
    episode.reverse()
    return user_texts, assistant_text, episode


def _json_size(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False).encode("utf-8"))


def _clip_json_text(text: str, render: Callable[[str], Any], budget: int) -> Any:
    """The largest prefix of ``text`` whose ``render``ed JSON form fits ``budget``
    encoded bytes; escapes count, so the raw length alone is not the measure."""
    if _json_size(render(text)) <= budget:
        return render(text)
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _json_size(render(text[:mid])) <= budget:
            lo = mid
        else:
            hi = mid - 1
    return render(text[:lo])


def _remote_slim_call(call: JsonDict, *fields: str) -> JsonDict:
    """The call trimmed to the payload fields the server command reads; the rest
    of a raw hook payload can be huge and must not push the body over 256 KB.
    One shared budget across the kept fields, measured on the JSON-encoded size,
    so escapes and the field envelope count too."""
    source = call.get("payload")
    payload: JsonDict = {}
    budget = REMOTE_CALL_FIELD_MAX_BYTES
    if isinstance(source, dict):
        for key in fields:
            value = source.get(key)
            if isinstance(value, str):
                kept = _clip_json_text(value, lambda text: text, budget)
            elif isinstance(value, dict) and isinstance(value.get("content"), str):
                kept = _clip_json_text(value["content"],
                                       lambda text, inner=value: {**inner, "content": text},
                                       budget)
            elif isinstance(value, dict):
                kept = value
            else:
                continue
            size = _json_size(kept)
            if size <= budget:
                payload[key] = kept
                budget -= size
    return {**call, "payload": payload}


def remote_session_end(payload: JsonDict, project: str, src_dir: Path, reason_text: str,
                       deadline: float) -> bool:
    """Flush the spool, then hand the pre-extracted session text to session-end.
    A failed flush keeps the spool for the next sweep and skips session-end:
    the turn boundary must not close before the turn's actions landed."""
    session = session_of(payload)
    cwd = str(payload.get("cwd") or os.getcwd())
    call = hook_call(payload, project)
    if not remote_flush_own(call, deadline):
        return False
    transcript = find_transcript(str(payload.get("transcript_path") or ""), cwd, session)
    user_texts, assistant_text, episode = remote_session_texts(transcript, src_dir)
    # Git before the deadline read: its timeout is not part of the POST budget.
    branch = _git(cwd, "branch", "--show-current")
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return False
    ok, _ = remote_post("session-end", call, {
        "user_texts": user_texts,
        "assistant_text": assistant_text,
        "episode_messages": episode,
        "branch": branch,
        "project": project,
        "reason": reason_text,
    }, remaining)
    remote_sweep_stale(call, deadline)
    return ok


# ── memory-trigger side effects via error-capture ────────────────────

def _remote_cli_options(args: Sequence[str]) -> dict[str, str]:
    options: dict[str, str] = {}
    index = 0
    while index < len(args):
        arg = str(args[index])
        if arg.startswith("--") and index + 1 < len(args) and not str(args[index + 1]).startswith("--"):
            options[arg] = str(args[index + 1])
            index += 2
        else:
            index += 1
    return options


class RemoteErrorCaptureRunner:
    """run_script stand-in for remote mode: auto_self_improve.py error/fix/check-patterns
    queue up as error-capture requests; flush() hands the whole ordered list to
    ONE detached remote-post worker, so the server applies them in the order
    memory_trigger issued them (a per-call worker could POST check-patterns
    before the error it counts). Every other script reports failure like a
    missing file."""

    def __init__(self, call: JsonDict) -> None:
        self._call = call
        self.requests: list[JsonDict] = []

    def __call__(self, _src_dir: Path, script: str, args: Sequence[str],
                 stdin_text: str | None = None) -> bool:
        if script != "auto_self_improve.py" or not args or args[0] not in ("error", "fix", "check-patterns"):
            return False
        options = _remote_cli_options(list(args[1:]))
        payload: JsonDict = {"action": str(args[0]),
                             "project": str(options.get("--project") or self._call.get("project") or "general")}
        if args[0] == "check-patterns":
            if "--days" in options:
                payload["days"] = options["--days"]
        else:
            payload["description"] = stdin_text or ""
            payload["category"] = str(options.get("--category") or "general")
            if args[0] == "error":
                payload["severity"] = str(options.get("--severity") or "medium")
            else:
                payload["fix"] = str(options.get("--fix") or "")
        self.requests.append({"command": "error-capture", "call": {**self._call, "payload": {}},
                              "payload": payload})
        return True

    def flush(self) -> None:
        """Spawn the single detached worker with the ordered request list."""
        if not self.requests:
            return
        helper = str(Path(__file__).resolve())
        request = json.dumps(self.requests, ensure_ascii=False)
        log_file = os.environ.get("TAM_HOOK_LOG_FILE")
        with ExitStack() as stack:
            sink: Any = subprocess.DEVNULL
            if log_file:
                try:
                    sink = stack.enter_context(open(log_file, "a", encoding="utf-8"))
                except OSError as exc:
                    log(f"cannot open {log_file}, remote-post output discarded: {exc}")
            try:
                child = subprocess.Popen([sys.executable, helper, "remote-post"],
                                         stdin=subprocess.PIPE, stdout=sink, stderr=sink,
                                         **_detached_kwargs())
            except OSError as exc:
                log(f"cannot start remote-post worker: {exc}")
                return
        if child.stdin is None:
            return
        try:
            child.stdin.write(request.encode("utf-8"))
            child.stdin.close()
        except OSError as exc:
            log(f"cannot pass requests to remote-post worker: {exc}")


def remote_error_capture_runner(call: JsonDict) -> RemoteErrorCaptureRunner:
    return RemoteErrorCaptureRunner(call)


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def _print_shell(assignments: dict[str, str]) -> None:
    for name, value in assignments.items():
        print(f"{name}={shlex.quote(value)}")


def cmd_fields(_args: argparse.Namespace) -> int:
    payload = read_payload()
    assignments = {FIELD_PREFIX + key: value for key, value in flatten(payload).items()}
    assignments.update(derived_fields(payload))
    assignments["HOOK_FIELDS_PARSED"] = "1"
    _print_shell(assignments)
    return 0


def cmd_emit(args: argparse.Namespace) -> int:
    return print_context(args.event, [sys.stdin.read()])


def cmd_system_message(_args: argparse.Namespace) -> int:
    rendered = system_message_json(sys.stdin.read())
    if rendered:
        print(rendered)
    return 0


def cmd_nudge(args: argparse.Namespace) -> int:
    payload = read_payload()
    if quiet_in_subagent(payload):
        # Subagent saves land in the parent's counters; a save never emits a nudge.
        if classify_tool(str(payload.get("tool_name") or "").strip()) == "memory_saves":
            nudge(payload, Path(args.memory_dir))
        return 0
    message = nudge(payload, Path(args.memory_dir))
    if not message:
        return 0
    if args.emit:
        return print_context(event_for(payload, args.emit), [message])
    print(message)
    return 0


def cmd_nudge_summary(args: argparse.Namespace) -> int:
    result = nudge_summary(Path(args.memory_dir), args.session, args.project)
    if result:
        print(result[1])
    return 0


def cmd_memory_trigger(args: argparse.Namespace) -> int:
    payload = read_payload()
    project = args.project or project_of(payload)
    if remote_mode():
        capture = remote_error_capture_runner(hook_call(payload, project))
        messages = memory_trigger(payload, Path(args.state_dir), Path(args.src),
                                  project, runner=capture)
        capture.flush()
    else:
        messages = memory_trigger(payload, Path(args.state_dir), Path(args.src),
                                  project, runner=run_script)
    return print_context(event_for(payload, args.emit), messages)


def check_runtime_version(environ: Any = None) -> bool:
    """Log (structured, hooks.log/stderr only) when these hook copies were installed from another release
    than the checkout they run against. True when they match or no stamp/version is available."""
    env = os.environ if environ is None else environ
    stamped = env.get("TAM_HOOKS_VERSION", "")
    install = env.get("CLAUDE_MEMORY_INSTALL_DIR", "")
    if not stamped or not install:
        return True
    try:
        match = re.search(r'^VERSION\s*=\s*"([^"]+)"', (Path(install) / "src" / "version.py").read_text(
            encoding="utf-8"), re.MULTILINE)
    except OSError:
        return True
    current = match.group(1) if match else ""
    if not current or current == stamped:
        return True
    log(json.dumps({"event": "hook_runtime_version_mismatch", "installed": stamped, "checkout": current,
                    "fix": "run update.sh or `python -m setup_wizard.refresh_hooks`"}))
    return False


def cmd_session_start_context(args: argparse.Namespace) -> int:
    check_runtime_version()
    if remote_mode():
        call = {"payload": {"source": args.source, "session_id": args.session, "cwd": args.cwd},
                "project": args.project, "session_id": args.session or "unknown", "cwd": args.cwd}
        deadline = time.monotonic() + REMOTE_SYNC_BUDGET_SEC
        remote_flush_own(call, deadline)
        remaining = deadline - time.monotonic()
        if remaining > 0:
            _ok, text = remote_post("session-start", call, {}, remaining)
            if text:
                print(text)
        remote_sweep_stale(call, deadline)
        return 0
    text = session_start_context(Path(args.db), args.project, args.cwd, args.session, args.source,
                                 Path(args.recovery_dir))
    if text:
        print(text)
    return 0


def cmd_stop(args: argparse.Namespace) -> int:
    payload = read_payload()
    if not remote_mode() and payload.get("stop_hook_active") is True:
        return 0
    memory_dir = Path(args.memory_dir)
    session = session_of(payload)
    project = args.project or project_of(payload)
    cwd = str(payload.get("cwd") or os.getcwd())
    if remote_mode():
        call = hook_call(payload, project)
        deadline = time.monotonic() + REMOTE_STOP_BUDGET_SEC
        flushed = remote_flush_own(call, deadline)
        if payload.get("stop_hook_active") is True:
            return 0
        notice = None if os.environ.get("MEMORY_NUDGE_DISABLE") == "1" else stop_notice(memory_dir, session)
        prune_nudge_state(memory_dir)
        turn_notice = ""
        if flushed:
            remaining = deadline - time.monotonic()
            if remaining > 0:
                slim = _remote_slim_call(call, "last_assistant_message")
                _ok, turn_notice = remote_post("turn-end", slim, {}, remaining)
        remote_sweep_stale(call, deadline)
        rendered = system_message_json("\n".join(n for n in [notice or "", turn_notice] if n))
        if rendered:
            print(rendered)
        return 0
    write_recovery(Path(args.recovery_dir), session, project, cwd,
                   "Turn ended (snapshot kept in case the session is interrupted)",
                   min_interval_sec=args.min_interval_sec)
    notice = None if os.environ.get("MEMORY_NUDGE_DISABLE") == "1" else stop_notice(memory_dir, session)
    prune_nudge_state(memory_dir)
    notices = [notice or ""]
    if args.src:
        notices.append(ambient_call("turn-end", hook_call(payload, project), memory_dir, Path(args.src)))
        notices.append(ambient_unavailable_notice(memory_dir, session))
    rendered = system_message_json("\n".join(n for n in notices if n))
    if rendered:
        print(rendered)
    return 0


def cmd_session_end(args: argparse.Namespace) -> int:
    payload = read_payload()
    project = args.project or project_of(payload)
    if remote_mode():
        remote_session_end(payload, project, Path(args.src), args.reason_text,
                           time.monotonic() + REMOTE_END_BUDGET_SEC)
        return 0
    for action in session_end(payload, Path(args.src), Path(args.recovery_dir), project, args.reason_text):
        log(f"SESSION_END {project}: {action}")
    return 0


def _run_worker(name: str, job: Callable[[], bool]) -> int:
    # Detached workers have no caller to report to: any failure is logged to hooks.log.
    try:
        job()
    except Exception as exc:  # noqa: BLE001 — top-level worker boundary, logged with type
        log(f"{name} failed: {type(exc).__name__}: {exc}")
        return 1
    return 0


def cmd_save_intent(args: argparse.Namespace) -> int:
    if remote_mode():
        payload = consume_input_file(Path(args.input))
        call = _remote_slim_call(hook_call(payload, project_of(payload)), "prompt", "user_message")
        remote_post("save-intent", call, {}, REMOTE_WRITE_TIMEOUT_SEC)
        return 0
    return _run_worker("save-intent", lambda: save_intent_from(
        consume_input_file(Path(args.input)), Path(args.src), Path(args.db)))


def cmd_capture_observation(args: argparse.Namespace) -> int:
    if remote_mode():
        # No remote command for the extract queue yet; still consume the cached payload.
        consume_input_file(Path(args.input))
        return 0
    return _run_worker("capture-observation", lambda: capture_observation_from(
        consume_input_file(Path(args.input)), Path(args.src), Path(args.memory_dir)))


def cmd_remote_post(_args: argparse.Namespace) -> int:
    """Detached worker: an ordered list of {"command", "call", "payload"}
    requests on stdin -> one POST each, in order (a lone object still works)."""
    raw = sys.stdin.read()
    try:
        data = json.loads(raw) if raw.strip() else []
    except ValueError as exc:
        log(f"remote-post stdin is not JSON: {exc}")
        return 0
    requests = data if isinstance(data, list) else [data]
    for item in requests:
        if not isinstance(item, dict):
            continue
        command = str(item.get("command") or "")
        call = item.get("call")
        post_payload = item.get("payload")
        if command and isinstance(call, dict) and isinstance(post_payload, dict):
            remote_post(command, call, post_payload, REMOTE_WRITE_TIMEOUT_SEC)
    return 0


def cmd_record_action(args: argparse.Namespace) -> int:
    payload = read_payload()
    if remote_mode():
        try:
            remote_record_action(payload, Path(args.src))
        except Exception as exc:  # noqa: BLE001 - a hook must never fail the client
            log(f"remote record-action failed: {type(exc).__name__}: {exc}")
        return 0
    ambient_call("record-action", hook_call(payload, project_of(payload)), Path(args.memory_dir), Path(args.src))
    return 0


def cmd_prompt_context(args: argparse.Namespace) -> int:
    payload = read_payload()
    if remote_mode():
        call = hook_call(payload, project_of(payload))
        deadline = time.monotonic() + REMOTE_SYNC_BUDGET_SEC
        text = ""
        if remote_flush_own(call, deadline):
            remaining = deadline - time.monotonic()
            if remaining > 0:
                slim = _remote_slim_call(call, "prompt", "user_message")
                _ok, text = remote_post("prompt-context", slim, {}, remaining)
        remote_sweep_stale(call, deadline)
        return print_context(event_for(payload, args.emit), [text])
    text = ambient_call("prompt-context", hook_call(payload, project_of(payload)), Path(args.memory_dir),
                        Path(args.src))
    return print_context(event_for(payload, args.emit), [text])


def cmd_rule_reminder(args: argparse.Namespace) -> int:
    payload = read_payload()
    if remote_mode():
        return 0  # no remote command for rule reminders yet
    if quiet_in_subagent(payload):
        return 0
    text = ambient_call("rule-reminder", hook_call(payload, project_of(payload)), Path(args.memory_dir),
                        Path(args.src))
    return print_context(event_for(payload, args.emit), [text])


# ──────────────────────────────────────────────
# Provenance (who wrote a memory): `_ctx` for the server, see src/provenance.py
# ──────────────────────────────────────────────

PROVENANCE_TOOL = re.compile(r"^mcp__(?:plugin_total-agent-memory_)?memory__")
PROVENANCE_AGENT = "claude-code"
PROVENANCE_CTX_KEY = "_ctx"
PROVENANCE_MARKER = "tam-provenance.json"
PROVENANCE_FINGERPRINT_KEYS = ("command", "args", "url")
# Server names Claude Code derives for MCP servers shipped by the TAM plugin itself.
PROVENANCE_PLUGIN_SERVER = re.compile(r"^plugin_total-agent-memory_memory$")


def workspace_of(cwd: str) -> tuple[str, str]:
    """(repo name, path of cwd relative to the repository root); never an absolute path."""
    if not cwd:
        return "", ""
    root = _git(cwd, "rev-parse", "--show-toplevel") if os.path.isdir(cwd) else ""
    if not root:
        return os.path.basename(os.path.normpath(cwd)), ""
    relative = os.path.relpath(os.path.realpath(cwd), os.path.realpath(root))
    return os.path.basename(root), "" if relative == "." else relative.replace(os.sep, "/")


def _json_file(path: Path) -> JsonDict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _fingerprint(entry: object) -> JsonDict:
    if not isinstance(entry, dict):
        return {}
    return {key: entry[key] for key in PROVENANCE_FINGERPRINT_KEYS if entry.get(key)}


def _mcpjson_server_approved(name: str, directory: Path, home: Path, project: object) -> bool:
    """Claude Code starts a server from a project .mcp.json only after approval (enableAllProjectMcpServers or
    enabledMcpjsonServers), and never one listed in disabledMcpjsonServers. The answers live in the project's
    settings files, the user settings and the project entry of ~/.claude.json."""
    sources = [_json_file(directory / ".claude" / "settings.local.json"), _json_file(directory / ".claude" / "settings.json"),
               _json_file(home / ".claude" / "settings.json"), project if isinstance(project, dict) else {}]

    def listed(key: str) -> bool:
        return any(isinstance(source.get(key), list) and name in source[key] for source in sources)

    if listed("disabledMcpjsonServers"):
        return False
    return listed("enabledMcpjsonServers") or any(source.get("enableAllProjectMcpServers") is True for source in sources)


def effective_server(name: str, cwd: str, home: Path) -> JsonDict | None:
    """The registration Claude Code uses for ``name`` at ``cwd``. Precedence follows the documented MCP scope order
    (code.claude.com/docs/en/mcp, "Scope hierarchy and precedence"): local scope, project .mcp.json, user scope.
    Local scope belongs to one project path only, so only the working directory and its repository root are
    looked up there, never their parents. A project .mcp.json server counts only when it is approved."""
    user_config = _json_file(home / ".claude.json")
    directories = [Path(cwd), *Path(cwd).parents] if cwd else []
    projects = user_config.get("projects")
    # ~/.claude.json keys projects by path, with forward slashes on Windows.
    by_path = {os.path.normcase(os.path.normpath(str(key))): value for key, value in projects.items()} \
        if isinstance(projects, dict) else {}
    project_roots = [cwd, _git(cwd, "rev-parse", "--show-toplevel") if os.path.isdir(cwd) else ""] if cwd else []
    scopes: list[Any] = [by_path[key].get("mcpServers") for key in (os.path.normcase(os.path.normpath(root))
                                                                      for root in project_roots if root)
                         if isinstance(by_path.get(key), dict)]
    scopes += [servers for servers, d in ((_json_file(d / ".mcp.json").get("mcpServers"), d) for d in directories)
               if not (isinstance(servers, dict) and name in servers)
               or _mcpjson_server_approved(name, d, home, by_path.get(os.path.normcase(os.path.normpath(str(d)))))]
    scopes.append(user_config.get("mcpServers"))
    for servers in scopes:
        if isinstance(servers, dict) and isinstance(servers.get(name), dict):
            return servers[name]
    return None


def is_tam_server(tool_name: str, cwd: str, home: Path) -> bool:
    """True when the tool belongs to the TAM server: our plugin's, or the "memory" server the installer registered."""
    match = PROVENANCE_TOOL.match(tool_name)
    if not match:
        return False
    server = match.group(0)[len("mcp__"):-len("__")]
    if PROVENANCE_PLUGIN_SERVER.match(server):
        return True
    recorded = _json_file(home / ".claude" / PROVENANCE_MARKER).get("servers")
    expected = _fingerprint(recorded.get(server)) if isinstance(recorded, dict) else {}
    return bool(expected) and _fingerprint(effective_server(server, cwd, home)) == expected


def provenance_input(payload: JsonDict, home: Path | None = None) -> JsonDict | None:
    """The tool input plus ``_ctx`` for a TAM tool call; None for any other call or one that already has it."""
    tool_input = payload.get("tool_input")
    if not isinstance(tool_input, dict) or PROVENANCE_CTX_KEY in tool_input \
            or not is_tam_server(str(payload.get("tool_name") or ""), str(payload.get("cwd") or ""),
                                 home or Path.home()):
        return None
    repo, workspace = workspace_of(str(payload.get("cwd") or ""))
    ctx = {"agent": PROVENANCE_AGENT, "session_id": session_of(payload), "workspace": workspace, "repo": repo,
           "tool_use_id": str(payload.get("tool_use_id") or "")}
    return {**tool_input, PROVENANCE_CTX_KEY: ctx}


def cmd_inject_provenance(_args: argparse.Namespace) -> int:
    updated = provenance_input(read_payload())
    if updated is not None:
        # No permissionDecision: the user's normal permission flow still applies to the (modified) call.
        print(json.dumps({"hookSpecificOutput": {"hookEventName": "PreToolUse", "updatedInput": updated}},
                         ensure_ascii=False))
    return 0


def cmd_codex_notify(args: argparse.Namespace) -> int:
    if remote_mode():
        return 0  # no remote command for Codex capture yet
    raw = args.payload if args.payload is not None else sys.stdin.read()
    if not raw.strip():
        return 0
    notification = parse_payload(raw)
    if not notification:
        return 0
    call = {"payload": notification, "project": "", "session_id": "", "cwd": os.getcwd()}
    ambient_call("codex-notify", call, Path(args.memory_dir), Path(args.src))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tam_hook.py", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("fields").set_defaults(func=cmd_fields)
    p = sub.add_parser("emit")
    p.add_argument("event")
    p.set_defaults(func=cmd_emit)
    sub.add_parser("system-message").set_defaults(func=cmd_system_message)

    p = sub.add_parser("nudge")
    p.add_argument("--memory-dir", required=True)
    p.add_argument("--emit", metavar="DEFAULT_EVENT", help="wrap as additionalContext JSON")
    p.set_defaults(func=cmd_nudge)

    p = sub.add_parser("nudge-summary")
    p.add_argument("--memory-dir", required=True)
    p.add_argument("--session", required=True)
    p.add_argument("--project", required=True)
    p.set_defaults(func=cmd_nudge_summary)

    p = sub.add_parser("memory-trigger")
    p.add_argument("--state-dir", required=True)
    p.add_argument("--src", required=True)
    p.add_argument("--project", default="")
    p.add_argument("--emit", default="PostToolUse", metavar="DEFAULT_EVENT")
    p.set_defaults(func=cmd_memory_trigger)

    p = sub.add_parser("session-start-context")
    p.add_argument("--db", required=True)
    p.add_argument("--project", required=True)
    p.add_argument("--cwd", required=True)
    p.add_argument("--session", default="unknown")
    p.add_argument("--source", default="")
    p.add_argument("--recovery-dir", required=True)
    p.set_defaults(func=cmd_session_start_context)

    p = sub.add_parser("stop")
    p.add_argument("--memory-dir", required=True)
    p.add_argument("--recovery-dir", required=True)
    p.add_argument("--project", default="")
    p.add_argument("--min-interval-sec", type=int, default=STOP_SNAPSHOT_MIN_INTERVAL_SEC)
    p.add_argument("--src", default="", help="TAM src/; enables the rule check and the turn summary")
    p.set_defaults(func=cmd_stop)

    p = sub.add_parser("session-end")
    p.add_argument("--src", required=True)
    p.add_argument("--recovery-dir", required=True)
    p.add_argument("--project", default="")
    p.add_argument("--reason-text", required=True)
    p.set_defaults(func=cmd_session_end)

    p = sub.add_parser("save-intent", help="detached worker: payload file -> intents table")
    p.add_argument("--src", required=True)
    p.add_argument("--db", required=True)
    p.add_argument("--input", required=True, help="cached payload file, deleted after reading")
    p.set_defaults(func=cmd_save_intent)

    sub.add_parser("remote-post", help="detached worker: remote POST request on stdin"
                   ).set_defaults(func=cmd_remote_post)

    p = sub.add_parser("capture-observation", help="detached worker: payload file -> extract queue")
    p.add_argument("--src", required=True)
    p.add_argument("--memory-dir", required=True)
    p.add_argument("--input", required=True, help="cached payload file, deleted after reading")
    p.set_defaults(func=cmd_capture_observation)

    p = sub.add_parser("record-action", help="PostToolUse(Failure): payload -> agent_actions")
    p.add_argument("--src", required=True)
    p.add_argument("--memory-dir", required=True)
    p.set_defaults(func=cmd_record_action)

    p = sub.add_parser("prompt-context", help="UserPromptSubmit: rules + background as additionalContext")
    p.add_argument("--src", required=True)
    p.add_argument("--memory-dir", required=True)
    p.add_argument("--emit", default="UserPromptSubmit", metavar="DEFAULT_EVENT")
    p.set_defaults(func=cmd_prompt_context)

    sub.add_parser("inject-provenance", help="PreToolUse(mcp__memory__*): add _ctx to the tool input"
                   ).set_defaults(func=cmd_inject_provenance)
    p = sub.add_parser("rule-reminder", help="PreToolUse: matching rules as additionalContext")
    p.add_argument("--src", required=True)
    p.add_argument("--memory-dir", required=True)
    p.add_argument("--emit", default="PreToolUse", metavar="DEFAULT_EVENT")
    p.set_defaults(func=cmd_rule_reminder)

    p = sub.add_parser("codex-notify", help="Codex notify JSON (argument or stdin) -> action log")
    p.add_argument("--src", required=True)
    p.add_argument("--memory-dir", required=True)
    p.add_argument("--payload", default=None, help="the notification JSON; read from stdin when omitted")
    p.set_defaults(func=cmd_codex_notify)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")
    stdin_reconfigure = getattr(sys.stdin, "reconfigure", None)
    if stdin_reconfigure is not None:
        # utf-8-sig: Windows PowerShell 5.1 can prefix what it pipes to a native command with a BOM.
        stdin_reconfigure(encoding="utf-8-sig", errors="replace")
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except OSError as exc:
        # Hooks must never break the session; the failure is still reported.
        log(f"{args.command} failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
