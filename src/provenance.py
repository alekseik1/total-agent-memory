"""Who wrote each memory: agent, session and repo-relative workspace, without touching any tool schema.

Two sources feed one record per written memory (table ``memory_provenance``, migration 046):

* ``_ctx`` in the tool arguments, injected by the installer-managed PreToolUse hook
  (``hooks/inject-provenance.*``, ``hookSpecificOutput.updatedInput``). The server strips it before
  the arguments are validated, logged or handed to a tool, so no schema ever sees it.
* The MCP request ``_meta``: Claude Code sends ``claudecode/toolUseId``; Codex sends
  ``x-codex-turn-metadata``; other clients may send a conversation or thread id.

Both are labels supplied by the client, not authentication. Workspaces are stored relative to the
repository root; an absolute path is never stored.

server.py wiring (one import, two calls) is described in the 14.10.0 integration report.
"""
from __future__ import annotations

import json
import logging
import re
import sqlite3
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from secret_redaction import redact_secrets

LOGGER = logging.getLogger(__name__)

CTX_KEY = "_ctx"
TABLE = "memory_provenance"
SQLITE_TIMEOUT_SEC = 2.0
MAX_FIELD_CHARS = 200
MAX_AGENT_CHARS = 40
SOURCE_HOOK = "hook"
SOURCE_META = "meta"
SOURCE_BOTH = "hook+meta"

AGENT_CLAUDE = "claude-code"
AGENT_CODEX = "codex"
AGENT_UNKNOWN = "unknown"
META_CLAUDE_TOOL_USE = "claudecode/toolUseId"
META_CODEX_TURN = "x-codex-turn-metadata"
META_SESSION_KEYS = ("session_id", "sessionId", "thread_id", "threadId", "conversation_id", "conversationId")
META_TURN_KEYS = ("turn_id", "turnId")
META_TOOL_USE_KEYS = ("tool_use_id", "toolUseId")

_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_AGENT_SAFE = re.compile(r"[^a-z0-9._-]")
_WINDOWS_DRIVE = re.compile(r"^[A-Za-z]:")


@dataclass(frozen=True)
class CallProvenance:
    """Provenance of one tool call; empty strings mean "not supplied"."""

    agent: str = ""
    session_id: str = ""
    workspace: str = ""
    repo: str = ""
    tool_use_id: str = ""
    turn_id: str = ""
    source: str = ""

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


def _text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    # Redact before truncating: a cut could otherwise leave a credential prefix the patterns no longer match.
    return redact_secrets(_CONTROL.sub("", value).strip())[0][:MAX_FIELD_CHARS]


def _agent(value: Any) -> str:
    cleaned = _AGENT_SAFE.sub("", _text(value).lower())[:MAX_AGENT_CHARS]
    return cleaned


def relative_workspace(value: Any) -> str:
    """A repository-relative path, '' for the root; anything absolute or escaping the repo is dropped."""
    text = _text(value).replace("\\", "/")
    if not text or text == ".":
        return ""
    if text.startswith("/") or _WINDOWS_DRIVE.match(text) or ".." in text.split("/"):
        return ""
    return text.strip("/")


def sanitize_ctx(raw: Any) -> CallProvenance | None:
    """The hook's ``_ctx`` object as a clean record; None when it is not an object or carries nothing."""
    if not isinstance(raw, Mapping):
        return None
    ctx = CallProvenance(
        agent=_agent(raw.get("agent")), session_id=_text(raw.get("session_id")),
        workspace=relative_workspace(raw.get("workspace")), repo=_text(raw.get("repo")),
        tool_use_id=_text(raw.get("tool_use_id")), turn_id=_text(raw.get("turn_id")), source=SOURCE_HOOK)
    return ctx if any((ctx.agent, ctx.session_id, ctx.repo, ctx.tool_use_id, ctx.turn_id)) else None


def _mapping(value: Any) -> Mapping[str, Any]:
    """A request _meta (dict, pydantic model with extras, or a JSON string) as a mapping."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return {}
    if isinstance(value, Mapping):
        return value
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        dumped = dump(by_alias=True)
        return dumped if isinstance(dumped, Mapping) else {}
    return {}


def _first(mapping: Mapping[str, Any], keys: Iterable[str]) -> str:
    for key in keys:
        text = _text(mapping.get(key))
        if text:
            return text
    return ""


def from_request_meta(request_meta: Any) -> CallProvenance | None:
    """What the MCP request ``_meta`` says about the caller; None when it says nothing useful."""
    meta = _mapping(request_meta)
    if not meta:
        return None
    agent = ""
    tool_use_id = _text(meta.get(META_CLAUDE_TOOL_USE))
    session_id = _first(meta, META_SESSION_KEYS)
    turn_id = _first(meta, META_TURN_KEYS)
    repo = ""
    if tool_use_id:
        agent = AGENT_CLAUDE
    codex = _mapping(meta.get(META_CODEX_TURN))
    if META_CODEX_TURN in meta:
        agent = AGENT_CODEX
        session_id = session_id or _first(codex, META_SESSION_KEYS)
        turn_id = turn_id or _first(codex, META_TURN_KEYS)
        tool_use_id = tool_use_id or _first(codex, META_TOOL_USE_KEYS)
        workspaces = codex.get("workspaces")
        if isinstance(workspaces, Mapping) and workspaces:
            # Codex reports absolute workspace roots; only the directory name is kept.
            repo = Path(str(next(iter(workspaces)))).name[:MAX_FIELD_CHARS]
    ctx = CallProvenance(agent=agent, session_id=session_id, repo=_text(repo), tool_use_id=tool_use_id,
                         turn_id=turn_id, source=SOURCE_META)
    return ctx if any((ctx.agent, ctx.session_id, ctx.repo, ctx.tool_use_id, ctx.turn_id)) else None


def merge(hook: CallProvenance | None, meta: CallProvenance | None) -> CallProvenance | None:
    """Hook values describe the workspace and session; the request _meta is authoritative for the call id."""
    if hook is None or meta is None:
        return hook or meta
    return CallProvenance(
        agent=hook.agent or meta.agent, session_id=hook.session_id or meta.session_id,
        workspace=hook.workspace, repo=hook.repo or meta.repo,
        tool_use_id=meta.tool_use_id or hook.tool_use_id, turn_id=hook.turn_id or meta.turn_id,
        source=SOURCE_BOTH)


def split_call(arguments: Any, request_meta: Any = None) -> tuple[Any, CallProvenance | None]:
    """``arguments`` without ``_ctx`` (a copy; the input is untouched) and the call's provenance.

    Call this before the arguments are validated, redacted, logged or passed on. Arguments that are not
    an object are returned unchanged.
    """
    hook = None
    if isinstance(arguments, Mapping) and CTX_KEY in arguments:
        hook = sanitize_ctx(arguments[CTX_KEY])
        arguments = {key: value for key, value in arguments.items() if key != CTX_KEY}
    return arguments, merge(hook, from_request_meta(request_meta))


def record(db_path: Path | str, tool: str, provenance: CallProvenance | None,
           knowledge_ids: Iterable[int] | None) -> int:
    """Store one row per written record; returns how many. A problem is logged and never fails the tool call."""
    ids = sorted({int(i) for i in knowledge_ids or () if isinstance(i, int) and not isinstance(i, bool)})
    if provenance is None or not ids:
        return 0
    now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    try:
        db = sqlite3.connect(str(db_path), timeout=SQLITE_TIMEOUT_SEC)
        try:
            if db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (TABLE,)).fetchone() is None:
                LOGGER.warning(json.dumps({"event": "provenance_not_recorded", "reason": "migration 046 not applied"}))
                return 0
            db.executemany(
                f"INSERT INTO {TABLE} (knowledge_id, tool, agent, session_id, workspace, repo, tool_use_id, "
                "turn_id, source, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [(i, _text(tool), provenance.agent, provenance.session_id, provenance.workspace, provenance.repo,
                  provenance.tool_use_id, provenance.turn_id, provenance.source, now) for i in ids])
            db.commit()
        finally:
            db.close()
    except sqlite3.Error as exc:
        LOGGER.warning(json.dumps({"event": "provenance_not_recorded", "error": f"{type(exc).__name__}: {exc}"}))
        return 0
    return len(ids)


def provenance_for(db: sqlite3.Connection, knowledge_id: int) -> list[dict[str, Any]]:
    """Every provenance row of one record, oldest first."""
    cursor = db.execute(
        f"SELECT tool, agent, session_id, workspace, repo, tool_use_id, turn_id, source, created_at FROM {TABLE} "
        "WHERE knowledge_id = ? ORDER BY id", (knowledge_id,))
    names = [column[0] for column in cursor.description]
    return [dict(zip(names, row, strict=True)) for row in cursor.fetchall()]
