"""Codex CLI turns into the action log (and, when enabled, a turn summary).

Codex has no per-tool hooks; its ``notify`` program receives one JSON argument
per finished agent turn::

    {"type": "agent-turn-complete", "thread-id": "...", "turn-id": "...",
     "cwd": "/path/to/project", "input-messages": ["..."],
     "last-assistant-message": "..."}

Each ``agent-turn-complete`` becomes one ``codex_turn`` row in
``agent_actions`` (client ``codex``, session = thread id) and, with
``TAM_TURN_AUTOSAVE`` on, one auto-saved turn summary. Other notification
types are ignored.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from ambient.actions import (
    CLIENT_CODEX,
    CODEX_TURN_TOOL,
    STATUS_OK,
    ActionLog,
    AgentAction,
    summarize_tool_input,
    table_exists,
)
from ambient.settings import AmbientSettings
from ambient.turn_end import save_turn_summary, summary_text
from memory_core.timestamps import format_utc

LOGGER = logging.getLogger(__name__)

TURN_COMPLETE: Final = "agent-turn-complete"
UNKNOWN_SESSION: Final = "codex-unknown"
SQLITE_TIMEOUT_SEC: Final = 2.0
MAX_ID_CHARS: Final = 200


class CodexNotificationError(ValueError):
    """The notify argument is not a Codex notification object."""


@dataclass(frozen=True)
class CodexTurn:
    session_id: str
    project: str
    cwd: str
    request: str
    outcome: str


@dataclass(frozen=True)
class CodexCaptureResult:
    recorded: bool
    saved_id: int | None


def parse_notification(raw: str, fallback_cwd: str = "") -> CodexTurn | None:
    """The turn in a notify payload string; None for other notification types."""
    try:
        data = json.loads(raw)
    except ValueError as exc:
        raise CodexNotificationError(f"notification is not JSON: {exc}") from None
    if not isinstance(data, Mapping):
        raise CodexNotificationError("notification must be a JSON object")
    return turn_from(data, fallback_cwd)


def turn_from(data: Mapping[str, object], fallback_cwd: str = "") -> CodexTurn | None:
    """The turn in a parsed notification; None for other notification types."""
    if data.get("type") != TURN_COMPLETE:
        return None
    cwd = str(data.get("cwd") or fallback_cwd or "")
    messages = data.get("input-messages")
    request = "\n".join(str(m) for m in messages if str(m).strip()) if isinstance(messages, list) else ""
    session = str(data.get("thread-id") or data.get("turn-id") or UNKNOWN_SESSION)[:MAX_ID_CHARS]
    return CodexTurn(session_id=session, project=project_for(cwd), cwd=cwd, request=request,
                     outcome=str(data.get("last-assistant-message") or ""))


def project_for(cwd: str) -> str:
    override = os.environ.get("MEMORY_PROJECT")
    if override:
        return override
    return os.path.basename(cwd.rstrip("/\\")) or "general"


def turn_action(turn: CodexTurn, now: datetime | None = None) -> AgentAction:
    summary = summarize_tool_input(CODEX_TURN_TOOL, {"description": turn.request} if turn.request else {})
    return AgentAction(ts=format_utc(now or datetime.now(UTC)), client=CLIENT_CODEX, session_id=turn.session_id,
                       project=turn.project, tool=CODEX_TURN_TOOL, summary=summary, status=STATUS_OK)


def capture_turn(db_path: Path, turn: CodexTurn, settings: AmbientSettings,
                 now: datetime | None = None) -> CodexCaptureResult:
    """Log the turn and, when enabled and there is something to say, save its summary."""
    if not db_path.is_file():
        return CodexCaptureResult(False, None)
    moment = now or datetime.now(UTC)
    action = turn_action(turn, moment)
    db = sqlite3.connect(str(db_path), timeout=SQLITE_TIMEOUT_SEC)
    try:
        recorded = False
        if settings.action_log and table_exists(db, "agent_actions"):
            ActionLog(db).record(action, retention_days=settings.action_log_retention_days, now=moment)
            recorded = True
        saved_id = None
        if settings.turn_autosave and turn.request.strip() and turn.outcome.strip():
            content = summary_text(project=turn.project, session_id=turn.session_id, request=turn.request,
                                   actions=[action], outcome=turn.outcome)
            saved_id = save_turn_summary(db, content=content, project=turn.project, session_id=turn.session_id,
                                         cwd=turn.cwd, now=moment)
    finally:
        db.close()
    return CodexCaptureResult(recorded, saved_id)

