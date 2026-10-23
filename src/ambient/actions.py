"""Agent action log: what the agent did, captured by client hooks (table ``agent_actions``).

One row per tool call (Claude Code PostToolUse / PostToolUseFailure) or per
Codex turn (notify hook). A row keeps a short, secret-redacted summary — the
command line, or the file path plus the start of the new text — never the full
input or any output. The Stop hook reads a session's rows back as evidence for
the rule compliance check and the turn summary.

Retention is bounded twice on every write: rows older than the configured
number of days go, and so does everything beyond ``MAX_ROWS``.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Final, Literal, Protocol

from memory_core.timestamps import format_utc
from secret_redaction import redact_secrets

LOGGER = logging.getLogger(__name__)

CLIENT_CLAUDE: Final = "claude-code"
CLIENT_CODEX: Final = "codex"
CODEX_TURN_TOOL: Final = "codex_turn"

Status = Literal["ok", "failed", "interrupted"]
STATUS_OK: Final = "ok"
STATUS_FAILED: Final = "failed"
STATUS_INTERRUPTED: Final = "interrupted"

SUMMARY_MAX_CHARS: Final = 400
# How much of an edit's new text is kept: enough to see what was introduced.
DETAIL_SNIPPET_CHARS: Final = 240
# Between an edited path and the start of its new text in a summary.
EDIT_DETAIL_SEPARATOR: Final = " :: "
MAX_ID_CHARS: Final = 200
MAX_TOOL_CHARS: Final = 120
MAX_ROWS: Final = 100_000
# Rows read back for one turn; a turn with more tool calls keeps the newest.
DEFAULT_TURN_LIMIT: Final = 500
# Hard erasure matches a record against action summaries by this much of its text.
ERASE_NEEDLE_CHARS: Final = 80
ERASE_MIN_CHARS: Final = 12
SECONDS_PER_DAY: Final = 86_400

EDIT_TOOLS: Final = frozenset({"Edit", "Write", "MultiEdit", "NotebookEdit"})
SHELL_TOOLS: Final = frozenset({"Bash"})
MUTATING_TOOLS: Final = EDIT_TOOLS | SHELL_TOOLS | {CODEX_TURN_TOOL}
# Read-only lookups are the bulk of tool calls and carry no evidence of what was done.
UNLOGGED_TOOLS: Final = frozenset({"Read", "Glob", "Grep", "LS", "TodoWrite", "TodoRead"})
# Tools whose input is a path or a pattern worth keeping verbatim.
_LOCATOR_KEYS: Final = ("file_path", "notebook_path", "path", "pattern", "url", "query", "description")
MCP_TOOL_PREFIX: Final = "mcp__"
_COLUMNS: Final = "ts, client, session_id, project, tool, summary, status, exit_code"


class Connection(Protocol):
    def execute(self, sql: str, parameters: Sequence[Any] = ..., /) -> Any: ...

    def commit(self) -> None: ...


@dataclass(frozen=True)
class AgentAction:
    ts: str
    client: str
    session_id: str
    project: str
    tool: str
    summary: str
    status: Status = STATUS_OK
    exit_code: int | None = None

    @property
    def at(self) -> datetime:
        return datetime.fromisoformat(self.ts).astimezone(UTC)

    @property
    def mutating(self) -> bool:
        return self.tool in MUTATING_TOOLS

    def evidence_text(self) -> str:
        """The action as rule-check evidence: tool name plus its summary."""
        return f"{self.tool} {self.summary}".strip()


def _one_line(text: str) -> str:
    return " ".join(str(text).split())


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: max(0, limit - 1)].rstrip() + "…"


def _snippet(value: Any) -> str:
    return _clip(_one_line(value), DETAIL_SNIPPET_CHARS) if isinstance(value, str) and value.strip() else ""


def summarize_tool_input(tool: str, tool_input: Mapping[str, Any] | None) -> str:
    """Short description of one tool call; redacted and at most ``SUMMARY_MAX_CHARS``."""
    data = tool_input or {}
    if tool in SHELL_TOOLS:
        text = _one_line(data.get("command") or "")
    elif tool in EDIT_TOOLS:
        target = str(data.get("file_path") or data.get("notebook_path") or "")
        if tool == "MultiEdit" and isinstance(data.get("edits"), list):
            pieces = [edit.get("new_string") for edit in data["edits"] if isinstance(edit, Mapping)]
            detail = _snippet(" ".join(piece for piece in pieces if isinstance(piece, str)))
        else:
            detail = _snippet(data.get("new_string") or data.get("content") or data.get("new_source") or "")
        text = f"{target}{EDIT_DETAIL_SEPARATOR}{detail}" if detail else target
    else:
        # MCP tools keep only argument names: their values can be whole memory records.
        locator = "" if tool.startswith(MCP_TOOL_PREFIX) else next(
            (str(data[key]) for key in _LOCATOR_KEYS if isinstance(data.get(key), str) and data[key]), "")
        text = _one_line(locator) if locator else ("keys: " + ", ".join(sorted(map(str, data))) if data else "")
    return _clip(redact_secrets(text)[0], SUMMARY_MAX_CHARS)


def status_for(failed: bool, interrupted: bool) -> Status:
    if interrupted:
        return STATUS_INTERRUPTED
    return STATUS_FAILED if failed else STATUS_OK


def parse_exit_code(raw: str | int | None) -> int | None:
    if raw is None or isinstance(raw, bool):
        return None
    try:
        return int(str(raw).strip())
    except ValueError:
        return None


def build_action(*, tool: str, tool_input: Mapping[str, Any] | None, session_id: str, project: str,
                 status: Status, exit_code: int | None, client: str = CLIENT_CLAUDE,
                 now: datetime | None = None) -> AgentAction | None:
    """An action from hook fields; None without a tool name or for a read-only lookup tool."""
    name = _one_line(tool)[:MAX_TOOL_CHARS]
    if not name or name in UNLOGGED_TOOLS:
        return None
    return AgentAction(
        ts=format_utc(now or datetime.now(UTC)),
        client=client,
        session_id=(session_id or "")[:MAX_ID_CHARS],
        project=(project or "general")[:MAX_ID_CHARS],
        tool=name,
        summary=summarize_tool_input(name, tool_input),
        status=status,
        exit_code=exit_code,
    )


def _row_action(row: Sequence[Any]) -> AgentAction:
    status = str(row[6]) if row[6] in (STATUS_OK, STATUS_FAILED, STATUS_INTERRUPTED) else STATUS_OK
    return AgentAction(str(row[0]), str(row[1]), str(row[2]), str(row[3]), str(row[4]), str(row[5] or ""),
                       status, None if row[7] is None else int(row[7]))  # type: ignore[arg-type]


class ActionLog:
    """``agent_actions`` operations over an injected connection; writes commit."""

    def __init__(self, db: Connection):
        self._db = db

    def record(self, action: AgentAction, *, retention_days: int, max_rows: int = MAX_ROWS,
               now: datetime | None = None) -> int:
        """Insert ``action`` and prune; returns the new row id."""
        cursor = self._db.execute(
            f"INSERT INTO agent_actions ({_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (action.ts, action.client, action.session_id, action.project, action.tool, action.summary,
             action.status, action.exit_code))
        row_id = int(cursor.lastrowid or 0)
        self.prune(retention_days, max_rows=max_rows, now=now, commit=False)
        self._db.commit()
        return row_id

    def prune(self, retention_days: int, *, max_rows: int = MAX_ROWS, now: datetime | None = None,
              commit: bool = True) -> int:
        """Delete rows older than ``retention_days`` and the oldest beyond ``max_rows``."""
        if retention_days < 1:
            raise ValueError("retention_days must be at least 1")
        cutoff = format_utc((now or datetime.now(UTC)) - timedelta(days=retention_days))
        removed = int(self._db.execute("DELETE FROM agent_actions WHERE ts < ?", (cutoff,)).rowcount or 0)
        boundary = self._db.execute(
            "SELECT id FROM agent_actions ORDER BY id DESC LIMIT 1 OFFSET ?", (max_rows,)).fetchone()
        if boundary is not None:
            removed += int(self._db.execute(
                "DELETE FROM agent_actions WHERE id <= ?", (int(boundary[0]),)).rowcount or 0)
        if commit:
            self._db.commit()
        return removed

    def session_actions(self, session_id: str, since: str | None = None,
                        limit: int = DEFAULT_TURN_LIMIT) -> list[AgentAction]:
        """``session_id``'s actions after ``since`` (canonical UTC text), oldest first."""
        rows = self._db.execute(
            f"SELECT {_COLUMNS} FROM agent_actions WHERE session_id = ? AND ts > ? "
            "ORDER BY ts DESC, id DESC LIMIT ?", (session_id, since or "", int(limit))).fetchall()
        return [_row_action(row) for row in reversed(rows)]

    def count(self, project: str | None = None) -> int:
        if project is None:
            return int(self._db.execute("SELECT COUNT(*) FROM agent_actions").fetchone()[0])
        return int(self._db.execute("SELECT COUNT(*) FROM agent_actions WHERE project = ?",
                                    (project,)).fetchone()[0])

    def delete(self, project: str | None = None) -> int:
        """Delete the whole log, or one project's part of it; commits."""
        if project is None:
            removed = self._db.execute("DELETE FROM agent_actions").rowcount
        else:
            removed = self._db.execute("DELETE FROM agent_actions WHERE project = ?", (project,)).rowcount
        self._db.commit()
        return int(removed or 0)

    def erase_mentions(self, texts: Iterable[str]) -> int:
        """Delete rows whose summary quotes one of ``texts`` (or is quoted by it). No commit.

        Used by hard erasure: a command or an edit can carry a record's text.
        """
        removed = 0
        for text in {t.strip() for t in texts if t and len(t.strip()) >= ERASE_MIN_CHARS}:
            needle = text[:ERASE_NEEDLE_CHARS]
            removed += int(self._db.execute(
                "DELETE FROM agent_actions WHERE instr(summary, ?) > 0 "
                "OR (length(summary) >= ? AND instr(?, summary) > 0)",
                (needle, ERASE_MIN_CHARS, text)).rowcount or 0)
        return removed

    def export(self, project: str | None = None) -> list[dict[str, Any]]:
        """Rows as dicts for ``memory_export``, oldest first."""
        where, params = ("WHERE project = ?", (project,)) if project else ("", ())
        rows = self._db.execute(f"SELECT {_COLUMNS} FROM agent_actions {where} ORDER BY ts, id", params).fetchall()
        names = [name.strip() for name in _COLUMNS.split(",")]
        return [dict(zip(names, row, strict=True)) for row in rows]


def table_exists(db: Connection, name: str) -> bool:
    return db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None


def log_event(event: str, **fields: Any) -> None:
    """One structured INFO line (JSON) for hook-side observability."""
    LOGGER.info(json.dumps({"event": event, **fields}, ensure_ascii=False, default=str))
