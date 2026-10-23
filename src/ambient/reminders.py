"""Rule reminder before a Bash call or a file edit (PreToolUse).

The command line, or the edited path plus the new text, is matched against the
active rules with :mod:`rule_surfaces`. A rule is reminded when the action
carries what a prohibition forbids ("never use sqlite" vs ``sqlite3 db.sqlite``)
or shares at least ``REMINDER_MIN_MATCHED`` tokens with it, one of them
distinctive. Each reminder is a recorded surface, so the Stop hook can tell
whether the rule was then followed; the same rule is not reminded to the same
session again within ``REMINDER_COOLDOWN_SEC``. The text goes to the model as
additionalContext; no permission decision is ever made.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

from ambient.actions import EDIT_TOOLS, SHELL_TOOLS
from ambient.prompt_context import rule_line
from ambient.state import SessionState
from rule_surfaces import FORBIDDEN_MIN_OVERLAP, RankedRule, RuleSurfacer, forbidden_terms, tokenize

REMINDER_LIMIT: Final = 2
REMINDER_COOLDOWN_SEC: Final = 900
REMINDER_MIN_MATCHED: Final = 2
QUERY_MAX_CHARS: Final = 4000
SOURCE_PRE_TOOL: Final = "ambient_pre_tool"
REMINDED_TOOLS: Final = SHELL_TOOLS | EDIT_TOOLS


@dataclass(frozen=True)
class Reminder:
    text: str
    rule_ids: tuple[int, ...]
    forbidden_hits: dict[int, tuple[str, ...]]


def action_query(tool: str, tool_input: Mapping[str, Any] | None) -> str:
    """The text a rule is matched against: the command, or the target path plus the new text."""
    data = tool_input or {}
    if tool in SHELL_TOOLS:
        text = str(data.get("command") or "")
    else:
        pieces = [data.get("file_path"), data.get("notebook_path"), data.get("new_string"), data.get("content"),
                  data.get("new_source")]
        if isinstance(data.get("edits"), list):
            pieces += [edit.get("new_string") for edit in data["edits"] if isinstance(edit, Mapping)]
        text = "\n".join(piece for piece in pieces if isinstance(piece, str) and piece)
    return text[:QUERY_MAX_CHARS]


def forbidden_hit(item: RankedRule, query_tokens: set[str]) -> tuple[str, ...]:
    forbidden = forbidden_terms(item.rule.get("content"))
    if not forbidden:
        return ()
    hits = forbidden & query_tokens
    return tuple(sorted(hits)) if len(hits) >= min(FORBIDDEN_MIN_OVERLAP, len(forbidden)) else ()


def build_reminder(db: sqlite3.Connection, *, tool: str, tool_input: Mapping[str, Any] | None, project: str,
                   session_id: str, state: SessionState, now: float) -> Reminder:
    """Rules to remind before ``tool`` runs; records their surfaces and updates ``state`` (not saved)."""
    if tool not in REMINDED_TOOLS:
        return Reminder("", (), {})
    query = action_query(tool, tool_input)
    if not query.strip():
        return Reminder("", (), {})
    surfacer = RuleSurfacer(db)
    ranked = surfacer.rank_rules(query, project, lexical_only=True,
                                 exclude_rule_ids=state.reminded_since(now - REMINDER_COOLDOWN_SEC))
    query_tokens = set(tokenize(query))
    chosen: list[tuple[RankedRule, tuple[str, ...]]] = []
    for item in ranked:
        hits = forbidden_hit(item, query_tokens)
        if hits or (len(item.matched) >= REMINDER_MIN_MATCHED and item.distinctive):
            chosen.append((item, hits))
    # Prohibitions the action seems to break come first.
    chosen.sort(key=lambda pair: not pair[1])
    chosen = chosen[:REMINDER_LIMIT]
    if not chosen:
        return Reminder("", (), {})
    surfacer.record_surfaces([item for item, _ in chosen], query, project, session_id, SOURCE_PRE_TOOL)
    state.remember_reminders((item.rule_id for item, _ in chosen), now)

    subject = "this command" if tool in SHELL_TOOLS else "this edit"
    lines = [f"[TAM rule reminder before {subject}]"]
    for item, hits in chosen:
        line = rule_line(item)
        if hits:
            line += f"\n  ! {subject} contains what the rule forbids: {', '.join(hits)}"
        lines.append(line)
    lines.append("Follow these rules, or say why one does not apply here.")
    return Reminder("\n".join(lines), tuple(item.rule_id for item, _ in chosen),
                    {item.rule_id: hits for item, hits in chosen if hits})
