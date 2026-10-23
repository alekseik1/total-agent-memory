"""The MCP ``instructions`` string this server hands clients at initialize.

Clients put it into the model's system context once per session, so it has to
be short: what TAM is, read before acting, what to save and what not to, and the
date the server started so relative dates in queries resolve to absolute ones.
``TAM_MCP_INSTRUCTIONS=false`` turns it off.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from datetime import date
from typing import Final

LOGGER = logging.getLogger(__name__)

INSTRUCTIONS_ENV: Final = "TAM_MCP_INSTRUCTIONS"
MAX_INSTRUCTIONS_CHARS: Final = 1500
_TRUE: Final = frozenset({"true", "1", "yes", "on"})
_FALSE: Final = frozenset({"false", "0", "no", "off"})
DEFAULT_ENABLED: Final = True

_TEMPLATE: Final = (
    "total-agent-memory (TAM) is long-term memory shared across sessions and projects: decisions, "
    "solutions, facts, lessons and conventions, a temporal knowledge graph and behaviour rules learned "
    "from past mistakes.\n"
    "Read first: at session start call session_init (last summary, next steps, pitfalls) and "
    "self_rules_context. Before answering about past work, a project's stack or earlier choices, call "
    "memory_recall instead of guessing. Before editing a file, file_context shows its known problems.\n"
    "Save: decisions with the reason in context, working solutions to non-obvious problems, lessons from "
    "mistakes (learn_error for reproducible failures), stable facts and conventions. When a stored fact "
    "changed, use memory_update instead of saving a contradicting copy. Close with session_end "
    "(summary, next_steps).\n"
    "Do not save: ephemeral state (current branch, task lists, temporary paths, command output, "
    "in-progress reasoning), secrets, or what the repository already records.\n"
    "Server start date: {today}. Resolve relative dates (yesterday, last week) against it and store "
    "absolute dates."
)


def enabled(environ: Mapping[str, str]) -> bool:
    raw = environ.get(INSTRUCTIONS_ENV, "").strip().lower()
    if not raw:
        return DEFAULT_ENABLED
    if raw in _TRUE:
        return True
    if raw in _FALSE:
        return False
    LOGGER.warning("%s=%r is not true/false; using the default (%s)", INSTRUCTIONS_ENV, raw, DEFAULT_ENABLED)
    return DEFAULT_ENABLED


def build(environ: Mapping[str, str], today: date) -> str | None:
    """The instructions text, or None when the setting turns it off."""
    if not enabled(environ):
        return None
    text = _TEMPLATE.format(today=today.isoformat())
    if len(text) > MAX_INSTRUCTIONS_CHARS:
        raise ValueError(f"MCP instructions are {len(text)} chars; keep them under {MAX_INSTRUCTIONS_CHARS}")
    return text
