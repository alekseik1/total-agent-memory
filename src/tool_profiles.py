"""Opt-in reduced MCP tool surfaces (``TAM_TOOL_PROFILE``).

Every tools/list answer costs the client model context on every turn. The full
catalogue (79 tools with long descriptions) is the default and stays exactly as
the published plugin and its ``memory-protocol`` skill expect it. Two smaller
profiles are opt-in:

* ``minimal`` — the memory loop itself. Read first (``session_init``,
  ``self_rules_context``, ``memory_recall``, ``memory_get``), write
  (``memory_save``, ``memory_update``, ``memory_delete``, ``kg_add_fact``),
  close the session (``session_end``) and the two pre-flight / capture tools the
  global agent rules call on every edit and every failure (``file_context``,
  ``learn_error``), plus ``self_rules`` so learned rules can be fired and rated.
* ``lean`` — ``minimal`` plus the tools agents reach for in ordinary work:
  cross-project analogies, workflow prediction, structured decisions, timelines,
  tags, history, relations, temporal-graph queries, episodes and insights.
  Admin, maintenance, evaluation and benchmark tools stay out.

Filtering only changes what tools/list advertises. A hidden tool stays callable,
so prompts written against the full catalogue keep working.

Both reduced profiles serve a one-line description with an inline signature in
place of the long prose, and drop per-parameter ``description`` texts from the
input schemas; types, enums, defaults and ``required`` stay intact.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

LOGGER = logging.getLogger(__name__)

PROFILE_ENV: Final = "TAM_TOOL_PROFILE"
FULL: Final = "full"
LEAN: Final = "lean"
MINIMAL: Final = "minimal"
DEFAULT_PROFILE: Final = FULL
PROFILE_NAMES: Final = (FULL, LEAN, MINIMAL)

MINIMAL_TOOLS: Final[frozenset[str]] = frozenset({
    "memory_recall", "memory_save", "memory_get", "memory_update", "memory_delete",
    "session_init", "session_end", "self_rules_context", "self_rules",
    "file_context", "learn_error", "kg_add_fact",
})

LEAN_TOOLS: Final[frozenset[str]] = MINIMAL_TOOLS | frozenset({
    "analogize", "workflow_predict", "workflow_track", "self_error_log", "self_insight",
    "save_decision", "memory_timeline", "memory_search_by_tag", "memory_history",
    "memory_relate", "memory_context_build", "kg_at", "kg_timeline", "kg_invalidate_fact",
    "memory_episode_save", "memory_episode_recall",
})

# One line per tool: an inline signature (required parameters bare, optional ones
# with `?` or `=default`) and what the tool is for. tests/test_tool_profiles.py
# checks every signature against the live input schema.
SHORT_DESCRIPTIONS: Final[Mapping[str, str]] = {
    "memory_recall": "memory_recall(query, project?, type=all, limit=10, mode=search, detail=full) — "
                     "hybrid search (BM25 + semantic + graph) over all past decisions, solutions, facts and "
                     "lessons. Call before answering about past work.",
    "memory_save": "memory_save(type, content, project=general, context?, tags?, importance=medium, "
                   "supersede=false) — store a decision/fact/solution/lesson/convention; put the why in context.",
    "memory_get": "memory_get(ids, detail=full) — full records by id (max 50), e.g. after a truncated recall hit.",
    "memory_update": "memory_update(new_content, id?, find?, project?, reason?) — replace a record with a new "
                     "version; the old one is kept in its history.",
    "memory_delete": "memory_delete(id, hard=false) — soft-delete a record; hard=true erases it and its versions.",
    "session_init": "session_init(project=general, mark_consumed=true) — previous session summary, next steps "
                    "and pitfalls. Call at session start.",
    "session_end": "session_end(session_id, summary?, next_steps?, pitfalls?, project=general) — save the session "
                   "summary that the next session_init returns.",
    "self_rules_context": "self_rules_context(project=general, categories?, phase?) — active behaviour rules "
                          "learned from past errors. Load at session start.",
    "self_rules": "self_rules(action, id?, content?, category?, project=general) — "
                  "list/fire/rate/suspend/activate/retire/add_manual a behaviour rule.",
    "file_context": "file_context(path, project?, limit=20) — past errors, hot spots and risk score for a file. "
                    "Call before editing it.",
    "learn_error": "learn_error(file, error, root_cause, fix, pattern, project=general, severity=medium) — record "
                   "a reproducible failure; repeated patterns become rules.",
    "kg_add_fact": "kg_add_fact(subject, predicate, object, project=general, valid_from?, "
                   "invalidate_previous=true) — add a temporal knowledge-graph fact, retiring the previous object.",
    "analogize": "analogize(text, exclude_project?, limit=10) — similar past solutions, lessons and decisions, "
                 "optionally from other projects only.",
    "workflow_predict": "workflow_predict(trigger?, workflow_id?, project?) — success probability and duration "
                        "of a known workflow.",
    "workflow_track": "workflow_track(workflow_id, outcome, duration_ms?, notes?) — record how a workflow ended.",
    "self_error_log": "self_error_log(category, description, fix?, project=general, severity=medium) — log an "
                      "agent mistake for self-improvement.",
    "self_insight": "self_insight(action, content?, id?, category?, project=general) — "
                    "add/upvote/downvote/edit/list/promote an insight distilled from errors.",
    "save_decision": "save_decision(title, options, criteria_matrix, selected, rationale, discarded?, project?) — "
                     "store a structured decision with its options and criteria.",
    "memory_timeline": "memory_timeline(query?, sessions_ago?, date_from?, date_to?, project?, limit=5) — what "
                       "happened in past sessions, by session or date range.",
    "memory_search_by_tag": "memory_search_by_tag(tag, project?) — records carrying a tag.",
    "memory_history": "memory_history(id) — version history of a record.",
    "memory_relate": "memory_relate(from_id, to_id, type) — link two records "
                     "(causal/solution/context/related/contradicts).",
    "memory_context_build": "memory_context_build(query, project?, max_tokens=4000) — token-budgeted context "
                            "from graph, episodes, skills and memories for a task.",
    "kg_at": "kg_at(timestamp?, subject?, predicate?, object?, project?) — knowledge-graph facts valid at a "
             "point in time.",
    "kg_timeline": "kg_timeline(subject, predicate?, project?) — how the facts about a subject changed over time.",
    "kg_invalidate_fact": "kg_invalidate_fact(subject, predicate, object, at?, reason?) — mark a fact as no "
                          "longer true.",
    "memory_episode_save": "memory_episode_save(narrative, outcome, key_insight?, concepts?, project=general) — "
                           "store a task episode (breakthrough/failure/routine/discovery).",
    "memory_episode_recall": "memory_episode_recall(query?, concepts?, outcome?, project?, limit=10) — past "
                             "episodes by topic or outcome.",
}

PROFILES: Final[Mapping[str, frozenset[str] | None]] = {
    FULL: None,  # None = every registered tool, descriptions untouched
    LEAN: LEAN_TOOLS,
    MINIMAL: MINIMAL_TOOLS,
}

_SIGNATURE = re.compile(r"^(?P<name>[a-z_]+)\((?P<params>[^)]*)\) — \S")


class ToolProfileError(ValueError):
    """An unknown profile name or a profile that names tools the server does not register."""


@dataclass(frozen=True)
class ToolProfile:
    name: str
    tools: frozenset[str] | None

    @property
    def is_full(self) -> bool:
        return self.tools is None

    def exposes(self, tool: str) -> bool:
        return self.tools is None or tool in self.tools


def resolve_profile(environ: Mapping[str, str]) -> ToolProfile:
    """The profile selected by ``TAM_TOOL_PROFILE``; unset or blank means full.

    An unknown value raises instead of falling back: a typo must not silently
    hand the model all 79 tools (or none).
    """
    raw = environ.get(PROFILE_ENV, "").strip().lower()
    name = raw or DEFAULT_PROFILE
    if name not in PROFILES:
        raise ToolProfileError(
            f"{PROFILE_ENV}={environ.get(PROFILE_ENV)!r} is not a tool profile; "
            f"expected one of: {', '.join(PROFILE_NAMES)}"
        )
    return ToolProfile(name=name, tools=PROFILES[name])


def validate_against(registered: Iterable[str]) -> None:
    """Fail when a profile or a short description names a tool the server does not register."""
    known = set(registered)
    problems = []
    for name, tools in PROFILES.items():
        missing = sorted((tools or frozenset()) - known)
        if missing:
            problems.append(f"profile {name!r} names unknown tools: {missing}")
    reduced = set().union(*(tools for tools in PROFILES.values() if tools is not None))
    undescribed = sorted(reduced - set(SHORT_DESCRIPTIONS))
    if undescribed:
        problems.append(f"tools without a short description: {undescribed}")
    stray = sorted(set(SHORT_DESCRIPTIONS) - known)
    if stray:
        problems.append(f"short descriptions for unknown tools: {stray}")
    if problems:
        raise ToolProfileError("; ".join(problems))


def signature_params(description: str) -> tuple[str, ...]:
    """Parameter names of the inline signature that opens a short description."""
    match = _SIGNATURE.match(description)
    if match is None:
        raise ToolProfileError(f"short description lacks a 'name(params) — summary' signature: {description!r}")
    params = match.group("params").strip()
    if not params:
        return ()
    return tuple(re.split(r"[?=]", part.strip(), maxsplit=1)[0] for part in params.split(","))


def strip_descriptions(schema: Any) -> Any:
    """A JSON schema without ``description`` annotations; parameter names are never touched."""
    if isinstance(schema, list):
        return [strip_descriptions(item) for item in schema]
    if not isinstance(schema, dict):
        return schema
    stripped: dict[str, Any] = {}
    for key, value in schema.items():
        if key == "description":
            continue
        if key in ("properties", "patternProperties", "$defs", "definitions") and isinstance(value, dict):
            stripped[key] = {name: strip_descriptions(sub) for name, sub in value.items()}
        else:
            stripped[key] = strip_descriptions(value)
    return stripped


def _schema_field(tool: Any) -> str:
    """mcp SDK 1.x names the Tool field ``inputSchema``, 2.x ``input_schema``."""
    return "input_schema" if "input_schema" in type(tool).model_fields else "inputSchema"


def apply_profile(tools: Sequence[Any], profile: ToolProfile) -> list[Any]:
    """The catalogue a client sees under ``profile``. The full profile returns the tools unchanged."""
    if profile.is_full:
        return list(tools)
    validate_against(tool.name for tool in tools)
    shaped = []
    for tool in tools:
        if not profile.exposes(tool.name):
            continue
        field = _schema_field(tool)
        shaped.append(tool.model_copy(update={
            "description": SHORT_DESCRIPTIONS[tool.name],
            field: strip_descriptions(getattr(tool, field)),
        }))
    return shaped


def note_call(tool: str, profile: ToolProfile) -> None:
    """Debug trace for a call to a tool the active profile does not advertise; the call still runs."""
    if not profile.exposes(tool):
        LOGGER.debug("hidden tool called: %s (profile=%s)", tool, profile.name)
