"""End of a turn (Stop hook): rule compliance check and turn summary auto-save.

The turn is every action the session logged since the previous Stop
(``ambient.state``). Two things happen with it:

* compliance — :meth:`rule_surfaces.RuleSurfacer.followcheck` judges the
  surfaces shown to the session against the actions. Prohibitions count: a
  "never X" rule whose X shows up in an action was *ignored* (violated), not
  followed. Each newly decided verdict lands in the rule's heuristic counters
  only (:func:`rule_surfaces.record_auto_outcome`: auto_success_count /
  auto_fail_count). Explicit ratings, success_rate, success_lower_bound and
  auto-suspension never see it. Loading or surfacing a rule never touches
  fire_count; only a confirmed follow does.
* summary — when the turn changed something (an edit, a shell command), a
  short secret-redacted summary is stored as an auto-saved ``fact``
  (``source='auto'``, tags ``auto-saved`` + ``turn-autosave``) and queued for
  embedding. ``TAM_TURN_AUTOSAVE=false`` turns it off; ``tam forget-auto``
  removes these records.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from ambient.actions import (
    EDIT_DETAIL_SEPARATOR,
    EDIT_TOOLS,
    SHELL_TOOLS,
    STATUS_FAILED,
    ActionLog,
    AgentAction,
    table_exists,
)
from ambient.settings import AmbientSettings
from ambient.state import SessionState
from memory_core.timestamps import format_utc
from rule_surfaces import ActionEvidence, FollowVerdict, RuleSurfacer, record_auto_outcome
from secret_redaction import redact_secrets

LOGGER = logging.getLogger(__name__)

AUTO_SOURCE: Final = "auto"
AUTO_SAVED_TAG: Final = "auto-saved"
TURN_AUTOSAVE_TAG: Final = "turn-autosave"
# SessionEnd's auto_session_save.py marks its records with this tag.
SESSION_AUTOSAVE_TAG: Final = "session-autosave"
AUTO_TAGS: Final = (TURN_AUTOSAVE_TAG, SESSION_AUTOSAVE_TAG)
SUMMARY_TYPE: Final = "fact"
# Below the 0.8 retention bar: never-recalled summaries age out with memory_forget.
SUMMARY_CONFIDENCE: Final = 0.5
REQUEST_CHARS: Final = 300
OUTCOME_CHARS: Final = 400
SUMMARY_MAX_CHARS: Final = 1500
MAX_LISTED_FILES: Final = 6
MAX_LISTED_COMMANDS: Final = 3
COMMAND_CHARS: Final = 120
SESSION_LABEL_CHARS: Final = 8
EMBEDDING_REASON: Final = "turn summary saved by the Stop hook"
SQLITE_TIMEOUT_SEC: Final = 2.0


@dataclass(frozen=True)
class ComplianceResult:
    verdicts: tuple[FollowVerdict, ...]
    violations: tuple[FollowVerdict, ...]
    followed: tuple[FollowVerdict, ...]

    def notice(self, rule_texts: dict[int, str]) -> str:
        """User-facing line about the rules this turn appears to have broken; '' when none."""
        if not self.violations:
            return ""
        parts = [f"#{v.rule_id} \"{rule_texts.get(v.rule_id, '')}\" (matched: {', '.join(v.forbidden_hits)})"
                 for v in self.violations]
        return "TAM rule check: this turn appears to break " + "; ".join(parts) + "."


def evidence_from(actions: Sequence[AgentAction]) -> list[ActionEvidence]:
    return [ActionEvidence(action.evidence_text(), action.at, action.tool) for action in actions]


def check_compliance(db: sqlite3.Connection, session_id: str, actions: Sequence[AgentAction],
                     now: datetime | None = None) -> ComplianceResult:
    """Judge the session's rule surfaces against ``actions``; count new verdicts in the auto_* counters."""
    if not actions:
        return ComplianceResult((), (), ())
    verdicts = RuleSurfacer(db).followcheck(session_id, evidence_from(actions), persist=True)
    stamp = format_utc(now or datetime.now(UTC))
    decided = [v for v in verdicts if v.persisted and v.verdict in ("followed", "ignored")]
    for verdict in decided:
        record_auto_outcome(db, verdict.rule_id, verdict.verdict == "followed", stamp)
    violations = tuple(v for v in decided if v.verdict == "ignored")
    followed = tuple(v for v in decided if v.verdict == "followed")
    return ComplianceResult(tuple(verdicts), violations, followed)


def rule_previews(db: sqlite3.Connection, rule_ids: Sequence[int], chars: int = COMMAND_CHARS) -> dict[int, str]:
    if not rule_ids:
        return {}
    marks = ",".join("?" for _ in rule_ids)
    rows = db.execute(f"SELECT id, content FROM rules WHERE id IN ({marks})", list(rule_ids)).fetchall()
    return {int(rule_id): _clip(" ".join(str(content or "").split()), chars) for rule_id, content in rows}


# ── turn summary ───────────────────────────────────────────────────────────


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: max(0, limit - 1)].rstrip() + "…"


def _one_line(text: str | None) -> str:
    return " ".join(str(text or "").split())


def significant(actions: Sequence[AgentAction]) -> bool:
    """A turn worth a summary changed something: an edit, a shell command or a Codex turn."""
    return any(action.mutating for action in actions)


def summary_text(*, project: str, session_id: str, request: str, actions: Sequence[AgentAction],
                 outcome: str) -> str:
    counts = Counter(action.tool for action in actions)
    failed = sum(1 for action in actions if action.status == STATUS_FAILED)
    files = list(dict.fromkeys(action.summary.split(EDIT_DETAIL_SEPARATOR, 1)[0]
                               for action in actions if action.tool in EDIT_TOOLS and action.summary))
    commands = [_clip(action.summary, COMMAND_CHARS) for action in actions
                if action.tool in SHELL_TOOLS and action.summary]
    lines = [f"Turn summary (auto-saved; project {project}, session {session_id[:SESSION_LABEL_CHARS]})"]
    if request:
        lines.append(f"Request: {_clip(_one_line(request), REQUEST_CHARS)}")
    tally = ", ".join(f"{tool} x{count}" for tool, count in counts.most_common())
    lines.append(f"Actions: {tally}" + (f" ({failed} failed)" if failed else ""))
    if files:
        lines.append("Files: " + ", ".join(files[:MAX_LISTED_FILES])
                     + (f" (+{len(files) - MAX_LISTED_FILES} more)" if len(files) > MAX_LISTED_FILES else ""))
    if commands:
        lines.append("Commands: " + "; ".join(commands[-MAX_LISTED_COMMANDS:]))
    if outcome:
        lines.append(f"Outcome: {_clip(_one_line(outcome), OUTCOME_CHARS)}")
    return _clip(redact_secrets("\n".join(lines))[0], SUMMARY_MAX_CHARS)


def last_request(db: sqlite3.Connection, session_id: str) -> str:
    """The newest prompt the UserPromptSubmit hook stored for the session (``intents``)."""
    try:
        row = db.execute("SELECT prompt FROM intents WHERE session_id = ? ORDER BY id DESC LIMIT 1",
                         (session_id,)).fetchone()
    except sqlite3.OperationalError as exc:
        LOGGER.info(json.dumps({"event": "turn_request_unavailable", "error": str(exc)}))
        return ""
    return str(row[0]) if row else ""


def save_turn_summary(db: sqlite3.Connection, *, content: str, project: str, session_id: str, cwd: str,
                      now: datetime | None = None) -> int | None:
    """Store ``content`` as an auto-saved fact and queue it for embedding; None when it is a repeat."""
    if db.execute("SELECT 1 FROM knowledge WHERE session_id = ? AND source = ? AND content = ? LIMIT 1",
                  (session_id, AUTO_SOURCE, content)).fetchone():
        return None
    stamp = format_utc(now or datetime.now(UTC))
    tags = json.dumps([AUTO_SAVED_TAG, TURN_AUTOSAVE_TAG, project])
    context = f"Project: {project}, CWD: {cwd}, Session: {session_id}, Turn ended: {stamp}"
    cursor = db.execute(
        "INSERT INTO knowledge (session_id, type, content, context, project, tags, status, confidence, source, "
        "created_at, last_confirmed, recall_count) VALUES (?, ?, ?, ?, ?, ?, 'active', ?, ?, ?, ?, 0)",
        (session_id, SUMMARY_TYPE, content, redact_secrets(context)[0], project, tags, SUMMARY_CONFIDENCE,
         AUTO_SOURCE, stamp, stamp))
    record_id = int(cursor.lastrowid or 0)
    if db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='embedding_pending'").fetchone():
        from embedding_backfill import EmbeddingBackfill

        EmbeddingBackfill(db).enqueue(record_id, EMBEDDING_REASON)
    db.commit()
    return record_id


# ── orchestration ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TurnEndResult:
    actions: int
    compliance: ComplianceResult | None
    saved_id: int | None
    notice: str


def run_turn_end(*, db_path: Path, memory_dir: Path, session_id: str, project: str, cwd: str,
                 assistant_message: str, settings: AmbientSettings, now: datetime | None = None) -> TurnEndResult:
    """Compliance check + turn summary for the actions since the previous Stop; moves the turn boundary."""
    moment = now or datetime.now(UTC)
    if not db_path.is_file() or not (settings.rule_compliance or settings.turn_autosave):
        return TurnEndResult(0, None, None, "")
    state = SessionState.load(memory_dir, session_id)
    db = sqlite3.connect(str(db_path), timeout=SQLITE_TIMEOUT_SEC)
    try:
        if not table_exists(db, "agent_actions"):
            LOGGER.info(json.dumps({"event": "turn_end_skipped", "reason": "agent_actions missing"}))
            return TurnEndResult(0, None, None, "")
        actions = ActionLog(db).session_actions(session_id, state.last_stop)
        compliance = None
        notice = ""
        if settings.rule_compliance and actions and table_exists(db, "rule_surfaces"):
            compliance = check_compliance(db, session_id, actions, moment)
            notice = compliance.notice(rule_previews(db, [v.rule_id for v in compliance.violations]))
        saved_id = None
        if settings.turn_autosave and significant(actions):
            content = summary_text(project=project, session_id=session_id, request=last_request(db, session_id),
                                   actions=actions, outcome=assistant_message)
            saved_id = save_turn_summary(db, content=content, project=project, session_id=session_id, cwd=cwd,
                                         now=moment)
    finally:
        db.close()
    state.last_stop = format_utc(moment)
    state.save()
    return TurnEndResult(len(actions), compliance, saved_id, notice)
