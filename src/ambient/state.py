"""Per-session ambient state: what was already injected, when the last turn ended.

One JSON file per client session under ``<memory dir>/state/ambient-<session>.json``
(pruned with the other hook state after ``STATE_MAX_AGE_DAYS``). It lets every
hook invocation — a fresh process each time — dedupe against the ones before it:

* ``records``   memory ids already shown to this session (newest last, capped);
* ``rules``     rule ids already shown with the prompt;
* ``reminders`` rule id -> epoch seconds of its last PreToolUse reminder;
* ``last_stop`` canonical UTC text of the previous Stop: the turn boundary;
* ``corrections`` short hashes of prompts already captured as corrections.

A lost or corrupt file only costs a repeated injection, never a failure.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

LOGGER = logging.getLogger(__name__)

STATE_SUBDIR: Final = "state"
STATE_PREFIX: Final = "ambient-"
STATE_MAX_AGE_DAYS: Final = 7
MAX_REMEMBERED_RECORDS: Final = 500
MAX_REMEMBERED_RULES: Final = 200
MAX_REMEMBERED_CORRECTIONS: Final = 100
SECONDS_PER_DAY: Final = 86_400


def safe_name(value: str) -> str:
    """Filename-safe session id; identical to the hook runtime's state naming."""
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in value) or "unknown"


def state_path(memory_dir: Path, session_id: str) -> Path:
    return Path(memory_dir) / STATE_SUBDIR / f"{STATE_PREFIX}{safe_name(session_id)}.json"


@dataclass
class SessionState:
    path: Path
    records: list[int] = field(default_factory=list)
    rules: list[int] = field(default_factory=list)
    reminders: dict[int, float] = field(default_factory=dict)
    last_stop: str = ""
    corrections: list[str] = field(default_factory=list)

    @classmethod
    def load(cls, memory_dir: Path, session_id: str) -> SessionState:
        path = state_path(memory_dir, session_id)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return cls(path)
        except (OSError, ValueError) as exc:
            LOGGER.warning(json.dumps({"event": "ambient_state_unreadable", "path": str(path), "error": str(exc)}))
            return cls(path)
        if not isinstance(data, dict):
            return cls(path)
        return cls(
            path,
            records=_ints(data.get("records")),
            rules=_ints(data.get("rules")),
            reminders={int(k): float(v) for k, v in (data.get("reminders") or {}).items()
                       if str(k).lstrip("-").isdigit() and isinstance(v, int | float)},
            last_stop=str(data.get("last_stop") or ""),
            corrections=[str(item) for item in data.get("corrections") or [] if isinstance(item, str)],
        )

    def remember_records(self, ids: Iterable[int]) -> None:
        self.records = _append_capped(self.records, ids, MAX_REMEMBERED_RECORDS)

    def remember_rules(self, ids: Iterable[int]) -> None:
        self.rules = _append_capped(self.rules, ids, MAX_REMEMBERED_RULES)

    def remember_reminders(self, ids: Iterable[int], at: float) -> None:
        for rule_id in ids:
            self.reminders[int(rule_id)] = at
        if len(self.reminders) > MAX_REMEMBERED_RULES:
            newest = sorted(self.reminders.items(), key=lambda item: item[1])[-MAX_REMEMBERED_RULES:]
            self.reminders = dict(newest)

    def remember_correction(self, key: str) -> None:
        self.corrections = [item for item in self.corrections if item != key][-(MAX_REMEMBERED_CORRECTIONS - 1):]
        self.corrections.append(key)

    def reminded_since(self, cutoff: float) -> set[int]:
        return {rule_id for rule_id, at in self.reminders.items() if at >= cutoff}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"records": self.records, "rules": self.rules,
                   "reminders": {str(k): v for k, v in self.reminders.items()}, "last_stop": self.last_stop,
                   "corrections": self.corrections}
        tmp = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(tmp, self.path)


CLAIM_PREFIX: Final = "ambient-tooluse-"
CLAIM_SUFFIX: Final = ".claim"
MAX_CLAIM_ID_CHARS: Final = 120


def claim_tool_use(memory_dir: Path, session_id: str, tool_use_id: str) -> bool:
    """True for the first hook process that claims this tool call, False for every later one.

    A plugin and a settings.json install both fire for the same call, in parallel; the exclusive create
    makes exactly one of them record it. A claim that cannot be made (read-only state dir) counts as won,
    so a broken state directory never silences the action log.
    """
    name = f"{CLAIM_PREFIX}{safe_name(session_id)}-{safe_name(tool_use_id)[:MAX_CLAIM_ID_CHARS]}{CLAIM_SUFFIX}"
    path = Path(memory_dir) / STATE_SUBDIR / name
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        os.close(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600))
    except FileExistsError:
        return False
    except OSError as exc:
        LOGGER.warning(json.dumps({"event": "ambient_claim_failed", "path": str(path), "error": str(exc)}))
    return True


def prune_states(memory_dir: Path, max_age_days: int = STATE_MAX_AGE_DAYS, now: float | None = None) -> int:
    """Remove ambient state files untouched for ``max_age_days``; returns how many."""
    cutoff = (time.time() if now is None else now) - max_age_days * SECONDS_PER_DAY
    removed = 0
    state_dir = Path(memory_dir) / STATE_SUBDIR
    for path in [*state_dir.glob(f"{STATE_PREFIX}*.json"), *state_dir.glob(f"{CLAIM_PREFIX}*{CLAIM_SUFFIX}")]:
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
                removed += 1
        except FileNotFoundError:
            continue
    return removed


def _ints(raw: object) -> list[int]:
    if not isinstance(raw, list):
        return []
    return [int(item) for item in raw if isinstance(item, int) and not isinstance(item, bool)]


def _append_capped(current: list[int], new: Iterable[int], cap: int) -> list[int]:
    # Newest last: a repeated id moves to its latest position.
    merged = [*current, *(int(item) for item in new)]
    newest_first = list(dict.fromkeys(reversed(merged)))
    return list(reversed(newest_first[:cap]))
