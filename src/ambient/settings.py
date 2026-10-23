"""Ambient-memory settings, resolved the way the local dashboard stores them.

Hooks run outside the MCP server, so ``local_settings.apply`` never ran for
them. :meth:`AmbientSettings.load` applies the same precedence itself: a value
saved in ``<memory dir>/settings.json`` wins over the process environment,
which wins over the default here. None of these settings is a secret, so the
plain ``values`` section is all that is read (no Fernet key needed).
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

LOGGER = logging.getLogger(__name__)

SETTINGS_FILE: Final = "settings.json"
SETTINGS_FORMAT_VERSION: Final = 1

ACTION_LOG_ENV: Final = "TAM_ACTION_LOG"
ACTION_LOG_RETENTION_ENV: Final = "TAM_ACTION_LOG_RETENTION_DAYS"
AMBIENT_RECALL_ENV: Final = "TAM_AMBIENT_RECALL"
AMBIENT_TOKEN_BUDGET_ENV: Final = "TAM_AMBIENT_TOKEN_BUDGET"
AMBIENT_LATENCY_ENV: Final = "TAM_AMBIENT_LATENCY_MS"
RULE_REMINDERS_ENV: Final = "TAM_RULE_REMINDERS"
RULE_COMPLIANCE_ENV: Final = "TAM_RULE_COMPLIANCE"
TURN_AUTOSAVE_ENV: Final = "TAM_TURN_AUTOSAVE"
CORRECTION_CAPTURE_ENV: Final = "TAM_CORRECTION_CAPTURE"

DEFAULT_ACTION_LOG: Final = True
DEFAULT_ACTION_LOG_RETENTION_DAYS: Final = 30
DEFAULT_AMBIENT_RECALL: Final = True
# About 1.6k characters: a few rules plus three short records per prompt.
DEFAULT_AMBIENT_TOKEN_BUDGET: Final = 400
# The UserPromptSubmit hook gets 5 s from Claude Code; ambient recall gives up
# long before that, so a slow store never delays the prompt noticeably.
DEFAULT_AMBIENT_LATENCY_MS: Final = 1500
DEFAULT_RULE_REMINDERS: Final = True
DEFAULT_RULE_COMPLIANCE: Final = True
DEFAULT_TURN_AUTOSAVE: Final = True
DEFAULT_CORRECTION_CAPTURE: Final = True

MIN_TOKEN_BUDGET: Final = 50
MAX_TOKEN_BUDGET: Final = 5000
MIN_LATENCY_MS: Final = 100
MAX_LATENCY_MS: Final = 4000

_TRUE: Final = frozenset({"true", "1", "yes", "on"})
_FALSE: Final = frozenset({"false", "0", "no", "off"})

SETTING_KEYS: Final = (ACTION_LOG_ENV, ACTION_LOG_RETENTION_ENV, AMBIENT_RECALL_ENV, AMBIENT_TOKEN_BUDGET_ENV,
                       AMBIENT_LATENCY_ENV, RULE_REMINDERS_ENV, RULE_COMPLIANCE_ENV, TURN_AUTOSAVE_ENV,
                       CORRECTION_CAPTURE_ENV)


@dataclass(frozen=True)
class AmbientSettings:
    action_log: bool = DEFAULT_ACTION_LOG
    action_log_retention_days: int = DEFAULT_ACTION_LOG_RETENTION_DAYS
    ambient_recall: bool = DEFAULT_AMBIENT_RECALL
    token_budget: int = DEFAULT_AMBIENT_TOKEN_BUDGET
    latency_ms: int = DEFAULT_AMBIENT_LATENCY_MS
    rule_reminders: bool = DEFAULT_RULE_REMINDERS
    rule_compliance: bool = DEFAULT_RULE_COMPLIANCE
    turn_autosave: bool = DEFAULT_TURN_AUTOSAVE
    correction_capture: bool = DEFAULT_CORRECTION_CAPTURE

    @classmethod
    def load(cls, memory_dir: Path, environ: Mapping[str, str] | None = None) -> AmbientSettings:
        env = os.environ if environ is None else environ
        values = {**{key: env[key] for key in SETTING_KEYS if env.get(key, "").strip()},
                  **saved_values(Path(memory_dir))}
        return cls(
            action_log=_flag(values, ACTION_LOG_ENV, DEFAULT_ACTION_LOG),
            action_log_retention_days=_bounded(values, ACTION_LOG_RETENTION_ENV,
                                               DEFAULT_ACTION_LOG_RETENTION_DAYS, 1, None),
            ambient_recall=_flag(values, AMBIENT_RECALL_ENV, DEFAULT_AMBIENT_RECALL),
            token_budget=_bounded(values, AMBIENT_TOKEN_BUDGET_ENV, DEFAULT_AMBIENT_TOKEN_BUDGET,
                                  MIN_TOKEN_BUDGET, MAX_TOKEN_BUDGET),
            latency_ms=_bounded(values, AMBIENT_LATENCY_ENV, DEFAULT_AMBIENT_LATENCY_MS,
                                MIN_LATENCY_MS, MAX_LATENCY_MS),
            rule_reminders=_flag(values, RULE_REMINDERS_ENV, DEFAULT_RULE_REMINDERS),
            rule_compliance=_flag(values, RULE_COMPLIANCE_ENV, DEFAULT_RULE_COMPLIANCE),
            turn_autosave=_flag(values, TURN_AUTOSAVE_ENV, DEFAULT_TURN_AUTOSAVE),
            correction_capture=_flag(values, CORRECTION_CAPTURE_ENV, DEFAULT_CORRECTION_CAPTURE),
        )


def saved_values(memory_dir: Path) -> dict[str, str]:
    """Ambient keys saved from the local dashboard; {} when there is no usable settings file."""
    path = memory_dir / SETTINGS_FILE
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        LOGGER.warning(json.dumps({"event": "ambient_settings_unreadable", "path": str(path), "error": str(exc)}))
        return {}
    if not isinstance(data, dict) or data.get("version") != SETTINGS_FORMAT_VERSION:
        LOGGER.warning(json.dumps({"event": "ambient_settings_unsupported", "path": str(path)}))
        return {}
    values = data.get("values")
    if not isinstance(values, dict):
        return {}
    return {key: value for key, value in values.items() if key in SETTING_KEYS and isinstance(value, str)}


def _flag(values: Mapping[str, str], key: str, default: bool) -> bool:
    raw = values.get(key, "").strip().lower()
    if not raw:
        return default
    if raw in _TRUE:
        return True
    if raw in _FALSE:
        return False
    LOGGER.warning(json.dumps({"event": "ambient_setting_invalid", "key": key, "value": raw}))
    return default


def _bounded(values: Mapping[str, str], key: str, default: int, low: int, high: int | None) -> int:
    raw = values.get(key, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        LOGGER.warning(json.dumps({"event": "ambient_setting_invalid", "key": key, "value": raw}))
        return default
    if value < low:
        return low
    return value if high is None else min(value, high)
