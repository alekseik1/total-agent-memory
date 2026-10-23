"""Per-project daily budget for high-importance saves.

``importance=critical`` multiplies a record's recall score (x1.5 by default).
Uncapped, an agent that marks everything critical flattens the ranking back to
nothing. Each project gets ``DEFAULT_CRITICAL_DAILY_BUDGET`` critical saves per
UTC day; beyond that a save is stored as ``high``, then (when a high budget is
configured) as ``medium``. The caller gets a note, and ``importance_budget_log``
keeps the requested level next to the applied one.

Rules and conventions are exempt: they are the records that are supposed to
win every ranking. Decisions count like any other type.

Settings (environment / local settings):
  MEMORY_CRITICAL_DAILY_BUDGET  critical saves per project per UTC day (0 = unlimited)
  MEMORY_HIGH_DAILY_BUDGET      high saves per project per UTC day (0 = unlimited, default)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

from memory_core.timestamps import format_utc

DEFAULT_CRITICAL_DAILY_BUDGET = 5
DEFAULT_HIGH_DAILY_BUDGET = 0
UNLIMITED = 0
CRITICAL_BUDGET_ENV = "MEMORY_CRITICAL_DAILY_BUDGET"
HIGH_BUDGET_ENV = "MEMORY_HIGH_DAILY_BUDGET"
EXEMPT_TYPES = frozenset({"convention", "rule"})
# Each capped level and the level a save over its budget drops to.
DOWNGRADE = {"critical": "high", "high": "medium"}
BUDGET_ENV = {"critical": CRITICAL_BUDGET_ENV, "high": HIGH_BUDGET_ENV}
BUDGET_DEFAULT = {"critical": DEFAULT_CRITICAL_DAILY_BUDGET, "high": DEFAULT_HIGH_DAILY_BUDGET}


class _Connection(Protocol):
    def execute(self, sql: str, parameters: Any = (), /) -> Any: ...


@dataclass(frozen=True)
class BudgetDecision:
    requested: str
    applied: str
    budget: int
    used: int

    @property
    def clamped(self) -> bool:
        return self.applied != self.requested

    def note(self) -> str:
        return (f"importance lowered from {self.requested} to {self.applied}: this project already has "
                f"{self.used} {self.requested} save(s) today (daily budget {self.budget}). Reserve "
                f"{self.requested} for migration-blocking decisions and incidents; "
                f"conventions are exempt.")

    def as_dict(self) -> dict[str, Any]:
        return {"requested": self.requested, "applied": self.applied, "budget": self.budget,
                "used_today": self.used, "note": self.note()}


def daily_budget(level: str, environ: dict[str, str] | None = None) -> int:
    """Budget for ``level`` from the environment; invalid values raise (misconfiguration)."""
    env = os.environ if environ is None else environ
    raw = env.get(BUDGET_ENV[level], "").strip()
    if not raw:
        return BUDGET_DEFAULT[level]
    value = int(raw)
    if value < 0:
        raise ValueError(f"{BUDGET_ENV[level]} must be >= 0, got {value}")
    return value


def day_start(now: datetime) -> str:
    """Canonical timestamp of the UTC midnight that starts ``now``'s day."""
    moment = now.astimezone(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    return format_utc(moment)


def used_today(db: _Connection, project: str, level: str, since: str) -> int:
    """Saves of ``level`` in ``project`` since ``since`` that count against the budget."""
    exempt = sorted(EXEMPT_TYPES)
    marks = ",".join("?" for _ in exempt)
    return int(db.execute(
        f"SELECT COUNT(*) FROM knowledge WHERE importance = ? AND project = ? AND created_at >= ? "
        f"AND type NOT IN ({marks})", (level, project, since, *exempt)).fetchone()[0])


def apply_budget(db: _Connection, *, project: str, ktype: str, importance: str,
                 now: datetime | None = None, environ: dict[str, str] | None = None) -> BudgetDecision:
    """Importance to store for a new record of ``ktype`` requested at ``importance``."""
    level = importance
    if ktype in EXEMPT_TYPES or level not in DOWNGRADE:
        return BudgetDecision(requested=importance, applied=importance, budget=UNLIMITED, used=0)
    since = day_start(now or datetime.now(UTC))
    first: BudgetDecision | None = None
    while level in DOWNGRADE:
        budget = daily_budget(level, environ)
        if budget == UNLIMITED:
            break
        used = used_today(db, project, level, since)
        if used < budget:
            break
        if first is None:
            first = BudgetDecision(requested=importance, applied=DOWNGRADE[level], budget=budget, used=used)
        level = DOWNGRADE[level]
    if first is None:
        return BudgetDecision(requested=importance, applied=importance, budget=UNLIMITED, used=0)
    return BudgetDecision(requested=importance, applied=level, budget=first.budget, used=first.used)


def record_clamp(db: _Connection, decision: BudgetDecision, *, knowledge_id: int, project: str,
                 ktype: str, created_at: str) -> None:
    """Audit row for a lowered importance (caller commits)."""
    db.execute(
        "INSERT INTO importance_budget_log "
        "(knowledge_id, project, ktype, requested, applied, budget, used, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (knowledge_id, project, ktype, decision.requested, decision.applied, decision.budget,
         decision.used, created_at))


def today_summary(db: _Connection, now: datetime | None = None,
                  environ: dict[str, str] | None = None) -> dict[str, Any]:
    """Budgets and today's clamps per project, for memory_stats."""
    since = day_start(now or datetime.now(UTC))
    rows = db.execute(
        "SELECT project, requested, COUNT(*) FROM importance_budget_log WHERE created_at >= ? "
        "GROUP BY project, requested ORDER BY project, requested", (since,)).fetchall()
    clamped: dict[str, dict[str, int]] = {}
    for project, requested, count in rows:
        clamped.setdefault(project, {})[requested] = int(count)
    return {"budgets": {level: daily_budget(level, environ) for level in DOWNGRADE},
            "exempt_types": sorted(EXEMPT_TYPES), "clamped_today": clamped}
