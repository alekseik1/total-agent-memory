"""Unit tests for the per-project daily importance budget (src/importance_budget.py)."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta

import pytest

import importance_budget as ib

NOON = datetime(2026, 10, 13, 12, 0, tzinfo=UTC)


@pytest.fixture
def db():
    conn = sqlite3.connect(":memory:")
    conn.executescript("""
        CREATE TABLE knowledge (id INTEGER PRIMARY KEY, type TEXT, project TEXT,
                                importance TEXT, created_at TEXT);
        CREATE TABLE importance_budget_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT, knowledge_id INTEGER NOT NULL, project TEXT NOT NULL,
            ktype TEXT NOT NULL, requested TEXT NOT NULL, applied TEXT NOT NULL,
            budget INTEGER NOT NULL, used INTEGER NOT NULL, created_at TEXT NOT NULL);
    """)
    yield conn
    conn.close()


def _add(db, importance, *, project="p", ktype="decision", at=NOON):
    db.execute("INSERT INTO knowledge (type, project, importance, created_at) VALUES (?, ?, ?, ?)",
               (ktype, project, importance, at.strftime("%Y-%m-%dT%H:%M:%S.%fZ")))


def _apply(db, importance="critical", *, environ=None, ktype="decision", project="p", now=NOON):
    return ib.apply_budget(db, project=project, ktype=ktype, importance=importance, now=now,
                           environ=environ or {})


def test_under_budget_keeps_the_level(db):
    for _ in range(ib.DEFAULT_CRITICAL_DAILY_BUDGET - 1):
        _add(db, "critical")
    decision = _apply(db)
    assert not decision.clamped and decision.applied == "critical"


def test_over_budget_drops_one_level_with_a_note(db):
    for _ in range(ib.DEFAULT_CRITICAL_DAILY_BUDGET):
        _add(db, "critical")
    decision = _apply(db)
    assert decision.clamped and decision.applied == "high"
    assert decision.used == ib.DEFAULT_CRITICAL_DAILY_BUDGET
    assert "lowered from critical to high" in decision.note()


def test_yesterday_and_other_projects_do_not_count(db):
    for _ in range(ib.DEFAULT_CRITICAL_DAILY_BUDGET):
        _add(db, "critical", at=NOON - timedelta(hours=13))   # 23:00 the day before
        _add(db, "critical", project="other")
    assert not _apply(db).clamped


def test_exempt_types_neither_count_nor_get_clamped(db):
    for _ in range(ib.DEFAULT_CRITICAL_DAILY_BUDGET):
        _add(db, "critical", ktype="convention")
    assert not _apply(db).clamped
    for _ in range(ib.DEFAULT_CRITICAL_DAILY_BUDGET):
        _add(db, "critical")
    assert not _apply(db, ktype="convention").clamped
    assert _apply(db, ktype="decision").clamped


def test_high_budget_cascades_when_configured(db):
    env = {ib.CRITICAL_BUDGET_ENV: "1", ib.HIGH_BUDGET_ENV: "1"}
    _add(db, "critical")
    _add(db, "high")
    decision = _apply(db, environ=env)
    assert (decision.requested, decision.applied) == ("critical", "medium")
    assert _apply(db, "high", environ=env).applied == "medium"
    assert _apply(db, "medium", environ=env).applied == "medium"


def test_zero_means_unlimited_and_negative_is_rejected(db):
    for _ in range(50):
        _add(db, "critical")
    assert not _apply(db, environ={ib.CRITICAL_BUDGET_ENV: "0"}).clamped
    with pytest.raises(ValueError):
        _apply(db, environ={ib.CRITICAL_BUDGET_ENV: "-1"})


def test_today_summary_lists_clamps_per_project(db):
    decision = ib.BudgetDecision(requested="critical", applied="high", budget=5, used=5)
    ib.record_clamp(db, decision, knowledge_id=1, project="p", ktype="decision",
                    created_at=NOON.strftime("%Y-%m-%dT%H:%M:%S.%fZ"))
    summary = ib.today_summary(db, now=NOON, environ={})
    assert summary["clamped_today"] == {"p": {"critical": 1}}
    assert summary["budgets"] == {"critical": ib.DEFAULT_CRITICAL_DAILY_BUDGET, "high": ib.DEFAULT_HIGH_DAILY_BUDGET}
