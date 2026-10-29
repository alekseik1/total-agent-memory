"""Tests for src/task_phases.py — v8.0."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from task_phases import TaskPhaseError, TaskPhases


@pytest.fixture
def tp_db():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    mig = Path(__file__).parent.parent / "migrations" / "012_task_phases.sql"
    conn.executescript(mig.read_text(encoding="utf-8"))
    # Also load procedural migration so complete_task hook is available.
    proc = Path(__file__).parent.parent / "migrations" / "009_procedural.sql"
    conn.executescript(proc.read_text(encoding="utf-8"))
    yield conn
    conn.close()


@pytest.fixture
def tp(tp_db):
    return TaskPhases(tp_db)


# ──────────────────────────────────────────────
# Creation
# ──────────────────────────────────────────────

def test_create_task_starts_in_van_phase(tp):
    r = tp.create_task("task-1", "refactor auth middleware to JWT-only", level=3)
    assert r["phase"] == "van"
    assert r["level"] == 3
    assert tp.current_phase("task-1") == "van"


def test_create_task_infers_level_from_description(tp):
    r = tp.create_task("task-auto", "fix typo in footer")
    assert r["level"] == 1


def test_create_task_duplicate_rejects(tp):
    tp.create_task("dup", "fix typo")
    with pytest.raises(TaskPhaseError):
        tp.create_task("dup", "fix typo again")


# ──────────────────────────────────────────────
# Transitions
# ──────────────────────────────────────────────

def test_phase_transition_closes_previous(tp, tp_db):
    tp.create_task("t2", "add /users endpoint", level=2)
    tp.phase_transition("t2", "plan", artifacts={"files": ["api.py"]})
    rows = list(tp_db.execute(
        "SELECT phase, exited_at FROM task_phases WHERE task_id = ? ORDER BY entered_at",
        ("t2",),
    ))
    assert rows[0]["phase"] == "van"
    assert rows[0]["exited_at"] is not None
    assert rows[1]["phase"] == "plan"
    assert rows[1]["exited_at"] is None


def test_phase_transition_invalid_for_level_rejects(tp):
    # L1 skips plan and creative — going van → creative is illegal.
    tp.create_task("t1", "fix typo", level=1)
    with pytest.raises(TaskPhaseError, match="not allowed for L1"):
        tp.phase_transition("t1", "creative")


def test_phase_transition_skip_rejects(tp):
    # L3 must go van → plan → creative → build → reflect → archive.
    tp.create_task("t3", "refactor auth", level=3)
    with pytest.raises(TaskPhaseError, match="skip"):
        tp.phase_transition("t3", "creative")


def test_phase_transition_backward_rejects(tp):
    tp.create_task("t2", "add endpoint", level=2)
    tp.phase_transition("t2", "plan")
    with pytest.raises(TaskPhaseError):
        tp.phase_transition("t2", "van")


def test_phase_transition_unknown_phase(tp):
    tp.create_task("tx", "fix typo", level=1)
    with pytest.raises(TaskPhaseError):
        tp.phase_transition("tx", "ship")


# ──────────────────────────────────────────────
# Listing / current
# ──────────────────────────────────────────────

def test_list_phases_chronological(tp):
    tp.create_task("t2", "add endpoint", level=2)
    tp.phase_transition("t2", "plan")
    tp.phase_transition("t2", "build")
    phases = tp.list_phases("t2")
    assert [p["phase"] for p in phases] == ["van", "plan", "build"]
    # first two closed, last open
    assert phases[0]["exited_at"] is not None
    assert phases[1]["exited_at"] is not None
    assert phases[2]["exited_at"] is None


def test_current_phase_for_completed_task_returns_archive(tp):
    tp.create_task("t1", "fix typo", level=1)
    for p in ("build", "reflect", "archive"):
        tp.phase_transition("t1", p)
    tp.complete_task("t1", final_notes="done")
    assert tp.current_phase("t1") == "archive"


def test_current_phase_unknown_task_returns_none(tp):
    assert tp.current_phase("nope") is None


# ──────────────────────────────────────────────
# Completion
# ──────────────────────────────────────────────

def test_complete_task_not_in_archive_rejects(tp):
    tp.create_task("t1", "fix typo", level=1)
    with pytest.raises(TaskPhaseError):
        tp.complete_task("t1")


def test_complete_task_triggers_workflow_track(tp, tp_db):
    from procedural import ProceduralMemory
    pm = ProceduralMemory(tp_db)
    # Seed a workflow with the same id as our task.
    task_id = "wf-task-7"
    # Override auto-generated id by pre-inserting the workflow row.
    tp_db.execute(
        """INSERT INTO workflows (id, name, steps, project, created_at, updated_at)
           VALUES (?, 'fix_typo', '["edit", "test"]', 'general',
                   '2026-04-19T00:00:00Z', '2026-04-19T00:00:00Z')""",
        (task_id,),
    )
    tp_db.commit()

    tp.create_task(task_id, "fix typo in docs", level=1)
    for p in ("build", "reflect", "archive"):
        tp.phase_transition(task_id, p)
    result = tp.complete_task(task_id, outcome="success")

    assert result["outcome"] == "success"
    assert result["tracked"] is not None
    assert result["tracked"]["workflow_id"] == task_id
    # Workflow aggregates should reflect the tracked outcome.
    wf = pm.get_workflow(task_id)
    assert wf["times_run"] == 1
    assert wf["success_count"] == 1
