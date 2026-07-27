"""Tests for the v10 quality gate.

Each test isolates `quality_gate` env knobs and the provider cache so the
gate's behavior is deterministic regardless of the host's MEMORY_*
configuration or whether Ollama is actually running.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from pathlib import Path

import pytest

# Re-use the package conftest's sys.path injection.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import quality_gate as qg
from base_schema import apply_full_schema


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    """Clear provider cache + sane env defaults for every test."""
    qg._reset_provider_cache()
    # Force the gate fully on so 'auto' branches don't depend on local Ollama.
    monkeypatch.setenv("MEMORY_QUALITY_GATE_ENABLED", "true")
    monkeypatch.delenv("MEMORY_QUALITY_THRESHOLD", raising=False)
    monkeypatch.delenv("MEMORY_QUALITY_MIN_CHARS", raising=False)
    monkeypatch.delenv("MEMORY_QUALITY_BYPASS_TYPES", raising=False)
    monkeypatch.delenv("MEMORY_QUALITY_LOG_ALL", raising=False)
    yield
    qg._reset_provider_cache()


class _FakeProvider:
    """Stand-in for `LLMProvider`. Returns a canned response from `complete`."""

    def __init__(self, response="", available=True, raises=None):
        self.name = "fake"
        self._response = response
        self._available = available
        self._raises = raises
        self.calls: list[tuple[str, dict]] = []

    def available(self) -> bool:
        return self._available

    def complete(self, prompt: str, **kwargs) -> str:
        self.calls.append((prompt, kwargs))
        if self._raises:
            raise self._raises
        return self._response


@pytest.fixture
def fake_provider(monkeypatch):
    """Patch `_get_provider()` to return a fake."""

    def _make(response="", available=True, raises=None):
        prov = _FakeProvider(response=response, available=available, raises=raises)
        monkeypatch.setattr(qg, "_get_provider", lambda: prov)
        # `_model_name()` is harmless — leave it alone.
        return prov

    return _make


@pytest.fixture
def gate_db(tmp_path):
    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    apply_full_schema(db)
    yield db
    db.close()


# ──────────────────────────────────────────────
# should_score gating
# ──────────────────────────────────────────────


def test_should_score_skips_short_content(monkeypatch):
    monkeypatch.setenv("MEMORY_QUALITY_MIN_CHARS", "200")
    ok, reason = qg.should_score("a" * 50, "fact")
    assert not ok
    assert "min_chars" in reason


def test_should_score_skips_bypass_types(monkeypatch):
    monkeypatch.setenv("MEMORY_QUALITY_BYPASS_TYPES", "transcript,raw")
    long = "x" * 400
    ok, reason = qg.should_score(long, "transcript")
    assert not ok
    assert "bypass" in reason


def test_should_score_disabled_via_env(monkeypatch):
    monkeypatch.setenv("MEMORY_QUALITY_GATE_ENABLED", "false")
    long = "x" * 400
    ok, reason = qg.should_score(long, "fact")
    assert not ok
    assert "disabled" in reason


def test_should_score_runs_when_enabled():
    long = "x" * 400
    ok, reason = qg.should_score(long, "fact")
    assert ok
    assert reason == ""


# ──────────────────────────────────────────────
# score_quality decisions
# ──────────────────────────────────────────────


def test_score_quality_pass_when_above_threshold(fake_provider, monkeypatch):
    monkeypatch.setenv("MEMORY_QUALITY_THRESHOLD", "0.5")
    fake_provider(
        response=json.dumps(
            {"specificity": 0.8, "actionability": 0.7, "verifiability": 0.9, "reason": "concrete"}
        )
    )
    score = qg.score_quality("a" * 200, "decision", "vito")
    assert score.decision == "pass"
    assert score.passed
    assert pytest.approx(score.total, abs=1e-6) == (0.8 + 0.7 + 0.9) / 3
    assert score.reason == "concrete"


def test_score_quality_drop_when_below_threshold(fake_provider, monkeypatch):
    monkeypatch.setenv("MEMORY_QUALITY_THRESHOLD", "0.5")
    fake_provider(
        response=json.dumps(
            {"specificity": 0.2, "actionability": 0.1, "verifiability": 0.3, "reason": "vague"}
        )
    )
    score = qg.score_quality("a" * 200, "decision", "vito")
    assert score.decision == "drop"
    assert not score.passed
    assert score.total < 0.5


def test_score_quality_skip_when_provider_unavailable(fake_provider):
    fake_provider(available=False)
    score = qg.score_quality("a" * 200, "fact")
    assert score.decision == "skip"
    assert score.passed  # gate fails open
    assert "unavailable" in score.reason


def test_score_quality_error_when_llm_raises(fake_provider):
    fake_provider(raises=RuntimeError("boom"))
    score = qg.score_quality("a" * 200, "fact")
    assert score.decision == "error"
    assert score.passed  # still fails open
    assert "boom" in score.reason


def test_score_quality_error_when_llm_returns_garbage(fake_provider):
    fake_provider(response="i am a banana")
    score = qg.score_quality("a" * 200, "fact")
    assert score.decision == "error"
    assert score.passed
    assert "unparsable" in score.reason


def test_score_quality_error_when_axes_missing(fake_provider):
    fake_provider(response=json.dumps({"specificity": 0.5, "reason": "incomplete"}))
    score = qg.score_quality("a" * 200, "fact")
    assert score.decision == "error"
    assert "missing axes" in score.reason


def test_score_quality_clamps_axes_to_unit_interval(fake_provider):
    fake_provider(
        response=json.dumps(
            {"specificity": 1.5, "actionability": -0.2, "verifiability": 0.6, "reason": "clamped"}
        )
    )
    score = qg.score_quality("a" * 200, "fact")
    assert score.decision == "pass"
    assert score.specificity == 1.0
    assert score.actionability == 0.0
    assert score.verifiability == 0.6


def test_score_quality_threshold_clamped_to_unit_interval(fake_provider, monkeypatch):
    monkeypatch.setenv("MEMORY_QUALITY_THRESHOLD", "10")
    fake_provider(
        response=json.dumps(
            {"specificity": 0.99, "actionability": 0.99, "verifiability": 0.99, "reason": "ok"}
        )
    )
    score = qg.score_quality("a" * 200, "fact")
    # threshold clamped to 1.0 → 0.99 < 1.0 → drop
    assert score.threshold == 1.0
    assert score.decision == "drop"


def test_score_quality_skip_short_content_does_not_call_llm(fake_provider, monkeypatch):
    monkeypatch.setenv("MEMORY_QUALITY_MIN_CHARS", "500")
    prov = fake_provider(response='{"specificity":1,"actionability":1,"verifiability":1}')
    score = qg.score_quality("short", "fact")
    assert score.decision == "skip"
    assert prov.calls == []  # no LLM call wasted on noise


def test_score_quality_skip_bypass_type_does_not_call_llm(fake_provider, monkeypatch):
    monkeypatch.setenv("MEMORY_QUALITY_BYPASS_TYPES", "transcript")
    prov = fake_provider(response='{"specificity":1,"actionability":1,"verifiability":1}')
    score = qg.score_quality("a" * 400, "transcript")
    assert score.decision == "skip"
    assert prov.calls == []


def test_score_quality_parses_json_inside_prose(fake_provider):
    """Some Ollama models prefix JSON with prose. We tolerate that."""
    fake_provider(
        response=(
            "Here is the score:\n"
            '{"specificity": 0.9, "actionability": 0.8, "verifiability": 0.7, "reason": "good"}'
        )
    )
    score = qg.score_quality("a" * 200, "fact")
    assert score.decision == "pass"
    assert pytest.approx(score.total, abs=1e-6) == 0.8


# ──────────────────────────────────────────────
# Audit log
# ──────────────────────────────────────────────


def test_log_decision_records_drops(gate_db):
    score = qg.QualityScore(
        decision="drop",
        total=0.3,
        specificity=0.4,
        actionability=0.2,
        verifiability=0.3,
        reason="vague reflection",
        threshold=0.5,
        provider="fake",
        model="qwen2.5",
        latency_ms=42,
    )
    qg.log_decision(
        gate_db, score, project="vito", ktype="fact", content="x" * 300
    )
    rows = gate_db.execute("SELECT * FROM quality_gate_log").fetchall()
    assert len(rows) == 1
    row = rows[0]
    assert row["decision"] == "drop"
    assert row["score_total"] == 0.3
    assert row["project"] == "vito"
    assert row["content_chars"] == 300
    # Preview must be truncated to <= 240 chars (see _PREVIEW_CHARS).
    assert len(row["content_preview"]) <= 240


def test_log_decision_skips_passes_by_default(gate_db):
    score = qg.QualityScore(
        decision="pass", total=0.9, specificity=0.9, actionability=0.9,
        verifiability=0.9, reason="ok", threshold=0.5,
    )
    qg.log_decision(gate_db, score, project="vito", ktype="fact", content="abc")
    rows = gate_db.execute("SELECT COUNT(*) AS c FROM quality_gate_log").fetchone()
    assert rows["c"] == 0


def test_log_decision_records_passes_when_log_all_set(gate_db, monkeypatch):
    monkeypatch.setenv("MEMORY_QUALITY_LOG_ALL", "1")
    score = qg.QualityScore(
        decision="pass", total=0.9, specificity=0.9, actionability=0.9,
        verifiability=0.9, reason="ok", threshold=0.5,
    )
    qg.log_decision(gate_db, score, project="vito", ktype="fact", content="abc")
    row = gate_db.execute("SELECT * FROM quality_gate_log").fetchone()
    assert row is not None
    assert row["decision"] == "pass"


def test_log_decision_records_errors_unconditionally(gate_db):
    score = qg.QualityScore(
        decision="error", total=None, specificity=None, actionability=None,
        verifiability=None, reason="LLM error", threshold=0.5,
    )
    qg.log_decision(gate_db, score, project=None, ktype=None, content="x")
    row = gate_db.execute("SELECT * FROM quality_gate_log").fetchone()
    assert row is not None
    assert row["decision"] == "error"
    assert row["score_total"] is None
