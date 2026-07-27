"""Tests for src/task_classifier.py — v8.0."""

from __future__ import annotations

import sqlite3
from unittest.mock import patch

import pytest

from task_classifier import LEVEL_PHASES, classify_task
from base_schema import apply_full_schema


# ──────────────────────────────────────────────
# Keyword-based classification
# ──────────────────────────────────────────────

def test_classify_quick_fix_is_L1():
    r = classify_task("fix typo in README")
    assert r["level"] == 1
    assert r["confidence"] == 1.0
    assert r["suggested_phases"] == LEVEL_PHASES[1]


def test_classify_feature_is_L2():
    r = classify_task("add /users endpoint with pagination")
    assert r["level"] == 2
    assert "plan" in r["suggested_phases"]
    assert "creative" not in r["suggested_phases"]


def test_classify_refactor_is_L3():
    r = classify_task("refactor auth middleware to JWT-only, touch 3 files")
    assert r["level"] == 3
    assert "creative" in r["suggested_phases"]


def test_classify_architecture_is_L4():
    r = classify_task("architecture redesign: split monolith into three microservices")
    assert r["level"] == 4
    assert "creative" in r["suggested_phases"]
    assert r["estimated_tokens"] >= 50_000


# ──────────────────────────────────────────────
# Structural invariants
# ──────────────────────────────────────────────

def test_classify_returns_phases_matching_level():
    for desc, expected in [
        ("fix typo", 1),
        ("add endpoint /ping", 2),
        ("refactor core", 3),
        ("redesign architecture", 4),
    ]:
        r = classify_task(desc)
        assert r["level"] == expected
        assert r["suggested_phases"] == LEVEL_PHASES[expected]
        assert r["estimated_tokens"] > 0
        assert 0.0 <= r["confidence"] <= 1.0
        assert isinstance(r["rationale"], str) and r["rationale"]


def test_classify_empty_description_raises():
    with pytest.raises(ValueError):
        classify_task("")
    with pytest.raises(ValueError):
        classify_task("   ")


def test_classify_length_fallback_without_keywords():
    # Short nonsense — L1 by length.
    r = classify_task("do the thing now")
    assert r["level"] == 1
    assert r["confidence"] == 0.5

    # Long nonsense — L3/L4 by length.
    long_desc = " ".join(["word"] * 70)
    r2 = classify_task(long_desc)
    assert r2["level"] == 4
    assert r2["confidence"] == 0.5


# ──────────────────────────────────────────────
# Analogy integration
# ──────────────────────────────────────────────

def test_classify_with_analogize_boosts_confidence(tmp_path):
    """When project+db supplied, AnalogyEngine hits boost confidence."""
    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    apply_full_schema(db)

    # Without analogy — confidence = 0.5 (length fallback).
    r_no_match = classify_task(
        "unique-zzzz-xxxx short",  # no keywords, length fallback path
        project="myproj",
        db=db,
    )
    base_conf = r_no_match["confidence"]

    # Mock analogize to return several hits.
    fake_hits = [
        {"analogy_score": 0.4}, {"analogy_score": 0.35}, {"analogy_score": 0.5}
    ]
    with patch("analogy.AnalogyEngine.find_analogies", return_value=fake_hits):
        r_with = classify_task(
            "unique-zzzz-xxxx short",
            project="myproj",
            db=db,
        )
    assert r_with["confidence"] > base_conf
    assert r_with["analogy"] is not None
    assert r_with["analogy"]["count"] == 3


def test_classify_analogy_without_db_is_safe():
    # project supplied but db=None — must still return a result.
    r = classify_task("refactor something", project="p")
    assert r["level"] == 3
    assert r["analogy"] is None
