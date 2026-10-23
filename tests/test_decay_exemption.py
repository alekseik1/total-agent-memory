"""Query-time decay exemption for decision / convention / lesson records (src/decay_policy.py)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

import decay_policy

HALF_LIFE_DAYS = 90
YEAR_DAYS = 365


@pytest.fixture(autouse=True)
def _default_policy(monkeypatch):
    monkeypatch.delenv(decay_policy.EXEMPT_TYPES_ENV, raising=False)
    monkeypatch.delenv(decay_policy.EXEMPT_FLOOR_ENV, raising=False)


def _days_ago(days: int) -> str:
    return (datetime.now(UTC) - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _decay(stamp, ktype=None, half_life=HALF_LIFE_DAYS):
    from server import Store
    return Store._decay_factor(stamp, half_life, ktype)


@pytest.mark.parametrize("ktype", ["decision", "convention", "lesson", "Decision"])
def test_exempt_types_never_fall_below_the_floor(ktype):
    assert _decay(_days_ago(YEAR_DAYS), ktype) == pytest.approx(decay_policy.DEFAULT_EXEMPT_FLOOR)
    # Fresh records keep their full score; the floor only lifts old ones.
    assert _decay(_days_ago(0), ktype) == pytest.approx(1.0)


@pytest.mark.parametrize("ktype", ["fact", "solution", None])
def test_other_types_decay_with_the_half_life(ktype):
    year_old = _decay(_days_ago(YEAR_DAYS), ktype)
    assert year_old == pytest.approx(2 ** (-YEAR_DAYS / HALF_LIFE_DAYS), rel=1e-3)
    assert year_old < decay_policy.DEFAULT_EXEMPT_FLOOR


def test_legacy_positional_calls_are_unchanged():
    from server import Store
    assert Store._decay_factor(_days_ago(HALF_LIFE_DAYS), HALF_LIFE_DAYS) == pytest.approx(0.5, rel=1e-2)
    assert Store._decay_factor("") == 0.5
    assert _decay("", "decision") == pytest.approx(decay_policy.DEFAULT_EXEMPT_FLOOR)


def test_settings_override_types_and_floor(monkeypatch):
    monkeypatch.setenv(decay_policy.EXEMPT_TYPES_ENV, "fact")
    monkeypatch.setenv(decay_policy.EXEMPT_FLOOR_ENV, "0.6")
    assert _decay(_days_ago(YEAR_DAYS), "fact") == pytest.approx(0.6)
    assert _decay(_days_ago(YEAR_DAYS), "decision") < 0.6
    monkeypatch.setenv(decay_policy.EXEMPT_TYPES_ENV, "")
    assert decay_policy.floor_for("fact") is None


def test_invalid_floor_falls_back_to_the_default(monkeypatch):
    monkeypatch.setenv(decay_policy.EXEMPT_FLOOR_ENV, "1.7")
    with pytest.raises(ValueError):
        decay_policy.exempt_floor()
    assert decay_policy.floor_for("decision") == decay_policy.DEFAULT_EXEMPT_FLOOR


def test_old_decision_outranks_equally_old_fact_in_recall_scoring():
    stamp = _days_ago(2 * YEAR_DAYS)
    assert _decay(stamp, "decision") > 10 * _decay(stamp, "fact")
