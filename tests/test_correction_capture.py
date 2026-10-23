"""Tests for src/correction_capture.py (correction detection + draft insights)."""

from __future__ import annotations

import json
import sqlite3
import time
from datetime import UTC, datetime, timedelta

import pytest

import correction_capture as cc

T0 = datetime(2026, 10, 13, 12, 0, tzinfo=UTC)

INSIGHTS_DDL = """
CREATE TABLE insights (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    content TEXT NOT NULL,
    context TEXT DEFAULT '',
    category TEXT NOT NULL,
    importance INTEGER NOT NULL DEFAULT 2,
    confidence REAL NOT NULL DEFAULT 0.5,
    source_error_ids TEXT DEFAULT '[]',
    project TEXT DEFAULT 'general',
    tags TEXT DEFAULT '[]',
    status TEXT DEFAULT 'active',
    promoted_to_rule_id INTEGER,
    fire_count INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


@pytest.fixture
def db():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(INSIGHTS_DDL)
    yield conn
    conn.close()


def at(minutes):
    return lambda: T0 + timedelta(minutes=minutes)


# ── detect ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize("text", [None, "", "   \n\t", "Please add a test for the parser",
                                  "Добавь, пожалуйста, тест для парсера", "OK", "Fix the TODO in README.md",
                                  "употреблять", "Я говорил с Петей вчера", "I told the team about it",
                                  "Run the tests again", "Do not touch the config"])
def test_detect_none(text):
    assert cc.detect(text) is None


@pytest.mark.parametrize(("text", "marker"), [
    ("I already told you to use Postgres", "already_told"),
    ("I've already said this twice", "already_told"),
    ("I told you not to mock the database", "already_told"),
    ("How many times do I have to repeat it?", "how_many_times"),
    ("For the last time: no force pushes", "how_many_times"),
    ("Я же говорил, не трогай миграции", "ru_already_told"),
    ("я уже просила тебя так не делать", "ru_already_told"),
    ("Сколько раз повторять: тесты перед коммитом", "ru_how_many_times"),
    ("сколько можно ломать сборку", "ru_how_many_times"),
    ("wtf is this output", "profanity"),
    ("This is fucking broken", "profanity"),
    ("Бля, опять сломал", "profanity"),
    ("ну пиздец, что это", "profanity"),
])
def test_detect_strong_phrases(text, marker):
    signal = cc.detect(text)
    assert signal is not None and signal.strength == "strong"
    assert marker in signal.markers


@pytest.mark.parametrize("text", ["STOP CHANGING THE SCHEMA", "НЕ ТРОГАЙ ЭТОТ ФАЙЛ", "WHY IS THIS STILL Broken"])
def test_detect_shout(text):
    signal = cc.detect(text)
    assert signal is not None and signal.strength == "strong" and "shout" in signal.markers


@pytest.mark.parametrize("text", ["You didn't run the tests again", "Again you do not check types",
                                  "Опять не запустил тесты", "ты снова не проверил типы",
                                  "it still doesn't compile"])
def test_detect_weak(text):
    signal = cc.detect(text)
    assert signal is not None and signal.strength == "weak"
    assert signal.markers == ("repeat", "negated_action")


def test_detect_excerpt_collapses_whitespace_and_truncates():
    signal = cc.detect("I already told you\n\n   use   tabs " + "x" * 1000)
    assert signal is not None
    assert "\n" not in signal.excerpt and "  " not in signal.excerpt
    assert len(signal.excerpt) == cc.MAX_INSIGHT_TEXT_CHARS


def test_detect_scans_head_and_tail_of_long_prompts():
    filler = "log line without anything special\n" * 500
    assert cc.detect("I already told you about this\n" + filler) is not None
    assert cc.detect(filler + "how many times do I have to say it") is not None
    assert cc.detect(filler[:2000] + "сколько раз" + filler[:2000]) is None


def test_detect_curly_apostrophe():
    assert cc.detect("I’ve already told you twice") is not None


# Guards against catastrophic regex backtracking (seconds per call), not against
# runner speed: a typical call takes well under a millisecond, and a loaded CI
# runner must not turn that into a failure.
DETECT_BUDGET_SECONDS = 0.02


def test_detect_is_fast():
    text = "Please refactor the parser and keep the public API stable. " * 200
    rounds = 200
    started = time.perf_counter()
    for _ in range(rounds):
        cc.detect(text)
    assert (time.perf_counter() - started) / rounds < DETECT_BUDGET_SECONDS


# ── capture ──────────────────────────────────────────────────────────────


def insight(db, insight_id):
    return dict(db.execute("SELECT * FROM insights WHERE id=?", (insight_id,)).fetchone())


def test_capture_no_signal_writes_nothing(db):
    assert cc.capture(db, "Add a parser test", "p", "s1", clock=at(0)) is None
    assert cc.capture(db, None, "p", "s1", clock=at(0)) is None
    assert db.execute("SELECT COUNT(*) FROM insights").fetchone()[0] == 0


def test_capture_creates_draft_insight(db):
    insight_id = cc.capture(db, "Я же говорил: не используй print в сервисе", "proj", "s1", clock=at(0))
    row = insight(db, insight_id)
    assert row["category"] == cc.CORRECTION_CATEGORY
    assert row["status"] == "active" and row["project"] == "proj" and row["session_id"] == "s1"
    assert (row["importance"], row["confidence"]) == (cc.DRAFT_IMPORTANCE, cc.DRAFT_CONFIDENCE)
    assert json.loads(row["tags"]) == [cc.DRAFT_TAG, cc.CORRECTION_TAG, "strength:strong"]
    assert row["content"] == "Я же говорил: не используй print в сервисе"
    assert "ru_already_told" in row["context"]


def test_capture_redacts_secrets(db):
    key = "AKIA" + "Q" * 16
    insight_id = cc.capture(db, f"I already told you the key is {key}", "p", "s1", clock=at(0))
    assert key not in insight(db, insight_id)["content"]


def test_capture_defaults_project(db):
    insight_id = cc.capture(db, "how many times, run the linter", None, "s1", clock=at(0))
    assert insight(db, insight_id)["project"] == cc.DEFAULT_PROJECT


def test_capture_dedupes_recent_similar_draft(db):
    first = cc.capture(db, "I already told you: run the linter before commit", "p", "s1", clock=at(0))
    second = cc.capture(db, "How many times: run the linter before every commit!", "p", "s1", clock=at(30))
    assert second == first
    row = insight(db, first)
    assert row["importance"] == cc.DRAFT_IMPORTANCE + cc.REPEAT_IMPORTANCE_STEP
    assert row["confidence"] == pytest.approx(cc.DRAFT_CONFIDENCE + cc.REPEAT_CONFIDENCE_STEP)
    assert db.execute("SELECT COUNT(*) FROM insights").fetchone()[0] == 1


def test_capture_new_draft_after_window_other_project_or_other_topic(db):
    first = cc.capture(db, "I already told you: run the linter before commit", "p", "s1", clock=at(0))
    late = cc.capture(db, "I already told you: run the linter before commit", "p", "s1", clock=at(91))
    other_project = cc.capture(db, "I already told you: run the linter before commit", "q", "s1", clock=at(1))
    other_topic = cc.capture(db, "I already told you: deploy only on Mondays", "p", "s1", clock=at(2))
    assert len({first, late, other_project, other_topic}) == 4


def test_capture_unicode_only_signal(db):
    insight_id = cc.capture(db, "СКОЛЬКО РАЗ ПОВТОРЯТЬ 😡😡", "p", "s1", clock=at(0))
    assert insight_id is not None
    assert "😡" in insight(db, insight_id)["content"]


def test_capture_draft_flows_through_insight_pipeline(monkeypatch, tmp_path):
    import server

    monkeypatch.setattr(server, "MEMORY_DIR", tmp_path)
    store = server.Store()
    try:
        insight_id = cc.capture(store.db, "Я же говорил: тесты перед коммитом", "proj", "s1")
        listed = store.manage_insight("s1", "list", project="proj")
        assert [row["id"] for row in listed["insights"]] == [insight_id]
        assert listed["insights"][0]["promotion_eligible"] is False
        upvoted = store.manage_insight("s1", "upvote", id=insight_id)
        assert upvoted["importance"] == cc.DRAFT_IMPORTANCE + 1
        assert store.promote_insight_to_rule("s1", insight_id)["error"] == "Not eligible"
    finally:
        store.db.close()
