from __future__ import annotations

import sqlite3
from datetime import datetime

import pytest

from memory_core.relative_dates import annotate, message_time


@pytest.mark.parametrize(("content", "expected"), [
    ("[1:56 pm on 8 May, 2023] Melanie: hi", datetime.fromisoformat("2023-05-08T13:56")),
    ("[2023/05/20 (Sat) 02:21]\nuser: hi", datetime.fromisoformat("2023-05-20T02:21")),
    ("[2023-05-20] note", datetime.fromisoformat("2023-05-20T00:00")),
    ("[8 May 2023] note", datetime.fromisoformat("2023-05-08T00:00")),
])
def test_message_time_reads_leading_stamp(content, expected):
    assert message_time(content, "2030-01-01T00:00:00Z") == expected


def test_message_time_falls_back_to_created_at():
    assert message_time("no stamp here", "2024-02-01T10:00:00Z") == datetime.fromisoformat("2024-02-01T10:00")
    assert message_time("[not a date] text", "2024-02-01T10:00:00") == datetime.fromisoformat("2024-02-01T10:00")
    assert message_time("text", None) is None
    assert message_time("text", "garbage") is None


def test_weekday_and_week_are_counted_from_the_message_not_today():
    text = annotate("[1:14 pm on 17 December, 2023] Evan: my son fell off his bike last Thursday, "
                    "and last week we went skiing.")
    assert "last Thursday [Thu 14 December 2023]" in text
    assert "last week [the week before Sun 17 December 2023]" in text


def test_year_boundary():
    text = annotate("[9:00 am on 3 January, 2024] Evan: we had a drunken night yesterday and last month")
    assert "yesterday [Tue 2 January 2024]" in text
    assert "last month [December 2023]" in text


def test_ago_phrases_are_approximate():
    text = annotate("[1:14 pm on 17 December, 2023] Two weeks ago I was sick; 3 months ago too; 5 days ago;"
                    " a week ago")
    assert "Two weeks ago [about 2 weeks before Sun 17 December 2023, around 3 December 2023]" in text
    assert "a week ago [the week before Sun 17 December 2023]" in text
    assert "3 months ago [around September 2023]" in text
    assert "5 days ago [Tue 12 December 2023]" in text


def test_weekends():
    text = annotate("[2023/05/20 (Sat) 02:21]\nuser: last weekend was fun; this weekend too")
    assert "last weekend [the weekend before Sat 20 May 2023]" in text
    assert "this weekend [the weekend of Sat 20 May 2023]" in text


def test_russian_phrases():
    text = annotate("[2024-01-03] Вчера я купил машину, а на прошлой неделе продал старую. "
                    "Две недели назад был в Москве, год назад тоже.")
    assert "Вчера [Tue 2 January 2024]" in text
    assert "на прошлой неделе [the week before Wed 3 January 2024]" in text
    assert "Две недели назад [about 2 weeks before Wed 3 January 2024, around 20 December 2023]" in text
    assert "год назад [around 2023]" in text


def test_few_days_ago_stays_relative_to_the_message():
    text = annotate("[10:04 am on 19 June, 2023] Jon: a few days ago I met her")
    assert "a few days ago [a few days before Mon 19 June 2023]" in text


def test_text_without_phrases_or_anchor_is_unchanged():
    plain = "[1:56 pm on 8 May, 2023] Melanie: I painted a sunrise."
    assert annotate(plain) == plain
    assert annotate("yesterday was fine") == "yesterday was fine"
    assert annotate("") == ""


def test_words_inside_other_words_are_not_matched():
    text = "[2023-05-20] the todays list and yesterdayish mood"
    assert annotate(text) == text


def test_annotation_is_idempotent_on_its_own_output():
    once = annotate("[2023-05-20] yesterday")
    assert annotate(once) == once


def test_context_mode_annotates_and_can_be_disabled(monkeypatch):
    from memory_core.evidence_context import EvidenceContext
    from memory_core.retrieval import SearchScope

    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    db.execute("CREATE TABLE knowledge (id INTEGER PRIMARY KEY, content TEXT, project TEXT, session_id TEXT, "
               "created_at TEXT, status TEXT, type TEXT, branch TEXT, tags TEXT)")
    db.execute("INSERT INTO knowledge VALUES (1, '[2023-05-20] I moved yesterday', 'p', 's', "
               "'2026-01-01T00:00:00Z', 'active', 'fact', '', '[]')")
    hit = {"id": 1, "content": "[2023-05-20] I moved yesterday", "created_at": "2026-01-01T00:00:00Z"}
    scope = SearchScope(project="p")

    monkeypatch.setenv("MEMORY_CONTEXT_RESOLVE_DATES", "on")
    built = EvidenceContext(db).build([hit], query="when did I move", scope=scope, radius=0)
    assert "yesterday [Fri 19 May 2023]" in built[0]["content"]

    monkeypatch.setenv("MEMORY_CONTEXT_RESOLVE_DATES", "off")
    built = EvidenceContext(db).build([hit], query="when did I move", scope=scope, radius=0)
    assert built[0]["content"] == "[2023-05-20] I moved yesterday"

    monkeypatch.setenv("MEMORY_CONTEXT_RESOLVE_DATES", "sometimes")
    with pytest.raises(ValueError):
        EvidenceContext(db).build([hit], query="when did I move", scope=scope, radius=0)
