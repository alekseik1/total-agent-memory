"""Property tests for functions that take untrusted text.

Recall queries, stored record content and tool arguments come straight from
agents and users. The functions below sit on that path, so they must never
raise and must keep a few structural invariants for *any* input, not only the
hand-picked examples in their unit tests.

Examples are derandomized (fixed seed, no example database) so the suite is
reproducible on every machine and in CI.
"""

from __future__ import annotations

import os
import sqlite3
import string
import sys
from datetime import date, datetime
from pathlib import Path

import pytest

pytest.importorskip("hypothesis", reason="hypothesis is a dev dependency (pip install -r requirements-dev.txt)")

from hypothesis import HealthCheck, assume, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import secret_redaction  # noqa: E402
import temporal_filter  # noqa: E402
from memory_core import query_terms, relative_dates  # noqa: E402
from memory_core.episodes import retriever as episodes_retriever  # noqa: E402
from memory_core.temporal.normalizer import normalize  # noqa: E402

# TAM_PROPERTY_EXAMPLES raises the budget for a local bug hunt; CI keeps the
# default so the run stays fast and reproducible.
PROPERTY_SETTINGS = settings(
    derandomize=os.environ.get("TAM_PROPERTY_RANDOM") != "1",
    database=None,
    deadline=None,
    max_examples=int(os.environ.get("TAM_PROPERTY_EXAMPLES", "300")),
    suppress_health_check=[HealthCheck.too_slow],
)

# Text that looks like what agents actually send: mixed scripts, digits,
# punctuation that FTS5 and regexes treat specially, and arbitrary unicode.
FTS_SPECIALS = '"*^:()-+{}[].,/\\\'`~!?@#$%&|<>=;'
WORDS = st.sampled_from(
    [
        "yesterday", "today", "tomorrow", "last", "next", "week", "month", "year", "ago",
        "monday", "friday", "weekend", "few", "couple", "of", "days", "the", "day", "before",
        "вчера", "сегодня", "завтра", "позавчера", "неделю", "назад", "месяц", "год", "лет",
        "прошлой", "следующей", "неделе", "пятницу", "в", "на", "два", "три",
        "AND", "OR", "NOT", "NEAR", "march", "2023", "spring", "in", "2024-02-30",
        "sk-", "password=", "token:", "Маша", "Маше", "context.Context", "orders.dlq",
    ]
)
MIXED_TEXT = st.one_of(
    st.text(max_size=200),
    st.text(alphabet=string.ascii_letters + string.digits + FTS_SPECIALS + " \t\n", max_size=200),
    st.lists(st.one_of(WORDS, st.text(alphabet=FTS_SPECIALS, min_size=1, max_size=3)), max_size=25).map(" ".join),
)
ANCHORS = st.datetimes(min_value=datetime(1900, 1, 1), max_value=datetime(2100, 12, 31))
MONTH_NAMES = ["january", "february", "march", "april", "may", "june", "july",
               "august", "september", "october", "november", "december"]


@pytest.fixture(scope="module")
def fts_db():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE VIRTUAL TABLE docs USING fts5(content)")
    conn.execute("INSERT INTO docs(content) VALUES ('маша staging port 5433 context.Context')")
    yield conn
    conn.close()


def assert_valid_match(conn: sqlite3.Connection, expression: str) -> None:
    conn.execute("SELECT rowid FROM docs WHERE docs MATCH ?", (expression,)).fetchall()


def is_subsequence(needle: str, haystack: str) -> bool:
    it = iter(haystack)
    return all(ch in it for ch in needle)


# --- temporal_filter --------------------------------------------------------


@PROPERTY_SETTINGS
@given(MIXED_TEXT)
def test_parse_query_dates_never_raises_and_ranges_are_ordered(text):
    ranges = temporal_filter.parse_query_dates(text)
    assert isinstance(ranges, list)
    for r in ranges:
        assert r.start <= r.end


@PROPERTY_SETTINGS
@given(st.dates(min_value=date(1000, 1, 1), max_value=date(9998, 12, 31)), st.sampled_from(["dmy", "mdy", "iso"]))
def test_parse_query_dates_finds_a_generated_date(day, layout):
    month = MONTH_NAMES[day.month - 1]
    query = {
        "dmy": f"what happened on {day.day} {month} {day.year}",
        "mdy": f"what happened on {month} {day.day}, {day.year}",
        "iso": f"what happened on {day:%Y-%m-%d}",
    }[layout]
    target = datetime(day.year, day.month, day.day)
    assert any(r.contains(target) for r in temporal_filter.parse_query_dates(query)), query


@PROPERTY_SETTINGS
@given(MIXED_TEXT)
def test_has_temporal_intent_and_entry_date_never_raise(text):
    assert isinstance(temporal_filter.has_temporal_intent(text), bool)
    parsed = temporal_filter.extract_entry_date(f"[{text}] Speaker: hello")
    assert parsed is None or isinstance(parsed, datetime)


@PROPERTY_SETTINGS
@given(MIXED_TEXT, st.lists(st.tuples(MIXED_TEXT, st.floats(-10, 10, allow_nan=False)), max_size=8))
def test_temporal_rerank_is_a_permutation(query, raw_entries):
    entries = [{"id": i, "content": c, "score": s} for i, (c, s) in enumerate(raw_entries)]
    out = temporal_filter.temporal_rerank(query, entries)
    assert sorted(e["id"] for e in out) == [e["id"] for e in entries]


# --- relative date normalization and annotation -----------------------------


@PROPERTY_SETTINGS
@given(MIXED_TEXT, ANCHORS)
def test_normalize_never_raises(phrase, anchor):
    result = normalize(phrase, anchor)
    if result is not None:
        datetime.fromisoformat(result.iso)


@PROPERTY_SETTINGS
@given(MIXED_TEXT, st.one_of(st.none(), st.text(max_size=40), ANCHORS.map(datetime.isoformat)))
def test_annotate_never_raises_and_only_inserts(content, created_at):
    out = relative_dates.annotate(content, created_at)
    assert isinstance(out, str)
    assert is_subsequence(content, out), "annotate must only insert, never drop or reorder text"


@PROPERTY_SETTINGS
@given(MIXED_TEXT, ANCHORS)
def test_annotate_is_idempotent(content, anchor):
    once = relative_dates.annotate(content, anchor.isoformat())
    assert relative_dates.annotate(once, anchor.isoformat()) == once


@PROPERTY_SETTINGS
@given(
    st.sampled_from(["yesterday", "tomorrow", "last friday", "next monday", "3 days ago",
                     "two weeks ago", "вчера", "завтра", "два дня назад", "в прошлую пятницу"]),
    ANCHORS,
)
def test_annotate_resolves_known_phrases(phrase, anchor):
    content = f"we shipped the release {phrase} after review"
    out = relative_dates.annotate(content, anchor.isoformat())
    assert f"{phrase} [" in out, out


@PROPERTY_SETTINGS
@given(st.one_of(st.none(), st.text(max_size=60)), MIXED_TEXT)
def test_message_time_never_raises(created_at, content):
    value = relative_dates.message_time(content, created_at)
    assert value is None or isinstance(value, datetime)


# --- query term extraction, stemming and FTS sanitizing ---------------------


@PROPERTY_SETTINGS
@given(MIXED_TEXT)
def test_lexical_terms_invariants(query):
    terms = query_terms.lexical_terms(query)
    folded = query.casefold()
    assert len(terms) <= query_terms.MAX_LEXICAL_TERMS
    assert len(terms) == len(set(terms))
    for term in terms:
        assert term and term == term.casefold()
        assert term in folded
    if any(t not in query_terms.QUESTION_STOPWORDS for t in terms):
        assert not set(terms) & query_terms.QUESTION_STOPWORDS


@PROPERTY_SETTINGS
@given(st.one_of(st.text(alphabet="абвгдеёжзийклмнопрстуфхцчшщъыьэюя", min_size=1, max_size=20), st.text(max_size=20)))
def test_russian_stem_shortens_cyrillic_and_keeps_other_words(word):
    stem = query_terms.russian_stem(word)
    assert isinstance(stem, str)
    if query_terms._CYRILLIC_WORD.fullmatch(word):
        assert len(stem) <= len(word)
    else:
        assert stem == word


@PROPERTY_SETTINGS
@given(MIXED_TEXT)
def test_fts_match_query_is_always_valid_fts5(fts_db, query):
    expression = query_terms.fts_match_query(query)
    assert expression
    assert_valid_match(fts_db, expression)


# Regression (hypothesis, 2026-10-13): a query with no \w term fell back to
# _fts_phrase(query) with the NUL kept, and FTS5 raised "unterminated string".
@pytest.mark.parametrize("query", ["\x00", "?\x00?", "\x00 \x00", "abc\x00def"])
def test_fts_match_query_with_nul_input(fts_db, query):
    assert "\x00" not in query_terms.fts_match_query(query)
    assert_valid_match(fts_db, query_terms.fts_match_query(query))


@PROPERTY_SETTINGS
@given(MIXED_TEXT)
def test_episode_fts_sanitizer_is_valid_or_empty(fts_db, query):
    expression = episodes_retriever._sanitize_fts(query)
    if expression:
        assert_valid_match(fts_db, expression)


@PROPERTY_SETTINGS
@given(st.text(max_size=60))
def test_fts_phrase_quoting_is_valid_fts5(fts_db, word):
    assert_valid_match(fts_db, query_terms._fts_phrase(word))


@PROPERTY_SETTINGS
@given(st.text(alphabet="абвгдеёжзийклмнопрстуфхцчшщэюя", min_size=4, max_size=12))
def test_russian_inflections_share_a_prefix_query(word):
    expression = query_terms.fts_match_query(word)
    stem = query_terms.russian_stem(word)
    if stem != word and len(stem) >= query_terms.MIN_PREFIX_STEM:
        assert expression == f'"{stem}"*'


# --- secret redaction -------------------------------------------------------

API_KEY_BODY = st.text(alphabet=string.ascii_letters + string.digits, min_size=20, max_size=64)
SEPARATORS = st.sampled_from([" ", "\n", "\t", "=", ": ", "(", "'", '"', ", "])


@PROPERTY_SETTINGS
@given(MIXED_TEXT)
def test_redact_secrets_is_idempotent_and_reports_change(text):
    once, changed = secret_redaction.redact_secrets(text)
    assert changed == (once != text)
    twice, changed_again = secret_redaction.redact_secrets(once)
    assert twice == once
    assert not changed_again


@PROPERTY_SETTINGS
@given(MIXED_TEXT, SEPARATORS, API_KEY_BODY, SEPARATORS, MIXED_TEXT,
       st.sampled_from(["sk-", "sk-proj-", "sk-ant-", "sk_live_", "sk_test_"]))
def test_generated_api_key_never_survives_redaction(prefix, sep_before, body, sep_after, suffix, key_prefix):
    key = key_prefix + body
    assume(body not in prefix and body not in suffix)
    text = f"{prefix}{sep_before}{key}{sep_after}{suffix}"
    redacted, changed = secret_redaction.redact_secrets(text)
    assert changed
    assert key not in redacted
    assert body not in redacted


JSON_LEAF = st.one_of(st.none(), st.booleans(), st.integers(), st.floats(allow_nan=False), MIXED_TEXT)
JSON_VALUE = st.recursive(
    JSON_LEAF,
    lambda children: st.one_of(
        st.lists(children, max_size=4),
        st.dictionaries(st.sampled_from(["content", "path", "query", "token", "file", "x"]), children, max_size=4),
    ),
    max_leaves=12,
)


def same_shape(a, b) -> bool:
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same_shape(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return isinstance(b, list) and len(a) == len(b) and all(same_shape(x, y) for x, y in zip(a, b))
    return type(a) is type(b)


@PROPERTY_SETTINGS
@given(JSON_VALUE)
def test_redact_value_keeps_shape_and_is_idempotent(value):
    once, _ = secret_redaction.redact_value(value)
    assert same_shape(value, once)
    twice, changed = secret_redaction.redact_value(once)
    assert twice == once
    assert not changed


@PROPERTY_SETTINGS
@given(MIXED_TEXT)
def test_redact_value_leaves_path_keys_alone(text):
    value = {"path": text, "file_path": [text], "content": text}
    out, _ = secret_redaction.redact_value(value)
    assert out["path"] == text
    assert out["file_path"] == [text]
    assert out["content"] == secret_redaction.redact_secrets(text)[0]


def test_property_settings_are_reproducible_by_default(monkeypatch):
    if os.environ.get("TAM_PROPERTY_RANDOM") == "1":
        pytest.skip("random exploration explicitly requested via TAM_PROPERTY_RANDOM=1")
    assert PROPERTY_SETTINGS.derandomize is True
    assert PROPERTY_SETTINGS.database is None
