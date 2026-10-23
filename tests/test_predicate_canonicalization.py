"""Predicate alias groups in the temporal KG (src/predicate_vocab.py + temporal_kg)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

import predicate_vocab
from predicate_vocab import PredicateVocabularyError, load, normalize, parse
from temporal_kg import TemporalKG


@pytest.fixture(autouse=True)
def _fresh_vocabulary(monkeypatch):
    monkeypatch.delenv(predicate_vocab.VOCAB_PATH_ENV, raising=False)
    monkeypatch.delenv(predicate_vocab.VOCAB_EXTRA_ENV, raising=False)
    predicate_vocab.reset_cache()
    yield
    predicate_vocab.reset_cache()


@pytest.fixture
def tkg():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript((Path(__file__).parent.parent / "migrations" / "008_temporal_kg.sql").read_text())
    yield TemporalKG(conn)
    conn.close()


def _current(tkg, subject):
    return [(row["predicate"], row["object"]) for row in tkg.get_current(subject=subject)]


# ─── vocabulary ─────────────────────────────────────────────────────────────

def test_bundled_vocabulary_groups_english_and_russian_forms():
    vocab = load()
    for form in ("city", "current_city", "location", "Lives In", "lives-in", "живет_в", "город"):
        assert vocab.canonical(form) == "lives_in", form
    for form in ("job", "employer", "works_for", "место_работы", "работодатель"):
        assert vocab.canonical(form) == "works_at", form
    assert vocab.canonical("Uses") == "uses"
    assert not vocab.is_known("uses")


def test_match_forms_keep_the_given_predicate_first():
    forms = load().match_forms("City")
    assert forms[0] == "City" and "lives_in" in forms and "город" in forms
    assert load().match_forms("depends_on") == ("depends_on",)


def test_conflicting_vocabulary_is_rejected():
    with pytest.raises(PredicateVocabularyError):
        parse("lives_in, city\nhometown, city\n")
    with pytest.raises(PredicateVocabularyError):
        parse("lives_in, city\ncity, town\n")


def test_extra_vocabulary_file_extends_the_bundled_one(tmp_path, monkeypatch, tkg):
    extra = tmp_path / "team_predicates.txt"
    extra.write_text("# team additions\nowns_service, service_owner, отвечает_за\n", encoding="utf-8")
    monkeypatch.setenv(predicate_vocab.VOCAB_EXTRA_ENV, str(extra))
    predicate_vocab.reset_cache()
    tkg.add_fact("alice", "service_owner", "billing")
    tkg.add_fact("alice", "отвечает_за", "payments")
    assert _current(tkg, "alice") == [("отвечает_за", "payments")]
    # The bundled groups still apply alongside.
    assert predicate_vocab.get_vocabulary().canonical("city") == "lives_in"


def test_missing_vocabulary_degrades_to_exact_matching(tmp_path, monkeypatch, tkg):
    monkeypatch.setenv(predicate_vocab.VOCAB_PATH_ENV, str(tmp_path / "absent.txt"))
    predicate_vocab.reset_cache()
    tkg.add_fact("bob", "city", "Berlin")
    tkg.add_fact("bob", "lives_in", "Paris")
    assert sorted(_current(tkg, "bob")) == [("city", "Berlin"), ("lives_in", "Paris")]


def test_normalize():
    assert normalize("  Current-City ") == "current_city"
    assert normalize("works  at") == "works_at"


# ─── temporal KG behaviour ──────────────────────────────────────────────────

def test_alias_supersedes_and_keeps_the_original_predicate(tkg):
    first = tkg.add_fact("user", "city", "Berlin", valid_from="2026-01-01T00:00:00Z")
    second = tkg.add_fact("user", "lives_in", "Paris", valid_from="2026-03-01T00:00:00Z")
    third = tkg.add_fact("user", "живет_в", "Тбилиси", valid_from="2026-06-01T00:00:00Z")
    assert _current(tkg, "user") == [("живет_в", "Тбилиси")]
    history = tkg.timeline("user")
    assert [(row["id"], row["predicate"], row["valid_to"], row["superseded_by"]) for row in history] == [
        (first, "city", "2026-03-01T00:00:00Z", second),
        (second, "lives_in", "2026-06-01T00:00:00Z", third),
        (third, "живет_в", None, None),
    ]


def test_alias_matches_in_dedup_and_filters(tkg):
    first = tkg.add_fact("user", "employer", "Acme")
    assert tkg.add_fact("user", "works_at", "Acme") == first  # same fact, other wording
    assert [row["id"] for row in tkg.get_current(subject="user", predicate="job")] == [first]
    assert [row["id"] for row in tkg.timeline("user", predicate="работает_в")] == [first]
    assert tkg.invalidate_fact("user", "works_for", "Acme") == 1
    assert tkg.get_current(subject="user") == []


def test_unrelated_predicates_do_not_invalidate_each_other(tkg):
    tkg.add_fact("user", "city", "Berlin")
    tkg.add_fact("user", "works_at", "Acme")
    tkg.add_fact("user", "uses", "Vim")
    assert sorted(_current(tkg, "user")) == [("city", "Berlin"), ("uses", "Vim"), ("works_at", "Acme")]


def test_out_of_order_alias_insert_keeps_bitemporal_intervals(tkg):
    # The later fact arrives first; the earlier one is inserted afterwards.
    late = tkg.add_fact("user", "lives_in", "Lisbon", valid_from="2026-05-01T00:00:00Z")
    early = tkg.add_fact("user", "city", "Madrid", valid_from="2026-02-01T00:00:00Z")
    assert _current(tkg, "user") == [("lives_in", "Lisbon")]
    rows = {row["id"]: row for row in tkg.timeline("user")}
    assert rows[early]["valid_to"] == "2026-05-01T00:00:00Z" and rows[early]["superseded_by"] == late
    assert rows[late]["valid_to"] is None
    at_march = tkg.query_at("2026-03-15T00:00:00Z", subject="user", predicate="current_city")
    assert [(row["predicate"], row["object"]) for row in at_march] == [("city", "Madrid")]
    at_june = tkg.query_at("2026-06-15T00:00:00Z", subject="user", predicate="город")
    assert [(row["predicate"], row["object"]) for row in at_june] == [("lives_in", "Lisbon")]


def test_middle_insert_between_aliases_splits_the_interval(tkg):
    jan = tkg.add_fact("user", "city", "Oslo", valid_from="2026-01-01T00:00:00Z")
    jun = tkg.add_fact("user", "lives_in", "Rome", valid_from="2026-06-01T00:00:00Z")
    mar = tkg.add_fact("user", "location", "Vienna", valid_from="2026-03-01T00:00:00Z")
    rows = {row["id"]: row for row in tkg.timeline("user")}
    assert rows[jan]["valid_to"] == "2026-03-01T00:00:00Z" and rows[jan]["superseded_by"] == mar
    assert rows[mar]["valid_to"] == "2026-06-01T00:00:00Z" and rows[mar]["superseded_by"] == jun
    assert rows[jun]["valid_to"] is None


def test_diff_reports_an_alias_change_as_a_change(tkg):
    tkg.add_fact("user", "city", "Berlin", valid_from="2026-01-01T00:00:00Z")
    tkg.add_fact("user", "lives_in", "Paris", valid_from="2026-03-01T00:00:00Z")
    result = tkg.diff("2026-02-01T00:00:00Z", "2026-04-01T00:00:00Z")
    assert result["added"] == [] and result["removed"] == []
    ((change,),) = (result["changed"],)
    assert (change["predicate"], change["from_object"], change["to_object"]) == ("lives_in", "Berlin", "Paris")
    assert change["assertion"]["predicate"] == "lives_in"
