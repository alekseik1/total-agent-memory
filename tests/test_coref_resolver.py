"""Tests for the v10 coreference resolver."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import coref_resolver as cr
from base_schema import apply_full_schema

# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    cr._reset_provider_cache()
    # Default: enable explicitly so tests don't need to opt in each time.
    monkeypatch.setenv("MEMORY_COREF_ENABLED", "true")
    monkeypatch.delenv("MEMORY_COREF_HISTORY_LIMIT", raising=False)
    monkeypatch.delenv("MEMORY_COREF_TIMEOUT_SEC", raising=False)
    yield
    cr._reset_provider_cache()


@pytest.fixture
def hist_db():
    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    apply_full_schema(db)
    yield db
    db.close()


def _seed(db, session_id, *contents):
    for c in contents:
        db.execute(
            "INSERT INTO knowledge (session_id, project, type, content, status, created_at) "
            "VALUES (?, 'test', 'fact', ?, 'active', '2026-04-14T00:00:00Z')",
            (session_id, c),
        )
    db.commit()


class _FakeProvider:
    def __init__(self, response="", available=True, raises=None):
        self.name = "fake"
        self._response = response
        self._available = available
        self._raises = raises
        self.calls = []

    def available(self) -> bool:
        return self._available

    def complete(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        if self._raises:
            raise self._raises
        return self._response


@pytest.fixture
def fake_provider(monkeypatch):
    def _make(response="", available=True, raises=None):
        prov = _FakeProvider(response=response, available=available, raises=raises)
        monkeypatch.setattr(cr, "_get_provider", lambda: prov)
        return prov
    return _make


# ──────────────────────────────────────────────
# needs_resolution — pre-filter
# ──────────────────────────────────────────────


def test_needs_resolution_detects_english_pronouns():
    assert cr.needs_resolution("after this it broke")
    assert cr.needs_resolution("They merged that PR")
    assert cr.needs_resolution("we deprecated it")


def test_needs_resolution_detects_russian_pronouns():
    assert cr.needs_resolution("после этого всё сломалось")
    assert cr.needs_resolution("он не запустился")
    assert cr.needs_resolution("это заработало")


def test_needs_resolution_negative():
    assert not cr.needs_resolution("Migration 422000001 broke batchUpsert")
    assert not cr.needs_resolution("")
    assert not cr.needs_resolution("Set DECAY_HALF_LIFE=90 in env")


def test_history_is_project_scoped(fake_provider, hist_db):
    _seed(hist_db, "shared", "Alice deployed the parser")
    hist_db.execute(
        "INSERT INTO knowledge (session_id, project, type, content, created_at) "
        "VALUES (?, ?, 'fact', ?, '2026-04-14T00:00:00Z')",
        ("shared", "other", "Bob deployed the SECRET compiler"),
    )
    provider = fake_provider(response="Alice deployed the parser and it failed")
    cr.resolve("After this it failed", db=hist_db, session_id="shared", project="test")
    assert "Alice" in provider.calls[0][0]
    assert "SECRET" not in provider.calls[0][0]


def test_missing_project_does_not_use_global_history(fake_provider, hist_db):
    _seed(hist_db, "shared", "Alice deployed the parser")
    provider = fake_provider(response="should never run")
    result = cr.resolve("After this it failed", db=hist_db, session_id="shared")
    assert result.decision == "skip"
    assert provider.calls == []


# ──────────────────────────────────────────────
# resolve — gating paths (no LLM call)
# ──────────────────────────────────────────────


def test_resolve_skips_when_disabled(monkeypatch, fake_provider, hist_db):
    monkeypatch.setenv("MEMORY_COREF_ENABLED", "false")
    prov = fake_provider(response="should not be called")
    out = cr.resolve("after this it broke", db=hist_db, project="test", session_id="s1")
    assert out.decision == "skip"
    assert "disabled" in out.reason
    assert prov.calls == []
    assert out.content == "after this it broke"


def test_resolve_noops_when_no_pronouns(fake_provider, hist_db):
    prov = fake_provider(response="should not be called")
    out = cr.resolve(
        "Migration 422000001 broke batchUpsert in vitamin_all dev branch",
        db=hist_db, project="test", session_id="s1",
    )
    assert out.decision == "noop"
    assert prov.calls == []


def test_resolve_skips_oversized_input(fake_provider, hist_db):
    prov = fake_provider(response="never called")
    huge = "after this it broke. " * 500  # ~10k chars
    out = cr.resolve(huge, db=hist_db, project="test", session_id="s1")
    assert out.decision == "skip"
    assert "too long" in out.reason
    assert prov.calls == []


def test_resolve_skips_when_no_history(fake_provider, hist_db):
    prov = fake_provider(response="never called")
    # No seeding → no history.
    out = cr.resolve("after this it broke", db=hist_db, project="test", session_id="empty-sess")
    assert out.decision == "skip"
    assert "no session history" in out.reason
    assert prov.calls == []


def test_resolve_skips_when_provider_unavailable(fake_provider, hist_db):
    fake_provider(available=False)
    _seed(hist_db, "s1", "Earlier note about migration 422000001")
    out = cr.resolve("after this it broke", db=hist_db, project="test", session_id="s1")
    assert out.decision == "skip"
    assert "unavailable" in out.reason


# ──────────────────────────────────────────────
# resolve — LLM path
# ──────────────────────────────────────────────


def test_resolve_rewrites_with_context(fake_provider, hist_db):
    _seed(
        hist_db,
        "s1",
        "Applied migration 422000001 to staging",
        "Ran batchUpsert benchmark afterwards",
    )
    rewrite = (
        "After migration 422000001 the batchUpsert call broke "
        "with a deadlock on the staging database"
    )
    fake_provider(response=rewrite)
    out = cr.resolve(
        "after this it broke with a deadlock on the staging database",
        db=hist_db, project="test", session_id="s1",
    )
    assert out.decision == "rewritten"
    assert out.content == rewrite
    assert out.latency_ms is not None


def test_resolve_passes_history_to_llm(fake_provider, hist_db):
    _seed(hist_db, "s1", "Note A about migration X", "Note B about deadlock Y")
    prov = fake_provider(response="something completely different and longer than the input text here")
    cr.resolve("after this it broke", db=hist_db, project="test", session_id="s1")
    assert len(prov.calls) == 1
    prompt = prov.calls[0][0]
    # Both history snippets must reach the prompt (oldest-first).
    assert "Note A" in prompt
    assert "Note B" in prompt
    assert prompt.index("Note A") < prompt.index("Note B")


def test_resolve_returns_error_on_llm_failure(fake_provider, hist_db):
    _seed(hist_db, "s1", "Earlier note")
    fake_provider(raises=RuntimeError("provider down"))
    out = cr.resolve("after this it broke", db=hist_db, project="test", session_id="s1")
    assert out.decision == "error"
    assert out.content == "after this it broke"  # original preserved
    assert "provider down" in out.reason


def test_resolve_rejects_truncated_rewrite(fake_provider, hist_db):
    _seed(hist_db, "s1", "Earlier note")
    # Rewrite is much shorter than input → suspected truncation.
    long_input = "after this it broke " * 30  # ~600 chars
    fake_provider(response="broke")
    out = cr.resolve(long_input, db=hist_db, project="test", session_id="s1")
    assert out.decision == "error"
    assert "truncation" in out.reason
    assert out.content == long_input


def test_resolve_noops_when_llm_returns_identical(fake_provider, hist_db):
    _seed(hist_db, "s1", "Earlier note")
    text = "after this it broke"
    fake_provider(response=text)
    out = cr.resolve(text, db=hist_db, project="test", session_id="s1")
    # Identical text → bumped from rewritten to noop (would also fail length check).
    assert out.decision in ("noop", "error")


def test_resolve_strips_markdown_fence(fake_provider, hist_db):
    _seed(hist_db, "s1", "Migration 422000001 deployed to prod")
    fenced = (
        "```\nAfter migration 422000001 the batchUpsert call broke "
        "with a deadlock on the staging database\n```"
    )
    fake_provider(response=fenced)
    out = cr.resolve(
        "after this it broke with a deadlock on the staging database",
        db=hist_db, project="test", session_id="s1",
    )
    assert out.decision == "rewritten"
    assert "```" not in out.content


def test_resolve_returns_error_on_empty_llm_response(fake_provider, hist_db):
    _seed(hist_db, "s1", "Earlier note")
    fake_provider(response="   ")
    out = cr.resolve(
        "after this it broke unexpectedly during deploy",
        db=hist_db, project="test", session_id="s1",
    )
    assert out.decision == "error"
    assert "empty" in out.reason


def test_prompt_instructs_language_preservation(fake_provider, hist_db):
    """LLM must keep the input language — RU stays RU, EN stays EN.

    Regression: 2026-04-27 qwen2.5-coder:7b translated a Russian record into
    English because the prompt only described the rewrite mechanic, never
    pinned the output language.
    """
    _seed(hist_db, "s1", "Запустил Postgres 18 для total-agent-memory")
    prov = fake_provider(
        response="После того как настроил Postgres 18, индексы по embeddings показали 0.85 cosine"
    )
    cr.resolve(
        "После того как настроил его, индексы по embeddings показали 0.85 cosine",
        db=hist_db, project="test", session_id="s1",
    )
    assert len(prov.calls) == 1
    # Collapse whitespace so newlines in the prompt template don't hide phrases.
    import re
    prompt = re.sub(r"\s+", " ", prov.calls[0][0]).lower()
    # Language-preservation guard must be present in the prompt.
    assert "same language" in prompt
    assert "do not translate" in prompt


# ──────────────────────────────────────────────
# History truncation
# ──────────────────────────────────────────────


def test_history_truncates_long_records(fake_provider, hist_db):
    big = "x" * 1000
    _seed(hist_db, "s1", big)
    prov = fake_provider(response="placeholder rewrite that is at least as long as the original input here please")
    cr.resolve(
        "after this it broke and the deploy rolled back automatically",
        db=hist_db, project="test", session_id="s1",
    )
    prompt = prov.calls[0][0]
    # 240 char cap + "…" suffix → no full 1000-char dump.
    assert "x" * 1000 not in prompt
    assert "…" in prompt
