"""Tests for session_end(auto_compress=True) — LLM-driven compaction.

No real network: provider is mocked via unittest.mock.patch.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from session_continuity import SessionContinuity  # noqa: E402
from base_schema import apply_full_schema  # noqa: E402


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────


@pytest.fixture
def sc_db():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    apply_full_schema(conn)
    yield conn
    conn.close()


@pytest.fixture
def sc(sc_db):
    return SessionContinuity(sc_db)


def _latest_row(db: sqlite3.Connection) -> sqlite3.Row:
    return db.execute(
        "SELECT * FROM session_summaries ORDER BY rowid DESC LIMIT 1"
    ).fetchone()


# ──────────────────────────────────────────────
# Happy path: provider returns JSON → fields saved
# ──────────────────────────────────────────────


def test_auto_compress_uses_llm_provider(sc, sc_db):
    """Mock provider.complete → JSON; assert session_summaries row reflects it."""
    fake_provider = MagicMock()
    fake_provider.name = "openai"
    fake_provider.available.return_value = True
    fake_provider.complete.return_value = json.dumps(
        {
            "summary": "Wired provider abstraction into all callers.",
            "next_steps": ["Add benchmark run", "Document new envs"],
            "pitfalls": ["Remember to clear cache in tests"],
        }
    )

    with patch("llm_provider.make_provider", return_value=fake_provider):
        result = sc.session_end(
            "sess_ac",
            auto_compress=True,
            transcript="user: do X\nassistant: did X, tested, passed",
            project="p",
        )

    row = _latest_row(sc_db)
    assert row["summary"] == "Wired provider abstraction into all callers."
    assert json.loads(row["next_steps"]) == ["Add benchmark run", "Document new envs"]
    assert json.loads(row["pitfalls"]) == ["Remember to clear cache in tests"]
    assert result["compressed_used"] is True
    assert "auto_compress_error" not in result
    fake_provider.complete.assert_called_once()


# ──────────────────────────────────────────────
# Explicit args override LLM output
# ──────────────────────────────────────────────


def test_auto_compress_explicit_args_override_llm(sc, sc_db):
    """Explicit summary is preserved; LLM-derived summary is discarded."""
    fake_provider = MagicMock()
    fake_provider.available.return_value = True
    fake_provider.complete.return_value = json.dumps(
        {
            "summary": "LLM-generated summary",
            "next_steps": ["llm-step"],
            "pitfalls": ["llm-pitfall"],
        }
    )

    with patch("llm_provider.make_provider", return_value=fake_provider):
        sc.session_end(
            "sess_ex",
            summary="Explicit summary wins",
            next_steps=["explicit step"],
            auto_compress=True,
            transcript="something",
            project="p",
        )

    row = _latest_row(sc_db)
    assert row["summary"] == "Explicit summary wins"
    assert json.loads(row["next_steps"]) == ["explicit step"]
    # pitfalls wasn't explicit → LLM value fills it.
    assert json.loads(row["pitfalls"]) == ["llm-pitfall"]


# ──────────────────────────────────────────────
# Fallback: provider unavailable
# ──────────────────────────────────────────────


def test_auto_compress_fallback_on_provider_unavailable(sc, sc_db, caplog):
    """provider.available()=False → save with what we have, don't call complete."""
    fake_provider = MagicMock()
    fake_provider.name = "openai"
    fake_provider.available.return_value = False

    with patch("llm_provider.make_provider", return_value=fake_provider):
        result = sc.session_end(
            "sess_na",
            summary="",  # will be replaced with "" in auto_compress path
            next_steps=["nonempty"],
            auto_compress=True,
            transcript="hi",
            project="p",
        )

    fake_provider.complete.assert_not_called()
    row = _latest_row(sc_db)
    # Explicit next_steps survives; summary empty because provider was out.
    assert json.loads(row["next_steps"]) == ["nonempty"]
    assert result["compressed_used"] is False
    assert result.get("auto_compress_error") == "provider_unavailable"


# ──────────────────────────────────────────────
# Fallback: malformed JSON
# ──────────────────────────────────────────────


def test_auto_compress_fallback_on_malformed_json(sc, sc_db):
    """LLM returns garbage → explicit args saved, compressed sections empty."""
    fake_provider = MagicMock()
    fake_provider.available.return_value = True
    fake_provider.complete.return_value = "definitely not json {{{"

    with patch("llm_provider.make_provider", return_value=fake_provider):
        result = sc.session_end(
            "sess_bad",
            summary="keep me",
            auto_compress=True,
            transcript="noise",
            project="p",
        )

    row = _latest_row(sc_db)
    assert row["summary"] == "keep me"
    # next_steps / pitfalls default to [] when neither explicit nor valid LLM output
    assert json.loads(row["next_steps"]) == []
    assert json.loads(row["pitfalls"]) == []
    assert result["compressed_used"] is False
    assert result.get("auto_compress_error") == "malformed_json"


# ──────────────────────────────────────────────
# auto_compress=False: unchanged legacy behavior
# ──────────────────────────────────────────────


def test_auto_compress_false_by_default_does_not_call_llm(sc, sc_db):
    sentinel = MagicMock()
    sentinel.available.return_value = True
    sentinel.complete.return_value = "should-not-be-read"

    with patch("session_continuity.make_provider", return_value=sentinel, create=True):
        sc.session_end(
            "sess_plain",
            summary="explicit",
            next_steps=["do-a", "do-b"],
            project="p",
        )

    sentinel.complete.assert_not_called()
    sentinel.available.assert_not_called()
    row = _latest_row(sc_db)
    assert row["summary"] == "explicit"
    assert json.loads(row["next_steps"]) == ["do-a", "do-b"]
