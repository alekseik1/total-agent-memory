"""Integration tests: session_continuity.py + active_context.py.

Verifies the markdown live-doc projection is written on session_end and
surfaced through session_init response. All filesystem I/O is confined to
``tmp_path`` via ``MEMORY_ACTIVECONTEXT_VAULT``.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from active_context import read_active_context, write_active_context
from base_schema import apply_full_schema
from session_continuity import SessionContinuity


@pytest.fixture
def sc_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    # The full schema, not one hand-picked migration: session_end writes the
    # summary *and* closes the row in `sessions`, so a fixture carrying only
    # 010 tests a database no installation ever has.
    apply_full_schema(conn)
    yield conn
    conn.close()


@pytest.fixture
def sc(sc_db: sqlite3.Connection) -> SessionContinuity:
    return SessionContinuity(sc_db)


@pytest.fixture
def vault(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point activeContext vault to tmp_path and ensure markdown is enabled."""
    monkeypatch.setenv("MEMORY_ACTIVECONTEXT_VAULT", str(tmp_path))
    monkeypatch.delenv("MEMORY_ACTIVECONTEXT_DISABLE", raising=False)
    return tmp_path


# ──────────────────────────────────────────────
# session_end ↔ markdown
# ──────────────────────────────────────────────


def test_session_end_writes_markdown_when_enabled(
    sc: SessionContinuity, vault: Path
) -> None:
    r = sc.session_end(
        "sess_1",
        "Worked on temporal KG",
        next_steps=["Wire MCP", "Ship docs"],
        pitfalls=["sqlite lock"],
        project="ctm",
    )
    md_path = vault / "ctm" / "activeContext.md"
    assert md_path.exists()
    assert r.get("active_context_path") == str(md_path)
    text = md_path.read_text(encoding="utf-8")
    assert "Worked on temporal KG" in text
    assert "- Wire MCP" in text
    assert "- sqlite lock" in text
    assert "**Session:** sess_1" in text


def test_session_end_skips_markdown_when_disabled(
    sc: SessionContinuity,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MEMORY_ACTIVECONTEXT_VAULT", str(tmp_path))
    monkeypatch.setenv("MEMORY_ACTIVECONTEXT_DISABLE", "1")
    r = sc.session_end("s1", "sum", project="p", next_steps=["n"])
    assert "active_context_path" not in r
    assert not (tmp_path / "p" / "activeContext.md").exists()


@pytest.mark.parametrize("val", ["true", "yes", "1", "TRUE", "Yes"])
def test_disable_env_accepts_multiple_truthy_values(
    sc: SessionContinuity,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    val: str,
) -> None:
    monkeypatch.setenv("MEMORY_ACTIVECONTEXT_VAULT", str(tmp_path))
    monkeypatch.setenv("MEMORY_ACTIVECONTEXT_DISABLE", val)
    sc.session_end("s1", "sum", project="p")
    assert not (tmp_path / "p" / "activeContext.md").exists()


# ──────────────────────────────────────────────
# session_init ↔ markdown
# ──────────────────────────────────────────────


def test_session_init_reads_markdown_into_response(
    sc: SessionContinuity, vault: Path
) -> None:
    sc.session_end(
        "s1",
        "DB summary",
        next_steps=["n1"],
        pitfalls=["p1"],
        project="proj",
    )
    r = sc.session_init(project="proj")
    assert r is not None
    assert "active_context" in r
    ac = r["active_context"]
    assert ac is not None
    assert ac["summary"] == "DB summary"
    assert ac["next_steps"] == ["n1"]
    assert ac["pitfalls"] == ["p1"]
    assert r["markdown_updated_at"] is not None
    assert r["markdown_stale"] is False


def test_session_init_handles_missing_markdown_gracefully(
    sc: SessionContinuity, vault: Path
) -> None:
    # Write to DB only, bypassing markdown via the disable flag
    import os
    os.environ["MEMORY_ACTIVECONTEXT_DISABLE"] = "1"
    try:
        sc.session_end("s1", "only in db", project="p", next_steps=["x"])
    finally:
        os.environ.pop("MEMORY_ACTIVECONTEXT_DISABLE", None)

    r = sc.session_init(project="p")
    assert r is not None
    assert r["summary"] == "only in db"
    assert r["active_context"] is None
    assert r["markdown_updated_at"] is None


def test_session_init_returns_markdown_only_when_db_empty(
    sc: SessionContinuity, vault: Path
) -> None:
    # No DB row, but there is a markdown doc dropped by an external writer.
    write_active_context(
        "orphan",
        "from markdown",
        ["m-step"],
        ["m-pit"],
        vault_root=vault,
        session_id="ext",
    )
    r = sc.session_init(project="orphan")
    assert r is not None
    assert r["source"] == "markdown"
    assert r["summary"] == "from markdown"
    assert r["next_steps"] == ["m-step"]
    assert r["pitfalls"] == ["m-pit"]


def test_db_and_markdown_diverge_flags_stale_markdown(
    sc: SessionContinuity, vault: Path
) -> None:
    # Seed markdown with an old summary
    write_active_context(
        "proj",
        "OLD markdown summary",
        ["old step"],
        [],
        vault_root=vault,
        session_id="old",
    )
    # Write fresh row to DB directly, skipping the markdown write so the
    # file stays "stale" vs. the DB content.
    import os
    os.environ["MEMORY_ACTIVECONTEXT_DISABLE"] = "1"
    try:
        sc.session_end(
            "new-sess",
            "NEW db summary",
            project="proj",
            next_steps=["new step"],
        )
    finally:
        os.environ.pop("MEMORY_ACTIVECONTEXT_DISABLE", None)

    r = sc.session_init(project="proj")
    assert r is not None
    # DB wins on summary content
    assert r["summary"] == "NEW db summary"
    # But markdown_stale flag is raised
    assert r["markdown_stale"] is True
    assert r["markdown_updated_at"] is not None
    assert r["active_context"]["summary"] == "OLD markdown summary"


def test_session_init_returns_none_when_both_empty(
    sc: SessionContinuity, vault: Path
) -> None:
    assert sc.session_init(project="empty") is None


def test_session_init_include_pitfalls_false_with_markdown_only(
    sc: SessionContinuity, vault: Path
) -> None:
    write_active_context(
        "p",
        "sum",
        [],
        ["hidden"],
        vault_root=vault,
    )
    r = sc.session_init(project="p", include_pitfalls=False)
    assert r is not None
    assert r["pitfalls"] == []
    # Raw active_context still carries the original pitfalls
    assert r["active_context"]["pitfalls"] == ["hidden"]
