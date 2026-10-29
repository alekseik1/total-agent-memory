"""Background writers: one redacting write path, TLS always verified, prompts never in argv, TAM memory dir."""

from __future__ import annotations

import re
import sqlite3
import ssl
import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC))

import background_writes as bw  # noqa: E402
import tool_support  # noqa: E402

PASSWORD = "p4ssw0rd-value-123"
PRIVATE_NOTE = "private note 98765"

# Files that may write to `knowledge` directly: the store itself, the shared helper, and
# code outside the background-writer set (derived/synthesised records, eval fixtures).
RAW_INSERT_ALLOWLIST = frozenset({
    "server.py", "background_writes.py", "auto_session_save.py", "temporal_index.py",
    "fact_merger.py", "fact_synthesizer.py", "tools/benchmark.py", "ambient/turn_end.py",
})
RAW_INSERT_RE = re.compile(r"INSERT\s+(?:OR\s+\w+\s+)?INTO\s+knowledge\b(?!_)", re.IGNORECASE)
LEGACY_DIR_PATTERNS = (
    re.compile(r"""os\.environ\.get\(\s*["']CLAUDE_MEMORY_DIR["']"""),
    re.compile(r"""Path\.home\(\)\s*/\s*["']\.claude-memory["']"""),
)


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.execute("CREATE TABLE knowledge(id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, type TEXT, content TEXT, "
              "context TEXT, project TEXT, tags TEXT, status TEXT DEFAULT 'active', confidence REAL, source TEXT, "
              "created_at TEXT, last_confirmed TEXT)")
    c.execute("CREATE TABLE embedding_pending(knowledge_id INTEGER PRIMARY KEY, attempts INTEGER NOT NULL DEFAULT 0, "
              "last_error TEXT, next_attempt_at TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'pending', "
              "created_at TEXT NOT NULL, updated_at TEXT NOT NULL)")
    yield c
    c.close()


def _save(conn, content, **kw):
    return bw.save_background_knowledge(conn, content=content, ktype="fact", project="p", session_id="s", **kw)


def test_secrets_private_blocks_and_context_are_cleaned(conn):
    saved = _save(conn, f"cfg password: {PASSWORD} <private>{PRIVATE_NOTE}</private>",
                  context=f"api_key: {PASSWORD}", tags=[f"token: {PASSWORD}"])
    row = conn.execute("SELECT content, context, tags FROM knowledge WHERE id=?", (saved.id,)).fetchone()
    joined = " ".join(row)
    assert PASSWORD not in joined and PRIVATE_NOTE not in joined


def test_unterminated_private_block_is_cut_to_the_end(conn):
    saved = _save(conn, f"visible text <private>{PRIVATE_NOTE}")
    assert PRIVATE_NOTE not in conn.execute("SELECT content FROM knowledge WHERE id=?", (saved.id,)).fetchone()[0]


def test_only_private_content_stores_nothing(conn):
    assert _save(conn, f"<private>{PRIVATE_NOTE}</private>") is None
    assert conn.execute("SELECT COUNT(*) FROM knowledge").fetchone()[0] == 0


def test_exact_duplicate_returns_existing_row(conn):
    first = _save(conn, "same body")
    second = _save(conn, "same body")
    assert first.created and not second.created and first.id == second.id


def test_new_row_is_queued_for_embedding(conn):
    saved = _save(conn, "needs a vector")
    assert conn.execute("SELECT knowledge_id FROM embedding_pending").fetchone()[0] == saved.id


def test_extra_column_names_are_validated(conn):
    with pytest.raises(ValueError):
        _save(conn, "x", extra={"content; DROP TABLE knowledge": 1})
    with pytest.raises(ValueError):
        _save(conn, "x", extra={"no_such_column": 1})


def test_no_raw_inserts_into_knowledge_outside_the_allowlist():
    offenders = []
    for path in SRC.rglob("*.py"):
        rel = path.relative_to(SRC).as_posix()
        if rel in RAW_INSERT_ALLOWLIST or "__pycache__" in rel:
            continue
        if RAW_INSERT_RE.search(path.read_text(encoding="utf-8")):
            offenders.append(rel)
    assert offenders == [], f"use background_writes.save_background_knowledge: {offenders}"


def test_tools_use_the_tam_memory_dir_not_the_legacy_default():
    offenders = []
    for folder in ("tools", "reflection", "ingestion", "workers"):
        for path in (SRC / folder).glob("*.py"):
            text = path.read_text(encoding="utf-8")
            if any(p.search(text) for p in LEGACY_DIR_PATTERNS):
                offenders.append(path.name)
    assert offenders == []


def test_ssl_context_always_verifies(monkeypatch):
    ctx = tool_support.make_ssl_context()
    assert ctx.verify_mode == ssl.CERT_REQUIRED and ctx.check_hostname is True
    # No certifi, and a host trust store that loads fine: the platform default is used. The
    # default context is stubbed so the result does not depend on the machine's certificate store
    # (one unparsable Windows certificate makes the real call raise "[ASN1] nested asn1 error").
    bundle = certifi_bundle()
    monkeypatch.setitem(sys.modules, "certifi", None)
    real = ssl.create_default_context
    monkeypatch.setattr(ssl, "create_default_context", lambda *a, **k: real(cafile=bundle))
    ctx = tool_support.make_ssl_context()
    assert ctx.verify_mode == ssl.CERT_REQUIRED and ctx.check_hostname is True


def certifi_bundle() -> str:
    import certifi

    return certifi.where()


def test_tool_ssl_contexts_verify():
    from tools import brain_autonomy, llm_router

    for module in (brain_autonomy, llm_router):
        ctx = module._make_ssl_context()
        assert ctx.verify_mode == ssl.CERT_REQUIRED and ctx.check_hostname is True


def test_no_module_disables_certificate_verification():
    for path in SRC.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "CERT_NONE" not in text or path.name == "test_background_writers.py", path
        assert "_create_unverified_context" not in text, path


def test_ollama_prompt_goes_through_stdin_not_argv(monkeypatch):
    seen = {}

    def fake_run(argv, **kwargs):
        seen["argv"], seen["input"] = argv, kwargs.get("input")
        return subprocess.CompletedProcess(argv, 0, stdout="\x1b[1mreply\x1b[0m\n", stderr="")

    monkeypatch.setattr(tool_support, "ollama_binary", lambda: "/opt/homebrew/bin/ollama")
    monkeypatch.setattr(tool_support.subprocess, "run", fake_run)
    out = tool_support.ollama_generate("secret decision text", model="m1")
    assert out == "reply"
    assert "secret decision text" not in " ".join(seen["argv"])
    assert seen["input"] == "secret decision text"


def test_ollama_binary_is_discovered_not_hardcoded(monkeypatch, tmp_path):
    fake = tmp_path / "ollama"
    fake.write_text("#!/bin/sh\n", encoding="utf-8")
    fake.chmod(0o755)
    monkeypatch.delenv(tool_support.OLLAMA_BIN_ENV, raising=False)
    monkeypatch.setattr(tool_support.shutil, "which", lambda name: str(fake))
    assert tool_support.ollama_binary() == str(fake)


def test_missing_ollama_binary_is_reported_not_raised(monkeypatch):
    messages = []
    monkeypatch.setattr(tool_support, "ollama_binary", lambda: None)
    assert tool_support.ollama_generate("p", log=messages.append) == ""
    assert messages and "not found" in messages[0]


def test_tools_have_no_hardcoded_ollama_path_or_argv_prompt():
    for path in (SRC / "tools").glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert '"/usr/local/bin/ollama"' not in text, path.name
        assert not re.search(r'\[\s*\w*OLLAMA_BIN\w*\s*,\s*"run"\s*,\s*\w+\s*,\s*prompt', text), path.name
    assert '"/usr/local/bin/ollama"' not in (SRC / "reflection" / "scheduler.py").read_text(encoding="utf-8")
