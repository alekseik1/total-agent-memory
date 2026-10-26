"""obsidian_sync: export stays inside Memory/, import redacts and validates types."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tools import obsidian_sync as osync  # noqa: E402

PASSWORD = "s3cr3t-pass-value"


@pytest.fixture
def store(tmp_path, monkeypatch):
    db_path = tmp_path / "memory.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE knowledge(id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, type TEXT, content TEXT,
            context TEXT, project TEXT, tags TEXT DEFAULT '[]', status TEXT DEFAULT 'active', confidence REAL DEFAULT 1.0,
            source TEXT, created_at TEXT, last_confirmed TEXT, recall_count INTEGER DEFAULT 0);
        CREATE TABLE embedding_pending(knowledge_id INTEGER PRIMARY KEY, attempts INTEGER NOT NULL DEFAULT 0,
            last_error TEXT, next_attempt_at TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'pending',
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
        """
    )
    conn.commit()
    conn.close()
    monkeypatch.setattr(osync, "DB_PATH", str(db_path))
    return db_path


def test_dot_only_project_cannot_escape_memory_dir():
    for name in ("..", ".", "...", "../..", ""):
        safe = osync._safe_filename(name)
        assert safe not in ("", ".", "..")
        assert "/" not in safe
    assert osync._safe_filename("..") != osync._safe_filename("general")


def test_export_with_dotdot_project_stays_in_vault_memory_dir(store, tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "facts.md").write_text("user note, must survive", encoding="utf-8")
    conn = sqlite3.connect(store)
    conn.execute("INSERT INTO knowledge(type, content, project, created_at) VALUES ('fact', 'x', '..', '2026-01-01')")
    conn.commit()
    conn.close()
    osync.export_memory_to_obsidian(vault, None, None)
    assert (vault / "facts.md").read_text(encoding="utf-8") == "user note, must survive"
    assert any((vault / "Memory").rglob("facts.md"))


def test_import_redacts_secrets_and_queues_embedding(store, tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "note.md").write_text(
        f"---\nmemory: true\ntype: fact\nproject: demo\n---\nservice config\npassword: {PASSWORD}\n"
        "<private>my pin 1234</private>\n", encoding="utf-8"
    )
    imported, _ = osync.import_obsidian_to_memory(vault, None, [])
    assert imported == 1
    conn = sqlite3.connect(store)
    content = conn.execute("SELECT content FROM knowledge").fetchone()[0]
    assert PASSWORD not in content and "my pin 1234" not in content
    assert conn.execute("SELECT COUNT(*) FROM embedding_pending").fetchone()[0] == 1


def test_import_rejects_unknown_type(store, tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "note.md").write_text("---\nmemory: true\ntype: exploit\nproject: demo\n---\nplain body text\n", encoding="utf-8")
    osync.import_obsidian_to_memory(vault, None, [])
    conn = sqlite3.connect(store)
    assert conn.execute("SELECT type FROM knowledge").fetchone()[0] == "fact"
