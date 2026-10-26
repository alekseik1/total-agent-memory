"""reembed.py rewrites a record's vector without touching its space metadata."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import reembed  # noqa: E402

MIGRATIONS = Path(__file__).parents[1] / "migrations"


@pytest.fixture
def db():
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE embeddings (knowledge_id INTEGER PRIMARY KEY, binary_vector BLOB NOT NULL, "
        "float32_vector BLOB NOT NULL, embed_model TEXT NOT NULL, embed_dim INTEGER NOT NULL, created_at TEXT NOT NULL)")
    conn.executescript((MIGRATIONS / "021_embedding_spaces.sql").read_text(encoding="utf-8"))
    yield conn
    conn.close()


def test_upsert_keeps_space_content_type_and_language(db):
    db.execute(
        "INSERT INTO embeddings VALUES (1, x'00', x'00000000', 'old', 1, 't0', 'fastembed', 'config', 'sql', 'en')")

    reembed.upsert_embedding(db, 1, [0.5, -0.5], "new", "fastembed", "t1")

    row = db.execute(
        "SELECT embed_model, embed_dim, embedding_provider, embedding_space, content_type, language "
        "FROM embeddings WHERE knowledge_id=1").fetchone()
    assert row == ("new", 2, "fastembed", "config", "sql", "en")


def test_upsert_records_the_backend_and_a_text_space_for_new_rows(db):
    reembed.upsert_embedding(db, 7, [1.0, 0.0], "model", "ollama", "t1")

    assert db.execute("SELECT embedding_provider, embedding_space FROM embeddings WHERE knowledge_id=7").fetchone() == (
        "ollama", "text")


def test_created_table_has_the_space_columns():
    conn = sqlite3.connect(":memory:")
    reembed.ensure_embeddings_table(conn)
    columns = {row[1] for row in conn.execute("PRAGMA table_info(embeddings)")}
    assert {"embedding_provider", "embedding_space", "content_type", "language"} <= columns
