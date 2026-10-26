"""Consolidation cosine dedup: embedding spaces/models/dimensions never mix, differing values never merge."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from workers.consolidation_daemon import _cosine_dedup  # noqa: E402

TEXT_DIM = 8
CODE_DIM = 5


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.execute("CREATE TABLE knowledge(id INTEGER PRIMARY KEY, project TEXT, type TEXT, content TEXT, "
              "status TEXT, superseded_by INT, created_at TEXT)")
    c.execute("CREATE TABLE embeddings(knowledge_id INT, float32_vector BLOB, embed_model TEXT, embedding_space TEXT)")
    yield c
    c.close()


def _add(conn, kid, content, vec, *, model="text-model", space="text", day=1):
    conn.execute("INSERT INTO knowledge VALUES(?,?,?,?,?,NULL,?)",
                 (kid, "p", "fact", content, "active", f"2026-03-{day:02d}"))
    conn.execute("INSERT INTO embeddings VALUES(?,?,?,?)",
                 (kid, np.asarray(vec, dtype=np.float32).tobytes(), model, space))


def _status(conn):
    return dict(conn.execute("SELECT id, status FROM knowledge").fetchall())


def _unit(dim):
    v = np.zeros(dim, dtype=np.float32)
    v[0] = 1.0
    return v


def test_mixed_dimensions_do_not_crash_and_are_not_compared(conn):
    _add(conn, 1, "same words here", _unit(TEXT_DIM), day=2)
    _add(conn, 2, "same words here", _unit(CODE_DIM), model="code-model", space="code", day=1)
    _add(conn, 3, "same words here", _unit(CODE_DIM), model="other-model", space="text", day=1)
    assert _cosine_dedup(conn, "p", pause_check=lambda: False) == 0
    assert set(_status(conn).values()) == {"active"}


def test_same_model_duplicates_still_merge(conn):
    _add(conn, 1, "deploy runs nightly", _unit(TEXT_DIM), day=2)
    _add(conn, 2, "deploy runs nightly", _unit(TEXT_DIM), day=1)
    assert _cosine_dedup(conn, "p", pause_check=lambda: False) == 1
    assert _status(conn) == {1: "active", 2: "merged"}


def test_high_cosine_with_different_numbers_does_not_merge(conn):
    _add(conn, 1, "rate limit is 500 rpm", _unit(TEXT_DIM), day=2)
    _add(conn, 2, "rate limit is 100 rpm", _unit(TEXT_DIM), day=1)
    assert _cosine_dedup(conn, "p", pause_check=lambda: False) == 0
