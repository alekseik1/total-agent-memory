import sqlite3

import pytest

from memory_core.fts_maintenance import maintain_fts


@pytest.mark.parametrize("optimize", [False, True])
def test_maintenance_preserves_ranked_results(optimize):
    with sqlite3.connect(":memory:") as db:
        db.execute("CREATE VIRTUAL TABLE knowledge_fts USING fts5(content)")
        db.executemany(
            "INSERT INTO knowledge_fts VALUES (?)",
            [("rabbit carrots",), ("rabbit rabbit",)],
        )
        before = db.execute(
            "SELECT rowid FROM knowledge_fts WHERE knowledge_fts MATCH 'rabbit' ORDER BY rank"
        ).fetchall()
        assert maintain_fts(db, optimize=optimize) == 1
        after = db.execute(
            "SELECT rowid FROM knowledge_fts WHERE knowledge_fts MATCH 'rabbit' ORDER BY rank"
        ).fetchall()
        assert before == after
