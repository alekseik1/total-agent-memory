import sqlite3
from pathlib import Path

import pytest

from memory_core.schema_migration import Migration, MigrationRunner


@pytest.fixture
def database():
    db = sqlite3.connect(":memory:")
    db.executescript(
        "CREATE TABLE knowledge(id INTEGER PRIMARY KEY, content TEXT);"
        "CREATE TABLE graph_nodes(id TEXT PRIMARY KEY);"
        "CREATE TABLE knowledge_nodes(knowledge_id INTEGER REFERENCES knowledge(id),"
        "node_id TEXT REFERENCES graph_nodes(id),role TEXT,strength REAL,"
        "PRIMARY KEY(knowledge_id,node_id));"
        "CREATE TABLE migrations(version TEXT PRIMARY KEY,description TEXT,applied_at TEXT);"
        "INSERT INTO knowledge VALUES(1,'original');"
        "INSERT INTO graph_nodes VALUES('node');"
        "INSERT INTO knowledge_nodes VALUES(1,'node','provides',0.9);"
        "INSERT INTO knowledge_nodes VALUES(2,'node','mentions',0.7);"
        "INSERT INTO knowledge_nodes VALUES(1,'missing','related',0.5);"
    )
    script = (
        Path(__file__).parents[1] / "migrations/031_graph_link_integrity.sql"
    ).read_text()
    MigrationRunner(db).apply(Migration("031", "graph link integrity", script))
    yield db
    db.close()


def test_archives_exact_orphan_rows_and_preserves_valid_links(database):
    assert database.execute("PRAGMA foreign_keys").fetchone()[0] == 0
    assert database.execute("PRAGMA foreign_key_check").fetchall() == []
    assert database.execute("SELECT * FROM knowledge_nodes").fetchall() == [
        (1, "node", "provides", 0.9)
    ]
    archived = database.execute(
        "SELECT knowledge_id,node_id,role,strength FROM knowledge_nodes_quarantine ORDER BY knowledge_id"
    ).fetchall()
    assert archived == [(1, "missing", "related", 0.5), (2, "node", "mentions", 0.7)]
    assert database.execute("SELECT content FROM knowledge").fetchone()[0] == "original"


@pytest.mark.parametrize("identity,node", [(2, "node"), (1, "missing"), (None, "node")])
@pytest.mark.parametrize(
    "operation",
    [
        "INSERT OR REPLACE INTO knowledge_nodes(knowledge_id,node_id) VALUES (?,?)",
        "UPDATE knowledge_nodes SET knowledge_id=?,node_id=?",
    ],
)
def test_rejects_invalid_links_without_foreign_keys(
    database, identity, node, operation
):
    with pytest.raises(sqlite3.IntegrityError, match="requires existing"):
        database.execute(operation, (identity, node))
    assert database.execute("SELECT count(*) FROM knowledge_nodes").fetchone()[0] == 1


@pytest.mark.parametrize("table", ["knowledge", "graph_nodes"])
def test_parent_delete_cleans_links_without_foreign_keys(database, table):
    database.execute(f"DELETE FROM {table}")
    assert database.execute("SELECT count(*) FROM knowledge_nodes").fetchone()[0] == 0
    assert (
        database.execute("SELECT count(*) FROM knowledge_nodes_quarantine").fetchone()[
            0
        ]
        == 2
    )


@pytest.mark.parametrize("table,identity", [("knowledge", 9), ("graph_nodes", "other")])
def test_parent_identity_change_cannot_orphan_links(database, table, identity):
    with pytest.raises(sqlite3.IntegrityError, match="Cannot change linked"):
        database.execute(f"UPDATE {table} SET id=?", (identity,))
    assert database.execute("PRAGMA foreign_key_check").fetchall() == []
