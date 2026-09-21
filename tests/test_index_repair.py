import sqlite3

import pytest

from memory_core.index_repair import (
    EMBEDDING_SQL,
    SOURCE_SQL,
    IndexRepair,
    PreparedEmbedding,
    RepairConflict,
    row_digest,
)


@pytest.fixture
def database():
    db = sqlite3.connect(":memory:")
    db.executescript(
        "CREATE TABLE knowledge(id INTEGER PRIMARY KEY,content TEXT,context TEXT,project TEXT,"
        "session_id TEXT,status TEXT,branch TEXT);"
        "CREATE TABLE embeddings(knowledge_id INTEGER PRIMARY KEY,binary_vector BLOB,"
        "float32_vector BLOB,embed_model TEXT,embed_dim INTEGER,created_at TEXT,"
        "embedding_provider TEXT,embedding_space TEXT,content_type TEXT,language TEXT);"
        "INSERT INTO knowledge VALUES(1,'first','','p','s','active','');"
        "INSERT INTO knowledge VALUES(2,'second','','p','s','active','');"
    )
    yield db
    db.close()


def prepared(db):
    return tuple(
        PreparedEmbedding(
            i,
            row_digest(db, SOURCE_SQL, i),
            row_digest(db, EMBEDDING_SQL, i),
            "model",
            (1.0, -1.0),
        )
        for i in (1, 2)
    )


def test_repair_keeps_originals_and_stores_matching_binary_and_float_vectors(database):
    before = database.execute("SELECT * FROM knowledge").fetchall()
    assert IndexRepair(database).apply(prepared(database)) == 2
    assert database.execute("SELECT * FROM knowledge").fetchall() == before
    assert database.execute(
        "SELECT binary_vector,embed_dim,embedding_space FROM embeddings"
    ).fetchall() == [(b"\x80", 2, "text"), (b"\x80", 2, "text")]


@pytest.mark.parametrize(
    "change",
    [
        "UPDATE knowledge SET content='new' WHERE id=2",
        "UPDATE knowledge SET status='deleted' WHERE id=2",
        "DELETE FROM knowledge WHERE id=2",
        "INSERT INTO embeddings(knowledge_id,embed_model) VALUES(2,'new-model')",
    ],
)
def test_changed_source_or_vector_rolls_back_entire_batch(database, change):
    batch = prepared(database)
    database.execute(change)
    database.commit()
    before = database.execute("SELECT * FROM embeddings").fetchall()
    with pytest.raises(RepairConflict, match="changed"):
        IndexRepair(database).apply(batch)
    assert database.execute("SELECT * FROM embeddings").fetchall() == before
    assert not database.in_transaction


def test_non_text_space_is_never_overwritten(database):
    database.execute(
        "INSERT INTO embeddings(knowledge_id,embedding_space) VALUES(2,'code')"
    )
    database.commit()
    with pytest.raises(RepairConflict, match="not in text space"):
        IndexRepair(database).apply(prepared(database))
    assert database.execute("SELECT count(*) FROM embeddings").fetchone()[0] == 1


def test_nonfinite_vector_rejected_before_transaction(database):
    batch = prepared(database)
    bad = PreparedEmbedding(
        1, batch[0].source_digest, batch[0].embedding_digest, "model", (float("nan"),)
    )
    with pytest.raises(ValueError, match="finite"):
        IndexRepair(database).apply((bad,))
    assert not database.in_transaction


def test_existing_classification_is_preserved(database):
    database.execute(
        "INSERT INTO embeddings(knowledge_id,embedding_space,content_type,language) "
        "VALUES(1,'text','document','ru')"
    )
    database.commit()
    IndexRepair(database).apply(prepared(database))
    assert database.execute(
        "SELECT content_type,language FROM embeddings WHERE knowledge_id=1"
    ).fetchone() == ("document", "ru")
