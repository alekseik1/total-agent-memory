from __future__ import annotations

import hashlib
import logging
import sqlite3
import struct
from dataclasses import dataclass

from memory_core.telemetry import counters, op_timer

SOURCE_SQL = (
    "SELECT content,context,project,session_id,status,branch FROM knowledge WHERE id=?"
)
EMBEDDING_SQL = (
    "SELECT binary_vector,float32_vector,embed_model,embed_dim,created_at,"
    "embedding_provider,embedding_space,content_type,language "
    "FROM embeddings WHERE knowledge_id=?"
)


class RepairConflict(RuntimeError):
    pass


@dataclass(frozen=True)
class PreparedEmbedding:
    knowledge_id: int
    source_digest: str
    embedding_digest: str
    model: str
    vector: tuple[float, ...]


def row_digest(db: sqlite3.Connection, query: str, identity: int) -> str:
    row = db.execute(query, (identity,)).fetchone()
    return hashlib.sha256(
        repr(tuple(row) if row is not None else None).encode()
    ).hexdigest()


class IndexRepair:
    def __init__(self, db: sqlite3.Connection):
        self.db = db

    def apply(self, prepared: tuple[PreparedEmbedding, ...]) -> int:
        import math

        if self.db.in_transaction:
            raise RepairConflict("Repair requires an idle connection")
        identities = {row.knowledge_id for row in prepared}
        if len(identities) != len(prepared):
            raise ValueError("Duplicate repair identity")
        for row in prepared:
            if (
                not row.model
                or not row.vector
                or not all(math.isfinite(v) for v in row.vector)
            ):
                raise ValueError("Repair requires a model and finite non-empty vectors")
        with op_timer("index_repair_ms"):
            self.db.execute("BEGIN IMMEDIATE")
            try:
                for row in prepared:
                    if (
                        row_digest(self.db, SOURCE_SQL, row.knowledge_id)
                        != row.source_digest
                        or row_digest(self.db, EMBEDDING_SQL, row.knowledge_id)
                        != row.embedding_digest
                    ):
                        raise RepairConflict(
                            f"Record {row.knowledge_id} changed since preparation"
                        )
                    source = self.db.execute(SOURCE_SQL, (row.knowledge_id,)).fetchone()
                    if source is None or source[4] != "active":
                        raise RepairConflict(
                            f"Record {row.knowledge_id} is no longer active"
                        )
                    existing = self.db.execute(
                        EMBEDDING_SQL, (row.knowledge_id,)
                    ).fetchone()
                    if existing is not None and existing[6] not in (None, "text"):
                        raise RepairConflict(
                            f"Record {row.knowledge_id} is not in text space"
                        )
                    binary = bytearray((len(row.vector) + 7) // 8)
                    for position, value in enumerate(row.vector):
                        if value > 0:
                            binary[position // 8] |= 1 << (7 - position % 8)
                    self.db.execute(
                        "INSERT INTO embeddings(knowledge_id,binary_vector,float32_vector,"
                        "embed_model,embed_dim,created_at,embedding_provider,embedding_space,"
                        "content_type,language) VALUES(?,?,?,?,?,strftime('%Y-%m-%dT%H:%M:%fZ','now'),"
                        "'fastembed','text','text',NULL) "
                        "ON CONFLICT(knowledge_id) DO UPDATE SET "
                        "binary_vector=excluded.binary_vector,float32_vector=excluded.float32_vector,"
                        "embed_model=excluded.embed_model,embed_dim=excluded.embed_dim,"
                        "created_at=excluded.created_at,embedding_provider=excluded.embedding_provider,"
                        "embedding_space=excluded.embedding_space",
                        (
                            row.knowledge_id,
                            bytes(binary),
                            struct.pack(f"{len(row.vector)}f", *row.vector),
                            row.model,
                            len(row.vector),
                        ),
                    )
                self.db.commit()
            except (
                sqlite3.Error,
                RepairConflict,
                ValueError,
                OverflowError,
                struct.error,
            ):
                self.db.rollback()
                counters.bump("index_repair_errors")
                logging.getLogger(__name__).exception("Index repair rolled back")
                raise
        counters.bump("index_repair_records", len(prepared))
        return len(prepared)
