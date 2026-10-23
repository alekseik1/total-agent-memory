"""The atomic-fact backfill skips a record whose extraction fails and keeps going."""

from __future__ import annotations

import pytest

import server
from atomic_facts_backfill import backfill
from memory_core.atomic_facts import InvalidExtraction


@pytest.fixture
def database(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "MEMORY_DIR", tmp_path)
    store = server.Store(background_queues=False)
    store._embed_mode = "none"
    for identity in (1, 2, 3, 4):
        store.db.execute(
            "INSERT INTO knowledge(id,content,session_id,project,type,status,created_at,last_confirmed) "
            "VALUES (?,?,'s','p','fact','active','2026-01-01','2026-01-01')",
            (identity, f"record number {identity}"),
        )
    store.db.commit()
    store.db.close()
    return tmp_path / "memory.db"


class _Extractor:
    def __init__(self, repository):
        self.seen: list[int] = []

    def extract(self, knowledge_id: int) -> int:
        self.seen.append(knowledge_id)
        if knowledge_id == 2:
            raise InvalidExtraction("Target exceeds extraction character limit")
        if knowledge_id == 3:
            raise TimeoutError("The read operation timed out")
        return 1


def test_failed_records_are_skipped_and_counted(database):
    made = {}

    def factory(repository):
        made["extractor"] = _Extractor(repository)
        return made["extractor"]

    processed, created, skipped = backfill(database, "p", 100, extractor_factory=factory)
    assert (processed, created, skipped) == (4, 2, 2)
    assert made["extractor"].seen == [1, 2, 3, 4]


def test_unexpected_errors_still_raise(database):
    class Broken:
        def __init__(self, repository):
            pass

        def extract(self, knowledge_id: int) -> int:
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        backfill(database, "p", 100, extractor_factory=Broken)
