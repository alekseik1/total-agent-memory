"""memory_save(supersede=true): a new value retires the stored one."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from memory_core.dedup import updates_value

VALUE_UPDATES = [
    ("Lionel Messi's country of citizenship is Armenia", "Lionel Messi's country of citizenship is Argentina"),
    ("Сервис billing работает на PostgreSQL 18", "Сервис billing работает на PostgreSQL 16"),
    ("Маша любит зелёный цвет", "Маша любит красный цвет"),
    ("Israel was founded by Philippe, Duke of Orléans", "Israel was founded by David Ben-Gurion"),
]

NOT_UPDATES = [
    # the subject differs, the value is shared
    ("The company that produced Chevrolet Caprice is General Motors",
     "The company that produced Buick Riviera is General Motors"),
    # the subject differs inside the prefix
    ("The capital of Tang Dynasty is Y", "The capital of Tang Empire is X"),
    ("Bob likes red", "Alice likes red"),
    # a different relation
    ("Monk was written in the language of Hebrew", "Monk was written by Matthew Lewis"),
    # the same statement
    ("Deploy runs nightly", "deploy runs nightly."),
]


@pytest.mark.parametrize(("new", "old"), VALUE_UPDATES)
def test_value_update_is_detected(new, old):
    assert updates_value(new, old)


@pytest.mark.parametrize(("new", "old"), NOT_UPDATES)
def test_other_changes_are_not_value_updates(new, old):
    assert not updates_value(new, old)


@pytest.fixture
def srv(monkeypatch, tmp_path):
    (tmp_path / "blobs").mkdir(exist_ok=True)
    (tmp_path / "chroma").mkdir(exist_ok=True)
    import server

    monkeypatch.setattr(server, "MEMORY_DIR", tmp_path)
    monkeypatch.setattr(server, "store", server.Store())
    monkeypatch.setattr(server, "recall", server.Recall(server.store))
    monkeypatch.setattr(server, "SID", "test")
    yield server
    server.store.db.close()


def save(srv, content, project="p", **extra):
    out = asyncio.run(srv._do("memory_save", {"content": content, "type": "fact", "project": project, **extra}))
    return json.loads(out)


def active(srv):
    return [row[0] for row in srv.store.db.execute("SELECT content FROM knowledge WHERE status='active' ORDER BY id")]


def test_without_flag_both_values_stay(srv):
    old, new = VALUE_UPDATES[0][1], VALUE_UPDATES[0][0]
    save(srv, old)
    assert "superseded" not in save(srv, new)
    assert active(srv) == [old, new]


def test_flag_retires_the_old_value(srv):
    old, new = VALUE_UPDATES[0][1], VALUE_UPDATES[0][0]
    old_id = save(srv, old)["id"]
    result = save(srv, new, supersede=True)
    assert result["superseded"] == [old_id]
    assert active(srv) == [new]
    status, by = srv.store.db.execute("SELECT status, superseded_by FROM knowledge WHERE id=?", (old_id,)).fetchone()
    assert (status, by) == ("superseded", result["id"])
    assert srv.store.db.execute("SELECT COUNT(*) FROM embeddings WHERE knowledge_id=?", (old_id,)).fetchone()[0] == 0


def test_flag_leaves_other_subjects_and_projects(srv):
    save(srv, NOT_UPDATES[0][1])
    save(srv, VALUE_UPDATES[0][1], project="other")
    assert "superseded" not in save(srv, NOT_UPDATES[0][0], supersede=True)
    assert "superseded" not in save(srv, VALUE_UPDATES[0][0], supersede=True)
    assert len(active(srv)) == 4


def test_fast_path_supports_the_flag(srv):
    old, new = VALUE_UPDATES[1][1], VALUE_UPDATES[1][0]
    save(srv, old)
    out = json.loads(asyncio.run(srv._do("memory_save_fast", {"content": new, "type": "fact", "project": "p",
                                                               "supersede": True})))
    assert len(out["superseded"]) == 1
    assert active(srv) == [new]
