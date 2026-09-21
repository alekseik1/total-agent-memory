"""knowledge_fts after migration 035: project token column and narrow triggers."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from memory_core.fts_schema import project_token, recreate_knowledge_fts


@pytest.fixture
def store(monkeypatch, tmp_path):
    (tmp_path / "blobs").mkdir(exist_ok=True)
    (tmp_path / "chroma").mkdir(exist_ok=True)
    import server

    monkeypatch.setattr(server, "MEMORY_DIR", tmp_path)
    s = server.Store()
    yield s
    s.db.close()


def fts_objects(db):
    return sorted(tuple(row) for row in db.execute(
        "SELECT name, sql FROM sqlite_master WHERE name IN ('knowledge_fts', 'k_fts_i', 'k_fts_u', 'k_fts_d')"
    ))


def test_migration_and_module_ddl_match(store):
    migrated = fts_objects(store.db)
    assert [name for name, _ in migrated] == ["k_fts_d", "k_fts_i", "k_fts_u", "knowledge_fts"]
    recreate_knowledge_fts(store.db)
    assert fts_objects(store.db) == migrated


@pytest.mark.parametrize("project", ["general", "tenant-042", "проект", "a b"])
def test_project_token_matches_generated_column(store, project):
    store.save_knowledge("s1", "Deploy runs nightly", "fact", project=project)
    stored = store.db.execute("SELECT fts_project FROM knowledge WHERE project=?", (project,)).fetchone()[0]
    assert stored == project_token(project)
    hits = store.db.execute(
        "SELECT count(*) FROM knowledge_fts WHERE knowledge_fts MATCH ?",
        (f"fts_project : {project_token(project)}",),
    ).fetchone()[0]
    assert hits == 1


def test_recall_counter_does_not_reindex(store):
    rid, *_ = store.save_knowledge("s1", "Deploy runs nightly", "fact", project="p")
    statements = []
    store.db.set_trace_callback(statements.append)
    store.bump_recall([rid])
    store.db.set_trace_callback(None)
    assert not any("k_fts_u" in s for s in statements)


def test_content_update_and_delete_keep_index_in_sync(store):
    rid, *_ = store.save_knowledge("s1", "Deploy runs nightly", "fact", project="p")
    store.db.execute("UPDATE knowledge SET content='Deploy runs hourly' WHERE id=?", (rid,))
    def matching(word):
        return [row[0] for row in store.db.execute("SELECT rowid FROM knowledge_fts WHERE knowledge_fts MATCH ?", (word,))]

    assert matching("hourly") == [rid]
    assert matching("nightly") == []
    store.db.execute("DELETE FROM knowledge WHERE id=?", (rid,))
    assert matching("hourly") == []


def test_scoped_recall_uses_project_token(store):
    import server

    for project in ("alpha", "beta"):
        for n in range(3):
            store.save_knowledge("s1", f"Backup of {project} service {n} runs at night", "fact", project=project)
    statements = []
    store.db.set_trace_callback(statements.append)
    found = server.Recall(store).search("backup night", project="alpha", limit=10, record_usage=False)
    store.db.set_trace_callback(None)

    hits = [hit for group in found["results"].values() for hit in group]
    assert hits and {hit["project"] for hit in hits} == {"alpha"}
    assert any(f"fts_project : {project_token('alpha')}" in s for s in statements)


def test_word_equal_to_project_name_does_not_match_token(store):
    store.save_knowledge("s1", "Unrelated note", "fact", project="alpha")
    hits = store.db.execute("SELECT count(*) FROM knowledge_fts WHERE knowledge_fts MATCH 'alpha'").fetchone()[0]
    assert hits == 0
