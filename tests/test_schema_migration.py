import sqlite3
from pathlib import Path

import pytest

from memory_core.schema_migration import Migration, MigrationFailed, MigrationRunner


def test_failed_migration_rolls_back_ddl_and_tracking():
    db = sqlite3.connect(":memory:")
    try:
        db.executescript(
            "CREATE TABLE migrations(version TEXT PRIMARY KEY,description TEXT,applied_at TEXT); "
            "CREATE TABLE example(id INTEGER PRIMARY KEY);"
        )
        runner = MigrationRunner(db)
        with pytest.raises(MigrationFailed, match="031"):
            runner.apply(
                Migration(
                    "031",
                    "example",
                    "ALTER TABLE example ADD COLUMN value TEXT; INSERT INTO missing VALUES (1);",
                )
            )
        assert [row[1] for row in db.execute("PRAGMA table_info(example)")] == ["id"]
        assert db.execute("SELECT count(*) FROM migrations").fetchone()[0] == 0
        assert not db.in_transaction
        runner.apply(
            Migration("031", "example", "ALTER TABLE example ADD COLUMN value TEXT;")
        )
        assert [row[1] for row in db.execute("PRAGMA table_info(example)")] == [
            "id",
            "value",
        ]
        assert db.execute("SELECT version FROM migrations").fetchone()[0] == "031"
    finally:
        db.close()


def test_upgrade_queues_legacy_extractions_without_deleting_originals():
    db = sqlite3.connect(":memory:")
    try:
        db.executescript(
            "CREATE TABLE knowledge(id INTEGER PRIMARY KEY,content TEXT,project TEXT,"
            "session_id TEXT,branch TEXT,status TEXT,type TEXT,created_at TEXT);"
            "INSERT INTO knowledge VALUES(1,'original','p','s','','active','fact','2026-01-01');"
        )
        migrations = Path(__file__).parents[1] / "migrations"
        db.executescript((migrations / "020_async_enrichment.sql").read_text())
        db.executescript((migrations / "029_atomic_facts.sql").read_text())
        db.execute("INSERT INTO atomic_fact_runs VALUES(1,'original',0,'old-model')")
        db.commit()
        db.executescript((migrations / "030_evidence_lifecycle.sql").read_text())
        assert (
            db.execute("SELECT knowledge_id FROM atomic_fact_rebuild").fetchone()[0]
            == 1
        )
        assert db.execute("SELECT count(*) FROM atomic_fact_runs").fetchone()[0] == 0
        assert db.execute("SELECT content FROM knowledge").fetchone()[0] == "original"
        assert db.execute(
            "SELECT target_id,source_id FROM atomic_fact_dependencies"
        ).fetchone() == (1, 1)
    finally:
        db.close()
