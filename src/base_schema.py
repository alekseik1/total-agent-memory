"""Single source of truth for the core SQLite schema.

`src/sql/base_schema.sql` holds the DDL; this module just locates and reads
it. Lives under `src/` (not `migrations/`) so it ships inside the `src`
package-data glob (`**/*.sql`) in a wheel install - `migrations/` is a
top-level non-package directory and is absent from a non-editable install.
Kept separate from `server.py` so tests can load the real schema without
paying for the server's heavy imports (chromadb, sentence-transformers).

Production reads it via `Store._schema()`; the pytest `db` fixture reads the
same text. That is the point - a fixture that invents its own tables lets code
ship against a schema production does not have.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

BASE_SCHEMA_PATH = Path(__file__).resolve().parent / "sql" / "base_schema.sql"
MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations"


def base_schema_sql() -> str:
    """Return the base schema DDL as an executescript-ready string."""
    return BASE_SCHEMA_PATH.read_text()


def apply_full_schema(db: sqlite3.Connection) -> None:
    """Build the whole schema production runs, in production's order.

    Mirrors ``Store.__init__``: base DDL, then the shared column migrations,
    then the self-improvement tables, then every ``migrations/*.sql`` in
    sorted order, then the reflection-report column migrations.

    Exists for tests. A fixture that hand-rolls its own ``knowledge`` table is
    how `fact_merger` shipped writing to a column production does not have with
    a fully green suite - so fixtures call this instead of inventing DDL.
    """
    db.executescript(base_schema_sql())
    apply_core_column_migrations(db)
    apply_self_improvement_tables(db)
    for sql_path in sorted(MIGRATIONS_DIR.glob("*.sql")):
        db.executescript(sql_path.read_text())
    apply_reflection_report_column_migrations(db)


def apply_core_column_migrations(db: sqlite3.Connection, log=lambda _msg: None) -> None:
    """Add the knowledge/sessions columns that older databases predate.

    Guarded by PRAGMA so it is a no-op on an up-to-date database, which is why
    it can run unconditionally at every startup. This is the ONLY list of these
    ALTERs: `Store._migrate()` calls it in production and the pytest fixture
    calls it too, so a column can never exist in tests but not in production.

    `recall_count`, `last_recalled` and `updated_at` ARE already declared in
    the base DDL (`src/sql/base_schema.sql`) - the base DDL covers fresh
    databases, this function covers pre-existing ones, and each ALTER here is
    PRAGMA-guarded so running both against the same column is safe. The rule
    this protects: no column may be added by a bare `ALTER TABLE` in a
    numbered `migrations/*.sql` file if it also lives in the base DDL, because
    a column present in both makes that migration's `ALTER TABLE` raise
    `duplicate column name` - for a migration numbered below
    `TRANSACTIONAL_SCHEMA_VERSION` that fails and retries on every single
    startup forever; at or above it, `MigrationRunner` has no such tolerance
    and raises `MigrationFailed` uncaught, crashing `Store.__init__` instead.
    """
    cols = {r[1] for r in db.execute("PRAGMA table_info(knowledge)").fetchall()}
    if "recall_count" not in cols:
        db.execute("ALTER TABLE knowledge ADD COLUMN recall_count INTEGER DEFAULT 0")
    if "last_recalled" not in cols:
        db.execute("ALTER TABLE knowledge ADD COLUMN last_recalled TEXT")
    # v4.0: branch-aware context
    if "branch" not in cols:
        db.execute("ALTER TABLE knowledge ADD COLUMN branch TEXT DEFAULT ''")
        log("Migration: added branch to knowledge table")
    # Claude Code v2.1.139+ subagent lineage (OTEL agent_id / parent_agent_id)
    if "agent_id" not in cols:
        db.execute("ALTER TABLE knowledge ADD COLUMN agent_id TEXT DEFAULT NULL")
        log("Migration: added agent_id to knowledge table")
    if "parent_agent_id" not in cols:
        db.execute("ALTER TABLE knowledge ADD COLUMN parent_agent_id TEXT DEFAULT NULL")
        log("Migration: added parent_agent_id to knowledge table")
    # Written by fact_merger (merge inserts + source archival) and by any future
    # in-place edit needing a last-touched timestamp. Databases created before
    # this column existed backfill from created_at so ordering by updated_at
    # never sees a NULL island.
    if "updated_at" not in cols:
        db.execute("ALTER TABLE knowledge ADD COLUMN updated_at TEXT")
        db.execute("UPDATE knowledge SET updated_at = created_at WHERE updated_at IS NULL")
        log("Migration: added updated_at to knowledge table")

    db.execute(
        "CREATE INDEX IF NOT EXISTS idx_k_agent_id "
        "ON knowledge(agent_id) WHERE agent_id IS NOT NULL"
    )
    db.execute(
        "CREATE INDEX IF NOT EXISTS idx_k_parent_agent_id "
        "ON knowledge(parent_agent_id) WHERE parent_agent_id IS NOT NULL"
    )
    db.execute("CREATE INDEX IF NOT EXISTS idx_k_updated_at ON knowledge(updated_at)")

    sess_cols = {r[1] for r in db.execute("PRAGMA table_info(sessions)").fetchall()}
    if "branch" not in sess_cols:
        db.execute("ALTER TABLE sessions ADD COLUMN branch TEXT DEFAULT ''")
        log("Migration: added branch to sessions table")


def apply_reflection_report_column_migrations(db: sqlite3.Connection, log=lambda _msg: None) -> None:
    """PRAGMA-guarded `reflection_reports` column work, idempotent like
    ``apply_core_column_migrations`` above.

    `reflection_reports` is created by `migrations/001_v5_schema.sql`, not the
    base DDL, so this cannot run from `apply_core_column_migrations` - that
    function runs from `Store._migrate()`, which executes before
    `Store._apply_sql_migrations()` has had a chance to run migration 001. It
    must instead run after `_apply_sql_migrations`'s loop, once the table is
    guaranteed to exist (the guard below also makes it a no-op against a
    database where `migrations/` is absent, e.g. a non-editable/wheel
    install that never bundled migration 001, or where migration 001 itself
    failed and was left unrecorded - both callers invoke this function after
    their loop, never mid-loop).

    This logic used to be bare `ALTER TABLE` statements in
    `migrations/035_reflection_phase_errors.sql` and
    `migrations/039_reflection_report_stat_names.sql` (numbered 029 and 033
    before the v14.2.0 merge renumbered them). A database that already
    applied them under the old numbers runs them again under the new ones,
    and `ADD COLUMN` / `RENAME COLUMN` / `DROP COLUMN` are not safe to
    repeat in SQLite - see the comment atop each of those files.

    Runs inside `BEGIN IMMEDIATE` so the write lock is taken before the
    `PRAGMA table_info` read below: two `Store.__init__` calls against the
    same file (a concurrent MCP session plus the launchd reflection runner,
    see the migration rule in project memory) can otherwise both read the
    pre-migration column set, and the second one's `ALTER TABLE` then raises
    `duplicate column name` / `no such column` - the PRAGMA guard alone only
    protects sequential replay, not a concurrent first run. With the lock
    taken first, the second caller blocks until the first commits, then
    re-reads the now-migrated schema and is a no-op.
    """
    tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    if "reflection_reports" not in tables:
        return

    db.execute("BEGIN IMMEDIATE")
    try:
        cols = {r[1] for r in db.execute("PRAGMA table_info(reflection_reports)").fetchall()}
        if "phase_errors" not in cols:
            db.execute("ALTER TABLE reflection_reports ADD COLUMN phase_errors JSON")
            log("Migration: added phase_errors to reflection_reports table")

        if "new_nodes" in cols:
            db.execute("ALTER TABLE reflection_reports RENAME COLUMN new_nodes TO edges_strengthened")
            log("Migration: renamed reflection_reports.new_nodes to edges_strengthened")
        if "patterns_found" in cols:
            db.execute("ALTER TABLE reflection_reports RENAME COLUMN patterns_found TO clusters_found")
            log("Migration: renamed reflection_reports.patterns_found to clusters_found")
        if "skills_refined" in cols:
            db.execute("ALTER TABLE reflection_reports RENAME COLUMN skills_refined TO skills_proposed")
            log("Migration: renamed reflection_reports.skills_refined to skills_proposed")
        if "rules_proposed" in cols:
            db.execute("ALTER TABLE reflection_reports DROP COLUMN rules_proposed")
            log("Migration: dropped reflection_reports.rules_proposed")
        db.commit()
    except Exception:
        db.rollback()
        raise


def apply_self_improvement_tables(db: sqlite3.Connection) -> None:
    """Create errors/insights/rules - the Self-Improving Agent tables.

    Shared by ``Store._create_self_improvement_tables`` and the
    ``apply_full_schema`` test fixture builder, for the same reason
    ``apply_core_column_migrations`` is shared: these tables live outside
    `src/sql/base_schema.sql` (they predate it and are created lazily), so a
    fixture that skips this step doesn't have the `errors` table at all - and
    any `migrations/*.sql` file that touches `errors` (e.g. 038) breaks every
    test built on `apply_full_schema` the moment it does. All statements are
    `IF NOT EXISTS`, so calling this unconditionally is safe.
    """
    db.executescript("""
        CREATE TABLE IF NOT EXISTS errors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            category TEXT NOT NULL,
            severity TEXT NOT NULL DEFAULT 'medium',
            description TEXT NOT NULL,
            context TEXT DEFAULT '',
            fix TEXT DEFAULT '',
            project TEXT DEFAULT 'general',
            tags TEXT DEFAULT '[]',
            status TEXT DEFAULT 'open',
            resolved_at TEXT,
            insight_id INTEGER,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_e_category ON errors(category);
        CREATE INDEX IF NOT EXISTS idx_e_project ON errors(project);
        CREATE INDEX IF NOT EXISTS idx_e_status ON errors(status);
        CREATE INDEX IF NOT EXISTS idx_e_session ON errors(session_id);
        CREATE INDEX IF NOT EXISTS idx_e_created ON errors(created_at DESC);

        CREATE VIRTUAL TABLE IF NOT EXISTS errors_fts USING fts5(
            description, context, fix, tags,
            content='errors', content_rowid='id'
        );
        CREATE TRIGGER IF NOT EXISTS e_fts_i AFTER INSERT ON errors BEGIN
            INSERT INTO errors_fts(rowid, description, context, fix, tags)
            VALUES (new.id, new.description, new.context, new.fix, new.tags);
        END;
        CREATE TRIGGER IF NOT EXISTS e_fts_u AFTER UPDATE ON errors BEGIN
            INSERT INTO errors_fts(errors_fts, rowid, description, context, fix, tags)
            VALUES ('delete', old.id, old.description, old.context, old.fix, old.tags);
            INSERT INTO errors_fts(rowid, description, context, fix, tags)
            VALUES (new.id, new.description, new.context, new.fix, new.tags);
        END;
        CREATE TRIGGER IF NOT EXISTS e_fts_d AFTER DELETE ON errors BEGIN
            INSERT INTO errors_fts(errors_fts, rowid, description, context, fix, tags)
            VALUES ('delete', old.id, old.description, old.context, old.fix, old.tags);
        END;

        CREATE TABLE IF NOT EXISTS insights (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            content TEXT NOT NULL,
            context TEXT DEFAULT '',
            category TEXT NOT NULL,
            importance INTEGER NOT NULL DEFAULT 2,
            confidence REAL NOT NULL DEFAULT 0.5,
            source_error_ids TEXT DEFAULT '[]',
            project TEXT DEFAULT 'general',
            tags TEXT DEFAULT '[]',
            status TEXT DEFAULT 'active',
            promoted_to_rule_id INTEGER,
            fire_count INTEGER DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_i_status ON insights(status);
        CREATE INDEX IF NOT EXISTS idx_i_category ON insights(category);
        CREATE INDEX IF NOT EXISTS idx_i_project ON insights(project);
        CREATE INDEX IF NOT EXISTS idx_i_importance ON insights(importance DESC);

        CREATE TABLE IF NOT EXISTS rules (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            content TEXT NOT NULL,
            context TEXT DEFAULT '',
            category TEXT NOT NULL,
            scope TEXT DEFAULT 'global',
            priority INTEGER NOT NULL DEFAULT 5,
            source_insight_id INTEGER,
            project TEXT DEFAULT 'general',
            tags TEXT DEFAULT '[]',
            status TEXT DEFAULT 'active',
            fire_count INTEGER DEFAULT 0,
            success_count INTEGER DEFAULT 0,
            fail_count INTEGER DEFAULT 0,
            success_rate REAL DEFAULT 0.0,
            last_fired TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_r_status ON rules(status);
        CREATE INDEX IF NOT EXISTS idx_r_scope ON rules(scope);
        CREATE INDEX IF NOT EXISTS idx_r_priority ON rules(priority DESC);
        CREATE INDEX IF NOT EXISTS idx_r_project ON rules(project);
    """)
