"""Single source of truth for the core SQLite schema.

`src/sql/base_schema.sql` holds the DDL; this module just locates and reads
it. Lives under `src/` (not `migrations/`) so it ships inside the `src`
package-data glob (`**/*.sql`) in a wheel install — `migrations/` is a
top-level non-package directory and is absent from a non-editable install.
Kept separate from `server.py` so tests can load the real schema without
paying for the server's heavy imports (chromadb, sentence-transformers).

Production reads it via `Store._schema()`; the pytest `db` fixture reads the
same text. That is the point — a fixture that invents its own tables lets code
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
    then every ``migrations/*.sql`` in sorted order.

    Exists for tests. A fixture that hand-rolls its own ``knowledge`` table is
    how `fact_merger` shipped writing to a column production does not have with
    a fully green suite — so fixtures call this instead of inventing DDL.
    """
    db.executescript(base_schema_sql())
    apply_core_column_migrations(db)
    for sql_path in sorted(MIGRATIONS_DIR.glob("*.sql")):
        db.executescript(sql_path.read_text())


def apply_core_column_migrations(db: sqlite3.Connection, log=lambda _msg: None) -> None:
    """Add the knowledge/sessions columns that older databases predate.

    Guarded by PRAGMA so it is a no-op on an up-to-date database, which is why
    it can run unconditionally at every startup. This is the ONLY list of these
    ALTERs: `Store._migrate()` calls it in production and the pytest fixture
    calls it too, so a column can never exist in tests but not in production.

    `recall_count`, `last_recalled` and `updated_at` ARE already declared in
    the base DDL (`src/sql/base_schema.sql`) — the base DDL covers fresh
    databases, this function covers pre-existing ones, and each ALTER here is
    PRAGMA-guarded so running both against the same column is safe. The rule
    this protects: no column may be added by a bare `ALTER TABLE` in a
    numbered `migrations/*.sql` file if it also lives in the base DDL, because
    `_apply_sql_migrations` retries any migration that raises — so a column
    present in both the base DDL and a migration file makes that migration
    fail, and retry, on every single startup, forever.
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
