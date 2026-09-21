"""Tests for src/base_schema.py — the single-source-of-truth SQLite schema."""

from __future__ import annotations

import sqlite3
import threading

_KNOWN_RUNNER_ASYMMETRY = {"migrations"}


def test_apply_core_column_migrations_upgrades_legacy_database_idempotently():
    from base_schema import apply_core_column_migrations

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE knowledge (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL, type TEXT NOT NULL,
            content TEXT NOT NULL, context TEXT DEFAULT '',
            project TEXT DEFAULT 'general', tags TEXT DEFAULT '[]',
            status TEXT DEFAULT 'active', superseded_by INTEGER,
            confidence REAL DEFAULT 1.0, source TEXT DEFAULT 'explicit',
            created_at TEXT NOT NULL, last_confirmed TEXT
        );
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY, started_at TEXT NOT NULL, ended_at TEXT,
            project TEXT DEFAULT 'general', status TEXT DEFAULT 'open',
            summary TEXT, log_count INTEGER DEFAULT 0
        );
        """
    )
    conn.execute(
        "INSERT INTO knowledge (session_id, type, content, created_at) "
        "VALUES ('s1', 'fact', 'legacy row', '2026-04-14T00:00:00Z')"
    )
    conn.commit()

    apply_core_column_migrations(conn)

    knowledge_cols = {r[1] for r in conn.execute("PRAGMA table_info(knowledge)").fetchall()}
    assert {
        "recall_count", "last_recalled", "branch", "agent_id", "parent_agent_id", "updated_at",
    } <= knowledge_cols
    session_cols = {r[1] for r in conn.execute("PRAGMA table_info(sessions)").fetchall()}
    assert "branch" in session_cols

    row = conn.execute(
        "SELECT created_at, updated_at FROM knowledge WHERE session_id='s1'"
    ).fetchone()
    assert row["updated_at"] == row["created_at"]

    apply_core_column_migrations(conn)

    row_again = conn.execute(
        "SELECT created_at, updated_at FROM knowledge WHERE session_id='s1'"
    ).fetchone()
    assert row_again["updated_at"] == row["updated_at"]
    knowledge_cols_again = {r[1] for r in conn.execute("PRAGMA table_info(knowledge)").fetchall()}
    assert knowledge_cols_again == knowledge_cols


def test_apply_full_schema_matches_real_store_bootstrap(tmp_path, monkeypatch):
    from base_schema import apply_full_schema

    reference = sqlite3.connect(":memory:")
    apply_full_schema(reference)
    reference_tables = [
        r[0]
        for r in reference.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
        if r[0] not in _KNOWN_RUNNER_ASYMMETRY
    ]
    assert reference_tables

    (tmp_path / "chroma").mkdir(exist_ok=True)
    import server

    monkeypatch.setattr(server, "MEMORY_DIR", tmp_path)
    store = server.Store()
    try:
        for table in reference_tables:
            ref_cols = {
                (r[1], r[2], r[3], r[4], r[5])
                for r in reference.execute(f"PRAGMA table_info({table})").fetchall()
            }
            real_cols = {
                (r[1], r[2], r[3], r[4], r[5])
                for r in store.db.execute(f"PRAGMA table_info({table})").fetchall()
            }
            assert ref_cols == real_cols, f"schema drift in table '{table}'"

        from base_schema import MIGRATIONS_DIR

        applied_versions = {
            r[0] for r in store.db.execute("SELECT version FROM migrations").fetchall()
        }
        file_versions = {
            p.stem.split("_", 1)[0] for p in MIGRATIONS_DIR.glob("*.sql")
        }
        assert applied_versions == file_versions
    finally:
        store.db.close()


def test_apply_full_schema_seeds_fact_merge_session():
    from base_schema import apply_full_schema

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    apply_full_schema(conn)

    row = conn.execute("SELECT id FROM sessions WHERE id='fact-merge'").fetchone()
    assert row is not None


def test_the_shared_schema_creates_the_self_improvement_tables():
    """`apply_full_schema` must build the same tables production has.

    It used to omit errors/insights/rules — they were created only by
    `Store._create_self_improvement_tables`, which the test fixture never
    called. Nothing noticed until migration 038 became the first migration to
    touch `errors` and broke every fixture built this way. A schema the tests
    use that production does not have (or the reverse) makes green meaningless.
    """
    from base_schema import apply_full_schema

    db = sqlite3.connect(":memory:")
    try:
        apply_full_schema(db)
        tables = {
            r[0] for r in db.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert {"errors", "insights", "rules"} <= tables, sorted(tables)
    finally:
        db.close()


def test_production_and_the_fixture_build_the_self_improvement_tables_alike():
    """Both paths delegate to one function, so they cannot drift apart again."""
    import inspect

    import server

    src = inspect.getsource(server.Store._create_self_improvement_tables)
    assert "apply_self_improvement_tables" in src, src


def test_reflection_report_column_migrations_serializes_concurrent_first_runs(tmp_path):
    """Two `Store.__init__` calls racing to run this migration on the same
    file (a concurrent MCP session plus the launchd reflection runner) must
    not crash with `duplicate column name` / `no such column`.

    `PRAGMA table_info` then `ALTER TABLE` with no transaction lets both
    connections read the pre-migration schema before either writes, so both
    attempt the same `ADD COLUMN` / `RENAME COLUMN`. `BEGIN IMMEDIATE` takes
    the write lock before the PRAGMA read, so the second connection blocks
    until the first commits, then re-reads the already-migrated schema and
    is a no-op.
    """
    from base_schema import apply_reflection_report_column_migrations

    db_path = tmp_path / "memory.db"
    seed = sqlite3.connect(str(db_path))
    seed.executescript(
        """
        CREATE TABLE reflection_reports (
            id TEXT PRIMARY KEY,
            period_start TEXT NOT NULL,
            period_end TEXT NOT NULL,
            type TEXT NOT NULL,
            new_nodes INTEGER DEFAULT 0,
            patterns_found INTEGER DEFAULT 0,
            skills_refined INTEGER DEFAULT 0,
            rules_proposed INTEGER DEFAULT 0
        );
        """
    )
    seed.commit()
    seed.close()

    conn_a = sqlite3.connect(str(db_path), check_same_thread=False)
    conn_b = sqlite3.connect(str(db_path), check_same_thread=False)
    conn_a.execute("PRAGMA busy_timeout=15000")
    conn_b.execute("PRAGMA busy_timeout=15000")

    started = threading.Barrier(2, timeout=5)
    errors: list[str] = []

    def run(conn):
        started.wait()
        try:
            apply_reflection_report_column_migrations(conn)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{type(e).__name__}: {e}")

    t_a = threading.Thread(target=run, args=(conn_a,))
    t_b = threading.Thread(target=run, args=(conn_b,))
    t_a.start()
    t_b.start()
    t_a.join(timeout=10)
    t_b.join(timeout=10)

    assert errors == [], errors

    cols = {r[1] for r in conn_a.execute("PRAGMA table_info(reflection_reports)").fetchall()}
    assert {"edges_strengthened", "clusters_found", "skills_proposed", "phase_errors"} <= cols
    assert not ({"new_nodes", "patterns_found", "skills_refined", "rules_proposed"} & cols)

    conn_a.close()
    conn_b.close()


def test_reflection_report_column_migrations_noops_on_an_open_transaction():
    """Neither current caller (`Store._apply_sql_migrations`,
    `apply_full_schema`) should ever hand this an in-progress transaction -
    but this function only gets a `Connection`, not the code that produced
    it, so it cannot verify that. `BEGIN IMMEDIATE` raises "cannot start a
    transaction within a transaction" into a connection that already has one
    open; a prior round's fix for that made this a no-op instead of letting
    the exception kill `Store.__init__`.
    """
    from base_schema import apply_full_schema, apply_reflection_report_column_migrations

    db = sqlite3.connect(":memory:")
    apply_full_schema(db)
    cols_before = {r[1] for r in db.execute("PRAGMA table_info(reflection_reports)").fetchall()}

    db.execute("INSERT INTO sessions (id, started_at) VALUES ('leak-probe', 'x')")
    assert db.in_transaction is True

    apply_reflection_report_column_migrations(db)  # must not raise

    assert db.in_transaction is True, "a no-op must leave the caller's transaction exactly as handed"
    cols_after = {r[1] for r in db.execute("PRAGMA table_info(reflection_reports)").fetchall()}
    assert cols_after == cols_before, "a no-op must not partially migrate the table"

    db.rollback()
    db.close()
