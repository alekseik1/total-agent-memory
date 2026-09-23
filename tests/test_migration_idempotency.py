"""Migration runner must survive columns that already exist.

SQLite has no ``ALTER TABLE ... ADD COLUMN IF NOT EXISTS``. When a database
already carries a column a migration adds — a restored backup, a DB whose
`migrations` tracker was reset, a column added out-of-band — `executescript`
aborts on the first statement. Because it is all-or-nothing, the CREATE INDEX
statements after it never run and, since the migration is never recorded, the
same failure repeats on every single startup.

The runner therefore replays such a script statement by statement, skipping
only the redundant ALTERs, and then records the migration as applied.
"""

from __future__ import annotations

import inspect
import re
import sqlite3
import sys
import threading
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import server  # noqa: E402


@pytest.fixture
def runner(tmp_path, monkeypatch):
    """A real Store on a throwaway memory dir.

    The base tables are created by `Store._create_tables`, not by a migration,
    so the migrator can only be exercised on top of a fully constructed Store.
    """
    monkeypatch.setattr(server, "MEMORY_DIR", tmp_path)
    store = server.Store()
    try:
        yield store
    finally:
        try:
            store.db.close()
        except Exception:
            pass


def _applied(runner) -> set[str]:
    return {
        r[0] for r in runner.db.execute("SELECT version FROM migrations").fetchall()
    }


def test_all_migrations_apply_on_a_fresh_database(runner):
    runner._apply_sql_migrations()
    versions = _applied(runner)
    on_disk = {p.stem.split("_", 1)[0] for p in (ROOT / "migrations").glob("*.sql")}
    assert versions == on_disk, f"unapplied: {sorted(on_disk - versions)}"


def test_rerunning_is_a_no_op(runner):
    runner._apply_sql_migrations()
    first = _applied(runner)
    runner._apply_sql_migrations()
    assert _applied(runner) == first


def test_rerunning_leaves_the_tracker_rows_identical(runner):
    """A second run must not just leave the same version SET applied - it
    must not touch any row's own content either. The renumber repair block
    deletes rows by (version, description) pair; if a future generation ever
    shared a key with an earlier one, a second run could silently delete and
    re-date a row that belongs to the CURRENT generation. Comparing full
    (version, description) pairs catches that even though the version set
    alone would look unchanged.
    """
    runner._apply_sql_migrations()
    first = sorted(
        (v, d) for v, d in runner.db.execute("SELECT version, description FROM migrations").fetchall()
    )
    runner._apply_sql_migrations()
    second = sorted(
        (v, d) for v, d in runner.db.execute("SELECT version, description FROM migrations").fetchall()
    )
    assert second == first


def test_migration_versions_are_unique():
    """Two v14 merge cycles each claimed a version key this project already
    used (029-033, then 035). Both collisions passed
    `test_all_migrations_apply_on_a_fresh_database` because it compares SETS
    of prefixes, which collapse a duplicate to one element on both sides.
    """
    versions = [p.stem.split("_", 1)[0] for p in (ROOT / "migrations").glob("*.sql")]
    assert len(versions) == len(set(versions)), sorted(
        v for v in set(versions) if versions.count(v) > 1
    )


def test_rerunning_leaves_reflection_reports_schema_unchanged(runner):
    """900/904 rename and drop reflection_reports columns.

    `_apply_sql_migrations` skips a version already recorded in `migrations`,
    so without forcing it, a second call never actually replays 900/904 -
    only `apply_reflection_report_column_migrations` (called unconditionally
    at the end of every `_apply_sql_migrations` run) would exercise anything.
    Deleting their tracker rows first forces MigrationRunner to genuinely
    replay both files, which must raise nothing (MigrationRunner has no
    duplicate-column/no-such-column tolerance for migrations >=
    TRANSACTIONAL_SCHEMA_VERSION) and leave the resulting column set
    identical.
    """
    runner._apply_sql_migrations()
    cols_first = {r[1] for r in runner.db.execute("PRAGMA table_info(reflection_reports)").fetchall()}

    runner.db.execute("DELETE FROM migrations WHERE version IN ('900', '904')")
    runner.db.commit()
    runner._apply_sql_migrations()
    cols_second = {r[1] for r in runner.db.execute("PRAGMA table_info(reflection_reports)").fetchall()}

    assert {"900", "904"} <= _applied(runner), "900/904 were not actually replayed"

    assert cols_second == cols_first
    assert {"edges_strengthened", "clusters_found", "skills_proposed", "phase_errors"} <= cols_second
    assert not ({"new_nodes", "patterns_found", "skills_refined", "rules_proposed"} & cols_second)


def test_preexisting_column_does_not_wedge_the_migration(runner):
    """The 028 lineage case: agent_id already there, tracker empty."""
    runner._apply_sql_migrations()
    runner.db.execute("DELETE FROM migrations WHERE version = '028'")
    runner.db.commit()
    assert "028" not in _applied(runner)

    runner._apply_sql_migrations()

    assert "028" in _applied(runner), "migration stayed unrecorded — will retry forever"
    cols = {r[1] for r in runner.db.execute("PRAGMA table_info(knowledge)").fetchall()}
    assert {"agent_id", "parent_agent_id"} <= cols
    indexes = {
        r[0]
        for r in runner.db.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        ).fetchall()
    }
    # The statements that follow the failing ALTER must still have run.
    assert {"idx_k_agent_id", "idx_k_parent_agent_id"} <= indexes


def test_a_genuinely_broken_migration_is_not_recorded(runner, tmp_path):
    """Only duplicate-column errors are tolerated; real errors still retry."""
    ok = runner._replay_migration_skipping_existing(
        "ALTER TABLE knowledge ADD COLUMN zzz TEXT;"
        "SELECT * FROM a_table_that_does_not_exist;",
        "999",
    )
    assert ok is False


def test_no_migration_fails_on_a_fresh_database(tmp_path, monkeypatch, capsys):
    """A clean install must apply every migration without a single failure.

    Regression for the ordering defect @juicetin reported in #12: `_migrate()`
    added the subagent-lineage columns before `_apply_sql_migrations()` ran
    `028_agent_lineage.sql`, so 028 hit "duplicate column name: agent_id" on
    every fresh database, aborted before its CREATE INDEX statements, and was
    never recorded — meaning it retried forever.
    """
    monkeypatch.setattr(server, "MEMORY_DIR", tmp_path)
    store = server.Store()
    try:
        logged = capsys.readouterr().err
        assert "failed" not in logged.lower(), f"a migration failed on a fresh DB:\n{logged}"

        applied = {
            r[0] for r in store.db.execute("SELECT version FROM migrations").fetchall()
        }
        on_disk = {p.stem.split("_", 1)[0] for p in (ROOT / "migrations").glob("*.sql")}
        assert applied == on_disk
    finally:
        store.db.close()


def test_lineage_columns_have_exactly_one_owner(tmp_path, monkeypatch):
    """028 owns agent_id/parent_agent_id; `_migrate()` must not also add them."""
    migrate_src = inspect.getsource(server.Store._migrate)
    assert "ADD COLUMN agent_id" not in migrate_src, (
        "_migrate() is adding a column that migration 028 also adds"
    )
    assert "ADD COLUMN parent_agent_id" not in migrate_src

    monkeypatch.setattr(server, "MEMORY_DIR", tmp_path)
    store = server.Store()
    try:
        cols = {r[1] for r in store.db.execute("PRAGMA table_info(knowledge)").fetchall()}
        assert {"agent_id", "parent_agent_id"} <= cols, "028 did not create them"
        indexes = {
            r[0]
            for r in store.db.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            ).fetchall()
        }
        assert {"idx_k_agent_id", "idx_k_parent_agent_id"} <= indexes
    finally:
        store.db.close()


def test_repair_lets_upstream_035_and_the_900s_reapply(runner):
    """Reproduces a live database's actual state: migrations 001-034
    applied for real, these five migrations still recorded at 035-039 under
    their v14.2.0 descriptions, the real 035 (fts_project_token) never run.
    The repair block must delete the stale 035-039 rows so the real 035
    re-applies for real (not a copy of its DDL) and these five migrations
    re-apply under 900-904.
    """
    runner._apply_sql_migrations()  # clean baseline: everything applied once

    # Undo migration 035's schema effect so a genuine re-apply is possible.
    # Triggers reference `new.fts_project` - drop them before the column.
    runner.db.execute("DROP TRIGGER IF EXISTS k_fts_i")
    runner.db.execute("DROP TRIGGER IF EXISTS k_fts_u")
    runner.db.execute("DROP TRIGGER IF EXISTS k_fts_d")
    runner.db.execute("DROP TABLE IF EXISTS knowledge_fts")
    runner.db.execute("ALTER TABLE knowledge DROP COLUMN fts_project")

    runner.db.execute(
        "DELETE FROM migrations WHERE version IN ('035','900','901','902','903','904')"
    )
    for version, description in [
        ("035", "reflection phase errors"),
        ("036", "backfill session rows"),
        ("037", "backfill orphan session rows"),
        ("038", "resolve learned errors"),
        ("039", "reflection report stat names"),
    ]:
        runner.db.execute(
            "INSERT INTO migrations (version, description, applied_at) VALUES (?, ?, ?)",
            (version, description, "2026-01-01T00:00:00Z"),
        )
    runner.db.commit()

    runner._apply_sql_migrations()

    row = runner.db.execute("SELECT description FROM migrations WHERE version='035'").fetchone()
    assert row["description"] == "fts project token"

    cols = {r[1] for r in runner.db.execute("PRAGMA table_xinfo(knowledge)").fetchall()}
    assert "fts_project" in cols, "PRAGMA table_info hides GENERATED columns - use table_xinfo"

    fts_cols = runner.db.execute("PRAGMA table_info(knowledge_fts)").fetchall()
    assert len(fts_cols) == 4

    triggers = {
        r[0] for r in runner.db.execute("SELECT name FROM sqlite_master WHERE type='trigger'").fetchall()
    }
    assert "k_fts_d" in triggers

    stale = runner.db.execute(
        "SELECT version FROM migrations WHERE version IN ('036','037','038','039')"
    ).fetchall()
    assert stale == []

    assert {"900", "901", "902", "903", "904"} <= _applied(runner)


def test_repair_does_not_delete_upstreams_own_035_row(runner):
    """The repair matches (version, description) pairs, not version alone -
    the real '035' row (description 'fts project token') must survive even
    though '035' is also a key the repair deletes under its old
    description.
    """
    runner._apply_sql_migrations()
    before = runner.db.execute(
        "SELECT version, description, applied_at FROM migrations WHERE version='035'"
    ).fetchone()
    assert before["description"] == "fts project token"

    runner._apply_sql_migrations()  # the repair runs again on every startup

    after = runner.db.execute(
        "SELECT version, description, applied_at FROM migrations WHERE version='035'"
    ).fetchone()
    assert tuple(after) == tuple(before)


def test_a_lost_migration_race_is_swallowed_not_raised(tmp_path, monkeypatch):
    """Two real `Store.__init__` calls (a concurrent MCP session and the
    launchd reflection runner) can both decide the same version needs
    applying before either commits its `INSERT INTO migrations` row.
    `MigrationRunner.apply`'s bare INSERT raises `MigrationFailed` for the
    loser on the duplicate primary key; `_apply_sql_migrations` must
    recognize that as a lost race against an already-recorded version, not a
    real failure, and must not crash `Store.__init__`.
    """
    monkeypatch.setattr(server, "MEMORY_DIR", tmp_path)
    seed = server.Store()
    seed.db.execute("DELETE FROM migrations WHERE version = '900'")
    seed.db.commit()
    seed.db.close()

    db_path = tmp_path / "memory.db"
    conn_a = sqlite3.connect(str(db_path), check_same_thread=False)
    conn_b = sqlite3.connect(str(db_path), check_same_thread=False)
    for conn in (conn_a, conn_b):
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=15000")

    started = threading.Barrier(2, timeout=5)
    errors: list[str] = []

    def run(conn):
        started.wait()
        stub = types.SimpleNamespace(db=conn)
        try:
            server.Store._apply_sql_migrations(stub)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{type(e).__name__}: {e}")

    t_a = threading.Thread(target=run, args=(conn_a,))
    t_b = threading.Thread(target=run, args=(conn_b,))
    t_a.start()
    t_b.start()
    t_a.join(timeout=20)
    t_b.join(timeout=20)

    assert errors == [], errors
    assert {
        r[0] for r in conn_a.execute("SELECT version FROM migrations WHERE version='900'")
    } == {"900"}

    conn_a.close()
    conn_b.close()


def test_no_column_is_added_by_both_python_and_sql_migrations():
    """Catch the next instance of the same class of bug."""
    migrate_src = inspect.getsource(server.Store._migrate)
    python_cols = set(re.findall(r"ADD COLUMN (\w+)", migrate_src))
    sql_cols: set[str] = set()
    for path in (ROOT / "migrations").glob("*.sql"):
        sql_cols |= set(re.findall(r"ADD COLUMN (\w+)", path.read_text()))

    overlap = python_cols & sql_cols
    assert not overlap, (
        f"{sorted(overlap)} added by both Store._migrate() and a SQL migration — "
        "whichever runs second will fail on a fresh database"
    )
