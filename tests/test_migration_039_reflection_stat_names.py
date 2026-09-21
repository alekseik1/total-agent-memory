"""Regression test for migrations/039_reflection_report_stat_names.sql.

`_save_report` (src/reflection/agent.py) wrote synthesis['edges_strengthened']
into column `new_nodes`, synthesis['clusters_found'] into `patterns_found`,
and synthesis['skills_proposed'] into `skills_refined` — a live report read
`new_nodes=116176`, which is strengthened graph edges, not new nodes. 039
renames the three columns to match what they actually hold, and drops
`rules_proposed`, which every INSERT hardcoded to 0.

039 (numbered 033 before the v14.2.0 merge renumbered it) is now a
comment-only historical marker; the actual work lives in
`base_schema.apply_reflection_report_column_migrations`, exercised directly
here, because a bare `RENAME COLUMN` / `DROP COLUMN` is not safe to repeat
against a database that already applied the old-numbered migration.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from base_schema import apply_reflection_report_column_migrations  # noqa: E402


def _pre_039_db() -> sqlite3.Connection:
    """A `reflection_reports` table shaped as migrations 001-038 left it."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE reflection_reports (
            id TEXT PRIMARY KEY,
            period_start TEXT NOT NULL,
            period_end TEXT NOT NULL,
            type TEXT NOT NULL CHECK (type IN ('session', 'periodic', 'weekly', 'manual')),
            new_nodes INTEGER DEFAULT 0,
            patterns_found INTEGER DEFAULT 0,
            skills_refined INTEGER DEFAULT 0,
            rules_proposed INTEGER DEFAULT 0,
            contradictions INTEGER DEFAULT 0,
            archived INTEGER DEFAULT 0,
            focus_areas JSON,
            key_findings JSON,
            proposed_changes JSON,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
            phase_errors JSON
        );
        """
    )
    conn.execute(
        "INSERT INTO reflection_reports "
        "(id, period_start, period_end, type, new_nodes, patterns_found, "
        "skills_refined, rules_proposed, contradictions, archived, created_at) "
        "VALUES ('rep1', '2026-08-01T00:00:00Z', '2026-08-08T00:00:00Z', "
        "'weekly', 116176, 277, 6, 0, 3, 12, '2026-08-08T00:05:00Z')"
    )
    conn.commit()
    return conn


def test_migration_renames_columns_and_preserves_existing_rows():
    conn = _pre_039_db()

    apply_reflection_report_column_migrations(conn)

    cols = {r[1] for r in conn.execute("PRAGMA table_info(reflection_reports)").fetchall()}
    assert {"edges_strengthened", "clusters_found", "skills_proposed"} <= cols
    assert not ({"new_nodes", "patterns_found", "skills_refined", "rules_proposed"} & cols)

    row = conn.execute(
        "SELECT edges_strengthened, clusters_found, skills_proposed, "
        "contradictions, archived FROM reflection_reports WHERE id='rep1'"
    ).fetchone()
    assert tuple(row) == (116176, 277, 6, 3, 12)


def test_migration_is_idempotent_against_an_already_migrated_database():
    conn = _pre_039_db()

    apply_reflection_report_column_migrations(conn)
    apply_reflection_report_column_migrations(conn)

    cols = {r[1] for r in conn.execute("PRAGMA table_info(reflection_reports)").fetchall()}
    assert {"edges_strengthened", "clusters_found", "skills_proposed", "phase_errors"} <= cols
    assert not ({"new_nodes", "patterns_found", "skills_refined", "rules_proposed"} & cols)


def test_migration_is_a_no_op_before_reflection_reports_exists():
    conn = sqlite3.connect(":memory:")
    apply_reflection_report_column_migrations(conn)  # must not raise
