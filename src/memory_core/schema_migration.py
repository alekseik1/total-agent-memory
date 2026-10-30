from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone

from memory_core.telemetry import counters, op_timer

TRANSACTIONAL_SCHEMA_VERSION = 30


_ADD_COLUMN_RE = re.compile(
    r"^\s*ALTER\s+TABLE\s+(?P<table>\w+)\s+ADD\s+COLUMN\s+(?P<column>\w+)", re.IGNORECASE
)
_COMMENT_LINE_RE = re.compile(r"^\s*--.*$", re.MULTILINE)


class MigrationFailed(RuntimeError):
    pass


def skip_existing_columns(db: sqlite3.Connection, script: str) -> str:
    """Drop `ALTER TABLE .. ADD COLUMN` statements whose column already exists.

    Runtime helpers (queue_retry.ensure_column) may add a column before its
    migration is recorded; replaying the ALTER would then fail for good.
    """
    kept: list[str] = []
    pending = ""
    for line in script.splitlines(keepends=True):
        pending += line
        if not sqlite3.complete_statement(pending):
            continue
        match = _ADD_COLUMN_RE.match(_COMMENT_LINE_RE.sub("", pending))
        if match:
            columns = {row[1].lower() for row in db.execute(f"PRAGMA table_info({match['table']})")}
            if match["column"].lower() in columns:
                pending = ""
                continue
        kept.append(pending)
        pending = ""
    kept.append(pending)
    return "".join(kept)


@dataclass(frozen=True)
class Migration:
    version: str
    description: str
    script: str


class MigrationRunner:
    def __init__(self, db: sqlite3.Connection):
        self.db = db

    def apply(self, migration: Migration) -> None:
        if self.db.in_transaction:
            raise MigrationFailed('Migration requires an idle database connection')
        with op_timer('schema_migration_ms'):
            try:
                self.db.executescript('BEGIN IMMEDIATE;\n' + skip_existing_columns(self.db, migration.script))
                self.db.execute(
                    'INSERT INTO migrations(version,description,applied_at) VALUES (?,?,?)',
                    (migration.version, migration.description, datetime.now(timezone.utc).isoformat()),
                )
                self.db.commit()
            except sqlite3.Error as error:
                self.db.rollback()
                counters.bump('schema_migration_errors')
                raise MigrationFailed(f'Migration {migration.version} failed: {error}') from error
