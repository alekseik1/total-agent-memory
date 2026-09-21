from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone

from memory_core.telemetry import counters, op_timer

TRANSACTIONAL_SCHEMA_VERSION = 30


class MigrationFailed(RuntimeError):
    pass


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
                self.db.executescript('BEGIN IMMEDIATE;\n' + migration.script)
                self.db.execute(
                    'INSERT INTO migrations(version,description,applied_at) VALUES (?,?,?)',
                    (migration.version, migration.description, datetime.now(timezone.utc).isoformat()),
                )
                self.db.commit()
            except sqlite3.Error as error:
                self.db.rollback()
                counters.bump('schema_migration_errors')
                raise MigrationFailed(f'Migration {migration.version} failed: {error}') from error
