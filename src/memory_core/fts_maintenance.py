from __future__ import annotations

import sqlite3

from memory_core.telemetry import counters, op_timer

FTS_TABLES = ("knowledge_fts", "atomic_facts_fts")


def maintain_fts(
    db: sqlite3.Connection, pages: int = 64, *, optimize: bool = False
) -> int:
    if type(pages) is not int or not 1 <= pages <= 1024:
        raise ValueError("FTS merge pages must be between 1 and 1024")
    processed = 0
    with op_timer("fts_maintenance_ms"):
        for table in FTS_TABLES:
            if (
                db.execute(
                    "SELECT 1 FROM sqlite_master WHERE name=? AND sql LIKE '%USING fts5%'",
                    (table,),
                ).fetchone()
                is None
            ):
                continue
            if optimize:
                db.execute(f"INSERT INTO {table}({table}) VALUES ('optimize')")
            else:
                db.execute(
                    f"INSERT INTO {table}({table},rank) VALUES ('merge',?)", (pages,)
                )
            processed += 1
    counters.bump("fts_maintenance_tables", processed)
    return processed
