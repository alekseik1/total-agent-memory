"""Opportunistic WAL reclaim when the personal SQLite store opens."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path

# A TRUNCATE checkpoint succeeds only with no other reader holding a snapshot.
CHECKPOINT_STATEMENT = "PRAGMA wal_checkpoint(TRUNCATE)"


def checkpoint_on_open(
    db: sqlite3.Connection,
    db_path: Path,
    *,
    checkpoint_timeout_ms: int,
    restore_timeout_ms: int,
    log: Callable[[str], None],
) -> None:
    """Try one TRUNCATE checkpoint with a short busy_timeout, then restore the regular timeout.

    A busy result (the common case with several sessions open) is logged and startup continues.
    A successful TRUNCATE reports (0, 0, 0), so the size to log is read from the file beforehand.
    """
    wal_path = db_path.with_name(db_path.name + "-wal")
    wal_bytes = wal_path.stat().st_size if wal_path.exists() else 0
    try:
        db.execute(f"PRAGMA busy_timeout={checkpoint_timeout_ms}")
        row = db.execute(CHECKPOINT_STATEMENT).fetchone()
        if row and row[0] != 0:
            log(f"WAL checkpoint busy: {row[1]} log frames, {row[2]} checkpointed")
        elif wal_bytes:
            log(f"WAL checkpoint: truncated {wal_bytes} bytes")
    except sqlite3.OperationalError as exc:
        log(f"WAL checkpoint skipped: {exc}")
    finally:
        db.execute(f"PRAGMA busy_timeout={restore_timeout_ms}")
