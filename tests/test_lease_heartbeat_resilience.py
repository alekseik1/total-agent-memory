"""The lease heartbeat survives a failed renewal instead of dying silently."""

from __future__ import annotations

import logging
import sqlite3
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from memory_core.leases import LeaseHeartbeat  # noqa: E402

HEARTBEAT_INTERVAL_SEC = 0.01
RENEWALS_WANTED = 3
WAIT_SEC = 5.0


def test_heartbeat_keeps_renewing_after_a_sqlite_error(tmp_path, monkeypatch, caplog):
    path = tmp_path / "q.db"
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE enrichment_queue (id INTEGER, lease_token TEXT, status TEXT, heartbeat_at TEXT)")
    db.commit()
    beat = LeaseHeartbeat(db, [(1, "t")], HEARTBEAT_INTERVAL_SEC, logging.getLogger("test"))
    calls = {"n": 0}
    enough = threading.Event()

    def flaky_renew(_conn):
        calls["n"] += 1
        if calls["n"] == 1:
            raise sqlite3.OperationalError("database is locked")
        if calls["n"] >= RENEWALS_WANTED:
            enough.set()

    monkeypatch.setattr(beat, "renew", flaky_renew)
    with caplog.at_level(logging.ERROR), beat:
        assert enough.wait(WAIT_SEC), "heartbeat stopped after the first sqlite error"
    assert calls["n"] >= RENEWALS_WANTED
