"""file_watcher: redaction, private stripping, symlink/size guards, stability and retry."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ingestion import file_watcher as fw  # noqa: E402

SECRET_PASSWORD = "hunter2-very-secret"
PRIVATE_TEXT = "my bank pin 123456"
SMALL_LIMIT = 1024


@pytest.fixture
def env(tmp_path, monkeypatch):
    mem = tmp_path / "mem"
    inbox = tmp_path / "inbox"
    mem.mkdir()
    inbox.mkdir()
    monkeypatch.setattr(fw, "MEMORY_DIR", mem)
    db_path = tmp_path / "memory.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE knowledge(id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, type TEXT, content TEXT,
            context TEXT, project TEXT DEFAULT 'general', tags TEXT, status TEXT DEFAULT 'active',
            confidence REAL DEFAULT 1.0, source TEXT, created_at TEXT);
        CREATE TABLE embedding_pending(knowledge_id INTEGER PRIMARY KEY, attempts INTEGER NOT NULL DEFAULT 0,
            last_error TEXT, next_attempt_at TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'pending',
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
        """
    )
    conn.commit()
    conn.close()
    return fw.FileProcessor(str(db_path), inbox), inbox, db_path


def _rows(db_path):
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute("SELECT content, context, tags FROM knowledge").fetchall()
    finally:
        conn.close()


def test_secrets_are_redacted_before_storage(env):
    processor, inbox, db_path = env
    path = inbox / "config.yaml"
    path.write_text(f"db:\n  password: {SECRET_PASSWORD}\n  host: db.internal\n", encoding="utf-8")
    assert processor.process_file(path) is True
    stored = " ".join(str(c) for row in _rows(db_path) for c in row)
    assert SECRET_PASSWORD not in stored
    assert "[REDACTED]" in stored


def test_private_blocks_are_stripped(env):
    processor, inbox, db_path = env
    path = inbox / "note.txt"
    path.write_text(f"public line <private>{PRIVATE_TEXT}</private> end", encoding="utf-8")
    processor.process_file(path)
    stored = _rows(db_path)[0][0]
    assert PRIVATE_TEXT not in stored and "public line" in stored


def test_row_is_queued_for_embedding(env):
    processor, inbox, db_path = env
    (inbox / "a.txt").write_text("a reasonably long note about the deployment pipeline", encoding="utf-8")
    processor.process_file(inbox / "a.txt")
    conn = sqlite3.connect(db_path)
    assert conn.execute("SELECT COUNT(*) FROM embedding_pending").fetchone()[0] == 1


def test_duplicate_drop_is_stored_once(env):
    processor, inbox, db_path = env
    for name in ("a.txt", "b.txt"):
        (inbox / name).write_text("identical note body", encoding="utf-8")
        processor.process_file(inbox / name)
    assert len(_rows(db_path)) == 1


def test_symlinks_are_refused(env, tmp_path, make_symlink):
    processor, inbox, db_path = env
    target = tmp_path / "outside.txt"
    target.write_text("outside secret content", encoding="utf-8")
    link = inbox / "link.txt"
    make_symlink(link, target)
    assert fw.is_plain_file(link) is False
    assert processor.process_file(link) is False
    assert _rows(db_path) == []


def test_oversized_file_is_skipped_without_reading(env, monkeypatch):
    processor, inbox, db_path = env
    monkeypatch.setattr(fw, "MAX_FILE_BYTES", SMALL_LIMIT)
    big = inbox / "big.txt"
    big.write_text("x" * (SMALL_LIMIT * 2), encoding="utf-8")
    assert processor.process_file(big) is False
    assert _rows(db_path) == []


def test_read_is_capped(tmp_path):
    big = tmp_path / "big.txt"
    big.write_text("y" * (fw.MAX_CONTENT_CHARS * 10), encoding="utf-8")
    assert len(fw.read_text_capped(big)) == fw.MAX_CONTENT_CHARS


def test_wait_until_stable_detects_growing_file(tmp_path):
    path = tmp_path / "growing.txt"
    path.write_text("a", encoding="utf-8")
    ticks = {"n": 0}

    def grow(_seconds):
        ticks["n"] += 1
        with path.open("a", encoding="utf-8") as handle:
            handle.write("more")

    assert fw.wait_until_stable(path, check_sec=0, max_wait_sec=0.05, sleep=grow) is False
    path2 = tmp_path / "done.txt"
    path2.write_text("done", encoding="utf-8")
    assert fw.wait_until_stable(path2, check_sec=0, max_wait_sec=1.0, sleep=lambda s: None) is True


def test_failed_file_is_retried_after_backoff():
    clock = {"t": 0.0}
    tracker = fw.RetryTracker(clock=lambda: clock["t"])
    assert tracker.record_failure("f") is True
    assert tracker.is_due("f") is False
    clock["t"] += 10_000
    assert tracker.is_due("f") is True
    outcomes = [tracker.record_failure("f") for _ in range(fw.MAX_PROCESS_ATTEMPTS - 1)]
    assert outcomes[-1] is False


def test_blob_is_removed_when_insert_fails(env, monkeypatch):
    processor, inbox, db_path = env
    image = inbox / "pic.png"
    image.write_bytes(b"\x89PNG fake")

    def boom(*args, **kwargs):
        raise sqlite3.OperationalError("disk full")

    monkeypatch.setattr(fw, "save_background_knowledge", boom)
    assert processor.process_file(image) is False
    assert list((fw.MEMORY_DIR / "blobs").glob("*")) == []
