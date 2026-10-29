"""extract_transcript: <private> blocks never reach disk, files are private, ids are validated."""

from __future__ import annotations

import json
import os
import sqlite3
import stat
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import extract_transcript as et  # noqa: E402

SECRET_NOTE = "my bank pin 123456"
SESSION = "0b8f2c1e-1111-4222-8333-444455556666"


def _transcript(tmp_path, user_text, tool_content=None):
    lines = [{"timestamp": "2026-01-01T00:00:00Z", "cwd": "/work/demo",
              "message": {"role": "user", "content": user_text}}]
    blocks = [{"type": "text", "text": "ok"}]
    if tool_content is not None:
        blocks.append({"type": "tool_use", "name": "mcp__memory__memory_save",
                       "input": {"content": tool_content, "type": "fact", "project": "demo"}})
    lines.append({"timestamp": "2026-01-01T00:00:01Z", "message": {"role": "assistant", "content": blocks}})
    path = tmp_path / "t.jsonl"
    path.write_text("\n".join(json.dumps(line) for line in lines), encoding="utf-8")
    return str(path)


def test_sanitize_strips_private_blocks():
    out = et.sanitize(f"note <private>{SECRET_NOTE}</private> after")
    assert SECRET_NOTE not in out and "<private>" not in out and "after" in out


def test_sanitize_strips_unterminated_private_block():
    assert SECRET_NOTE not in et.sanitize(f"note <private>{SECRET_NOTE}")


def test_private_block_cut_by_truncation_does_not_leak(tmp_path):
    filler = "x" * (et.MAX_USER_TEXT - 5)
    text = f"{filler}<private>{SECRET_NOTE}</private>"
    data = et.extract(_transcript(tmp_path, text), SESSION, str(tmp_path / "q"))
    assert SECRET_NOTE not in json.dumps(data)


def test_private_text_is_absent_from_queue_file_and_records(tmp_path):
    out_dir = tmp_path / "q"
    data = et.extract(_transcript(tmp_path, f"fix bug <private>{SECRET_NOTE}</private>", f"<private>{SECRET_NOTE}</private> saved"),
                      SESSION, str(out_dir))
    assert SECRET_NOTE not in (out_dir / f"pending-{SESSION}.json").read_text(encoding="utf-8")
    db_path = tmp_path / "memory.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE knowledge(id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, type TEXT, content TEXT, "
                 "context TEXT, project TEXT, tags TEXT, status TEXT DEFAULT 'active', confidence REAL, source TEXT, "
                 "created_at TEXT, last_confirmed TEXT, recall_count INTEGER)")
    conn.commit()
    conn.close()
    ids = et.auto_save_knowledge(str(db_path), SESSION, data)
    assert ids
    conn = sqlite3.connect(db_path)
    stored = " ".join(row[0] for row in conn.execute("SELECT content FROM knowledge"))
    assert SECRET_NOTE not in stored


def test_memory_save_preview_is_sanitized(tmp_path):
    data = et.extract(_transcript(tmp_path, "hello there", "password: hunter2hunter2 value"), SESSION, str(tmp_path / "q"))
    assert "hunter2hunter2" not in json.dumps(data["memory_saves_in_session"])


@pytest.mark.skipif(os.name != "posix", reason="Windows protects the profile with ACLs, not mode bits")
def test_queue_file_is_owner_only(tmp_path):
    out_dir = tmp_path / "q"
    et.extract(_transcript(tmp_path, "hello there"), SESSION, str(out_dir))
    mode = stat.S_IMODE(os.stat(out_dir / f"pending-{SESSION}.json").st_mode)
    assert mode == et.QUEUE_FILE_MODE


@pytest.mark.parametrize("bad", ["../escape", "a/b", "", "..", "x" * 300, "with space"])
def test_session_id_is_validated(tmp_path, bad):
    with pytest.raises(ValueError):
        et.extract(_transcript(tmp_path, "hello there"), bad, str(tmp_path / "q"))
