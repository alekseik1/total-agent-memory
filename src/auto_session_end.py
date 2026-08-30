#!/usr/bin/env python3
"""
Write a session_summaries row when a session ends, so the next session's
`session_init` has something to resume from.

Called by the session-end.sh hook. The hook has been invoking this script for
664 logged session ends while the file did not exist: `hook_run_script`
returns silently when its target is missing and the hook logs success
regardless, so automatic end-of-session capture has been a no-op and every
stored summary came from someone calling the MCP tool by hand.

The session id is the one the hook knows — the Claude Code session (the
transcript's basename). That is not the id the MCP server issues for itself,
so `SessionContinuity.session_end` creates the session row for it.

The summary written here is deterministic, built from what the hook extracted.
It is a floor, not a substitute for a real `session_end` call: an agent that
summarises its own session says far more than the last few messages do.
"""

import argparse
import os
import sqlite3
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import memory_dir
from session_continuity import SessionContinuity

DB_PATH = os.path.join(str(memory_dir()), "memory.db")

# The hook fires on /clear and /compact as well as on exit, and those can land
# seconds apart. Two rows for one session are not wrong — each is a real end —
# but a burst of near-identical ones is noise.
_DEDUP_WINDOW_SEC = 300


def _recent_duplicate(db: sqlite3.Connection, session_id: str) -> bool:
    row = db.execute(
        "SELECT ended_at FROM session_summaries WHERE session_id = ? "
        "ORDER BY ended_at DESC LIMIT 1",
        (session_id,),
    ).fetchone()
    if not row or not row[0]:
        return False
    try:
        last = datetime.fromisoformat(str(row[0]).replace("Z", "+00:00"))
    except ValueError:
        return False
    return (datetime.now(timezone.utc) - last).total_seconds() < _DEDUP_WINDOW_SEC


def build_summary(reason: str, user_context: str, assistant_context: str) -> str:
    user = (user_context or "").strip()[:800]
    assistant = (assistant_context or "").strip()[:800]
    parts = [f"Session ended: {reason}." if reason else "Session ended."]
    if user:
        parts.append(f"User was working on: {user}")
    if assistant:
        parts.append(f"Assistant context: {assistant}")
    return "\n".join(parts)


def main() -> int:
    p = argparse.ArgumentParser(description="Write a session summary on session end")
    p.add_argument("--session-id", required=True)
    p.add_argument("--project", required=True)
    p.add_argument("--cwd", default="")
    p.add_argument("--branch", default="")
    p.add_argument("--reason", default="")
    p.add_argument("--user-context", default="")
    p.add_argument("--assistant-context", default="")
    a = p.parse_args()

    if not os.path.exists(DB_PATH):
        print(f"Memory DB not found at {DB_PATH}", file=sys.stderr)
        return 1

    # Nothing extracted means nothing worth resuming from — a summary saying
    # only "session ended" costs a row and tells the next session nothing.
    if not (a.user_context or "").strip() and not (a.assistant_context or "").strip():
        return 0

    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA busy_timeout=15000")
    try:
        if _recent_duplicate(db, a.session_id):
            return 0
        result = SessionContinuity(db).session_end(
            a.session_id,
            build_summary(a.reason, a.user_context, a.assistant_context),
            project=a.project,
            branch=a.branch or None,
        )
        print(f"session summary {result['id']} for {a.project} ({a.session_id})")
        return 0
    except Exception as e:  # noqa: BLE001 — a hook must not fail the session end
        print(f"auto_session_end failed: {e}", file=sys.stderr)
        return 1
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
