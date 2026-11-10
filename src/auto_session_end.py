#!/usr/bin/env python3
"""
Write a session_summaries row when a session ends, so the next session's
`session_init` has something to resume from.

Called by the session-end.sh hook. The hook has been invoking this script for
664 logged session ends while the file did not exist: `hook_run_script`
returns silently when its target is missing and the hook logs success
regardless, so automatic end-of-session capture has been a no-op and every
stored summary came from someone calling the MCP tool by hand.

The session id is the one the hook knows - the Claude Code session (the
transcript's basename). On Claude Code that is the same id the MCP server
uses for itself: the server reads `CLAUDE_CODE_SESSION_ID` from its own
environment and adopts it as its session id whenever the host publishes
one, instead of minting `mcp_<ts>_<pid>`. On Codex, Cursor and Docker no
such variable is published, so the ids differ, and there is no equivalent
hook at all - `SessionContinuity.session_end` creates the session row for
it either way.

The summary written here is deterministic, built from what the hook extracted
and passed on stdin (argv is visible to every local user in `ps`). User prompts
are never stored: they stay in the redacted recovery file and the transcript.
It is a floor, not a substitute for a real `session_end` call: an agent that
summarises its own session says far more than the last few messages do.
"""

import argparse
import json
import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from background_writes import PRIVATE_PLACEHOLDER, sanitize_text
from paths import memory_dir
from session_continuity import SessionContinuity
from stdio_utf8 import use_utf8_stdio

DB_PATH = os.path.join(str(memory_dir()), "memory.db")


def _llm_available() -> bool:
    """Whether the LLM gate is open. A probe must never fail the session end."""
    try:
        import config

        return bool(config.has_llm())
    except Exception:  # noqa: BLE001
        return False


def build_summary(reason: str, assistant: str) -> str:
    """``assistant`` is already sanitized."""
    reason = sanitize_text(reason)
    assistant = assistant[:800]
    parts = [f"Session ended: {reason}." if reason else "Session ended."]
    if assistant:
        parts.append(f"Assistant context: {assistant}")
    return "\n".join(parts)


def run_session_end(*, session_id: str, project: str, assistant_text: str = "",
                    branch: str = "", reason: str = "", db_path: str | None = None) -> dict:
    """Write the session_summaries row for a finished session and close its sessions row.

    ``assistant_text`` is already-extracted assistant context (the hook side
    reads it from the transcript; the remote transport sends it in the request).
    Returns a result dict; never raises for a missing DB or a failed write.
    """
    path = db_path or DB_PATH
    assistant = sanitize_text(assistant_text).strip()

    if not os.path.exists(path):
        return {"ok": False, "error": f"Memory DB not found at {path}"}

    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA busy_timeout=15000")
    try:
        # No assistant text left after redaction means nothing worth resuming
        # from (user prompts are never stored) - a summary saying only
        # "session ended" costs a row and tells the next session nothing. The
        # `sessions` row must still close, though: this cannot early-return
        # before that, or every such session keeps ended_at NULL forever.
        if not assistant.replace(PRIVATE_PLACEHOLDER, "").strip():
            SessionContinuity(db).close_session_row(
                session_id, project=project, branch=branch or None
            )
            db.commit()
            return {"ok": True, "closed_only": True}

        fallback = build_summary(reason, assistant)
        # With an LLM reachable, let it write the summary from the session's
        # own artifacts - the handful of messages the hook could extract is a
        # thin thing to resume from. `session_end` only asks the LLM when
        # `summary` is None (an explicit one always wins), so the deterministic
        # text cannot be passed as a safety net: it would suppress the
        # compression it is meant to back up. It is written afterwards instead,
        # whenever compression did not actually produce anything.
        compress = _llm_available()
        # Always call session_end, even for a session that already has a
        # recent summary: it decides for itself whether the write is a
        # duplicate, but it must still close the `sessions` row every time -
        # returning early here would leave that row open forever whenever a
        # duplicate was detected. producer="hook" on both branches, compressed
        # or not: even an LLM-written summary here only compresses what this
        # script itself extracted, so within the 300s dedup window it must
        # never displace a real session_end call's summary (see
        # _dedup_action in session_continuity.py).
        result = SessionContinuity(db).session_end(
            session_id,
            None if compress else fallback,
            project=project,
            branch=branch or None,
            producer="hook",
            auto_compress=compress,
        )
        if result.get("deduped"):
            return {"ok": True, "deduped": True}
        if compress and not result.get("compressed_used"):
            db.execute(
                "UPDATE session_summaries SET summary = ? WHERE id = ?",
                (fallback, result["id"]),
            )
            db.commit()
        return {"ok": True, "id": result["id"],
                "compressed_used": bool(result.get("compressed_used"))}
    except Exception as e:  # noqa: BLE001 - a hook must not fail the session end
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    finally:
        db.close()


def main() -> int:
    p = argparse.ArgumentParser(description="Write a session summary on session end")
    p.add_argument("--session-id", required=True)
    p.add_argument("--project", required=True)
    p.add_argument("--cwd", default="")
    p.add_argument("--branch", default="")
    p.add_argument("--reason", default="")
    a = p.parse_args()
    use_utf8_stdio()
    # Unreadable stdin reads as nothing extracted: the sessions row must still close.
    try:
        context = json.loads(sys.stdin.read() or "{}")
    except ValueError:
        context = {}
    if not isinstance(context, dict):
        context = {}

    result = run_session_end(
        session_id=a.session_id, project=a.project,
        assistant_text=str(context.get("assistant") or ""),
        branch=a.branch, reason=a.reason,
    )
    if not result["ok"]:
        print(f"auto_session_end failed: {result['error']}", file=sys.stderr)
        return 1
    if result.get("deduped"):
        print(f"session summary deduped for {a.project} ({a.session_id})")
    elif not result.get("closed_only"):
        print(f"session summary {result['id']} for {a.project} ({a.session_id})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
