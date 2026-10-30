#!/usr/bin/env python3
"""
Auto-save session context to MCP Memory on session end.
Called by session-end.sh hook to preserve context
even when Claude didn't manually save.

Saves a lightweight session summary as a 'fact' record
with tags=['session-autosave', 'context-recovery'].
"""

import argparse
import sqlite3
import os
import sys
import json
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from memory_core.timestamps import utc_now
from paths import memory_dir
from stdio_utf8 import use_utf8_stdio
from secret_redaction import redact_secrets

MEMORY_DIR = str(memory_dir())
DB_PATH = os.path.join(MEMORY_DIR, "memory.db")


def _count_lines(text: str) -> int:
    return sum(1 for line in text.splitlines() if line.strip())


def save_session_context(project: str, cwd: str, reason: str,
                         user_context: str, assistant_context: str) -> None:
    """Save session context to memory.db as a fact."""
    if not os.path.exists(DB_PATH):
        print(f"Memory DB not found at {DB_PATH}", file=sys.stderr)
        return

    user_text = redact_secrets(user_context.strip())[0] if user_context else ""
    assistant_text = redact_secrets(assistant_context.strip())[0] if assistant_context else ""

    if not user_text and not assistant_text:
        return  # Nothing to save

    timestamp = utc_now()
    session_id = f"auto_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Only a size summary is stored: prompts and replies stay in the (redacted, private) recovery file and
    # the transcript, never in an active fact that recall, export and team sync would spread.
    content = (f"Session auto-save ({redact_secrets(reason)[0]}). "
               f"{_count_lines(user_text)} user message(s), {len(user_text) + len(assistant_text)} characters "
               "of context were captured to the session recovery file.\n")

    context_field = f"Project: {project}, CWD: {cwd}, Ended: {timestamp}"
    tags = json.dumps(["session-autosave", "context-recovery", project])

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Check for recent duplicate (same project, within 5 minutes)
        cursor.execute("""
            SELECT id FROM knowledge
            WHERE project = ? AND status = 'active'
              AND tags LIKE '%session-autosave%'
              AND created_at > strftime('%Y-%m-%dT%H:%M:%f000Z', 'now', '-5 minutes')
            LIMIT 1
        """, (project,))

        if cursor.fetchone():
            conn.close()
            return  # Skip duplicate

        cursor.execute("""
            INSERT INTO knowledge (session_id, type, content, context, project, tags, status, confidence, source, created_at, last_confirmed, recall_count)
            VALUES (?, 'fact', ?, ?, ?, ?, 'active', 0.6, 'auto', ?, ?, 0)
        """, (session_id, content, context_field, project, tags, timestamp, timestamp))

        conn.commit()
        record_id = cursor.lastrowid
        conn.close()

        print(f"Auto-saved session context: id={record_id}, project={project}")

    except Exception as e:
        print(f"Error saving session context: {e}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Auto-save session context to memory")
    parser.add_argument("--project", required=True, help="Project name")
    parser.add_argument("--cwd", required=True, help="Working directory")
    parser.add_argument("--reason", required=True, help="Session end reason")
    parser.add_argument("--user-context", default="", help="Last user messages (visible in ps; prefer stdin)")
    parser.add_argument("--assistant-context", default="", help="Last assistant context")
    parser.add_argument("--context-stdin", action="store_true",
                        help='Read {"user": ..., "assistant": ...} from stdin instead of argv')

    args = parser.parse_args()
    use_utf8_stdio()
    user_context, assistant_context = args.user_context, args.assistant_context
    if args.context_stdin:
        payload = json.loads(sys.stdin.read() or "{}")
        user_context = str(payload.get("user") or "")
        assistant_context = str(payload.get("assistant") or "")

    save_session_context(
        project=args.project,
        cwd=args.cwd,
        reason=args.reason,
        user_context=user_context,
        assistant_context=assistant_context,
    )


if __name__ == "__main__":
    main()
