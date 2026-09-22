"""
Session continuity — v7.0 Phase G.

Provides `session_end` to capture a structured summary and `session_init`
to load the most recent unconsumed summary into a new session. This replaces
shell-hook-based recovery with first-class MCP tools.
"""

from __future__ import annotations

import json
import sqlite3
import sys
import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from active_context import (
    read_active_context,
    write_active_context,
)
from config import get_active_context_vault, is_active_context_enabled

LOG = lambda msg: sys.stderr.write(f"[session-continuity] {msg}\n")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _new_id() -> str:
    return uuid.uuid4().hex


# A caller can retry/burst session_end for the same session_id seconds apart
# (the hook fires on /clear, /compact and exit, and can race a real
# agent-authored call in either order) - see _dedup_action's docstring below
# for what happens to it.
_DEDUP_WINDOW_SEC = 300


class SessionContinuity:
    """End-of-session summary capture + start-of-session resume."""

    def __init__(self, db: sqlite3.Connection) -> None:
        self.db = db

    def _dedup_action(
        self, session_id: str, project: str, producer: str
    ) -> tuple[str, str | None]:
        """Decide what this session_end call does to session_summaries.

        Returns ("insert", None) when there is no burst to join, ("skip", id)
        when the burst's existing row must be left alone, or ("overwrite", id)
        when it must be replaced in place - `id` is the row already there.

        A session that has never closed before is never deduped: the first
        call to close it might itself be the real, agent-authored summary,
        and a later floor-text hook fire (see auto_session_end.py's module
        docstring - that text is "a floor, not a substitute for a real
        session_end call") must not be allowed to look earlier than it and
        win by skipping the real one instead.

        Once the session has closed at least once, a further call within the
        window is the same burst - what happens to it depends on `producer`,
        never on who wrote the row that's already there: that identity
        travels only as the current call's argument and is never stored. A
        burst is scoped to (session_id, project): two ends under one
        session_id for two different projects within the window are two
        different events, not a retry of the same close, so each gets its
        own row rather than the second repurposing the first's.

        producer="hook" (the session-end hook, see auto_session_end.py) never
        displaces a row already in the burst, unconditionally - the
        row's `consumed` state plays no part in that branch, since hook
        never overwrites or inserts differently depending on it. Any other
        producer ("tool" - a real session_end call) overwrites it in place:
        since the existing row's own producer is never stored, there is no
        way to tell "this was already a real one" from "this was a floor",
        so a tool call always wins - UNLESS that row was already consumed
        (handed to a session by session_init). A consumed row already did
        its job, so the burst it belonged to is over: this call starts a
        fresh row instead of resurrecting a delivered one, which would reset
        `consumed` and let the same summary be served to a second session.

        Check-then-act: the SELECT below and the caller's subsequent write
        are not one transaction, and the two producers are separate
        processes (the MCP server and the hook script) that can race it.
        Accepted rather than locked: the worst outcome is an extra row in
        the same burst - a case every action above already tolerates (a
        pre-close burst, a cross-project burst) - not a lost or corrupted
        write, since SQLite's own busy_timeout still serializes the actual
        INSERT/UPDATE statements.
        """
        row = self.db.execute(
            "SELECT ended_at FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if not row or not row[0]:
            return "insert", None
        summary_row = self.db.execute(
            "SELECT id, ended_at, consumed FROM session_summaries "
            "WHERE session_id = ? AND project = ? "
            "ORDER BY ended_at DESC, rowid DESC LIMIT 1",
            (session_id, project),
        ).fetchone()
        if not summary_row or not summary_row[1]:
            return "insert", None
        try:
            last = datetime.fromisoformat(str(summary_row[1]).replace("Z", "+00:00"))
            now = datetime.fromisoformat(_now().replace("Z", "+00:00"))
            elapsed = (now - last).total_seconds()
        except (ValueError, TypeError):
            # TypeError covers a naive `last` (no "Z"/offset) subtracted from
            # an aware `now` - fromisoformat parses a naive string without
            # raising, only the subtraction itself fails.
            return "insert", None
        if elapsed >= _DEDUP_WINDOW_SEC:
            return "insert", None
        if producer == "hook":
            return "skip", summary_row[0]
        if summary_row[2]:
            return "insert", None
        return "overwrite", summary_row[0]

    def close_session_row(
        self,
        session_id: str,
        project: str = "general",
        branch: str | None = None,
        started_at: str | None = None,
        *,
        now: str | None = None,
    ) -> str:
        """Ensure `sessions` has a row for `session_id` and mark it ended.

        Runs on every session_end call, dedup or not - a deduped/skipped
        summary is a reason to skip a second summary row, not a reason to
        leave the session looking open. Also called directly by the hook
        script when nothing was extracted to summarise, so that path still
        closes the row without going through a session_summaries write at
        all (session_end requires a summary; this does not). Caller commits.

        A session the server never opened still gets a row here. Ends arrive
        under identities the MCP process does not own - a hook naming the
        Claude Code session, a subagent, a caller that made an id up - and
        without this, `sessions` and `session_summaries` stay two
        disconnected lists: 231 of 256 summaries had no row to belong to.
        `started_at` is the caller's if given, else the end time, never a
        guess at when it began. Project and branch on an existing row are
        only overwritten while they still hold the 'general'/'' placeholder
        `Store.session_start` defaults to - a row that already knows better
        is not downgraded by a caller who omitted the argument.
        """
        now = now or _now()
        self.db.execute(
            "INSERT OR IGNORE INTO sessions (id, started_at, project, branch) VALUES (?,?,?,?)",
            (session_id, started_at or now, project, branch or ""),
        )
        self.db.execute(
            """UPDATE sessions
                  SET ended_at = ?,
                      project = CASE WHEN COALESCE(project, 'general') = 'general'
                                     THEN ? ELSE project END,
                      branch = CASE WHEN COALESCE(branch, '') = ''
                                    THEN ? ELSE branch END
                WHERE id = ?""",
            (now, project, branch or "", session_id),
        )
        return now

    # ──────────────────────────────────────────────
    # End of session
    # ──────────────────────────────────────────────

    def session_end(
        self,
        session_id: str,
        summary: str | None = None,
        *,
        highlights: list[str] | None = None,
        pitfalls: list[str] | None = None,
        next_steps: list[str] | None = None,
        open_questions: list[str] | None = None,
        context_blob: str | None = None,
        project: str = "general",
        branch: str | None = None,
        producer: Literal["tool", "hook"] = "tool",
        started_at: str | None = None,
        auto_compress: bool = False,
        transcript: str | None = None,
    ) -> dict[str, Any]:
        if not session_id:
            raise ValueError("session_id required")

        project_inferred = False
        if not project or project == "general":
            inferred = self._infer_project(session_id)
            if inferred:
                project = inferred
                project_inferred = True
                LOG(f"project inferred as '{project}' for session {session_id}")

        # A non-auto call must always name a real summary, whatever dedup
        # will end up doing with it - checked before dedup is even consulted
        # so a genuinely bad argument still raises (see _dedup_action's
        # docstring for why a still-open session is never deduped, and why
        # the policy branches on `producer`, not on the row that's there).
        if not auto_compress and not summary:
            raise ValueError("summary required")

        action, target_id = self._dedup_action(session_id, project, producer)

        # Explicit args always win over LLM output. We compute LLM-derived
        # fields first so they can fill any gaps, then overlay the explicit
        # arguments on top. Skipped entirely when this call's own content is
        # about to be discarded anyway (action == "skip") - a floor that will
        # never be written is not worth an LLM call.
        compressed_used = False
        llm_error: str | None = None
        if auto_compress:
            if action != "skip":
                llm_summary, llm_next_steps, llm_pitfalls, llm_error = self._compress_session(
                    session_id=session_id,
                    project=project,
                    transcript=transcript,
                )
                if summary is None and llm_summary:
                    summary = llm_summary
                if next_steps is None and llm_next_steps:
                    next_steps = llm_next_steps
                if pitfalls is None and llm_pitfalls:
                    pitfalls = llm_pitfalls
                compressed_used = llm_error is None and (
                    bool(llm_summary) or bool(llm_next_steps) or bool(llm_pitfalls)
                )
            # Ensure we always have *some* summary so downstream NOT NULL holds.
            if summary is None:
                summary = ""

        sid = None
        now = _now()
        if action == "insert":
            sid = _new_id()
            self.db.execute(
                """INSERT INTO session_summaries
                   (id, session_id, project, branch, summary, highlights, pitfalls,
                    next_steps, open_questions, context_blob, started_at,
                    ended_at, consumed, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)""",
                (
                    sid, session_id, project, branch, summary,
                    json.dumps(highlights or []),
                    json.dumps(pitfalls or []),
                    json.dumps(next_steps or []),
                    json.dumps(open_questions or []),
                    context_blob, started_at, now, now,
                ),
            )
        elif action == "overwrite":
            sid = target_id
            # branch/started_at use COALESCE(?, existing): a second call in
            # the burst that omits them (e.g. a hook retry with no --branch)
            # must not null out what the first call already stored there.
            self.db.execute(
                """UPDATE session_summaries
                      SET project = ?, branch = COALESCE(?, branch), summary = ?,
                          highlights = ?, pitfalls = ?, next_steps = ?,
                          open_questions = ?, context_blob = ?,
                          started_at = COALESCE(?, started_at), ended_at = ?,
                          consumed = 0
                    WHERE id = ?""",
                (
                    project, branch, summary,
                    json.dumps(highlights or []),
                    json.dumps(pitfalls or []),
                    json.dumps(next_steps or []),
                    json.dumps(open_questions or []),
                    context_blob, started_at, now,
                    target_id,
                ),
            )
        else:
            # action == "skip": no summary write - a hook fire never
            # displaces whatever is already in the burst's row. `sid` still
            # names that row so the caller (and mark_unconsumed) can address
            # it, even though this call didn't touch it.
            sid = target_id

        self.close_session_row(session_id, project, branch, started_at, now=now)
        self.db.commit()

        result: dict[str, Any] = {
            "id": sid,
            "session_id": session_id,
            "project": project,
            "ended_at": now,
            "summary_len": len(summary),
            "next_steps_count": len(next_steps or []),
        }
        if action == "skip":
            result["deduped"] = True
        if project_inferred:
            result["project_inferred"] = True
        if auto_compress:
            result["auto_compress"] = True
            result["compressed_used"] = compressed_used
            if llm_error:
                result["auto_compress_error"] = llm_error

        # Markdown live-doc projection (optional, env-gated). Skipped along
        # with the summary write itself: a deduped hook floor must not
        # repaint the live document over whatever a real summary (or an
        # earlier call in this burst) already wrote there.
        if action != "skip" and is_active_context_enabled():
            try:
                path = write_active_context(
                    project,
                    summary,
                    next_steps or [],
                    pitfalls or [],
                    vault_root=get_active_context_vault(),
                    session_id=session_id,
                )
                result["active_context_path"] = str(path)
            except OSError as e:
                LOG(f"active_context write failed: {e}")
                result["active_context_error"] = str(e)

        return result

    def _infer_project(self, session_id: str) -> str | None:
        """Best-effort project inference for callers that omit `project`."""
        queries = (
            """SELECT project FROM knowledge
               WHERE session_id = ? AND project != 'general'
               ORDER BY created_at DESC LIMIT 1""",
            """SELECT project FROM session_summaries
               WHERE session_id = ? AND project != 'general'
               ORDER BY ended_at DESC, rowid DESC LIMIT 1""",
        )
        for q in queries:
            try:
                row = self.db.execute(q, (session_id,)).fetchone()
            except sqlite3.OperationalError:
                continue
            if row and row[0]:
                return row[0]
        return None

    # ──────────────────────────────────────────────
    # Auto-compress helpers
    # ──────────────────────────────────────────────

    # Rough token estimate: 1 token ≈ 4 chars. Cap context at ~6k tokens.
    _MAX_TRANSCRIPT_CHARS = 6000 * 4

    _COMPRESS_PROMPT = (
        "Summarize the following coding session into:\n"
        "1. SUMMARY: 2-3 sentences describing what was done\n"
        "2. NEXT_STEPS: bullet list of actionable items (what to do next session)\n"
        "3. PITFALLS: bullet list of gotchas/constraints to remember\n\n"
        "Session log:\n{context}\n\n"
        'Return JSON ONLY (no markdown fences, no commentary):\n'
        '{{"summary": "...", "next_steps": ["..."], "pitfalls": ["..."]}}'
    )

    def _collect_session_context(self, session_id: str, project: str) -> str:
        """Stitch together saved artifacts for ``session_id`` into a flat log.

        Pulls knowledge rows (memory_save / learn_error / kg_add_fact all land
        there as distinct types) scoped to this session. If the table is
        missing or no rows match, returns ''.
        """
        try:
            cur = self.db.cursor()
            rows = cur.execute(
                """SELECT type, content FROM knowledge
                   WHERE session_id = ? AND project = ?
                   ORDER BY id ASC""",
                (session_id, project),
            ).fetchall()
        except sqlite3.Error as exc:
            LOG(f"collect_session_context query failed: {exc}")
            return ""

        parts: list[str] = []
        for row in rows:
            rtype = row["type"] if isinstance(row, sqlite3.Row) else row[0]
            content = row["content"] if isinstance(row, sqlite3.Row) else row[1]
            if not content:
                continue
            parts.append(f"[{rtype}] {content}")
        return "\n".join(parts)

    def _compress_session(
        self,
        *,
        session_id: str,
        project: str,
        transcript: str | None,
    ) -> tuple[str, list[str], list[str], str | None]:
        """Run the LLM provider to distil session_id into summary/next/pitfalls.

        Returns (summary, next_steps, pitfalls, error_msg). On any failure
        error_msg is set and the first three fields come back empty — caller
        is expected to overlay explicit args on top.
        """
        ctx = transcript if transcript is not None else self._collect_session_context(
            session_id, project
        )
        if not ctx.strip():
            return "", [], [], "empty_context"

        # Truncate to protect provider-side token budgets.
        if len(ctx) > self._MAX_TRANSCRIPT_CHARS:
            ctx = ctx[: self._MAX_TRANSCRIPT_CHARS]

        try:
            import config as _config
            from llm_provider import make_provider

            provider = make_provider(_config.get_llm_provider())
        except Exception as exc:  # noqa: BLE001 — bad provider config etc.
            LOG(f"auto_compress provider init failed: {exc}")
            return "", [], [], f"provider_init: {exc}"

        if not provider.available():
            LOG(f"auto_compress provider '{getattr(provider, 'name', '?')}' unavailable")
            return "", [], [], "provider_unavailable"

        prompt = self._COMPRESS_PROMPT.format(context=ctx)
        try:
            raw = provider.complete(prompt, max_tokens=2000, temperature=0.2)
        except Exception as exc:  # noqa: BLE001 — urllib/http/runtime
            LOG(f"auto_compress LLM call failed: {exc}")
            return "", [], [], f"llm_call: {exc}"

        parsed = self._parse_compress_response(raw)
        if parsed is None:
            LOG("auto_compress LLM returned malformed JSON")
            return "", [], [], "malformed_json"

        summary = str(parsed.get("summary") or "").strip()
        next_steps = [str(x).strip() for x in (parsed.get("next_steps") or []) if x]
        pitfalls = [str(x).strip() for x in (parsed.get("pitfalls") or []) if x]
        return summary, next_steps, pitfalls, None

    @staticmethod
    def _parse_compress_response(raw: str) -> dict[str, Any] | None:
        """Extract a JSON object from an LLM response.

        Handles markdown code fences and leading/trailing chatter.
        """
        if not raw:
            return None
        text = raw.strip()
        # Strip ```json … ``` / ``` … ``` fences if present.
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:]
            text = text.strip()
        # Locate the outermost JSON object.
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end < start:
            return None
        try:
            obj = json.loads(text[start : end + 1])
        except (json.JSONDecodeError, ValueError):
            return None
        return obj if isinstance(obj, dict) else None

    # ──────────────────────────────────────────────
    # Start of session (resume)
    # ──────────────────────────────────────────────

    def session_init(
        self,
        *,
        project: str = "general",
        mark_consumed: bool = True,
        include_pitfalls: bool = True,
    ) -> dict[str, Any] | None:
        """Fetch the most recent unconsumed summary for `project`.

        Returns None if nothing to resume. When `mark_consumed=True`, sets the
        consumed flag so the same summary is not replayed twice.
        """
        cur = self.db.cursor()
        cur.row_factory = sqlite3.Row
        row = cur.execute(
            """SELECT * FROM session_summaries
               WHERE project = ? AND consumed = 0
               ORDER BY ended_at DESC, rowid DESC LIMIT 1""",
            (project,),
        ).fetchone()

        # Always attempt to surface markdown projection even if DB is empty
        md_doc = self._read_markdown(project)

        if not row:
            # Fallback to markdown only when the project has never had a row
            # in session_summaries. If there are consumed rows, the DB is the
            # source of truth and "nothing to resume" must mean None.
            has_any = cur.execute(
                "SELECT 1 FROM session_summaries WHERE project = ? LIMIT 1",
                (project,),
            ).fetchone()
            if has_any or md_doc is None:
                return None
            # Orphan markdown (no DB row ever) — surface it
            return {
                "summary": md_doc.get("summary", ""),
                "next_steps": md_doc.get("next_steps", []),
                "pitfalls": md_doc.get("pitfalls", []) if include_pitfalls else [],
                "active_context": md_doc,
                "markdown_updated_at": md_doc.get("updated_at"),
                "source": "markdown",
            }

        d = dict(row)
        for k in ("highlights", "pitfalls", "next_steps", "open_questions"):
            if d.get(k):
                try:
                    d[k] = json.loads(d[k])
                except (json.JSONDecodeError, TypeError):
                    d[k] = []
            else:
                d[k] = []

        if not include_pitfalls:
            d["pitfalls"] = []

        # Attach markdown projection; DB wins on summary conflicts but we
        # still expose markdown_updated_at so caller can detect staleness.
        d["active_context"] = md_doc
        d["markdown_updated_at"] = md_doc.get("updated_at") if md_doc else None
        if md_doc and md_doc.get("summary") and md_doc["summary"] != d.get("summary"):
            d["markdown_stale"] = True
        else:
            d["markdown_stale"] = False

        if mark_consumed:
            self.db.execute(
                "UPDATE session_summaries SET consumed = 1 WHERE id = ?",
                (d["id"],),
            )
            self.db.commit()

        return d

    @staticmethod
    def _read_markdown(project: str) -> dict[str, Any] | None:
        """Read the markdown projection. Returns None on missing file or error."""
        try:
            return read_active_context(project, vault_root=get_active_context_vault())
        except (OSError, ValueError):
            return None

    # ──────────────────────────────────────────────
    # Listing / stats
    # ──────────────────────────────────────────────

    def list_summaries(
        self,
        *,
        project: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        cur = self.db.cursor()
        cur.row_factory = sqlite3.Row
        if project:
            rows = cur.execute(
                """SELECT id, session_id, project, summary, consumed, ended_at
                   FROM session_summaries WHERE project = ?
                   ORDER BY ended_at DESC, rowid DESC LIMIT ?""",
                (project, limit),
            ).fetchall()
        else:
            rows = cur.execute(
                """SELECT id, session_id, project, summary, consumed, ended_at
                   FROM session_summaries
                   ORDER BY ended_at DESC, rowid DESC LIMIT ?""",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def stats(self, *, project: str | None = None) -> dict[str, int]:
        cur = self.db.cursor()
        if project:
            total = cur.execute(
                "SELECT COUNT(*) FROM session_summaries WHERE project = ?",
                (project,),
            ).fetchone()[0]
            pending = cur.execute(
                """SELECT COUNT(*) FROM session_summaries
                   WHERE project = ? AND consumed = 0""",
                (project,),
            ).fetchone()[0]
        else:
            total = cur.execute(
                "SELECT COUNT(*) FROM session_summaries"
            ).fetchone()[0]
            pending = cur.execute(
                "SELECT COUNT(*) FROM session_summaries WHERE consumed = 0"
            ).fetchone()[0]
        return {"total_summaries": total, "pending": pending,
                "consumed": total - pending}

    def mark_unconsumed(self, summary_id: str) -> bool:
        cur = self.db.execute(
            "UPDATE session_summaries SET consumed = 0 WHERE id = ?",
            (summary_id,),
        )
        self.db.commit()
        return cur.rowcount > 0
