"""One safe write path into `knowledge` for every out-of-process writer.

The server's `save_knowledge` redacts secrets, strips ``<private>`` blocks,
deduplicates and embeds. Background jobs (file watcher, Obsidian import, git
observer, idea engine, reflection scheduler, ...) used to run their own
``INSERT INTO knowledge`` and skipped all of that: a dropped ``config.yaml``
landed verbatim in recall, exports and the vault. They now call
`save_background_knowledge`, which

  1. strips ``<private>...</private>`` blocks,
  2. redacts credentials (``secret_redaction.redact_secrets``),
  3. skips exact duplicates already stored in the same project and type,
  4. inserts the row, and
  5. queues it in ``embedding_pending`` so vector recall and the cosine dedup
     can see it (the drain runs from the reflection runner and at server start).
"""
from __future__ import annotations

import json
import re
import sqlite3
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from secret_redaction import REDACTED, redact_secrets, strip_private_blocks

LOG = lambda msg: sys.stderr.write(f"[background-writes] {msg}\n")

PRIVATE_PLACEHOLDER = REDACTED
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_FIXED_COLUMNS = frozenset({
    "session_id", "type", "content", "context", "project", "tags", "status", "confidence", "source", "created_at",
})


@dataclass(frozen=True)
class SavedKnowledge:
    """Outcome of a background write: the row id and whether a new row was created."""

    id: int
    created: bool


def sanitize_text(text: str | None) -> str:
    """Strip ``<private>`` blocks, then redact credentials."""
    if not text:
        return ""
    stripped = strip_private_blocks(text, PRIVATE_PLACEHOLDER)
    cleaned, _ = redact_secrets(stripped)
    return cleaned


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")}  # noqa: S608 - fixed table names


def _sanitize_tags(tags: Sequence[str] | str | None) -> str:
    if tags is None:
        return "[]"
    if isinstance(tags, str):
        return sanitize_text(tags)
    return json.dumps([sanitize_text(str(tag)) for tag in tags], ensure_ascii=False)


def save_background_knowledge(
    conn: sqlite3.Connection,
    *,
    content: str,
    ktype: str,
    project: str,
    session_id: str,
    tags: Sequence[str] | str | None = None,
    context: str | None = None,
    source: str | None = None,
    confidence: float | None = None,
    status: str = "active",
    created_at: str | None = None,
    extra: Mapping[str, Any] | None = None,
    queue_embedding: bool = True,
) -> SavedKnowledge | None:
    """Insert a knowledge row through the shared redaction/dedup/embedding path.

    Returns None when nothing is left to store (the content was empty or only
    private text). `extra` carries additional columns (e.g. ``last_confirmed``);
    values are bound as parameters, names are validated as identifiers and must
    exist on the table.
    """
    clean = sanitize_text(content).strip()
    if not clean or clean == PRIVATE_PLACEHOLDER:
        return None

    existing = conn.execute(
        "SELECT id FROM knowledge WHERE project = ? AND type = ? AND content = ? "
        "AND COALESCE(status, 'active') = 'active' ORDER BY id LIMIT 1",
        (project, ktype, clean),
    ).fetchone()
    if existing is not None:
        return SavedKnowledge(id=int(existing[0]), created=False)

    values: dict[str, Any] = {
        "session_id": session_id,
        "type": ktype,
        "content": clean,
        "context": sanitize_text(context) if context is not None else None,
        "project": project,
        "tags": _sanitize_tags(tags),
        "status": status,
        "confidence": confidence,
        "source": source,
        "created_at": created_at or _now(),
    }
    for name, value in (extra or {}).items():
        if not _IDENTIFIER_RE.match(name) or name in _FIXED_COLUMNS:
            raise ValueError(f"invalid extra column name: {name!r}")
        values[name] = sanitize_text(value) if isinstance(value, str) else value

    available = _columns(conn, "knowledge")
    unknown = [name for name in values if name not in available and values[name] is not None]
    if unknown:
        raise ValueError(f"knowledge has no column(s): {', '.join(sorted(unknown))}")
    chosen = {name: value for name, value in values.items() if name in available and value is not None}
    column_list = ", ".join(chosen)
    placeholders = ", ".join("?" for _ in chosen)
    cursor = conn.execute(
        f"INSERT INTO knowledge ({column_list}) VALUES ({placeholders})",  # noqa: S608 - names validated above
        tuple(chosen.values()),
    )
    knowledge_id = int(cursor.lastrowid)
    conn.commit()
    if queue_embedding:
        _queue_embedding(conn, knowledge_id)
    return SavedKnowledge(id=knowledge_id, created=True)


def _queue_embedding(conn: sqlite3.Connection, knowledge_id: int) -> None:
    if not _table_exists(conn, "embedding_pending"):
        return
    from embedding_backfill import EmbeddingBackfill

    EmbeddingBackfill(conn).enqueue(knowledge_id, "queued by background writer")
