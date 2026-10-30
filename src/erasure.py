"""Hard erasure of a personal-memory record: the row, its earlier versions and everything derived.

`memory_delete` without `hard` only hides a record. This removes it for good: SQLite
rows (triggers take the FTS index, atomic facts, graph links and evidence passages),
vectors in every embedding space, queue and log rows, quotes of it inside later
versions, agent action log rows that quote it, and the copies of its text in the raw
call log. `secure_delete` zeroes the freed pages and a WAL checkpoint truncates the
log, so the text is not left on disk.
"""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import time
from pathlib import Path

from ambient.actions import ActionLog

ERASED = "[erased]"
ACTION_LOG_TABLE = "agent_actions"
SECONDS_PER_DAY = 86_400
# Entry types the server writes to the raw call log (src/server.py `_call_tool_impl`).
RAW_CALL_TYPE = "tool_call"
RAW_RESULT_TYPE = "tool_result"

# Tables whose `knowledge_id` rows describe a record. Triggers on DELETE FROM knowledge
# already cover knowledge_fts, atomic_fact*, knowledge_nodes and passage_sources.
DERIVED_TABLES = (
    "embeddings", "enrichment_queue", "deep_enrichment_queue", "representations_queue",
    "triple_extraction_queue", "knowledge_representations", "knowledge_enrichment", "episode_facts",
    "entity_dedup_log", "filter_savings", "quality_gate_log", "write_intents", "knowledge_nodes_quarantine",
    "evidence_passages",
)
PAIR_TABLES = (("contradiction_log", ("new_knowledge_id", "candidate_knowledge_id")),
               ("relations", ("from_id", "to_id")))
# `memory_update` stores "Was: <first 200 chars of the old content>" in the new version's context.
QUOTED_PREFIX_CHARS = 200


class ErasureUnsupported(RuntimeError):
    """Hard erasure runs on the personal SQLite store only."""


def version_chain(db, record_id: int) -> list[int]:
    """`record_id` and every earlier version that was superseded into it, oldest last."""
    chain, frontier = [record_id], [record_id]
    while frontier:
        marks = ",".join("?" * len(frontier))
        rows = db.execute(f"SELECT id FROM knowledge WHERE superseded_by IN ({marks})", frontier).fetchall()
        frontier = [row[0] for row in rows if row[0] not in chain]
        chain.extend(frontier)
    return chain


def _existing_tables(db) -> set[str]:
    return {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def _fts_tables(db) -> list[str]:
    return [row[0] for row in db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND sql LIKE 'CREATE VIRTUAL TABLE%USING fts5%'")]


def _merge_fts(db) -> None:
    """FTS5 keeps deleted tokens in older index segments until a merge; `optimize` rewrites them."""
    for table in _fts_tables(db):
        db.execute(f"INSERT INTO {table}({table}) VALUES('optimize')")


CHROMA_DELETE_OPERATION = 3


def compact_chroma(chroma_dir: Path, ids: list[int]) -> None:
    """Remove the erased documents' text from Chroma's own SQLite file.

    Chroma replays writes from `embeddings_queue`, whose `metadata` holds the document text. The
    add/upsert entries of erased ids lose it (their DELETE entries stay for the vector index);
    then its FTS is merged and the file vacuumed.
    """
    path = chroma_dir / "chroma.sqlite3"
    if not path.is_file():
        return
    db = sqlite3.connect(path, timeout=30)
    try:
        db.execute("PRAGMA secure_delete=ON")
        with db:
            if db.execute("SELECT 1 FROM sqlite_master WHERE name='embeddings_queue'").fetchone():
                marks = ",".join("?" * len(ids))
                db.execute(f"UPDATE embeddings_queue SET metadata=NULL WHERE id IN ({marks}) AND operation != ?",
                           (*[str(i) for i in ids], CHROMA_DELETE_OPERATION))
            _merge_fts(db)
        db.execute("VACUUM")
        db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        db.close()


def _linked_call_ids(paths: list[Path], record_ids: set[int]) -> set[str]:
    """Ids of logged tool calls whose `tool_result` line reports one of `record_ids`."""
    call_ids: set[str] = set()
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if RAW_RESULT_TYPE not in line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("type") == RAW_RESULT_TYPE and record_ids.intersection(entry.get("knowledge_ids") or ()):
                call_ids.add(str(entry.get("call_id")))
    return call_ids


def _erase_call_args(line: str, call_ids: set[str]) -> str:
    """The logged call with its arguments blanked, when the call produced an erased record."""
    try:
        entry = json.loads(line)
    except json.JSONDecodeError:
        return line
    if not isinstance(entry, dict) or entry.get("type") != RAW_CALL_TYPE or str(entry.get("call_id")) not in call_ids:
        return line
    entry["args"] = ERASED
    newline = "\n" if line.endswith("\n") else ""
    return json.dumps(entry, ensure_ascii=False, default=str) + newline


def scrub_raw_logs(raw_dir: Path, texts: list[str], record_ids: list[int] | None = None) -> int:
    """Erase the record from the JSONL call logs; returns the number of files rewritten.

    The log holds what the caller *submitted*, which differs from the stored text whenever
    the save pipeline rewrote it (filters, coreference, redaction), so a text search alone
    misses it. Each logged call carries a `call_id`; the `tool_result` line written after it
    names the records it produced. Calls that produced an erased record lose their arguments;
    every remaining copy of `texts` is replaced as well.
    """
    needles = sorted({json.dumps(t, ensure_ascii=False)[1:-1] for t in texts if t and t.strip()}, key=len,
                     reverse=True)
    if not raw_dir.is_dir():
        return 0
    paths = sorted(raw_dir.glob("*.jsonl"))
    call_ids = _linked_call_ids(paths, set(record_ids)) if record_ids else set()
    if not needles and not call_ids:
        return 0
    rewritten = 0
    for path in paths:
        original = path.read_text(encoding="utf-8")
        cleaned = original
        if call_ids:
            cleaned = "".join(_erase_call_args(line, call_ids) for line in cleaned.splitlines(keepends=True))
        for needle in needles:
            cleaned = cleaned.replace(needle, ERASED)
        if cleaned == original:
            continue
        fd, temp = tempfile.mkstemp(dir=raw_dir, prefix=".erase-", suffix=".jsonl")
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(cleaned)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
        rewritten += 1
    return rewritten


def erase(store, record_id: int, raw_dir: Path, chroma_dir: Path | None = None) -> dict | None:
    """Erase `record_id` with its earlier versions; None when no such record exists."""
    return erase_many(store, [record_id], raw_dir, chroma_dir)


def erase_many(store, record_ids: list[int], raw_dir: Path, chroma_dir: Path | None = None) -> dict | None:
    """Erase every record in `record_ids` with its earlier versions in one pass; None when none exists.

    One FTS merge, one WAL checkpoint and one raw-log scrub for the whole batch, so
    `tam forget-auto` costs the same for one record or a thousand.
    """
    if getattr(store, "is_postgres", False):
        raise ErasureUnsupported("Hard delete is available for personal (SQLite) memory only")
    db = store.db
    existing = [rid for rid in dict.fromkeys(record_ids)
                if db.execute("SELECT 1 FROM knowledge WHERE id=?", (rid,)).fetchone() is not None]
    if not existing:
        return None
    ids: list[int] = []
    for rid in existing:
        ids.extend(kid for kid in version_chain(db, rid) if kid not in ids)
    marks = ",".join("?" * len(ids))
    rows = db.execute(f"SELECT id, content, context, project FROM knowledge WHERE id IN ({marks})", ids).fetchall()
    texts = [value for row in rows for value in (row[1], row[2]) if value]
    projects = sorted({row[3] for row in rows if row[3]})
    spaces = {row[0] for row in db.execute(
        f"SELECT DISTINCT embedding_space FROM embeddings WHERE knowledge_id IN ({marks})", ids)}

    tables = _existing_tables(db)
    actions_erased = 0
    db.execute("PRAGMA secure_delete=ON")
    try:
        for kid in ids:
            store._delete_embedding(kid)
        for table in DERIVED_TABLES:
            if table in tables:
                db.execute(f"DELETE FROM {table} WHERE knowledge_id IN ({marks})", ids)
        for table, columns in PAIR_TABLES:
            if table in tables:
                where = " OR ".join(f"{column} IN ({marks})" for column in columns)
                db.execute(f"DELETE FROM {table} WHERE {where}", ids * len(columns))
        for row in rows:
            quoted = (row[1] or "")[:QUOTED_PREFIX_CHARS]
            if quoted.strip():
                db.execute(f"UPDATE knowledge SET context=replace(context, ?, ?) "
                           f"WHERE instr(context, ?) > 0 AND id NOT IN ({marks})", (quoted, ERASED, quoted, *ids))
        if ACTION_LOG_TABLE in tables:
            # A logged command or edit can carry the record's text (src/ambient/actions.py).
            actions_erased = ActionLog(db).erase_mentions(texts)
        db.execute(f"DELETE FROM knowledge WHERE id IN ({marks})", ids)
        _merge_fts(db)
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.execute("PRAGMA secure_delete=OFF")
    db.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    result = {"erased": ids, "projects": projects, "raw_logs_rewritten": scrub_raw_logs(raw_dir, texts, ids),
              "actions_erased": actions_erased}
    for space in sorted({"text", *spaces}):
        collection = store._chroma_collection_for(space)
        if collection is None:
            continue
        try:
            collection.delete(ids=[str(kid) for kid in ids])
        except Exception as error:  # noqa: BLE001 — chroma raises its own types; SQLite erasure already committed
            result.setdefault("vector_store_errors", []).append(f"{space}: {error}")
    if chroma_dir is not None and store._chroma_client is not None:
        try:
            compact_chroma(chroma_dir, ids)
        except sqlite3.Error as error:
            result.setdefault("vector_store_errors", []).append(f"compact: {error}")
    return result


def prune_raw_logs(raw_dir: Path, days: str | None, now: float | None = None) -> int:
    """Remove call logs not written for `days` days; unset, blank or invalid keeps everything."""
    if not days or not str(days).strip().isdigit() or int(days) <= 0 or not raw_dir.is_dir():
        return 0
    cutoff = (time.time() if now is None else now) - int(days) * SECONDS_PER_DAY
    removed = 0
    for path in raw_dir.glob("*.jsonl"):
        if path.stat().st_mtime < cutoff:
            path.unlink()
            removed += 1
    return removed
