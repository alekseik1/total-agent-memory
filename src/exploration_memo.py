"""Exploration memos: conclusions about a codebase, tied to the exact bytes they were drawn from.

Agents re-read the same files every session to answer the same questions.
``memory_exploration_record`` stores a question, the conclusion and the
SHA-256 of every file read to reach it; ``memory_exploration_recall`` finds
memos for a new question (semantic similarity over the stored question
embedding, blended with TAM's word-level Jaccard, exact question match = 1.0)
and re-hashes the files: ``fresh`` means every file is byte-identical, so the
conclusion can be reused without re-reading; ``stale`` lists the changed and
missing files that need another look.

Files are stored relative to the project root. A file outside the root
(including through a symlink) is refused at record time and reported as
``outside_root`` at recall time, so a memo never makes the server read
outside the tree it was given.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
import struct
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from memory_core.timestamps import utc_now
from secret_redaction import redact_secrets

LOGGER = logging.getLogger(__name__)

RECORD_TOOL = "memory_exploration_record"
RECALL_TOOL = "memory_exploration_recall"
TOOL_NAMES = frozenset({RECORD_TOOL, RECALL_TOOL})

MAX_QUESTION_CHARS = 2_000
MAX_CONCLUSION_CHARS = 20_000
MAX_PROJECT_CHARS = 200
MAX_PATH_CHARS = 1_024
MAX_FILES_PER_MEMO = 64
# Files larger than this are refused at record time (and reported changed at recall).
MAX_FILE_BYTES = 32 * 1024 * 1024
HASH_CHUNK_BYTES = 1024 * 1024
DEFAULT_RECALL_LIMIT = 5
MAX_RECALL_LIMIT = 50
# Most recently updated memos of a project scored per recall.
CANDIDATE_SCAN_LIMIT = 2_000
# Memos scoring below this are not returned.
MIN_RECALL_SCORE = 0.2
SEMANTIC_WEIGHT = 0.8
LEXICAL_WEIGHT = 0.2
EXACT_MATCH_SCORE = 1.0
FLOAT32_BYTES = 4

STATUS_FRESH = "fresh"
STATUS_STALE = "stale"

Embedder = Callable[[list[str]], Sequence[Sequence[float] | None] | None]
LexicalScorer = Callable[[str, str], float]
FileState = tuple[str, str | None]


class ExplorationError(ValueError):
    """A memo request the caller has to fix (bad path, limit exceeded, unknown root)."""


class RecordArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS,
                          description="The question the exploration answered, e.g. 'how is auth wired?'")
    conclusion: str = Field(min_length=1, max_length=MAX_CONCLUSION_CHARS,
                            description="What was concluded, self-contained enough to reuse without re-reading")
    files: list[str] = Field(min_length=1, max_length=MAX_FILES_PER_MEMO,
                             description="Files read to reach the conclusion: paths relative to root, or absolute "
                                         "paths inside root")
    project: str = Field(min_length=1, max_length=MAX_PROJECT_CHARS, description="Project name")
    root: str | None = Field(default=None, max_length=MAX_PATH_CHARS,
                             description="Absolute project root directory; default: the server's working directory")


class RecallArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS, description="The question being asked now")
    project: str = Field(min_length=1, max_length=MAX_PROJECT_CHARS, description="Project name")
    root: str | None = Field(default=None, max_length=MAX_PATH_CHARS,
                             description="Absolute project root to verify files against; default: the server's "
                                         "working directory")
    limit: int = Field(default=DEFAULT_RECALL_LIMIT, ge=1, le=MAX_RECALL_LIMIT, description="Maximum memos returned")
    include_stale: bool = Field(default=True, description="Also return memos whose files changed since recording")


RECORD_DESCRIPTION = (
    "Record the conclusion of a codebase exploration together with the SHA-256 of every file read to reach it. "
    "Call it after answering a structural question by reading files (how X is wired, where Y is decided). "
    "Paths are stored relative to `root` (default: the server's working directory); files outside root are "
    f"refused. Up to {MAX_FILES_PER_MEMO} files of at most {MAX_FILE_BYTES // (1024 * 1024)} MB each. Recording "
    "the same question again for the project replaces the earlier memo."
)
RECALL_DESCRIPTION = (
    "Before re-reading files to answer a codebase question, look up earlier exploration conclusions. Results are "
    "ranked by question similarity; each carries status 'fresh' (every source file is byte-identical to when the "
    "conclusion was recorded: reuse it) or 'stale' (changed_files / missing_files list what to re-read)."
)


def _schema(model: type[BaseModel]) -> dict[str, Any]:
    schema = model.model_json_schema()
    schema.pop("title", None)
    schema["additionalProperties"] = False
    return schema


def record_input_schema() -> dict[str, Any]:
    return _schema(RecordArguments)


def recall_input_schema() -> dict[str, Any]:
    return _schema(RecallArguments)


def normalise_question(question: str) -> str:
    return re.sub(r"\s+", " ", question.casefold()).strip().rstrip("?.!¿¡ ").strip()


def question_hash(question: str) -> str:
    return hashlib.sha256(normalise_question(question).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_root(raw: str | None) -> Path:
    candidate = Path(raw).expanduser() if raw else Path.cwd()
    if not candidate.is_absolute():
        raise ExplorationError(f"root must be an absolute path, got {raw!r}")
    try:
        root = candidate.resolve(strict=True)
    except (FileNotFoundError, RuntimeError) as error:
        raise ExplorationError(f"root {candidate} does not exist") from error
    if not root.is_dir():
        raise ExplorationError(f"root {root} is not a directory")
    return root


def _inside(root: Path, path: Path) -> str | None:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return None


@dataclass(frozen=True, slots=True)
class FileFingerprint:
    path: str
    sha256: str
    size: int


def fingerprint(root: Path, raw: str) -> FileFingerprint:
    """Hash one file named relative to ``root`` (or absolute inside it); refuses anything outside root."""
    if not raw.strip() or len(raw) > MAX_PATH_CHARS or "\x00" in raw:
        raise ExplorationError(f"invalid file path {raw[:80]!r}")
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        resolved = candidate.resolve(strict=True)
    except (FileNotFoundError, RuntimeError) as error:
        raise ExplorationError(f"file {raw!r} does not exist under {root}") from error
    relative = _inside(root, resolved)
    if relative is None:
        raise ExplorationError(f"file {raw!r} resolves outside the project root {root}")
    if not resolved.is_file():
        raise ExplorationError(f"{relative} is not a regular file")
    size = resolved.stat().st_size
    if size > MAX_FILE_BYTES:
        raise ExplorationError(f"{relative} is {size} bytes; the limit is {MAX_FILE_BYTES}")
    return FileFingerprint(relative, sha256_file(resolved), size)


def _pack(vector: Sequence[float]) -> bytes:
    return struct.pack(f"{len(vector)}f", *vector)


def _unpack(blob: bytes | memoryview | None) -> list[float] | None:
    if blob is None:
        return None
    data = bytes(blob)
    if not data or len(data) % FLOAT32_BYTES:
        return None
    return list(struct.unpack(f"{len(data) // FLOAT32_BYTES}f", data))


def _cosine(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or not left:
        return None
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    norm = (sum(a * a for a in left) ** 0.5) * (sum(b * b for b in right) ** 0.5)
    return dot / norm if norm else None


class ExplorationMemos:
    """Record and recall exploration memos on a TAM store connection."""

    def __init__(self, db: Any, *, lexical: LexicalScorer, embed: Embedder | None = None,
                 model_name: str | None = None) -> None:
        self._db = db
        self._lexical = lexical
        self._embed = embed
        self._model_name = model_name

    def _vector(self, text: str) -> list[float] | None:
        if self._embed is None or not self._model_name:
            return None
        try:
            vectors = self._embed([text])
        except (RuntimeError, ValueError, OSError) as error:
            LOGGER.warning(json.dumps({"event": "exploration_embed_failed", "error": str(error)}))
            return None
        if not vectors or vectors[0] is None:
            return None
        return [float(value) for value in vectors[0]]

    def record(self, arguments: RecordArguments) -> dict[str, Any]:
        root = resolve_root(arguments.root)
        question = redact_secrets(arguments.question.strip())[0]
        conclusion = redact_secrets(arguments.conclusion.strip())[0]
        project = arguments.project.strip()
        if not question or not conclusion or not project:
            raise ExplorationError("question, conclusion and project must be non-blank")
        prints: dict[str, FileFingerprint] = {}
        for raw in arguments.files:
            item = fingerprint(root, raw)
            prints[item.path] = item
        vector = self._vector(question)
        digest = question_hash(question)
        now = utc_now()
        try:
            existing = self._db.execute(
                "SELECT id, created_at FROM exploration_memos WHERE project=? AND question_hash=?",
                (project, digest)).fetchone()
            values = (question, conclusion, str(root), _pack(vector) if vector else None,
                      self._model_name if vector else None, now)
            if existing is not None:
                memo_id = int(existing[0])
                self._db.execute(
                    "UPDATE exploration_memos SET question=?, conclusion=?, root=?, question_vector=?, "
                    "vector_model=?, updated_at=? WHERE id=?", (*values, memo_id))
                self._db.execute("DELETE FROM exploration_memo_files WHERE memo_id=?", (memo_id,))
            else:
                cursor = self._db.execute(
                    "INSERT INTO exploration_memos (project, question, question_hash, conclusion, root, "
                    "question_vector, vector_model, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (project, question, digest, conclusion, str(root), values[3], values[4], now, now))
                memo_id = int(cursor.lastrowid)
            for item in prints.values():
                self._db.execute(
                    "INSERT INTO exploration_memo_files (memo_id, path, sha256, size) VALUES (?, ?, ?, ?)",
                    (memo_id, item.path, item.sha256, item.size))
            self._db.commit()
        except sqlite3.Error:
            self._db.rollback()
            raise
        return {
            "recorded": True, "id": memo_id, "project": project, "root": str(root),
            "replaced": existing is not None, "semantic_index": vector is not None,
            "files": [{"path": item.path, "sha256": item.sha256, "size": item.size} for item in prints.values()],
        }

    def recall(self, arguments: RecallArguments) -> dict[str, Any]:
        root = resolve_root(arguments.root)
        project = arguments.project.strip()
        question = arguments.question.strip()
        digest = question_hash(question)
        rows = self._db.execute(
            "SELECT id, question, question_hash, conclusion, question_vector, vector_model, created_at, updated_at "
            "FROM exploration_memos WHERE project=? ORDER BY updated_at DESC, id DESC LIMIT ?",
            (project, CANDIDATE_SCAN_LIMIT)).fetchall()
        query_vector = self._vector(question) if rows else None
        scored: list[tuple[float, Any]] = []
        for row in rows:
            score = self._score(question, digest, query_vector, row)
            if score >= MIN_RECALL_SCORE:
                scored.append((score, row))
        scored.sort(key=lambda item: (-item[0], -int(item[1][0])))
        hashes: dict[str, FileState] = {}
        results: list[dict[str, Any]] = []
        for score, row in scored:
            entry = self._verify(root, row, hashes)
            if entry["status"] == STATUS_STALE and not arguments.include_stale:
                continue
            entry["score"] = round(score, 4)
            results.append(entry)
            if len(results) >= arguments.limit:
                break
        return {
            "question": question, "project": project, "root": str(root),
            "fresh": sum(1 for entry in results if entry["status"] == STATUS_FRESH),
            "stale": sum(1 for entry in results if entry["status"] == STATUS_STALE),
            "results": results,
        }

    def _score(self, question: str, digest: str, query_vector: list[float] | None, row: Any) -> float:
        if row[2] == digest:
            return EXACT_MATCH_SCORE
        lexical = float(self._lexical(question, row[1]))
        stored = _unpack(row[4]) if row[5] == self._model_name else None
        cosine = _cosine(query_vector, stored) if query_vector is not None and stored is not None else None
        if cosine is None:
            return lexical
        return SEMANTIC_WEIGHT * max(cosine, 0.0) + LEXICAL_WEIGHT * lexical

    def _current_hash(self, root: Path, relative: str, cache: dict[str, FileState]) -> FileState:
        """("ok", sha) or (missing | outside_root | too_large, None) for one stored path, cached per recall."""
        if relative in cache:
            return cache[relative]
        state: FileState = ("ok", None)
        try:
            resolved = (root / relative).resolve(strict=True)
        except (FileNotFoundError, RuntimeError):
            state = ("missing", None)
        else:
            if _inside(root, resolved) is None:
                state = ("outside_root", None)
            elif not resolved.is_file():
                state = ("missing", None)
            elif resolved.stat().st_size > MAX_FILE_BYTES:
                state = ("too_large", None)
            else:
                try:
                    state = ("ok", sha256_file(resolved))
                except OSError:
                    state = ("missing", None)
        cache[relative] = state
        return state

    def _verify(self, root: Path, row: Any, cache: dict[str, FileState]) -> dict[str, Any]:
        files = self._db.execute(
            "SELECT path, sha256, size FROM exploration_memo_files WHERE memo_id=? ORDER BY path",
            (int(row[0]),)).fetchall()
        changed: list[str] = []
        missing: list[str] = []
        unreadable: list[dict[str, str]] = []
        for path, recorded, _size in files:
            state, current = self._current_hash(root, path, cache)
            if state == "missing":
                missing.append(path)
            elif state != "ok":
                unreadable.append({"path": path, "reason": state})
            elif current != recorded:
                changed.append(path)
        fresh = not (changed or missing or unreadable)
        entry: dict[str, Any] = {
            "id": int(row[0]), "question": row[1], "conclusion": row[3],
            "status": STATUS_FRESH if fresh else STATUS_STALE,
            "files": [path for path, _sha, _size in files],
            "recorded_at": row[6], "updated_at": row[7],
        }
        if not fresh:
            entry["changed_files"] = changed
            entry["missing_files"] = missing
            if unreadable:
                entry["unverifiable_files"] = unreadable
        return entry


def handle(name: str, raw: dict[str, Any], db: Any, *, lexical: LexicalScorer, embed: Embedder | None = None,
           model_name: str | None = None) -> dict[str, Any]:
    """Validate the arguments of one exploration tool call and run it."""
    memos = ExplorationMemos(db, lexical=lexical, embed=embed, model_name=model_name)
    if name == RECORD_TOOL:
        return memos.record(RecordArguments.model_validate(raw))
    if name == RECALL_TOOL:
        return memos.recall(RecallArguments.model_validate(raw))
    raise ExplorationError(f"unknown exploration tool {name!r}")


__all__ = [
    "RECALL_DESCRIPTION", "RECALL_TOOL", "RECORD_DESCRIPTION", "RECORD_TOOL", "TOOL_NAMES",
    "ExplorationError", "ExplorationMemos", "RecallArguments", "RecordArguments", "handle",
    "recall_input_schema", "record_input_schema",
]
