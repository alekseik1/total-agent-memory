"""``tam forget-auto``: hard-delete the records the hooks saved on their own.

Auto-saved records are ``source='auto'`` knowledge rows tagged ``turn-autosave``
(Stop hook, Codex notify) or ``session-autosave`` (SessionEnd). They are erased
through :func:`erasure.erase_many` — vectors, FTS, raw-log copies and action-log
quotes included — not merely hidden. ``--include-actions`` also clears the
agent action log. ``--dry-run`` only counts.

Exit codes: 0 done (or nothing to do), 1 the store could not be changed, 2 usage.
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from collections.abc import Sequence
from contextlib import closing
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

from ambient.actions import ActionLog, table_exists
from ambient.turn_end import AUTO_SOURCE, AUTO_TAGS

LOGGER = logging.getLogger(__name__)

EXIT_OK: Final = 0
EXIT_FAILURE: Final = 1
EXIT_USAGE: Final = 2
DB_NAME: Final = "memory.db"
SQLITE_TIMEOUT_SEC: Final = 30.0


class ForgetError(RuntimeError):
    """The store is missing or could not be changed."""


@dataclass(frozen=True)
class ForgetReport:
    memory_dir: str
    project: str | None
    dry_run: bool
    records: int
    erased_ids: tuple[int, ...]
    actions: int

    def as_dict(self) -> dict[str, object]:
        return {**asdict(self), "erased_ids": list(self.erased_ids)}


def auto_record_ids(db: sqlite3.Connection, project: str | None = None) -> list[int]:
    """Ids of auto-saved records (any status), optionally within one project."""
    tag_match = " OR ".join("instr(tags, ?) > 0" for _ in AUTO_TAGS)
    params: list[object] = [AUTO_SOURCE, *(json.dumps(tag) for tag in AUTO_TAGS)]
    sql = f"SELECT id FROM knowledge WHERE source = ? AND ({tag_match})"
    if project:
        sql += " AND project = ?"
        params.append(project)
    return [int(row[0]) for row in db.execute(sql + " ORDER BY id", params).fetchall()]


def preview(memory_dir: Path, project: str | None, include_actions: bool) -> ForgetReport:
    db_path = memory_dir / DB_NAME
    if not db_path.is_file():
        raise ForgetError(f"no memory store at {db_path}")
    with closing(sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=SQLITE_TIMEOUT_SEC)) as db:
        ids = auto_record_ids(db, project)
        actions = ActionLog(db).count(project) if include_actions and table_exists(db, "agent_actions") else 0
    return ForgetReport(str(memory_dir), project, True, len(ids), tuple(ids), actions)


def forget(memory_dir: Path, project: str | None, include_actions: bool) -> ForgetReport:
    """Erase the auto-saved records (and the action log when asked); opens the full store."""
    if not (memory_dir / DB_NAME).is_file():
        raise ForgetError(f"no memory store at {memory_dir / DB_NAME}")
    import erasure
    from cli_store import open_store

    with open_store(memory_dir) as store:
        ids = auto_record_ids(store.db, project)
        result = erasure.erase_many(store, ids, memory_dir / "raw", memory_dir / "chroma") if ids else None
        actions = 0
        if include_actions and table_exists(store.db, "agent_actions"):
            actions = ActionLog(store.db).delete(project)
        if result is not None and getattr(store, "cache", None) is not None:
            for name in result["projects"]:
                store.cache.invalidate(project=name)
    erased = tuple(result["erased"]) if result else ()
    LOGGER.info(json.dumps({"event": "forget_auto", "project": project, "records": len(erased),
                            "actions": actions}))
    return ForgetReport(str(memory_dir), project, False, len(erased), erased, actions)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tam forget-auto",
        description="Hard-delete the memory records TAM hooks saved automatically "
                    "(turn and session summaries).")
    parser.add_argument("--memory-dir", type=Path, default=None, help="memory directory (default: TAM_MEMORY_DIR)")
    parser.add_argument("--project", default=None, help="only this project's auto-saved records")
    parser.add_argument("--include-actions", action="store_true",
                        help="also delete the agent action log (of --project, or all of it)")
    parser.add_argument("--dry-run", action="store_true", help="count what would be deleted, change nothing")
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
    except SystemExit as exc:
        return EXIT_OK if exc.code in (0, None) else EXIT_USAGE
    from cli_store import pin_memory_dir

    memory_dir = pin_memory_dir(args.memory_dir)
    try:
        report = (preview if args.dry_run else forget)(memory_dir, args.project, args.include_actions)
    except (ForgetError, sqlite3.Error, OSError) as exc:
        sys.stderr.write(f"tam forget-auto: {exc}\n")
        return EXIT_FAILURE
    if args.json:
        sys.stdout.write(json.dumps(report.as_dict()) + "\n")
        return EXIT_OK
    verb = "would delete" if report.dry_run else "deleted"
    scope = f" in project {report.project}" if report.project else ""
    actions = f" and {report.actions} action log row(s)" if args.include_actions else ""
    sys.stdout.write(f"tam forget-auto: {verb} {report.records} auto-saved record(s){scope}{actions}.\n")
    return EXIT_OK
