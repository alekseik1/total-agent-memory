"""`tam import <kind> <path>`: import an export file or a Markdown tree into the local store."""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from pathlib import Path

from importers import KINDS, SOURCES, ChatOptions, ImportFormatError, ImportStats
from importers.service import ImportService

LOGGER = logging.getLogger(__name__)
EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_USAGE = 2
DEFAULT_PROJECT = "general"
READ_TIMEOUT_SECONDS = 10


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tam import",
        description="Import chat history or notes into memory. Re-running an import skips what is already stored.",
    )
    p.add_argument("kind", choices=KINDS, help="chatgpt / claude: conversations.json of the data export; "
                                               "mem0: JSON from get_all(); markdown: a directory of notes")
    p.add_argument("path", type=Path, help="export file or Markdown directory")
    p.add_argument("--project", default=DEFAULT_PROJECT, help=f"project to store into (default: {DEFAULT_PROJECT})")
    p.add_argument("--dry-run", action="store_true", help="parse and count, store nothing")
    p.add_argument("--include-assistant", action="store_true",
                   help="chatgpt / claude: also import assistant replies (default: user turns only)")
    p.add_argument("--memory-dir", type=Path, help="memory directory (default: TAM_MEMORY_DIR or ~/.tam)")
    p.add_argument("--quiet", action="store_true", help="no progress lines on stderr")
    return p


def _progress_printer(quiet: bool):
    if quiet:
        return None

    def report(stats: ImportStats) -> None:
        done = stats.would_import if stats.dry_run else stats.imported
        verb = "would import" if stats.dry_run else "imported"
        sys.stderr.write(
            f"[tam import] {stats.units_seen} seen: {done} {verb}, {stats.deduplicated} duplicates, "
            f"{stats.already_imported} already imported, "
            f"{stats.skipped_role + stats.skipped_empty + stats.skipped_short} skipped\n"
        )

    return report


def _dry_run_ledger(root: Path) -> sqlite3.Connection | None:
    database = root / "memory.db"
    if not database.is_file():
        return None
    return sqlite3.connect(database.as_uri() + "?mode=ro", uri=True, timeout=READ_TIMEOUT_SECONDS)


def run(kind: str, path: Path, *, project: str, dry_run: bool, include_assistant: bool, memory_dir: Path,
        quiet: bool = True) -> ImportStats:
    """Import ``path`` with the ``kind`` reader; raises ImportFormatError on malformed input."""
    if kind not in SOURCES:
        raise ValueError(f"unknown import kind {kind!r}; choose one of {', '.join(KINDS)}")
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")
    if kind != "markdown" and not path.is_file():
        raise ImportFormatError(f"{kind} import expects a JSON file, got {path}")
    if not project.strip():
        raise ValueError("project must be non-empty")
    progress = _progress_printer(quiet)
    if dry_run:
        connection = _dry_run_ledger(memory_dir)
        try:
            service = ImportService(kind=kind, project=project, store=None, ledger_db=connection, dry_run=True,
                                    progress=progress)
            stats = service.run(SOURCES[kind](path, ChatOptions(include_assistant), service.stats))
        finally:
            if connection is not None:
                connection.close()
        return stats
    from cli_store import open_store

    with open_store(memory_dir) as store:
        service = ImportService(kind=kind, project=project, store=store, ledger_db=store.db, dry_run=False,
                                progress=progress)
        return service.run(SOURCES[kind](path, ChatOptions(include_assistant), service.stats))


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    from cli_store import pin_memory_dir

    root = pin_memory_dir(args.memory_dir)
    try:
        stats = run(args.kind, args.path.expanduser(), project=args.project, dry_run=args.dry_run,
                    include_assistant=args.include_assistant, memory_dir=root, quiet=args.quiet)
    except (ImportFormatError, FileNotFoundError, ValueError) as error:
        sys.stderr.write(f"tam import: {error}\n")
        return EXIT_USAGE
    except RecursionError:
        sys.stderr.write("tam import: input is nested too deeply\n")
        return EXIT_USAGE
    except (OSError, sqlite3.Error) as error:
        LOGGER.error(json.dumps({"event": "import_failed", "kind": args.kind, "error": str(error)}))
        sys.stderr.write(f"tam import: failed: {error}\n")
        return EXIT_FAILURE
    print(json.dumps(stats.as_dict(), indent=2, ensure_ascii=False))
    return EXIT_OK

