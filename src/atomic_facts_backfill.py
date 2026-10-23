from __future__ import annotations

import argparse
import json
import logging
import sqlite3
from contextlib import closing
from pathlib import Path

import config
from ai_layer.atomic_fact_extractor import FactExtractor
from llm_provider import make_provider
from memory_core.atomic_facts import FactRepository, InvalidExtraction
from memory_core.fts_maintenance import maintain_fts

log = logging.getLogger(__name__)


# One record's failure must not stop the sweep: a too-long source, a provider
# timeout or a malformed completion is logged and the record stays pending for
# the next run. Anything else (a broken database, a missing table) still raises.
SKIPPED_ERRORS = (InvalidExtraction, OSError, ValueError, json.JSONDecodeError)


def backfill(database: Path, project: str, limit: int, *, optimize_fts: bool = False,
             extractor_factory=None) -> tuple[int, int, int]:
    """Extract atomic facts for the project's pending records.

    Returns (records tried, facts created, records skipped because of an error).
    """
    if not database.is_file():
        raise ValueError("Database must already exist and have current migrations")
    with closing(sqlite3.connect(database, timeout=30)) as db:
        db.row_factory = sqlite3.Row
        repository = FactRepository(db)
        if extractor_factory is None:
            extractor = FactExtractor(repository, make_provider(config.get_phase_provider("enrich")),
                                      config.get_phase_model("enrich"))
        else:
            extractor = extractor_factory(repository)
        ids = repository.pending(project, limit)
        created = 0
        failed = 0
        for identity in ids:
            try:
                created += extractor.extract(identity)
            except SKIPPED_ERRORS as error:
                failed += 1
                db.rollback()
                log.warning("atomic fact extraction skipped", extra={"knowledge_id": identity, "error": str(error)})
                continue
            db.commit()
        if optimize_fts:
            maintain_fts(db, optimize=True)
            db.commit()
        return len(ids), created, failed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--optimize-fts", action="store_true")
    args = parser.parse_args()
    processed, created, failed = backfill(args.database, args.project, args.limit, optimize_fts=args.optimize_fts)
    print(json.dumps({"processed": processed, "facts_created": created, "skipped": failed}))


if __name__ == "__main__":
    main()
