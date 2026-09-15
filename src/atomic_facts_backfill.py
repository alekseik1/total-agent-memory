from __future__ import annotations

import argparse
import json
import sqlite3
from contextlib import closing
from pathlib import Path

import config
from ai_layer.atomic_fact_extractor import FactExtractor
from llm_provider import make_provider
from memory_core.atomic_facts import FactRepository
from memory_core.fts_maintenance import maintain_fts


def backfill(database: Path, project: str, limit: int, *, optimize_fts: bool = False) -> tuple[int, int]:
    if not database.is_file():
        raise ValueError("Database must already exist and have current migrations")
    with closing(sqlite3.connect(database, timeout=30)) as db:
        db.row_factory = sqlite3.Row
        repository = FactRepository(db)
        extractor = FactExtractor(repository, make_provider(config.get_phase_provider("enrich")),
                                  config.get_phase_model("enrich"))
        ids = repository.pending(project, limit)
        created = 0
        for identity in ids:
            created += extractor.extract(identity)
            db.commit()
        if optimize_fts:
            maintain_fts(db, optimize=True)
            db.commit()
        return len(ids), created


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--optimize-fts", action="store_true")
    args = parser.parse_args()
    processed, created = backfill(args.database, args.project, args.limit, optimize_fts=args.optimize_fts)
    print(json.dumps({"processed": processed, "facts_created": created}))


if __name__ == "__main__":
    main()
