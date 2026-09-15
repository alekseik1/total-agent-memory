from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from memory_core.evidence_pack import pack_evidence
from memory_core.retrieval import MemoryHit

CONTEXT_MAX_BYTES = 24000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contexts", type=Path, required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads(args.contexts.read_text())
    db = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    try:
        for row in rows:
            evidence: list[MemoryHit] = []
            for identity in row["hit_ids"]:
                source = db.execute(
                    "SELECT id,content,project FROM knowledge WHERE id=? AND status='active'",
                    (identity,),
                ).fetchone()
                if source is None:
                    raise ValueError(f"Missing source {identity}")
                evidence.append({"id": source["id"], "content": source["content"],
                                 "project": source["project"], "source_ref": f"knowledge:{identity}"})
            if len({hit["project"] for hit in evidence}) > 1:
                raise ValueError("Cross-project evidence")
            start = time.perf_counter()
            row["context"] = pack_evidence(
                evidence, query=row["question"], max_bytes=CONTEXT_MAX_BYTES,
            )
            row["packing_ms"] = (time.perf_counter() - start) * 1000
            if len(row["context"].encode("utf-8")) > CONTEXT_MAX_BYTES:
                raise ValueError("Context exceeds external runner cap")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(rows, ensure_ascii=False))
    finally:
        db.close()


if __name__ == "__main__":
    main()
