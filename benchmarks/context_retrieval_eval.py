from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import tempfile
import time
from pathlib import Path
from typing import TypedDict


class ContextRow(TypedDict, total=False):
    id: str
    conversation: str | int
    question: str
    gold: str
    category: str | int
    context: str
    hit_ids: list[int]
    anchor_ids: list[int]
    retrieval_ms: float
    context_ms: float


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--kind", choices=("locomo", "longmemeval"), required=True)
    parser.add_argument("--radius", type=int, choices=range(4), required=True)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--questions", type=Path)
    args = parser.parse_args()
    if args.kind == "longmemeval" and args.questions is None:
        parser.error("LongMemEval requires --questions to preserve the original retrieval query")
    queries = (
        {row["question_id"]: row["question"] for row in json.loads(args.questions.read_text())}
        if args.questions else None
    )
    rows: list[ContextRow] = json.loads(args.input.read_text())
    with tempfile.TemporaryDirectory(prefix="tam-context-eval-") as temporary:
        os.environ["TAM_MEMORY_DIR"] = temporary
        source = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
        target = sqlite3.connect(Path(temporary) / "memory.db")
        try:
            source.backup(target)
        finally:
            target.close()
            source.close()
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
        import server
        from config import get_recall_excluded_tags
        from memory_core.evidence_context import EvidenceContext
        from memory_core.evidence_pack import pack_evidence
        from memory_core.retrieval import SearchScope, flatten_results

        store = server.Store()
        recall = server.Recall(store)
        builder = EvidenceContext(store.db, get_recall_excluded_tags())
        try:
            for row in rows:
                prefix = "locomo" if args.kind == "locomo" else "lme"
                project = f"{prefix}_{row['conversation']}"
                if store.cache is not None:
                    store.cache.invalidate()
                if getattr(store, "v9_cache", None) is not None:
                    store.v9_cache.invalidate_all()
                started = time.perf_counter()
                search_query = queries[row["id"]] if queries is not None else row["question"]
                hits = flatten_results(recall.search(
                    search_query, project=project, limit=args.limit,
                    detail="full", record_usage=False,
                ))
                row["retrieval_ms"] = (time.perf_counter() - started) * 1000
                row["anchor_ids"] = [hit["id"] for hit in hits]
                started = time.perf_counter()
                evidence = builder.build(
                    hits, query=row["question"], scope=SearchScope(project=project),
                    radius=args.radius, max_bytes=24000,
                )
                if any(hit["project"] != project for hit in evidence):
                    raise ValueError("Cross-project evidence")
                row["context"] = pack_evidence(evidence, query=row["question"], max_bytes=24000)
                row["context_ms"] = (time.perf_counter() - started) * 1000
                row["hit_ids"] = [hit["id"] for hit in evidence]
                if len(row["context"].encode("utf-8")) > 24000:
                    raise ValueError("Context exceeds byte budget")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(rows, ensure_ascii=False))
        finally:
            store.db.close()


if __name__ == "__main__":
    main()
