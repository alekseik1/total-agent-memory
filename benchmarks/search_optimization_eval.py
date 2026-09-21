from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--kind", choices=("locomo", "longmemeval"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--record-usage", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source_paths = [args.code_root / 'src' / name for name in (
        'server.py', 'memory_core/vector_search.py', 'memory_core/retrieval.py', 'memory_core/vector_math.py',
    )]
    source_hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths if path.exists()}
    inputs = json.loads((root / f"docs/benchmarks/context-v14/{args.kind}-contexts.json").read_text())
    indexes = np.linspace(0, len(inputs) - 1, min(args.count, len(inputs)), dtype=int)
    questions = [inputs[index] for index in indexes]
    raw_questions = {}
    if args.kind == "longmemeval":
        raw_questions = {r["question_id"]: r["question"] for r in json.loads((root / "benchmarks/data/longmemeval_s.json").read_text())}
    with tempfile.TemporaryDirectory(prefix="search-optimization-") as temporary:
        location = Path(temporary)
        with (
            sqlite3.connect(f"file:{root}/backups/standard-v14-20260911/{args.kind}.db?mode=ro", uri=True) as source,
            sqlite3.connect(location / "memory.db") as target,
        ):
            source.backup(target)
        os.environ.update(TAM_MEMORY_DIR=temporary, CLAUDE_MEMORY_DIR=temporary,
            MEMORY_MODE="fast", MEMORY_LLM_ENABLED="false", MEMORY_ASYNC_ENRICHMENT="false",
            MEMORY_OUTBOX_ENABLED="false", MEMORY_QUALITY_GATE_ENABLED="false")
        sys.path.insert(0, str(args.code_root.resolve() / "src"))
        import server
        from memory_core.retrieval import flatten_results
        from memory_core.telemetry import counters

        store = server.Store()
        recall = server.Recall(store)
        records = []
        try:
            for number, row in enumerate(questions):
                first = store.db.execute("SELECT project FROM knowledge WHERE id=?", (row["anchor_ids"][0],)).fetchone()
                project = first[0]
                question = raw_questions.get(row["id"], row["question"])
                def search(question=question, project=project):
                    if store.cache is not None:
                        store.cache.invalidate()
                    if getattr(store, "v9_cache", None) is not None:
                        store.v9_cache.invalidate_all()
                    return recall.search(question, project=project, limit=10 if args.kind == "locomo" else 5,
                                         detail="full", record_usage=args.record_usage)
                search()
                for temperature in ("cold", "warm"):
                    if temperature == "cold" and hasattr(store, "_vector_search"):
                        store._vector_search.clear()
                    before = counters.snapshot()
                    statements = []
                    store.db.set_trace_callback(statements.append)
                    started = time.perf_counter()
                    hits = flatten_results(search())
                    elapsed = (time.perf_counter() - started) * 1000
                    store.db.set_trace_callback(None)
                    after = counters.snapshot()
                    records.append({"id": row["id"], "temperature": temperature, "ms": elapsed,
                        "hit_ids": [hit["id"] for hit in hits],
                        "scores": [hit.get("rrf_score", hit.get("score")) for hit in hits],
                        "sql_count": len(statements),
                        "single_record_selects": sum("SELECT * FROM knowledge WHERE id=" in sql for sql in statements),
                        "vector_loads": sum("float32_vector" in sql or "binary_vector" in sql for sql in statements),
                        "sql_families": Counter(re.sub(r"'(?:''|[^'])*'|\b\d+\b", "?", sql)[:240] for sql in statements).most_common(10) if number == 0 else [],
                        "telemetry": {name: value - before.get(name, 0) for name, value in after.items()
                                      if value != before.get(name, 0) and "bucket" not in name}})
                if (number + 1) % 25 == 0:
                    sys.stderr.write(json.dumps({"completed": number + 1}) + "\n")
            report = {"kind": args.kind, "count": len(questions), "record_usage": args.record_usage,
                      "code_root": str(args.code_root), "source_hashes": source_hashes, "rows": records, "summary": {}}
            for path, digest in source_hashes.items():
                assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
            for temperature in ("cold", "warm"):
                selected = [row for row in records if row["temperature"] == temperature]
                report["summary"][temperature] = {
                    "p50_ms": float(np.percentile([row["ms"] for row in selected], 50)),
                    "p95_ms": float(np.percentile([row["ms"] for row in selected], 95)),
                    "median_sql": float(np.median([row["sql_count"] for row in selected])),
                    "single_record_selects": sum(row["single_record_selects"] for row in selected),
                    "vector_loads": sum(row["vector_loads"] for row in selected),
                }
            args.output.write_text(json.dumps(report, indent=2))
            sys.stdout.write(json.dumps(report["summary"]) + "\n")
        finally:
            store.db.close()


if __name__ == "__main__":
    main()
