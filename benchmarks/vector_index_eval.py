from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

import faiss
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("locomo", "longmemeval"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--scale", type=int, default=0)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    faiss.omp_set_num_threads(1)
    rows = json.loads((root / f"docs/benchmarks/context-v14/{args.kind}-contexts.json").read_text())
    rows = [rows[i] for i in np.linspace(0, len(rows) - 1, args.count, dtype=int)]
    raw_questions = {}
    if args.kind == "longmemeval":
        raw_questions = {row["question_id"]: row["question"] for row in json.loads(
            (root / "benchmarks/data/longmemeval_s.json").read_text()
        )}
    with tempfile.TemporaryDirectory() as temporary:
        os.environ.update(TAM_MEMORY_DIR=temporary, CLAUDE_MEMORY_DIR=temporary, MEMORY_MODE="fast",
                          MEMORY_LLM_ENABLED="false", MEMORY_ASYNC_ENRICHMENT="false")
        sys.path.insert(0, str(root / "src"))
        import server
        from memory_core.vector_math import POPCOUNT

        store = server.Store()
        try:
            queries = np.asarray([
                store.embed([raw_questions.get(row["id"], row["question"])])[0] for row in rows
            ], dtype=np.float32)
            model = store._active_embed_model_name()
        finally:
            store.db.close()
    with sqlite3.connect(f"file:{root}/backups/standard-v14-20260911/{args.kind}.db?mode=ro", uri=True) as db:
        blobs = db.execute("SELECT float32_vector FROM embeddings e JOIN knowledge k ON k.id=e.knowledge_id "
                           "WHERE k.status='active' AND e.embed_dim=? AND e.embed_model=? ORDER BY e.knowledge_id",
                           (queries.shape[1], model)).fetchall()
    matrix = np.frombuffer(b"".join(row[0] for row in blobs), dtype=np.float32).reshape(-1, queries.shape[1]).copy()
    real_count = len(matrix)
    if args.scale:
        rng = np.random.default_rng(42)
        matrix = rng.standard_normal(size=(args.scale, queries.shape[1]), dtype=np.float32)
    faiss.normalize_L2(matrix)
    faiss.normalize_L2(queries)
    binary = np.packbits(matrix > 0, axis=1)
    exact = faiss.IndexFlatIP(matrix.shape[1])
    exact.add(matrix)
    _, reference = exact.search(queries, 10)
    results = {}

    def measure(name, search):
        search(queries[0])
        times, recalls = [], []
        for query, expected in zip(queries, reference, strict=True):
            started = time.perf_counter()
            found = search(query)
            times.append((time.perf_counter() - started) * 1000)
            recalls.append(len(set(map(int, found)) & set(map(int, expected))) / len(expected))
        results[name] = {"p50_ms": float(np.percentile(times, 50)), "p95_ms": float(np.percentile(times, 95)),
                         "mean_exact_top10_recall": float(np.mean(recalls)), "worst_recall": min(recalls)}
        sys.stderr.write(json.dumps({"completed": name, **results[name]}) + "\n")

    measure("exact_faiss", lambda q: exact.search(q.reshape(1, -1), 10)[1][0])
    measure("exact_numpy", lambda q: np.argsort(-np.einsum("ij,j->i", matrix, q), kind="stable")[:10])
    for candidates in (50, 150, 300):
        def binary_search(q, candidates=candidates):
            distances = POPCOUNT[np.bitwise_xor(binary, np.packbits(q > 0))].sum(axis=1)
            count = min(candidates, len(distances))
            boundary = np.partition(distances, count - 1)[count - 1]
            selected = np.flatnonzero(distances < boundary)
            ties = np.flatnonzero(distances == boundary)[:count - len(selected)]
            selected = np.concatenate((selected, ties))
            selected.sort()
            scores = np.einsum("ij,j->i", matrix[selected], q)
            return selected[np.argsort(-scores, kind="stable")[:10]]
        measure(f"binary_{candidates}", binary_search)
    report = {"kind": args.kind, "real_records": real_count, "measured_records": len(matrix),
              "synthetic": bool(args.scale), "dimension": matrix.shape[1], "queries": len(queries),
              "matrix_bytes": matrix.nbytes, "binary_bytes": binary.nbytes,
              "hnsw_build_ms": None, "hnsw_status": "building", "faiss_version": faiss.__version__,
              "threads": 1, "results": results,
              "scope": "unscoped vector algorithm comparison against exact top10; not source recall or QA; excludes DB and embedding latency"}
    args.output.write_text(json.dumps(report, indent=2))
    started = time.perf_counter()
    sys.stderr.write(json.dumps({"building_hnsw_records": len(matrix)}) + "\n")
    hnsw = faiss.IndexHNSWFlat(matrix.shape[1], 32, faiss.METRIC_INNER_PRODUCT)
    hnsw.hnsw.efConstruction = 100
    hnsw.add(matrix)
    build_ms = (time.perf_counter() - started) * 1000
    for effort in (64, 128):
        hnsw.hnsw.efSearch = effort
        measure(f"hnsw_{effort}", lambda q: hnsw.search(q.reshape(1, -1), 10)[1][0])
    report.update(hnsw_build_ms=build_ms, hnsw_status="complete")
    args.output.write_text(json.dumps(report, indent=2))
    sys.stdout.write(json.dumps(report) + "\n")


if __name__ == "__main__":
    main()
