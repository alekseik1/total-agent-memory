#!/usr/bin/env python3
"""Evidence recall of `memory_recall` on LoCoMo and LongMemEval stores, without an LLM.

For each question it records where the first gold evidence record lands in the
ranked search list (R@1/5/10/50), whether any gold evidence reaches the context
that `mode="context"` hands a reader (`ctx`), and whether all of it does
(`full`: every evidence turn of a multi-hop question). Gold evidence is the
LoCoMo `dia_id` tags and the LongMemEval answer session ids that the QA
harnesses store as record tags; abstention questions are skipped.

    python benchmarks/retrieval_eval.py --bench locomo --store DIR --split dev
    python benchmarks/retrieval_eval.py --bench lme --store DIR --split dev --limit 20 --neighbors 2

The store is copied first, so the run leaves it untouched.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCOMO = ROOT / "benchmarks" / "data" / "locomo" / "data" / "locomo10.json"
LONGMEMEVAL = ROOT / "benchmarks" / "data" / "longmemeval_s.json"
LOCOMO_CATEGORIES = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop"}
LONGMEMEVAL_DEV_SIZE = 100
RANKS = (1, 5, 10, 50)
SEARCH_LIMIT = 50


def questions(bench: str, split: str) -> list[dict]:
    if bench == "locomo":
        conversations = {"dev": range(3), "test": range(3, 10), "all": range(10)}[split]
        data = json.loads(LOCOMO.read_text())
        return [{"question": qa["question"], "project": f"locomo_{index}", "kind": LOCOMO_CATEGORIES[qa["category"]],
                 "gold": {e.lower() for e in qa.get("evidence", []) if e}}
                for index in conversations for qa in data[index]["qa"]
                if qa.get("category") in LOCOMO_CATEGORIES and "answer" in qa]
    data = json.loads(LONGMEMEVAL.read_text())
    ranked = sorted(data, key=lambda entry: hashlib.sha256(entry["question_id"].encode()).hexdigest())
    chosen = {"dev": ranked[:LONGMEMEVAL_DEV_SIZE], "test": ranked[LONGMEMEVAL_DEV_SIZE:], "all": ranked}[split]
    return [{"question": entry["question"], "project": f"lme_{entry['question_id']}", "kind": entry["question_type"],
             "gold": {sid.lower() for sid in entry["answer_session_ids"]}}
            for entry in chosen if not entry["question_id"].endswith("_abs")]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bench", choices=("locomo", "lme"), required=True)
    parser.add_argument("--store", type=Path, required=True, help="memory dir the QA harness ingested")
    parser.add_argument("--split", choices=("dev", "test", "all"), default="dev")
    parser.add_argument("--limit", type=int, default=50, help="context-mode limit, as in the QA run")
    parser.add_argument("--neighbors", type=int, default=1, help="context-mode neighbour radius, as in the QA run")
    parser.add_argument("--context-chars", type=int, default=60000)
    parser.add_argument("--out", type=Path, help="per-question rows (JSONL)")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="retrieval-eval-") as scratch:
        store = Path(scratch) / "store"
        shutil.copytree(args.store, store)
        os.environ["TAM_MEMORY_DIR"] = str(store)
        os.environ["CLAUDE_MEMORY_DIR"] = str(store)
        os.environ.setdefault("MEMORY_MODE", "fast")
        os.environ.setdefault("MEMORY_LLM_ENABLED", "false")
        os.environ.setdefault("MEMORY_CROSS_RERANK", "on")
        sys.path.insert(0, str(ROOT / "src"))
        import server as srv

        srv.store = srv.Store()
        srv.recall = srv.Recall(srv.store)
        srv.SID = "retrieval-eval"
        srv.BRANCH = ""

        def evidence_keys(ids: list[int]) -> list[set[str]]:
            if not ids:
                return []
            rows = srv.store.db.execute(
                f"SELECT id, tags FROM knowledge WHERE id IN ({','.join('?' * len(ids))})", ids).fetchall()
            # The store lower-cases tags on save.
            tags = {row[0]: {tag.lower() for tag in json.loads(row[1] or "[]")} for row in rows}
            return [tags.get(record_id, set()) for record_id in ids]

        def recall(arguments: dict) -> dict:
            raw = asyncio.run(srv._do("memory_recall", arguments))
            return json.loads(raw[0].text if isinstance(raw, list) else raw)

        totals: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
        rows = []
        started = time.perf_counter()
        for item in questions(args.bench, args.split):
            found = recall({"query": item["question"], "project": item["project"], "limit": SEARCH_LIMIT,
                            "detail": "compact"})
            groups = found.get("results", {})
            ranked = [hit["id"] for group in (groups.values() if isinstance(groups, dict) else [groups])
                      for hit in group]
            first = next((rank for rank, keys in enumerate(evidence_keys(ranked)) if keys & item["gold"]), None)
            context = recall({"query": item["question"], "project": item["project"], "limit": args.limit,
                              "mode": "context", "neighbors": args.neighbors,
                              "context_max_chars": args.context_chars, "detail": "full"})
            covered = set().union(*(keys & item["gold"]
                                    for keys in evidence_keys([hit["id"] for hit in context.get("results", [])])))
            for bucket in ("all", item["kind"]):
                for rank in RANKS:
                    totals[bucket][f"R@{rank}"].append(int(first is not None and first < rank))
                totals[bucket]["ctx"].append(int(bool(covered)))
                totals[bucket]["full"].append(int(covered >= item["gold"]))
            rows.append({"question": item["question"], "project": item["project"], "kind": item["kind"],
                         "first_rank": first, "in_context": bool(covered), "full": covered >= item["gold"]})
        srv.store.db.close()

    summary = {bucket: {"n": len(values["ctx"]),
                        **{name: round(100 * sum(v) / len(v), 1) for name, v in values.items()}}
               for bucket, values in sorted(totals.items())}
    summary["seconds"] = round(time.perf_counter() - started, 1)
    print(json.dumps(summary, indent=1))
    if args.out:
        args.out.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
