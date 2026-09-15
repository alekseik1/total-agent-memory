from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@dataclass(frozen=True)
class QualityCase:
    id: str
    query: str
    project: str
    category: str
    evidence: tuple[tuple[int, ...], ...]
    reference: str
    answerable: bool
    branch: str = ""


@dataclass(frozen=True)
class CaseResult:
    id: str
    category: str
    hit_ids: tuple[int, ...]
    complete_evidence: bool | None
    scope_leaks: int
    retrieval_ms: float
    sql_statements: int


def load_cases(data: dict) -> list[QualityCase]:
    records = {row["id"]: row for row in data["records"]}
    if len(records) != len(data["records"]):
        raise ValueError("Duplicate record IDs")
    cases = []
    for item in data["cases"]:
        case = QualityCase(
            **{**item, "evidence": tuple(tuple(group) for group in item["evidence"])}
        )
        if (
            type(case.answerable) is not bool
            or not case.id
            or not case.query
            or not case.project
        ):
            raise ValueError("Invalid case contract")
        if case.answerable and not case.evidence:
            raise ValueError("Answerable cases require evidence alternatives")
        for group in case.evidence:
            if not group or any(identity not in records for identity in group):
                raise ValueError("Evidence refers to missing records")
            if any(records[identity]["project"] != case.project for identity in group):
                raise ValueError("Gold evidence crosses project boundary")
        cases.append(case)
    if not cases or len({case.id for case in cases}) != len(cases):
        raise ValueError("Empty dataset or duplicate case IDs")
    return cases


def judge_audit(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate answer IDs")
    equivalent: dict[tuple[str, str, str], set[bool]] = defaultdict(set)
    for row in rows:
        if row.get("error") or row.get("prediction") is None:
            continue
        normalize = lambda text: " ".join(text.casefold().split()).rstrip(".")
        key = (
            normalize(row["question"]),
            normalize(row["gold"]),
            normalize(row["prediction"]),
        )
        if type(row.get("correct")) is not bool:
            raise ValueError("Missing boolean judge verdict")
        equivalent[key].add(row["correct"])
    return {
        "rows": len(rows),
        "api_errors": sum(bool(row.get("error")) for row in rows),
        "identical_answer_conflicts": sum(
            len(verdicts) > 1 for verdicts in equivalent.values()
        ),
        "correct": sum(
            row.get("correct") is True and not row.get("error") for row in rows
        ),
    }


def evaluate(dataset: Path, *, top_k: int = 5) -> dict:
    if not 1 <= top_k <= 50:
        raise ValueError("top_k must be between 1 and 50")
    data = json.loads(dataset.read_text())
    cases = load_cases(data)
    import server
    from memory_core.evidence_chains import EvidenceChains
    from memory_core.evidence_window import EvidenceWindow
    from memory_core.retrieval import SearchScope, flatten_results

    results: list[CaseResult] = []
    with tempfile.TemporaryDirectory(prefix="tam-quality-") as directory:
        original_dir = server.MEMORY_DIR
        server.MEMORY_DIR = Path(directory)
        store = None
        try:
            store = server.Store()
            recall = server.Recall(store)
            for row in data["records"]:
                store.db.execute(
                    "INSERT INTO knowledge(id,content,project,session_id,created_at,last_confirmed,type,status,branch) "
                    "VALUES (?,?,?,?,?,?,?,?,?)",
                    (
                        row["id"],
                        row["content"],
                        row["project"],
                        row.get("session_id", ""),
                        row["created_at"],
                        row["created_at"],
                        "fact",
                        row.get("status", "active"),
                        row.get("branch", ""),
                    ),
                )
            store.db.commit()
            for case in cases:
                statements = []
                store.db.set_trace_callback(statements.append)
                started = time.perf_counter()
                scope = SearchScope(project=case.project, branch=case.branch)
                hits = flatten_results(
                    recall.search(
                        case.query,
                        project=case.project,
                        limit=top_k,
                        detail="full",
                        branch=case.branch,
                        _explain=True,
                    )
                )
                hits = EvidenceWindow(store.db).expand(hits, scope=scope)
                hits = EvidenceChains(store.db).expand(hits, scope)
                elapsed = (time.perf_counter() - started) * 1000
                store.db.set_trace_callback(None)
                ids = {hit["id"] for hit in hits}
                complete = (
                    any(set(group) <= ids for group in case.evidence)
                    if case.answerable
                    else None
                )
                leaks = sum(not scope.allows(hit, store.db) for hit in hits)
                results.append(
                    CaseResult(
                        case.id,
                        case.category,
                        tuple(hit["id"] for hit in hits),
                        complete,
                        leaks,
                        elapsed,
                        len(statements),
                    )
                )
        finally:
            if store is not None:
                if store._enrich_worker:
                    store._enrich_worker.stop()
                    store._enrich_worker.join()
                store.db.close()
            server.MEMORY_DIR = original_dir
    groups: dict[str, list[CaseResult]] = defaultdict(list)
    for result in results:
        groups[result.category].append(result)
    return {
        "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "provenance": data.get("provenance", "unspecified"),
        "top_k": top_k,
        "mode": "lexical_context_no_ingestion_embeddings_no_llm",
        "cases": [asdict(result) for result in results],
        "categories": {
            name: {
                "n": len(rows),
                "answerable": sum(r.complete_evidence is not None for r in rows),
                "complete_evidence": sum(r.complete_evidence is True for r in rows),
                "scope_leaks": sum(r.scope_leaks for r in rows),
                "p50_ms": statistics.median(r.retrieval_ms for r in rows),
            }
            for name, rows in groups.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--answers", type=Path)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="tam-quality-runtime-") as directory:
        os.environ["TAM_MEMORY_DIR"] = directory
        os.environ["MEMORY_ASYNC_ENRICHMENT"] = "false"
        os.environ["MEMORY_MODE"] = "fast"
        result = evaluate(args.dataset, top_k=args.top_k)
        if args.answers:
            result["judge_audit"] = judge_audit(args.answers)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
