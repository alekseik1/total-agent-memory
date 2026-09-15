import argparse
import hashlib
import importlib.util
import json
import sqlite3
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from memory_core.atomic_facts import FactRepository
from memory_core.evidence_window import EvidenceWindow
from memory_core.retrieval import SearchScope


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


parser = argparse.ArgumentParser()
parser.add_argument("--baseline", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
old_a = load("old_atomic", args.baseline / "atomic_facts.py")
old_w = load("old_window", args.baseline / "evidence_window.py")
db = sqlite3.connect(":memory:")
db.row_factory = sqlite3.Row
db.executescript(
    "CREATE TABLE knowledge(id INTEGER PRIMARY KEY,content TEXT,project TEXT,session_id TEXT,branch TEXT DEFAULT '',status TEXT DEFAULT 'active',type TEXT DEFAULT 'fact',created_at TEXT,tags TEXT DEFAULT '[]'); CREATE TABLE embeddings(knowledge_id INTEGER, embedding_space TEXT); CREATE INDEX idx_neighbor ON knowledge(project,session_id,status,created_at,id);"
)
db.executescript(
    (
        Path(__file__).resolve().parents[1] / "migrations/029_atomic_facts.sql"
    ).read_text()
)
for i in range(1, 2001):
    db.execute(
        "INSERT INTO knowledge(id,content,project,session_id,created_at) VALUES (?,?,?,?,?)",
        (
            i,
            f"Owner{i % 20} maintains database topic{i % 50}.",
            "p",
            f"s{i // 40}",
            "2026-01-01",
        ),
    )
    db.execute(
        "INSERT INTO atomic_facts(knowledge_id,subject,predicate,object,content) VALUES (?,?,?,?,?)",
        (
            i,
            f"Owner{i % 20}",
            "maintains",
            f"topic{i % 50}",
            f"Owner{i % 20} maintains database topic{i % 50}",
        ),
    )
    db.execute(
        "INSERT INTO atomic_fact_sources VALUES (?,?,?)", (i, i, f"Owner{i % 20}")
    )
db.commit()
scope = SearchScope(project="p")
anchors = [{"id": i} for i in range(50, 550, 50)]


def probe(call):
    call()
    queries = []
    db.set_trace_callback(queries.append)
    call()
    db.set_trace_callback(None)
    times = []
    for _ in range(200):
        t = time.perf_counter()
        call()
        times.append((time.perf_counter() - t) * 1000)
    times.sort()
    return {
        "p50_ms": statistics.median(times),
        "p95_ms": times[int(len(times) * 0.95) - 1],
        "p99_ms": times[int(len(times) * 0.99) - 1],
        "sql_statements": sum(s.startswith(("SELECT", "WITH")) for s in queries),
    }


calls = {
    "atomic_before": lambda: old_a.FactRepository(db).search("Owner4 database", scope),
    "atomic_after": lambda: FactRepository(db).search("Owner4 database", scope),
    "window_before": lambda: old_w.EvidenceWindow(db).expand(anchors, scope=scope),
    "window_after": lambda: EvidenceWindow(db).expand(anchors, scope=scope),
}
assert calls["atomic_before"]() == calls["atomic_after"]()
assert calls["window_before"]() == calls["window_after"]()
report = {name: probe(call) for name, call in calls.items()}
report["fixture"] = {
    "records": 2000,
    "anchors": 10,
    "iterations": 200,
    "same_result_assertions": True,
    "scope": "synthetic warm SQLite operations, not end-to-end SLA",
}
report["baseline_sha256"] = {
    p.name: hashlib.sha256(p.read_bytes()).hexdigest()
    for p in args.baseline.glob("*.py")
}
report["fts_plans"] = {
    order: [
        list(row)
        for row in db.execute(
            "EXPLAIN QUERY PLAN SELECT rowid FROM atomic_facts_fts WHERE atomic_facts_fts MATCH ? ORDER BY "
            + order
            + " LIMIT 30",
            ("database",),
        )
    ]
    for order in ("rank", "bm25(atomic_facts_fts)")
}


def fts_order(order):
    return db.execute(
        "SELECT f.id FROM atomic_facts_fts JOIN atomic_facts f ON f.id=atomic_facts_fts.rowid "
        "JOIN knowledge k ON k.id=f.knowledge_id WHERE atomic_facts_fts MATCH ? "
        "AND k.status='active' AND k.project='p' ORDER BY " + order + " LIMIT 30",
        ('"owner4" OR "database"',),
    ).fetchall()


assert fts_order("rank") == fts_order("bm25(atomic_facts_fts)")
report["fts_order_latency"] = {
    order: probe(lambda order=order: fts_order(order)) for order in ("rank", "bm25(atomic_facts_fts)")
}
args.output.write_text(json.dumps(report, indent=2) + "\n")
