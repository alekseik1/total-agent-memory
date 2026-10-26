"""v10.5 — latency micro-benchmark for memory_save / memory_recall.

Hermetic enough to run in CI: spins a fresh tmp Store, exercises
the dispatcher the same way the MCP loop does, and reports
percentiles for save/recall in two configurations:

  • sync  — default v10.0 hot path
  • async — v10.1 inbox/outbox worker (MEMORY_ASYNC_ENRICHMENT=true)

Run:
    cd ~/claude-memory-server
    ./.venv/bin/python benchmarks/v10_5_latency.py [--rounds N] [--out path.json]

Skips the LLM-bound heavy stages by default (gate/contradiction off)
so numbers are repeatable; pass `--with-llm` if you want the realistic
slow-path numbers.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import uuid
from pathlib import Path
from statistics import quantiles

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def _percentiles(samples: list[float]) -> dict:
    if not samples:
        return {"min": None, "p50": None, "p95": None, "p99": None, "max": None, "mean": None}
    q = quantiles(samples, n=100, method="inclusive")
    return {
        "min":  round(min(samples), 2),
        "p50":  round(q[49], 2),
        "p95":  round(q[94], 2),
        "p99":  round(q[98], 2),
        "max":  round(max(samples), 2),
        "mean": round(sum(samples) / len(samples), 2),
    }


CORPUS = [
    ("PostgreSQL 18 настроен с pgvector в claude-memory-server, конфиг docker-compose.yml:42.", "fact"),
    ("Решено: LISTEN/NOTIFY вместо Redis Streams для очереди событий.", "decision"),
    ("Фикс _binary_search server.py:697 — np.argpartition требует kth STRICTLY < N.", "solution"),
    ("Convention: SQL миграции 0NN_short_name.sql, Store применяет идемпотентно.", "convention"),
    ("Outbox journal через write_intents с sha1 hash для дедупликации.", "fact"),
    ("Quality gate threshold 0.5, fail-open при ошибке LLM.", "convention"),
    ("Episodic events идут в graph_nodes type=event, mentioned_in edges на entities.", "fact"),
    ("Async enrichment worker дренирует enrichment_queue каждые 100ms.", "fact"),
    ("Soft-drop semantic: запись помечается quality_dropped после INSERT в async.", "convention"),
    ("Stale-processing recovery: rows зависшие в processing > 60s flip back to pending.", "convention"),
]

RECALL_QUERIES = [
    "PostgreSQL конфигурация",
    "очередь событий",
    "binary_search фикс",
    "SQL миграция",
    "async enrichment worker",
    "quality gate",
    "episodic events",
]


async def _run_bench(async_mode: bool, rounds: int, with_llm: bool) -> dict:
    label = "async" if async_mode else "sync"
    os.environ["MEMORY_ASYNC_ENRICHMENT"] = "true" if async_mode else "false"
    os.environ["MEMORY_QUALITY_GATE_ENABLED"] = "auto" if with_llm else "false"
    os.environ["MEMORY_CONTRADICTION_DETECT_ENABLED"] = "auto" if with_llm else "false"
    os.environ["MEMORY_OUTBOX_ENABLED"] = "true"
    os.environ["MEMORY_EPISODIC_ENABLED"] = "true"

    # Fresh import to pick up env-time-resolved MEMORY_DIR.
    if "server" in sys.modules:
        for mod_name in list(sys.modules):
            if mod_name == "server" or mod_name.startswith("server."):
                del sys.modules[mod_name]
    import server as srv  # noqa

    # Tmp store so we don't touch the user DB.
    tmp_dir = Path(f"/tmp/v10_5_bench_{label}_{uuid.uuid4().hex[:6]}")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    srv.MEMORY_DIR = tmp_dir
    srv.store = srv.Store()
    srv.recall = srv.Recall(srv.store)
    srv.SID = f"bench_{label}"
    srv.BRANCH = "bench"

    # Warmup — first call hits cold caches (FastEmbed model load, etc.)
    await srv._do("memory_save", {
        "type": "fact", "content": "warmup record about claude memory v10.5",
        "project": "bench-warmup", "tags": ["warmup"],
    })

    save_samples_ms: list[float] = []
    recall_samples_ms: list[float] = []

    for r in range(rounds):
        for content, ktype in CORPUS:
            t0 = time.perf_counter()
            await srv._do("memory_save", {
                "type": ktype, "content": content,
                "project": f"bench-{label}", "tags": [ktype, "v10.5"],
            })
            save_samples_ms.append((time.perf_counter() - t0) * 1000)

        # Let the async worker drain a batch before recall (only async needs this)
        if async_mode:
            await asyncio.sleep(0.5)

        for q in RECALL_QUERIES:
            t0 = time.perf_counter()
            await srv._do("memory_recall", {
                "query": q, "project": f"bench-{label}",
                "limit": 5, "detail": "compact",
            })
            recall_samples_ms.append((time.perf_counter() - t0) * 1000)

    # Drain async worker fully so the test doesn't leave pending rows.
    if async_mode:
        await asyncio.sleep(2.0)
        try:
            import enrichment_worker as ew
            ew.run_pending(srv.store.db, store=srv.store, max_rows=999)
        except Exception:
            pass

    return {
        "label": label,
        "with_llm": with_llm,
        "rounds": rounds,
        "save_count": len(save_samples_ms),
        "recall_count": len(recall_samples_ms),
        "save_ms": _percentiles(save_samples_ms),
        "recall_ms": _percentiles(recall_samples_ms),
    }


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=2,
                        help="Repetitions of the corpus (default 2 → ~20 saves, ~14 recalls per config)")
    parser.add_argument("--with-llm", action="store_true",
                        help="Keep LLM-bound stages (quality_gate / contradiction) ON for realistic numbers")
    parser.add_argument("--out", default=str(ROOT / "benchmarks" / "results" / "v10_5_latency.json"),
                        help="Where to write the JSON report")
    args = parser.parse_args()

    print(f"=== v10.5 latency bench (rounds={args.rounds}, with_llm={args.with_llm}) ===\n")

    results: list[dict] = []
    for async_mode in (False, True):
        out = await _run_bench(async_mode=async_mode, rounds=args.rounds, with_llm=args.with_llm)
        results.append(out)
        s = out["save_ms"]
        r = out["recall_ms"]
        print(f"--- {out['label']} ---")
        print(f"  save n={out['save_count']:<3} "
              f"min={s['min']:>5}  p50={s['p50']:>5}  p95={s['p95']:>5}  p99={s['p99']:>5}  max={s['max']:>6}  mean={s['mean']:>5}")
        print(f"  recall n={out['recall_count']:<3} "
              f"min={r['min']:>5}  p50={r['p50']:>5}  p95={r['p95']:>5}  p99={r['p99']:>5}  max={r['max']:>6}  mean={r['mean']:>5}")
        print()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "version": "10.5.0",
        "host": os.uname().sysname + "-" + os.uname().machine,
        "with_llm": args.with_llm,
        "rounds": args.rounds,
        "configs": results,
    }, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
