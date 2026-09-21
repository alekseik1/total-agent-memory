"""Scale benchmark: how the SQLite core behaves at 10k / 100k / 1M records.

Loads a synthetic multi-tenant corpus through the real `Store.save_knowledge`
path (same tables, triggers, graph links and episodic rows as memory_save),
with embeddings precomputed in batches, then measures at each size:

  • recall latency through the MCP dispatcher (`_do("memory_recall")`),
    tenant-scoped and unscoped, and whether the planted fact is in the top 5;
  • recall right after a write (the vector pool is rebuilt on every write);
  • save latency through `_do("memory_save")` with a real embedding;
  • DB size and peak RSS.

Separate subcommands measure concurrency: many clients against one HTTP
server, and many writer processes against one DB file.

Run (from the repo root; the DB lives in --dir, never in ~/.tam):
    .venv/bin/python benchmarks/scale_bench.py embed   --dir D --total 1000000
    .venv/bin/python benchmarks/scale_bench.py load    --dir D --upto 10000
    .venv/bin/python benchmarks/scale_bench.py measure --dir D
    .venv/bin/python benchmarks/scale_bench.py http    --dir D --clients 1,4,16
    .venv/bin/python benchmarks/scale_bench.py writers --dir D --procs 1,2,4,8
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import resource
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from statistics import quantiles

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"

EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBED_DIM = 384
EMBED_BATCH = 256
EMBED_CHUNK = 20_000
LOAD_REPORT_EVERY = 10_000

TENANTS = 200
NEEDLES = 300
NEEDLE_SPAN = 10_000  # needles live among the oldest records
PROBE_SAVES = 50
TOP_K = 5

HTTP_PORT = 3799
HTTP_DURATION_S = 30
HTTP_SAVE_SHARE = 0.1
HTTP_READY_TIMEOUT_S = 180
WRITER_RECORDS = 400

FIRST_NAMES = [
    "Ольга", "Иван", "Мария", "Пётр", "Анна", "Сергей", "Елена", "Дмитрий", "Наталья", "Алексей",
    "Ирина", "Максим", "Татьяна", "Никита", "Светлана", "Артём", "Юлия", "Роман", "Ксения", "Павел",
    "Alice", "Bob", "Carol", "David", "Emma", "Frank", "Grace", "Henry", "Irene", "Jack",
]
SERVICES = [
    "billing", "auth", "gateway", "search", "notify", "ledger", "catalog", "checkout", "profile", "media",
    "reports", "scheduler", "inventory", "pricing", "shipping", "chat", "export", "import", "audit", "crm",
]
TECH = [
    "PostgreSQL 18", "Redis 7", "RabbitMQ 4", "Kafka", "ClickHouse", "Elasticsearch", "MinIO", "Nginx",
    "Go 1.25", "PHP 8.4", "Symfony 8", "Vue 3", "Nuxt 4", "Docker Compose", "Kubernetes", "Terraform",
    "gRPC", "GraphQL", "OpenTelemetry", "Prometheus", "Grafana", "Sentry", "Keycloak", "Temporal",
]
ACTIONS = [
    "перевели {svc} на {tech}, миграцию вёл {who}",
    "{who} настроил алерты {svc} в {tech}, порог p95 {num} мс",
    "решили хранить сессии {svc} в {tech}: меньше задержка, проще откат",
    "инцидент {num}: {svc} падал из-за пула соединений {tech}, исправил {who}",
    "{who} предпочитает ревью {svc} по вторникам, созвоны переносить нельзя",
    "для {svc} лимит запросов {num} в минуту на клиента, проверка в {tech}",
    "{svc} пишет события в {tech}, ретеншн {num} дней, ответственный {who}",
    "договорились: релизы {svc} только после прогона нагрузочного теста на {tech}",
    "{who} заметил утечку памяти в {svc} после обновления {tech}, откатили",
    "клиент попросил выгрузку {svc} в CSV раз в {num} часов через {tech}",
    "{svc} migrated to {tech}; {who} owns the rollout, error budget {num} min",
    "{who} prefers async standups for {svc}; decisions go to the {tech} wiki",
    "rate limit for {svc} raised to {num} rps after the {tech} upgrade",
    "postmortem {num}: {svc} timeouts traced to {tech} connection churn, fixed by {who}",
]
TAGS = ["infra", "decision", "incident", "preference", "release", "security", "billing", "perf",
        "migration", "oncall", "api", "data", "client", "team", "monitoring", "cost"]
TYPES = ["fact", "decision", "solution", "lesson", "convention"]

NEEDLE_FACTS = [
    ("Вебхуки сервиса {svc}-{n} слушают порт {port}, настраивала {who}",
     "какой порт у вебхуков {svc}-{n}"),
    ("Ключ шифрования бэкапов {svc}-{n} хранится в хранилище секретов под именем vault-{port}",
     "где хранится ключ шифрования бэкапов {svc}-{n}"),
    ("Окно обслуживания {svc}-{n}: по четвергам с {h}:00 до {h2}:00 по UTC, согласовал {who}",
     "когда окно обслуживания у {svc}-{n}"),
]


def tenant(i: int) -> str:
    return f"tenant-{i % TENANTS:03d}"


def needle_positions() -> dict[int, int]:
    step = NEEDLE_SPAN // NEEDLES
    return {k * step + step // 2: k for k in range(NEEDLES)}


NEEDLE_AT = needle_positions()


def needle(k: int) -> dict:
    rnd = random.Random(10_000 + k)
    tpl, q = NEEDLE_FACTS[k % len(NEEDLE_FACTS)]
    h = rnd.randint(0, 20)
    slots = {
        "svc": rnd.choice(SERVICES), "n": 9000 + k, "port": 40_000 + rnd.randint(0, 20_000),
        "who": rnd.choice(FIRST_NAMES), "h": h, "h2": h + 2,
    }
    return {"content": tpl.format(**slots), "query": q.format(**slots)}


def record(i: int) -> dict:
    """Deterministic synthetic record #i."""
    if i in NEEDLE_AT:
        k = NEEDLE_AT[i]
        n = needle(k)
        return {"content": n["content"], "context": "", "project": tenant(i),
                "tags": ["needle", "infra"], "type": "fact", "needle": k}
    rnd = random.Random(i)
    parts = []
    for _ in range(rnd.randint(1, 3)):
        parts.append(rnd.choice(ACTIONS).format(
            svc=f"{rnd.choice(SERVICES)}-{rnd.randint(1, 8999)}",
            tech=rnd.choice(TECH), who=rnd.choice(FIRST_NAMES), num=rnd.randint(2, 5000),
        ))
    content = "; ".join(parts) + "."
    content = content[0].upper() + content[1:]
    return {"content": content, "context": "", "project": tenant(i),
            "tags": rnd.sample(TAGS, rnd.randint(1, 3)), "type": rnd.choice(TYPES), "needle": None}


def embed_text(rec: dict) -> str:
    return f"{rec['content']} {rec['context']}"


def pct(samples: list[float]) -> dict:
    if len(samples) < 2:
        return {"n": len(samples)}
    q = quantiles(samples, n=100, method="inclusive")
    return {"n": len(samples), "p50": round(q[49], 1), "p95": round(q[94], 1),
            "p99": round(q[98], 1), "max": round(max(samples), 1), "mean": round(sum(samples) / len(samples), 1)}


def vectors_path(d: Path) -> Path:
    return d / "vectors.f32"


def open_vectors(d: Path, total: int, mode: str) -> np.memmap:
    return np.memmap(vectors_path(d), dtype="<f4", mode=mode, shape=(total, EMBED_DIM))


# ── embed ─────────────────────────────────────────────────────────────


def cmd_embed(args) -> None:
    from fastembed import TextEmbedding

    d = Path(args.dir)
    d.mkdir(parents=True, exist_ok=True)
    meta = d / "vectors.json"
    done = json.loads(meta.read_text())["done"] if meta.exists() else 0
    vecs = open_vectors(d, args.total, "r+" if vectors_path(d).exists() else "w+")
    model = TextEmbedding(EMBED_MODEL)
    started = time.perf_counter()
    first = done
    while done < args.total:
        end = min(done + EMBED_CHUNK, args.total)
        texts = [embed_text(record(i)) for i in range(done, end)]
        out = np.asarray(list(model.embed(texts, batch_size=EMBED_BATCH, parallel=args.parallel)), dtype="<f4")
        vecs[done:end] = out
        vecs.flush()
        done = end
        meta.write_text(json.dumps({"done": done, "total": args.total, "model": EMBED_MODEL}))
        rate = (done - first) / (time.perf_counter() - started)
        print(f"embedded {done}/{args.total}  {rate:.0f}/s  eta {(args.total - done) / rate / 60:.1f} min", flush=True)


# ── server bootstrap ──────────────────────────────────────────────────


def bench_env(d: Path, async_enrichment: bool) -> None:
    os.environ["TAM_MEMORY_DIR"] = str(d / "store")
    os.environ["MEMORY_MODE"] = "fast"
    os.environ["MEMORY_LLM_ENABLED"] = "false"
    os.environ["MEMORY_ASYNC_ENRICHMENT"] = "true" if async_enrichment else "false"


def import_server(d: Path, async_enrichment: bool):
    bench_env(d, async_enrichment)
    sys.path.insert(0, str(SRC))
    import server as srv

    srv.MEMORY_DIR = d / "store"
    srv.store = srv.Store()
    srv.recall = srv.Recall(srv.store)
    srv.SID = "scale-bench"
    srv.BRANCH = ""
    return srv


class PrecomputedEmbed:
    """Serves loader vectors by text; anything else goes to the real model."""

    def __init__(self, real):
        self.real = real
        self.table: dict[str, list[float]] = {}
        self.misses = 0

    def __call__(self, texts):
        if len(texts) == 1 and texts[0] in self.table:
            return [self.table.pop(texts[0])]
        self.misses += len(texts)
        return self.real(texts)


def loaded_count(srv) -> int:
    return srv.store.db.execute("SELECT COUNT(*) FROM knowledge WHERE session_id='scale-load'").fetchone()[0]


# ── load ──────────────────────────────────────────────────────────────


def cmd_load(args) -> None:
    d = Path(args.dir)
    total = json.loads((d / "vectors.json").read_text())["total"]
    done_embedded = json.loads((d / "vectors.json").read_text())["done"]
    if args.upto > done_embedded:
        raise SystemExit(f"only {done_embedded} vectors embedded, need {args.upto}")
    vecs = open_vectors(d, total, "r")
    srv = import_server(d, async_enrichment=False)
    store = srv.store
    embed = PrecomputedEmbed(store.embed)
    store.embed = embed
    start = loaded_count(srv)
    started = time.perf_counter()
    for i in range(start, args.upto):
        rec = record(i)
        embed.table[embed_text(rec)] = vecs[i].tolist()
        rid = store.save_knowledge("scale-load", rec["content"], rec["type"], project=rec["project"],
                                   tags=rec["tags"], context=rec["context"], skip_dedup=True, skip_quality=True)[0]
        if rid is None:
            raise SystemExit(f"record {i} was not saved")
        if rec["needle"] is not None:
            store.db.execute("INSERT OR REPLACE INTO bench_needles(k, knowledge_id) VALUES (?, ?)", (rec["needle"], rid))
            store.db.commit()
        if (i + 1) % LOAD_REPORT_EVERY == 0:
            rate = (i + 1 - start) / (time.perf_counter() - started)
            print(f"loaded {i + 1}/{args.upto}  {rate:.0f}/s  eta {(args.upto - i - 1) / rate / 60:.1f} min  "
                  f"embed misses {embed.misses}", flush=True)
    print(f"loaded {args.upto}; embed misses {embed.misses}")


def ensure_needle_table(d: Path) -> None:
    (d / "store").mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(d / "store" / "memory.db")
    db.execute("CREATE TABLE IF NOT EXISTS bench_needles (k INTEGER PRIMARY KEY, knowledge_id INTEGER NOT NULL)")
    db.commit()
    db.close()


# ── measure ───────────────────────────────────────────────────────────


def top_ids(payload) -> list[int]:
    if isinstance(payload, list):
        payload = payload[0].text if hasattr(payload[0], "text") else payload[0]
    data = json.loads(payload) if isinstance(payload, str) else payload
    items = [item for group in data.get("results", {}).values() for item in group]
    items.sort(key=lambda it: it.get("score", 0), reverse=True)
    return [it["id"] for it in items[:TOP_K]]


def db_bytes(d: Path) -> int:
    return sum(p.stat().st_size for p in (d / "store").glob("memory.db*"))


def cmd_measure(args) -> None:
    d = Path(args.dir)
    srv = import_server(d, async_enrichment=True)
    db = srv.store.db
    n = db.execute("SELECT COUNT(*) FROM knowledge WHERE status='active'").fetchone()[0]
    needles = dict(db.execute("SELECT k, knowledge_id FROM bench_needles ORDER BY k LIMIT ?", (args.queries,)).fetchall())
    counts = {t: db.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
              for t in ("knowledge", "embeddings", "graph_nodes", "graph_edges", "knowledge_nodes")}
    print(f"records {n}, needles {len(needles)}, rows {counts}", flush=True)

    async def recall(query: str, project: str | None) -> tuple[float, list[int]]:
        payload = {"query": query, "limit": 10, "detail": "compact"}
        if project:
            payload["project"] = project
        t0 = time.perf_counter()
        out = await srv._do("memory_recall", payload)
        return (time.perf_counter() - t0) * 1000, top_ids(out)

    async def run() -> dict:
        report: dict = {"records": n, "rows": counts}
        t0 = time.perf_counter()
        await recall("прогрев индекса", None)
        report["first_recall_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        for label, scoped in (("recall_tenant", True), ("recall_all", False)):
            lat, hits = [], 0
            for k, kid in sorted(needles.items()):
                pos = next(p for p, kk in NEEDLE_AT.items() if kk == k)
                ms, ids = await recall(needle(k)["query"], tenant(pos) if scoped else None)
                lat.append(ms)
                hits += kid in ids
            report[label] = {**pct(lat), "hit_at_5": round(hits / len(needles), 3)}
            print(label, report[label], flush=True)

        save_ms, after_write_ms = [], []
        rnd = random.Random(n)
        for j in range(PROBE_SAVES):
            rec = record(10_000_000 + n + j)
            t0 = time.perf_counter()
            await srv._do("memory_save", {"type": rec["type"], "content": rec["content"],
                                           "project": rec["project"], "tags": rec["tags"]})
            save_ms.append((time.perf_counter() - t0) * 1000)
            k = rnd.choice(sorted(needles))
            pos = next(p for p, kk in NEEDLE_AT.items() if kk == k)
            ms, _ = await recall(needle(k)["query"], tenant(pos))
            after_write_ms.append(ms)
        report["save"] = pct(save_ms)
        report["recall_after_write"] = pct(after_write_ms)
        print("save", report["save"], flush=True)
        print("recall_after_write", report["recall_after_write"], flush=True)
        return report

    report = asyncio.run(run())
    report["db_mb"] = round(db_bytes(d) / 2**20, 1)
    report["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 1)
    out = d / f"measure-{n}{'-' + args.label if args.label else ''}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, ensure_ascii=False), flush=True)
    print(f"wrote {out}")


# ── http concurrency ──────────────────────────────────────────────────


def wait_healthy(proc: subprocess.Popen, httpx) -> None:
    deadline = time.time() + HTTP_READY_TIMEOUT_S
    while True:
        if proc.poll() is not None:
            raise SystemExit("HTTP server exited; see http-server.log")
        try:
            if httpx.get(f"http://127.0.0.1:{HTTP_PORT}/healthz", timeout=2).status_code == 200:
                return
        except httpx.HTTPError:
            if time.time() > deadline:
                raise SystemExit("HTTP server did not start; see http-server.log") from None
        time.sleep(1)


async def http_client(url: str, cid: int, needle_keys: list[int], stop_at: float,
                      recall_ms: list, save_ms: list, errors: list) -> None:
    import httpx
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client
    from mcp.shared.exceptions import MCPError

    rnd = random.Random(cid)
    async with streamable_http_client(url) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        j = 0
        while time.perf_counter() < stop_at:
            t0 = time.perf_counter()
            try:
                if rnd.random() < HTTP_SAVE_SHARE:
                    rec = record(20_000_000 + cid * 100_000 + j)
                    res = await session.call_tool("memory_save", {
                        "type": rec["type"], "content": rec["content"],
                        "project": rec["project"], "tags": rec["tags"]})
                    bucket = save_ms
                else:
                    k = rnd.choice(needle_keys)
                    pos = next(p for p, kk in NEEDLE_AT.items() if kk == k)
                    res = await session.call_tool("memory_recall", {
                        "query": needle(k)["query"], "project": tenant(pos),
                        "limit": 10, "detail": "compact"})
                    bucket = recall_ms
                if res.is_error:
                    errors.append(str(res.content)[:200])
                else:
                    bucket.append((time.perf_counter() - t0) * 1000)
            except (MCPError, httpx.HTTPError, OSError, TimeoutError) as exc:
                errors.append(repr(exc)[:200])
            j += 1


async def http_level(url: str, clients: int, needle_keys: list[int]) -> dict:
    recall_ms, save_ms, errors = [], [], []
    stop_at = time.perf_counter() + HTTP_DURATION_S
    await asyncio.gather(*(http_client(url, c, needle_keys, stop_at, recall_ms, save_ms, errors)
                           for c in range(clients)))
    done = len(recall_ms) + len(save_ms)
    return {"clients": clients, "ops_per_s": round(done / HTTP_DURATION_S, 1),
            "recall": pct(recall_ms), "save": pct(save_ms), "errors": len(errors),
            "error_samples": errors[:3]}


def cmd_http(args) -> None:
    import httpx

    d = Path(args.dir)
    db = sqlite3.connect(d / "store" / "memory.db")
    n = db.execute("SELECT COUNT(*) FROM knowledge WHERE status='active'").fetchone()[0]
    needle_keys = [k for (k,) in db.execute("SELECT k FROM bench_needles")]
    db.close()

    env = dict(os.environ, TAM_MEMORY_DIR=str(d / "store"), MEMORY_MODE="fast", MEMORY_LLM_ENABLED="false",
               MCP_TRANSPORT="http", MCP_HTTP_HOST="127.0.0.1", MCP_HTTP_PORT=str(HTTP_PORT),
               MCP_HTTP_WORKERS=str(args.workers))
    url = f"http://127.0.0.1:{HTTP_PORT}/mcp"
    results = []
    with open(d / "http-server.log", "w") as log:
        proc = subprocess.Popen([sys.executable, str(SRC / "server.py")], env=env,
                                stdout=log, stderr=subprocess.STDOUT)
        try:
            wait_healthy(proc, httpx)
            for c in [int(x) for x in args.clients.split(",")]:
                r = asyncio.run(http_level(url, c, needle_keys))
                print(json.dumps(r, ensure_ascii=False), flush=True)
                results.append(r)
        finally:
            proc.terminate()
            proc.wait(timeout=30)
    out = d / f"http-{n}-w{args.workers}.json"
    out.write_text(json.dumps({"records": n, "workers": args.workers, "duration_s": HTTP_DURATION_S,
                               "save_share": HTTP_SAVE_SHARE, "levels": results}, indent=2, ensure_ascii=False))
    print(f"wrote {out}")


# ── multi-process writers ─────────────────────────────────────────────


def writer_proc(args) -> None:
    d = Path(args.dir)
    srv = import_server(d, async_enrichment=True)
    ok, deduped, locked, other, lat = 0, 0, 0, 0, []
    base = 30_000_000 + args.writer_id * WRITER_RECORDS
    for j in range(WRITER_RECORDS):
        rec = record(base + j)
        t0 = time.perf_counter()
        try:
            _, was_dedup, *_ = srv.store.save_knowledge(f"writer-{args.writer_id}", rec["content"], rec["type"],
                                                        project=rec["project"], tags=rec["tags"])
            ok += 1
            deduped += bool(was_dedup)
            lat.append((time.perf_counter() - t0) * 1000)
        except sqlite3.OperationalError as exc:
            if "locked" in str(exc) or "busy" in str(exc):
                locked += 1
            else:
                other += 1
                print(f"writer {args.writer_id}: {exc}", file=sys.stderr)
    print(json.dumps({"ok": ok, "deduped": deduped, "locked": locked, "other": other, "lat": lat}))


def cmd_writers(args) -> None:
    d = Path(args.dir)
    results = []
    db = sqlite3.connect(d / "store" / "memory.db")
    writer_rows = "SELECT COUNT(*) FROM knowledge WHERE session_id LIKE 'writer-%'"
    for procs in [int(x) for x in args.procs.split(",")]:
        before = db.execute(writer_rows).fetchone()[0]
        first_id = db.execute("SELECT COALESCE(MAX(CAST(SUBSTR(session_id, 8) AS INTEGER)) + 1, 0) "
                              "FROM knowledge WHERE session_id LIKE 'writer-%'").fetchone()[0]
        t0 = time.perf_counter()
        children = [subprocess.Popen([sys.executable, __file__, "writer", "--dir", str(d), "--writer-id", str(first_id + w)],
                                     stdout=subprocess.PIPE, text=True) for w in range(procs)]
        outs = [c.communicate()[0] for c in children]
        wall = time.perf_counter() - t0
        parsed = [json.loads(o.strip().splitlines()[-1]) for o in outs]
        lat = [x for p in parsed for x in p["lat"]]
        ok = sum(p["ok"] for p in parsed)
        r = {"procs": procs, "saved": ok, "deduped": sum(p["deduped"] for p in parsed), "rows_in_db": db.execute(writer_rows).fetchone()[0] - before,
             "locked": sum(p["locked"] for p in parsed),
             "other_errors": sum(p["other"] for p in parsed), "wall_s": round(wall, 1),
             "saves_per_s_incl_startup": round(ok / wall, 1), "save": pct(lat)}
        print(json.dumps(r, ensure_ascii=False), flush=True)
        results.append(r)
    n = db.execute("SELECT COUNT(*) FROM knowledge WHERE status='active'").fetchone()[0]
    db.close()
    (d / f"writers-{n}.json").write_text(json.dumps({"records": n, "per_process": WRITER_RECORDS,
                                                    "levels": results}, indent=2, ensure_ascii=False))


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("embed"); e.add_argument("--dir", required=True); e.add_argument("--total", type=int, required=True)
    e.add_argument("--parallel", type=int, default=None)
    ld = sub.add_parser("load"); ld.add_argument("--dir", required=True); ld.add_argument("--upto", type=int, required=True)
    m = sub.add_parser("measure"); m.add_argument("--dir", required=True)
    m.add_argument("--queries", type=int, default=NEEDLES, help="needle queries per recall series")
    m.add_argument("--label", default="", help="suffix for the report file name")
    h = sub.add_parser("http"); h.add_argument("--dir", required=True); h.add_argument("--clients", default="1,4,16")
    h.add_argument("--workers", type=int, default=1, help="MCP_HTTP_WORKERS for the server")
    w = sub.add_parser("writers"); w.add_argument("--dir", required=True); w.add_argument("--procs", default="1,2,4,8")
    wp = sub.add_parser("writer"); wp.add_argument("--dir", required=True); wp.add_argument("--writer-id", type=int, required=True)
    args = p.parse_args()
    if args.cmd == "load":
        ensure_needle_table(Path(args.dir))
    {"embed": cmd_embed, "load": cmd_load, "measure": cmd_measure, "http": cmd_http,
     "writers": cmd_writers, "writer": writer_proc}[args.cmd](args)


if __name__ == "__main__":
    main()
