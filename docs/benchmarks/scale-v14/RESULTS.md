# Scale: 10k, 100k and 1M records

How the SQLite core behaves as a store grows, on 14.3.0 and 14.3.1.

Host: Apple M2 Max, 12 cores, 64 GB RAM, macOS. Harness: `benchmarks/scale_bench.py`. Raw reports: `raw/v14.3.0/`, `raw/v14.3.1/`, and `raw/intermediate/` (the same fixes before migration 035 and multi-process HTTP).

## Corpus

- **Records.** Synthetic records spread over 200 tenants (`project = tenant-000 … tenant-199`). Each record joins one to three Russian or English templates about services, people and technologies ("перевели billing-4121 на Redis 7, миграцию вёл Ольга"), with varied types and tags.
- **Planted facts.** 300 facts sit among the oldest 10,000 records, each with a unique service id ("Вебхуки сервиса media-9000 слушают порт 51234") and a question about it ("какой порт у вебхуков media-9000"). At 1M records they are the oldest 1%.
- **Load path.** Records go through the real `Store.save_knowledge`, so the same tables, FTS and revision triggers, graph links and episodic event nodes as `memory_save` are written. The only differences are precomputed embeddings (paraphrase-multilingual-MiniLM-L12-v2, 384d, passed through unchanged) and `skip_dedup=True`.
- **Store size.** 1M records make a 6.9 GB database with 1.0M graph nodes, 3.4M graph edges and 4.4M knowledge–node links.
- **Growth between rounds.** The measurements themselves add records, so the 14.3.1 rows ran at 15.6k, 105k and 1.016M records.

## What is measured

| Series | How |
|---|---|
| `recall_tenant` | 300 `memory_recall` calls through the MCP dispatcher (`_do`), one per planted fact, scoped to its tenant; `limit=10`, `detail=compact` |
| `recall_all` | the same questions without a project |
| `hit_at_5` | the planted record is among the first five results |
| `save` | 50 `memory_save` calls with a real embedding |
| `recall_after_write` | a scoped recall right after each of those saves |
| HTTP | one `MCP_TRANSPORT=http` server; 1, 4 and 16 MCP clients for 30 s each; 90% recall, 10% save |
| writers | 1, 4 and 8 processes, each saving 400 records into the same database file |

Fast mode (the default). No LLM is called.

## Results

Latency in ms, p50 / p95.

| | 14.3.0 | 14.3.1 |
|---|---:|---:|
| **Recall, tenant-scoped** | | |
| 10k | 62 / 104 | 19 / 25 |
| 100k | 781 / 1,901 | 25 / 35 |
| 1M | 6,331–63,977 (5 queries) | 105 / 147 |
| **Recall, all tenants** | | |
| 10k | 59 / 86 | 22 / 32 |
| 100k | 589 / 924 | 55 / 105 |
| 1M | 4,980–7,999 (5 queries) | 349 / 805 |
| **Save** | | |
| 10k | 48 / 64 | 31 / 57 |
| 100k | 149 / 194 | 43 / 65 |
| 1M | — | 52 / 78 |
| **HTTP, calls/s: 1 client · 16 clients** | | |
| 10k, one process | 12.7 · 13.4 | 41.4 · 47.8 |
| 10k, `MCP_HTTP_WORKERS=4` | — | 41.3 · 167.7 |
| 100k, one process | 1.1 · 1.3 | 32.7 · 37.4 |
| 100k, `MCP_HTTP_WORKERS=4` | — | 33.1 · 117.3 |
| 1M, one process | — | 9.2 · 9.8 |
| 1M, `MCP_HTTP_WORKERS=4` | — | 7.5 · 24.9 |
| **8 writer processes, saves/s** | | |
| 10k | 108 | 166 ¹ |
| 100k | 45 | 147 ¹ |
| 1M | — | 94 ¹ |

¹ From `raw/intermediate/`. The writer path in 14.3.1 differs only by migration 035's triggers.

- **`hit_at_5`.** Tenant-scoped it is 1.00 at every size, before and after. Unscoped it is 0.97–0.98 on both versions; those misses were not analysed.
- **14.3.0 at 1M.** A full series did not finish within 29 minutes. Five timed queries are in `raw/v14.3.0/recall-1m-5-queries.json`.
- **Migration 035.** It rebuilds the full-text index. On the 1M store, opening it for the first time, migration included, took 29 s.

## What was slow in 14.3.0

Profiled at 110k records. Before the fixes, five queries accounted for 98% of recall time and 70% of save time.

1. **Scoped full-text tier: 650 ms per recall.** With `k.project=?` in the join, SQLite walked the tenant's rows and ran MATCH once per row.
   - 14.3.1 gives `knowledge_fts` a fifth column, `fts_project`: one token per project, read from a generated column of `knowledge` (migration 035).
   - The scoped query ANDs that token into MATCH, so FTS5 intersects doclists and scores only the tenant's matches. The project column gets bm25 weight 0.
   - This brought p95 at 1M from 456 ms (the intermediate fix, a materialized CTE) to 147 ms.
2. **Graph seeds: 540 ms per recall.** `find_seed_nodes` compared `LOWER(name)`, which no index covers, across `graph_nodes`, 4–5 times per recall. Every save adds an episodic event node, so the table grows with the store.
   - It now uses the indexed `name_norm` with `lower(trim(?))`, the expression the migration 026 triggers store.
   - The prefix fallback is an index range with `+status`; without statistics the planner otherwise prefers the status index.
3. **Dedup lookup: 126 ms per save.** An FTS `OR` over the first twelve words scored every record that shared a common word.
   - Dedup now looks only for exact repeats, so it ANDs the record's words (three letters or longer) and the project token.
   - It takes 3–8 ms.
4. **Solutions lookup: 850 ms per unscoped recall at 1M.** The `available_solutions` enrichment ran `content LIKE '%word%'` over all 203k solutions. It now uses the full-text index (word prefixes).
5. **Recall counter re-indexing.** The FTS update trigger fired on every `UPDATE knowledge`, including the recall counter bumped for each of the ten results of every recall. It now fires only when content, context, tags or project change. The missing delete trigger was added too.

The HTTP server ran every tool call synchronously on one event loop, so its throughput was 1 / latency however many clients connected. `MCP_HTTP_WORKERS=N` starts N server processes on one listening socket. Each is a fresh interpreter with its own store connection, and they share the database through SQLite WAL. Sessions are stateless in that mode.

## Limits that remain

- **One process still serves one call at a time.** Workers add throughput; one client's latency does not drop. With 4 workers the 1M store reaches 25 calls/s at 16 clients. More than 4 workers was not measured.
- **Unscoped recall at 1M: 349 ms p50, 805 ms p95.** Without a project, the full-text tier still ranks every match of a common word across the whole store.
- **Save stalls at 1M: p99 1.4 s.** Occasional save stalls, most likely WAL checkpoints; not investigated.
- **First recall after start: 10.8 s at 1M.** The binary vector pool (56 B per record) loads from SQLite on the first query and again after any write, because the cache is invalidated.

## Caveats

- **The planted questions are easy.** Each contains a unique service id, so `hit_at_5` confirms that the fixes return the same records rather than measuring retrieval quality. Quality is covered by LongMemEval and `docs/benchmarks/knowledge-update-v14/`.
- **Mixed-code load.** The 1M store was partly loaded with fixed code. The load uses `skip_dedup=True`, and no fixed path writes the rows differently.
- **Overlap.** The 14.3.0 100k series overlapped for a few seconds with an unrelated test process. Every other series ran with nothing else on the machine.
