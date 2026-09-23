# MemoryAgentBench — FactConsolidation

total-agent-memory as a memory method in [MemoryAgentBench](https://github.com/HUST-AI-HYZ/MemoryAgentBench), on the Conflict Resolution split: FactConsolidation single-hop (FC-SH) and multi-hop (FC-MH), 6k and 262k contexts. The method module is `methods/total_agent_memory.py` in the MemoryAgentBench PR; this directory holds the per-question results.

## Method

- **Store.** TAM 14.3.1 (the release that fixes dedup, see `CHANGELOG.md`) runs as its own process over MCP stdio, in fast mode. It gets a fresh `TAM_MEMORY_DIR` per context and no API keys, so it makes no LLM calls.
- **Write.** The context is parsed into one fact per record, in context order. The parser is the harness's `parse_fact_lines`, the same one the knowl and agentmemory methods use. Each fact is one `memory_save` (`type=fact`), called sequentially, so recorded times grow with position.
- **Read.** `memory_recall` with `limit=10` and `detail=full` runs on the question that `_extract_retrieval_query` extracts. Hits keep TAM's rank order. The reader assembly is shared with the RAG family (`Memory i:` labels, instruction after the facts, generic system message) and is the same as knowl and agentmemory.
- **Two configs.** They differ only in whether the reader sees each hit's recorded time:
  - `dates` prefixes each hit with `[recorded 2026-09-21T13:05:59.181144Z]`;
  - `nodates` passes the fact text alone.
- **Supersession at write time is a third config.** By default a later value is stored beside the earlier one and both remain retrievable. TAM 14.4.0 adds `memory_save(supersede=true)`: a fact with the same opening words and a different trailing value retires the earlier record. The `supersede` config (`tam_supersede: true`) saves every fact that way and shows recorded dates; everything else matches `dates`. Latest-wins answering in `memory_answer` returns an answer rather than passages, so it is not used here.
- **Reader and settings.** gpt-4o-mini at temperature 0.7. `retrieve_num: 10`, `buffer_length: 200` and `input_length_limit: 10000000` are copied from `Simple_rag_bm25`.

## Results

SubEM; each TAM figure is one run. The 14.3.1 rows are the 14.3.1 release code, the supersede row is the 14.4.0 code.

| | FC-SH 6k | FC-SH 262k | FC-MH 6k | FC-MH 262k |
|---|---:|---:|---:|---:|
| TAM 14.4.0, supersede | 99.0 | 93.0 | 27.0 | 9.0 |
| TAM 14.3.1, dates | 82.0 | 85.0 | 13.0 | 3.0 |
| TAM 14.3.1, no dates | 71.0 | 81.0 | 11.0 | 6.0 |
| BM25 (this harness, same run setup) | 83.0 | 46.0 | 12.0 | 3.0 |
| knowl (PR #23) | 95.0 | 89.0 | — | — |
| agentmemory (PR #23) | 83.0 | 79.0 | — | — |
| TAM 14.3.0, dates | 71.0 | — | — | — |

- **Run-to-run spread.** FC-SH was run three times at temperature 0.7, during development and on the release code. With dates the scores were 80 / 84 / 82 on 6k and 86 / 86 / 85 on 262k; without dates 71 / 72 / 71 and 79 / 80 / 81. The earlier two runs predate a change to how exact repeats are stored, which touched one fact in FC-SH 262k and none in FC-SH 6k.
- **Cost.** The 14.3.x runs in this directory cost $1.65 of gpt-4o-mini in total (11.0M input tokens).
- **Ingest time.** Writing the 18,332 facts of a 262k context took 4.3–4.5 minutes per run (the harness's `memory_construction_time`) on an M2 Max, with two TAM runs going in parallel.

## Findings

1. **Retrieval is not the bottleneck on FC-SH.** A retrieval-only probe on FC-SH 6k found the gold answer string among the 10 hits for 100 of 100 questions. What goes wrong is choosing among the conflicting values. For 12 of the 20 misses in the first dates run (`run1-dates`), the probe ranked the newest fact first, and gpt-4o-mini still answered with the real-world value: Satya Nadella over the recorded "Steve Jobs", Chang'an over "Beaumont", jazz over "post-punk". The FC instruction talks about serial numbers, and a recorded timestamp evidently carries less weight with the reader than a serial number would.
2. **The recorded date is worth 4–12 points on FC-SH across the three runs.** It is the only recency signal TAM's search path returns; in the release run, withholding it drops FC-SH from 82 / 85 to 71 / 81.
3. **At 262k, TAM keeps its 6k score while BM25 falls to 46.** BM25 over 4096-character chunks returns the whole 6k context (about 6,700 tokens) but only a slice of the 262k one. TAM's input stays at about 560 tokens at both sizes.
4. **FC-MH is near floor for every method measured here.** Two-hop questions need the intermediate entity, which a single retrieval on the question rarely returns.
5. **14.3.0 lost facts on write.** Its dedup treated near-identical texts as repeats, so 36 of the 455 FC-SH 6k facts, all of them updates, were never stored, and the score was 71.0. 14.3.1 stores a record unless it has the same words in the same order as one already stored; an exact repeat replaces the stored record, so it carries the later date. In FC-SH 262k that happens once, a case-only repeat ("Safety is associated with…" / "safety is associated with…").
6. **Retiring the old value at write time closes most of the FC-SH gap.** With `supersede`, the reader no longer has to pick between conflicting values: FC-SH goes from 82 / 85 to 99 / 93 and FC-MH from 13 / 3 to 27 / 9, above knowl (95 / 89) on FC-SH. The rule fits this dataset, where every fact is single-valued. On a real 5,128-record store it would have retired 138 records that were not updates (multi-valued relations such as "likes jazz" / "likes rock", logs with a shared header), which is why it is opt-in per record and not the default.

## Files

`raw/<run>/<dataset>.json` are the harness's result files, with every question, answer, model output and metric:

- `tam-14.3.1-dates`, `tam-14.3.1-nodates`: the release-code runs in the table.
- `tam-14.4.0-supersede`: the `supersede` config on the 14.4.0 code.
- `run1-*`, `run2-*`: the two earlier FC runs (spread).
- `tam-14.3.0-dates`: 14.3.0 on FC-SH 6k.
- `bm25`: `Simple_rag_bm25` in this environment. It needs `langchain-core<1`, because the harness calls `get_relevant_documents`.
