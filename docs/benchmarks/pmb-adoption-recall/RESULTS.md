# Recall pipeline changes (14.9): free retrieval evaluation

Six changes to `memory_recall`, each measured before and after on the free retrieval harness
(no LLM, no paid API): the rank of the first gold record in the ranked list (R@1, R@5, R@10, MRR)
and, where a question has several gold records, `coverage@10` (mean share of a question's gold
records in the top 10). Defaults change only where the numbers do not regress.

| # | change | flag / default | evidence |
|---|---|---|---|
| 1 | "What was it before?" puts the older value first | always on (regex detector) | historical questions 0/48 -> 48/48 correct; current questions 48/48 -> 48/48; LoCoMo/LongMemEval-S ranks unchanged |
| 2 | Usage statistics written off the read path | `MEMORY_RECALL_TOUCH_FLUSH_MS=250` | search under a held write lock 1,008-1,108 ms -> 16-17 ms; counts identical |
| 3 | Intent -> record-type multiplier | `MEMORY_RECALL_INTENT_BOOSTS` decision/lesson/howto = 1.3 | benchmarks hold only `fact` records: LoCoMo dev/test identical with and without |
| 4 | Compound queries: each part's best record gets a slot | `MEMORY_RECALL_COMPOUND_SLOTS=on` (default on) | LoCoMo natural questions identical; compound pairs coverage@10 0.583 -> 0.629 and 0.593 -> 0.637 |
| 5 | `query_entity` RRF tier | tier weight `query_entity=0` (off) | with an entity graph: held-out R@10 -0.1 to -0.4 at weights 0.3-1.0, so it ships off |
| 6a | Singleflight for identical concurrent recalls | `MEMORY_RECALL_SINGLEFLIGHT=on`, wait 30 s | correctness tests; no effect on single-threaded benchmarks |
| 6b | Cross-lingual FTS damp | `MEMORY_RECALL_FTS_CROSSLINGUAL_DAMP=0.3` | fires on 1/383 LoCoMo dev questions (gold from 5th place to 1st), 0/1,153 held-out, 0/95 LongMemEval-S; no regression |

Plus one fix found on the way: the result cache stored entries under a key without the tier
weights and looked them up with it, so no lookup ever hit. Lookup and store now share one key that
also covers the cross-encoder's readiness and the ranking switches read from the environment, and
cached results are handed out as copies (the `memory_recall` handler edits its result in place).

## Setup

- Stores and scoring of [`benchmarks/tier_ablation.py`](../../../benchmarks/tier_ablation.py): one store per
  LoCoMo conversation / LongMemEval-S question, one record per turn, built through `Store.save_knowledge`
  with the default local embedder (`paraphrase-multilingual-MiniLM-L12-v2`, fastembed); searches through
  `Recall.search` with `record_usage=False`, `_explain=True`.
- Configurations that differ in environment switches as well as tier weights run through
  [`benchmarks/recall_changes_eval.py`](../../../benchmarks/recall_changes_eval.py), which also records
  every gold rank (`coverage@K`), builds a proxy entity graph (`--entity-graph`) and compound pairs
  (`--compound-pairs`).
- Baseline: `git archive HEAD` of `f2f124f` extracted to a separate directory and run with the same
  scripts (the 14.9 scripts tolerate the older `Recall` without `close()`). After: this tree.
- LoCoMo: conversations 0-2 (dev, 383 questions), 3-9 (held-out, 1,153). LongMemEval-S: the dev id
  list of head-to-head-v14 (100 ids; the 5 abstention questions have no gold turn, so 95 are scored). Cross-encoder off unless stated (to see the pipeline, as in
  tier-ablation-v14); one run with it on.
- Apple M2 Max, Python 3.13, other jobs running on the machine (latencies are relative, not absolute).

## Baseline vs after, natural questions (defaults)

| bench | split | cross-encoder | baseline R@1 / R@5 / R@10 / MRR | after R@1 / R@5 / R@10 / MRR | coverage@10 base -> after |
|---|---|---|---|---|---|
| LoCoMo | dev (383) | off | 0.303 / 0.561 / 0.689 / 0.423 | 0.305 / 0.561 / 0.689 / 0.425 | 0.636 -> 0.636 |
| LoCoMo | held-out (1,153) | off | 0.314 / 0.586 / 0.672 / 0.439 | 0.314 / 0.586 / 0.672 / 0.439 | 0.599 -> 0.599 |
| LoCoMo | dev (383) | on (window 50) | 0.467 / 0.739 / 0.828 / 0.586 | 0.467 / 0.739 / 0.828 / 0.586 | 0.767 -> 0.767 |
| LongMemEval-S | dev (95 of 100) | off | 0.463 / 0.832 / 0.916 / 0.619 | 0.463 / 0.832 / 0.916 / 0.619 | 0.836 -> 0.836 |

What fired on the natural questions (after, defaults): LoCoMo dev — historical 1, FTS damp 1, compound 0;
held-out — historical 8, compound 3, FTS damp 0; LongMemEval-S dev — historical 4, compound 0, FTS damp 0,
every gold rank identical to the baseline. Intent boosts, FTS damp and compound slots switched off one at
a time leave LongMemEval-S dev unchanged as well (`intent_off`, `damp_off`, `slots_off` in
[`raw/lme.md`](raw/lme.md)). The one LoCoMo dev change is the FTS damp on
"Which city have both Jean and John visited?" (none of its content words is in that conversation's
index; the gold turn moves from 5th place to 1st).

## 1. Historical-value questions

Stage 4.9 (`order_value_updates`) puts the newest of several values of one statement first, so a
question about the earlier value got the later one. `query_shapes.is_historical_value_query` now
detects the question in English and Russian and the stage fills the group's places oldest first; the
cross-encoder's own update ordering is skipped for such a question.

Rule: a comparison frame ("before or after", "earlier than", "раньше, чем", "что было раньше: X или Y")
or an anchor to the later state ("after", "now", "instead of", "после", "сейчас", "вместо") is never
historical; a strong cue decides alone ("previously", "formerly", "originally", "initially", "used to"
with a person as subject, "my old / previous / former X", "no longer", "раньше", "ранее", "прежде",
"изначально", "когда-то", "бывший", "прежний", "старый", "до этого", "больше не"); a weak cue ("before
the migration", "before we switched", "at first", "сначала", "до переезда") counts only in a question
that is not a procedure. Not historical: "Before deploying, run tests", "Should I back up before
migrating?", "What is the tool used to deploy?", "What ingredient did Nate use to make the ice cream?",
"Which tournament did Nate win in the beginning of November?", "Он по-прежнему работает в Ozon?"
A reference to the dialogue itself ("in our previous conversation you recommended...", "в прошлом
разговоре") and a time window ("in the past three months", "my previous year's time") are removed
before the cue search: the first version of the detector fired on 16 of the 95 LongMemEval-S dev
questions through them ("previous X" as a strong cue), the current one on 4 ("initially", "in the past",
"my old keyboard"). No rank moved in either case, because those stores hold no value updates of one
statement, but on a store that does, a misfire would put the older value first
(tests: `tests/test_historical_value_queries.py`).

[`benchmarks/historical_value_eval.py`](../../../benchmarks/historical_value_eval.py): 8 subjects x 3
attributes, each saved with an old and a new value (plus noise), per language; share of questions
whose first value record is the expected one:

| | baseline, CE off | after, CE off | baseline, CE on | after, CE on |
|---|---:|---:|---:|---:|
| EN current ("What is the database of the billing service?") | 1.000 | 1.000 | 1.000 | 1.000 |
| EN historical ("What was ... before?") | 0.000 | **1.000** | 0.000 | **1.000** |
| RU current ("Какая основная база данных сервиса биллинга?") | 1.000 | 1.000 | 1.000 | 1.000 |
| RU historical ("... была раньше?") | 0.000 | **1.000** | 0.000 | **1.000** |

## 2. Usage statistics off the read path

`Recall.search` used to run one UPDATE per returned record and a COMMIT inside every search
(`Store.bump_recall`), taking the SQLite write lock on a read. `memory_core.recall_touch.RecallTouchBuffer`
now collects the ids (no I/O in the search), coalesces repeats and writes them with one `executemany`
in one transaction from a background thread every `MEMORY_RECALL_TOUCH_FLUSH_MS` (250). The flusher has
its own connection of the Store's connection class (SQLite file or PostgreSQL workspace, so audit
triggers keep working); it runs only while something is pending, exits after a failed flush keeping the
counts, and every buffer is flushed at interpreter exit and at multiprocessing worker exit.
`record_usage=False` still records nothing. The result cache keys on `PRAGMA data_version` of the
flusher's connection, which sees every commit except its own, so usage flushes do not invalidate
cached results while other commits still do.

Search latency on a LoCoMo store (450 searches, distinct texts, `record_usage=True`):

| | p50 ms | p95 ms | recall_count written |
|---|---:|---:|---:|
| baseline (UPDATE + COMMIT in the search) | 25.0 | 34.8 | 4,500 |
| after (buffered) | 24.1 | 31.1 | 4,500 |

With another connection holding the write lock for 1 s, five searches: baseline 1,016 / 1,101 / 1,108 /
1,008 / 1,018 ms (each waits for the lock), after 17 / 16 / 17 / 16 / 17 ms.

## 3. Intent -> type weighting

`query_shapes.query_intents` detects decision ("what did we decide", "why did we choose", "какое
решение", "почему выбрали"), lesson/rule ("lessons learned", "rules for", "best practices", "какие
грабли", "правила") and how-to/solution ("how did we fix", "how to configure", "workaround", "как
починить") questions; the fused score of `decision`, `lesson`+`convention` and `solution` records is
multiplied by `MEMORY_RECALL_INTENT_BOOSTS` (default 1.3 each, 1.0 switches an intent off). The
benchmark stores hold only `fact` records, so the boost cannot move them: LoCoMo dev and held-out are
identical with it on and off (`intent_off` rows in the raw tables). The detector fires on 5/383 dev and
3/1,153 held-out questions (e.g. "What decision did James and Samantha make"); "scuba diving lessons" and
"gaming convention" are excluded explicitly. The default 1.3 matches the user-turn boost; it is a
moderate prior, not a tuned value, because no free corpus has typed records.

## 4. Compound queries: guaranteed slots

`query_shapes.compound_parts` splits "X; Y", "X? Y?", "X, also Y", "X, а также Y" always, and "X and
how/what/where ...", "X, and what ...", "X и где ..." only before a new question with two content words
on each side ("Caroline and Melanie", "her family and friends", "A, B, and C" stay whole). Each part is
searched on its own (FTS + semantic fused per part); its best record is guaranteed a place in the
caller's top K, taking the lowest places held by non-winners.

Natural questions: the split fires on 0/383 LoCoMo dev and 3/1,153 held-out questions and changes no
rank (K=10: R@10 0.674 / 0.661 with and without). Compound pairs: two consecutive single-hop questions
of a held-out conversation joined into one query, gold = both questions' evidence, K=10:

| pairs (318) | config | R@10 (first gold) | coverage@10 | p50 ms |
|---|---|---:|---:|---:|
| "Q1? Q2?" | baseline | 0.802 | 0.583 | 40 |
| "Q1? Q2?" | after, slots off | 0.802 | 0.583 | 40 |
| "Q1? Q2?" | after, slots on | **0.852** | **0.629** | 63 |
| "Q1, and q2?" | baseline | 0.805 | 0.593 | 41 |
| "Q1, and q2?" | after, slots off | 0.805 | 0.593 | 41 |
| "Q1, and q2?" | after, slots on | **0.852** | **0.637** | 64 |

No regression on natural questions and +4.4-4.6 points of coverage on compound ones, so the flag ships on.
The cost is one FTS + semantic search per part (~20 ms) on the queries it fires on.

## 5. query_entity tier

`retrieval_tiers.query_entity_candidates`: name variants of the query's entities and content words
(as typed, lower case, capitalised, adjacent pairs) are matched exactly against `graph_nodes.name`
(hub types and nodes linked to more than `MEMORY_GRAPH_HUB_DEGREE` records skipped); each node weighs
IDF = 1/ln(2 + mentions); a record's weight is the sum of its matched entities' IDF times
1 + 0.5 x (N - 1); score w/(1 + w); fused as the RRF tier `query_entity`. While its weight is 0 the
relational router's post-rerank injection (first two entities, flat 0.6 x max) runs as before; when it
is on, that injection is skipped.

Benchmark stores have no graph, so the tier was measured on copies with a proxy entity graph: one node
per speaker and per capitalised word that does not open a sentence, linked to every record containing it.

| LoCoMo, entity graph | query_entity | R@1 | R@5 | R@10 | MRR |
|---|---:|---:|---:|---:|---:|
| dev (383) | 0 | 0.285 | 0.553 | 0.687 | 0.411 |
| dev | 0.3 | 0.274 | 0.561 | 0.692 | 0.406 |
| dev | 0.5 | 0.274 | 0.564 | 0.692 | 0.408 |
| dev | 1.0 | 0.272 | 0.553 | 0.692 | 0.402 |
| held-out (1,153) | 0 | 0.304 | 0.588 | 0.673 | 0.433 |
| held-out | 0.3 | 0.300 | 0.587 | 0.672 | 0.431 |
| held-out | 0.5 | 0.300 | 0.586 | 0.670 | 0.431 |
| held-out | 1.0 | 0.299 | 0.586 | 0.669 | 0.429 |

A first proxy graph that also counted sentence-initial words ("Hey", "Thanks") cost 3.7 points of dev
R@10 at weight 1.0 (that graph builder was replaced; its run is not in `raw/`). On the cleaner graph the tier gains 0.5 points of dev R@10
and loses 0.1-0.4 on the held-out split, R@1 and MRR lower everywhere: the default weight is 0. On an
enriched store with curated entity nodes (triple extraction) it may behave differently; switch it on
with `MEMORY_RECALL_TIER_WEIGHTS=query_entity=0.3`.

## 6. Singleflight and cross-lingual FTS damp

Singleflight (`memory_core.singleflight`): concurrent `Recall.search` calls with the same project,
normalised query (case and whitespace folded), limit and options share one execution; waiters get a
copy of the leader's result and record their own usage; a waiter that waits longer than
`MEMORY_RECALL_SINGLEFLIGHT_WAIT_SECONDS` (30) or whose leader failed runs its own search. The
benchmarks call search from one thread, so it never engages there.

FTS damp (`memory_core.fts_vocab`): when the fts tier found something but none of the query's content
words (index tokens of at least 3 characters, not a function word, not a number) occurs in the
full-text vocabulary, the fts tier's fusion weight is multiplied by
`MEMORY_RECALL_FTS_CROSSLINGUAL_DAMP` (0.3; 1.0 = off). SQLite reads an `fts5vocab` table in the
connection's temp schema (nothing is written to the file), PostgreSQL the `knowledge_tsv` lexemes;
Russian words match by Snowball stem prefix. It fired on 1 LoCoMo dev question (5th place -> 1st, the
R@1/MRR difference in the first table) and on none of the held-out or LongMemEval-S questions; a
Russian-records / English-question test lives in `tests/test_recall_fts_damp.py`.

## Tests

- Baseline (`f2f124f`, same command): 3,643 passed, 541 skipped, 0 failed.
- After: 3,855 passed, 555 skipped, 0 failed (212 new tests; the 14 extra skips are their PostgreSQL variants).
- New: `tests/test_historical_value_queries.py`, `test_recall_touch.py` (SQLite + PostgreSQL),
  `test_recall_intents.py`, `test_recall_compound_slots.py`, `test_recall_query_entity_tier.py`
  (SQLite + PostgreSQL), `test_recall_fts_damp.py` (SQLite + PostgreSQL), `test_recall_singleflight.py`,
  `test_recall_settings.py` (the environment knobs: defaults, valid and malformed values; `bump_recall`);
  `test_recall_usage_feedback.py` flushes the buffer before reading the counters.
- PostgreSQL variants: `--backend=postgres -m postgres` (throwaway pgvector container), 18 passed, 1 skipped
  (a test of the SQLite result cache).

## Reproduce

```bash
W=$(mktemp -d); OUT=$PWD/out
# baseline tree with the 14.9 eval scripts and the benchmark data
B=$(mktemp -d); git archive f2f124f | tar -x -C $B
cp benchmarks/recall_changes_eval.py benchmarks/historical_value_eval.py $B/benchmarks/
rm -rf $B/benchmarks/data && ln -s $PWD/benchmarks/data $B/benchmarks/data
# baseline (builds the stores once per split; LongMemEval-S dev takes ~80 min on an M2 Max)
(cd $B && python benchmarks/tier_ablation.py --bench locomo --split dev  --work-root $W --output $OUT/base_locomo_dev.json --only-extra)
(cd $B && python benchmarks/tier_ablation.py --bench locomo --split test --work-root $W --output $OUT/base_locomo_test.json --only-extra)
(cd $B && python benchmarks/tier_ablation.py --bench lme    --split dev  --work-root $W --output $OUT/base_lme_dev.json --only-extra)
# baseline coverage@10, cross-encoder, compound pairs, historical values
(cd $B && python benchmarks/recall_changes_eval.py --bench locomo --split dev --work-root $W --output $OUT/base_cov_locomo_dev.json)
(cd $B && python benchmarks/recall_changes_eval.py --bench lme    --split dev --work-root $W --output $OUT/base_cov_lme.json)
(cd $B && python benchmarks/recall_changes_eval.py --bench locomo --split dev --work-root $W --output $OUT/base_dev_ce.json --cross-rerank on)
(cd $B && python benchmarks/recall_changes_eval.py --bench locomo --split test --work-root $W --output $OUT/base_pairs_q.json \
    --search-limit 10 --compound-pairs question)
(cd $B && python benchmarks/recall_changes_eval.py --bench locomo --split test --work-root $W --output $OUT/base_pairs_and.json \
    --search-limit 10 --compound-pairs and)
(cd $B && python benchmarks/historical_value_eval.py --work-root $(mktemp -d) --output $OUT/base_historical.json)
(cd $B && python benchmarks/historical_value_eval.py --work-root $(mktemp -d) --output $OUT/base_historical_ce.json --cross-rerank on)
# after (this tree, same stores): defaults and switches
python benchmarks/recall_changes_eval.py --bench locomo --split dev  --work-root $W --output $OUT/dev.json \
    --config default --config intent_off --config damp_off --config qe_off --config slots_off
python benchmarks/recall_changes_eval.py --bench locomo --split test --work-root $W --output $OUT/test.json --config default
python benchmarks/recall_changes_eval.py --bench locomo --split dev  --work-root $W --output $OUT/dev_ce.json \
    --cross-rerank on --config default
python benchmarks/recall_changes_eval.py --bench lme    --split dev  --work-root $W --output $OUT/lme.json \
    --config default --config intent_off --config damp_off --config slots_off
# compound pairs at K=10
python benchmarks/recall_changes_eval.py --bench locomo --split test --work-root $W --output $OUT/pairs_q.json \
    --search-limit 10 --compound-pairs question --config slots_off --config slots_on
python benchmarks/recall_changes_eval.py --bench locomo --split test --work-root $W --output $OUT/pairs_and.json \
    --search-limit 10 --compound-pairs and --config slots_off --config slots_on
# query_entity on copies with a proxy entity graph
for split in dev test; do
  python benchmarks/recall_changes_eval.py --bench locomo --split $split --work-root $W --output $OUT/kg_$split.json \
      --entity-graph --graph-root $W-kg --config qe_off --config qe_0.3 --config qe_0.5 --config qe_1
done
# historical values (own throwaway store)
python benchmarks/historical_value_eval.py --work-root $(mktemp -d) --output $OUT/historical.json
python benchmarks/historical_value_eval.py --work-root $(mktemp -d) --output $OUT/historical_ce.json --cross-rerank on
```

The LoCoMo and LongMemEval-S files go to `benchmarks/data/` as for the other runs. Summaries of every
run above (without per-question rows) are in [`raw/`](raw/). On macOS some of these runs ended with
`libc++abi: ... recursive_mutex lock failed` at interpreter exit, after their output was written; the
numbers are unaffected. The cause was onnxruntime's telemetry upload thread outliving interpreter
shutdown; `memory_core.fastembed_loader` now disables onnxruntime telemetry before the first model loads.
