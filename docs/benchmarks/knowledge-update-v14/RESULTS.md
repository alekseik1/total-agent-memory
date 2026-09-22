# Knowledge update: answering with the current value of a fact

2026-09-21. Question: memory holds "Mary loves red" and, saved later, "Mary no longer
likes red; she has fallen for green". Does `memory_answer` say *green*?

## Setup

- Harness: `benchmarks/knowledge_update_eval.py`. It runs the shipping in-process path
  (`Store.save_knowledge` → `Recall.search` → `answer_endpoint.answer_response`) on a
  throwaway database. Every version runs from a frozen copy of its source tree.
- Reader, verifier, contradiction scorer: `claude-haiku-4-5` (the provider of the
  maintainer's installation). Judge: `gpt-4.1-mini-2025-04-14` with the v14 QA judge prompt,
  behind a spending ledger. Total judge spend: $0.03.
- **Synthetic**: `benchmarks/data/knowledge_update_scenarios.json`. 30 scenarios (ru + en):
  16 updates, 4 retractions, 6 later records about the past, 4 later records about another
  person. They are mixed with 300 deterministic noise records per language about other
  people, and records are stamped months apart. A refusal counts as wrong, because every
  reference states a fact.
- **LongMemEval KU**: all 78 knowledge-update questions of LongMemEval-S. Each question gets
  its own project, and each session is stamped with its haystack date.
- **Control**: the first 10 questions of each of the other five LongMemEval-S categories, to
  catch regressions. Reader and verifier prompts are shared with every question.

## Results (paired, same questions)

| Suite | 14.1.0 | v4 (this change) | wins / losses | sign test p |
|---|---|---|---|---|
| LongMemEval knowledge-update | 12 / 78 | **35 / 78** | +24 / −1 | 1.5·10⁻⁶ |
| Synthetic ru + en | 8 / 30 | **22 / 30** | +14 / −0 | 1.2·10⁻⁴ |
| Control (5 other categories) | 11 / 50 | 16 / 50 | +6 / −1 | 0.12 |

Intermediate versions, in the order they were built:

| Version | Change | KU | Synthetic |
|---|---|---|---|
| v1 | reader and verifier see each record's date; hard contradiction resolved by the reader instead of refused; contradiction scorer sees the question | — | 19 |
| v2 | + Russian stemming in the lexical recall tier | 18 | 18 |
| v3 | + inflection-aware claim grounding (Маша ~ Маше); a value stays current until a later record changes it | 36 | 20 |
| v4 | + verifier rules for records that recall the past and for retractions; canonical UTC timestamps | 35 | 22 |

## Why 14.1.0 failed

1. **The reader never saw dates.** It received `{id, content}` only, so "loves red" and
   "now green" were equal and it often cited the older one. The verifier (which also lacked
   dates, but saw the in-text dates of LongMemEval sessions) then rejected the stale answer,
   giving *Not enough information*. Example 6a1eabeb: the draft said 27:12, and a later
   session says 25:50.
2. **A hard contradiction refused before reading.** In a realistic database the veto also
   fired on conflicts about *other* people ("Fedor loves maroon" vs "Fedor loves white"
   blocked a question about Mary).
3. **Russian morphology.** FTS5 `unicode61` matches exact word forms, so the question's
   "Маша" never found the record's "Маше". Claim grounding required the literal name, so it
   rejected a correct quote.
4. **The verifier rejected values stated before the question date** ("no newer record
   confirms 30 videos").
5. **Timestamps came in three formats** (`…Z`, `…+00:00`, local-time `YYYY-MM-DD HH:MM:SS`).
   The zone-less ones were local time stored as if zone-free, so string order did not match
   time order.

## Remaining failures (v4)

- Later records that recall the past ("As a child, Chloe adored pink"): the reader answers
  correctly ("blue"), but the Haiku verifier still rejects it, so the answer is a refusal
  rather than a wrong fact. The prompt examples deliberately avoid the scenarios' wording.
- Retractions without a replacement ("Alice can't stand teal now"): the verifier sometimes
  demands a current favourite that memory does not hold.
- Control: 4 wrong answers against 1 in 14.1.0, because fewer refusals mean some answers
  are now wrong instead of withheld. Net correct is +5.

## Jev as the contradiction scorer (14.3.0)

`MEMORY_CONTRADICTION_SCORER=jev` sends every (supporting, opposing) pair of the negative
pass as one `noul` question, all in a single request to TypeSafe's System One API
(`jev-latest`, resolved to `jev-1.13.0`). The reader and the verifier stay on Claude
Haiku 4.5. Both arms ran the same code (v5 directories) on the same questions; only the
scorer differs.

| Suite | Haiku scorer | Jev scorer | wins / losses | p |
|---|---|---|---|---|
| LongMemEval knowledge-update | 36 / 78 | 35 / 78 | +1 / −2 | 1.0 |
| Synthetic ru + en | 21 / 30 | 21 / 30 | +1 / −1 | 1.0 |
| Control (5 other categories) | 16 / 50 | 15 / 50 | +3 / −4 | 1.0 |

Accuracy is unchanged within noise. What changes is the cost of the pass:

| Median per question | Haiku scorer | Jev scorer |
|---|---|---|
| Negative pass (inversion + search + scoring), knowledge-update | 3.1 s | 1.9 s |
| Whole `memory_answer`, knowledge-update | 8.7 s | 7.7 s |
| Whole `memory_answer`, control | 10.0 s | 7.7 s |

Jev billed 911,226 input tokens for all 78 knowledge-update questions: $0.038 at
$0.042 per million (output tokens are free). Haiku 4.5 input costs $1 per million, 24×
more per token; its token count for the same step was not recorded, so no dollar
comparison is claimed for it. Jev declared a hard contradiction less often (6 vs 24 times
on knowledge-update) without losing accuracy. In a direct probe it scored a conflict about
another person ("Fedor loves maroon / white", question about Masha) at 0.15, a later
recollection of the past at 0.12, and real updates at 0.87.

If the Jev call fails, the negative pass reports *no contradiction* and the answer
proceeds; a missing or malformed `TYPESAFE_API_KEY` fails at startup of the call without
echoing the key.

## Reproduce

```
export SSL_CERT_FILE=$(python -c "import certifi; print(certifi.where())")
python benchmarks/knowledge_update_eval.py --suite synthetic --out <dir>
for i in 0 1 2 3; do python benchmarks/knowledge_update_eval.py --suite longmemeval --shard $i/4 --out <dir> & done
python benchmarks/knowledge_update_eval.py --suite longmemeval --summarize --out <dir>
python benchmarks/knowledge_update_eval.py --suite longmemeval --out <dir>/control \
  --types multi-session,temporal-reasoning,single-session-user,single-session-assistant,single-session-preference --per-type 10
```

For the scorer comparison, prefix both commands with `MEMORY_CONTRADICTION_SCORER=jev`
(the harness reads `TYPESAFE_API_KEY` from the environment).

Per-question answers, verifier reasons, per-question contradiction-pass timings and judge
ledgers are in the version directories.
