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

## Reproduce

```
export SSL_CERT_FILE=$(python -c "import certifi; print(certifi.where())")
python benchmarks/knowledge_update_eval.py --suite synthetic --out <dir>
for i in 0 1 2 3; do python benchmarks/knowledge_update_eval.py --suite longmemeval --shard $i/4 --out <dir> & done
python benchmarks/knowledge_update_eval.py --suite longmemeval --summarize --out <dir>
python benchmarks/knowledge_update_eval.py --suite longmemeval --out <dir>/control \
  --types multi-session,temporal-reasoning,single-session-user,single-session-assistant,single-session-preference --per-type 10
```

Per-question answers, verifier reasons and judge ledgers are in the version directories.
