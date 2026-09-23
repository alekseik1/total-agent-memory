# LoCoMo and LongMemEval QA accuracy, under the protocols the public numbers use

> The 14.5.0 results, graded next to Mem0 Platform's published answers, are in [head-to-head-v14](../head-to-head-v14/RESULTS.md).

2026-09-22. Retrieval runs through the product (`memory_recall` over the MCP dispatcher). An OpenAI model answers from what it returns, and each benchmark's published grader scores the answer. Every number here is one run.

## Protocol

| | LoCoMo | LongMemEval-S |
|---|---|---|
| Questions | 1,540 of categories 1-4 (adversarial excluded, as every published figure excludes it) | 500 |
| Store | one project per conversation; each dialogue turn is a record, prefixed with its timestamp | one project per question; each user/assistant round of the haystack is a record, prefixed with the session date |
| Retrieval | `memory_recall`, context mode, limit 50, neighbour radius 1 | `memory_recall`, context mode, limit 20, neighbour radius 2 |
| Answering | gpt-4.1-mini-2025-04-14, temperature 0, the product's own `answer_guidance` plus an instruction to reason before answering | same |
| Grading | the lenient LoCoMo grader Zep and Mem0 publish, verbatim, gpt-4o-mini, temperature 0 | the task-specific prompts of the official evaluator (`src/evaluation/evaluate_qa.py`), verbatim, gpt-4o-2024-08-06 |
| Development split | conversations 0-2 (385 questions) | the 100 questions with the smallest SHA-256 of their id |
| Held-out split | conversations 3-9 (1,155 questions) | the other 400 questions |

Configurations were chosen on the development split; every number below is the held-out split. Harnesses: `benchmarks/locomo_qa.py`, `benchmarks/longmemeval_qa.py`. Each run starts from a fresh copy of the store, because retrieval records usage.

## Results, held-out

| | 14.3.1 | this change | paired |
|---|---:|---:|---|
| **LoCoMo**, 1,155 questions | 85.63 | **87.01** | +1.40 pts, 95% CI [-0.09; +2.88], 46 wins / 30 losses, sign test p = 0.085 |
| multi-hop (208) | 75.00 | 78.37 | |
| temporal (231) | 83.12 | 84.42 | |
| single-hop (641) | 92.51 | 93.29 | |
| open-domain (75) | 64.00 | 65.33 | |
| **LongMemEval-S**, 400 questions | 84.50 | **87.75** | +3.25 pts, 95% CI [+0.50; +6.25], 24 wins / 11 losses, sign test p = 0.041 |
| single-session-user | 98.11 | 100.00 | |
| knowledge-update | 95.16 | 96.77 | |
| temporal-reasoning | 82.29 | 88.54 | |
| single-session-assistant | 88.37 | 88.37 | |
| abstention | 84.00 | 84.00 | |
| multi-session | 78.12 | 81.25 | |
| single-session-preference | 56.00 | 64.00 | |

On LongMemEval the gain is significant at the 5% level; on LoCoMo it is not — the interval includes zero.

The development split reads higher than the held-out one (LoCoMo 92.21 vs 87.01): conversations 0-2 are easier, and the configuration was picked on them. That difference is what a development split is for.

## What changed

1. **Cross-encoder over the fused window.** Evidence recall@10 on the development split: 64.4% → 75.5%. Same candidates, different order.
2. **`recall_count` dropped from ranking.** Measured separately on the development split, sequentially over its 385 questions: 66.8% → 62.1% evidence recall@10 *with* the old feedback, on a store that started clean. The loop cost 4.7 points within a single pass.

## Against published claims

| System | LoCoMo | LongMemEval | Answering model |
|---|---:|---:|---|
| total-agent-memory (this run, held-out) | 87.0 | 87.8 | gpt-4.1-mini |
| Mem0 (self-reported, 2026) | 92.5 | 94.4 | not stated |
| ByteRover 2.0 (self-reported, 2026) | 92.2 | 92.8 | Gemini 3 Pro / Flash |
| Zep (self-reported) | 94.7 claimed | 71.2 | not stated |
| Mem0 as measured by ByteRover | 66.9 | — | Gemini 3 Flash |

These are not like-for-like. The answering model, the grader and any post-processing differ, and the same system scores 66.9 or 92.5 depending on who runs it. What this table does show is that the published leaders sit about 5 points above this run. Part of that gap is retrieval — evidence recall@10 is 75.5%, so the answer is missing from the context of roughly a quarter of the questions — and part is the answer itself, as the reader comparison below shows.

**A stronger answering model did not help.** On 400 held-out LoCoMo questions with identical memory and configuration, gpt-4.1 scored 82.25 against gpt-4.1-mini's 84.25. It refused twice as often (24 vs 11 "not enough information") and hedged where the gold answer is exact ("at least three hikes" for "four"), which the lenient grader counts as wrong. The remaining gap is in how the answer is written, not in what the memory returns.

## Files

`raw/` holds the per-question answers and the summary of every cell: `locomo-answers-new.jsonl` and `locomo-answers-14.3.1.jsonl` (held-out, 1,155 each), `longmemeval-answers-*.jsonl` (400 each), and `locomo-answers-reader-gpt41.jsonl` (the 400-question reader comparison). Each row carries the question, the retrieved record ids, the answer and the grade.

## Limits

- One run per cell, temperature 0 for answering; the graders are LLMs and were not validated against human labels here.
- LoCoMo's open-domain category has 75 questions in the held-out split and 21 in development, so its per-category numbers move several points on noise.
- The 14.3.1 column runs the released code against the same stores and the same prompts; it is not the published 14.3.1 figure.
- The LongMemEval store keeps each round as a record. Storing whole sessions instead scored 80.0 against 86.0 on the development split, so ingest granularity matters as much as retrieval here.
- Cost of both held-out columns plus the reader comparison: about $12 of OpenAI usage.
