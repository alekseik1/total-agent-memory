# LongMemEval-S after the 14.8.0 retrieval work (2026-10-08)

One decisive run on the held-out split, after every retrieval change of 14.8.0 and the relative-date
anchoring that followed it. The question it answers: did the retrieval work move the answer accuracy?

**It did not.** Held-out accuracy is 88.0, against 87.75 for 14.5.1 on the same 400 questions with
the same reader, judge and settings ([head-to-head-v14](../head-to-head-v14/RESULTS.md)). The
retrieval side is at 96-99% evidence coverage; 42 of the 48 wrong answers had every gold session in
the reader's context. The rest of the gap to the published leaders is on the reader.

## Setup

- Store: one project per question, one record per user/assistant round
  (`benchmarks/longmemeval_qa.py --ingest round`), 99,516 records for the 400 held-out questions,
  24,835 for the 100 dev questions. The splits are those of head-to-head-v14 (`splits/`).
- Retrieval: `memory_recall(mode="context", limit=20, neighbors=2, context_max_chars=48000)`, cross-encoder
  on, `as_of` = the question's date (relative phrases such as "last month" are counted from it).
- Reader gpt-4.1-mini-2025-04-14 with the reasoning instruction; judge gpt-4o-2024-08-06 with the
  official task-specific prompts. Context about 9,200 tokens per question.
- TAM at `d3c8901` (branch `feat/relative-dates`): 14.8.0 plus `as_of`.

## Held-out, 400 questions

| Type (n) | 14.5.1 | 14.8.0 + as_of |
|---|---:|---:|
| all (400) | 87.75 | **88.0** |
| abstention (25) | 88 | 84 |
| knowledge-update (62) | - | 95.2 |
| multi-session (96) | 82.3 | 80.2 |
| single-session-assistant (43) | - | 97.7 |
| single-session-preference (25) | 56 | 56.0 |
| single-session-user (53) | - | 100 |
| temporal-reasoning (96) | 87.5 | 89.6 |

Cost of the run: $1.71. Rows: [`raw/tam-14.9-mini-test.jsonl`](raw/tam-14.9-mini-test.jsonl).

### Where the 48 wrong answers come from

Each wrong answer was checked against the gold session ids: were they among the records the reader saw?

| cause | wrong answers | share |
|---|---:|---:|
| reader error, every gold session in context | 38 | 79% |
| abstention judgement (answered when the gold is "not enough information") | 4 | 8% |
| partial retrieval (some gold sessions in context) | 4 | 8% |
| retrieval miss (no gold session in context) | 2 | 4% |

The 38 reader errors: multi-session counting and arithmetic (17, "how many hours in total did I spend
driving", "what time did I go to bed the day before the appointment"), preference answers (10, the
judge expects the recommendation to draw on the stored preference), temporal arithmetic (8, "how many
days ago did I meet Emma"), knowledge updates (3). With perfect retrieval the same reader and prompt
would score about 89.5.

## Dev, 100 questions (seen data, used for the choices above)

| run | accuracy | notes |
|---|---:|---|
| gpt-4.1-mini, 14.5 code (2026-09-22) | 86 | from the head-to-head work |
| gpt-4.1-mini, 14.8.0 | 92.0 | preference 100, temporal 93.5 |
| gpt-4.1, 14.8.0 | 86.0 | the stronger reader is worse: more "not enough information", more date errors |
| gpt-4.1-mini, 14.8.0 + as_of | 91.0 | one flip inside the noise; the gold records moved up |

Retrieval coverage on dev (95 questions with evidence, cross-encoder on):

| | R@1 | R@5 | R@10 | gold session in context | every gold session |
|---|---:|---:|---:|---:|---:|
| 14.8.0 | 90.5 | 97.9 | 97.9 | 98.9 | 95.8 |
| + as_of | 92.6 | 98.9 | 98.9 | 98.9 | 95.8 |
| temporal-reasoning, 14.8.0 (31) | 80.6 | 93.5 | 93.5 | 96.8 | 87.1 |
| temporal-reasoning, + as_of | 87.1 | 96.8 | 96.8 | 96.8 | 87.1 |

Summaries: [`raw/`](raw/).

## What this means

- The published leaders (94-95) are not ahead on retrieval. On this harness the memory already hands
  the reader the evidence for 96-99% of the questions; the missing points are counting, date
  arithmetic, tailored recommendations and the decision to abstain.
- The next lever is the reader prompt and its reasoning format, measured with this same harness and
  the same judge, not another retrieval tier. A stronger reader model alone did not help (gpt-4.1
  scored below gpt-4.1-mini on dev).
- Retrieval work still pays where coverage is low: LoCoMo open-domain and multi-hop (R@10 0.40 and
  0.57 without the cross-encoder, [tier-ablation-v14](../tier-ablation-v14/RESULTS.md)) and the
  enterprise half of LongMemEval-V2.

## Reproduce

```bash
.venv/bin/python benchmarks/longmemeval_qa.py --store /tmp/tam/lme-test --split test --ingest round --variant x --out /tmp/tam/out
set -a; source ~/.config/tam-bench/openai.env; set +a
MEMORY_CROSS_RERANK=on MEMORY_ASYNC_ENRICHMENT=false .venv/bin/python benchmarks/longmemeval_qa.py --store /tmp/tam/lme-test --split test --variant tam-mini --limit 20 --neighbors 2 --context-chars 48000 --reasoning --budget-usd 8 --out /tmp/tam/out
```

Run the reader with nothing else touching the store: a run started while another process copied the
store returned an empty context for 87 of 100 questions (see the session notes of 2026-10-08).
