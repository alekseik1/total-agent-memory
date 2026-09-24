# TAM against Mem0 Platform on the same questions, graded by the same judge per column

2026-09-23, revised 2026-09-24 after external review (see *Revisions*). Mem0 publishes the per-question answers behind its LoCoMo (92.5) and LongMemEval (94.4) figures. This report grades those answers and TAM's answers to the same held-out questions. Within each grading configuration, both systems' saved answers are evaluated by the same model and prompt, so the difference between the systems within a column is not a difference between graders. The two LoCoMo configurations share a judge model and differ in the prompt; the two LongMemEval configurations differ in both judge model and rubric.

## Result

LoCoMo: 1,144 questions of conversations 3-9 (categories 1-4; the 11 of 1,155 that repeat a question text are dropped). LongMemEval-S: the 400 held-out questions of `benchmarks/longmemeval_qa.py`. Accuracy in %, in brackets the paired difference to Mem0 on the same questions with its sign-test p.

| System (answering model) | LoCoMo, published judge | LoCoMo, Mem0 judge | LongMemEval, official judge | LongMemEval, Mem0 judge |
|---|---:|---:|---:|---:|
| **Mem0 Platform** (gpt-5, top 200 memories) | 88.46 | 94.32 | 91.00 | 91.75 |
| TAM, English embedding preset (gpt-4.1-mini) | 88.02 (−0.44, p 0.74) | **94.32** (0.00, p 1.00) | — | — |
| TAM, English embedding preset (gpt-5) | 86.54 (−1.92, p 0.10) | 94.23 (−0.09, p 1.00) | — | — |
| TAM, default configuration (gpt-4.1-mini) | 87.50 (−0.96, p 0.42) | 92.57 (−1.75, p 0.06) | 87.50 (−3.50, p 0.06) | 88.25 (−3.50, p 0.05) |
| TAM, default configuration (gpt-5) | — | — | **92.25** (+1.25, p 0.50) | 90.75 (−1.00, p 0.64) |

What this supports:

- **LoCoMo: no statistically detected difference.** With the English embedding preset and gpt-4.1-mini answering, TAM scores within half a point of Mem0 Platform with gpt-5 answering under both judges; no difference reaches significance on these 1,144 questions.
- **LongMemEval: no statistically detected difference at the same answering model.** With gpt-5 answering from TAM's context, TAM is correct on 369 of 400 against Mem0's 364 under the official judge (TAM-only / Mem0-only correct 20 / 15, exact two-sided sign test p 0.50) and on 363 against 367 under Mem0's judge (18 / 22, p 0.64). By question type (table below), TAM is ahead on preference questions (23 vs 19 of 25) and non-abstention multi-session questions (83 vs 79 of 96), behind on non-abstention knowledge updates (58 vs 60 of 62). With gpt-4.1-mini answering, TAM is 3.5 points behind (p 0.05-0.06).
- TAM retrieves locally, with no LLM call when it writes or searches. Mem0 Platform extracts memories with an LLM on every write.

It does not support "first place" for either system, and it does not demonstrate equivalence: on these questions no difference between TAM and Mem0 Platform at the same answering model is statistically detected, on either benchmark or under either judge, and a sample of this size cannot rule out differences of a few points.

LongMemEval held-out questions by type, gpt-5 answering (correct / questions; the 25 abstention questions, whose ids end in `_abs`, are counted separately, as the evaluator reports them):

| Question type | Questions | Mem0, official judge | TAM, official judge | Mem0, Mem0 judge | TAM, Mem0 judge |
|---|---:|---:|---:|---:|---:|
| knowledge-update | 62 | 60 | 58 | 60 | 56 |
| multi-session | 96 | 79 | 83 | 81 | 79 |
| single-session-assistant | 43 | 42 | 42 | 43 | 43 |
| single-session-preference | 25 | 19 | 23 | 22 | 25 |
| single-session-user | 53 | 52 | 52 | 52 | 52 |
| temporal-reasoning | 96 | 89 | 88 | 87 | 85 |
| abstention (all types) | 25 | 23 | 23 | 22 | 23 |
| **total** | **400** | **364** | **369** | **367** | **363** |

Judges: the published LoCoMo judge is the prompt Zep and Mem0 published (verbatim in `benchmarks/locomo_qa.py`), run on gpt-4o-mini; the official LongMemEval judge is the evaluator's task-specific prompt set (verbatim in `benchmarks/longmemeval_qa.py`), run on gpt-4o-2024-08-06. The Mem0 judges are the prompts of `mem0ai/memory-benchmarks` at commit `4b61c5d`, run on gpt-4o-mini. Both systems' answers go through the identical calls. Re-running a judge moves a cell by up to ±0.3 points. The paired comparison within a column holds; the difference between the two LongMemEval columns changes the judge model and the rubric at once, so it cannot isolate the effect of the prompt (`judge_models` in `raw/crossgrade-lme.json`).

## Protocol

- **Splits.** LongMemEval-S: the 100 questions with the smallest SHA-256 of their `question_id` are the development split, the other 400 are reported. The exact lists are `splits/longmemeval-dev-ids.txt` and `splits/longmemeval-heldout-ids.txt`; the development split has 11 knowledge-update, 27 multi-session, 13 single-session-assistant, 5 single-session-preference, 12 single-session-user and 32 temporal-reasoning questions. LoCoMo: conversations 0-2 (385 questions) are development, conversations 3-9 are reported; the 11 question texts that occur twice there are graded once (`splits/locomo-heldout-dropped-duplicates.txt`). No full-500 or full-LoCoMo score is reported here; one would include development questions.
- **Tuning history.** Changes were chosen on the development splits, but the configuration was not frozen before the reported questions were first scored. The reported LongMemEval questions were scored with TAM's own harness judge on 2026-09-22 (14.3.1 against the 14.4.0 candidate, and a gpt-4.1 reader experiment) and on the morning of 2026-09-23 after that day's first changes (neighbour-aware re-ranking, relative dates, answer guidance); rank-weighted context packing was added later that day, chosen on the development split, and the columns above were then produced. The reported split therefore measures a configuration chosen on development data by an author who had seen earlier scores on the reported split; a deterministic split alone does not establish the absence of tuning on it.
- **Ingest.** LongMemEval stores each user/assistant round as its own record, with speaker labels kept, the session date and the session id; the LongMemEval paper considers round-level decomposition in §3.1 and §5.2. LoCoMo stores each turn as a record, prefixed with its session date and speaker.
- **Retrieval into the reader's context.** `memory_recall(mode="context")`: LongMemEval takes the top 20 rounds and adds 2 neighbouring rounds of the same session on each side (`--limit 20 --neighbors 2`) within a 48,000-character budget shared by rank; LoCoMo takes the top 50 turns with 1 neighbour on each side within 60,000 characters. The cross-encoder re-ranks the fused candidates in both. The context averaged about 9,000 tokens on LongMemEval and 3,000 on LoCoMo.

## How Mem0's published numbers were produced

Everything below is in `mem0ai/memory-benchmarks` (Apache-2.0); commit hashes refer to that repository.

1. **Managed platform, not the open-source package.** Mem0's documentation: "Scores reflect Mem0's managed platform, which includes proprietary optimizations not available in the open-source SDK."
2. **gpt-5 answers and gpt-5 judges**, from the top 200 retrieved memories (`results/platform/*.json`, `metadata`). The TAM rows above answer from 50 LoCoMo turns or 20 LongMemEval rounds with their session neighbours, about 3,000 and 9,000 tokens.
3. **A more lenient judge than the one the earlier public numbers used.** The LoCoMo judge counts an answer correct when it contains one item of a list, when a date is within 14 days, when a duration is within 50%, or when it names "the same referent" (`benchmarks/locomo/prompts.py`; the date and duration rule arrived in `edcd6f1`, 2026-04-09, with the results). The LongMemEval judge tells the grader "You have a tendency to say 'no' too quickly … When in doubt, lean toward 'yes'". On the same answers, the Mem0 LoCoMo judge scores 5-6 points higher than the published one.
4. **156 LoCoMo questions were re-run and merged** into the published file (`metadata.merged_from_questions`). 26 of the merged answers are correct, against 91% across the file.
5. **The LoCoMo answer prompt carries hints that match individual LoCoMo gold answers.** Since the first commit (`7ba1bd3`, 2026-03-30), before the published run (2026-04-06):

   | Line in `ANSWER_GENERATION_PROMPT` | LoCoMo question and gold answer |
   |---|---|
   | "Memory shows store with a lot of working people -> store employs a lot of people" | conv 9: "Does Dave's shop employ a lot of people?" — *Yes* |
   | "a game exclusive to one platform implies ownership of that platform" | conv 3: "What Console does Nate own?" — *A Nintendo Switch; since the game "Xenoblade 2" is made for this console* |
   | "An unnamed company deal can be linked to a previously expressed brand preference" | conv 4: "Which outdoor gear company likely signed up John for an endorsement deal?" — *Under Armour* |
   | "you may name it (e.g., "Eternal Sunshine of the Spotless Mind")" | conv 3: "What is one of Joanna's favorite movies?" — *Eternal Sunshine of the Spotless Mind* |
   | "the nearby lake IS Lake Tahoe" | conv 8: "Where did Sam and his mate plan to try kayaking?" — *Lake Tahoe* |
   | "All events occurred in 2022-2024. Never output 2025 or 2026." | the date range of the LoCoMo conversations |

   The prompt also forbids answering "not specified" or "not mentioned", which suits LoCoMo's categories 1-4 and would be wrong on its adversarial category 5, which every published figure leaves out.

TAM's answers here come from the product's own answer guidance (`memory_recall(mode="context")` returns it as `answer_guidance`), which is written for any conversation and tells the reader to say "Not enough information" when a premise is missing. With gpt-5, that instruction made TAM refuse 69 LoCoMo questions against 32 with gpt-4.1-mini, which is most of why gpt-5 did not score higher.

## Reproduce

Needs Python 3.11+, this repository, an OpenAI key (about $40 for every cell below, $6 without the gpt-5 cells), and about 6 hours of machine time (the LongMemEval stores take most of it).

```bash
# 0. Setup
git clone https://github.com/vbcherepanov/total-agent-memory && cd total-agent-memory
python -m venv .venv && . .venv/bin/activate && pip install -e .
git clone https://github.com/snap-research/locomo benchmarks/data/locomo       # data/locomo10.json
curl -L -o benchmarks/data/longmemeval_s.json \
  https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json   # 277,383,467 bytes
git clone https://github.com/mem0ai/memory-benchmarks /tmp/mem0-mb
git -C /tmp/mem0-mb checkout 4b61c5d31b9c668a12b4f5e78064248a02c82d2b
export OPENAI_API_KEY=...

# 1. Stores: LoCoMo turns (5 min each), LongMemEval rounds (about 3 h for the 400 held-out questions)
python benchmarks/locomo_bench_llm.py --db-path /tmp/tam/locomo --limit-qa 0
MEMORY_TEXT_EMBED_MODEL=BAAI/bge-base-en-v1.5 \
  python benchmarks/locomo_bench_llm.py --db-path /tmp/tam/locomo-en --limit-qa 0
python benchmarks/longmemeval_qa.py --store /tmp/tam/lme --split test --ingest round --variant x --out /tmp/tam/out

# 2. Answers (each run works on its own copy: recall writes usage counters)
cp -R /tmp/tam/locomo-en /tmp/tam/run-en
MEMORY_TEXT_EMBED_MODEL=BAAI/bge-base-en-v1.5 MEMORY_CROSS_RERANK=on \
  python benchmarks/locomo_qa.py --store /tmp/tam/run-en --split test --variant tam-en \
  --limit 50 --neighbors 1 --context-chars 60000 --reasoning --out /tmp/tam/out        # ~$2.5
cp -R /tmp/tam/lme /tmp/tam/run-lme
MEMORY_CROSS_RERANK=on python benchmarks/longmemeval_qa.py --store /tmp/tam/run-lme --split test \
  --variant tam-mini --limit 20 --neighbors 2 --context-chars 48000 --reasoning --out /tmp/tam/out   # ~$1.7
# gpt-5 cells: add --answer-model gpt-5-2025-08-07 --budget-usd 20 and drop --reasoning (~$15 LoCoMo, ~$9 LongMemEval)

# 3. Grade both systems with both judges
python benchmarks/crossgrade_mem0.py --bench locomo --mem0-repo /tmp/mem0-mb \
  --tam tam-en=/tmp/tam/out/tam-en-test.jsonl --out /tmp/tam/crossgrade                  # ~$0.7
python benchmarks/crossgrade_mem0.py --bench lme --mem0-repo /tmp/mem0-mb --mem0-judge-model gpt-4o-mini \
  --tam tam-mini=/tmp/tam/out/tam-mini-test.jsonl --out /tmp/tam/crossgrade             # ~$0.6

# 4. Retrieval alone, no API key
python benchmarks/retrieval_eval.py --bench locomo --store /tmp/tam/locomo --split test
```

`crossgrade_mem0.py` writes `crossgrade-<bench>.json` (every cell, per-category accuracy, paired statistics) and `crossgrade-<bench>-marks.jsonl` (each question's verdict per system and judge).

## Files

`raw/` holds TAM's answers for every cell (`locomo-*.jsonl`, `longmemeval-*.jsonl`: question, retrieved record ids, answer, grade under the harness judge) and the cross-grading output (`crossgrade-*.json`, `crossgrade-*-marks.jsonl`). Mem0's answers are not copied; they are in Mem0's repository at the commit above.

## Limits

- One answering run per cell. The judges are LLMs; neither judge was validated against human labels here.
- Mem0's answers are its published ones: its retrieval was not re-run, so the comparison takes its best reported run, re-runs included.
- The English preset uses an English-only embedding model (`BAAI/bge-base-en-v1.5`); TAM's default is multilingual, and that default is the row below it.
- No other system was graded. Zep's and ByteRover's per-question answers are not published.

## Revisions

2026-09-24, after an external review that recalculated every LongMemEval accuracy cell, paired win/loss count and exact sign-test p from the saved verdicts at `4e8c81a` and found them in agreement:

- The introduction said one judge model and two prompts; that holds for LoCoMo only. The LongMemEval configurations differ in judge model and rubric, as `judge_models` in the raw output already recorded.
- "A tie" became "no statistically detected difference": the test does not demonstrate equivalence.
- Added the exact split id lists, the per-type table with denominators, the tuning history, and the ingest and context-budget description. No number changed.

External saved-verdict review and reporting feedback: Youngseok Oh (@YS-OH-CORE), with substantial technical analysis, code, and execution assistance from Zero (ChatGPT).

The check covers the six LongMemEval accuracy cells and four paired comparisons from the saved verdicts. It does not certify retrieval, answer generation, fresh judge decisions, tuning independence, overall system quality, or institutional endorsement. The tuning history above was read by the reviewers, not independently audited.
