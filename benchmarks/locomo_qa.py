#!/usr/bin/env python3
"""LoCoMo QA accuracy under the protocol the public leaderboard uses.

Retrieval goes through the product's MCP dispatcher (`memory_recall`), so the
number measures what an agent gets from TAM. An OpenAI model answers from the
retrieved excerpts, and gpt-4o-mini grades the answer with the grading prompt
published by Zep and Mem0 (verbatim below), over the 1,540 questions of
categories 1-4. Category 5 (adversarial) is excluded, as every published
LoCoMo figure excludes it.

Conversations 0-2 are the development split; 3-9 are held out. Tune on dev,
report test and all.

The store must hold the LoCoMo turns as `benchmarks/locomo_bench_llm.py`
ingests them (project `locomo_<conversation index>`):

    python benchmarks/locomo_bench_llm.py --db-path DIR --limit-qa 1   # ingest only
    python benchmarks/locomo_qa.py --store DIR --split dev --variant context-k10 --out OUT

Answers and grades are cached in OUT by content hash, so a rerun only pays
for what changed. Spend is capped by --budget-usd.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _qa_common import BudgetExceeded, Cache, OpenAIClient, digest

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "benchmarks" / "data" / "locomo" / "data" / "locomo10.json"
DEFAULT_ANSWER_MODEL = "gpt-4.1-mini-2025-04-14"
JUDGE_MODEL = "gpt-4o-mini"
CATEGORIES = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop"}
SPLITS = {"dev": range(3), "test": range(3, 10), "all": range(10)}
MAX_ANSWER_TOKENS = 400
# Reasoning comes before the <answer> tag; a short cap cut answers off mid-reasoning.
MAX_REASONING_TOKENS = 1500

# Verbatim (apart from trailing whitespace) from Zep's published LoCoMo evaluation (the prompt Mem0 also uses),
# typo included, so scores are graded exactly as the leaderboard grades them.
JUDGE_SYSTEM = """
        You are an expert grader that determines if answers to questions match a gold standard answer
        """
JUDGE_PROMPT = """
    Your task is to label an answer to a question as ’CORRECT’ or ’WRONG’. You williolw23 be given the following data:
        (1) a question (posed by one user to another user),
        (2) a ’gold’ (ground truth) answer,
        (3) a generated answer
    which you will score as CORRECT/WRONG.

    The point of the question is to ask about something one user should know about the other user based on their prior conversations.
    The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
    Question: Do you remember what I got the last time I went to Hawaii?
    Gold answer: A shell necklace
    The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.

    For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

    Now it’s time for the real question:
    Question: {question}
    Gold answer: {gold_answer}
    Generated answer: {response}

    First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG.
    Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

    Just return the label CORRECT or WRONG in a json format with the key as "label".
    """


@dataclass(frozen=True)
class Variant:
    name: str
    limit: int
    neighbors: int
    context_chars: int
    reasoning: bool


def load_questions(split: str, limit: int) -> list[dict]:
    data = json.loads(DATASET.read_text())
    questions = []
    for index in SPLITS[split]:
        for qa in data[index]["qa"]:
            if qa.get("category") in CATEGORIES and "answer" in qa:
                questions.append({"conversation": index, "question": qa["question"],
                                  "gold": str(qa["answer"]), "category": qa["category"]})
    return questions[:limit] if limit else questions


def open_store(store_dir: Path):
    os.environ["TAM_MEMORY_DIR"] = str(store_dir)
    os.environ["CLAUDE_MEMORY_DIR"] = str(store_dir)
    os.environ.setdefault("MEMORY_MODE", "fast")
    os.environ.setdefault("MEMORY_LLM_ENABLED", "false")
    sys.path.insert(0, str(ROOT / "src"))
    import server as srv

    srv.store = srv.Store()
    srv.recall = srv.Recall(srv.store)
    srv.SID = "locomo-qa"
    srv.BRANCH = ""
    return srv


def retrieve(srv, question: dict, variant: Variant) -> tuple[list[dict], str, float]:
    started = time.perf_counter()
    raw = asyncio.run(srv._do("memory_recall", {
        "query": question["question"], "project": f"locomo_{question['conversation']}",
        "limit": variant.limit, "mode": "context", "neighbors": variant.neighbors,
        "context_max_chars": variant.context_chars, "detail": "full",
    }))
    elapsed = (time.perf_counter() - started) * 1000
    text = raw[0].text if isinstance(raw, list) else raw
    result = json.loads(text) if isinstance(text, str) else text
    return result.get("results", []), result.get("answer_guidance", ""), elapsed


def answer_messages(question: str, evidence: list[dict], guidance: str, variant: Variant) -> list[dict]:
    excerpts = "\n".join(f"[{item['id']}] {item['content']}" for item in evidence)
    instruction = guidance
    if variant.reasoning:
        instruction += (" First reason step by step over the excerpts inside <reasoning> tags: list the relevant"
                        " excerpts with their dates, resolve relative dates against the message date, and combine"
                        " facts. Then give the final answer inside <answer> tags.")
    user = f"Conversation excerpts:\n{excerpts}\n\nQuestion: {question}"
    return [{"role": "system", "content": instruction}, {"role": "user", "content": user}]


def final_answer(text: str) -> str:
    if "<answer>" in text:
        return text.split("<answer>", 1)[1].split("</answer>", 1)[0].strip()
    if "</reasoning>" in text:
        return text.split("</reasoning>", 1)[1].strip()
    return text.strip()


def judge(client: OpenAIClient, cache: Cache, question: dict, answer: str) -> bool:
    key = digest("judge", JUDGE_MODEL, question["question"], question["gold"], answer)
    cached = cache.get(key)
    if cached is not None:
        return cached["correct"]
    prompt = JUDGE_PROMPT.format(question=question["question"], gold_answer=question["gold"], response=answer)
    text, _ = client.complete(JUDGE_MODEL, [{"role": "system", "content": JUDGE_SYSTEM},
                                            {"role": "user", "content": prompt}], max_tokens=200, json_mode=True)
    try:
        label = str(json.loads(text).get("label", ""))
    except json.JSONDecodeError:
        label = text
    correct = label.strip().upper().startswith("CORRECT")
    cache.put(key, {"correct": correct, "label": label})
    return correct


def summarize(rows: list[dict]) -> dict:
    by_category: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        by_category[CATEGORIES[row["category"]]].append(row["correct"])
    total = [row["correct"] for row in rows]
    retrieval = sorted(row["retrieval_ms"] for row in rows)
    return {
        "n": len(rows),
        "accuracy": round(100 * sum(total) / len(total), 2) if total else 0.0,
        "by_category": {name: {"n": len(v), "accuracy": round(100 * sum(v) / len(v), 2)}
                        for name, v in sorted(by_category.items())},
        "errors": sum(1 for row in rows if row.get("error")),
        "context_tokens_mean": round(sum(row["context_tokens"] for row in rows) / len(rows)) if rows else 0,
        "retrieval_ms_p50": round(retrieval[len(retrieval) // 2], 1) if retrieval else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", type=Path, required=True, help="TAM memory dir holding the ingested LoCoMo turns")
    parser.add_argument("--split", choices=tuple(SPLITS), default="dev")
    parser.add_argument("--variant", required=True, help="name for this configuration's output files")
    parser.add_argument("--limit", type=int, default=10, help="memory_recall limit")
    parser.add_argument("--neighbors", type=int, default=1, help="context-mode neighbor radius")
    parser.add_argument("--context-chars", type=int, default=24000)
    parser.add_argument("--reasoning", action="store_true", help="reason over excerpts before answering")
    parser.add_argument("--answer-model", default=DEFAULT_ANSWER_MODEL,
                        help="model that answers from the retrieved context")
    parser.add_argument("--questions", type=int, default=0, help="first N questions of the split (0 = all)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--budget-usd", type=float, default=5.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is required")
    variant = Variant(args.variant, args.limit, args.neighbors, args.context_chars, args.reasoning)
    args.out.mkdir(parents=True, exist_ok=True)
    answers = Cache(args.out / "answers-cache.jsonl")
    grades = Cache(args.out / "judge-cache.jsonl")
    client = OpenAIClient(api_key, args.budget_usd)
    answer_model = args.answer_model
    questions = load_questions(args.split, args.questions)
    srv = open_store(args.store)

    # Retrieval runs on this thread: the store's connection is not shared.
    prepared = []
    for question in questions:
        evidence, guidance, elapsed = retrieve(srv, question, variant)
        prepared.append((question, evidence, guidance, elapsed))

    def evaluate(item: tuple[dict, list[dict], str, float]) -> dict:
        question, evidence, guidance, elapsed = item
        messages = answer_messages(question["question"], evidence, guidance, variant)
        max_tokens = MAX_REASONING_TOKENS if variant.reasoning else MAX_ANSWER_TOKENS
        key = digest("answer", answer_model, max_tokens, messages)
        row = {**question, "variant": variant.name, "retrieval_ms": round(elapsed, 2),
               "context_tokens": sum(len(item["content"]) for item in evidence) // 4,
               "evidence_ids": [item["id"] for item in evidence]}
        try:
            cached = answers.get(key)
            if cached is None or not cached["text"].strip():
                text, usage = client.complete(answer_model, messages, max_tokens=max_tokens)
                cached = {"text": text, "usage": usage}
                answers.put(key, cached)
            row["answer"] = final_answer(cached["text"])
            row["finish_truncated"] = "<reasoning>" in cached["text"] and "</reasoning>" not in cached["text"]
            row["correct"] = judge(client, grades, question, row["answer"])
        except BudgetExceeded:
            raise
        except RuntimeError as exc:
            row.update(answer="", correct=False, error=str(exc))
        return row

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(evaluate, prepared))

    out_rows = args.out / f"{variant.name}-{args.split}.jsonl"
    out_rows.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    summary = {"variant": variant.__dict__, "split": args.split, "answer_model": answer_model,
               "judge_model": JUDGE_MODEL, "spent_usd_this_run": round(client.spent, 4), **summarize(rows)}
    (args.out / f"{variant.name}-{args.split}.summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
