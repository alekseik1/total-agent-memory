#!/usr/bin/env python3
"""LongMemEval-S QA accuracy under the official protocol.

Each question's haystack sits in its own project (`lme_<question_id>`), as
`benchmarks/longmemeval_bench.py --modes store` ingests it: one record per
session, prefixed with the session date. Retrieval goes through the product's
MCP dispatcher (`memory_recall`, context mode). An OpenAI model answers from
the retrieved sessions and the question date, and gpt-4o-2024-08-06 grades
the answer with the task-specific prompts of the official evaluator
(src/evaluation/evaluate_qa.py in xiaowu0162/LongMemEval, verbatim below).
Abstention questions (`_abs`) are graded on recognising that the question
cannot be answered.

Split: the 100 questions with the smallest SHA-256 of their id are the
development set; the other 400 are held out.

`--ingest round` stores each user/assistant round as its own record, the way
a client saves a conversation as it goes, with the session date and the
session id in `context`; context mode then adds neighbouring rounds.

    python benchmarks/longmemeval_qa.py --store DIR --split dev --ingest round --variant x --out OUT  # load
    python benchmarks/longmemeval_qa.py --store DIR --split dev --variant k5 --out OUT
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
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
DATASET = ROOT / "benchmarks" / "data" / "longmemeval_s.json"
DEFAULT_ANSWER_MODEL = "gpt-4.1-mini-2025-04-14"
JUDGE_MODEL = "gpt-4o-2024-08-06"
DEV_SIZE = 100
MAX_ANSWER_TOKENS = 600
# Reasoning comes before the <answer> tag; a short cap cut answers off mid-reasoning.
MAX_REASONING_TOKENS = 1500

_CORE = ("I will give you a question, a correct answer, and a response from a model. Please answer yes if the "
         "response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct "
         "answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the "
         "response only contains a subset of the information required by the answer, answer no. ")
_TAIL = "\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
JUDGE_TEMPLATES = {
    "single-session-user": _CORE + _TAIL,
    "single-session-assistant": _CORE + _TAIL,
    "multi-session": _CORE + _TAIL,
    "temporal-reasoning": _CORE + (
        "In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number "
        "of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer "
        "is 18), the model's response is still correct. ") + _TAIL,
    "knowledge-update": (
        "I will give you a question, a correct answer, and a response from a model. Please answer yes if the "
        "response contains the correct answer. Otherwise, answer no. If the response contains some previous "
        "information along with an updated answer, the response should be considered as correct as long as the "
        "updated answer is the required answer." + _TAIL),
    "single-session-preference": (
        "I will give you a question, a rubric for desired personalized response, and a response from a model. "
        "Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not "
        "need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes "
        "the user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\n"
        "Is the model response correct? Answer yes or no only."),
}
ABSTENTION_TEMPLATE = (
    "I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if "
    "the model correctly identifies the question as unanswerable. The model could say that the information is "
    "incomplete, or some other information is given but the asked information is not.\n\nQuestion: {}\n\n"
    "Explanation: {}\n\nModel Response: {}\n\nDoes the model correctly identify the question as unanswerable? "
    "Answer yes or no only.")


@dataclass(frozen=True)
class Variant:
    name: str
    limit: int
    context_chars: int
    reasoning: bool
    neighbors: int


def load_questions(split: str, limit: int) -> list[dict]:
    data = json.loads(DATASET.read_text())
    ranked = sorted(data, key=lambda entry: hashlib.sha256(entry["question_id"].encode()).hexdigest())
    chosen = {"dev": ranked[:DEV_SIZE], "test": ranked[DEV_SIZE:], "all": ranked}[split]
    questions = [{"question_id": entry["question_id"], "type": entry["question_type"],
                  "question": entry["question"], "gold": str(entry["answer"]),
                  "question_date": entry.get("question_date", ""),
                  "abstention": entry["question_id"].endswith("_abs")} for entry in chosen]
    return questions[:limit] if limit else questions


def rounds(session: list[dict]) -> list[str]:
    """User turns with the assistant replies that follow them."""
    grouped: list[list[str]] = []
    for turn in session:
        text = turn["content"].strip()
        if not text:
            continue
        if turn["role"] == "user" or not grouped:
            grouped.append([])
        grouped[-1].append(f"{turn['role']}: {text}")
    return ["\n".join(lines) for lines in grouped]


def ingest_rounds(srv, question_ids: set[str]) -> int:
    """Store every haystack round of the chosen questions; skip projects already loaded."""
    saved = 0
    for entry in json.loads(DATASET.read_text()):
        qid = entry["question_id"]
        if qid not in question_ids:
            continue
        project = f"lme_{qid}"
        if srv.store.db.execute("SELECT 1 FROM knowledge WHERE project=? LIMIT 1", (project,)).fetchone():
            continue
        for sid, date, session in zip(entry["haystack_session_ids"], entry["haystack_dates"],
                                      entry["haystack_sessions"], strict=True):
            session_key = f"lme__{project}__{sid}"
            srv.store.session_start(session_key, project=project)
            for index, text in enumerate(rounds(session)):
                srv.store.save_knowledge(sid=session_key, content=f"[{date}]\n{text}", ktype="fact",
                                         project=project, tags=[sid], skip_dedup=True, skip_quality=True,
                                         context=f"longmemeval session={sid} round={index}")
                saved += 1
    return saved


def open_store(store_dir: Path):
    os.environ["TAM_MEMORY_DIR"] = str(store_dir)
    os.environ["CLAUDE_MEMORY_DIR"] = str(store_dir)
    os.environ.setdefault("MEMORY_MODE", "fast")
    os.environ.setdefault("MEMORY_LLM_ENABLED", "false")
    sys.path.insert(0, str(ROOT / "src"))
    import server as srv

    srv.store = srv.Store()
    srv.recall = srv.Recall(srv.store)
    srv.SID = "longmemeval-qa"
    srv.BRANCH = ""
    return srv


def retrieve(srv, question: dict, variant: Variant) -> tuple[list[dict], str, float]:
    started = time.perf_counter()
    raw = asyncio.run(srv._do("memory_recall", {
        "query": question["question"], "project": f"lme_{question['question_id']}",
        "limit": variant.limit, "mode": "context", "neighbors": variant.neighbors,
        "context_max_chars": variant.context_chars, "detail": "full",
    }))
    elapsed = (time.perf_counter() - started) * 1000
    text = raw[0].text if isinstance(raw, list) else raw
    result = json.loads(text) if isinstance(text, str) else text
    return result.get("results", []), result.get("answer_guidance", ""), elapsed


def answer_messages(question: dict, evidence: list[dict], guidance: str, variant: Variant) -> list[dict]:
    history = "\n\n".join(f"### Session {n}\n{item['content']}" for n, item in enumerate(evidence, 1))
    instruction = guidance
    if variant.reasoning:
        instruction += (" First reason step by step inside <reasoning> tags: list the relevant sessions with their"
                        " dates, work out dates and counts against the current date, and prefer the latest value"
                        " when a fact changed. Then give the final answer inside <answer> tags.")
    user = f"History chats:\n\n{history}\n\nCurrent date: {question['question_date']}\nQuestion: {question['question']}"
    return [{"role": "system", "content": instruction}, {"role": "user", "content": user}]


def final_answer(text: str) -> str:
    if "<answer>" in text:
        return text.split("<answer>", 1)[1].split("</answer>", 1)[0].strip()
    if "</reasoning>" in text:
        return text.split("</reasoning>", 1)[1].strip()
    return text.strip()


def judge(client: OpenAIClient, cache: Cache, question: dict, answer: str) -> bool:
    template = ABSTENTION_TEMPLATE if question["abstention"] else JUDGE_TEMPLATES[question["type"]]
    prompt = template.format(question["question"], question["gold"], answer)
    key = digest("judge", JUDGE_MODEL, prompt)
    cached = cache.get(key)
    if cached is not None:
        return cached["correct"]
    text, _ = client.complete(JUDGE_MODEL, [{"role": "user", "content": prompt}], max_tokens=10)
    correct = "yes" in text.lower()
    cache.put(key, {"correct": correct, "label": text})
    return correct


def summarize(rows: list[dict]) -> dict:
    by_type: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        by_type["abstention" if row["abstention"] else row["type"]].append(row["correct"])
    total = [row["correct"] for row in rows]
    retrieval = sorted(row["retrieval_ms"] for row in rows)
    return {
        "n": len(rows),
        "accuracy": round(100 * sum(total) / len(total), 2) if total else 0.0,
        "by_type": {name: {"n": len(v), "accuracy": round(100 * sum(v) / len(v), 2)}
                    for name, v in sorted(by_type.items())},
        "errors": sum(1 for row in rows if row.get("error")),
        "context_tokens_mean": round(sum(row["context_tokens"] for row in rows) / len(rows)) if rows else 0,
        "retrieval_ms_p50": round(retrieval[len(retrieval) // 2], 1) if retrieval else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", type=Path, required=True, help="TAM memory dir holding the ingested haystacks")
    parser.add_argument("--split", choices=("dev", "test", "all"), default="dev")
    parser.add_argument("--variant", required=True, help="name for this configuration's output files")
    parser.add_argument("--limit", type=int, default=5, help="memory_recall limit (sessions)")
    parser.add_argument("--context-chars", type=int, default=32000)
    parser.add_argument("--reasoning", action="store_true", help="reason over sessions before answering")
    parser.add_argument("--neighbors", type=int, default=0, help="context-mode neighbor radius (rounds)")
    parser.add_argument("--ingest", choices=("round",), help="load the split's haystacks into --store and exit")
    parser.add_argument("--answer-model", default=DEFAULT_ANSWER_MODEL,
                        help="model that answers from the retrieved context")
    parser.add_argument("--questions", type=int, default=0, help="first N questions of the split (0 = all)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--budget-usd", type=float, default=5.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.ingest:
        srv = open_store(args.store)
        saved = ingest_rounds(srv, {q["question_id"] for q in load_questions(args.split, args.questions)})
        print(json.dumps({"ingested_records": saved, "store": str(args.store)}))
        return 0
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is required")
    variant = Variant(args.variant, args.limit, args.context_chars, args.reasoning, args.neighbors)
    args.out.mkdir(parents=True, exist_ok=True)
    answers = Cache(args.out / "answers-cache.jsonl")
    grades = Cache(args.out / "judge-cache.jsonl")
    client = OpenAIClient(api_key, args.budget_usd)
    answer_model = args.answer_model
    questions = load_questions(args.split, args.questions)
    srv = open_store(args.store)

    # Retrieval runs on this thread: the store's connection is not shared.
    prepared = [(question, *retrieve(srv, question, variant)) for question in questions]

    def evaluate(item: tuple[dict, list[dict], str, float]) -> dict:
        question, evidence, guidance, elapsed = item
        messages = answer_messages(question, evidence, guidance, variant)
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

    (args.out / f"{variant.name}-{args.split}.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    summary = {"variant": variant.__dict__, "split": args.split, "answer_model": answer_model,
               "judge_model": JUDGE_MODEL, "spent_usd_this_run": round(client.spent, 4), **summarize(rows)}
    (args.out / f"{variant.name}-{args.split}.summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
