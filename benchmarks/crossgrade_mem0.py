#!/usr/bin/env python3
"""Grade TAM's answers and Mem0 Platform's published answers with the same judges.

Mem0 publishes the per-question answers behind its LoCoMo and LongMemEval
figures (github.com/mem0ai/memory-benchmarks, results/platform/*.json: gpt-5
answering from the top 200 memories, gpt-5 judging). This script takes those
answers and the answers the TAM harnesses wrote (`locomo_qa.py`,
`longmemeval_qa.py`), restricts both to the same held-out questions, and grades
every answer under two judges; within a judge, both systems' answers get the
same model and prompt. `published` keeps each benchmark's own judge model
(gpt-4o-mini for LoCoMo, gpt-4o-2024-08-06 for LongMemEval); `mem0` runs on
--mem0-judge-model, which defaults to the same model. The report's LongMemEval
run set it to gpt-4o-mini, so there the two judges differ in model and rubric:

* `published` — the judge the public numbers used before 2026: for LoCoMo the
  prompt Zep and Mem0 published (verbatim in locomo_qa.py), for LongMemEval
  the official evaluator's task-specific prompts (verbatim in longmemeval_qa.py);
* `mem0` — the judge prompt of Mem0's current benchmark repository, imported
  from the checkout given with --mem0-repo.

Each (system, judge) cell gets its accuracy, per-category accuracy, and a
paired comparison with Mem0 on the same questions: difference, 95% bootstrap
interval, wins/losses and a two-sided sign test.

    git clone https://github.com/mem0ai/memory-benchmarks.git /tmp/mem0-mb
    git -C /tmp/mem0-mb checkout 4b61c5d31b9c668a12b4f5e78064248a02c82d2b
    python benchmarks/crossgrade_mem0.py --bench locomo --mem0-repo /tmp/mem0-mb \\
        --tam "tam-en=OUT/final-en-test.jsonl" --out OUT/crossgrade
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import locomo_qa
import longmemeval_qa
from _qa_common import Cache, OpenAIClient, digest

MEM0_LABEL = "mem0-platform"
MEM0_RESULTS = {"locomo": "results/platform/locomo_results.json",
                "lme": "results/platform/longmemeval_results.json"}
MEM0_PROMPTS = {"locomo": "benchmarks/locomo/prompts.py", "lme": "benchmarks/longmemeval/prompts.py"}
ANSWER_PREFIX = "Generated answer:"
BOOTSTRAP_SAMPLES = 4000
MAX_JUDGE_TOKENS = {"published": 200, "mem0": 1200}


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def question_key(bench: str, row: dict) -> str:
    if bench == "locomo":
        return f"{row['conversation']}|{row['question']}"
    return row["question_id"]


def held_out(bench: str) -> dict[str, dict]:
    if bench == "locomo":
        return {question_key(bench, q): q for q in locomo_qa.load_questions("test", 0)}
    return {question_key(bench, q): q for q in longmemeval_qa.load_questions("test", 0)}


def mem0_answers(bench: str, repo: Path) -> dict[str, str]:
    rows = json.loads((repo / MEM0_RESULTS[bench]).read_text())["evaluations"]
    if bench == "locomo":
        return {f"{row['conversation_idx']}|{row['question']}": row["cutoff_results"]["top_200"]["generated_answer"]
                for row in rows}
    answers = {}
    for row in rows:
        reason = row["cutoff_results"]["top_200"]["reason"]
        if not reason.startswith(ANSWER_PREFIX):
            raise ValueError(f"{row['question_id']}: unexpected reason format {reason[:40]!r}")
        answers[row["question_id"]] = reason[len(ANSWER_PREFIX):].strip()
    return answers


def judge_messages(bench: str, judge: str, question: dict, answer: str, prompts) -> list[dict]:
    if bench == "locomo":
        if judge == "published":
            return [{"role": "system", "content": locomo_qa.JUDGE_SYSTEM},
                    {"role": "user", "content": locomo_qa.JUDGE_PROMPT.format(
                        question=question["question"], gold_answer=question["gold"], response=answer)}]
        gold = prompts.preprocess_answer(question["category"], question["gold"])
        return [{"role": "system", "content": prompts.JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": prompts.get_judge_prompt(
                    question["category"], question["question"], gold, answer)}]
    if judge == "published":
        template = (longmemeval_qa.ABSTENTION_TEMPLATE if question["abstention"]
                    else longmemeval_qa.JUDGE_TEMPLATES[question["type"]])
        return [{"role": "user", "content": template.format(question["question"], question["gold"], answer)}]
    return [{"role": "user", "content": prompts.get_judge_prompt(
        question["type"], question["question_id"], question["question"], question["gold"], answer,
        question_date=question["question_date"])}]


def verdict(bench: str, judge: str, text: str) -> bool:
    if bench == "locomo":
        try:
            label = str(json.loads(text).get("label", ""))
        except json.JSONDecodeError:
            label = text
        return label.strip().upper().startswith("CORRECT")
    if judge == "mem0":
        # Mem0's parser reads the verdict after </judge_thinking>. Smaller judge models often write
        # it as the last line inside the tag; read it there too rather than count it as "no".
        after = text.rsplit("</judge_thinking>", 1)[-1]
        for region in (after, text):
            lines = [line.strip().strip("*.").lower() for line in region.splitlines() if line.strip()]
            for line in reversed(lines):
                if line in ("yes", "no"):
                    return line == "yes"
            words = re.findall(r"\b(yes|no)\b", region.lower())
            if words:
                return words[-1] == "yes"
        return False
    return "yes" in text.lower()


def category(bench: str, question: dict) -> str:
    if bench == "locomo":
        return locomo_qa.CATEGORIES[question["category"]]
    return "abstention" if question["abstention"] else question["type"]


def paired(a: list[bool], b: list[bool]) -> dict:
    wins = sum(1 for x, y in zip(a, b) if x and not y)
    losses = sum(1 for x, y in zip(a, b) if y and not x)
    n = wins + losses
    p = min(1.0, 2 * sum(comb(n, i) for i in range(min(wins, losses) + 1)) / 2 ** n) if n else 1.0
    diffs = [int(x) - int(y) for x, y in zip(a, b)]
    rng = random.Random(0)
    samples = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) * 100 for _ in range(BOOTSTRAP_SAMPLES))
    return {"difference": round(100 * sum(diffs) / len(diffs), 2),
            "ci95": [round(samples[int(0.025 * BOOTSTRAP_SAMPLES)], 2),
                     round(samples[int(0.975 * BOOTSTRAP_SAMPLES) - 1], 2)],
            "wins": wins, "losses": losses, "sign_test_p": round(p, 4)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bench", choices=("locomo", "lme"), required=True)
    parser.add_argument("--mem0-repo", type=Path, required=True, help="checkout of mem0ai/memory-benchmarks")
    parser.add_argument("--tam", action="append", required=True, metavar="LABEL=FILE",
                        help="answers written by locomo_qa.py / longmemeval_qa.py (repeatable)")
    parser.add_argument("--judge-model", help="model of the published judge (default: the harness's judge model)")
    parser.add_argument("--mem0-judge-model", help="model of Mem0's judge prompt (default: --judge-model)")
    parser.add_argument("--budget-usd", type=float, default=5.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is required")
    published_model = args.judge_model or (
        locomo_qa.JUDGE_MODEL if args.bench == "locomo" else longmemeval_qa.JUDGE_MODEL)
    judge_models = {"published": published_model, "mem0": args.mem0_judge_model or published_model}
    prompts = load_module(args.mem0_repo / MEM0_PROMPTS[args.bench], f"mem0_{args.bench}_prompts")
    questions = held_out(args.bench)
    systems = {MEM0_LABEL: mem0_answers(args.bench, args.mem0_repo)}
    for spec in args.tam:
        label, _, path = spec.partition("=")
        if not path:
            raise SystemExit(f"--tam expects LABEL=FILE, got {spec!r}")
        systems[label] = {question_key(args.bench, row): row["answer"]
                          for row in map(json.loads, Path(path).read_text().splitlines()) if row.get("answer")}
    common = sorted(set(questions).intersection(*systems.values()))
    if not common:
        raise SystemExit("no held-out question is answered by every system")

    args.out.mkdir(parents=True, exist_ok=True)
    cache = Cache(args.out / "judge-cache.jsonl")
    client = OpenAIClient(api_key, args.budget_usd)

    def grade(item: tuple[str, str, str]) -> bool:
        judge, system, key = item
        messages = judge_messages(args.bench, judge, questions[key], systems[system][key], prompts)
        cache_key = digest("crossgrade", judge_models[judge], messages)
        hit = cache.get(cache_key)
        if hit is None:
            text, _ = client.complete(judge_models[judge], messages, max_tokens=MAX_JUDGE_TOKENS[judge],
                                      json_mode=args.bench == "locomo")
            hit = {"text": text}
            cache.put(cache_key, hit)
        return verdict(args.bench, judge, hit["text"])

    marks: dict[tuple[str, str], list[bool]] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for judge in ("published", "mem0"):
            for system in systems:
                marks[(judge, system)] = list(pool.map(grade, [(judge, system, key) for key in common]))

    report = {"bench": args.bench, "questions": len(common), "judge_models": judge_models,
              "mem0_repo_commit": subprocess.run(["git", "-C", str(args.mem0_repo), "rev-parse", "HEAD"],
                                                 capture_output=True, text=True, check=False).stdout.strip() or None,
              "cells": []}
    for (judge, system), values in marks.items():
        by_category: dict[str, list[bool]] = {}
        for key, value in zip(common, values):
            by_category.setdefault(category(args.bench, questions[key]), []).append(value)
        cell = {"system": system, "judge": judge, "accuracy": round(100 * sum(values) / len(values), 2),
                "by_category": {name: round(100 * sum(v) / len(v), 2) for name, v in sorted(by_category.items())}}
        if system != MEM0_LABEL:
            cell["vs_mem0"] = paired(values, marks[(judge, MEM0_LABEL)])
        report["cells"].append(cell)
        print(f"{system:24s} judge={judge:9s} {cell['accuracy']:6.2f}  {cell.get('vs_mem0', '')}")
    report["spent_usd"] = round(client.spent, 4)
    (args.out / f"crossgrade-{args.bench}.json").write_text(json.dumps(report, indent=2))
    (args.out / f"crossgrade-{args.bench}-marks.jsonl").write_text("".join(
        json.dumps({"key": key, **{f"{s}|{j}": marks[(j, s)][i] for (j, s) in marks}}) + "\n"
        for i, key in enumerate(common)))
    print(f"spent ${client.spent:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
