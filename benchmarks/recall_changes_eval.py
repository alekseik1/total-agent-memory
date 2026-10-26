#!/usr/bin/env python3
"""Free retrieval evaluation of the 14.9 recall changes (no LLM, no paid API).

Runs `benchmarks/tier_ablation.py`'s stores and scoring (rank of the first gold
record: R@1/5/10, MRR) under several pipeline configurations that differ in
environment knobs as well as tier weights:

  default        everything at its shipped default
  intent_off     MEMORY_RECALL_INTENT_BOOSTS=decision=1,lesson=1,howto=1
  damp_off       MEMORY_RECALL_FTS_CROSSLINGUAL_DAMP=1
  qe_off         tier weight query_entity=0 (the legacy relational injection runs instead)
  qe_<w>         tier weight query_entity=<w>
  slots_on       MEMORY_RECALL_COMPOUND_SLOTS=on
  slots_off      MEMORY_RECALL_COMPOUND_SLOTS=off

`--entity-graph` copies every unit store to `--graph-root` and links each
record to a graph node per named-entity-like token of its text (the speaker and
capitalised words that do not open a sentence: `entity_tokens`), a crude
stand-in for the entity nodes that triple extraction and auto-linking give an
enriched store. Benchmark stores have no graph otherwise, so the query_entity
tier would never fire.

`--search-limit` sets the K of the search (tier_ablation uses 50); the compound
slots guarantee places inside the caller's K, so they are measured at K=10.

Usage (from the repository root, after tier_ablation.py ingested the stores):
  python benchmarks/recall_changes_eval.py --bench locomo --split dev \\
      --work-root WORK --output OUT.json --config default --config intent_off
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
sys.path.insert(0, str(ROOT / "src"))

ENTITY_NODE_TYPE = "concept"
ENTITY_LINK_ROLE = "mentions"
# "[date] Speaker: text" (LoCoMo) or "[date] user: text" (LongMemEval).
SPEAKER = re.compile(r"^\s*\[[^\]]*\]\s*([A-Z][\w-]{2,}):")
# A capitalised word preceded by a lower-case word or a comma, i.e. not opening a sentence.
MID_SENTENCE_CAPITAL = re.compile(r"(?<=[a-z,;] )([A-Z][\w-]{2,})")
ENTITY_STOPWORDS = frozenset({"i'm", "i've", "i'll", "i'd", "the", "and", "but", "what", "when", "where", "who",
                              "how", "why", "this", "that", "thanks", "yes", "yeah", "wow", "hey", "omg"})
CONFIG_ENV: dict[str, dict[str, str]] = {
    "default": {},
    "intent_off": {"MEMORY_RECALL_INTENT_BOOSTS": "decision=1,lesson=1,howto=1"},
    "damp_off": {"MEMORY_RECALL_FTS_CROSSLINGUAL_DAMP": "1"},
    "slots_on": {"MEMORY_RECALL_COMPOUND_SLOTS": "on"},
    "slots_off": {"MEMORY_RECALL_COMPOUND_SLOTS": "off"},
}


def config_of(name: str, defaults: dict[str, float]) -> tuple[dict[str, float], dict[str, str]]:
    """(tier weights, environment) of a named configuration."""
    if name == "qe_off":
        return {**defaults, "query_entity": 0.0}, {}
    if name.startswith("qe_"):
        return {**defaults, "query_entity": float(name[3:])}, {}
    if name not in CONFIG_ENV:
        raise SystemExit(f"unknown config {name!r}; known: {', '.join(CONFIG_ENV)}, qe_off, qe_<weight>")
    return dict(defaults), CONFIG_ENV[name]


def entity_tokens(content: str) -> list[str]:
    """Named-entity-like tokens: the speaker, and capitalised words that do not open a sentence.

    A crude stand-in for the entities triple extraction finds: "Caroline",
    "Boston", "LGBTQ" mid-sentence count; "Hey", "Thanks", "What" opening a
    sentence do not.
    """
    tokens: list[str] = []
    speaker = SPEAKER.match(content)
    if speaker:
        tokens.append(speaker.group(1))
    for match in MID_SENTENCE_CAPITAL.finditer(content):
        word = match.group(1)
        if word.lower() not in ENTITY_STOPWORDS:
            tokens.append(word)
    return list(dict.fromkeys(tokens))


def build_entity_graph(unit_dir: Path) -> dict:
    """Link every record of the unit store to one node per entity-like token of its content."""
    import sqlite3

    db = sqlite3.connect(unit_dir / "memory.db")
    try:
        if db.execute("SELECT COUNT(*) FROM graph_nodes WHERE id LIKE 'eval-entity:%'").fetchone()[0]:
            return {"unit": unit_dir.name, "skipped": "graph present"}
        nodes: dict[str, str] = {}
        links = 0
        rows = db.execute("SELECT id, content FROM knowledge WHERE status='active'").fetchall()
        for kid, content in rows:
            for token in entity_tokens(content or ""):
                name = token.lower()
                node_id = nodes.get(name)
                if node_id is None:
                    node_id = nodes[name] = f"eval-entity:{name}"
                    db.execute("INSERT INTO graph_nodes(id,type,name,status,first_seen_at,last_seen_at) "
                               "VALUES (?,?,?,'active','2026-01-01','2026-01-01')", (node_id, ENTITY_NODE_TYPE, name))
                db.execute("INSERT OR IGNORE INTO knowledge_nodes(knowledge_id,node_id,role,strength) "
                           "VALUES (?,?,?,1.0)", (kid, node_id, ENTITY_LINK_ROLE))
                links += 1
        db.commit()
        return {"unit": unit_dir.name, "records": len(rows), "nodes": len(nodes), "links": links}
    finally:
        db.close()


PAIR_CATEGORY = "single-hop"


def compound_pairs(questions: list[dict], style: str) -> list[dict]:
    """Consecutive, non-overlapping pairs of single-hop questions of one conversation as one compound query."""
    singles = [q for q in questions if q["category"] == PAIR_CATEGORY]
    pairs = []
    for first, second in zip(singles[::2], singles[1::2]):
        one, two = first["question"].strip(), second["question"].strip()
        if style == "question":
            text = f"{one.rstrip('?')}? {two.rstrip('?')}?"
        else:
            text = f"{one.rstrip('?')}, and {two[:1].lower()}{two[1:].rstrip('?')}?"
        pairs.append({"id": f"{first['id']}+{second['id']}", "question": text,
                      "gold": first["gold"] | second["gold"], "category": f"pair-{style}"})
    return pairs


def run_config(tier_ablation, srv, evaluator, weights: dict[str, float]) -> list[dict]:
    """tier_ablation.Evaluator.run for one configuration, plus every gold rank.

    `gold_ranks` lists the rank of each gold id found, so `coverage@K` (share
    of a question's gold records in the top K) can be computed: the measure a
    compound question needs, whose parts have separate evidence.
    """
    rows: list[dict] = []
    for unit_id, unit_dir, fragments, questions in evaluator.units:
        store = tier_ablation.open_store(srv, unit_dir, fragments)
        recall = srv.Recall(store)
        try:
            for question in questions:
                started = time.perf_counter()
                found = recall.search(question["question"], project=question.get("project", "ablation"),
                                      limit=tier_ablation.SEARCH_LIMIT, detail="compact", record_usage=False,
                                      _explain=True, tier_weights=weights)
                elapsed_ms = (time.perf_counter() - started) * 1000
                groups = found.get("results", {})
                ranked = [hit["id"] for group in (groups.values() if isinstance(groups, dict) else [groups])
                          for hit in group]
                gold_ranks: dict[str, int] = {}
                for rank, keys in enumerate(tier_ablation.gold_of(store, ranked)):
                    for key in keys & question["gold"]:
                        gold_ranks.setdefault(key, rank)
                explain = found.get("_explain", {})
                rows.append({"unit": unit_id, "id": question["id"], "category": question["category"],
                             "first_rank": min(gold_ranks.values(), default=None), "gold": len(question["gold"]),
                             "gold_ranks": sorted(gold_ranks.values()), "ms": round(elapsed_ms, 1),
                             "tiers_used": explain.get("tiers_used", []),
                             "compound": bool(explain.get("compound_parts")),
                             "historical": bool(explain.get("historical_query")),
                             "fts_damped": bool(explain.get("fts_damped"))})
        finally:
            # Recall.close exists from 14.9; the script also measures older trees for the baseline.
            close = getattr(recall, "close", None)
            if close is not None:
                close()
            store.db.close()
    return rows


def coverage(rows: list[dict], k: int) -> float:
    """Mean share of each question's gold records found in the top k."""
    if not rows:
        return 0.0
    return round(sum(sum(1 for rank in row["gold_ranks"] if rank < k) / row["gold"] for row in rows) / len(rows), 4)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bench", choices=("locomo", "lme"), required=True)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--limit", type=int, default=0, help="lme: first N questions of the split")
    parser.add_argument("--work-root", type=Path, required=True, help="tier_ablation.py work root (stores)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", action="append", default=[], help="configuration name (repeatable)")
    parser.add_argument("--cross-rerank", choices=("on", "off"), default="off")
    parser.add_argument("--search-limit", type=int, default=0, help="K of each search (default: tier_ablation's)")
    parser.add_argument("--entity-graph", action="store_true", help="evaluate on copies with an entity graph")
    parser.add_argument("--graph-root", type=Path, help="where the copies go (default: <work-root>-kg)")
    parser.add_argument("--compound-pairs", choices=("question", "and"), default=None,
                        help="replace each unit's questions by pairs of consecutive single-hop questions, "
                             "joined as 'Q1? Q2?' (question) or 'Q1, and q2?' (and); gold = union")
    args = parser.parse_args()

    os.environ["MEMORY_MODE"] = "fast"
    os.environ["MEMORY_LLM_ENABLED"] = "false"
    os.environ["MEMORY_ASYNC_ENRICHMENT"] = "false"
    os.environ["MEMORY_CROSS_RERANK"] = args.cross_rerank
    os.environ.setdefault("TAM_MEMORY_DIR", str(args.work_root / "bootstrap"))
    os.environ.setdefault("CLAUDE_MEMORY_DIR", str(args.work_root / "bootstrap"))
    import config
    import server as srv
    import tier_ablation

    if args.search_limit:
        tier_ablation.SEARCH_LIMIT = args.search_limit
    work_root = args.work_root
    graph_report = []
    units = list(tier_ablation.locomo_units(args.split) if args.bench == "locomo"
                 else tier_ablation.lme_units(args.split, args.limit))
    if args.compound_pairs:
        if args.bench != "locomo":
            raise SystemExit("--compound-pairs needs --bench locomo (several questions per store)")
        units = [(unit_id, fragments, compound_pairs(questions, args.compound_pairs))
                 for unit_id, fragments, questions in units]
    if args.entity_graph:
        source_root, work_root = args.work_root, args.graph_root or Path(f"{args.work_root}-kg")
        for unit_id, _, questions in units:
            if not questions:
                continue
            source, target = source_root / args.bench / unit_id, work_root / args.bench / unit_id
            if not (source / "ingested.json").exists():
                raise SystemExit(f"{source} is not ingested; run tier_ablation.py --ingest-only first")
            if not target.exists():
                shutil.copytree(source, target)
            graph_report.append(build_entity_graph(target))
    evaluator = tier_ablation.Evaluator(srv, work_root, args.bench)
    evaluator.load(units)
    for _, unit_dir, _, _ in evaluator.units:
        if not (unit_dir / "ingested.json").exists():
            raise SystemExit(f"{unit_dir} is not ingested; run tier_ablation.py --ingest-only first")

    defaults = dict(config.DEFAULT_RECALL_TIER_WEIGHTS)
    names = args.config or ["default"]
    results, rows_by_config, started = {}, {}, time.perf_counter()
    for name in names:
        weights, env = config_of(name, defaults)
        saved = {key: os.environ.get(key) for key in env}
        os.environ.update(env)
        try:
            rows = run_config(tier_ablation, srv, evaluator, weights)
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        rows_by_config[name] = rows
        results[name] = tier_ablation.summarize(rows, config.RECALL_TIERS)
        results[name]["coverage@10"] = coverage(rows, 10)
        compound = [row for row in rows if row["compound"]]
        results[name]["compound"] = {"n": len(compound), "coverage@5": coverage(compound, 5),
                                     "coverage@10": coverage(compound, 10)}
        results[name]["fired"] = {flag: sum(1 for row in rows if row[flag])
                                  for flag in ("compound", "historical", "fts_damped")}
        print(json.dumps({"config": name, "all": results[name]["all"]}), flush=True)
    categories = sorted({c for summary in results.values() for c in summary
                         if c not in ("all", "tier_fired", "latency_ms", "coverage@10", "compound", "fired")})
    table = tier_ablation.markdown(results, categories)
    payload = {"bench": args.bench, "split": args.split, "limit": args.limit, "cross_rerank": args.cross_rerank,
               "search_limit": tier_ablation.SEARCH_LIMIT, "entity_graph": args.entity_graph,
               "graph": graph_report, "configs": {name: config_of(name, defaults) for name in names},
               "results": results, "rows": rows_by_config, "seconds": round(time.perf_counter() - started, 1)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    args.output.with_suffix(".md").write_text(
        f"# {args.bench} {args.split} (cross-rerank {args.cross_rerank}, K={tier_ablation.SEARCH_LIMIT}, "
        f"entity graph {'on' if args.entity_graph else 'off'})\n\n{table}\n\n"
        + "\n".join(f"- {name}: coverage@10 {summary['coverage@10']:.3f}, compound {json.dumps(summary['compound'])}, "
                    f"fired {json.dumps(summary['fired'])}" for name, summary in results.items()) + "\n", encoding="utf-8")
    print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
