#!/usr/bin/env python3
"""Free check of value-update ordering: current questions vs "what was it before" questions.

Builds a throwaway store with SUBJECTS x ATTRIBUTES statements, each saved twice
with a different value (the second one later), plus unrelated noise records,
through the same `Store.save_knowledge` / `Recall.search` as the server
(local embedding model, no LLM). For every statement it asks

  current     "What is the <attribute> of <subject>?"      -> the newer value should rank first
  historical  "What was the <attribute> of <subject> before?" -> the older value should rank first

in English and Russian, and reports the share of questions whose first-ranked
value record is the expected one. Before 14.9 Stage 4.9 always put the newer
value first, so the historical share was ~0 by construction.

Usage:
  python benchmarks/historical_value_eval.py --work-root DIR --output OUT.json [--cross-rerank on|off]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEARCH_LIMIT = 10
OLD_CREATED = "2025-01-01T00:00:00Z"
NEW_CREATED = "2026-01-01T00:00:00Z"

EN = {
    "subjects": ["billing service", "search service", "auth gateway", "report worker", "mobile backend",
                 "payments api", "analytics pipeline", "notification hub"],
    "attributes": [("database", "MySQL", "PostgreSQL"), ("message broker", "RabbitMQ", "Kafka"),
                   ("deploy tool", "Ansible", "Helm")],
    "record": "The {attribute} of the {subject} is {value}",
    "current": "What is the {attribute} of the {subject}?",
    "historical": "What was the {attribute} of the {subject} before?",
    "noise": "The {subject} team meets on {day} to review incidents",
}
RU = {
    "subjects": ["сервиса биллинга", "сервиса поиска", "шлюза авторизации", "воркера отчётов", "мобильного бэкенда",
                 "платёжного api", "аналитического конвейера", "центра уведомлений"],
    "attributes": [("база данных", "MySQL", "PostgreSQL"), ("брокер сообщений", "RabbitMQ", "Kafka"),
                   ("инструмент деплоя", "Ansible", "Helm")],
    "record": "Основная {attribute} {subject} это {value}",
    "current": "Какая основная {attribute} {subject}?",
    "historical": "Какая основная {attribute} {subject} была раньше?",
    "noise": "Команда {subject} встречается по {day} разбирать инциденты",
}
DAYS = ("monday", "tuesday", "wednesday", "thursday", "friday")


def build(store, lang: str, spec: dict) -> list[dict]:
    """Save the statements of one language; returns one probe per statement."""
    probes = []
    for subject in spec["subjects"]:
        for day in DAYS[:2]:
            store.save_knowledge(sid="eval", content=spec["noise"].format(subject=subject, day=day), ktype="fact",
                                 project=lang, skip_dedup=True, skip_quality=True)
        for attribute, old, new in spec["attributes"]:
            ids = {}
            for value, created in ((old, OLD_CREATED), (new, NEW_CREATED)):
                rid, *_ = store.save_knowledge(sid="eval", content=spec["record"].format(
                    attribute=attribute, subject=subject, value=value), ktype="fact", project=lang,
                    skip_dedup=True, skip_quality=True)
                store.db.execute("UPDATE knowledge SET created_at=? WHERE id=?", (created, rid))
                ids[value] = rid
            probes.append({"lang": lang, "subject": subject, "attribute": attribute,
                           "old": ids[old], "new": ids[new],
                           "current": spec["current"].format(attribute=attribute, subject=subject),
                           "historical": spec["historical"].format(attribute=attribute, subject=subject)})
    store.db.commit()
    return probes


def first_value(result: dict, probe: dict) -> int | None:
    ids = [hit["id"] for group in result.get("results", {}).values() for hit in group]
    return next((kid for kid in ids if kid in (probe["old"], probe["new"])), None)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cross-rerank", choices=("on", "off"), default="off")
    args = parser.parse_args()
    args.work_root.mkdir(parents=True, exist_ok=True)
    os.environ.update(MEMORY_MODE="fast", MEMORY_LLM_ENABLED="false", MEMORY_ASYNC_ENRICHMENT="false",
                      MEMORY_CROSS_RERANK=args.cross_rerank, TAM_MEMORY_DIR=str(args.work_root),
                      CLAUDE_MEMORY_DIR=str(args.work_root))
    sys.path.insert(0, str(ROOT / "src"))
    import server

    server.MEMORY_DIR = args.work_root
    store = server.Store(background_queues=False)
    probes = build(store, "en", EN) + build(store, "ru", RU)
    recall = server.Recall(store)
    rows, started = [], time.perf_counter()
    for probe in probes:
        for kind, expected in (("current", probe["new"]), ("historical", probe["old"])):
            result = recall.search(probe[kind], project=probe["lang"], limit=SEARCH_LIMIT, record_usage=False)
            rows.append({"lang": probe["lang"], "kind": kind, "query": probe[kind],
                         "correct": first_value(result, probe) == expected,
                         "found": first_value(result, probe) is not None})
    summary = {}
    for lang in ("en", "ru"):
        for kind in ("current", "historical"):
            group = [row for row in rows if row["lang"] == lang and row["kind"] == kind]
            summary[f"{lang}/{kind}"] = {"n": len(group),
                                         "correct": round(sum(row["correct"] for row in group) / len(group), 3),
                                         "found": round(sum(row["found"] for row in group) / len(group), 3)}
    payload = {"cross_rerank": args.cross_rerank, "summary": summary, "rows": rows,
               "seconds": round(time.perf_counter() - started, 1)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
