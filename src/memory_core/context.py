from __future__ import annotations

import json
from typing import NotRequired, TypedDict

from memory_core.telemetry import counters, op_timer


class ContextItem(TypedDict, total=False):
    id: int | str
    content: str
    narrative: str
    source: str
    source_ref: str


class ContextBundle(TypedDict):
    knowledge: list[ContextItem]
    episodes: list[ContextItem]
    skills: list[ContextItem]
    rules: list[ContextItem]
    competency: ContextItem | None
    blind_spots: list[ContextItem]
    total_tokens: int
    budget_method: NotRequired[str]
    omitted_items: NotRequired[int]


SECTIONS = ("rules", "knowledge", "episodes", "skills", "blind_spots")
SOURCE_TABLES = {"rules": "rules", "knowledge": "knowledge", "episodes": "episodes",
                 "skills": "skills", "blind_spots": "blind_spots"}


def bound_context(bundle: ContextBundle, max_tokens: int) -> ContextBundle:
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 0:
        raise ValueError("max_tokens must be a non-negative integer")
    with op_timer("context_build_ms"):
        counters.bump("context_build_calls")
        result: ContextBundle = {**bundle, "total_tokens": 0,
                                 "budget_method": "evidence_utf8_bytes_upper_bound",
                                 "omitted_items": 0}
        for section in SECTIONS:
            result[section] = []
            for original in bundle[section]:
                item = dict(original)
                if "id" in item:
                    table = "graph_nodes" if item.get("source") == "graph" else SOURCE_TABLES[section]
                    item["source_ref"] = f"{table}:{item['id']}"
                cost = _cost(item)
                if result["total_tokens"] + cost <= max_tokens:
                    result[section].append(item)
                    result["total_tokens"] += cost
                else:
                    result["omitted_items"] += 1
        if bundle["competency"] is not None:
            cost = _cost(bundle["competency"])
            if result["total_tokens"] + cost <= max_tokens:
                result["total_tokens"] += cost
            else:
                result["competency"] = None
                result["omitted_items"] += 1
        return result


def _cost(item: ContextItem) -> int:
    return len(json.dumps(item, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
