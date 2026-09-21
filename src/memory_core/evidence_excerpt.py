from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from memory_core.query_terms import lexical_terms

EXCERPT_CHUNK_CHARS = 384
EXCERPT_HEADER_CHARS = 160
OMISSION = " …[excerpt truncated]… "
_WORD = re.compile(r"\w+", re.UNICODE)


@dataclass(frozen=True)
class Passage:
    start: int
    end: int
    terms: frozenset[str]


def _clip(text: str, budget: int, cost: Callable[[str], int]) -> str:
    low, high = 0, min(len(text), budget)
    while low < high:
        middle = (low + high + 1) // 2
        if cost(text[:middle]) <= budget:
            low = middle
        else:
            high = middle - 1
    return text[:low]


def _passages(content: str) -> list[Passage]:
    passages = []
    start = 0
    while start < len(content):
        end = min(len(content), start + EXCERPT_CHUNK_CHARS)
        if end < len(content):
            boundary = max(content.rfind("\n", start, end), content.rfind(". ", start, end))
            if boundary > start:
                end = boundary + 1
        passages.append(Passage(start, end, frozenset(_WORD.findall(content[start:end].casefold()))))
        start = end
    return passages


def excerpt(content: str, query: str, budget: int, cost: Callable[[str], int]) -> str:
    if cost(content) <= budget:
        return content
    marker_cost = cost(OMISSION)
    if budget <= marker_cost:
        return _clip(OMISSION, budget, cost)
    terms = frozenset(_WORD.findall(" ".join(lexical_terms(query)))) if query else frozenset()
    passages = _passages(content) if terms else []
    scores = [len(passage.terms & terms) for passage in passages]
    if not any(scores):
        usable = budget - marker_cost
        front = _clip(content, (usable + 1) // 2, cost)
        tail = _clip(content[::-1], usable // 2, cost)[::-1]
        return front + OMISSION + tail

    header_end = min(content.find("\n") + 1 or EXCERPT_HEADER_CHARS, EXCERPT_HEADER_CHARS)
    intervals = [(0, header_end)]

    def render(ranges: list[tuple[int, int]]) -> str:
        merged: list[tuple[int, int]] = []
        for start, end in sorted(ranges):
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
            else:
                merged.append((start, end))
        result = OMISSION.join(content[start:end] for start, end in merged)
        return result + (OMISSION if merged[-1][1] < len(content) else "")

    covered: set[str] = set()
    remaining = {index for index, score in enumerate(scores) if score}
    while remaining:
        index = max(remaining, key=lambda i: (
            len((passages[i].terms & terms) - covered), scores[i], -i,
        ))
        remaining.remove(index)
        for radius in (1, 0):
            start = passages[max(0, index - radius)].start
            end = passages[min(len(passages) - 1, index + radius)].end
            proposed = intervals + [(start, end)]
            if cost(render(proposed)) <= budget:
                intervals = proposed
                covered.update(passages[index].terms & terms)
                break
    rendered = render(intervals)
    if len(intervals) == 1:
        best = passages[max(range(len(scores)), key=scores.__getitem__)]
        prefix = _clip(content[:header_end], max(0, (budget - marker_cost * 2) // 3), cost)
        available = budget - cost(prefix) - marker_cost * 2
        body = _clip(content[best.start:best.end], max(0, available), cost)
        rendered = prefix + OMISSION + body + OMISSION
    return _clip(rendered, budget, cost)
