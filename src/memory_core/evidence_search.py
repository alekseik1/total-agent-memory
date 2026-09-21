from __future__ import annotations

import re
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass

from memory_core.evidence_pack import pack_evidence, pack_evidence_records
from memory_core.evidence_window import EvidenceWindow
from memory_core.grounding import MissingRelation
from memory_core.passage_index import (
    PASSAGE_CHARS,
    EmbedTexts,
    PassageIndex,
    RankedPassage,
    split_passages,
)
from memory_core.retrieval import MemoryHit, SearchScope
from memory_core.telemetry import counters, op_timer

MAX_CANDIDATES = 50
PASSAGES_PER_SOURCE = 3
SEMANTIC_PASSAGES_PER_SOURCE = 2
MIN_EVIDENCE_WEIGHT = 0.25
_WORD = re.compile(r'\w+', re.UNICODE)
Search = Callable[[str, int], list[MemoryHit]]


def evidence_weight(content: str, seen: list[set[str]]) -> float:
    terms = set(_WORD.findall(content.casefold()))
    overlap = max((len(terms & other) / max(1, len(terms | other)) for other in seen), default=0)
    weight = max(MIN_EVIDENCE_WEIGHT, (1 - overlap) / (1 + len(seen)))
    seen.append(terms)
    return weight


@dataclass(frozen=True)
class FollowupPlan:
    query: str
    missing_terms: tuple[str, ...]
    reason: str


@dataclass
class EvidenceResult:
    evidence: list[MemoryHit]
    context: str
    followup: FollowupPlan | None
    search_calls: int


class EvidenceSearch:
    def __init__(self, db: sqlite3.Connection, search: Search, embed: EmbedTexts,
                 model: str, excluded_tags: tuple[str, ...] = (), neighbor_radius: int = 0,
                 query_embed: Callable[[str], list[float]] | None = None):
        self.db = db
        self.search = search
        self.index = PassageIndex(db, embed, model, query_embed)
        self.window = EvidenceWindow(db, excluded_tags)
        self.neighbor_radius = neighbor_radius

    def _select(self, query: str, hits: list[MemoryHit], scope: SearchScope, limit: int) -> list[MemoryHit]:
        current = self.window.expand(hits, scope=scope, radius=self.neighbor_radius)
        if all(len(hit['content']) <= PASSAGE_CHARS for hit in current):
            seen: list[set[str]] = []
            return [{**hit, 'evidence_weight': evidence_weight(hit['content'], seen),
                     'passages': [{'start': p.start, 'end': p.end, 'speaker': p.speaker}
                                  for p in split_passages(hit['content'])]} for hit in current[:limit]]
        self.index.ensure(current)
        ranked = self.index.rank(query, [hit['id'] for hit in current])
        grouped: dict[int, list[RankedPassage]] = {}
        for item in ranked:
            grouped.setdefault(item.source_id, []).append(item)
        valid = {hit['id']: hit for hit in self.window.expand(current, scope=scope, radius=0)}
        sources = {hit['id']: hit for hit in current
                   if hit['id'] in valid and hit['content'] == valid[hit['id']]['content']}
        grouped = {kid: passages for kid, passages in grouped.items() if kid in sources}
        order = sorted(grouped, key=lambda kid: (-grouped[kid][0].score, kid))[:limit]
        result = []
        seen_terms: list[set[str]] = []
        for kid in order:
            source = sources[kid]
            blocks = [item.passage for item in sorted(grouped[kid], key=lambda item: item.passage.ordinal)]
            chosen = set()
            if blocks and re.match(r'^\[[^\n]+\]', blocks[0].content):
                chosen.add(0)
            centers = []
            semantic = sorted(grouped[kid], key=lambda item: item.semantic_rank)
            for candidates, cap in ((semantic, SEMANTIC_PASSAGES_PER_SOURCE), (grouped[kid], PASSAGES_PER_SOURCE)):
                for candidate in candidates:
                    if len(centers) >= cap:
                        break
                    if all(abs(candidate.passage.ordinal - previous.passage.ordinal) > 1 for previous in centers):
                        centers.append(candidate)
            for candidate in centers:
                ordinal = candidate.passage.ordinal
                chosen.update(range(max(0, ordinal - 1), min(len(blocks), ordinal + 2)))
            selected = [blocks[index] for index in sorted(chosen)]
            content = '\n …[passage boundary]… \n'.join(p.content for p in selected)
            weight = evidence_weight(content, seen_terms)
            result.append({**source, 'content': content, 'evidence_weight': weight,
                'passages': [{'start': p.start, 'end': p.end, 'speaker': p.speaker} for p in selected]})
        return result

    def run(self, query: str, scope: SearchScope, *, limit: int = 10, max_bytes: int = 24000,
            followup: bool = True, initial: list[MemoryHit] | None = None,
            missing_relation: MissingRelation | None = None) -> EvidenceResult:
        if type(limit) is not int or not 1 <= limit <= MAX_CANDIDATES:
            raise ValueError('Evidence source limit must be between 1 and 50')
        if type(max_bytes) is not int or max_bytes < 512:
            raise ValueError('Evidence byte budget must be at least 512')
        with op_timer('evidence_search_ms'):
            candidate_limit = min(MAX_CANDIDATES, limit * 2)
            hits = initial if initial is not None else self.search(query, candidate_limit)
            selected = self._select(query, hits, scope, limit)
            plan = None
            if followup and limit > 1 and missing_relation is not None:
                focused = missing_relation.query(query)
                if focused.casefold() != query.strip().casefold():
                    plan = FollowupPlan(focused, (missing_relation.relation,), 'missing_relation')
            if plan is not None:
                extra = self.search(plan.query, candidate_limit)
                existing = {hit['id'] for hit in selected}
                unique = {hit['id']: hit for hit in extra if hit['id'] not in existing}
                additions = self._select(plan.query, list(unique.values())[:MAX_CANDIDATES], scope, max(1, limit // 3))
                selected = selected[:limit - len(additions)] + [
                    {**hit, 'via': [*hit.get('via', []), 'focused_followup']} for hit in additions]
                counters.bump('evidence_followup_calls')
            packed = pack_evidence_records(selected, query=query, max_bytes=max_bytes)
            counters.bump('evidence_search_calls')
            return EvidenceResult(packed, pack_evidence(packed, query=query, max_bytes=max_bytes), plan, 2 if plan else 1)
