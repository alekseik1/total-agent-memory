from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass

from ai_layer.grounded_reader import (
    REFUSAL,
    GroundedDraft,
    GroundedReader,
    Verification,
)
from memory_core.evidence_context import EvidenceContext
from memory_core.evidence_pack import pack_evidence_records
from memory_core.evidence_window import EvidenceWindow
from memory_core.grounding import InvalidGrounding
from memory_core.negative_retrieval import (
    ContradictionBatchFn,
    LLMLike,
    NegativeEvidenceResult,
    negative_retrieve,
)
from memory_core.retrieval import MemoryHit, SearchScope
from memory_core.telemetry import counters, op_timer

MAX_SOURCES = 20
MAX_ANSWER_CONTEXT_BYTES = 24000
Search = Callable[[str, int], list[MemoryHit]]


@dataclass(frozen=True)
class GroundedAnswer:
    answer: str
    draft: GroundedDraft
    verification: Verification | None
    search_calls: int
    followup_query: str | None
    evidence: list[MemoryHit]
    negative: NegativeEvidenceResult | None = None
    caveat: str | None = None


class GroundedAnswerService:
    def __init__(self, db: sqlite3.Connection, search: Search, reader: GroundedReader,
                 excluded_tags: tuple[str, ...] = (), *,
                 contradiction_scorer: ContradictionBatchFn | None = None,
                 inversion_client: LLMLike | None = None):
        self.db, self.search, self.reader = db, search, reader
        self.contradiction_scorer, self.inversion_client = contradiction_scorer, inversion_client
        self.window = EvidenceWindow(db, excluded_tags)
        self.context = EvidenceContext(db, excluded_tags)

    def _context(self, query: str, scope: SearchScope, limit: int, max_bytes: int) -> tuple[list[MemoryHit], list[MemoryHit]]:
        hits = self.search(query, limit)
        self.db.execute('SAVEPOINT grounded_evidence_read')
        try:
            evidence = self.context.build(hits, query=query, scope=scope, radius=0, max_bytes=max_bytes)
            originals = self.window.expand(evidence, scope=scope, radius=0)
            self.db.execute('RELEASE grounded_evidence_read')
            return evidence, originals
        except Exception:
            self.db.execute('ROLLBACK TO grounded_evidence_read')
            self.db.execute('RELEASE grounded_evidence_read')
            raise

    def _validate_current(self, originals: list[MemoryHit], scope: SearchScope) -> None:
        current = {hit['id']: hit for hit in self.window.expand(originals, scope=scope, radius=0)}
        fields = ('content', 'project', 'branch', 'type', 'status')
        if any(hit['id'] not in current or any(hit.get(key) != current[hit['id']].get(key) for key in fields) for hit in originals):
            raise InvalidGrounding('Evidence changed during answer generation; retry the query')

    def _merge(self, pack_query: str, scope: SearchScope, evidence: list[MemoryHit], originals: list[MemoryHit],
               extra: list[MemoryHit], extra_originals: list[MemoryHit], limit: int,
               max_bytes: int) -> tuple[list[MemoryHit], list[MemoryHit]]:
        self._validate_current(originals, scope)
        originals = list({hit['id']: hit for hit in [*originals, *extra_originals]}.values())
        primary_ids = {hit['id'] for hit in evidence}
        additions = [hit for hit in extra if hit['id'] not in primary_ids][:max(1, limit // 3)]
        source_by_id = {hit['id']: hit for hit in originals}
        selected = [{**hit, 'content': source_by_id[hit['id']]['content']} for hit in [*evidence, *additions]]
        return pack_evidence_records(selected, query=pack_query, max_bytes=max_bytes), originals

    def _negative(self, query: str, scope: SearchScope, evidence: list[MemoryHit], max_bytes: int
                  ) -> tuple[NegativeEvidenceResult, list[MemoryHit], list[MemoryHit]] | None:
        if self.contradiction_scorer is None or not evidence:
            return None
        found: dict[str, list[MemoryHit]] = {'extra': [], 'originals': []}

        def search(inverted: str, k: int = 5, project: str | None = None) -> list[MemoryHit]:
            found['extra'], found['originals'] = self._context(inverted, scope, k, max_bytes)
            return found['extra']

        with op_timer('grounded_negative_ms'):
            counters.bump('grounded_negative_calls')
            result = negative_retrieve(query, evidence, search_fn=search,
                                       contradiction_batch_fn=self.contradiction_scorer,
                                       project=scope.project, llm_client=self.inversion_client)
        counters.bump(f'grounded_negative_{result.decision}')
        return result, found['extra'], found['originals']

    def answer(self, query: str, scope: SearchScope, *, limit: int = 10,
               max_bytes: int = MAX_ANSWER_CONTEXT_BYTES, followup: bool = True) -> GroundedAnswer:
        if not query.strip() or type(limit) is not int or not 1 <= limit <= MAX_SOURCES:
            raise ValueError('Answer requires a query and source limit between 1 and 20')
        if type(max_bytes) is not int or not 512 <= max_bytes <= MAX_ANSWER_CONTEXT_BYTES:
            raise ValueError('Answer context budget must be between 512 and 24000 bytes')
        with op_timer('grounded_answer_ms'):
            counters.bump('grounded_answer_calls')
            evidence, originals = self._context(query, scope, limit, max_bytes)
            searches = 1
            negative, caveat = None, None
            checked = self._negative(query, scope, evidence, max_bytes)
            if checked is not None:
                negative, negative_extra, negative_originals = checked
                searches += 1 if negative.inverted_query else 0
                if negative.decision == 'hard_contradict':
                    # Strong conflict wins: abstain instead of picking a side.
                    self._validate_current(originals, scope)
                    counters.bump('grounded_negative_vetoes')
                    draft = GroundedDraft('insufficient', REFUSAL, (), None, negative.rationale)
                    return GroundedAnswer(REFUSAL, draft, None, searches, None, evidence, negative, None)
                if negative.decision == 'soft_contradict':
                    evidence, originals = self._merge(f'{query}\n{negative.inverted_query}', scope, evidence, originals,
                                                      negative_extra, negative_originals, limit, max_bytes)
                    caveat = f'Evidence is mixed: {negative.rationale}'
            draft = self.reader.read(query, evidence)
            focused = None
            if followup and draft.missing is not None:
                bridge_quotes = tuple(ref.quote for claim in draft.claims for ref in claim.citations)
                focused = draft.missing.query(query, bridge_quotes)
                if focused.casefold() == query.strip().casefold():
                    focused = None
            if focused is not None:
                extra, extra_originals = self._context(focused, scope, limit, max_bytes)
                refreshed, originals = self._merge(f'{query}\n{focused}', scope, evidence, originals,
                                                   extra, extra_originals, limit, max_bytes)
                searches += 1
                if refreshed != evidence:
                    evidence = refreshed
                    draft = self.reader.read(query, evidence)
                counters.bump('grounded_followup_calls')
            verdict = self.reader.verify(query, draft, evidence) if draft.status in ('supported', 'inferred') else None
            if draft.rejection:
                verdict = Verification(False, draft.rejection)
            self._validate_current(originals, scope)
            answer = draft.answer if verdict is not None and verdict.supported else REFUSAL
            if verdict is not None and not verdict.supported:
                counters.bump('grounded_answer_vetoes')
            if answer == REFUSAL:
                caveat = None
            elif caveat is not None:
                answer = f'{answer}\n\nCaveat: {caveat}'
            return GroundedAnswer(answer, draft, verdict, searches, focused, evidence, negative, caveat)
