from __future__ import annotations

import sqlite3

import config as _cfg
from memory_core.evidence_chains import EvidenceChains
from memory_core.evidence_pack import DEFAULT_EVIDENCE_CHARS, pack_evidence_records
from memory_core.evidence_window import EvidenceWindow
from memory_core.relative_dates import annotate
from memory_core.retrieval import MemoryHit, SearchScope
from memory_core.telemetry import counters, op_timer


class EvidenceContext:
    def __init__(self, db: sqlite3.Connection, excluded_tags: tuple[str, ...] = ()):
        self.window = EvidenceWindow(db, excluded_tags)
        self.chains = EvidenceChains(db, excluded_tags)

    def build(
        self, hits: list[MemoryHit], *, query: str, scope: SearchScope,
        radius: int = 1, max_chars: int = DEFAULT_EVIDENCE_CHARS, max_bytes: int | None = None,
    ) -> list[MemoryHit]:
        with op_timer("evidence_context_ms"):
            expanded = self.window.expand(hits, scope=scope, radius=radius)
            linked = self.chains.expand(expanded, scope)
            linked = rank_weighted(linked)
            if _cfg.context_resolves_dates():
                linked = [{**hit, "content": annotate(str(hit.get("content", "")), hit.get("created_at"))}
                          for hit in linked]
            result = pack_evidence_records(
                linked, query=query, max_chars=max_chars, max_bytes=max_bytes,
            )
            counters.bump("evidence_context_sources", len(result))
            return result


def rank_weighted(evidence: list[MemoryHit]) -> list[MemoryHit]:
    """Share of the context budget by search rank.

    An even split cut every long record to an excerpt around the query's words: an
    assistant's list of a hundred items lost the one asked about although its round
    ranked first. The first hit now gets four shares, the second 2.5, the tenth 1.3;
    session neighbours, there for context, get half a share.
    """
    rank = 0
    weighted = []
    for hit in evidence:
        if "session_neighbor" in (hit.get("via") or []):
            weighted.append({**hit, "evidence_weight": 0.5})
            continue
        weighted.append({**hit, "evidence_weight": float(hit.get("evidence_weight", 1.0)) * (1 + 3 / (1 + rank))})
        rank += 1
    return weighted
