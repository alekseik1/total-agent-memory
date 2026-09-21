from __future__ import annotations

import sqlite3

from memory_core.evidence_chains import EvidenceChains
from memory_core.evidence_pack import DEFAULT_EVIDENCE_CHARS, pack_evidence_records
from memory_core.evidence_window import EvidenceWindow
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
            result = pack_evidence_records(
                linked, query=query, max_chars=max_chars, max_bytes=max_bytes,
            )
            counters.bump("evidence_context_sources", len(result))
            return result
