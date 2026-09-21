from __future__ import annotations

import sqlite3

from memory_core.retrieval import EventEvidence, MemoryHit, SearchScope
from memory_core.telemetry import counters, op_timer


class EvidenceChains:
    def __init__(self, db: sqlite3.Connection, excluded_tags: tuple[str, ...] = ()):
        self.db = db
        self.excluded_tags = excluded_tags

    def expand(
        self, hits: list[MemoryHit], scope: SearchScope, max_sources: int = 40
    ) -> list[MemoryHit]:
        if type(max_sources) is not int or not 0 <= max_sources <= 100:
            raise ValueError("max_sources must be between 0 and 100")
        if not hits:
            return []
        with op_timer("evidence_chain_ms"):
            predicates, params = scope.sql()
            for tag in self.excluded_tags:
                predicates.append("instr(lower(COALESCE(k.tags,'')),lower(?))=0")
                params.append(tag)
            input_ids = list(dict.fromkeys(hit["id"] for hit in hits))
            allowed_anchors = {
                row[0]
                for row in self.db.execute(
                    "SELECT k.id FROM knowledge k NOT INDEXED WHERE k.id IN ("
                    + ",".join("?" for _ in input_ids)
                    + ") AND "
                    + " AND ".join(predicates),
                    [*input_ids, *params],
                )
            }
            hits = [hit for hit in hits if hit["id"] in allowed_anchors]
            if not hits or not max_sources:
                return hits
            if (
                self.db.execute(
                    "SELECT 1 FROM sqlite_master WHERE name='atomic_fact_sources'"
                ).fetchone()
                is None
            ):
                return hits
            ids = list(dict.fromkeys(hit["id"] for hit in hits))
            placeholders = ",".join("?" for _ in ids)
            rows = self.db.execute(
                "SELECT s.fact_id, k.* FROM atomic_fact_sources s CROSS JOIN knowledge k ON k.id=s.knowledge_id "
                f"WHERE s.fact_id IN (SELECT fact_id FROM atomic_fact_sources WHERE knowledge_id IN ({placeholders})) "
                "ORDER BY s.fact_id,k.id",
                ids,
            ).fetchall()
            predicates, params = scope.sql()
            for tag in self.excluded_tags:
                predicates.append("instr(lower(COALESCE(k.tags,'')),lower(?))=0")
                params.append(tag)
            source_ids = list(dict.fromkeys(row["id"] for row in rows))
            if not source_ids:
                return hits
            allowed = {
                row[0]
                for row in self.db.execute(
                    "SELECT k.id FROM knowledge k NOT INDEXED WHERE k.id IN ("
                    + ",".join("?" for _ in source_ids)
                    + ") AND "
                    + " AND ".join(predicates),
                    [*source_ids, *params],
                )
            }
            groups: dict[int, list[MemoryHit]] = {}
            for row in rows:
                hit = dict(row)
                groups.setdefault(hit.pop("fact_id"), []).append(hit)
            events = self._events(list(groups))
            result = [dict(hit) for hit in hits]
            seen = set(ids)
            added = 0
            for identity, group in groups.items():
                members = {hit["id"] for hit in group}
                missing = members - seen
                if not members <= allowed or added + len(missing) > max_sources:
                    counters.bump("evidence_chain_omitted")
                    continue
                for hit in group:
                    if hit["id"] in missing:
                        result.append(
                            {
                                **hit,
                                "via": ["atomic_evidence"],
                                "source_ref": f"knowledge:{hit['id']}",
                            }
                        )
                seen.update(missing)
                added += len(missing)
                for hit in result:
                    if hit["id"] in members:
                        hit["evidence_ids"] = sorted(
                            set(hit.get("evidence_ids", [])) | members
                        )
                        if identity in events:
                            hit["events"] = [
                                *hit.get("events", []),
                                {**events[identity], "source_ids": sorted(members)},
                            ]
            counters.bump("evidence_chain_sources", added)
            return result

    def _events(self, ids: list[int]) -> dict[int, EventEvidence]:
        if not ids:
            return {}
        columns = {row[1] for row in self.db.execute("PRAGMA table_info(atomic_facts)")}
        if "event_key" not in columns:
            return {}
        rows = self.db.execute(
            "SELECT id,event_key,observed_at,event_start,event_end,event_precision,temporal_text "
            ",temporal_anchor_at FROM atomic_facts WHERE id IN ("
            + ",".join("?" for _ in ids)
            + ")",
            ids,
        ).fetchall()
        return {
            row[0]: EventEvidence(
                key=row[1],
                observed_at=row[2],
                temporal_anchor_at=row[7],
                start=row[3],
                end=row[4],
                precision=row[5],
                expression=row[6],
                source_ids=[],
            )
            for row in rows
        }
