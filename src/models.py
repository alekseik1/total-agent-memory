"""
Super Memory v5.0 — Domain Models

All dataclasses and enums for the MCP memory server.
Pure stdlib: dataclasses + enum + typing. No pydantic, no attrs.

Every model supports:
  - to_dict()     → plain dict (JSON-safe, lists/dicts for JSON fields)
  - from_dict()   → construct from dict (e.g. from JSON API payload)
  - from_row()    → construct from sqlite3.Row or tuple (DB layer)

JSON fields stored as TEXT in SQLite are handled via json_loads_safe().
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def uuid7() -> str:
    """Generate UUID v7 (time-ordered, RFC 9562).

    Layout (128 bits):
      - 48 bits: unix timestamp in milliseconds
      - 4 bits:  version (0b0111)
      - 12 bits: random
      - 2 bits:  variant (0b10)
      - 62 bits: random
    """
    ts_ms = int(time.time() * 1000)
    rand_bytes = os.urandom(10)

    # First 6 bytes: timestamp
    ts_bytes = ts_ms.to_bytes(6, "big")

    # Byte 6-7: version (4 bits) + 12 random bits
    rand_a = int.from_bytes(rand_bytes[:2], "big")
    rand_a = (rand_a & 0x0FFF) | 0x7000  # version 7

    # Byte 8-15: variant (2 bits) + 62 random bits
    rand_b = int.from_bytes(rand_bytes[2:], "big")
    rand_b = (rand_b & 0x3FFFFFFFFFFFFFFF) | 0x8000000000000000  # variant 10

    hex_str = (
        ts_bytes.hex()
        + f"{rand_a:04x}"
        + f"{rand_b:016x}"
    )
    return str(uuid.UUID(hex_str))


def json_loads_safe(text: str | None, default: Any = None) -> Any:
    """Safe JSON parse for SQLite TEXT fields.

    Returns *default* on None, empty string, or malformed JSON.
    """
    if not text:
        return default
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return default


def _json_dumps(obj: Any) -> str | None:
    """Serialize to compact JSON string, or None if obj is None."""
    if obj is None:
        return None
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def _iso_now() -> str:
    """Current UTC time as ISO 8601 string."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class NodeType(str, Enum):
    """Knowledge graph node types (22 total)."""

    RULE = "rule"
    CONVENTION = "convention"
    PROHIBITION = "prohibition"
    SKILL = "skill"
    PROCEDURE = "procedure"
    EPISODE = "episode"
    FACT = "fact"
    SOLUTION = "solution"
    DECISION = "decision"
    LESSON = "lesson"
    CONCEPT = "concept"
    PATTERN = "pattern"
    TECHNOLOGY = "technology"
    REPO = "repo"
    ARTICLE = "article"
    DOC = "doc"
    PERSON = "person"
    PROJECT = "project"
    COMPANY = "company"
    BLINDSPOT = "blindspot"
    COMPETENCY = "competency"
    PREFERENCE = "preference"


class RelationType(str, Enum):
    """Knowledge graph edge / relation types (28 total)."""

    IS_A = "is_a"
    PART_OF = "part_of"
    HAS_PART = "has_part"
    WORKS_AT = "works_at"
    WORKS_ON = "works_on"
    OWNS = "owns"
    USES = "uses"
    DEPENDS_ON = "depends_on"
    ALTERNATIVE_TO = "alternative_to"
    INTEGRATES_WITH = "integrates_with"
    REPLACED_BY = "replaced_by"
    PROVIDES = "provides"
    REQUIRES = "requires"
    COMPOSABLE_WITH = "composable_with"
    SOLVES = "solves"
    CAUSES = "causes"
    CONTRADICTS = "contradicts"
    SUPERSEDES = "supersedes"
    GENERALIZES = "generalizes"
    EXAMPLE_OF = "example_of"
    GOVERNS = "governs"
    ENFORCED_BY = "enforced_by"
    APPLIES_TO = "applies_to"
    LED_TO = "led_to"
    PRECEDED_BY = "preceded_by"
    STRUGGLES_WITH = "struggles_with"
    PREFERS = "prefers"
    MENTIONED_WITH = "mentioned_with"


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class GraphNode:
    """A node in the knowledge graph."""

    id: str
    type: NodeType
    name: str
    content: str | None = None
    properties: dict[str, Any] = field(default_factory=dict)
    source: str = "manual"
    importance: float = 0.5
    first_seen_at: str = field(default_factory=_iso_now)
    last_seen_at: str = field(default_factory=_iso_now)
    mention_count: int = 1
    status: str = "active"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "name": self.name,
            "content": self.content,
            "properties": self.properties,
            "source": self.source,
            "importance": self.importance,
            "first_seen_at": self.first_seen_at,
            "last_seen_at": self.last_seen_at,
            "mention_count": self.mention_count,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GraphNode:
        return cls(
            id=data["id"],
            type=NodeType(data["type"]),
            name=data["name"],
            content=data.get("content"),
            properties=data.get("properties") or {},
            source=data.get("source", "manual"),
            importance=float(data.get("importance", 0.5)),
            first_seen_at=data.get("first_seen_at", _iso_now()),
            last_seen_at=data.get("last_seen_at", _iso_now()),
            mention_count=int(data.get("mention_count", 1)),
            status=data.get("status", "active"),
        )

    @classmethod
    def from_row(cls, row: tuple | sqlite3_Row) -> GraphNode:
        """Construct from DB row: (id, type, name, content, properties_json,
        source, importance, first_seen_at, last_seen_at, mention_count, status)."""
        return cls(
            id=row[0],
            type=NodeType(row[1]),
            name=row[2],
            content=row[3],
            properties=json_loads_safe(row[4], {}),
            source=row[5] or "manual",
            importance=float(row[6]) if row[6] is not None else 0.5,
            first_seen_at=row[7] or _iso_now(),
            last_seen_at=row[8] or _iso_now(),
            mention_count=int(row[9]) if row[9] is not None else 1,
            status=row[10] or "active",
        )

    def to_row_values(self) -> tuple:
        """Values for INSERT/UPDATE (matches from_row column order)."""
        return (
            self.id,
            self.type.value,
            self.name,
            self.content,
            _json_dumps(self.properties),
            self.source,
            self.importance,
            self.first_seen_at,
            self.last_seen_at,
            self.mention_count,
            self.status,
        )


@dataclass(slots=True)
class GraphEdge:
    """A directed edge in the knowledge graph."""

    id: str
    source_id: str
    target_id: str
    relation_type: RelationType
    weight: float = 1.0
    context: str | None = None
    created_at: str = field(default_factory=_iso_now)
    last_reinforced_at: str | None = None
    reinforcement_count: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation_type": self.relation_type.value,
            "weight": self.weight,
            "context": self.context,
            "created_at": self.created_at,
            "last_reinforced_at": self.last_reinforced_at,
            "reinforcement_count": self.reinforcement_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GraphEdge:
        return cls(
            id=data["id"],
            source_id=data["source_id"],
            target_id=data["target_id"],
            relation_type=RelationType(data["relation_type"]),
            weight=float(data.get("weight", 1.0)),
            context=data.get("context"),
            created_at=data.get("created_at", _iso_now()),
            last_reinforced_at=data.get("last_reinforced_at"),
            reinforcement_count=int(data.get("reinforcement_count", 1)),
        )

    @classmethod
    def from_row(cls, row: tuple) -> GraphEdge:
        """Construct from DB row: (id, source_id, target_id, relation_type,
        weight, context, created_at, last_reinforced_at, reinforcement_count)."""
        return cls(
            id=row[0],
            source_id=row[1],
            target_id=row[2],
            relation_type=RelationType(row[3]),
            weight=float(row[4]) if row[4] is not None else 1.0,
            context=row[5],
            created_at=row[6] or _iso_now(),
            last_reinforced_at=row[7],
            reinforcement_count=int(row[8]) if row[8] is not None else 1,
        )

    def to_row_values(self) -> tuple:
        return (
            self.id,
            self.source_id,
            self.target_id,
            self.relation_type.value,
            self.weight,
            self.context,
            self.created_at,
            self.last_reinforced_at,
            self.reinforcement_count,
        )


# ---------------------------------------------------------------------------
# Episodes (experiential memory)
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class Episode:
    """A recorded work episode — captures what happened, how, and why."""

    id: str
    session_id: str
    project: str
    timestamp: str
    narrative: str
    approaches_tried: list[str] = field(default_factory=list)
    key_insight: str | None = None
    outcome: str = "routine"  # breakthrough | failure | routine | discovery
    impact_score: float = 0.5
    frustration_signals: int = 0
    user_corrections: list[str] = field(default_factory=list)
    concepts: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    duration_minutes: int | None = None
    similar_to: list[str] = field(default_factory=list)
    led_to: str | None = None
    contradicts: str | None = None
    created_at: str = field(default_factory=_iso_now)
    embedding_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "project": self.project,
            "timestamp": self.timestamp,
            "narrative": self.narrative,
            "approaches_tried": self.approaches_tried,
            "key_insight": self.key_insight,
            "outcome": self.outcome,
            "impact_score": self.impact_score,
            "frustration_signals": self.frustration_signals,
            "user_corrections": self.user_corrections,
            "concepts": self.concepts,
            "entities": self.entities,
            "tools_used": self.tools_used,
            "duration_minutes": self.duration_minutes,
            "similar_to": self.similar_to,
            "led_to": self.led_to,
            "contradicts": self.contradicts,
            "created_at": self.created_at,
            "embedding_id": self.embedding_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Episode:
        return cls(
            id=data["id"],
            session_id=data["session_id"],
            project=data["project"],
            timestamp=data["timestamp"],
            narrative=data["narrative"],
            approaches_tried=data.get("approaches_tried") or [],
            key_insight=data.get("key_insight"),
            outcome=data.get("outcome", "routine"),
            impact_score=float(data.get("impact_score", 0.5)),
            frustration_signals=int(data.get("frustration_signals", 0)),
            user_corrections=data.get("user_corrections") or [],
            concepts=data.get("concepts") or [],
            entities=data.get("entities") or [],
            tools_used=data.get("tools_used") or [],
            duration_minutes=data.get("duration_minutes"),
            similar_to=data.get("similar_to") or [],
            led_to=data.get("led_to"),
            contradicts=data.get("contradicts"),
            created_at=data.get("created_at", _iso_now()),
            embedding_id=data.get("embedding_id"),
        )

    @classmethod
    def from_row(cls, row: tuple) -> Episode:
        """Construct from DB row: (id, session_id, project, timestamp, narrative,
        approaches_tried_json, key_insight, outcome, impact_score,
        frustration_signals, user_corrections_json, concepts_json,
        entities_json, tools_used_json, duration_minutes, similar_to_json,
        led_to, contradicts, created_at, embedding_id)."""
        return cls(
            id=row[0],
            session_id=row[1],
            project=row[2],
            timestamp=row[3],
            narrative=row[4],
            approaches_tried=json_loads_safe(row[5], []),
            key_insight=row[6],
            outcome=row[7] or "routine",
            impact_score=float(row[8]) if row[8] is not None else 0.5,
            frustration_signals=int(row[9]) if row[9] is not None else 0,
            user_corrections=json_loads_safe(row[10], []),
            concepts=json_loads_safe(row[11], []),
            entities=json_loads_safe(row[12], []),
            tools_used=json_loads_safe(row[13], []),
            duration_minutes=int(row[14]) if row[14] is not None else None,
            similar_to=json_loads_safe(row[15], []),
            led_to=row[16],
            contradicts=row[17],
            created_at=row[18] or _iso_now(),
            embedding_id=row[19] if len(row) > 19 else None,
        )

    def to_row_values(self) -> tuple:
        return (
            self.id,
            self.session_id,
            self.project,
            self.timestamp,
            self.narrative,
            _json_dumps(self.approaches_tried),
            self.key_insight,
            self.outcome,
            self.impact_score,
            self.frustration_signals,
            _json_dumps(self.user_corrections),
            _json_dumps(self.concepts),
            _json_dumps(self.entities),
            _json_dumps(self.tools_used),
            self.duration_minutes,
            _json_dumps(self.similar_to),
            self.led_to,
            self.contradicts,
            self.created_at,
            self.embedding_id,
        )


# ---------------------------------------------------------------------------
# Skills (procedural memory)
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class Skill:
    """A learned reusable procedure with tracked effectiveness."""

    id: str
    name: str
    trigger_pattern: str
    steps: list[str] = field(default_factory=list)
    anti_patterns: list[str] = field(default_factory=list)
    times_used: int = 0
    success_rate: float = 0.0
    avg_steps_to_solve: float | None = None
    version: int = 1
    learned_from: list[str] = field(default_factory=list)
    last_refined_at: str | None = None
    projects: list[str] = field(default_factory=list)
    stack: list[str] = field(default_factory=list)
    related_skills: list[str] = field(default_factory=list)
    status: str = "draft"  # draft | active | mastered | deprecated
    created_at: str = field(default_factory=_iso_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "trigger_pattern": self.trigger_pattern,
            "steps": self.steps,
            "anti_patterns": self.anti_patterns,
            "times_used": self.times_used,
            "success_rate": self.success_rate,
            "avg_steps_to_solve": self.avg_steps_to_solve,
            "version": self.version,
            "learned_from": self.learned_from,
            "last_refined_at": self.last_refined_at,
            "projects": self.projects,
            "stack": self.stack,
            "related_skills": self.related_skills,
            "status": self.status,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Skill:
        return cls(
            id=data["id"],
            name=data["name"],
            trigger_pattern=data["trigger_pattern"],
            steps=data.get("steps") or [],
            anti_patterns=data.get("anti_patterns") or [],
            times_used=int(data.get("times_used", 0)),
            success_rate=float(data.get("success_rate", 0.0)),
            avg_steps_to_solve=data.get("avg_steps_to_solve"),
            version=int(data.get("version", 1)),
            learned_from=data.get("learned_from") or [],
            last_refined_at=data.get("last_refined_at"),
            projects=data.get("projects") or [],
            stack=data.get("stack") or [],
            related_skills=data.get("related_skills") or [],
            status=data.get("status", "draft"),
            created_at=data.get("created_at", _iso_now()),
        )

    @classmethod
    def from_row(cls, row: tuple) -> Skill:
        """Construct from DB row: (id, name, trigger_pattern, steps_json,
        anti_patterns_json, times_used, success_rate, avg_steps_to_solve,
        version, learned_from_json, last_refined_at, projects_json,
        stack_json, related_skills_json, status, created_at)."""
        return cls(
            id=row[0],
            name=row[1],
            trigger_pattern=row[2],
            steps=json_loads_safe(row[3], []),
            anti_patterns=json_loads_safe(row[4], []),
            times_used=int(row[5]) if row[5] is not None else 0,
            success_rate=float(row[6]) if row[6] is not None else 0.0,
            avg_steps_to_solve=float(row[7]) if row[7] is not None else None,
            version=int(row[8]) if row[8] is not None else 1,
            learned_from=json_loads_safe(row[9], []),
            last_refined_at=row[10],
            projects=json_loads_safe(row[11], []),
            stack=json_loads_safe(row[12], []),
            related_skills=json_loads_safe(row[13], []),
            status=row[14] or "draft",
            created_at=row[15] or _iso_now(),
        )

    def to_row_values(self) -> tuple:
        return (
            self.id,
            self.name,
            self.trigger_pattern,
            _json_dumps(self.steps),
            _json_dumps(self.anti_patterns),
            self.times_used,
            self.success_rate,
            self.avg_steps_to_solve,
            self.version,
            _json_dumps(self.learned_from),
            self.last_refined_at,
            _json_dumps(self.projects),
            _json_dumps(self.stack),
            _json_dumps(self.related_skills),
            self.status,
            self.created_at,
        )


@dataclass(slots=True)
class SkillUse:
    """A single recorded usage of a skill."""

    id: str
    skill_id: str
    episode_id: str | None = None
    success: bool = True
    steps_used: int | None = None
    notes: str | None = None
    used_at: str = field(default_factory=_iso_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "skill_id": self.skill_id,
            "episode_id": self.episode_id,
            "success": self.success,
            "steps_used": self.steps_used,
            "notes": self.notes,
            "used_at": self.used_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SkillUse:
        return cls(
            id=data["id"],
            skill_id=data["skill_id"],
            episode_id=data.get("episode_id"),
            success=bool(data.get("success", True)),
            steps_used=data.get("steps_used"),
            notes=data.get("notes"),
            used_at=data.get("used_at", _iso_now()),
        )

    @classmethod
    def from_row(cls, row: tuple) -> SkillUse:
        """Construct from DB row: (id, skill_id, episode_id, success,
        steps_used, notes, used_at)."""
        return cls(
            id=row[0],
            skill_id=row[1],
            episode_id=row[2],
            success=bool(row[3]) if row[3] is not None else True,
            steps_used=int(row[4]) if row[4] is not None else None,
            notes=row[5],
            used_at=row[6] or _iso_now(),
        )

    def to_row_values(self) -> tuple:
        return (
            self.id,
            self.skill_id,
            self.episode_id,
            int(self.success),
            self.steps_used,
            self.notes,
            self.used_at,
        )


# ---------------------------------------------------------------------------
# Competency & blind spots (metacognitive memory)
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class CompetencyScore:
    """Tracked competency level in a specific domain."""

    domain: str
    level: float = 0.5       # 0.0 - 1.0
    confidence: float = 0.5  # 0.0 - 1.0
    based_on: int = 0
    trend: str = "unknown"   # improving | stable | declining | stable_low | unknown
    last_updated: str = field(default_factory=_iso_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "level": self.level,
            "confidence": self.confidence,
            "based_on": self.based_on,
            "trend": self.trend,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CompetencyScore:
        return cls(
            domain=data["domain"],
            level=float(data.get("level", 0.5)),
            confidence=float(data.get("confidence", 0.5)),
            based_on=int(data.get("based_on", 0)),
            trend=data.get("trend", "unknown"),
            last_updated=data.get("last_updated", _iso_now()),
        )

    @classmethod
    def from_row(cls, row: tuple) -> CompetencyScore:
        """Construct from DB row: (domain, level, confidence, based_on,
        trend, last_updated)."""
        return cls(
            domain=row[0],
            level=float(row[1]) if row[1] is not None else 0.5,
            confidence=float(row[2]) if row[2] is not None else 0.5,
            based_on=int(row[3]) if row[3] is not None else 0,
            trend=row[4] or "unknown",
            last_updated=row[5] or _iso_now(),
        )

    def to_row_values(self) -> tuple:
        return (
            self.domain,
            self.level,
            self.confidence,
            self.based_on,
            self.trend,
            self.last_updated,
        )


@dataclass(slots=True)
class BlindSpot:
    """A recognized knowledge gap or recurring weakness."""

    id: str
    description: str
    domains: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    severity: float = 0.5   # 0.0 - 1.0
    status: str = "active"   # active | resolved | monitoring
    discovered_at: str = field(default_factory=_iso_now)
    resolved_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "domains": self.domains,
            "evidence": self.evidence,
            "severity": self.severity,
            "status": self.status,
            "discovered_at": self.discovered_at,
            "resolved_at": self.resolved_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BlindSpot:
        return cls(
            id=data["id"],
            description=data["description"],
            domains=data.get("domains") or [],
            evidence=data.get("evidence") or [],
            severity=float(data.get("severity", 0.5)),
            status=data.get("status", "active"),
            discovered_at=data.get("discovered_at", _iso_now()),
            resolved_at=data.get("resolved_at"),
        )

    @classmethod
    def from_row(cls, row: tuple) -> BlindSpot:
        """Construct from DB row: (id, description, domains_json,
        evidence_json, severity, status, discovered_at, resolved_at)."""
        return cls(
            id=row[0],
            description=row[1],
            domains=json_loads_safe(row[2], []),
            evidence=json_loads_safe(row[3], []),
            severity=float(row[4]) if row[4] is not None else 0.5,
            status=row[5] or "active",
            discovered_at=row[6] or _iso_now(),
            resolved_at=row[7],
        )

    def to_row_values(self) -> tuple:
        return (
            self.id,
            self.description,
            _json_dumps(self.domains),
            _json_dumps(self.evidence),
            self.severity,
            self.status,
            self.discovered_at,
            self.resolved_at,
        )


# ---------------------------------------------------------------------------
# User model
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class UserModelEntry:
    """A single key-value entry in the user preference/behavior model."""

    key: str
    value: Any
    confidence: float = 0.5
    evidence_count: int = 1
    last_updated: str = field(default_factory=_iso_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "value": self.value,
            "confidence": self.confidence,
            "evidence_count": self.evidence_count,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UserModelEntry:
        return cls(
            key=data["key"],
            value=data["value"],
            confidence=float(data.get("confidence", 0.5)),
            evidence_count=int(data.get("evidence_count", 1)),
            last_updated=data.get("last_updated", _iso_now()),
        )

    @classmethod
    def from_row(cls, row: tuple) -> UserModelEntry:
        """Construct from DB row: (key, value_json, confidence,
        evidence_count, last_updated)."""
        return cls(
            key=row[0],
            value=json_loads_safe(row[1]),
            confidence=float(row[2]) if row[2] is not None else 0.5,
            evidence_count=int(row[3]) if row[3] is not None else 1,
            last_updated=row[4] or _iso_now(),
        )

    def to_row_values(self) -> tuple:
        return (
            self.key,
            _json_dumps(self.value),
            self.confidence,
            self.evidence_count,
            self.last_updated,
        )


# ---------------------------------------------------------------------------
# Ingestion pipeline
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class IngestItem:
    """An item queued for ingestion into the knowledge base."""

    id: str
    source: str
    content_type: str
    raw_content: bytes | None = None
    text_content: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    status: str = "pending"  # pending | processing | completed | failed
    error_message: str | None = None
    created_at: str = field(default_factory=_iso_now)
    processed_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "content_type": self.content_type,
            "raw_content": self.raw_content.hex() if self.raw_content else None,
            "text_content": self.text_content,
            "metadata": self.metadata,
            "status": self.status,
            "error_message": self.error_message,
            "created_at": self.created_at,
            "processed_at": self.processed_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IngestItem:
        raw = data.get("raw_content")
        if isinstance(raw, str):
            raw = bytes.fromhex(raw)
        return cls(
            id=data["id"],
            source=data["source"],
            content_type=data["content_type"],
            raw_content=raw,
            text_content=data.get("text_content"),
            metadata=data.get("metadata") or {},
            status=data.get("status", "pending"),
            error_message=data.get("error_message"),
            created_at=data.get("created_at", _iso_now()),
            processed_at=data.get("processed_at"),
        )

    @classmethod
    def from_row(cls, row: tuple) -> IngestItem:
        """Construct from DB row: (id, source, content_type, raw_content,
        text_content, metadata_json, status, error_message, created_at,
        processed_at)."""
        return cls(
            id=row[0],
            source=row[1],
            content_type=row[2],
            raw_content=row[3] if isinstance(row[3], bytes) else None,
            text_content=row[4],
            metadata=json_loads_safe(row[5], {}),
            status=row[6] or "pending",
            error_message=row[7],
            created_at=row[8] or _iso_now(),
            processed_at=row[9],
        )

    def to_row_values(self) -> tuple:
        return (
            self.id,
            self.source,
            self.content_type,
            self.raw_content,
            self.text_content,
            _json_dumps(self.metadata),
            self.status,
            self.error_message,
            self.created_at,
            self.processed_at,
        )


@dataclass(slots=True)
class Chunk:
    """A processed chunk of ingested content, ready for embedding & retrieval."""

    id: str
    parent_id: str
    content: str
    summary: str
    chunk_index: int = 0
    concepts: list[str] = field(default_factory=list)
    capabilities: list[str] = field(default_factory=list)
    composable_with: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    relations: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: bytes | None = None
    binary_vector: bytes | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "parent_id": self.parent_id,
            "content": self.content,
            "summary": self.summary,
            "chunk_index": self.chunk_index,
            "concepts": self.concepts,
            "capabilities": self.capabilities,
            "composable_with": self.composable_with,
            "entities": self.entities,
            "relations": self.relations,
            "metadata": self.metadata,
            "embedding": self.embedding.hex() if self.embedding else None,
            "binary_vector": (
                self.binary_vector.hex() if self.binary_vector else None
            ),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Chunk:
        emb = data.get("embedding")
        if isinstance(emb, str):
            emb = bytes.fromhex(emb)
        bv = data.get("binary_vector")
        if isinstance(bv, str):
            bv = bytes.fromhex(bv)
        return cls(
            id=data["id"],
            parent_id=data["parent_id"],
            content=data["content"],
            summary=data["summary"],
            chunk_index=int(data.get("chunk_index", 0)),
            concepts=data.get("concepts") or [],
            capabilities=data.get("capabilities") or [],
            composable_with=data.get("composable_with") or [],
            entities=data.get("entities") or [],
            relations=data.get("relations") or [],
            metadata=data.get("metadata") or {},
            embedding=emb,
            binary_vector=bv,
        )

    @classmethod
    def from_row(cls, row: tuple) -> Chunk:
        """Construct from DB row: (id, parent_id, content, summary,
        chunk_index, concepts_json, capabilities_json, composable_with_json,
        entities_json, relations_json, metadata_json, embedding, binary_vector)."""
        return cls(
            id=row[0],
            parent_id=row[1],
            content=row[2],
            summary=row[3],
            chunk_index=int(row[4]) if row[4] is not None else 0,
            concepts=json_loads_safe(row[5], []),
            capabilities=json_loads_safe(row[6], []),
            composable_with=json_loads_safe(row[7], []),
            entities=json_loads_safe(row[8], []),
            relations=json_loads_safe(row[9], []),
            metadata=json_loads_safe(row[10], {}),
            embedding=row[11] if isinstance(row[11], bytes) else None,
            binary_vector=row[12] if len(row) > 12 and isinstance(row[12], bytes) else None,
        )

    def to_row_values(self) -> tuple:
        return (
            self.id,
            self.parent_id,
            self.content,
            self.summary,
            self.chunk_index,
            _json_dumps(self.concepts),
            _json_dumps(self.capabilities),
            _json_dumps(self.composable_with),
            _json_dumps(self.entities),
            _json_dumps(self.relations),
            _json_dumps(self.metadata),
            self.embedding,
            self.binary_vector,
        )


# ---------------------------------------------------------------------------
# Reflection & proposals
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ReflectionReport:
    """A periodic self-reflection summary."""

    id: str
    period_start: str
    period_end: str
    type: str = "session"  # session | periodic | weekly | manual
    edges_strengthened: int = 0
    clusters_found: int = 0
    skills_proposed: int = 0
    contradictions: int = 0
    archived: int = 0
    focus_areas: list[str] = field(default_factory=list)
    key_findings: list[str] = field(default_factory=list)
    proposed_changes: list[dict[str, Any]] = field(default_factory=list)
    created_at: str = field(default_factory=_iso_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "period_start": self.period_start,
            "period_end": self.period_end,
            "type": self.type,
            "edges_strengthened": self.edges_strengthened,
            "clusters_found": self.clusters_found,
            "skills_proposed": self.skills_proposed,
            "contradictions": self.contradictions,
            "archived": self.archived,
            "focus_areas": self.focus_areas,
            "key_findings": self.key_findings,
            "proposed_changes": self.proposed_changes,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReflectionReport:
        return cls(
            id=data["id"],
            period_start=data["period_start"],
            period_end=data["period_end"],
            type=data.get("type", "session"),
            edges_strengthened=int(data.get("edges_strengthened", 0)),
            clusters_found=int(data.get("clusters_found", 0)),
            skills_proposed=int(data.get("skills_proposed", 0)),
            contradictions=int(data.get("contradictions", 0)),
            archived=int(data.get("archived", 0)),
            focus_areas=data.get("focus_areas") or [],
            key_findings=data.get("key_findings") or [],
            proposed_changes=data.get("proposed_changes") or [],
            created_at=data.get("created_at", _iso_now()),
        )

    @classmethod
    def from_row(cls, row: tuple) -> ReflectionReport:
        """Construct from DB row: (id, period_start, period_end, type,
        edges_strengthened, clusters_found, skills_proposed,
        contradictions, archived, focus_areas_json, key_findings_json,
        proposed_changes_json, created_at)."""
        return cls(
            id=row[0],
            period_start=row[1],
            period_end=row[2],
            type=row[3] or "session",
            edges_strengthened=int(row[4]) if row[4] is not None else 0,
            clusters_found=int(row[5]) if row[5] is not None else 0,
            skills_proposed=int(row[6]) if row[6] is not None else 0,
            contradictions=int(row[7]) if row[7] is not None else 0,
            archived=int(row[8]) if row[8] is not None else 0,
            focus_areas=json_loads_safe(row[9], []),
            key_findings=json_loads_safe(row[10], []),
            proposed_changes=json_loads_safe(row[11], []),
            created_at=row[12] or _iso_now(),
        )

    def to_row_values(self) -> tuple:
        return (
            self.id,
            self.period_start,
            self.period_end,
            self.type,
            self.edges_strengthened,
            self.clusters_found,
            self.skills_proposed,
            self.contradictions,
            self.archived,
            _json_dumps(self.focus_areas),
            _json_dumps(self.key_findings),
            _json_dumps(self.proposed_changes),
            self.created_at,
        )


@dataclass(slots=True)
class Proposal:
    """A proposed change generated by reflection (rule, skill update, etc.)."""

    id: str
    type: str  # rule | skill | claude_md_update | blind_spot
    content: str
    evidence: list[str] = field(default_factory=list)
    confidence: float | None = None
    status: str = "pending"  # pending | approved | rejected
    created_at: str = field(default_factory=_iso_now)
    reviewed_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "content": self.content,
            "evidence": self.evidence,
            "confidence": self.confidence,
            "status": self.status,
            "created_at": self.created_at,
            "reviewed_at": self.reviewed_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Proposal:
        return cls(
            id=data["id"],
            type=data["type"],
            content=data["content"],
            evidence=data.get("evidence") or [],
            confidence=data.get("confidence"),
            status=data.get("status", "pending"),
            created_at=data.get("created_at", _iso_now()),
            reviewed_at=data.get("reviewed_at"),
        )

    @classmethod
    def from_row(cls, row: tuple) -> Proposal:
        """Construct from DB row: (id, type, content, evidence_json,
        confidence, status, created_at, reviewed_at)."""
        return cls(
            id=row[0],
            type=row[1],
            content=row[2],
            evidence=json_loads_safe(row[3], []),
            confidence=float(row[4]) if row[4] is not None else None,
            status=row[5] or "pending",
            created_at=row[6] or _iso_now(),
            reviewed_at=row[7],
        )

    def to_row_values(self) -> tuple:
        return (
            self.id,
            self.type,
            self.content,
            _json_dumps(self.evidence),
            self.confidence,
            self.status,
            self.created_at,
            self.reviewed_at,
        )


# ---------------------------------------------------------------------------
# Associative retrieval results (read-only, no DB storage)
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class AssociationResult:
    """Result of an associative / spreading-activation query."""

    query_concepts: list[str] = field(default_factory=list)
    activated_nodes: int = 0
    memories: list[dict[str, Any]] = field(default_factory=list)
    composition: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_concepts": self.query_concepts,
            "activated_nodes": self.activated_nodes,
            "memories": self.memories,
            "composition": self.composition,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AssociationResult:
        return cls(
            query_concepts=data.get("query_concepts") or [],
            activated_nodes=int(data.get("activated_nodes", 0)),
            memories=data.get("memories") or [],
            composition=data.get("composition"),
        )


@dataclass(slots=True)
class Composition:
    """A composed answer from multiple knowledge sources."""

    sources: list[dict[str, Any]] = field(default_factory=list)
    coverage_percent: float = 0.0
    conflicts: list[dict[str, Any]] = field(default_factory=list)
    integration_plan: str | None = None
    gaps: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sources": self.sources,
            "coverage_percent": self.coverage_percent,
            "conflicts": self.conflicts,
            "integration_plan": self.integration_plan,
            "gaps": self.gaps,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Composition:
        return cls(
            sources=data.get("sources") or [],
            coverage_percent=float(data.get("coverage_percent", 0.0)),
            conflicts=data.get("conflicts") or [],
            integration_plan=data.get("integration_plan"),
            gaps=data.get("gaps") or [],
        )


@dataclass(slots=True)
class ContextBundle:
    """A pre-assembled context package for session startup."""

    knowledge: list[dict[str, Any]] = field(default_factory=list)
    competency: dict[str, Any] | None = None
    blind_spots: list[dict[str, Any]] = field(default_factory=list)
    predicted_needs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "knowledge": self.knowledge,
            "competency": self.competency,
            "blind_spots": self.blind_spots,
            "predicted_needs": self.predicted_needs,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ContextBundle:
        return cls(
            knowledge=data.get("knowledge") or [],
            competency=data.get("competency"),
            blind_spots=data.get("blind_spots") or [],
            predicted_needs=data.get("predicted_needs") or [],
        )


@dataclass(slots=True)
class SessionSignals:
    """Aggregated user signals from a session for satisfaction tracking."""

    correction_count: int = 0
    retry_count: int = 0
    positive_count: int = 0
    total_messages: int = 0
    satisfaction_score: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return {
            "correction_count": self.correction_count,
            "retry_count": self.retry_count,
            "positive_count": self.positive_count,
            "total_messages": self.total_messages,
            "satisfaction_score": self.satisfaction_score,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionSignals:
        return cls(
            correction_count=int(data.get("correction_count", 0)),
            retry_count=int(data.get("retry_count", 0)),
            positive_count=int(data.get("positive_count", 0)),
            total_messages=int(data.get("total_messages", 0)),
            satisfaction_score=float(data.get("satisfaction_score", 0.5)),
        )


# ---------------------------------------------------------------------------
# Type alias for sqlite3.Row compatibility
# ---------------------------------------------------------------------------

# sqlite3.Row supports index access, so from_row() works with both
# tuples and sqlite3.Row objects. This alias is for documentation only.
sqlite3_Row = Any


# ---------------------------------------------------------------------------
# Convenience: all model classes for iteration / registration
# ---------------------------------------------------------------------------

ALL_MODELS: list[type] = [
    GraphNode,
    GraphEdge,
    Episode,
    Skill,
    SkillUse,
    CompetencyScore,
    BlindSpot,
    UserModelEntry,
    IngestItem,
    Chunk,
    ReflectionReport,
    Proposal,
    AssociationResult,
    Composition,
    ContextBundle,
    SessionSignals,
]
