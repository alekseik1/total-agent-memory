#!/usr/bin/env python3
"""Context Layers — L0/L1/L2/L3 hierarchical context loading system.

Inspired by MemPalace but optimized for token efficiency.
Provides always-loaded identity (L0), critical facts (L1),
project context (L2), and on-demand deep search (L3, existing).

Layers:
  L0 (~50-100 tokens)  — Identity, ALWAYS loaded
  L1 (~200-500 tokens)  — Critical facts, loaded on session start
  L2 (~300-500 tokens)  — Project context, loaded on project switch
  L3 (variable)         — Deep search via memory_recall (existing)

Integration points (do NOT modify these files):
  - src/cognitive/engine.py: on_session_start() should call wake_up()
  - src/server.py: memory_context_build tool should use layers
"""

from __future__ import annotations

import json
import os
import sqlite3
import re
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paths import memory_dir  # noqa: E402

MEMORY_DIR = memory_dir()
DB_PATH = MEMORY_DIR / "memory.db"
IDENTITY_PATH = MEMORY_DIR / "vito_identity.txt"
LAYER1_PATH = MEMORY_DIR / "vito_layer1.txt"
ENTITY_REGISTRY_PATH = MEMORY_DIR / "entity_registry.json"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

L1_MAX_CHARS = 2000
L2_MAX_CHARS = 1500
RECORD_CONTENT_LIMIT = 100
TOKENS_PER_CHAR = 0.3  # rough estimate for mixed en/ru text


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _estimate_tokens(text: str) -> int:
    """Rough token estimate for mixed English/Russian text."""
    return int(len(text) * TOKENS_PER_CHAR)


def _truncate(text: str, limit: int) -> str:
    """Truncate text to limit chars, adding ellipsis if cut."""
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def _get_db(db_path: str | Path | None = None) -> sqlite3.Connection:
    """Open a read-only SQLite connection."""
    path = str(db_path or DB_PATH)
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _safe_json_loads(text: str | None) -> list[str] | list[Any]:
    """Parse JSON text or return empty list."""
    if not text:
        return []
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return []


# ---------------------------------------------------------------------------
# Compact record format
# ---------------------------------------------------------------------------

def compact_record(record: dict[str, Any]) -> str:
    """Compress a knowledge record into pipe-delimited format.

    Input: dict with keys like type, content, project, confidence, tags.
    Output: ~30-50 tokens instead of ~150-200.

    Format: T:type | content[:100] | P:project | W:confidence | #tag1,tag2
    """
    rec_type = record.get("type", "?")
    content = record.get("content", "")
    project = record.get("project", "general")
    confidence = record.get("confidence", 1.0)
    tags = record.get("tags", [])

    # Normalize tags
    if isinstance(tags, str):
        tags = _safe_json_loads(tags)

    # Clean content: collapse whitespace, strip newlines
    content_clean = re.sub(r"\s+", " ", content).strip()
    content_trunc = _truncate(content_clean, RECORD_CONTENT_LIMIT)

    # Build tag string
    tag_str = ",".join(tags[:5]) if tags else ""

    parts = [f"T:{rec_type}", content_trunc, f"P:{project}", f"W:{confidence:.1f}"]
    if tag_str:
        parts.append(f"#{tag_str}")

    return " | ".join(parts)


# ---------------------------------------------------------------------------
# Entity shortcoding
# ---------------------------------------------------------------------------

def _load_entity_registry() -> dict[str, str]:
    """Load or create entity registry (name -> 3-letter code)."""
    if ENTITY_REGISTRY_PATH.exists():
        try:
            data = json.loads(ENTITY_REGISTRY_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_entity_registry(registry: dict[str, str]) -> None:
    """Persist entity registry to disk."""
    ENTITY_REGISTRY_PATH.write_text(
        json.dumps(registry, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _build_entity_registry(db_path: str | Path | None = None) -> dict[str, str]:
    """Build entity registry from frequently mentioned projects and names.

    Scans knowledge DB for project names and high-frequency terms,
    assigns 3-letter codes to each.
    """
    registry = _load_entity_registry()
    used_codes: set[str] = set(registry.values())

    conn = _get_db(db_path)
    try:
        # Get all project names with counts
        rows = conn.execute(
            "SELECT project, COUNT(*) as cnt FROM knowledge "
            "WHERE status='active' AND project != 'general' "
            "GROUP BY project ORDER BY cnt DESC LIMIT 30"
        ).fetchall()

        for row in rows:
            name = row["project"]
            if name and name not in registry:
                code = _generate_code(name, used_codes)
                registry[name] = code
                used_codes.add(code)
    finally:
        conn.close()

    _save_entity_registry(registry)
    return registry


def _generate_code(name: str, used: set[str]) -> str:
    """Generate a unique 3-letter uppercase code for a name."""
    # Try first 3 chars
    clean = re.sub(r"[^a-zA-Z0-9]", "", name).upper()
    if len(clean) >= 3:
        candidate = clean[:3]
        if candidate not in used:
            return candidate

    # Try consonants
    consonants = re.sub(r"[AEIOU]", "", clean)
    if len(consonants) >= 3:
        candidate = consonants[:3]
        if candidate not in used:
            return candidate

    # Try first + last + len
    if len(clean) >= 2:
        candidate = f"{clean[0]}{clean[-1]}{len(clean) % 10}"
        if candidate.upper() not in used:
            return candidate.upper()

    # Fallback: incremental
    for i in range(100):
        candidate = f"{clean[:2]}{i}" if len(clean) >= 2 else f"X{clean[:1]}{i}"
        if candidate not in used:
            return candidate

    return clean[:3] or "ZZZ"


def encode_entities(text: str, db_path: str | Path | None = None) -> str:
    """Replace frequent entity names with 3-letter shortcodes.

    Useful for compressing context layers to save tokens.
    """
    registry = _build_entity_registry(db_path)
    result = text
    # Sort by name length descending to avoid partial replacements
    for name, code in sorted(registry.items(), key=lambda x: -len(x[0])):
        if len(name) > len(code) + 1:  # only replace if actually saves space
            result = result.replace(name, f"[{code}]")
    return result


def decode_entities(text: str, db_path: str | Path | None = None) -> str:
    """Reverse shortcode mapping — expand [CODE] back to full names."""
    registry = _build_entity_registry(db_path)
    reverse = {f"[{code}]": name for name, code in registry.items()}
    result = text
    for code, name in reverse.items():
        result = result.replace(code, name)
    return result


# ---------------------------------------------------------------------------
# L0 — Identity
# ---------------------------------------------------------------------------

def load_identity(identity_path: str | Path | None = None) -> str:
    """Load L0 identity text from file.

    Returns the raw identity string (~50-100 tokens).
    Falls back to a minimal default if file is missing.
    """
    path = Path(identity_path or IDENTITY_PATH)
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return "Vito — autonomous memory brain. No identity file found."


# ---------------------------------------------------------------------------
# L1 — Critical Facts
# ---------------------------------------------------------------------------

def generate_layer1(db_path: str | Path | None = None) -> str:
    """Generate L1 critical facts from the knowledge DB.

    Combines:
      1. Top 10 most-recalled knowledge records
      2. Active SOUL rules
      3. Recent decisions (last 7 days)

    Output is capped at L1_MAX_CHARS and saved to vito_layer1.txt.
    """
    conn = _get_db(db_path)
    lines: list[str] = []
    total_len = 0

    try:
        # --- Top recalled knowledge ---
        lines.append("## TOP KNOWLEDGE")
        rows = conn.execute(
            "SELECT type, content, project, confidence, tags "
            "FROM knowledge WHERE status='active' AND recall_count > 0 "
            "ORDER BY recall_count DESC LIMIT 10"
        ).fetchall()

        for row in rows:
            rec = compact_record(dict(row))
            if total_len + len(rec) + 1 > L1_MAX_CHARS:
                break
            lines.append(rec)
            total_len += len(rec) + 1

        # --- Active SOUL rules ---
        lines.append("## SOUL RULES")
        rules = conn.execute(
            "SELECT content, category, scope, priority, project, success_rate "
            "FROM rules WHERE status='active' "
            "ORDER BY priority DESC, success_rate DESC LIMIT 10"
        ).fetchall()

        for rule in rules:
            content_trunc = _truncate(
                re.sub(r"\s+", " ", rule["content"]).strip(),
                RECORD_CONTENT_LIMIT,
            )
            line = (
                f"R:{rule['category']} | {content_trunc} "
                f"| S:{rule['scope']} | P:{rule['priority']} "
                f"| SR:{rule['success_rate']:.0%}"
            )
            if total_len + len(line) + 1 > L1_MAX_CHARS:
                break
            lines.append(line)
            total_len += len(line) + 1

        # --- Recent decisions (7 days) ---
        lines.append("## RECENT DECISIONS")
        cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
        decisions = conn.execute(
            "SELECT content, project, confidence, tags "
            "FROM knowledge WHERE status='active' AND type='decision' "
            "AND created_at > ? ORDER BY created_at DESC LIMIT 5",
            (cutoff,),
        ).fetchall()

        for dec in decisions:
            rec = compact_record(dict(dec))
            if total_len + len(rec) + 1 > L1_MAX_CHARS:
                break
            lines.append(rec)
            total_len += len(rec) + 1

    finally:
        conn.close()

    output = "\n".join(lines)

    # Save to file
    LAYER1_PATH.write_text(output, encoding="utf-8")

    return output


# ---------------------------------------------------------------------------
# L2 — Project Context
# ---------------------------------------------------------------------------

def generate_layer2(db_path: str | Path | None = None, project: str = "general") -> str:
    """Generate L2 project-specific context.

    Combines:
      1. Conventions for the project
      2. Recent solutions (last 14 days)
      3. Active blind spots (insights with low confidence)

    Output capped at L2_MAX_CHARS.
    """
    conn = _get_db(db_path)
    lines: list[str] = []
    total_len = 0

    try:
        # --- Conventions ---
        lines.append(f"## [{project}] CONVENTIONS")
        conventions = conn.execute(
            "SELECT content, confidence, tags FROM knowledge "
            "WHERE status='active' AND type='convention' AND project=? "
            "ORDER BY recall_count DESC LIMIT 8",
            (project,),
        ).fetchall()

        for conv in conventions:
            rec = compact_record({**dict(conv), "type": "convention", "project": project})
            if total_len + len(rec) + 1 > L2_MAX_CHARS:
                break
            lines.append(rec)
            total_len += len(rec) + 1

        # --- Recent solutions (14 days) ---
        lines.append(f"## [{project}] RECENT SOLUTIONS")
        cutoff = (datetime.now(timezone.utc) - timedelta(days=14)).isoformat()
        solutions = conn.execute(
            "SELECT content, confidence, tags FROM knowledge "
            "WHERE status='active' AND type='solution' AND project=? "
            "AND created_at > ? ORDER BY created_at DESC LIMIT 8",
            (project, cutoff),
        ).fetchall()

        for sol in solutions:
            rec = compact_record({**dict(sol), "type": "solution", "project": project})
            if total_len + len(rec) + 1 > L2_MAX_CHARS:
                break
            lines.append(rec)
            total_len += len(rec) + 1

        # --- Blind spots (low-confidence insights) ---
        lines.append(f"## [{project}] BLIND SPOTS")
        blind = conn.execute(
            "SELECT content, category, confidence FROM insights "
            "WHERE status='active' AND project=? AND confidence < 0.5 "
            "ORDER BY importance DESC LIMIT 5",
            (project,),
        ).fetchall()

        for spot in blind:
            content_trunc = _truncate(
                re.sub(r"\s+", " ", spot["content"]).strip(),
                RECORD_CONTENT_LIMIT,
            )
            line = f"BLIND:{spot['category']} | {content_trunc} | C:{spot['confidence']:.1f}"
            if total_len + len(line) + 1 > L2_MAX_CHARS:
                break
            lines.append(line)
            total_len += len(line) + 1

    finally:
        conn.close()

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Wake-up — unified context loader
# ---------------------------------------------------------------------------

def wake_up(
    db_path: str | Path | None = None,
    project: str | None = None,
) -> dict[str, Any]:
    """Load all relevant context layers for session start.

    Returns a dict with:
      - l0: identity text (always present)
      - l1: critical facts (regenerated from DB)
      - l2: project context (if project given, else None)
      - total_tokens: estimated token count
      - layers_loaded: list of layer names loaded

    Usage:
        context = wake_up(project="impatient")
        # Feed context["l0"] + context["l1"] + context["l2"] to system prompt
    """
    l0 = load_identity()
    l1 = generate_layer1(db_path)
    l2 = generate_layer2(db_path, project) if project else None

    combined = l0 + "\n" + l1
    if l2:
        combined += "\n" + l2

    layers = ["l0", "l1"]
    if l2:
        layers.append("l2")

    return {
        "l0": l0,
        "l1": l1,
        "l2": l2,
        "total_tokens": _estimate_tokens(combined),
        "total_chars": len(combined),
        "layers_loaded": layers,
    }


# ---------------------------------------------------------------------------
# CLI — for testing and manual generation
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    proj = sys.argv[1] if len(sys.argv) > 1 else None
    result = wake_up(project=proj)

    print(f"=== L0 Identity ({_estimate_tokens(result['l0'])} tokens) ===")
    print(result["l0"])
    print()
    print(f"=== L1 Critical Facts ({_estimate_tokens(result['l1'])} tokens) ===")
    print(result["l1"])
    print()
    if result["l2"]:
        print(f"=== L2 Project Context ({_estimate_tokens(result['l2'])} tokens) ===")
        print(result["l2"])
        print()
    print(f"Total: ~{result['total_tokens']} tokens, {result['total_chars']} chars")
    print(f"Layers: {result['layers_loaded']}")
