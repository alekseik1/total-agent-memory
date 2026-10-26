#!/usr/bin/env python3
"""
Obsidian Bidirectional Sync — syncs between Obsidian vault and memory.db.

Memory -> Obsidian: exports knowledge records as grouped .md files.
Obsidian -> Memory: imports .md files with `memory: true` frontmatter.

Usage:
    python src/tools/obsidian_sync.py --vault /path/to/vault [--project NAME]
"""

import argparse
import hashlib
import json
import os
import re
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from background_writes import save_background_knowledge  # noqa: E402
from paths import memory_dir  # noqa: E402

MEMORY_DIR = str(memory_dir())
DB_PATH = os.path.join(MEMORY_DIR, "memory.db")
SYNC_STATE_PATH = os.path.join(MEMORY_DIR, "obsidian_sync_state.json")

# Knowledge types in display order
TYPE_ORDER: list[str] = ["decision", "solution", "convention", "fact", "lesson", "reflection"]
# Types a vault note may declare; anything else is stored as a plain fact.
IMPORTABLE_TYPES = frozenset(TYPE_ORDER)
DEFAULT_IMPORT_TYPE = "fact"

# Frontmatter regex: content between --- delimiters at file start
FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def load_sync_state() -> dict:
    """Load sync state from JSON file."""
    if os.path.exists(SYNC_STATE_PATH):
        try:
            with open(SYNC_STATE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {"last_sync": None, "exported_ids": [], "imported_files": []}


def save_sync_state(state: dict) -> None:
    """Persist sync state to JSON file."""
    os.makedirs(os.path.dirname(SYNC_STATE_PATH), exist_ok=True)
    with open(SYNC_STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Parse YAML-like frontmatter from markdown text.

    Returns (metadata_dict, body_text). Handles simple key: value pairs
    and basic lists without requiring PyYAML.
    """
    match = FRONTMATTER_RE.match(text)
    if not match:
        return {}, text

    raw = match.group(1)
    body = text[match.end():]
    meta: dict = {}

    current_key: str | None = None
    current_list: list[str] | None = None

    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        # List item under current key
        if stripped.startswith("- ") and current_key is not None:
            if current_list is None:
                current_list = []
            current_list.append(stripped[2:].strip().strip('"').strip("'"))
            meta[current_key] = current_list
            continue

        # Key: value pair
        if ":" in stripped:
            if current_list is not None:
                current_list = None

            key, _, value = stripped.partition(":")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            current_key = key

            if value:
                # Handle inline lists: [a, b, c]
                if value.startswith("[") and value.endswith("]"):
                    items = [v.strip().strip('"').strip("'") for v in value[1:-1].split(",") if v.strip()]
                    meta[key] = items
                elif value.lower() == "true":
                    meta[key] = True
                elif value.lower() == "false":
                    meta[key] = False
                else:
                    meta[key] = value
                current_list = None
            else:
                current_list = []
                meta[key] = current_list

    return meta, body


def export_memory_to_obsidian(
    vault_path: Path,
    project_filter: str | None,
    last_sync: str | None,
) -> int:
    """Export knowledge records from memory.db to Obsidian vault.

    Groups records by project and type into separate .md files.
    Returns count of exported records.
    """
    if not os.path.exists(DB_PATH):
        print(f"Error: memory.db not found at {DB_PATH}", file=sys.stderr)
        return 0

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    query = "SELECT * FROM knowledge WHERE status = 'active'"
    params: list = []

    if project_filter:
        query += " AND project = ?"
        params.append(project_filter)

    if last_sync:
        query += " AND (created_at > ? OR last_confirmed > ?)"
        params.extend([last_sync, last_sync])

    query += " ORDER BY project, type, created_at DESC"
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        print("No records to export.")
        return 0

    # Group by project -> type
    grouped: dict[str, dict[str, list]] = {}
    for row in rows:
        proj = row["project"] or "general"
        rtype = row["type"] or "fact"
        grouped.setdefault(proj, {}).setdefault(rtype, []).append(dict(row))

    exported = 0
    memory_dir = vault_path / "Memory"
    memory_dir.mkdir(parents=True, exist_ok=True)

    export_root = memory_dir.resolve()
    for project, types_dict in grouped.items():
        proj_dir = memory_dir / _safe_filename(project)
        if not proj_dir.resolve().is_relative_to(export_root):
            print(f"  Skipped project {project!r}: export path escapes the Memory/ directory", file=sys.stderr)
            continue
        proj_dir.mkdir(parents=True, exist_ok=True)

        for rtype in TYPE_ORDER:
            if rtype not in types_dict:
                continue

            records = types_dict[rtype]
            filename = f"{rtype}s.md"
            filepath = proj_dir / filename

            lines: list[str] = [
                f"---",
                f"project: {project}",
                f"type: {rtype}",
                f"generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                f"count: {len(records)}",
                f"---",
                f"",
                f"# {rtype.capitalize()}s — {project}",
                f"",
            ]

            for rec in records:
                tags = _parse_tags(rec.get("tags", "[]"))
                confidence = rec.get("confidence", 1.0)
                created = rec.get("created_at", "unknown")

                lines.append(f"## [{rec['id']}] {created}")
                lines.append(f"")
                lines.append(f"**Confidence:** {confidence:.1f} | **Tags:** {', '.join(tags)}")
                lines.append(f"")
                lines.append(rec.get("content", "").strip())
                if rec.get("context"):
                    lines.append(f"")
                    lines.append(f"> Context: {rec['context']}")
                lines.append(f"")
                lines.append(f"---")
                lines.append(f"")
                exported += 1

            filepath.write_text("\n".join(lines), encoding="utf-8")
            print(f"  Exported {len(records)} {rtype}s -> {filepath.relative_to(vault_path)}")

    return exported


def import_obsidian_to_memory(
    vault_path: Path,
    project_filter: str | None,
    imported_files: list[str],
) -> tuple[int, list[str]]:
    """Import Obsidian notes with `memory: true` frontmatter into memory.db.

    Scans all .md files in the vault (except Memory/ export dir).
    Returns (count_imported, updated_imported_files_list).
    """
    if not os.path.exists(DB_PATH):
        print(f"Error: memory.db not found at {DB_PATH}", file=sys.stderr)
        return 0, imported_files

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    imported = 0
    new_imported: list[str] = list(imported_files)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    session_id = f"obsidian_import_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    for md_file in vault_path.rglob("*.md"):
        # Skip our own export directory
        try:
            md_file.relative_to(vault_path / "Memory")
            continue
        except ValueError:
            pass

        rel_path = str(md_file.relative_to(vault_path))

        # Read and check for memory frontmatter
        try:
            text = md_file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        meta, body = parse_frontmatter(text)
        if not meta.get("memory"):
            continue

        # Extract metadata
        record_type = meta.get("type", DEFAULT_IMPORT_TYPE)
        if not isinstance(record_type, str) or record_type not in IMPORTABLE_TYPES:
            print(f"  Unknown type {record_type!r} in {rel_path}; storing as {DEFAULT_IMPORT_TYPE}", file=sys.stderr)
            record_type = DEFAULT_IMPORT_TYPE
        record_project = meta.get("project", "general")
        if not isinstance(record_project, str) or not record_project.strip():
            record_project = "general"
        raw_tags = meta.get("tags", [])
        if isinstance(raw_tags, str):
            raw_tags = [t.strip() for t in raw_tags.split(",")]
        tags = list(raw_tags) + ["obsidian-import"]

        # Apply project filter
        if project_filter and record_project != project_filter:
            continue

        content = body.strip()
        if not content:
            continue

        # Dedup: check if similar content already exists
        cursor.execute("""
            SELECT id, content FROM knowledge
            WHERE project = ? AND type = ? AND status = 'active'
            ORDER BY created_at DESC LIMIT 20
        """, (record_project, record_type))

        duplicate_found = False
        for row in cursor.fetchall():
            existing_content = row[1] or ""
            # Simple similarity: check if >85% of words match
            if _content_similar(content, existing_content, threshold=0.85):
                duplicate_found = True
                # Update last_confirmed for existing record
                cursor.execute(
                    "UPDATE knowledge SET last_confirmed = ? WHERE id = ?",
                    (timestamp, row[0]),
                )
                break

        if duplicate_found:
            continue

        # Insert new record (redaction, private-block stripping, embedding queue in the shared helper)
        saved = save_background_knowledge(
            conn, content=content, ktype=record_type, project=record_project,
            session_id=session_id, tags=tags, context=f"Imported from Obsidian: {rel_path}",
            source="obsidian", confidence=0.8, created_at=timestamp,
            extra={"last_confirmed": timestamp},
        )
        if saved is None:
            continue

        imported += 1
        new_imported.append(rel_path)
        print(f"  Imported: {rel_path} -> {record_type} ({record_project})")

    conn.commit()
    conn.close()
    return imported, new_imported


def _content_similar(a: str, b: str, threshold: float = 0.85) -> bool:
    """Check if two texts are similar by word overlap (Jaccard)."""
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return False
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union) >= threshold


def _parse_tags(tags_str: str) -> list[str]:
    """Parse tags from JSON string or return empty list."""
    try:
        tags = json.loads(tags_str)
        return tags if isinstance(tags, list) else []
    except (json.JSONDecodeError, TypeError):
        return []


def _safe_filename(name: str) -> str:
    """Convert a string to a safe directory/file name."""
    # Dots are kept inside a name but never at its edges: ".." must not become a path component.
    safe = re.sub(r"[^\w\-.]", "_", name).strip("_.")
    if safe:
        return safe
    # Empty or dot-only names get a stable distinct directory instead of colliding with "general".
    return f"project-{hashlib.sha1(name.encode('utf-8')).hexdigest()[:8]}" if name else "general"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bidirectional sync between Obsidian vault and memory.db.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python src/tools/obsidian_sync.py --vault ~/Obsidian/MyVault\n"
            "  python src/tools/obsidian_sync.py --vault ~/Obsidian/MyVault --project myapp\n"
            "  python src/tools/obsidian_sync.py --vault ~/Obsidian/MyVault --export-only\n"
            "  python src/tools/obsidian_sync.py --vault ~/Obsidian/MyVault --import-only"
        ),
    )
    parser.add_argument("--vault", "-v", required=True, help="Path to Obsidian vault")
    parser.add_argument("--project", "-p", default=None, help="Filter by project name")
    parser.add_argument("--export-only", action="store_true", help="Only export memory -> Obsidian")
    parser.add_argument("--import-only", action="store_true", help="Only import Obsidian -> memory")
    parser.add_argument("--full", action="store_true", help="Ignore last sync timestamp, export all")

    args = parser.parse_args()

    vault_path = Path(args.vault).resolve()
    if not vault_path.is_dir():
        print(f"Error: vault path {vault_path} is not a directory", file=sys.stderr)
        sys.exit(1)

    state = load_sync_state()
    last_sync = None if args.full else state.get("last_sync")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"Obsidian Sync — Vault: {vault_path}")
    if last_sync:
        print(f"Last sync: {last_sync}")
    else:
        print("Full sync (no previous state)")
    print("=" * 60)

    exported = 0
    imported = 0

    if not args.import_only:
        print("\n--- Export: Memory -> Obsidian ---")
        exported = export_memory_to_obsidian(vault_path, args.project, last_sync)
        print(f"Exported {exported} records.")

    if not args.export_only:
        print("\n--- Import: Obsidian -> Memory ---")
        imported, new_imported = import_obsidian_to_memory(
            vault_path, args.project, state.get("imported_files", []),
        )
        state["imported_files"] = new_imported
        print(f"Imported {imported} records.")

    # Update sync state
    state["last_sync"] = now
    save_sync_state(state)

    print(f"\n{'=' * 60}")
    print(f"Sync complete. Exported: {exported}, Imported: {imported}")
    print(f"State saved to {SYNC_STATE_PATH}")


if __name__ == "__main__":
    main()
