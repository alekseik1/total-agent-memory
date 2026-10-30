#!/usr/bin/env python3
"""
Knowledge Export — generates readable documentation from memory.db.

Queries knowledge records and formats them as clean Markdown or JSON.

Usage:
    python src/tools/export_knowledge.py --project strata --format markdown
    python src/tools/export_knowledge.py --project strata --format json --output export.json
    python src/tools/export_knowledge.py --all --output-dir ./exports
"""

import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paths import memory_dir  # noqa: E402

MEMORY_DIR = str(memory_dir())
DB_PATH = os.path.join(MEMORY_DIR, "memory.db")

# Display order: decisions first, then solutions, conventions, facts, lessons
TYPE_ORDER: list[str] = ["decision", "solution", "convention", "fact", "lesson", "reflection"]
TYPE_LABELS: dict[str, str] = {
    "decision": "Decisions",
    "solution": "Solutions",
    "convention": "Conventions",
    "fact": "Facts",
    "lesson": "Lessons Learned",
    "reflection": "Reflections",
}


def fetch_knowledge(project: str | None = None) -> list[dict]:
    """Fetch all active knowledge records, optionally filtered by project."""
    if not os.path.exists(DB_PATH):
        print(f"Error: memory.db not found at {DB_PATH}", file=sys.stderr)
        return []

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    if project:
        cursor.execute(
            "SELECT * FROM knowledge WHERE status = 'active' AND project = ? ORDER BY type, created_at DESC",
            (project,),
        )
    else:
        cursor.execute(
            "SELECT * FROM knowledge WHERE status = 'active' ORDER BY project, type, created_at DESC",
        )

    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def list_projects() -> list[str]:
    """List all projects that have active knowledge records."""
    if not os.path.exists(DB_PATH):
        return []

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT DISTINCT project FROM knowledge WHERE status = 'active' ORDER BY project",
    )
    projects = [row[0] for row in cursor.fetchall() if row[0]]
    conn.close()
    return projects


def _parse_tags(tags_str: str) -> list[str]:
    """Parse tags from JSON string."""
    try:
        tags = json.loads(tags_str)
        return tags if isinstance(tags, list) else []
    except (json.JSONDecodeError, TypeError):
        return []


def group_by_type(records: list[dict]) -> dict[str, list[dict]]:
    """Group records by type in display order."""
    grouped: dict[str, list[dict]] = {}
    for rec in records:
        rtype = rec.get("type", "fact")
        grouped.setdefault(rtype, []).append(rec)
    return grouped


def format_markdown(project: str, records: list[dict]) -> str:
    """Format knowledge records as a clean Markdown document with TOC."""
    grouped = group_by_type(records)
    lines: list[str] = []

    # Header
    lines.append(f"# Knowledge Base: {project}")
    lines.append(f"")
    lines.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*")
    lines.append(f"*Records: {len(records)}*")
    lines.append(f"")

    # Table of Contents
    lines.append("## Table of Contents")
    lines.append("")
    for rtype in TYPE_ORDER:
        if rtype in grouped:
            label = TYPE_LABELS.get(rtype, rtype.capitalize())
            count = len(grouped[rtype])
            anchor = label.lower().replace(" ", "-")
            lines.append(f"- [{label}](#{anchor}) ({count})")
    lines.append("")

    # Sections
    for rtype in TYPE_ORDER:
        if rtype not in grouped:
            continue

        label = TYPE_LABELS.get(rtype, rtype.capitalize())
        items = grouped[rtype]

        lines.append(f"---")
        lines.append(f"")
        lines.append(f"## {label}")
        lines.append(f"")

        for i, rec in enumerate(items, 1):
            tags = _parse_tags(rec.get("tags", "[]"))
            confidence = rec.get("confidence", 1.0)
            created = rec.get("created_at", "")
            recall_count = rec.get("recall_count", 0)
            content = (rec.get("content") or "").strip()

            # Title: first line of content or truncated content
            first_line = content.split("\n")[0][:80] if content else "(empty)"

            lines.append(f"### {i}. {first_line}")
            lines.append(f"")

            # Metadata line
            meta_parts: list[str] = []
            if created:
                meta_parts.append(f"Date: {created[:10]}")
            meta_parts.append(f"Confidence: {confidence:.1f}")
            if recall_count:
                meta_parts.append(f"Recalls: {recall_count}")
            if tags:
                meta_parts.append(f"Tags: `{'`, `'.join(tags)}`")
            lines.append(f"*{' | '.join(meta_parts)}*")
            lines.append(f"")

            # Content
            lines.append(content)

            # Context
            if rec.get("context"):
                lines.append(f"")
                lines.append(f"> {rec['context']}")

            lines.append(f"")

    # Handle types not in TYPE_ORDER
    extra_types = set(grouped.keys()) - set(TYPE_ORDER)
    for rtype in sorted(extra_types):
        items = grouped[rtype]
        lines.append(f"---")
        lines.append(f"")
        lines.append(f"## {rtype.capitalize()}")
        lines.append(f"")
        for i, rec in enumerate(items, 1):
            content = (rec.get("content") or "").strip()
            first_line = content.split("\n")[0][:80] if content else "(empty)"
            lines.append(f"### {i}. {first_line}")
            lines.append(f"")
            lines.append(content)
            lines.append(f"")

    return "\n".join(lines)


def format_json(project: str, records: list[dict]) -> str:
    """Format knowledge records as JSON."""
    output = {
        "project": project,
        "generated": datetime.now().isoformat(),
        "total_records": len(records),
        "records": [],
    }

    for rec in records:
        output["records"].append({
            "id": rec.get("id"),
            "type": rec.get("type"),
            "content": rec.get("content", ""),
            "context": rec.get("context", ""),
            "tags": _parse_tags(rec.get("tags", "[]")),
            "confidence": rec.get("confidence", 1.0),
            "recall_count": rec.get("recall_count", 0),
            "created_at": rec.get("created_at", ""),
            "last_confirmed": rec.get("last_confirmed"),
            "source": rec.get("source", ""),
        })

    return json.dumps(output, indent=2, ensure_ascii=False)


def export_single_project(
    project: str,
    fmt: str,
    output_path: str | None = None,
) -> None:
    """Export knowledge for a single project."""
    records = fetch_knowledge(project)
    if not records:
        print(f"No active knowledge records found for project '{project}'.", file=sys.stderr)
        return

    if fmt == "json":
        result = format_json(project, records)
    else:
        result = format_markdown(project, records)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(result, encoding="utf-8")
        print(f"Exported {len(records)} records for '{project}' -> {output_path}")
    else:
        print(result)


def export_all_projects(fmt: str, output_dir: str) -> None:
    """Export all projects, one file each, to output_dir."""
    projects = list_projects()
    if not projects:
        print("No projects with active knowledge found.", file=sys.stderr)
        return

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ext = ".json" if fmt == "json" else ".md"
    total = 0

    for project in projects:
        records = fetch_knowledge(project)
        if not records:
            continue

        safe_name = "".join(c if c.isalnum() or c in "-_." else "_" for c in project)
        filepath = out / f"{safe_name}{ext}"

        if fmt == "json":
            result = format_json(project, records)
        else:
            result = format_markdown(project, records)

        filepath.write_text(result, encoding="utf-8")
        total += len(records)
        print(f"  {project}: {len(records)} records -> {filepath.name}")

    print(f"\nExported {total} total records across {len(projects)} projects to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export knowledge from memory.db as Markdown or JSON.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python src/tools/export_knowledge.py --project myapp --format markdown\n"
            "  python src/tools/export_knowledge.py --project myapp --format json -o export.json\n"
            "  python src/tools/export_knowledge.py --all --output-dir ./exports\n"
            "  python src/tools/export_knowledge.py --list"
        ),
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--project", "-p", help="Export knowledge for a specific project")
    group.add_argument("--all", action="store_true", help="Export all projects")
    group.add_argument("--list", action="store_true", help="List all projects with knowledge")

    parser.add_argument(
        "--format", "-f",
        choices=["markdown", "json"],
        default="markdown",
        help="Output format (default: markdown)",
    )
    parser.add_argument("--output", "-o", default=None, help="Output file path (single project)")
    parser.add_argument("--output-dir", "-d", default="./exports", help="Output directory (--all mode)")

    args = parser.parse_args()

    if args.list:
        projects = list_projects()
        if not projects:
            print("No projects found.")
            return
        print("Projects with active knowledge:")
        for p in projects:
            records = fetch_knowledge(p)
            print(f"  {p}: {len(records)} records")
        return

    if args.all:
        export_all_projects(args.format, args.output_dir)
    else:
        export_single_project(args.project, args.format, args.output)


if __name__ == "__main__":
    main()
