#!/usr/bin/env python3
"""
Code Analysis Script — scans a project directory and saves
architecture info (stack, file counts, structure) to memory.db.

Usage:
    python src/tools/analyze_project.py /path/to/project [--project-name NAME]
"""

import argparse
import json
import os
import sqlite3
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from memory_core.timestamps import utc_now

MEMORY_DIR = os.environ.get("CLAUDE_MEMORY_DIR", os.path.expanduser("~/.claude-memory"))
DB_PATH = os.path.join(MEMORY_DIR, "memory.db")

# Stack detection: filename -> (stack_name, ecosystem)
STACK_MARKERS: dict[str, tuple[str, str]] = {
    "go.mod": ("Go", "go"),
    "composer.json": ("PHP/Composer", "php"),
    "package.json": ("Node.js/NPM", "javascript"),
    "requirements.txt": ("Python/pip", "python"),
    "pyproject.toml": ("Python/Poetry", "python"),
    "Cargo.toml": ("Rust/Cargo", "rust"),
    "Gemfile": ("Ruby/Bundler", "ruby"),
    "build.gradle": ("Java/Gradle", "java"),
    "pom.xml": ("Java/Maven", "java"),
    "mix.exs": ("Elixir/Mix", "elixir"),
}

# Key infrastructure files
KEY_FILES: list[str] = [
    "Dockerfile",
    "docker-compose.yml",
    "docker-compose.yaml",
    "Makefile",
    "README.md",
    "README.rst",
    ".env.example",
    ".env.dist",
    ".github/workflows",
    ".gitlab-ci.yml",
    "Jenkinsfile",
    "Procfile",
    "Vagrantfile",
    "terraform.tf",
]

# Extensions to count
CODE_EXTENSIONS: set[str] = {
    ".go", ".php", ".ts", ".tsx", ".js", ".jsx", ".vue", ".py",
    ".rs", ".rb", ".java", ".kt", ".swift", ".c", ".cpp", ".h",
    ".css", ".scss", ".html", ".sql", ".proto", ".graphql",
    ".yaml", ".yml", ".json", ".toml", ".xml", ".sh", ".bash",
}

# Directories to skip when scanning
SKIP_DIRS: set[str] = {
    ".git", "node_modules", "vendor", ".venv", "venv", "__pycache__",
    ".idea", ".vscode", "dist", "build", "target", ".next", ".nuxt",
    "coverage", ".tox", ".mypy_cache", ".pytest_cache", "bin", "obj",
}


def detect_stack(project_root: Path) -> list[tuple[str, str]]:
    """Detect project stack by looking for marker files."""
    found: list[tuple[str, str]] = []
    for marker, info in STACK_MARKERS.items():
        if (project_root / marker).exists():
            found.append(info)
    return found


def count_files_by_extension(project_root: Path) -> Counter:
    """Count source files grouped by extension."""
    counts: Counter = Counter()
    for root, dirs, files in os.walk(project_root):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for f in files:
            ext = Path(f).suffix.lower()
            if ext in CODE_EXTENSIONS:
                counts[ext] += 1
    return counts


def find_key_files(project_root: Path) -> list[str]:
    """Find key infrastructure files that exist in the project."""
    found: list[str] = []
    for kf in KEY_FILES:
        target = project_root / kf
        if target.exists():
            found.append(kf)
    return found


def get_directory_structure(project_root: Path, max_depth: int = 2) -> list[str]:
    """Build directory tree up to max_depth levels."""
    lines: list[str] = []

    def _walk(path: Path, prefix: str, depth: int) -> None:
        if depth > max_depth:
            return
        try:
            entries = sorted(path.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower()))
        except PermissionError:
            return

        dirs_list = [e for e in entries if e.is_dir() and e.name not in SKIP_DIRS]
        files_list = [e for e in entries if e.is_file()]

        # Show files at this level (limit to 15 per dir to avoid noise)
        for f in files_list[:15]:
            lines.append(f"{prefix}{f.name}")
        if len(files_list) > 15:
            lines.append(f"{prefix}... and {len(files_list) - 15} more files")

        for d in dirs_list:
            lines.append(f"{prefix}{d.name}/")
            _walk(d, prefix + "  ", depth + 1)

    lines.append(f"{project_root.name}/")
    _walk(project_root, "  ", 1)
    return lines


def read_readme(project_root: Path, max_chars: int = 2000) -> str | None:
    """Read README.md content, truncated to max_chars."""
    for name in ("README.md", "README.rst", "README.txt", "README"):
        readme = project_root / name
        if readme.exists():
            try:
                text = readme.read_text(encoding="utf-8", errors="replace")
                return text[:max_chars]
            except OSError:
                continue
    return None


def parse_dependencies(project_root: Path, stacks: list[tuple[str, str]]) -> dict[str, list[str]]:
    """Extract top-level dependencies from package files."""
    deps: dict[str, list[str]] = {}

    for _, ecosystem in stacks:
        if ecosystem == "go":
            go_mod = project_root / "go.mod"
            if go_mod.exists():
                mod_deps: list[str] = []
                in_require = False
                for line in go_mod.read_text(errors="replace").splitlines():
                    stripped = line.strip()
                    if stripped.startswith("require ("):
                        in_require = True
                        continue
                    if in_require and stripped == ")":
                        in_require = False
                        continue
                    if in_require and stripped and not stripped.startswith("//"):
                        parts = stripped.split()
                        if parts:
                            mod_deps.append(parts[0])
                deps["go"] = mod_deps[:30]

        elif ecosystem == "php":
            cj = project_root / "composer.json"
            if cj.exists():
                try:
                    data = json.loads(cj.read_text(errors="replace"))
                    php_deps = list(data.get("require", {}).keys())[:30]
                    deps["php"] = php_deps
                except (json.JSONDecodeError, OSError):
                    pass

        elif ecosystem == "javascript":
            pj = project_root / "package.json"
            if pj.exists():
                try:
                    data = json.loads(pj.read_text(errors="replace"))
                    js_deps = list(data.get("dependencies", {}).keys())[:30]
                    deps["javascript"] = js_deps
                except (json.JSONDecodeError, OSError):
                    pass

        elif ecosystem == "python":
            req = project_root / "requirements.txt"
            if req.exists():
                py_deps: list[str] = []
                for line in req.read_text(errors="replace").splitlines():
                    stripped = line.strip()
                    if stripped and not stripped.startswith("#") and not stripped.startswith("-"):
                        pkg = stripped.split("==")[0].split(">=")[0].split("<=")[0].split("~=")[0].strip()
                        if pkg:
                            py_deps.append(pkg)
                deps["python"] = py_deps[:30]

        elif ecosystem == "rust":
            cargo = project_root / "Cargo.toml"
            if cargo.exists():
                rust_deps: list[str] = []
                in_deps = False
                for line in cargo.read_text(errors="replace").splitlines():
                    stripped = line.strip()
                    if stripped == "[dependencies]":
                        in_deps = True
                        continue
                    if in_deps and stripped.startswith("["):
                        in_deps = False
                        continue
                    if in_deps and "=" in stripped:
                        rust_deps.append(stripped.split("=")[0].strip())
                deps["rust"] = rust_deps[:30]

    return deps


def save_to_memory(records: list[dict]) -> int:
    """Save knowledge records to memory.db. Returns count of saved records."""
    if not os.path.exists(DB_PATH):
        print(f"Error: memory.db not found at {DB_PATH}", file=sys.stderr)
        return 0

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    saved = 0
    timestamp = utc_now()
    session_id = f"analyze_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    for rec in records:
        # Dedup: check for existing record with same project + similar tags
        tag_check = rec["tags"][0] if rec["tags"] else "code-analysis"
        cursor.execute("""
            SELECT id FROM knowledge
            WHERE project = ? AND status = 'active'
              AND tags LIKE ?
              AND tags LIKE ?
            ORDER BY created_at DESC LIMIT 1
        """, (rec["project"], f"%{tag_check}%", f"%{rec['tags'][1] if len(rec['tags']) > 1 else ''}%"))

        existing = cursor.fetchone()
        if existing:
            # Update existing record
            cursor.execute("""
                UPDATE knowledge
                SET content = ?, context = ?, tags = ?,
                    last_confirmed = ?, confidence = 0.9
                WHERE id = ?
            """, (
                rec["content"],
                rec.get("context", ""),
                json.dumps(rec["tags"]),
                timestamp,
                existing[0],
            ))
        else:
            # Insert new record
            cursor.execute("""
                INSERT INTO knowledge
                    (session_id, type, content, context, project, tags, status,
                     confidence, source, created_at, last_confirmed)
                VALUES (?, ?, ?, ?, ?, ?, 'active', 0.9, 'tool', ?, ?)
            """, (
                session_id,
                rec["type"],
                rec["content"],
                rec.get("context", ""),
                rec["project"],
                json.dumps(rec["tags"]),
                timestamp,
                timestamp,
            ))
        saved += 1

    conn.commit()
    conn.close()
    return saved


def analyze_project(project_path: str, project_name: str | None = None) -> None:
    """Main analysis function."""
    root = Path(project_path).resolve()
    if not root.is_dir():
        print(f"Error: {root} is not a directory", file=sys.stderr)
        sys.exit(1)

    name = project_name or root.name
    print(f"Analyzing project: {name} ({root})")
    print("=" * 60)

    # 1. Detect stack
    stacks = detect_stack(root)
    stack_names = [s[0] for s in stacks]
    print(f"\nStack: {', '.join(stack_names) if stack_names else 'Unknown'}")

    # 2. Count files
    file_counts = count_files_by_extension(root)
    total_files = sum(file_counts.values())
    print(f"\nFiles ({total_files} total):")
    for ext, count in file_counts.most_common(15):
        print(f"  {ext:10s} {count:>5d}")

    # 3. Key files
    key_files = find_key_files(root)
    print(f"\nKey files: {', '.join(key_files) if key_files else 'None found'}")

    # 4. Directory structure
    structure = get_directory_structure(root)
    print(f"\nDirectory structure:")
    for line in structure[:50]:
        print(f"  {line}")
    if len(structure) > 50:
        print(f"  ... ({len(structure) - 50} more entries)")

    # 5. Dependencies
    deps = parse_dependencies(root, stacks)

    # 6. README
    readme = read_readme(root)

    # Build records for memory
    records: list[dict] = []

    # Architecture summary
    arch_parts: list[str] = [
        f"Project: {name}",
        f"Path: {root}",
        f"Stack: {', '.join(stack_names) if stack_names else 'Unknown'}",
        f"Total source files: {total_files}",
        f"Key files: {', '.join(key_files)}",
    ]
    if readme:
        arch_parts.append(f"\nREADME (excerpt):\n{readme[:500]}")

    records.append({
        "type": "fact",
        "content": "\n".join(arch_parts),
        "context": f"Auto-analyzed on {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "project": name,
        "tags": ["code-analysis", "architecture"],
    })

    # Stack + dependencies
    dep_parts: list[str] = [f"Project: {name} — Stack & Dependencies"]
    for ecosystem, dep_list in deps.items():
        dep_parts.append(f"\n{ecosystem} dependencies ({len(dep_list)}):")
        for d in dep_list:
            dep_parts.append(f"  - {d}")
    top_extensions = ", ".join(f"{ext}({c})" for ext, c in file_counts.most_common(10))
    dep_parts.append(f"\nFile distribution: {top_extensions}")

    records.append({
        "type": "fact",
        "content": "\n".join(dep_parts),
        "context": f"Auto-analyzed on {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "project": name,
        "tags": ["code-analysis", "stack"],
    })

    # Directory structure
    structure_text = "\n".join(structure[:80])
    records.append({
        "type": "fact",
        "content": f"Project: {name} — Directory Structure\n\n{structure_text}",
        "context": f"Auto-analyzed on {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "project": name,
        "tags": ["code-analysis", "structure"],
    })

    # Save to memory
    saved = save_to_memory(records)
    print(f"\n{'=' * 60}")
    print(f"Saved {saved} knowledge records to memory.db for project '{name}'")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze a project directory and save architecture info to memory.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example:\n  python src/tools/analyze_project.py ~/projects/myapp --project-name myapp",
    )
    parser.add_argument("project_path", help="Path to the project directory")
    parser.add_argument(
        "--project-name", "-n",
        help="Project name (defaults to directory name)",
        default=None,
    )

    args = parser.parse_args()
    analyze_project(args.project_path, args.project_name)


if __name__ == "__main__":
    main()
