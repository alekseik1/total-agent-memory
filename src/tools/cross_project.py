#!/usr/bin/env python3
"""
Cross-Project Intelligence -- finds patterns, reusable code, and
architectural insights across ALL monitored projects.

Scheduled: weekly Sunday 2:00 AM via reflection/scheduler.py
Manual:    python src/tools/cross_project.py [--db PATH]

Features:
  1. Pattern Bridge     -- finds similar solutions across projects
  2. Reuse Detector     -- finds duplicate utility code
  3. Architecture Advisor -- structural analysis + anti-pattern detection
  4. Tech Debt Radar    -- TODO/FIXME/HACK scanning + dep freshness
"""

import json
import os
import re
import sqlite3
import sys
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from background_writes import save_background_knowledge  # noqa: E402
from tool_support import ollama_generate  # noqa: E402

from paths import memory_dir  # noqa: E402

MEMORY_DIR = memory_dir()
MONITORED_PROJECTS_PATH = MEMORY_DIR / "monitored_projects.json"
TELEGRAM_ENV_PATH = Path(__file__).parent.parent.parent / "telegram" / ".env"

LOG = lambda msg: sys.stderr.write(
    f"[cross-project] {datetime.now().strftime('%H:%M:%S')} {msg}\n"
)

# Directories to skip when scanning project trees
SKIP_DIRS: set[str] = {
    ".git", "node_modules", "vendor", ".venv", "venv", "__pycache__",
    ".idea", ".vscode", "dist", "build", "target", ".next", ".nuxt",
    "coverage", ".tox", ".mypy_cache", ".pytest_cache", "bin", "obj",
    ".cache", ".gradle", ".dart_tool",
}

# File extensions considered as source code
CODE_EXTENSIONS: set[str] = {
    ".go", ".php", ".ts", ".tsx", ".js", ".jsx", ".vue", ".py",
    ".rs", ".rb", ".java", ".kt", ".swift", ".c", ".cpp", ".h",
    ".css", ".scss", ".html", ".sql", ".proto", ".graphql",
    ".yaml", ".yml", ".json", ".toml", ".xml", ".sh", ".bash",
}

# Utility file name patterns
UTILITY_PATTERNS: list[str] = [
    r"util", r"helper", r"common", r"shared", r"lib",
]

# Architecture pattern markers
ARCH_MARKERS: dict[str, list[str]] = {
    "MVC": ["controllers", "models", "views"],
    "Clean Architecture": ["domain", "application", "infrastructure", "interfaces"],
    "DDD": ["domain", "aggregate", "repository", "value_object", "entity"],
    "Microservices": ["cmd", "internal", "api", "proto", "gateway"],
    "Hexagonal": ["ports", "adapters", "domain", "application"],
    "CQRS": ["commands", "queries", "handlers", "events"],
    "Laravel-style": ["app/http", "app/models", "database/migrations", "routes"],
    "Symfony-style": ["src/controller", "src/entity", "src/repository", "config"],
}


# ---------------------------------------------------------------------------
# Helpers (imported pattern from scheduler.py)
# ---------------------------------------------------------------------------


def _load_dotenv(path: Path) -> dict[str, str]:
    """Load .env file and return key-value pairs."""
    result: dict[str, str] = {}
    if not path.is_file():
        return result
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("'\"")
            if key:
                result[key] = value
    return result


def _telegram_send(token: str, chat_id: int, text: str, parse_mode: str = "HTML") -> bool:
    """Send a Telegram message via Bot API."""
    import urllib.request

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = json.dumps({
        "chat_id": chat_id,
        "text": text[:4096],
        "parse_mode": parse_mode,
    }).encode()
    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"},
    )
    try:
        from tls_context import verified_context
        ctx = verified_context()
        with urllib.request.urlopen(req, timeout=30, context=ctx) as resp:
            result = json.loads(resp.read())
            return result.get("ok", False)
    except Exception as e:
        LOG(f"Telegram send error: {e}")
        return False


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from text (terminal colors, cursor moves, etc.)."""
    return re.sub(r"\x1b\[[0-9;]*[A-Za-z]|\x1b\].*?\x07", "", text)


def _ollama_generate(prompt: str, model: str | None = None, timeout: int = 120) -> str:
    """Generate text with the Ollama CLI (prompt on stdin). Returns "" on failure."""
    return ollama_generate(prompt, model, timeout, log=LOG)


def _save_knowledge_record(
    db_path: str,
    content: str,
    record_type: str = "fact",
    tags: list[str] | None = None,
    source: str = "cross-project",
    confidence: float = 0.6,
    project: str = "general",
) -> None:
    """Save a knowledge record to the DB."""
    db = sqlite3.connect(db_path)
    try:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        save_background_knowledge(
            db, content=content, ktype=record_type, project=project,
            session_id=f"{source}_{now}_{uuid.uuid4().hex[:6]}",
            tags=tags or [], source=source, confidence=confidence, created_at=now,
        )
    except Exception as e:
        LOG(f"Error saving knowledge ({source}): {e}")
    finally:
        db.close()


def _send_digest(lines: list[str], title: str) -> None:
    """Send a Telegram digest message."""
    env = _load_dotenv(TELEGRAM_ENV_PATH)
    token = env.get("TELEGRAM_BOT_TOKEN", "")
    allowed = env.get("TELEGRAM_ALLOWED_USERS", "")
    if not token or not allowed:
        LOG("Telegram credentials not configured, skipping digest")
        return

    message = "\n".join(lines)
    for uid in allowed.split(","):
        uid = uid.strip()
        if uid:
            try:
                _telegram_send(token, int(uid), message)
            except ValueError:
                pass
    LOG(f"Telegram digest sent: {title}")


def _load_monitored_projects() -> list[dict[str, str]]:
    """Load monitored projects config.

    Expected format: [{"name": "project-name", "path": "/abs/path"}, ...]
    Falls back to scanning knowledge DB for known project paths.
    """
    if MONITORED_PROJECTS_PATH.is_file():
        try:
            with open(MONITORED_PROJECTS_PATH, encoding="utf-8") as f:
                projects = json.load(f)
            if isinstance(projects, list):
                valid = [
                    p for p in projects
                    if isinstance(p, dict) and p.get("path") and Path(p["path"]).is_dir()
                ]
                if valid:
                    return valid
        except (json.JSONDecodeError, IOError) as e:
            LOG(f"Error reading monitored_projects.json: {e}")

    LOG("No monitored_projects.json found or empty, trying knowledge DB fallback")
    return []


def _get_projects_from_db(db_path: str) -> list[dict[str, str]]:
    """Extract project names and paths from knowledge DB records."""
    projects: list[dict[str, str]] = []
    seen_names: set[str] = set()
    try:
        db = sqlite3.connect(db_path)
        db.row_factory = sqlite3.Row
        rows = db.execute(
            """SELECT DISTINCT project, content FROM knowledge
               WHERE status = 'active'
                 AND tags LIKE '%code-analysis%'
                 AND content LIKE '%Path:%'
               ORDER BY created_at DESC"""
        ).fetchall()
        db.close()

        for row in rows:
            name = row["project"]
            if not name or name in seen_names or name == "general":
                continue
            match = re.search(r"Path:\s*(.+)", row["content"])
            if match:
                path_str = match.group(1).strip()
                if Path(path_str).is_dir():
                    projects.append({"name": name, "path": path_str})
                    seen_names.add(name)
    except Exception as e:
        LOG(f"Error querying projects from DB: {e}")

    return projects


def _scan_file_lines(file_path: Path, max_lines: int = 200) -> list[str]:
    """Read first N lines of a file safely."""
    try:
        with open(file_path, encoding="utf-8", errors="replace") as f:
            lines: list[str] = []
            for i, line in enumerate(f):
                if i >= max_lines:
                    break
                lines.append(line.rstrip())
            return lines
    except (OSError, PermissionError):
        return []


# ---------------------------------------------------------------------------
# 1. Pattern Bridge
# ---------------------------------------------------------------------------


def run_pattern_bridge(db_path: str) -> int:
    """Find similar solutions across different projects.

    Uses FTS5 matching and tag overlap to identify cross-project patterns.
    Returns count of patterns found.
    """
    LOG("=== PATTERN BRIDGE STARTING ===")

    try:
        db = sqlite3.connect(db_path)
        db.row_factory = sqlite3.Row
    except Exception as e:
        LOG(f"DB connection error: {e}")
        return 0

    patterns_found = 0
    digest_lines: list[str] = ["<b>Cross-Project Pattern Bridge</b>", ""]

    try:
        rows = db.execute(
            """SELECT id, project, content, tags FROM knowledge
               WHERE status = 'active' AND type = 'solution'
                 AND project IS NOT NULL AND project != 'general'
               ORDER BY project, created_at DESC"""
        ).fetchall()

        if not rows:
            LOG("No solutions found in knowledge DB")
            db.close()
            return 0

        # Group by project
        by_project: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            tags_list: list[str] = []
            try:
                tags_list = json.loads(row["tags"]) if row["tags"] else []
            except (json.JSONDecodeError, TypeError):
                pass
            by_project[row["project"]].append({
                "id": row["id"],
                "content": row["content"],
                "tags": set(tags_list),
            })

        projects = list(by_project.keys())
        if len(projects) < 2:
            LOG(f"Only {len(projects)} project(s) with solutions, need 2+ for cross-project analysis")
            db.close()
            return 0

        LOG(f"Analyzing {len(projects)} projects: {', '.join(projects)}")

        # Compare solutions across project pairs
        cross_patterns: list[dict[str, Any]] = []
        skip_tags = {"reusable", "solution", "auto", "session-autosave", "context-recovery"}

        for i, proj_a in enumerate(projects):
            for proj_b in projects[i + 1:]:
                sols_a = by_project[proj_a]
                sols_b = by_project[proj_b]

                for sol_a in sols_a:
                    for sol_b in sols_b:
                        meaningful_a = sol_a["tags"] - skip_tags
                        meaningful_b = sol_b["tags"] - skip_tags
                        overlap = meaningful_a & meaningful_b

                        if len(overlap) >= 2:
                            cross_patterns.append({
                                "project_a": proj_a,
                                "project_b": proj_b,
                                "shared_tags": list(overlap),
                                "content_a": sol_a["content"][:300],
                                "content_b": sol_b["content"][:300],
                            })
                        elif _content_similarity(sol_a["content"], sol_b["content"]) > 0.3:
                            cross_patterns.append({
                                "project_a": proj_a,
                                "project_b": proj_b,
                                "shared_tags": list(overlap) if overlap else ["keyword-match"],
                                "content_a": sol_a["content"][:300],
                                "content_b": sol_b["content"][:300],
                            })

        # Deduplicate by project pair + similar tags
        seen_pairs: set[str] = set()
        unique_patterns: list[dict[str, Any]] = []
        for p in cross_patterns:
            pair_key = f"{p['project_a']}:{p['project_b']}:{','.join(sorted(p['shared_tags']))}"
            if pair_key not in seen_pairs:
                seen_pairs.add(pair_key)
                unique_patterns.append(p)

        LOG(f"Found {len(unique_patterns)} unique cross-project patterns")

        # Process top patterns with Ollama
        for pattern in unique_patterns[:5]:
            prompt = (
                f"Compare these two solutions from different projects.\n\n"
                f"Project '{pattern['project_a']}':\n{pattern['content_a']}\n\n"
                f"Project '{pattern['project_b']}':\n{pattern['content_b']}\n\n"
                f"Shared tags: {', '.join(pattern['shared_tags'])}\n\n"
                f"In 2-3 sentences: What's the common pattern? Should the approach be unified? "
                f"Which approach is better and why?"
            )
            recommendation = _ollama_generate(prompt, timeout=120)
            if not recommendation:
                recommendation = (
                    f"Projects {pattern['project_a']} and {pattern['project_b']} "
                    f"share pattern: {', '.join(pattern['shared_tags'])}. Review for unification."
                )

            record_content = (
                f"[Pattern Bridge] Cross-project pattern found\n"
                f"Projects: {pattern['project_a']}, {pattern['project_b']}\n"
                f"Shared: {', '.join(pattern['shared_tags'])}\n\n"
                f"Recommendation: {recommendation}"
            )
            _save_knowledge_record(
                db_path,
                content=record_content,
                record_type="lesson",
                tags=["cross-project", "pattern-bridge", "unification",
                      pattern["project_a"], pattern["project_b"]],
                source="pattern-bridge",
                confidence=0.65,
            )
            patterns_found += 1
            digest_lines.append(
                f"<b>{pattern['project_a']} + {pattern['project_b']}</b>: "
                f"{', '.join(pattern['shared_tags'][:3])}"
            )
            LOG(f"Saved pattern: {pattern['project_a']} <-> {pattern['project_b']}")

    except Exception as e:
        LOG(f"Pattern bridge error: {e}")
    finally:
        db.close()

    if patterns_found > 0:
        digest_lines.append(f"\nTotal: {patterns_found} cross-project patterns found")
        _send_digest(digest_lines, "Pattern Bridge")

    LOG(f"=== PATTERN BRIDGE COMPLETE === ({patterns_found} patterns)")
    return patterns_found


def _content_similarity(text_a: str, text_b: str) -> float:
    """Compute simple keyword-based similarity between two texts.

    Uses normalized word overlap (Jaccard on significant words).
    """
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "in", "on", "at",
        "to", "for", "of", "with", "and", "or", "not", "this", "that",
        "it", "be", "as", "by", "from", "but", "has", "had", "have",
        "i", "we", "you", "use", "using", "used", "can", "will",
    }

    def _extract_words(text: str) -> set[str]:
        words = re.findall(r"[a-zA-Z_]{3,}", text.lower())
        return {w for w in words if w not in stop_words and len(w) > 2}

    words_a = _extract_words(text_a)
    words_b = _extract_words(text_b)

    if not words_a or not words_b:
        return 0.0

    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union) if union else 0.0


# ---------------------------------------------------------------------------
# 2. Reuse Detector
# ---------------------------------------------------------------------------


def _detect_reusable_code(db_path: str) -> list[dict]:
    """Scan monitored projects for similar utility code.

    Looks for files matching utility patterns and compares across projects
    to find duplication opportunities.
    """
    LOG("--- Reuse Detector starting ---")

    projects = _load_monitored_projects()
    if not projects:
        projects = _get_projects_from_db(db_path)
    if len(projects) < 2:
        LOG(f"Need 2+ projects for reuse detection, found {len(projects)}")
        return []

    # Collect utility files per project
    project_utils: dict[str, list[dict[str, Any]]] = {}

    for proj in projects:
        proj_name = proj["name"]
        proj_path = Path(proj["path"])
        if not proj_path.is_dir():
            LOG(f"Project path not found: {proj_path}")
            continue

        utils: list[dict[str, Any]] = []
        for root, dirs, files in os.walk(proj_path):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            rel_root = Path(root).relative_to(proj_path)

            is_pkg_dir = any(
                part in str(rel_root).lower()
                for part in ("pkg/", "lib/", "shared/", "common/", "utils/")
            )

            for fname in files:
                fpath = Path(root) / fname
                ext = fpath.suffix.lower()
                if ext not in CODE_EXTENSIONS:
                    continue

                name_lower = fname.lower()
                is_util_file = any(
                    re.search(pat, name_lower) for pat in UTILITY_PATTERNS
                )

                if is_util_file or is_pkg_dir:
                    lines = _scan_file_lines(fpath, max_lines=200)
                    if lines:
                        utils.append({
                            "file": str(fpath.relative_to(proj_path)),
                            "name": fname,
                            "content_preview": "\n".join(lines[:100]),
                            "line_count": len(lines),
                        })

        if utils:
            project_utils[proj_name] = utils
            LOG(f"  {proj_name}: found {len(utils)} utility files")

    if len(project_utils) < 2:
        LOG("Not enough projects with utility files for comparison")
        return []

    # Compare utility files across projects
    recommendations: list[dict] = []
    project_names = list(project_utils.keys())

    for i, proj_a in enumerate(project_names):
        for proj_b in project_names[i + 1:]:
            for util_a in project_utils[proj_a]:
                for util_b in project_utils[proj_b]:
                    sim = _content_similarity(
                        util_a["content_preview"],
                        util_b["content_preview"],
                    )
                    if sim > 0.25:
                        recommendations.append({
                            "project_a": proj_a,
                            "file_a": util_a["file"],
                            "project_b": proj_b,
                            "file_b": util_b["file"],
                            "similarity": round(sim, 2),
                            "preview_a": util_a["content_preview"][:200],
                            "preview_b": util_b["content_preview"][:200],
                        })

    recommendations.sort(key=lambda r: r["similarity"], reverse=True)

    # Process top recommendations with Ollama
    for rec in recommendations[:3]:
        prompt = (
            f"Two utility files in different projects seem similar.\n\n"
            f"Project '{rec['project_a']}', file '{rec['file_a']}':\n"
            f"{rec['preview_a']}\n\n"
            f"Project '{rec['project_b']}', file '{rec['file_b']}':\n"
            f"{rec['preview_b']}\n\n"
            f"Similarity score: {rec['similarity']}\n\n"
            f"In 2-3 sentences: Do these do similar things? "
            f"Should they be extracted to a shared library?"
        )
        analysis = _ollama_generate(prompt, timeout=120)
        if analysis:
            rec["analysis"] = analysis

        record_content = (
            f"[Reuse Detector] Similar utility code found\n"
            f"{rec['project_a']}/{rec['file_a']} <-> {rec['project_b']}/{rec['file_b']}\n"
            f"Similarity: {rec['similarity']}\n\n"
            f"Analysis: {rec.get('analysis', 'Ollama unavailable')}"
        )
        _save_knowledge_record(
            db_path,
            content=record_content,
            record_type="decision",
            tags=["reuse-detector", "refactoring", rec["project_a"], rec["project_b"]],
            source="reuse-detector",
            confidence=0.55,
        )
        LOG(f"Saved reuse recommendation: {rec['file_a']} <-> {rec['file_b']}")

    LOG(f"--- Reuse Detector complete: {len(recommendations)} similarities found ---")
    return recommendations


# ---------------------------------------------------------------------------
# 3. Architecture Advisor
# ---------------------------------------------------------------------------


def run_architecture_advisor(db_path: str) -> int:
    """Analyze each monitored project's architecture.

    Checks structure, detects patterns, finds anti-patterns.
    Returns count of recommendations generated.
    """
    LOG("=== ARCHITECTURE ADVISOR STARTING ===")

    projects = _load_monitored_projects()
    if not projects:
        projects = _get_projects_from_db(db_path)
    if not projects:
        LOG("No projects found for architecture analysis")
        return 0

    recommendations_count = 0
    digest_lines: list[str] = ["<b>Architecture Advisor Report</b>", ""]

    for proj in projects:
        proj_name = proj["name"]
        proj_path = Path(proj["path"])
        if not proj_path.is_dir():
            LOG(f"Skipping {proj_name}: path not found ({proj_path})")
            continue

        LOG(f"Analyzing architecture: {proj_name}")

        try:
            analysis = _analyze_project_structure(proj_path, proj_name)
        except Exception as e:
            LOG(f"Error analyzing {proj_name}: {e}")
            continue

        anti_patterns = _detect_anti_patterns(proj_path)

        # Build prompt for Ollama
        prompt_parts: list[str] = [
            f"Analyze this project's architecture:\n",
            f"Project: {proj_name}",
            f"Files: {analysis['total_files']}, Dirs: {analysis['total_dirs']}",
            f"Max depth: {analysis['max_depth']}",
            f"Languages: {', '.join(f'{k}({v})' for k, v in analysis['languages'].most_common(5))}",
            f"Detected patterns: {', '.join(analysis['detected_patterns']) or 'none'}",
        ]

        if anti_patterns:
            prompt_parts.append(f"\nAnti-patterns found:")
            for ap in anti_patterns[:10]:
                prompt_parts.append(f"  - {ap}")

        best_practices = _load_best_practices(db_path, proj_name)
        if best_practices:
            prompt_parts.append(f"\nExisting conventions/rules from knowledge base:")
            for bp in best_practices[:5]:
                prompt_parts.append(f"  - {bp[:150]}")

        prompt_parts.append(
            "\nIn 3-5 bullet points: What's good? What needs improvement? "
            "Any architectural risks? Be specific and actionable."
        )

        recommendation = _ollama_generate("\n".join(prompt_parts), timeout=180)
        if not recommendation:
            recommendation = (
                f"Structure: {analysis['total_files']} files, "
                f"patterns: {', '.join(analysis['detected_patterns']) or 'none'}, "
                f"anti-patterns: {len(anti_patterns)}"
            )

        record_content = (
            f"[Architecture Advisor] {proj_name}\n"
            f"Files: {analysis['total_files']}, Dirs: {analysis['total_dirs']}, "
            f"Depth: {analysis['max_depth']}\n"
            f"Languages: {', '.join(f'{k}({v})' for k, v in analysis['languages'].most_common(5))}\n"
            f"Patterns: {', '.join(analysis['detected_patterns']) or 'none'}\n"
            f"Anti-patterns: {len(anti_patterns)}\n\n"
            f"Recommendation:\n{recommendation}"
        )
        _save_knowledge_record(
            db_path,
            content=record_content,
            record_type="decision",
            tags=["architecture-advisor", "improvement", proj_name],
            source="architecture-advisor",
            confidence=0.6,
            project=proj_name,
        )
        recommendations_count += 1

        digest_lines.append(
            f"<b>{proj_name}</b>: {analysis['total_files']} files, "
            f"{', '.join(analysis['detected_patterns'][:2]) or 'no pattern'}, "
            f"{len(anti_patterns)} anti-patterns"
        )

        # Run tech debt analysis for this project
        _analyze_tech_debt(db_path, str(proj_path), proj_name)

    # Also run reuse detection across all projects
    reuse_results = _detect_reusable_code(db_path)
    if reuse_results:
        digest_lines.append(f"\nReuse opportunities: {len(reuse_results)}")

    if recommendations_count > 0:
        digest_lines.append(f"\nTotal: {recommendations_count} projects analyzed")
        _send_digest(digest_lines, "Architecture Advisor")

    LOG(f"=== ARCHITECTURE ADVISOR COMPLETE === ({recommendations_count} projects)")
    return recommendations_count


def _analyze_project_structure(proj_path: Path, proj_name: str) -> dict[str, Any]:
    """Analyze a project's directory structure."""
    total_files = 0
    total_dirs = 0
    max_depth = 0
    languages: Counter = Counter()
    dir_names: set[str] = set()

    base_depth = len(proj_path.parts)

    for root, dirs, files in os.walk(proj_path):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        current_depth = len(Path(root).parts) - base_depth
        max_depth = max(max_depth, current_depth)
        total_dirs += len(dirs)

        for d in dirs:
            dir_names.add(d.lower())

        for f in files:
            ext = Path(f).suffix.lower()
            if ext in CODE_EXTENSIONS:
                total_files += 1
                languages[ext] += 1

    # Detect architecture patterns
    detected_patterns: list[str] = []
    dir_names_str = " ".join(dir_names)

    for pattern_name, markers in ARCH_MARKERS.items():
        matches = sum(1 for m in markers if m.lower() in dir_names_str)
        if matches >= len(markers) * 0.5:
            detected_patterns.append(pattern_name)

    return {
        "total_files": total_files,
        "total_dirs": total_dirs,
        "max_depth": max_depth,
        "languages": languages,
        "detected_patterns": detected_patterns,
        "dir_names": dir_names,
    }


def _detect_anti_patterns(proj_path: Path) -> list[str]:
    """Detect common anti-patterns in project structure."""
    anti_patterns: list[str] = []

    for root, dirs, files in os.walk(proj_path):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]

        for fname in files:
            fpath = Path(root) / fname
            ext = fpath.suffix.lower()
            if ext not in CODE_EXTENSIONS:
                continue

            try:
                line_count = sum(1 for _ in open(fpath, errors="replace", encoding="utf-8"))
                if line_count > 500:
                    rel = fpath.relative_to(proj_path)
                    anti_patterns.append(f"God file: {rel} ({line_count} lines)")
            except (OSError, PermissionError):
                pass

            if len(anti_patterns) >= 20:
                break
        if len(anti_patterns) >= 20:
            break

    # Check for too many top-level code files
    try:
        root_files = [
            f for f in proj_path.iterdir()
            if f.is_file() and f.suffix.lower() in CODE_EXTENSIONS
        ]
        if len(root_files) > 15:
            anti_patterns.append(f"Messy root: {len(root_files)} code files at top level")
    except PermissionError:
        pass

    # Check for deeply nested directories (>8 levels)
    for root, dirs, _ in os.walk(proj_path):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        depth = len(Path(root).parts) - len(proj_path.parts)
        if depth > 8:
            rel = Path(root).relative_to(proj_path)
            anti_patterns.append(f"Deep nesting: {rel} (depth {depth})")
            break

    return anti_patterns


def _load_best_practices(db_path: str, project_name: str) -> list[str]:
    """Load conventions and rules from knowledge DB for a project."""
    practices: list[str] = []
    try:
        db = sqlite3.connect(db_path)
        db.row_factory = sqlite3.Row
        rows = db.execute(
            """SELECT content FROM knowledge
               WHERE status = 'active'
                 AND (type = 'convention' OR type = 'rule' OR tags LIKE '%convention%')
                 AND (project = ? OR project = 'general')
               ORDER BY confidence DESC
               LIMIT 10""",
            (project_name,),
        ).fetchall()
        for row in rows:
            practices.append(row["content"][:200])
        db.close()
    except Exception as e:
        LOG(f"Error loading best practices: {e}")
    return practices


# ---------------------------------------------------------------------------
# 4. Tech Debt Radar
# ---------------------------------------------------------------------------


def _analyze_tech_debt(db_path: str, project_path: str, project_name: str) -> dict:
    """Scan project for tech debt indicators.

    Checks: TODO/FIXME/HACK comments, deprecated patterns, file sizes.
    Returns summary dict with debt level assessment.
    """
    LOG(f"--- Tech Debt Radar: {project_name} ---")

    proj_path = Path(project_path)
    if not proj_path.is_dir():
        LOG(f"Project path not found: {project_path}")
        return {}

    todo_count = 0
    fixme_count = 0
    hack_count = 0
    deprecated_count = 0
    large_files = 0
    todo_samples: list[str] = []

    todo_pattern = re.compile(r"\b(TODO|FIXME|HACK|XXX|WORKAROUND)\b", re.IGNORECASE)
    deprecated_pattern = re.compile(
        r"@deprecated|#\s*deprecated|//\s*deprecated|\bDeprecated\b", re.IGNORECASE
    )

    files_scanned = 0
    for root, dirs, files in os.walk(proj_path):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]

        for fname in files:
            fpath = Path(root) / fname
            ext = fpath.suffix.lower()
            if ext not in CODE_EXTENSIONS:
                continue

            files_scanned += 1
            try:
                with open(fpath, encoding="utf-8", errors="replace") as f:
                    line_count = 0
                    for line in f:
                        line_count += 1
                        match = todo_pattern.search(line)
                        if match:
                            marker = match.group(1).upper()
                            if marker == "TODO":
                                todo_count += 1
                            elif marker == "FIXME":
                                fixme_count += 1
                            elif marker in ("HACK", "XXX", "WORKAROUND"):
                                hack_count += 1

                            if len(todo_samples) < 10:
                                rel = fpath.relative_to(proj_path)
                                snippet = line.strip()[:100]
                                todo_samples.append(f"{rel}:{line_count}: {snippet}")

                        if deprecated_pattern.search(line):
                            deprecated_count += 1

                    if line_count > 500:
                        large_files += 1

            except (OSError, PermissionError):
                continue

    # Score tech debt
    total_markers = todo_count + fixme_count + hack_count
    markers_per_file = total_markers / max(files_scanned, 1)

    if markers_per_file > 0.5 or hack_count > 10 or large_files > 10:
        debt_level = "high"
    elif markers_per_file > 0.2 or hack_count > 3 or large_files > 5:
        debt_level = "medium"
    else:
        debt_level = "low"

    summary = {
        "project": project_name,
        "files_scanned": files_scanned,
        "todo_count": todo_count,
        "fixme_count": fixme_count,
        "hack_count": hack_count,
        "deprecated_count": deprecated_count,
        "large_files": large_files,
        "debt_level": debt_level,
        "markers_per_file": round(markers_per_file, 3),
        "samples": todo_samples,
    }

    # Save to knowledge DB
    record_parts: list[str] = [
        f"[Tech Debt Radar] {project_name}",
        f"Level: {debt_level.upper()}",
        f"Files scanned: {files_scanned}",
        f"TODO: {todo_count}, FIXME: {fixme_count}, HACK: {hack_count}",
        f"Deprecated: {deprecated_count}, Large files (>500 lines): {large_files}",
        f"Markers/file: {markers_per_file:.3f}",
    ]
    if todo_samples:
        record_parts.append("\nSamples:")
        for s in todo_samples[:5]:
            record_parts.append(f"  {s}")

    _save_knowledge_record(
        db_path,
        content="\n".join(record_parts),
        record_type="fact",
        tags=["tech-debt", project_name, debt_level],
        source="tech-debt-radar",
        confidence=0.7,
        project=project_name,
    )

    LOG(
        f"  {project_name}: {debt_level} debt "
        f"(TODO:{todo_count} FIXME:{fixme_count} HACK:{hack_count} "
        f"large:{large_files} files:{files_scanned})"
    )

    return summary


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------


def run_cross_project_intelligence(db_path: str) -> None:
    """Full cross-project analysis. Runs weekly on Sunday 2:00 AM."""
    LOG("====== CROSS-PROJECT INTELLIGENCE STARTING ======")
    start_time = datetime.now()

    results: dict[str, Any] = {}

    # Phase 1: Pattern Bridge
    try:
        patterns = run_pattern_bridge(db_path)
        results["patterns_found"] = patterns
    except Exception as e:
        LOG(f"Pattern Bridge failed: {e}")
        results["patterns_found"] = 0

    # Phase 2: Architecture Advisor (includes Reuse Detector + Tech Debt)
    try:
        recommendations = run_architecture_advisor(db_path)
        results["recommendations"] = recommendations
    except Exception as e:
        LOG(f"Architecture Advisor failed: {e}")
        results["recommendations"] = 0

    elapsed = (datetime.now() - start_time).total_seconds()
    LOG(
        f"====== CROSS-PROJECT INTELLIGENCE COMPLETE ====== "
        f"({elapsed:.1f}s, patterns:{results.get('patterns_found', 0)}, "
        f"recommendations:{results.get('recommendations', 0)})"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Cross-Project Intelligence")
    parser.add_argument("--db", default=str(MEMORY_DIR / "memory.db"))
    parser.add_argument(
        "--component",
        choices=["all", "patterns", "architecture", "reuse", "tech-debt"],
        default="all",
        help="Run specific component",
    )
    parser.add_argument(
        "--project-path",
        help="Path for single-project tech-debt scan",
    )
    parser.add_argument(
        "--project-name",
        help="Name for single-project tech-debt scan",
    )

    args = parser.parse_args()

    if args.component == "all":
        run_cross_project_intelligence(args.db)
    elif args.component == "patterns":
        run_pattern_bridge(args.db)
    elif args.component == "architecture":
        run_architecture_advisor(args.db)
    elif args.component == "reuse":
        _detect_reusable_code(args.db)
    elif args.component == "tech-debt":
        if args.project_path:
            name = args.project_name or Path(args.project_path).name
            result = _analyze_tech_debt(args.db, args.project_path, name)
            print(json.dumps(result, indent=2))
        else:
            print("Error: --project-path required for tech-debt component", file=sys.stderr)
            sys.exit(1)
