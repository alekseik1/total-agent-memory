#!/usr/bin/env python3
"""
Git Observer & Error Pattern Miner -- autonomous coding pattern analysis.

Features:
1. Git Observer (daily at 11:00 PM):
   - Analyzes git history across all monitored projects
   - Tracks work patterns: commits/day, active dirs, languages, peak hours
   - Generates weekly coding profile with Ollama

2. Code Style Evolution Tracker (part of Git Observer):
   - Detects new patterns in commits (new libs, new coding patterns)
   - Compares current week vs previous weeks from memory
   - Saves evolution insights

3. Error Pattern Miner (daily at 11:30 PM):
   - Analyzes errors table in memory.db for recurring patterns
   - Searches web for solutions to common errors
   - Saves found solutions automatically

Usage:
    python src/tools/git_observer.py --db ~/.claude-memory/memory.db [--run git|errors|all]

Scheduler integration (in scheduler.py):
    from tools.git_observer import run_git_observer, run_error_pattern_miner

    # APScheduler:
    scheduler.add_job(run_git_observer, CronTrigger(hour=23, minute=0),
                      args=[self.db_path], id="git_observer", ...)
    scheduler.add_job(run_error_pattern_miner, CronTrigger(hour=23, minute=30),
                      args=[self.db_path], id="error_miner", ...)

    # Simple loop:
    last_git_observer_day = -1
    last_error_miner_day = -1
    if current_hour == 23 and datetime.now().minute < 30 and last_git_observer_day != today_ordinal:
        run_git_observer(self.db_path)
        last_git_observer_day = today_ordinal
    if current_hour == 23 and datetime.now().minute >= 30 and last_error_miner_day != today_ordinal:
        run_error_pattern_miner(self.db_path)
        last_error_miner_day = today_ordinal

    # --run-now choices: add "git-observer", "error-miner"
"""

import json
import re
import sqlite3
import subprocess
import sys
import time
import uuid
from collections import Counter
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from background_writes import save_background_knowledge  # noqa: E402
from tool_support import ollama_generate  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent))

from paths import memory_dir  # noqa: E402

MEMORY_DIR = memory_dir()
TELEGRAM_ENV_PATH = Path(__file__).parent.parent.parent / "telegram" / ".env"
MONITORED_PROJECTS_PATH = MEMORY_DIR / "monitored_projects.json"

LOG = lambda msg: sys.stderr.write(
    f"[git-observer] {datetime.now().strftime('%H:%M:%S')} {msg}\n"
)


# ---------------------------------------------------------------------------
# Shared helpers (same signatures as scheduler.py)
# ---------------------------------------------------------------------------


def _load_dotenv(path: Path) -> dict[str, str]:
    """Load .env file and return key-value pairs (no side effects)."""
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


def _make_ssl_context():
    """Verified TLS context: certifi, else the platform store, never unverified."""
    from tls_context import verified_context

    return verified_context()

def _telegram_send(token: str, chat_id: int, text: str, parse_mode: str = "HTML") -> bool:
    """Send a Telegram message via Bot API. Returns True on success."""
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
        ctx = _make_ssl_context()
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
    source: str = "auto",
    confidence: float = 0.5,
    project: str = "general",
) -> None:
    """Save a knowledge record to DB. Reusable helper."""
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


def _notify_telegram(message: str) -> None:
    """Send notification to all configured Telegram users."""
    env = _load_dotenv(TELEGRAM_ENV_PATH)
    token = env.get("TELEGRAM_BOT_TOKEN", "")
    allowed = env.get("TELEGRAM_ALLOWED_USERS", "")
    if not token or not allowed:
        return
    for uid in allowed.split(","):
        uid = uid.strip()
        if uid:
            try:
                _telegram_send(token, int(uid), message)
            except ValueError:
                pass


# ---------------------------------------------------------------------------
# Git Analysis -- constants
# ---------------------------------------------------------------------------

# File extension to language mapping
_EXT_LANG: dict[str, str] = {
    ".py": "Python", ".go": "Go", ".php": "PHP",
    ".js": "JavaScript", ".ts": "TypeScript", ".tsx": "TypeScript",
    ".jsx": "JavaScript", ".vue": "Vue", ".rs": "Rust",
    ".java": "Java", ".kt": "Kotlin", ".swift": "Swift",
    ".rb": "Ruby", ".cs": "C#", ".cpp": "C++", ".c": "C",
    ".sh": "Shell", ".bash": "Shell", ".zsh": "Shell",
    ".sql": "SQL", ".html": "HTML", ".css": "CSS",
    ".scss": "SCSS", ".yaml": "YAML", ".yml": "YAML",
    ".json": "JSON", ".md": "Markdown", ".proto": "Protobuf",
    ".dockerfile": "Docker", ".tf": "Terraform",
}

# Lock files that indicate dependency changes
_LOCK_FILES: set[str] = {
    "go.sum", "composer.lock", "package-lock.json", "yarn.lock",
    "pnpm-lock.yaml", "Pipfile.lock", "poetry.lock", "Cargo.lock",
    "Gemfile.lock", "pubspec.lock",
}


# ---------------------------------------------------------------------------
# Git Analysis -- core functions
# ---------------------------------------------------------------------------


def _run_git_cmd(project_path: str, args: list[str], timeout: int = 30) -> str:
    """Run a git command in the given project directory. Returns stdout."""
    try:
        result = subprocess.run(
            ["git", "-C", project_path] + args,
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout,
        )
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        LOG(f"Git command timeout in {project_path}: {args}")
        return ""
    except Exception as e:
        LOG(f"Git command error in {project_path}: {e}")
        return ""


def _analyze_project_git(project_path: str, project_name: str) -> dict[str, Any]:
    """Analyze git history for a single project over the last 7 days.

    Args:
        project_path: Absolute path to the project root.
        project_name: Human-readable project name.

    Returns:
        Dict with keys: name, path, total_commits, commits_by_day,
        files_changed, lines_added, lines_removed, active_dirs,
        languages, commit_hours, lock_file_changes, recent_commit_messages.
    """
    analysis: dict[str, Any] = {
        "name": project_name,
        "path": project_path,
        "total_commits": 0,
        "commits_by_day": {},
        "files_changed": 0,
        "lines_added": 0,
        "lines_removed": 0,
        "active_dirs": [],
        "languages": {},
        "commit_hours": [],
        "lock_file_changes": [],
        "recent_commit_messages": [],
    }

    # Verify .git directory exists
    if not Path(project_path, ".git").is_dir():
        LOG(f"  Skipping {project_name}: no .git directory")
        return analysis

    # ---- Commit log (last 7 days) ----
    log_output = _run_git_cmd(project_path, [
        "log", "--since=7 days ago",
        "--format=%H|%an|%ad|%s", "--date=short",
    ])

    if not log_output:
        LOG(f"  {project_name}: no commits in last 7 days")
        return analysis

    commits_by_day: Counter = Counter()
    commit_messages: list[str] = []

    for line in log_output.splitlines():
        parts = line.split("|", 3)
        if len(parts) < 4:
            continue
        _hash, _author, date_str, subject = parts
        commits_by_day[date_str] += 1
        commit_messages.append(subject)
        analysis["total_commits"] += 1

    analysis["commits_by_day"] = dict(commits_by_day)
    analysis["recent_commit_messages"] = commit_messages[:20]

    # ---- Commit hour distribution ----
    time_output = _run_git_cmd(project_path, [
        "log", "--since=7 days ago", "--format=%H|%aI",
    ])
    commit_hours: list[int] = []
    if time_output:
        for line in time_output.splitlines():
            parts = line.split("|", 1)
            if len(parts) == 2:
                hour_match = re.search(r"T(\d{2}):", parts[1])
                if hour_match:
                    commit_hours.append(int(hour_match.group(1)))
    analysis["commit_hours"] = commit_hours

    # ---- Diff stats ----
    commit_count = min(analysis["total_commits"], 20)
    if commit_count > 0:
        diff_output = _run_git_cmd(project_path, [
            "diff", "--stat", f"HEAD~{commit_count}..HEAD",
        ])

        if diff_output:
            dir_counter: Counter = Counter()
            lang_counter: Counter = Counter()
            lock_changes: list[str] = []

            for line in diff_output.splitlines():
                # File stat line: " src/foo.py | 10 ++-"
                match = re.match(r"\s*(.+?)\s*\|\s*(\d+)", line)
                if match:
                    filepath = match.group(1).strip()
                    changes = int(match.group(2))

                    # Track top-level directories
                    path_parts = filepath.split("/")
                    if len(path_parts) >= 2:
                        dir_counter["/".join(path_parts[:2])] += changes
                    else:
                        dir_counter["."] += changes

                    # Track languages by extension
                    ext = Path(filepath).suffix.lower()
                    if ext in _EXT_LANG:
                        lang_counter[_EXT_LANG[ext]] += changes

                    # Track lock file changes
                    basename = Path(filepath).name
                    if basename in _LOCK_FILES:
                        lock_changes.append(basename)

                # Summary line: " 15 files changed, 200 insertions(+), 50 deletions(-)"
                summary = re.match(
                    r"\s*(\d+) files? changed"
                    r"(?:,\s*(\d+) insertions?\(\+\))?"
                    r"(?:,\s*(\d+) deletions?\(-\))?",
                    line,
                )
                if summary:
                    analysis["files_changed"] = int(summary.group(1))
                    analysis["lines_added"] = int(summary.group(2) or 0)
                    analysis["lines_removed"] = int(summary.group(3) or 0)

            analysis["active_dirs"] = [d for d, _ in dir_counter.most_common(5)]
            analysis["languages"] = dict(lang_counter.most_common(10))
            analysis["lock_file_changes"] = lock_changes

    LOG(
        f"  {project_name}: {analysis['total_commits']} commits, "
        f"+{analysis['lines_added']}/-{analysis['lines_removed']}"
    )
    return analysis


def _generate_coding_profile(analyses: list[dict[str, Any]]) -> str:
    """Generate a weekly coding profile summary using Ollama.

    Aggregates data from all project analyses and creates a natural language
    summary of the developer's work patterns.

    Args:
        analyses: List of per-project analysis dicts (from _analyze_project_git).

    Returns:
        A human-readable profile string.
    """
    total_commits = sum(a["total_commits"] for a in analyses)
    total_added = sum(a["lines_added"] for a in analyses)
    total_removed = sum(a["lines_removed"] for a in analyses)
    total_files = sum(a["files_changed"] for a in analyses)

    if total_commits == 0:
        return "No coding activity detected this week."

    active_projects = [a["name"] for a in analyses if a["total_commits"] > 0]

    # Aggregate languages across projects
    all_langs: Counter = Counter()
    for a in analyses:
        all_langs.update(a["languages"])
    top_langs = [lang for lang, _ in all_langs.most_common(5)]

    # Aggregate commit hours
    all_hours: list[int] = []
    for a in analyses:
        all_hours.extend(a["commit_hours"])

    peak_hours = ""
    if all_hours:
        hour_counter = Counter(all_hours)
        top_hours = hour_counter.most_common(3)
        peak_hours = ", ".join(f"{h}:00" for h, _ in top_hours)

    # Aggregate active directories
    all_dirs: list[str] = []
    for a in analyses:
        all_dirs.extend(a["active_dirs"][:3])

    # Commits per day
    days_active: set[str] = set()
    for a in analyses:
        days_active.update(a["commits_by_day"].keys())
    commits_per_day = total_commits / max(len(days_active), 1)

    # Lock file changes
    all_locks: list[str] = []
    for a in analyses:
        all_locks.extend(a["lock_file_changes"])

    # Recent commit themes
    all_messages: list[str] = []
    for a in analyses:
        all_messages.extend(a["recent_commit_messages"][:5])

    # Build Ollama prompt
    context = (
        f"Weekly coding stats:\n"
        f"- Active projects: {', '.join(active_projects)}\n"
        f"- Total commits: {total_commits} ({commits_per_day:.1f}/day)\n"
        f"- Files changed: {total_files}\n"
        f"- Lines: +{total_added} / -{total_removed}\n"
        f"- Languages: {', '.join(top_langs) if top_langs else 'N/A'}\n"
        f"- Peak coding hours: {peak_hours or 'N/A'}\n"
        f"- Active directories: {', '.join(all_dirs[:8]) if all_dirs else 'N/A'}\n"
        f"- Dependency changes: {', '.join(set(all_locks)) if all_locks else 'none'}\n"
    )
    if all_messages:
        context += f"- Recent commit themes: {'; '.join(all_messages[:10])}\n"

    prompt = (
        f"You are analyzing a developer's weekly coding activity. "
        f"Write a concise 3-5 sentence summary that highlights:\n"
        f"1. Main focus areas and projects\n"
        f"2. Productivity patterns (velocity, peak hours)\n"
        f"3. Any notable trends (new languages, areas of growth)\n\n"
        f"{context}\n\n"
        f"Write the summary in a professional but friendly tone. "
        f"Start with 'This week' and be specific about what was accomplished."
    )

    profile = _ollama_generate(prompt, timeout=180)

    if not profile:
        # Fallback: plain-text profile without Ollama
        profile = (
            f"This week you focused on {', '.join(active_projects)}. "
            f"Main languages: {', '.join(top_langs[:3]) if top_langs else 'various'}. "
            f"Peak coding hours: {peak_hours or 'varied'}. "
            f"Commit velocity: {commits_per_day:.1f}/day "
            f"(+{total_added}/-{total_removed} lines across {total_files} files)."
        )

    return profile


# ---------------------------------------------------------------------------
# Code Style Evolution Tracker
# ---------------------------------------------------------------------------


def _detect_style_evolution(
    db_path: str, current_analyses: list[dict[str, Any]]
) -> list[str]:
    """Detect how coding patterns change over time.

    Compares current week's analysis with previous weeks' data stored in memory.

    Args:
        db_path: Path to memory.db.
        current_analyses: This week's per-project analysis dicts.

    Returns:
        List of human-readable evolution insight strings.
    """
    insights: list[str] = []

    # Load previous coding profiles from memory
    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row
    try:
        previous = db.execute(
            """SELECT content FROM knowledge
               WHERE tags LIKE '%git-observer%'
                 AND tags LIKE '%coding-pattern%'
                 AND status = 'active'
               ORDER BY created_at DESC LIMIT 4""",
        ).fetchall()
    except Exception:
        previous = []
    finally:
        db.close()

    prev_contents = [row["content"] for row in previous]

    # Current week aggregates
    current_langs: Counter = Counter()
    current_dirs: set[str] = set()
    current_locks: set[str] = set()
    current_messages: list[str] = []

    for a in current_analyses:
        current_langs.update(a["languages"])
        current_dirs.update(a["active_dirs"])
        current_locks.update(a["lock_file_changes"])
        current_messages.extend(a["recent_commit_messages"])

    # 1. New dependencies detected
    if current_locks:
        insights.append(
            f"New dependency changes detected: {', '.join(current_locks)}. "
            f"This may indicate new libraries or framework updates."
        )

    # 2. Coding pattern keywords in commit messages
    pattern_keywords: dict[str, str] = {
        "functional options": r"functional.?option",
        "table-driven tests": r"table.?driven|table.?test",
        "dependency injection": r"inject|DI|constructor.?inject",
        "middleware": r"middleware|interceptor",
        "generics": r"generic|type.?param",
        "async/await": r"async|await|coroutine",
        "error handling": r"error.?handl|wrap.?error|sentinel",
        "observability": r"metric|trace|observ|prometheus|grafana",
        "containerization": r"docker|container|k8s|kubernetes",
        "CI/CD": r"ci.?cd|pipeline|github.?action|workflow",
    }

    messages_text = " ".join(current_messages).lower()
    detected_patterns: list[str] = []
    for pattern_name, regex in pattern_keywords.items():
        if re.search(regex, messages_text, re.IGNORECASE):
            detected_patterns.append(pattern_name)

    if detected_patterns:
        insights.append(
            f"Active coding patterns this week: {', '.join(detected_patterns)}."
        )

    # 3. Compare with previous weeks
    if prev_contents:
        # Languages that are new compared to recent history
        prev_langs_mentioned: set[str] = set()
        for content in prev_contents:
            for lang in _EXT_LANG.values():
                if lang.lower() in content.lower():
                    prev_langs_mentioned.add(lang)

        new_langs = set(current_langs.keys()) - prev_langs_mentioned
        if new_langs:
            insights.append(
                f"New languages used this week (not seen in recent history): "
                f"{', '.join(new_langs)}."
            )

        # New focus directories
        current_top_dirs = [d for d in current_dirs if "/" in d][:5]
        if current_top_dirs:
            prev_dirs_text = " ".join(prev_contents)
            new_dirs = [d for d in current_top_dirs if d not in prev_dirs_text]
            if new_dirs:
                insights.append(
                    f"New focus areas: {', '.join(new_dirs)}. "
                    f"These directories weren't prominent in recent weeks."
                )

    return insights


# ---------------------------------------------------------------------------
# Monitored projects config
# ---------------------------------------------------------------------------


def _load_monitored_projects() -> list[dict[str, str]]:
    """Load the list of monitored projects from JSON config.

    File format: [{"path": "/path/to/project", "name": "project-name"}, ...]
    Creates the file with an empty list if it doesn't exist.
    """
    if not MONITORED_PROJECTS_PATH.is_file():
        LOG(f"Creating empty monitored projects file: {MONITORED_PROJECTS_PATH}")
        MONITORED_PROJECTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        MONITORED_PROJECTS_PATH.write_text("[]", encoding="utf-8")
        return []

    try:
        data = json.loads(MONITORED_PROJECTS_PATH.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            LOG("monitored_projects.json is not a list, returning empty")
            return []
        return data
    except (json.JSONDecodeError, OSError) as e:
        LOG(f"Error reading monitored_projects.json: {e}")
        return []


# ---------------------------------------------------------------------------
# Git Observer -- main entry point
# ---------------------------------------------------------------------------


def run_git_observer(db_path: str) -> None:
    """Main entry point for the Git Observer.

    Analyzes git history across all monitored projects, generates a weekly
    coding profile, detects style evolution, and saves results to memory.

    Args:
        db_path: Path to memory.db.
    """
    LOG("=== GIT OBSERVER STARTING ===")

    # Dedup: skip if already ran today
    _state_file = MEMORY_DIR / "git_observer_state.json"
    _today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    try:
        if _state_file.is_file():
            with open(_state_file, encoding="utf-8") as _sf:
                _st = json.loads(_sf.read())
            if _st.get("last_run_date") == _today:
                LOG("Already ran today, skipping (dedup)")
                return
    except Exception:
        pass

    projects = _load_monitored_projects()
    if not projects:
        LOG(
            "No monitored projects configured. Add projects to "
            f"{MONITORED_PROJECTS_PATH}"
        )
        LOG(
            'Format: [{"path": "/path/to/project", "name": "project-name"}, ...]'
        )
        LOG("=== GIT OBSERVER COMPLETE (no projects) ===")
        return

    # Analyze each project
    analyses: list[dict[str, Any]] = []
    for proj in projects:
        project_path = proj.get("path", "")
        project_name = proj.get("name", Path(project_path).name)

        if not project_path or not Path(project_path).is_dir():
            LOG(f"Skipping {project_name}: path not found ({project_path})")
            continue

        LOG(f"Analyzing {project_name}...")
        analysis = _analyze_project_git(project_path, project_name)
        analyses.append(analysis)

    active_analyses = [a for a in analyses if a["total_commits"] > 0]

    if not active_analyses:
        LOG("No git activity found in any monitored project")
        LOG("=== GIT OBSERVER COMPLETE (no activity) ===")
        return

    # Generate weekly coding profile
    LOG("Generating coding profile...")
    profile = _generate_coding_profile(active_analyses)
    LOG(f"Profile: {profile[:200]}...")

    # Save coding profile to memory
    total_commits = sum(a["total_commits"] for a in active_analyses)
    active_names = [a["name"] for a in active_analyses]

    _save_knowledge_record(
        db_path,
        content=(
            f"[Weekly Coding Profile] {datetime.now().strftime('%Y-%m-%d')}\n\n"
            f"{profile}\n\n"
            f"Raw stats: {total_commits} commits across "
            f"{', '.join(active_names)}"
        ),
        record_type="fact",
        tags=["git-observer", "coding-pattern", "weekly"],
        source="git-observer",
        confidence=0.8,
        project="general",
    )

    # Detect style evolution
    LOG("Detecting style evolution...")
    evolution_insights = _detect_style_evolution(db_path, active_analyses)

    if evolution_insights:
        LOG(f"Found {len(evolution_insights)} evolution insight(s)")
        for insight in evolution_insights:
            _save_knowledge_record(
                db_path,
                content=(
                    f"[Code Style Evolution] "
                    f"{datetime.now().strftime('%Y-%m-%d')}\n\n"
                    f"{insight}"
                ),
                record_type="lesson",
                tags=["code-evolution", "skill-tracking", "git-observer"],
                source="git-observer",
                confidence=0.6,
                project="general",
            )
            LOG(f"  Saved: {insight[:100]}...")

    # Telegram digest
    total_added = sum(a["lines_added"] for a in active_analyses)
    total_removed = sum(a["lines_removed"] for a in active_analyses)

    telegram_msg = (
        f"<b>Git Observer Weekly</b>\n\n"
        f"{profile}\n\n"
        f"<b>Stats:</b> {total_commits} commits, "
        f"+{total_added}/-{total_removed} lines\n"
        f"<b>Projects:</b> {', '.join(active_names)}"
    )

    if evolution_insights:
        telegram_msg += "\n\n<b>Evolution:</b>\n"
        for ins in evolution_insights[:3]:
            telegram_msg += f"- {ins[:200]}\n"

    _notify_telegram(telegram_msg)

    # Save state to prevent duplicates
    try:
        with open(_state_file, "w", encoding="utf-8") as _sf:
            json.dump({"last_run_date": _today}, _sf)
    except Exception:
        pass

    LOG(
        f"=== GIT OBSERVER COMPLETE === "
        f"({len(active_analyses)} projects, {total_commits} commits, "
        f"{len(evolution_insights)} evolution insights)"
    )


# ---------------------------------------------------------------------------
# Error Pattern Miner -- main entry point
# ---------------------------------------------------------------------------


def run_error_pattern_miner(db_path: str) -> None:
    """Analyze the errors table for recurring patterns and search for solutions.

    Groups errors by category, finds patterns with 3+ occurrences in 30 days,
    searches web for solutions, and saves them to memory.

    Args:
        db_path: Path to memory.db.
    """
    LOG("=== ERROR PATTERN MINER STARTING ===")

    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row

    cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )

    # Find recurring error categories (3+ in 30 days)
    try:
        patterns = db.execute(
            """SELECT category, COUNT(*) as cnt,
                      GROUP_CONCAT(description, ' ||| ') as descriptions
               FROM errors
               WHERE created_at > ? AND status = 'open'
               GROUP BY category
               HAVING cnt >= 3
               ORDER BY cnt DESC
               LIMIT 10""",
            (cutoff,),
        ).fetchall()
    except Exception as e:
        LOG(f"Error querying patterns: {e}")
        db.close()
        return

    if not patterns:
        LOG("No recurring error patterns found (threshold: 3+ in 30 days)")
        db.close()
        LOG("=== ERROR PATTERN MINER COMPLETE (no patterns) ===")
        return

    LOG(f"Found {len(patterns)} recurring error pattern(s)")

    # Check which patterns already have mined solutions (avoid re-mining)
    already_mined: set[str] = set()
    try:
        recent_solutions = db.execute(
            """SELECT content FROM knowledge
               WHERE tags LIKE '%error-pattern%'
                 AND tags LIKE '%auto-fix%'
                 AND status = 'active'
                 AND created_at > ?
               ORDER BY created_at DESC""",
            (cutoff,),
        ).fetchall()
        for row in recent_solutions:
            cat_match = re.search(r"\[Error Pattern: (.+?)\]", row["content"])
            if cat_match:
                already_mined.add(cat_match.group(1).lower())
    except Exception:
        pass

    db.close()

    # Import web_search (from scheduler.py or fallback)
    _web_search = None
    try:
        from reflection.scheduler import web_search
        _web_search = web_search
    except ImportError:
        LOG("web_search not available from scheduler, trying local import")
        try:
            from tools.task_manager import web_search as ws2
            _web_search = ws2
        except ImportError:
            LOG("web_search not available at all, skipping web research")

    solutions_found = 0

    for pattern in patterns:
        category = pattern["category"]
        count = pattern["cnt"]
        descriptions_raw = pattern["descriptions"] or ""

        # Skip already-mined categories
        if category.lower() in already_mined:
            LOG(f"  Skipping {category}: already mined recently")
            continue

        # Parse unique error descriptions (take first 3)
        descriptions = [d.strip() for d in descriptions_raw.split("|||")][:3]
        desc_summary = "; ".join(d[:100] for d in descriptions)

        LOG(f"  Mining: {category} ({count} occurrences)")
        LOG(f"    Examples: {desc_summary[:200]}")

        # Web search for solutions (if available)
        web_snippets = ""
        if _web_search is not None:
            # Build search query from category + sample description words
            words = descriptions[0].split()[:6] if descriptions else []
            search_query = f"{' '.join(words)} {category} fix solution"

            try:
                results = _web_search(search_query, max_results=3)
                if results:
                    snippets_list = []
                    for r in results[:3]:
                        if r.get("snippet"):
                            snippets_list.append(
                                f"- {r['title']}: {r['snippet']}"
                            )
                    web_snippets = "\n".join(snippets_list)
            except Exception as e:
                LOG(f"  Web search error for {category}: {e}")

            time.sleep(2)  # Rate limit

        # Ask Ollama to synthesize a solution
        prompt = (
            f"I keep encountering this error pattern: {category}\n\n"
            f"Example errors:\n{desc_summary}\n\n"
        )
        if web_snippets:
            prompt += f"Web search results:\n{web_snippets}\n\n"
        prompt += (
            f"Provide a concise actionable fix (3-5 sentences). "
            f"Include specific code patterns or commands if applicable."
        )

        solution = _ollama_generate(prompt, timeout=120)
        if not solution:
            LOG(f"  Ollama returned empty for {category}")
            continue

        # Save solution to memory
        _save_knowledge_record(
            db_path,
            content=(
                f"[Error Pattern: {category}] "
                f"({count} occurrences in 30 days)\n\n"
                f"Examples: {desc_summary}\n\n"
                f"Solution: {solution}"
            ),
            record_type="solution",
            tags=["error-pattern", "auto-fix", category, "reusable"],
            source="error-miner",
            confidence=0.5,
            project="general",
        )
        solutions_found += 1
        LOG(f"  Saved solution for: {category}")

        time.sleep(2)  # Rate limit between Ollama calls

    # Telegram notification
    if solutions_found > 0:
        _notify_telegram(
            f"<b>Error Pattern Miner</b>\n\n"
            f"Found solutions for {solutions_found} recurring error "
            f"pattern(s).\n"
            f"Total patterns analyzed: {len(patterns)}\n\n"
            f"Use <code>/search error-pattern</code> to see them."
        )

    LOG(
        f"=== ERROR PATTERN MINER COMPLETE === "
        f"({solutions_found} solutions found from {len(patterns)} patterns)"
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Git Observer & Error Pattern Miner"
    )
    parser.add_argument(
        "--db",
        default=str(MEMORY_DIR / "memory.db"),
        help="Path to memory.db",
    )
    parser.add_argument(
        "--run",
        choices=["git", "errors", "all"],
        default="all",
        help="Which analysis to run (default: all)",
    )
    args = parser.parse_args()

    if args.run in ("git", "all"):
        run_git_observer(args.db)

    if args.run in ("errors", "all"):
        run_error_pattern_miner(args.db)
