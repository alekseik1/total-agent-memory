#!/usr/bin/env python3
"""
Predictive Assistant -- anticipates user needs and prepares information in advance.

Features:
- Smart Morning Briefing (daily 8:00 AM): enhanced daily overview with git activity,
  project predictions, task dashboard, dependency alerts, and code weather
- Pre-Research Engine (every 4 hours): proactively researches topics found in recent
  knowledge entries
- Context Predictor: predicts which project the user will work on today
- Deadline Intelligence: velocity-based deadline feasibility analysis

Usage:
    from tools.predictive import run_smart_briefing, run_pre_research
    run_smart_briefing("/path/to/memory.db")
    run_pre_research("/path/to/memory.db")
"""

from __future__ import annotations

import json
import re
import sqlite3
import subprocess
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from background_writes import save_background_knowledge  # noqa: E402
from tool_support import ollama_generate  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent))

from paths import memory_dir  # noqa: E402

MEMORY_DIR = memory_dir()
TELEGRAM_ENV_PATH = Path(__file__).parent.parent.parent / "telegram" / ".env"

LOG = lambda msg: sys.stderr.write(
    f"[predictive] {datetime.now().strftime('%H:%M:%S')} {msg}\n"
)

# ---------------------------------------------------------------------------
# Reusable helpers (imported from scheduler patterns)
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

def _telegram_send(
    token: str, chat_id: int, text: str, parse_mode: str = "HTML"
) -> bool:
    """Send a Telegram message via Bot API. Returns True on success."""
    import urllib.request

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = json.dumps(
        {
            "chat_id": chat_id,
            "text": text[:4096],
            "parse_mode": parse_mode,
        }
    ).encode()
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
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
    """Save a knowledge record to DB."""
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


def _send_to_all_users(message: str) -> int:
    """Send Telegram message to all allowed users. Returns count of sends."""
    env = _load_dotenv(TELEGRAM_ENV_PATH)
    token = env.get("TELEGRAM_BOT_TOKEN", "")
    allowed = env.get("TELEGRAM_ALLOWED_USERS", "")
    if not token or not allowed:
        LOG("Telegram credentials not configured")
        return 0
    sent = 0
    for uid in allowed.split(","):
        uid = uid.strip()
        if uid:
            try:
                if _telegram_send(token, int(uid), message):
                    sent += 1
            except ValueError:
                pass
    return sent


# ---------------------------------------------------------------------------
# Web search helpers (from scheduler)
# ---------------------------------------------------------------------------


def _web_search(query: str, max_results: int = 5) -> list[dict[str, str]]:
    """Search DuckDuckGo HTML and return list of {title, url, snippet}."""
    import html as _html
    import urllib.parse
    import urllib.request

    ctx = _make_ssl_context()
    encoded_query = urllib.parse.quote_plus(query)
    url = f"https://html.duckduckgo.com/html/?q={encoded_query}"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 ClaudeMemory/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30, context=ctx) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        LOG(f"web_search fetch error: {e}")
        return []

    results: list[dict[str, str]] = []
    title_pattern = re.compile(
        r'<a[^>]+class="result__a"[^>]+href="([^"]*)"[^>]*>(.*?)</a>',
        re.DOTALL | re.IGNORECASE,
    )
    snippet_pattern = re.compile(
        r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>',
        re.DOTALL | re.IGNORECASE,
    )
    titles_urls = title_pattern.findall(html)
    snippets = snippet_pattern.findall(html)

    for i, (raw_url, raw_title) in enumerate(titles_urls):
        if i >= max_results:
            break
        clean_title = _html.unescape(re.sub(r"<[^>]+>", "", raw_title).strip())
        clean_snippet = ""
        if i < len(snippets):
            clean_snippet = _html.unescape(
                re.sub(r"<[^>]+>", "", snippets[i]).strip()
            )
        if "uddg=" in raw_url:
            match = re.search(r"uddg=([^&]+)", raw_url)
            if match:
                raw_url = urllib.parse.unquote(match.group(1))
        if clean_title and raw_url:
            results.append(
                {"title": clean_title, "url": raw_url, "snippet": clean_snippet}
            )
    return results


def _fetch_and_summarize(url: str, max_chars: int = 5000) -> str:
    """Fetch URL content, strip HTML, truncate to max_chars."""
    import html as _html
    import urllib.request

    ctx = _make_ssl_context()
    req = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0 ClaudeMemory/1.0"}
    )
    try:
        with urllib.request.urlopen(req, timeout=20, context=ctx) as resp:
            content_type = resp.headers.get("Content-Type", "")
            if "text" not in content_type and "html" not in content_type:
                return ""
            raw = resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        LOG(f"fetch_and_summarize error for {url}: {e}")
        return ""

    text = re.sub(r"<script[^>]*>.*?</script>", " ", raw, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = _html.unescape(text)
    return text[:max_chars]


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

# Projects with git repos -- paths detected from common locations
_PROJECT_DIRS: list[Path] = []


def _discover_project_dirs() -> list[Path]:
    """Discover git project directories from home and common dev folders."""
    global _PROJECT_DIRS
    if _PROJECT_DIRS:
        return _PROJECT_DIRS

    home = Path.home()
    candidates: list[Path] = []

    # Check common dev directories
    for dev_dir in [home / "projects", home / "dev", home / "code", home / "work", home]:
        if dev_dir.is_dir():
            try:
                for child in dev_dir.iterdir():
                    if child.is_dir() and (child / ".git").is_dir():
                        candidates.append(child)
            except PermissionError:
                pass

    # Limit to 15 most recently modified
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    _PROJECT_DIRS = candidates[:15]
    return _PROJECT_DIRS


def _git_log_yesterday(project_dir: Path) -> list[str]:
    """Get git log entries from yesterday for a project directory."""
    try:
        result = subprocess.run(
            ["git", "log", "--since=yesterday", "--oneline", "--no-merges"],
            capture_output=True,
            text=True, encoding="utf-8", errors="replace",
            timeout=10,
            cwd=str(project_dir),
        )
        if result.returncode == 0 and result.stdout.strip():
            return [
                line.strip()
                for line in result.stdout.strip().split("\n")
                if line.strip()
            ]
    except Exception:
        pass
    return []


# ---------------------------------------------------------------------------
# 3. Context Predictor
# ---------------------------------------------------------------------------


def _predict_today_context(db_path: str) -> dict[str, Any]:
    """Predict what project the user will work on today.

    Analyzes:
    - Day-of-week patterns from knowledge history
    - Recent activity (last 3 days)
    - Task deadlines

    Returns:
        {"predicted_project": str, "confidence": float,
         "reason": str, "suggested_focus": list[str]}
    """
    now = datetime.now(timezone.utc)
    today_weekday = now.weekday()  # 0=Monday, 6=Sunday
    cutoff_3d = (now - timedelta(days=3)).strftime("%Y-%m-%dT%H:%M:%SZ")
    cutoff_30d = (now - timedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%SZ")

    project_scores: dict[str, float] = {}
    reasons: dict[str, list[str]] = {}

    try:
        db = sqlite3.connect(db_path)
        db.row_factory = sqlite3.Row

        # Factor 1: Day-of-week patterns (last 30 days)
        # Which projects were active on the same weekday?
        rows = db.execute(
            """SELECT project, created_at FROM knowledge
               WHERE status = 'active'
                 AND project IS NOT NULL AND project != 'general'
                 AND created_at > ?
               ORDER BY created_at DESC""",
            (cutoff_30d,),
        ).fetchall()

        weekday_counts: dict[str, int] = {}
        for row in rows:
            try:
                dt = datetime.fromisoformat(row["created_at"].replace("Z", "+00:00"))
                if dt.weekday() == today_weekday:
                    proj = row["project"]
                    weekday_counts[proj] = weekday_counts.get(proj, 0) + 1
            except (ValueError, AttributeError):
                pass

        if weekday_counts:
            max_count = max(weekday_counts.values())
            for proj, cnt in weekday_counts.items():
                score = (cnt / max_count) * 0.4  # Weight: 40%
                project_scores[proj] = project_scores.get(proj, 0) + score
                if cnt >= 2:
                    day_names = [
                        "Monday", "Tuesday", "Wednesday", "Thursday",
                        "Friday", "Saturday", "Sunday",
                    ]
                    reasons.setdefault(proj, []).append(
                        f"active on {day_names[today_weekday]}s ({cnt} times/30d)"
                    )

        # Factor 2: Recent activity (last 3 days) -- momentum
        recent_rows = db.execute(
            """SELECT project, COUNT(*) as cnt FROM knowledge
               WHERE status = 'active'
                 AND project IS NOT NULL AND project != 'general'
                 AND created_at > ?
               GROUP BY project ORDER BY cnt DESC LIMIT 10""",
            (cutoff_3d,),
        ).fetchall()

        if recent_rows:
            max_recent = recent_rows[0]["cnt"] if recent_rows else 1
            for row in recent_rows:
                proj = row["project"]
                score = (row["cnt"] / max_recent) * 0.35  # Weight: 35%
                project_scores[proj] = project_scores.get(proj, 0) + score
                reasons.setdefault(proj, []).append(
                    f"recent momentum ({row['cnt']} entries/3d)"
                )

        # Factor 3: Task deadlines -- urgency
        try:
            cutoff_deadline = (now + timedelta(days=3)).strftime("%Y-%m-%dT%H:%M:%SZ")
            now_str = now.strftime("%Y-%m-%dT%H:%M:%SZ")
            # Tasks table may have research_query hinting at project
            task_rows = db.execute(
                """SELECT title, deadline, research_query FROM tasks
                   WHERE status NOT IN ('done')
                     AND deadline IS NOT NULL
                     AND deadline >= ? AND deadline <= ?
                   ORDER BY deadline ASC""",
                (now_str, cutoff_deadline),
            ).fetchall()

            for task in task_rows:
                # Try to match task title keywords with known projects
                title_lower = (task["title"] or "").lower()
                for proj in set(project_scores.keys()):
                    proj_lower = proj.lower()
                    if proj_lower in title_lower or title_lower in proj_lower:
                        project_scores[proj] = project_scores.get(proj, 0) + 0.25
                        reasons.setdefault(proj, []).append(
                            f"deadline task: {task['title'][:40]}"
                        )
                        break
        except Exception:
            pass  # tasks table may not exist

        db.close()
    except Exception as e:
        LOG(f"Context predictor DB error: {e}")

    if not project_scores:
        return {
            "predicted_project": "unknown",
            "confidence": 0.0,
            "reason": "insufficient data",
            "suggested_focus": [],
        }

    # Sort by score
    sorted_projects = sorted(project_scores.items(), key=lambda x: x[1], reverse=True)
    top_project = sorted_projects[0][0]
    top_score = sorted_projects[0][1]

    # Confidence: normalize to 0-1 (max theoretical score = 1.0)
    confidence = min(top_score, 1.0)

    # Suggested focus: top reasons for the predicted project
    focus = reasons.get(top_project, [])[:3]

    # Add runner-ups as alternatives
    suggested = list(focus)
    if len(sorted_projects) > 1:
        runner_up = sorted_projects[1][0]
        suggested.append(f"also consider: {runner_up}")

    return {
        "predicted_project": top_project,
        "confidence": round(confidence, 2),
        "reason": "; ".join(reasons.get(top_project, ["pattern analysis"])),
        "suggested_focus": suggested,
    }


# ---------------------------------------------------------------------------
# 4. Deadline Intelligence
# ---------------------------------------------------------------------------


def _analyze_deadlines(db_path: str) -> str:
    """Analyze task deadlines vs velocity to estimate feasibility.

    Returns Telegram-friendly HTML text.
    """
    now = datetime.now(timezone.utc)
    now_str = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    cutoff_14d = (now - timedelta(days=14)).strftime("%Y-%m-%dT%H:%M:%SZ")

    try:
        db = sqlite3.connect(db_path)
        db.row_factory = sqlite3.Row

        # Calculate velocity: tasks completed in last 14 days
        done_rows = db.execute(
            """SELECT COUNT(*) as cnt FROM tasks
               WHERE status = 'done' AND updated_at > ?""",
            (cutoff_14d,),
        ).fetchone()
        tasks_done_14d = done_rows["cnt"] if done_rows else 0
        daily_velocity = tasks_done_14d / 14.0

        # Get all pending/active tasks with deadlines
        upcoming_tasks = db.execute(
            """SELECT id, title, deadline, status FROM tasks
               WHERE status NOT IN ('done')
                 AND deadline IS NOT NULL AND deadline >= ?
               ORDER BY deadline ASC""",
            (now_str,),
        ).fetchall()

        db.close()
    except Exception as e:
        LOG(f"Deadline intelligence DB error: {e}")
        return ""

    if not upcoming_tasks:
        return ""

    lines: list[str] = []
    lines.append(f"<b>\u23f0 Deadline Intelligence</b>")
    lines.append(
        f"Velocity: {daily_velocity:.1f} tasks/day "
        f"({tasks_done_14d} done in 14d)"
    )
    lines.append("")

    at_risk = 0
    for task in upcoming_tasks:
        try:
            deadline_dt = datetime.fromisoformat(
                task["deadline"].replace("Z", "+00:00")
            )
            days_left = (deadline_dt - now).total_seconds() / 86400
            title = task["title"][:45]
            deadline_str = deadline_dt.strftime("%b %d")

            if days_left < 1:
                status_icon = "\u2757"  # exclamation
                at_risk += 1
            elif days_left < 2 and daily_velocity < 1:
                status_icon = "\u26a0\ufe0f"  # warning
                at_risk += 1
            else:
                status_icon = "\u2705"  # check

            lines.append(
                f"  {status_icon} #{task['id']} {title} "
                f"({deadline_str}, {days_left:.0f}d left)"
            )
        except (ValueError, AttributeError):
            continue

    if at_risk > 0:
        lines.append("")
        lines.append(
            f"\u26a0\ufe0f <b>{at_risk} task(s) at risk</b> based on current velocity"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helper: Code Weather (sentiment from errors vs solutions)
# ---------------------------------------------------------------------------


def _code_weather(db_path: str, days: int = 7) -> tuple[str, str]:
    """Analyze recent errors vs solutions to determine 'code weather'.

    Returns (emoji, description).
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )

    errors_count = 0
    solutions_count = 0

    try:
        db = sqlite3.connect(db_path)
        db.row_factory = sqlite3.Row

        # Count errors
        row = db.execute(
            "SELECT COUNT(*) as cnt FROM errors WHERE created_at > ?",
            (cutoff,),
        ).fetchone()
        errors_count = row["cnt"] if row else 0

        # Count solutions
        row = db.execute(
            """SELECT COUNT(*) as cnt FROM knowledge
               WHERE type = 'solution' AND status = 'active' AND created_at > ?""",
            (cutoff,),
        ).fetchone()
        solutions_count = row["cnt"] if row else 0

        db.close()
    except Exception:
        pass

    total = errors_count + solutions_count
    if total == 0:
        return "\u2601\ufe0f", "Cloudy (no data)"

    ratio = solutions_count / total

    if ratio >= 0.7:
        return "\u2600\ufe0f", f"Sunny ({solutions_count} solutions, {errors_count} errors)"
    elif ratio >= 0.4:
        return "\u26c5", f"Partly Cloudy ({solutions_count} solutions, {errors_count} errors)"
    elif ratio >= 0.2:
        return "\u2601\ufe0f", f"Cloudy ({solutions_count} solutions, {errors_count} errors)"
    else:
        return "\u26c8\ufe0f", f"Stormy ({errors_count} errors, only {solutions_count} solutions)"


# ---------------------------------------------------------------------------
# 1. Smart Morning Briefing
# ---------------------------------------------------------------------------


def run_smart_briefing(db_path: str) -> None:
    """Enhanced daily morning briefing with predictions and intelligence.

    Replaces the basic advisor. Runs daily at 8:00 AM.

    Sections:
    - Git Activity Summary (yesterday's commits per project)
    - Today's Prediction (which project user will likely work on)
    - Task Dashboard (upcoming deadlines within 3 days)
    - Dependency Alerts (new releases from tech-radar/GitHub monitor)
    - Knowledge Growth (yesterday + total)
    - Open Blind Spots (relevant to predicted project)
    - Weather of Code (error vs solution sentiment)
    """
    LOG("=== SMART MORNING BRIEFING STARTING ===")

    today = datetime.now().strftime("%A, %B %d")
    now = datetime.now(timezone.utc)
    yesterday_str = (now - timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")

    sections: list[str] = []
    sections.append(f"<b>\u2615 Smart Morning Briefing -- {today}</b>")
    sections.append("")

    # ----- Section 1: Git Activity Summary -----
    LOG("[briefing] Collecting git activity...")
    git_lines: list[str] = []
    project_dirs = _discover_project_dirs()
    active_git_projects: list[str] = []

    for proj_dir in project_dirs:
        commits = _git_log_yesterday(proj_dir)
        if commits:
            proj_name = proj_dir.name
            active_git_projects.append(proj_name)
            git_lines.append(f"  <b>{proj_name}</b> ({len(commits)} commits):")
            for commit in commits[:3]:
                # Truncate long commit messages
                git_lines.append(f"    \u2022 {commit[:70]}")
            if len(commits) > 3:
                git_lines.append(f"    ... and {len(commits) - 3} more")

    if git_lines:
        sections.append("<b>\U0001f4bb Git Activity (Yesterday)</b>")
        sections.extend(git_lines)
    else:
        sections.append("<b>\U0001f4bb Git Activity</b>: No commits yesterday")
    sections.append("")

    # ----- Section 2: Today's Prediction -----
    LOG("[briefing] Predicting today's context...")
    prediction = _predict_today_context(db_path)
    pred_project = prediction["predicted_project"]
    pred_confidence = prediction["confidence"]
    pred_reason = prediction["reason"]

    confidence_bar = "\u2588" * int(pred_confidence * 10) + "\u2591" * (
        10 - int(pred_confidence * 10)
    )

    sections.append("<b>\U0001f52e Today's Prediction</b>")
    if pred_project != "unknown":
        sections.append(
            f"  Project: <b>{pred_project}</b> "
            f"[{confidence_bar}] {pred_confidence:.0%}"
        )
        sections.append(f"  Why: {pred_reason[:120]}")
        if prediction["suggested_focus"]:
            for focus in prediction["suggested_focus"][:3]:
                sections.append(f"  \u2022 {focus}")
    else:
        sections.append("  Not enough data for prediction yet")
    sections.append("")

    # ----- Section 3: Task Dashboard -----
    LOG("[briefing] Building task dashboard...")
    try:
        db = sqlite3.connect(db_path)
        db.row_factory = sqlite3.Row

        cutoff_3d = (now + timedelta(days=3)).strftime("%Y-%m-%dT%H:%M:%SZ")
        now_str = now.strftime("%Y-%m-%dT%H:%M:%SZ")

        tasks = db.execute(
            """SELECT id, title, deadline, status, research_results FROM tasks
               WHERE status NOT IN ('done')
               ORDER BY deadline ASC NULLS LAST
               LIMIT 10""",
        ).fetchall()

        db.close()

        if tasks:
            sections.append("<b>\U0001f4cb Task Dashboard</b>")
            for task in tasks:
                status_icons = {
                    "pending": "\u23f3",
                    "researching": "\U0001f50d",
                    "ready": "\u2705",
                    "overdue": "\u26a0\ufe0f",
                }
                icon = status_icons.get(task["status"], "\u2753")
                title = task["title"][:50]
                deadline_info = ""
                if task["deadline"]:
                    try:
                        dl = datetime.fromisoformat(
                            task["deadline"].replace("Z", "+00:00")
                        )
                        days_left = (dl - now).total_seconds() / 86400
                        if days_left < 0:
                            deadline_info = " <b>OVERDUE</b>"
                        elif days_left < 1:
                            deadline_info = " \u2757 TODAY"
                        elif days_left < 2:
                            deadline_info = " tomorrow"
                        else:
                            deadline_info = f" ({dl.strftime('%b %d')})"
                    except ValueError:
                        pass

                has_research = "\U0001f4da" if task["research_results"] else ""
                sections.append(
                    f"  {icon} #{task['id']} {title}{deadline_info} {has_research}"
                )
            sections.append("")
    except Exception as e:
        LOG(f"[briefing] Task dashboard error: {e}")

    # Deadline intelligence (velocity analysis)
    deadline_intel = _analyze_deadlines(db_path)
    if deadline_intel:
        sections.append(deadline_intel)
        sections.append("")

    # ----- Section 4: Dependency Alerts -----
    LOG("[briefing] Checking dependency alerts...")
    try:
        db = sqlite3.connect(db_path)
        db.row_factory = sqlite3.Row

        # Check for new releases saved by github_monitor in last 24h
        release_rows = db.execute(
            """SELECT content FROM knowledge
               WHERE source = 'github' AND status = 'active'
                 AND created_at > ?
               ORDER BY created_at DESC LIMIT 5""",
            (yesterday_str,),
        ).fetchall()

        db.close()

        if release_rows:
            sections.append("<b>\U0001f4e6 New Releases (24h)</b>")
            for row in release_rows:
                # First line is usually "New release: owner/repo vX.Y.Z"
                first_line = row["content"].split("\n")[0][:80]
                sections.append(f"  \u2022 {first_line}")
            sections.append("")
    except Exception as e:
        LOG(f"[briefing] Dependency alerts error: {e}")

    # ----- Section 5: Knowledge Growth -----
    LOG("[briefing] Calculating knowledge growth...")
    try:
        db = sqlite3.connect(db_path)
        db.row_factory = sqlite3.Row

        yesterday_count_row = db.execute(
            "SELECT COUNT(*) as cnt FROM knowledge WHERE created_at > ? AND status = 'active'",
            (yesterday_str,),
        ).fetchone()
        total_row = db.execute(
            "SELECT COUNT(*) as cnt FROM knowledge WHERE status = 'active'"
        ).fetchone()

        yesterday_count = yesterday_count_row["cnt"] if yesterday_count_row else 0
        total_count = total_row["cnt"] if total_row else 0

        db.close()

        sections.append("<b>\U0001f4c8 Knowledge Growth</b>")
        sections.append(f"  Yesterday: +{yesterday_count} memories")
        sections.append(f"  Total: {total_count} active records")
        sections.append("")
    except Exception as e:
        LOG(f"[briefing] Knowledge growth error: {e}")

    # ----- Section 6: Open Blind Spots -----
    LOG("[briefing] Checking blind spots...")
    try:
        db = sqlite3.connect(db_path)
        db.row_factory = sqlite3.Row

        blind_spots = db.execute(
            """SELECT description, severity, domains FROM blind_spots
               WHERE status = 'active'
               ORDER BY severity DESC LIMIT 5"""
        ).fetchall()

        db.close()

        if blind_spots:
            # Filter for those relevant to predicted project
            relevant: list[dict] = []
            other: list[dict] = []
            for bs in blind_spots:
                domains_raw = bs["domains"] or "[]"
                try:
                    domains = json.loads(domains_raw) if domains_raw else []
                except (json.JSONDecodeError, TypeError):
                    domains = []
                bs_dict = {
                    "description": bs["description"],
                    "severity": bs["severity"],
                    "domains": domains,
                }
                if pred_project != "unknown" and any(
                    pred_project.lower() in d.lower() for d in domains
                ):
                    relevant.append(bs_dict)
                else:
                    other.append(bs_dict)

            all_spots = relevant + other
            if all_spots:
                sections.append(
                    f"<b>\U0001f441 Blind Spots</b> ({len(all_spots)} active)"
                )
                for bs in all_spots[:3]:
                    sev = bs["severity"]
                    sev_icon = (
                        "\U0001f534" if sev >= 0.7
                        else "\U0001f7e1" if sev >= 0.4
                        else "\U0001f7e2"
                    )
                    desc = bs["description"][:60]
                    sections.append(f"  {sev_icon} {desc}")
                sections.append("")
    except Exception as e:
        LOG(f"[briefing] Blind spots error: {e}")

    # ----- Section 7: Weather of Code -----
    weather_emoji, weather_desc = _code_weather(db_path, days=7)
    sections.append(f"<b>{weather_emoji} Code Weather</b>: {weather_desc}")
    sections.append("")

    # ----- Footer -----
    sections.append("<i>Generated by Predictive Assistant</i>")

    message = "\n".join(sections)

    # Send via Telegram
    sent = _send_to_all_users(message)
    LOG(f"[briefing] Message sent to {sent} user(s)")

    # Save to memory
    _save_knowledge_record(
        db_path,
        content=f"[Smart Morning Briefing] {today}\n\n"
        + "\n".join(
            line
            for line in sections
            if not line.startswith("<i>") and not line.startswith("<b>\u2615")
        ),
        record_type="fact",
        tags=["morning-briefing", datetime.now().strftime("%Y-%m-%d")],
        source="predictive",
        confidence=0.6,
    )

    LOG("=== SMART MORNING BRIEFING COMPLETE ===")


# ---------------------------------------------------------------------------
# 2. Pre-Research Engine
# ---------------------------------------------------------------------------


def run_pre_research(db_path: str) -> None:
    """Proactively research topics mentioned in recent knowledge entries.

    Runs every 4 hours. Looks for:
    - New technologies not well-documented in memory
    - Unanswered questions
    - TODOs from recent sessions

    For each topic (max 2 per run):
    - Web search via DuckDuckGo
    - Fetch top 3 articles
    - Summarize with Ollama
    - Save as knowledge
    - Notify via Telegram
    """
    LOG("=== PRE-RESEARCH ENGINE STARTING ===")

    cutoff_24h = (datetime.now(timezone.utc) - timedelta(hours=24)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )

    topics_to_research: list[dict[str, str]] = []

    try:
        db = sqlite3.connect(db_path)
        db.row_factory = sqlite3.Row

        # Source 1: Find recent entries with questions (? or question words)
        question_rows = db.execute(
            """SELECT content, project FROM knowledge
               WHERE status = 'active' AND created_at > ?
                 AND (content LIKE '%?%'
                      OR LOWER(content) LIKE '%how to%'
                      OR LOWER(content) LIKE '%как %'
                      OR LOWER(content) LIKE '%what is%'
                      OR LOWER(content) LIKE '%что такое%'
                      OR LOWER(content) LIKE '%why does%'
                      OR LOWER(content) LIKE '%почему%')
               ORDER BY created_at DESC LIMIT 10""",
            (cutoff_24h,),
        ).fetchall()

        for row in question_rows:
            content = row["content"]
            # Extract the question part
            for line in content.split("\n"):
                line = line.strip()
                if "?" in line and len(line) > 15 and len(line) < 200:
                    topics_to_research.append(
                        {"topic": line, "source": "question", "project": row["project"] or "general"}
                    )
                    break

        # Source 2: Find TODOs
        todo_rows = db.execute(
            """SELECT content, project FROM knowledge
               WHERE status = 'active' AND created_at > ?
                 AND (LOWER(content) LIKE '%todo%'
                      OR LOWER(content) LIKE '%to-do%'
                      OR LOWER(content) LIKE '%need to research%'
                      OR LOWER(content) LIKE '%изучить%'
                      OR LOWER(content) LIKE '%разобраться%')
               ORDER BY created_at DESC LIMIT 5""",
            (cutoff_24h,),
        ).fetchall()

        for row in todo_rows:
            content = row["content"]
            for line in content.split("\n"):
                line_lower = line.strip().lower()
                if any(
                    kw in line_lower
                    for kw in ["todo", "to-do", "need to research", "изучить", "разобраться"]
                ) and 15 < len(line.strip()) < 200:
                    topics_to_research.append(
                        {"topic": line.strip(), "source": "todo", "project": row["project"] or "general"}
                    )
                    break

        # Source 3: New technology mentions not well-documented
        # Find tags that appeared only 1-2 times recently (potentially new tech)
        tag_rows = db.execute(
            """SELECT tags FROM knowledge
               WHERE status = 'active' AND created_at > ?""",
            (cutoff_24h,),
        ).fetchall()

        tag_counts: dict[str, int] = {}
        for row in tag_rows:
            try:
                tags = json.loads(row["tags"]) if row["tags"] else []
                for tag in tags:
                    if tag and len(tag) > 3 and tag not in (
                        "auto", "session-autosave", "context-recovery",
                        "reusable", "morning-briefing", "pre-research",
                        "proactive", "rss", "news", "advisor", "daily-tip",
                        "curiosity", "auto-research", "dream", "auto-insight",
                        "cross-project", "github", "release",
                    ):
                        tag_counts[tag] = tag_counts.get(tag, 0) + 1
            except (json.JSONDecodeError, TypeError):
                pass

        # Check which of these rare tags have few entries overall
        for tag, recent_count in tag_counts.items():
            if recent_count <= 2:
                try:
                    total_row = db.execute(
                        """SELECT COUNT(*) as cnt FROM knowledge
                           WHERE status = 'active' AND tags LIKE ?""",
                        (f'%"{tag}"%',),
                    ).fetchone()
                    total = total_row["cnt"] if total_row else 0

                    if total <= 3:
                        topics_to_research.append(
                            {
                                "topic": f"{tag} (new technology/concept)",
                                "source": "new_tech",
                                "project": "general",
                            }
                        )
                except Exception:
                    pass

        db.close()
    except Exception as e:
        LOG(f"Pre-research topic scan error: {e}")
        return

    if not topics_to_research:
        LOG("[pre-research] No topics found for research")
        return

    # Deduplicate by topic similarity
    seen: set[str] = set()
    unique_topics: list[dict[str, str]] = []
    for t in topics_to_research:
        key = t["topic"][:50].lower()
        if key not in seen:
            seen.add(key)
            unique_topics.append(t)

    LOG(f"[pre-research] Found {len(unique_topics)} unique topics to research")

    # Process max 2 topics per run
    topics_researched = 0
    researched_names: list[str] = []

    for topic_info in unique_topics[:2]:
        topic = topic_info["topic"]
        project = topic_info["project"]
        LOG(f"[pre-research] Researching: {topic[:80]}...")

        # Clean up topic for search query
        search_query = re.sub(r"[\[\]#*\-\u2022]", "", topic).strip()
        search_query = re.sub(r"\b(todo|to-do|need to research|изучить|разобраться)\b", "", search_query, flags=re.IGNORECASE).strip()
        if len(search_query) < 5:
            continue

        try:
            # Search
            results = _web_search(search_query[:100], max_results=5)
            if not results:
                LOG(f"[pre-research] No search results for: {search_query[:60]}")
                time.sleep(2)
                continue

            # Fetch top 3 articles
            combined_text = ""
            fetched_count = 0
            for r in results:
                if fetched_count >= 3:
                    break
                content = _fetch_and_summarize(r["url"], max_chars=3000)
                if content and len(content) > 100:
                    combined_text += (
                        f"\n\n--- Source: {r['title']} ({r['url']}) ---\n{content}"
                    )
                    fetched_count += 1
                time.sleep(2)  # Rate limit

            if not combined_text.strip():
                LOG(f"[pre-research] No content fetched for: {topic[:60]}")
                continue

            # Summarize with Claude API (reliable) → fallback Ollama
            try:
                from tools.llm_router import research_and_summarize
                summary = research_and_summarize(topic, combined_text[:8000])
            except ImportError:
                summary = ""
            if not summary:
                # Fallback to Ollama
                prompt = (
                    f"Summarize this research for a software developer. "
                    f"Extract key facts, code examples if any, and actionable advice.\n\n"
                    f"Research topic: {topic}\n\n"
                    f"Sources:\n{combined_text[:8000]}\n\n"
                    f"Write a concise summary (max 300 words). "
                    f"Include links to the most useful sources."
                )
                summary = _ollama_generate(prompt, timeout=180)
            if not summary:
                LOG(f"[pre-research] Ollama returned empty for: {topic[:60]}")
                continue

            # Save to knowledge
            _save_knowledge_record(
                db_path,
                content=f"[Pre-Research] {topic}\n\n{summary}",
                record_type="fact",
                tags=["pre-research", "proactive", topic_info["source"]],
                source="predictive",
                confidence=0.5,
                project=project,
            )

            topics_researched += 1
            researched_names.append(topic[:50])
            LOG(f"[pre-research] Saved research on: {topic[:60]}")

        except Exception as e:
            LOG(f"[pre-research] Error researching '{topic[:60]}': {e}")

        time.sleep(2)  # Rate limit between topics

    # Telegram notification
    if topics_researched > 0:
        msg = (
            f"\U0001f4da Pre-researched {topics_researched} topic(s) for you:\n"
            + "\n".join(f"  \u2022 {name}" for name in researched_names)
        )
        sent = _send_to_all_users(msg)
        LOG(f"[pre-research] Notification sent to {sent} user(s)")

    LOG(
        f"=== PRE-RESEARCH ENGINE COMPLETE === "
        f"({topics_researched} topics researched)"
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Predictive Assistant")
    parser.add_argument("--db", default=str(MEMORY_DIR / "memory.db"))
    parser.add_argument(
        "--run",
        choices=["briefing", "pre-research", "predict"],
        required=True,
        help="Which function to run",
    )
    args = parser.parse_args()

    if args.run == "briefing":
        run_smart_briefing(args.db)
    elif args.run == "pre-research":
        run_pre_research(args.db)
    elif args.run == "predict":
        result = _predict_today_context(args.db)
        print(json.dumps(result, indent=2, ensure_ascii=False))
