#!/usr/bin/env python3
"""
Task Manager -- secretary system with deadlines, auto-research, and briefings.

Stores tasks in the `tasks` table of memory.db. Supports natural language
deadline parsing (Russian + English), web research via DuckDuckGo, and
briefing generation via Ollama.

Usage:
    from tools.task_manager import TaskManager
    tm = TaskManager("/path/to/memory.db")
    task_id = tm.create_task("Research microservices", deadline="through 4 days")
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import ssl
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tool_support import ollama_generate  # noqa: E402

LOG = lambda msg: sys.stderr.write(f"[task-manager] {msg}\n")


# ---------------------------------------------------------------------------
# Deadline parsing
# ---------------------------------------------------------------------------

_RU_MONTHS: dict[str, int] = {
    "январ": 1, "феврал": 2, "март": 3, "апрел": 4,
    "ма": 5, "июн": 6, "июл": 7, "август": 8,
    "сентябр": 9, "октябр": 10, "ноябр": 11, "декабр": 12,
}

_EN_MONTHS: dict[str, int] = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def _parse_deadline(text: str) -> str | None:
    """Parse a deadline from natural language. Returns ISO 8601 string or None."""
    text = text.strip().lower()
    now = datetime.now(timezone.utc)

    # ISO date: 2026-04-15 or 2026-04-15T...
    iso_match = re.match(r"(\d{4}-\d{2}-\d{2})", text)
    if iso_match:
        try:
            dt = datetime.fromisoformat(iso_match.group(1)).replace(tzinfo=timezone.utc)
            # Set to end of day
            dt = dt.replace(hour=23, minute=59, second=59)
            return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            pass

    # "tomorrow" / "zavtra"
    if text in ("завтра", "tomorrow"):
        dt = now + timedelta(days=1)
        return dt.replace(hour=23, minute=59, second=59).strftime("%Y-%m-%dT%H:%M:%SZ")

    # "послезавтра" / "day after tomorrow"
    if text in ("послезавтра", "day after tomorrow"):
        dt = now + timedelta(days=2)
        return dt.replace(hour=23, minute=59, second=59).strftime("%Y-%m-%dT%H:%M:%SZ")

    # "через N дней/дня/день" / "in N days/day"
    m = re.match(r"(?:через|in)\s+(\d+)\s+(?:дн[ейяь]|день|days?)", text)
    if m:
        days = int(m.group(1))
        dt = now + timedelta(days=days)
        return dt.replace(hour=23, minute=59, second=59).strftime("%Y-%m-%dT%H:%M:%SZ")

    # "через неделю" / "in a week"
    if re.match(r"(?:через\s+неделю|in\s+a?\s*week)", text):
        dt = now + timedelta(weeks=1)
        return dt.replace(hour=23, minute=59, second=59).strftime("%Y-%m-%dT%H:%M:%SZ")

    # "через N недель/недели" / "in N weeks"
    m = re.match(r"(?:через|in)\s+(\d+)\s+(?:недел[ьию]|weeks?)", text)
    if m:
        weeks = int(m.group(1))
        dt = now + timedelta(weeks=weeks)
        return dt.replace(hour=23, minute=59, second=59).strftime("%Y-%m-%dT%H:%M:%SZ")

    # "через N часов/часа/час" / "in N hours"
    m = re.match(r"(?:через|in)\s+(\d+)\s+(?:час[аов]*|hours?)", text)
    if m:
        hours = int(m.group(1))
        dt = now + timedelta(hours=hours)
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    # "через месяц" / "in a month"
    if re.match(r"(?:через\s+месяц|in\s+a?\s*month)", text):
        dt = now + timedelta(days=30)
        return dt.replace(hour=23, minute=59, second=59).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Russian date: "25 апреля", "5 мая"
    m = re.match(r"(\d{1,2})\s+([а-яё]+)", text)
    if m:
        day = int(m.group(1))
        month_prefix = m.group(2)[:5]
        for prefix, month_num in _RU_MONTHS.items():
            if month_prefix.startswith(prefix):
                year = now.year
                try:
                    dt = datetime(year, month_num, day, 23, 59, 59, tzinfo=timezone.utc)
                    if dt < now:
                        dt = dt.replace(year=year + 1)
                    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
                except ValueError:
                    break

    # English date: "April 25", "May 5"
    m = re.match(r"([a-z]+)\s+(\d{1,2})", text)
    if m:
        month_str = m.group(1)[:3]
        day = int(m.group(2))
        month_num = _EN_MONTHS.get(month_str)
        if month_num:
            year = now.year
            try:
                dt = datetime(year, month_num, day, 23, 59, 59, tzinfo=timezone.utc)
                if dt < now:
                    dt = dt.replace(year=year + 1)
                return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            except ValueError:
                pass

    # "к пятнице" / "by friday" -- next occurrence of weekday
    weekdays_ru = {
        "понедельник": 0, "вторник": 1, "сред": 2, "четверг": 3,
        "пятниц": 4, "суббот": 5, "воскресень": 6,
    }
    weekdays_en = {
        "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
        "friday": 4, "saturday": 5, "sunday": 6,
    }
    clean = re.sub(r"^(к|by|до|ко)\s+", "", text)
    for name, wd in {**weekdays_ru, **weekdays_en}.items():
        if clean.startswith(name):
            days_ahead = (wd - now.weekday()) % 7
            if days_ahead == 0:
                days_ahead = 7
            dt = now + timedelta(days=days_ahead)
            return dt.replace(hour=23, minute=59, second=59).strftime("%Y-%m-%dT%H:%M:%SZ")

    return None


def _extract_deadline_and_title(text: str) -> tuple[str | None, str]:
    """Split raw input into (deadline_str, title).

    Tries to detect deadline keywords at the start or after 'add'.
    Returns (parsed_deadline_iso, remaining_title).
    """
    text = text.strip()

    # Patterns that start with a deadline indicator
    deadline_patterns = [
        # "через 4 дня подготовить ..."
        (r"^(через\s+\S+\s+\S+)\s+(.+)", None),
        # "in 4 days prepare ..."
        (r"^(in\s+\d+\s+\S+)\s+(.+)", None),
        # "завтра сделать ..."
        (r"^(завтра|послезавтра|tomorrow)\s+(.+)", None),
        # "через неделю ..."
        (r"^(через\s+неделю)\s+(.+)", None),
        # "in a week ..."
        (r"^(in\s+a?\s*week)\s+(.+)", None),
        # "к пятнице ..."
        (r"^(к\s+\S+|ко\s+\S+|by\s+\S+|до\s+\S+)\s+(.+)", None),
        # "25 апреля ..."
        (r"^(\d{1,2}\s+[а-яёa-z]+)\s+(.+)", None),
        # ISO date at start
        (r"^(\d{4}-\d{2}-\d{2})\s+(.+)", None),
        # "April 25 ..." (English month + day)
        (r"^([A-Za-z]+\s+\d{1,2})\s+(.+)", None),
    ]

    for pattern, _ in deadline_patterns:
        m = re.match(pattern, text, re.IGNORECASE)
        if m:
            deadline_part = m.group(1)
            title_part = m.group(2)
            parsed = _parse_deadline(deadline_part)
            if parsed:
                return parsed, title_part.strip()

    # No deadline found -- whole text is the title
    return None, text


# ---------------------------------------------------------------------------
# Web search helpers
# ---------------------------------------------------------------------------

class _SimpleHTMLTextExtractor(HTMLParser):
    """Minimal HTML to text converter."""

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._skip: bool = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in ("script", "style", "nav", "footer", "header", "svg"):
            self._skip = True

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style", "nav", "footer", "header", "svg"):
            self._skip = False
        if tag in ("p", "div", "br", "h1", "h2", "h3", "h4", "li", "tr"):
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self._parts.append(data)

    def get_text(self) -> str:
        raw = "".join(self._parts)
        # Collapse whitespace
        lines = [line.strip() for line in raw.split("\n")]
        return "\n".join(line for line in lines if line)


def _make_ssl_context() -> ssl.SSLContext:
    """Verified TLS context: certifi, else the platform store, never unverified."""
    from tls_context import verified_context

    return verified_context()

def web_search(query: str, max_results: int = 5) -> list[dict[str, str]]:
    """Search DuckDuckGo HTML and return list of {title, url, snippet}."""
    encoded = urllib.parse.quote_plus(query)
    url = f"https://html.duckduckgo.com/html/?q={encoded}"

    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        },
    )

    results: list[dict[str, str]] = []
    try:
        ctx = _make_ssl_context()
        with urllib.request.urlopen(req, timeout=15, context=ctx) as resp:
            html = resp.read().decode("utf-8", errors="replace")

        # Parse DuckDuckGo HTML results
        # Links are in <a class="result__a" href="...">title</a>
        # Snippets in <a class="result__snippet" ...>text</a>
        link_pattern = re.compile(
            r'<a[^>]+class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
            re.DOTALL,
        )
        snippet_pattern = re.compile(
            r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>',
            re.DOTALL,
        )

        links = link_pattern.findall(html)
        snippets = snippet_pattern.findall(html)

        for i, (href, title_html) in enumerate(links[:max_results]):
            # DuckDuckGo redirects through uddg param
            actual_url = href
            if "uddg=" in href:
                parsed = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
                actual_url = parsed.get("uddg", [href])[0]

            # Strip HTML from title and snippet
            title = re.sub(r"<[^>]+>", "", title_html).strip()
            snippet = ""
            if i < len(snippets):
                snippet = re.sub(r"<[^>]+>", "", snippets[i]).strip()

            if title and actual_url:
                results.append({
                    "title": title,
                    "url": actual_url,
                    "snippet": snippet,
                })
    except Exception:
        pass

    return results


def fetch_and_summarize(url: str, max_chars: int = 3000) -> str:
    """Fetch a URL and return cleaned text content (truncated)."""
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        },
    )
    try:
        ctx = _make_ssl_context()
        with urllib.request.urlopen(req, timeout=15, context=ctx) as resp:
            content_type = resp.headers.get("Content-Type", "")
            if "text/html" not in content_type and "text/plain" not in content_type:
                return ""
            raw = resp.read().decode("utf-8", errors="replace")

        if "text/html" in content_type:
            extractor = _SimpleHTMLTextExtractor()
            extractor.feed(raw)
            text = extractor.get_text()
        else:
            text = raw

        return text[:max_chars]
    except Exception:
        return ""


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from text (terminal colors, cursor moves, etc.)."""
    return re.sub(r"\x1b\[[0-9;]*[A-Za-z]|\x1b\].*?\x07", "", text)


def _ollama_generate(prompt: str, model: str | None = None, timeout: int = 120) -> str:
    """Generate text with the Ollama CLI (prompt on stdin). Returns "" on failure."""
    return ollama_generate(prompt, model, timeout, log=LOG)


# ---------------------------------------------------------------------------
# TaskManager
# ---------------------------------------------------------------------------

class TaskManager:
    """Task storage and processing engine backed by SQLite."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._ensured = False

    def _get_db(self) -> sqlite3.Connection:
        """Open a database connection with row factory."""
        db = sqlite3.connect(self.db_path)
        db.row_factory = sqlite3.Row
        if not self._ensured:
            self.ensure_table(db)
            self._ensured = True
        return db

    def ensure_table(self, db: sqlite3.Connection | None = None) -> None:
        """Create the tasks table if it does not exist."""
        close = False
        if db is None:
            db = sqlite3.connect(self.db_path)
            close = True
        db.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                description TEXT,
                deadline TEXT,
                status TEXT DEFAULT 'pending',
                research_query TEXT,
                research_results TEXT,
                briefing TEXT,
                created_at TEXT,
                updated_at TEXT,
                notified_at TEXT,
                recurring TEXT,
                last_recurring_at TEXT
            )
        """)
        # Migration: add recurring columns if missing
        try:
            db.execute("SELECT recurring FROM tasks LIMIT 1")
        except Exception:
            try:
                db.execute("ALTER TABLE tasks ADD COLUMN recurring TEXT")
                db.execute("ALTER TABLE tasks ADD COLUMN last_recurring_at TEXT")
            except Exception:
                pass
        db.commit()
        if close:
            db.close()

    def _now(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def create_task(
        self,
        title: str,
        description: str | None = None,
        deadline: str | None = None,
        research_query: str | None = None,
    ) -> int:
        """Create a new task. Deadline is parsed from natural language.

        Returns the integer task ID.
        """
        # Parse deadline if it looks like natural language
        parsed_deadline: str | None = None
        if deadline:
            parsed_deadline = _parse_deadline(deadline)
            if not parsed_deadline:
                # Maybe it's already ISO
                try:
                    datetime.fromisoformat(deadline.replace("Z", "+00:00"))
                    parsed_deadline = deadline
                except ValueError:
                    parsed_deadline = None

        # Auto-generate research query from title
        if not research_query:
            # Remove common filler words
            stop_words = {
                "подготовить", "сделать", "написать", "создать", "найти",
                "prepare", "make", "write", "create", "find", "do", "get",
                "по", "для", "the", "a", "an", "to", "and", "or", "of",
                "в", "на", "с", "и", "к", "от", "из",
            }
            words = [w for w in title.split() if w.lower() not in stop_words and len(w) > 2]
            research_query = " ".join(words) if words else title

        now = self._now()
        db = self._get_db()
        try:
            cursor = db.execute(
                """INSERT INTO tasks
                   (title, description, deadline, status, research_query,
                    created_at, updated_at)
                   VALUES (?, ?, ?, 'pending', ?, ?, ?)""",
                (title, description, parsed_deadline, research_query, now, now),
            )
            db.commit()
            task_id = cursor.lastrowid
            assert task_id is not None
            return task_id
        finally:
            db.close()

    def list_tasks(self, status: str | None = None) -> list[dict[str, Any]]:
        """List tasks, optionally filtered by status."""
        db = self._get_db()
        try:
            if status:
                rows = db.execute(
                    "SELECT * FROM tasks WHERE status = ? ORDER BY deadline ASC NULLS LAST, id ASC",
                    (status,),
                ).fetchall()
            else:
                rows = db.execute(
                    "SELECT * FROM tasks WHERE status NOT IN ('done') "
                    "ORDER BY deadline ASC NULLS LAST, id ASC"
                ).fetchall()
            return [dict(r) for r in rows]
        finally:
            db.close()

    def get_task(self, task_id: int) -> dict[str, Any] | None:
        """Get a single task by ID."""
        db = self._get_db()
        try:
            row = db.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
            return dict(row) if row else None
        finally:
            db.close()

    def update_status(self, task_id: int, status: str) -> bool:
        """Update task status. Returns True if task was found."""
        db = self._get_db()
        try:
            cursor = db.execute(
                "UPDATE tasks SET status = ?, updated_at = ? WHERE id = ?",
                (status, self._now(), task_id),
            )
            db.commit()
            return cursor.rowcount > 0
        finally:
            db.close()

    def save_research(self, task_id: int, results: list[dict[str, Any]]) -> bool:
        """Save research results as JSON."""
        db = self._get_db()
        try:
            cursor = db.execute(
                "UPDATE tasks SET research_results = ?, updated_at = ? WHERE id = ?",
                (json.dumps(results, ensure_ascii=False), self._now(), task_id),
            )
            db.commit()
            return cursor.rowcount > 0
        finally:
            db.close()

    def save_briefing(self, task_id: int, briefing: str) -> bool:
        """Save the final prepared briefing."""
        db = self._get_db()
        try:
            cursor = db.execute(
                "UPDATE tasks SET briefing = ?, status = 'ready', updated_at = ? WHERE id = ?",
                (briefing, self._now(), task_id),
            )
            db.commit()
            return cursor.rowcount > 0
        finally:
            db.close()

    def get_upcoming(self, days: int = 3) -> list[dict[str, Any]]:
        """Get tasks with deadline within the next N days (not done)."""
        cutoff = (datetime.now(timezone.utc) + timedelta(days=days)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        now = self._now()
        db = self._get_db()
        try:
            rows = db.execute(
                "SELECT * FROM tasks "
                "WHERE deadline IS NOT NULL AND deadline <= ? AND deadline >= ? "
                "AND status NOT IN ('done') "
                "ORDER BY deadline ASC",
                (cutoff, now),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            db.close()

    def get_overdue(self) -> list[dict[str, Any]]:
        """Get tasks past deadline that are not done."""
        now = self._now()
        db = self._get_db()
        try:
            rows = db.execute(
                "SELECT * FROM tasks "
                "WHERE deadline IS NOT NULL AND deadline < ? "
                "AND status NOT IN ('done', 'overdue') "
                "ORDER BY deadline ASC",
                (now,),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            db.close()

    def delete_task(self, task_id: int) -> bool:
        """Delete a task by ID. Returns True if task was found."""
        db = self._get_db()
        try:
            cursor = db.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
            db.commit()
            return cursor.rowcount > 0
        finally:
            db.close()

    def update_notified(self, task_id: int) -> None:
        """Mark task as notified (update notified_at)."""
        db = self._get_db()
        try:
            db.execute(
                "UPDATE tasks SET notified_at = ? WHERE id = ?",
                (self._now(), task_id),
            )
            db.commit()
        finally:
            db.close()

    # ------------------------------------------------------------------
    # Research & Briefing pipeline
    # ------------------------------------------------------------------

    def research_task(self, task_id: int) -> list[dict[str, str]]:
        """Run web search + memory search for a task. Returns collected results."""
        task = self.get_task(task_id)
        if not task:
            return []

        query = task.get("research_query") or task["title"]
        collected: list[dict[str, str]] = []

        # Phase 1: Web search
        search_results = web_search(query, max_results=5)
        for sr in search_results[:5]:
            url = sr.get("url", "")
            title = sr.get("title", "")
            snippet = sr.get("snippet", "")

            # Fetch full content for top 3
            content = ""
            if len(collected) < 3 and url:
                content = fetch_and_summarize(url, max_chars=2000)

            collected.append({
                "source": "web",
                "title": title,
                "url": url,
                "snippet": snippet,
                "content": content[:2000] if content else snippet,
            })

        # Phase 2: Memory search
        db = self._get_db()
        try:
            # Search knowledge base for related content
            keywords = query.split()[:5]
            like_clauses = " OR ".join(["content LIKE ?"] * len(keywords))
            params = [f"%{kw}%" for kw in keywords]

            rows = db.execute(
                f"SELECT content, type, project, source FROM knowledge "
                f"WHERE status='active' AND ({like_clauses}) "
                f"ORDER BY created_at DESC LIMIT 5",
                params,
            ).fetchall()

            for row in rows:
                collected.append({
                    "source": "memory",
                    "title": f"[{row['type']}] {row['project'] or 'general'}",
                    "url": "",
                    "snippet": "",
                    "content": row["content"][:1500],
                })
        except Exception:
            pass
        finally:
            db.close()

        # Save research results
        self.save_research(task_id, collected)
        self.update_status(task_id, "researching")

        return collected

    def generate_briefing(self, task_id: int) -> str:
        """Generate a briefing from research results using Ollama."""
        task = self.get_task(task_id)
        if not task:
            return ""

        research_raw = task.get("research_results", "")
        if not research_raw:
            return ""

        try:
            research_items: list[dict[str, str]] = json.loads(research_raw)
        except (json.JSONDecodeError, TypeError):
            return ""

        # Build context from research
        context_parts: list[str] = []
        for i, item in enumerate(research_items[:8], 1):
            source = item.get("source", "?")
            title = item.get("title", "")
            content = item.get("content", item.get("snippet", ""))[:800]
            context_parts.append(f"[{i}] ({source}) {title}\n{content}")

        context = "\n\n".join(context_parts)

        prompt = (
            f"You are a personal secretary preparing a briefing.\n\n"
            f"Task: {task['title']}\n"
            f"{'Description: ' + task['description'] if task.get('description') else ''}\n"
            f"Deadline: {task.get('deadline', 'no deadline')}\n\n"
            f"Research collected:\n{context}\n\n"
            f"Create a comprehensive briefing with:\n"
            f"1. Key findings summary (3-5 bullet points)\n"
            f"2. Recommended action plan (step by step)\n"
            f"3. Important details to consider\n"
            f"4. Potential risks or things to watch out for\n\n"
            f"Be specific and actionable. Write in the same language as the task title. "
            f"Max 500 words."
        )

        briefing = _ollama_generate(prompt, timeout=180)
        if briefing:
            self.save_briefing(task_id, briefing)

        return briefing


# ---------------------------------------------------------------------------
# Formatting helpers for Telegram
# ---------------------------------------------------------------------------

def format_task_list(tasks: list[dict[str, Any]]) -> str:
    """Format a list of tasks for Telegram display (HTML)."""
    if not tasks:
        return "No tasks found."

    lines: list[str] = []
    status_emoji = {
        "pending": "\u23f3",     # hourglass
        "researching": "\U0001f50d",  # magnifying glass
        "ready": "\u2705",       # green check
        "done": "\u2714\ufe0f",  # check mark
        "overdue": "\u26a0\ufe0f",   # warning
    }

    for task in tasks:
        emoji = status_emoji.get(task["status"], "\u2753")
        deadline_str = ""
        if task.get("deadline"):
            try:
                dt = datetime.fromisoformat(task["deadline"].replace("Z", "+00:00"))
                deadline_str = f" | {dt.strftime('%b %d')}"
            except ValueError:
                deadline_str = f" | {task['deadline'][:10]}"

        title = task["title"][:60]
        lines.append(f"{emoji} <b>#{task['id']}</b> {title}{deadline_str} [{task['status']}]")

    return "\n".join(lines)


def format_task_detail(task: dict[str, Any]) -> str:
    """Format a single task detail for Telegram (HTML)."""
    lines: list[str] = []
    lines.append(f"<b>Task #{task['id']}: {task['title']}</b>")
    lines.append(f"Status: {task['status']}")

    if task.get("description"):
        lines.append(f"Description: {task['description']}")

    if task.get("deadline"):
        try:
            dt = datetime.fromisoformat(task["deadline"].replace("Z", "+00:00"))
            lines.append(f"Deadline: {dt.strftime('%Y-%m-%d %H:%M UTC')}")
            remaining = dt - datetime.now(timezone.utc)
            if remaining.total_seconds() > 0:
                days = remaining.days
                hours = remaining.seconds // 3600
                lines.append(f"Time left: {days}d {hours}h")
            else:
                lines.append("OVERDUE!")
        except ValueError:
            lines.append(f"Deadline: {task['deadline']}")

    if task.get("research_query"):
        lines.append(f"Research: {task['research_query'][:100]}")

    lines.append(f"Created: {task.get('created_at', '?')[:16]}")

    if task.get("briefing"):
        lines.append(f"\n<b>Briefing:</b>\n{task['briefing'][:3500]}")
    elif task.get("research_results"):
        try:
            results = json.loads(task["research_results"])
            lines.append(f"\nResearch: {len(results)} sources collected (briefing pending)")
        except (json.JSONDecodeError, TypeError):
            pass

    if task.get("recurring"):
        lines.append(f"\n🔄 Recurring: {task['recurring']}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Recurring task support
# ---------------------------------------------------------------------------


def get_recurring_tasks(db_path: str) -> list[dict[str, Any]]:
    """Get tasks marked as recurring that need re-processing today."""
    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row
    try:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        rows = db.execute(
            """SELECT * FROM tasks
               WHERE recurring IS NOT NULL
                 AND recurring != ''
                 AND status NOT IN ('done')
                 AND (last_recurring_at IS NULL OR last_recurring_at < ?)
               ORDER BY id ASC""",
            (today,),
        ).fetchall()
        return [dict(r) for r in rows]
    except Exception:
        return []
    finally:
        db.close()


def mark_recurring_done(db_path: str, task_id: int) -> None:
    """Mark a recurring task as processed for today — reset for next run."""
    db = sqlite3.connect(db_path)
    try:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        db.execute(
            "UPDATE tasks SET last_recurring_at = ?, status = 'pending', "
            "research_results = NULL, briefing = NULL, updated_at = ? WHERE id = ?",
            (today, datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), task_id),
        )
        db.commit()
    finally:
        db.close()
