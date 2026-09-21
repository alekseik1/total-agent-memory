#!/usr/bin/env python3
"""
Brain Autonomy — Vito decides what to learn and do next.

Self-tasking: analyzes knowledge gaps, trending topics, user interests,
and creates tasks for itself. All actions logged to Telegram.

Runs every 6 hours via scheduler.

Usage:
    from tools.brain_autonomy import run_brain_autonomy
    run_brain_autonomy("/path/to/memory.db")
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import ssl
import subprocess
import sys
import time
import uuid
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

MEMORY_DIR = Path(os.environ.get("CLAUDE_MEMORY_DIR", os.path.expanduser("~/.claude-memory")))
OLLAMA_BIN = "/usr/local/bin/ollama"
TELEGRAM_ENV_PATH = Path(__file__).parent.parent.parent / "telegram" / ".env"
LOG = lambda msg: sys.stderr.write(f"[brain-autonomy] {datetime.now().strftime('%H:%M:%S')} {msg}\n")

# Max tasks brain can create per run
MAX_SELF_TASKS_PER_RUN = 3
# Max research actions per run
MAX_RESEARCH_PER_RUN = 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences."""
    return re.sub(r"\x1b\[[0-9;]*[A-Za-z]|\x1b\].*?\x07", "", text)


def _ollama_generate(prompt: str, model: str = "vitalii-brain", timeout: int = 180) -> str:
    """Generate text with Ollama. Returns clean text or empty string."""
    try:
        env = os.environ.copy()
        env["NO_COLOR"] = "1"
        result = subprocess.run(
            [OLLAMA_BIN, "run", model, prompt],
            capture_output=True, text=True, timeout=timeout, env=env,
        )
        return _strip_ansi(result.stdout).strip()
    except subprocess.TimeoutExpired:
        LOG(f"Ollama timeout after {timeout}s")
        return ""
    except Exception as e:
        LOG(f"Ollama error: {e}")
        return ""


def _make_ssl_context():
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _load_dotenv(path: Path) -> dict[str, str]:
    """Load .env file into dict."""
    env = {}
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def _telegram_send(token: str, chat_id: int, text: str, parse_mode: str = "HTML") -> bool:
    """Send Telegram message."""
    text = _strip_ansi(text)
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = json.dumps({
        "chat_id": chat_id,
        "text": text[:4096],
        "parse_mode": parse_mode,
    }).encode()
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    try:
        ctx = _make_ssl_context()
        with urllib.request.urlopen(req, timeout=30, context=ctx) as resp:
            return json.loads(resp.read()).get("ok", False)
    except Exception as e:
        LOG(f"Telegram send error: {e}")
        return False


def _notify_all(message: str) -> int:
    """Send message to all allowed Telegram users."""
    env = _load_dotenv(TELEGRAM_ENV_PATH)
    token = env.get("TELEGRAM_BOT_TOKEN", "")
    allowed = env.get("TELEGRAM_ALLOWED_USERS", "")
    if not token or not allowed:
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


def _log_action(action: str, details: str, status: str = "ok") -> None:
    """Log brain action to Telegram with structured format."""
    emoji = {"ok": "\u2705", "start": "\U0001f9e0", "error": "\u274c", "idea": "\U0001f4a1",
             "task": "\U0001f4cb", "research": "\U0001f50d", "learn": "\U0001f4da"}.get(status, "\u2139\ufe0f")
    ts = datetime.now().strftime("%H:%M")
    msg = f"{emoji} <b>[Vito {ts}]</b> {action}\n<i>{details[:500]}</i>"
    _notify_all(msg)
    LOG(f"[{status}] {action}: {details[:100]}")


# ---------------------------------------------------------------------------
# Knowledge Gap Analysis
# ---------------------------------------------------------------------------

def _find_knowledge_gaps(db: sqlite3.Connection) -> list[dict[str, str]]:
    """Analyze knowledge base and find gaps worth filling."""
    gaps: list[dict[str, str]] = []

    # 1. Topics mentioned but never researched
    try:
        rows = db.execute("""
            SELECT content, project FROM knowledge
            WHERE status = 'active'
              AND created_at > strftime('%Y-%m-%dT%H:%M:%f000Z', 'now', '-7 days')
              AND (LOWER(content) LIKE '%todo%'
                   OR LOWER(content) LIKE '%изучить%'
                   OR LOWER(content) LIKE '%разобраться%'
                   OR LOWER(content) LIKE '%need to learn%'
                   OR LOWER(content) LIKE '%should investigate%'
                   OR LOWER(content) LIKE '%look into%')
            ORDER BY created_at DESC LIMIT 10
        """).fetchall()

        for row in rows:
            content = row["content"]
            for line in content.split("\n"):
                line = line.strip()
                if any(kw in line.lower() for kw in ["todo", "изучить", "разобраться", "need to learn", "look into"]):
                    if 10 < len(line) < 200:
                        gaps.append({
                            "topic": line,
                            "reason": "mentioned but not researched",
                            "project": row["project"] or "general",
                            "priority": "medium",
                        })
                        break
    except Exception as e:
        LOG(f"Gap analysis (todos): {e}")

    # 2. Technologies used but poorly documented
    try:
        tech_mentions = db.execute("""
            SELECT tags, COUNT(*) as cnt FROM knowledge
            WHERE status = 'active' AND tags != '[]'
            GROUP BY tags
            HAVING cnt < 3
            ORDER BY cnt ASC LIMIT 10
        """).fetchall()

        for row in tech_mentions:
            try:
                tags = json.loads(row["tags"])
                for tag in tags:
                    if len(tag) > 3 and tag not in ("reusable", "session-autosave", "context-recovery"):
                        # Check if we have enough knowledge about this tag
                        count = db.execute(
                            "SELECT COUNT(*) FROM knowledge WHERE status='active' AND tags LIKE ?",
                            (f'%"{tag}"%',)
                        ).fetchone()[0]
                        if count < 3:
                            gaps.append({
                                "topic": f"Deep dive: {tag}",
                                "reason": f"only {count} records with tag '{tag}'",
                                "project": "general",
                                "priority": "low",
                            })
            except (json.JSONDecodeError, TypeError):
                pass
    except Exception as e:
        LOG(f"Gap analysis (tech): {e}")

    # 3. Questions from recent sessions without answers
    try:
        questions = db.execute("""
            SELECT content, project FROM knowledge
            WHERE status = 'active'
              AND type = 'fact'
              AND created_at > strftime('%Y-%m-%dT%H:%M:%f000Z', 'now', '-3 days')
              AND content LIKE '%?%'
            ORDER BY created_at DESC LIMIT 5
        """).fetchall()

        for row in questions:
            for line in row["content"].split("\n"):
                if "?" in line and 15 < len(line.strip()) < 200:
                    gaps.append({
                        "topic": line.strip(),
                        "reason": "unanswered question",
                        "project": row["project"] or "general",
                        "priority": "high",
                    })
                    break
    except Exception as e:
        LOG(f"Gap analysis (questions): {e}")

    # Source 4: Active blind spots from self-model
    try:
        blind_spots = db.execute(
            "SELECT id, description, domains, severity FROM blind_spots WHERE status = 'active' ORDER BY severity DESC"
        ).fetchall()
        for bs in blind_spots:
            gaps.append({
                "topic": f"Learn about: {bs['description']}",
                "reason": f"blind spot (severity {bs['severity']})",
                "project": "general",
                "priority": "high" if bs['severity'] > 0.5 else "medium",
            })
    except Exception as e:
        LOG(f"Gap analysis (blind_spots): {e}")

    # Deduplicate
    seen = set()
    unique = []
    for g in gaps:
        key = g["topic"][:50].lower()
        if key not in seen:
            seen.add(key)
            unique.append(g)

    return unique[:10]


# ---------------------------------------------------------------------------
# Interest Discovery — what should Vito learn next?
# ---------------------------------------------------------------------------

def _discover_interests(db: sqlite3.Connection) -> list[dict[str, str]]:
    """Use Ollama to analyze knowledge and suggest what to learn next."""
    # Get recent knowledge summary
    try:
        rows = db.execute("""
            SELECT type, project, content FROM knowledge
            WHERE status = 'active'
            ORDER BY created_at DESC LIMIT 20
        """).fetchall()
    except Exception:
        return []

    if not rows:
        return []

    summary = "\n".join(
        f"- [{r['type']}] {r['project']}: {r['content'][:100]}"
        for r in rows
    )

    # Collect active blind spots for additional context
    blind_spots_ctx = ""
    try:
        bs_rows = db.execute(
            "SELECT description, domains, severity FROM blind_spots "
            "WHERE status = 'active' ORDER BY severity DESC LIMIT 10"
        ).fetchall()
        if bs_rows:
            bs_lines = "\n".join(
                f"- {bs['description']} (domains: {bs['domains']}, severity: {bs['severity']})"
                for bs in bs_rows
            )
            blind_spots_ctx = f"\n\nKnown blind spots (areas with weak knowledge):\n{bs_lines}\n"
    except Exception:
        pass  # Table may not exist yet

    prompt = (
        "Based on this knowledge base activity, suggest 2-3 specific topics "
        "that would be valuable to research next. Consider:\n"
        "- What's trending in these technology areas\n"
        "- What knowledge gaps exist\n"
        "- What could help the user be more productive\n"
        "- Known blind spots that need attention\n\n"
        f"Recent knowledge:\n{summary[:3000]}\n"
        f"{blind_spots_ctx}\n"
        "Return ONLY a JSON array of objects with 'topic' and 'reason' fields. "
        "Each topic should be specific and actionable (not generic).\n"
        "Example: [{\"topic\": \"PostgreSQL 18 new features\", \"reason\": \"user works with PG, v18 just released\"}]"
    )

    # Use Claude for interest discovery (better reasoning), fallback Ollama
    try:
        from tools.llm_router import generate_cheap
        response = generate_cheap(prompt)
    except ImportError:
        response = ""
    if not response:
        response = _ollama_generate(prompt, timeout=120)
    if not response:
        return []

    # Parse JSON from response
    try:
        # Try to find JSON array in response
        match = re.search(r'\[.*\]', response, re.DOTALL)
        if match:
            interests = json.loads(match.group())
            return [
                {"topic": i["topic"], "reason": i.get("reason", ""), "project": "general", "priority": "medium"}
                for i in interests
                if isinstance(i, dict) and "topic" in i
            ][:3]
    except (json.JSONDecodeError, KeyError):
        LOG(f"Failed to parse interests JSON: {response[:200]}")

    return []


# ---------------------------------------------------------------------------
# Self-Tasking — create tasks from gaps and interests
# ---------------------------------------------------------------------------

def _create_self_task(db_path: str, title: str, reason: str, project: str = "general") -> int | None:
    """Create a task that the brain assigned to itself."""
    db = sqlite3.connect(db_path)
    try:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        deadline = (datetime.now(timezone.utc) + timedelta(days=2)).strftime("%Y-%m-%dT%H:%M:%SZ")

        # Check if similar task already exists
        existing = db.execute(
            "SELECT id FROM tasks WHERE status IN ('pending', 'researching') "
            "AND LOWER(title) LIKE ? LIMIT 1",
            (f"%{title[:30].lower()}%",)
        ).fetchone()

        if existing:
            LOG(f"Similar task already exists: #{existing[0]}")
            return None

        cursor = db.execute(
            "INSERT INTO tasks (title, description, status, deadline, project, "
            "created_at, source, tags) VALUES (?, ?, 'pending', ?, ?, ?, 'brain-autonomy', ?)",
            (
                f"[Auto] {title[:200]}",
                f"Self-assigned by Vito brain.\nReason: {reason}",
                deadline,
                project,
                now,
                json.dumps(["self-task", "brain-autonomy", project]),
            )
        )
        db.commit()
        task_id = cursor.lastrowid
        LOG(f"Self-task created: #{task_id} — {title[:60]}")
        return task_id
    except Exception as e:
        LOG(f"Create self-task error: {e}")
        return None
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Web Research (lightweight)
# ---------------------------------------------------------------------------

def _web_search(query: str, max_results: int = 5) -> list[dict[str, str]]:
    """Search DuckDuckGo HTML. Returns list of {title, url}."""
    try:
        q = urllib.parse.quote_plus(query)
        url = f"https://html.duckduckgo.com/html/?q={q}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        ctx = _make_ssl_context()
        with urllib.request.urlopen(req, timeout=15, context=ctx) as resp:
            html = resp.read().decode("utf-8", errors="ignore")

        results = []
        links = re.findall(r'<a[^>]+class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', html)
        for href, title in links[:max_results]:
            clean_title = re.sub(r"<[^>]+>", "", title).strip()
            if href.startswith("//duckduckgo.com/l/"):
                match = re.search(r"uddg=([^&]+)", href)
                if match:
                    href = urllib.parse.unquote(match.group(1))
            if clean_title and href.startswith("http"):
                results.append({"title": clean_title, "url": href})
        return results
    except Exception as e:
        LOG(f"Web search error: {e}")
        return []


def _fetch_url(url: str, max_chars: int = 3000) -> str:
    """Fetch URL content as plain text."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        ctx = _make_ssl_context()
        with urllib.request.urlopen(req, timeout=15, context=ctx) as resp:
            html = resp.read().decode("utf-8", errors="ignore")[:50000]
        text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:max_chars]
    except Exception as e:
        LOG(f"Fetch error ({url[:60]}): {e}")
        return ""


def _research_topic(topic: str) -> str | None:
    """Research a topic: web search → fetch → summarize with Ollama."""
    _log_action("Research", topic, "research")

    results = _web_search(topic, max_results=5)
    if not results:
        return None

    combined = ""
    sources: list[str] = []
    for r in results[:3]:
        text = _fetch_url(r["url"])
        if text and len(text) > 100:
            combined += f"\n\n--- {r['title']} ({r['url']}) ---\n{text}"
            sources.append(f'<a href="{r["url"]}">{r["title"][:60]}</a>')
        time.sleep(2)

    if not combined:
        return None

    # Use Claude API for summarization (reliable), fallback Ollama
    try:
        from tools.llm_router import research_and_summarize
        summary = research_and_summarize(topic, combined[:8000])
    except ImportError:
        summary = ""
    if not summary:
        prompt = (
            f"Summarize this research concisely for a senior developer.\n"
            f"Topic: {topic}\n\n"
            f"Sources:\n{combined[:8000]}\n\n"
            f"Write a clear summary (max 200 words). Include key facts and actionable advice."
        )
        summary = _ollama_generate(prompt, timeout=180)
    if not summary:
        return None

    # Log to Telegram with links
    source_list = "\n".join(sources[:3]) if sources else "No sources"
    _log_action(
        f"Researched: {topic[:60]}",
        f"{summary[:300]}...\n\n<b>Sources:</b>\n{source_list}",
        "learn"
    )
    return summary


def _save_research(db_path: str, topic: str, summary: str, project: str = "general") -> None:
    """Save research result to knowledge base."""
    db = sqlite3.connect(db_path)
    try:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        db.execute(
            "INSERT INTO knowledge (session_id, type, content, project, tags, source, "
            "confidence, created_at, status) VALUES (?, 'fact', ?, ?, ?, 'brain-autonomy', 0.5, ?, 'active')",
            (
                f"brain_{now}_{uuid.uuid4().hex[:6]}",
                f"[Brain Research] {topic}\n\n{summary}",
                project,
                json.dumps(["brain-autonomy", "self-research", project]),
                now,
            )
        )
        db.commit()
    except Exception as e:
        LOG(f"Save research error: {e}")
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Main: Brain Autonomy Loop
# ---------------------------------------------------------------------------

def run_brain_autonomy(db_path: str) -> None:
    """Main autonomous brain cycle.

    1. Analyze knowledge gaps
    2. Ask Ollama what to learn next
    3. Create self-tasks
    4. Execute immediate research (max 2 topics)
    5. Log everything to Telegram
    """
    LOG("=== BRAIN AUTONOMY ENGINE STARTING ===")
    _log_action("Brain Autonomy", "Starting autonomous cycle", "start")

    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row

    tasks_created = 0
    research_done = 0

    try:
        # Phase 1: Find knowledge gaps
        LOG("[autonomy] Phase 1: Knowledge gap analysis...")
        gaps = _find_knowledge_gaps(db)
        LOG(f"[autonomy] Found {len(gaps)} knowledge gaps")

        # Phase 2: Ask Vito what interests it
        LOG("[autonomy] Phase 2: Interest discovery...")
        interests = _discover_interests(db)
        LOG(f"[autonomy] Vito suggested {len(interests)} topics")

        # Merge and prioritize
        all_topics = gaps + interests
        # Sort: high priority first, then medium, then low
        priority_order = {"high": 0, "medium": 1, "low": 2}
        all_topics.sort(key=lambda x: priority_order.get(x.get("priority", "low"), 2))

        if not all_topics:
            _log_action("No gaps found", "Knowledge base is well-covered. Resting.", "ok")
            LOG("=== BRAIN AUTONOMY: nothing to do ===")
            return

        # Phase 3: Create self-tasks
        LOG("[autonomy] Phase 3: Creating self-tasks...")
        for topic_info in all_topics[:MAX_SELF_TASKS_PER_RUN]:
            topic = topic_info["topic"]
            reason = topic_info.get("reason", "knowledge gap")
            project = topic_info.get("project", "general")

            task_id = _create_self_task(db_path, topic, reason, project)
            if task_id:
                tasks_created += 1
                _log_action(
                    f"Self-task #{task_id}",
                    f"{topic[:100]}\nReason: {reason}",
                    "task"
                )

        # Phase 4: Immediate research on top priorities
        LOG("[autonomy] Phase 4: Immediate research...")
        for topic_info in all_topics[:MAX_RESEARCH_PER_RUN]:
            if topic_info.get("priority") in ("high", "medium"):
                topic = topic_info["topic"]
                project = topic_info.get("project", "general")

                # Clean topic for search
                search_q = re.sub(r"[\[\]#*\-\u2022]", "", topic).strip()
                search_q = re.sub(
                    r"\b(todo|to-do|need to research|изучить|разобраться|look into)\b",
                    "", search_q, flags=re.IGNORECASE
                ).strip()

                if len(search_q) < 5:
                    continue

                summary = _research_topic(search_q[:100])
                if summary:
                    _save_research(db_path, topic, summary, project)
                    research_done += 1

                time.sleep(3)  # Rate limit

    except Exception as e:
        LOG(f"Brain autonomy error: {e}")
        _log_action("Error", str(e)[:200], "error")
    finally:
        db.close()

    # Final summary
    summary_msg = (
        f"Autonomous cycle complete:\n"
        f"• {len(gaps)} gaps found\n"
        f"• {len(interests)} interests suggested\n"
        f"• {tasks_created} self-tasks created\n"
        f"• {research_done} topics researched"
    )
    _log_action("Cycle Complete", summary_msg, "ok")
    LOG(f"=== BRAIN AUTONOMY COMPLETE === ({tasks_created} tasks, {research_done} research)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Brain Autonomy Engine")
    parser.add_argument("--db", default=str(MEMORY_DIR / "memory.db"))
    args = parser.parse_args()
    run_brain_autonomy(args.db)
