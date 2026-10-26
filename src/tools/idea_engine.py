#!/usr/bin/env python3
"""
Idea Generation and Innovation Engine -- autonomous creative module.

Generates new ideas, challenges assumptions, proposes improvements
based on the user's knowledge base, projects, and current tech trends.

Schedules:
- Idea Generator: Friday 6:00 PM
- Innovation Digest: 1st and 15th of month at 10:00 AM
- Blind Spot Challenger: Monday 9:00 AM

Usage (standalone):
    python src/tools/idea_engine.py --db ~/.claude-memory/memory.db --run ideas
    python src/tools/idea_engine.py --db ~/.claude-memory/memory.db --run innovation
    python src/tools/idea_engine.py --db ~/.claude-memory/memory.db --run challenge
"""

import json
import re
import sqlite3
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

LOG = lambda msg: sys.stderr.write(
    f"[idea-engine] {datetime.now().strftime('%H:%M:%S')} {msg}\n"
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


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from text (terminal colors, cursor moves, etc.)."""
    return re.sub(r"\x1b\[[0-9;]*[A-Za-z]|\x1b\].*?\x07", "", text)


def _ollama_generate(prompt: str, model: str | None = None, timeout: int = 300) -> str:
    """Generate text with the Ollama CLI (prompt on stdin). Returns "" on failure."""
    return ollama_generate(prompt, model, timeout, log=LOG)


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


def _telegram_broadcast(message: str) -> int:
    """Send message to all configured Telegram users. Returns count of sends."""
    env = _load_dotenv(TELEGRAM_ENV_PATH)
    token = env.get("TELEGRAM_BOT_TOKEN", "")
    allowed = env.get("TELEGRAM_ALLOWED_USERS", "")
    if not token or not allowed:
        LOG("Telegram credentials not configured, skipping broadcast")
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
# Knowledge Base Analysis Helpers
# ---------------------------------------------------------------------------

# Tags to exclude from technology extraction -- these are internal/meta tags
_SKIP_TAGS = frozenset({
    "auto", "session-autosave", "context-recovery", "reusable",
    "rss", "news", "github", "release", "dream", "auto-insight",
    "cross-project", "curiosity", "auto-research", "advisor",
    "daily-tip", "idea-generator", "innovation", "weekly",
    "blind-spot-challenge", "growth", "innovation-digest",
    "opportunities", "error_pattern", "blind_spot",
    "uncertain_knowledge", "session-end", "session-start",
})


def _extract_user_profile(db_path: str) -> dict[str, Any]:
    """Analyze knowledge base to build a profile of user expertise.

    Returns dict with:
        technologies: list of (tech, count) -- most used technologies
        projects: list of (project, count) -- most active projects
        solutions: list of str -- recent solution summaries
        patterns: list of str -- recurring architectural patterns
        total_records: int
    """
    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row
    profile: dict[str, Any] = {
        "technologies": [],
        "projects": [],
        "solutions": [],
        "patterns": [],
        "total_records": 0,
    }

    try:
        # Total knowledge count
        row = db.execute(
            "SELECT COUNT(*) as cnt FROM knowledge WHERE status = 'active'"
        ).fetchone()
        profile["total_records"] = row["cnt"] if row else 0

        # Most active projects (by record count, last 90 days)
        cutoff_90d = (datetime.now(timezone.utc) - timedelta(days=90)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        projects = db.execute(
            """SELECT project, COUNT(*) as cnt FROM knowledge
               WHERE status = 'active' AND project IS NOT NULL
                 AND project != 'general' AND created_at > ?
               GROUP BY project ORDER BY cnt DESC LIMIT 10""",
            (cutoff_90d,),
        ).fetchall()
        profile["projects"] = [(r["project"], r["cnt"]) for r in projects]

        # Technology tags (from all active records)
        all_tags = db.execute(
            """SELECT tags FROM knowledge
               WHERE status = 'active' AND tags IS NOT NULL AND tags != '[]'"""
        ).fetchall()
        tag_counter: Counter = Counter()
        for row in all_tags:
            try:
                tags = json.loads(row["tags"]) if row["tags"] else []
                for tag in tags:
                    if tag and tag.lower() not in _SKIP_TAGS and len(tag) > 1:
                        tag_counter[tag.lower()] += 1
            except (json.JSONDecodeError, TypeError):
                pass
        profile["technologies"] = tag_counter.most_common(20)

        # Recent solutions (last 60 days)
        cutoff_60d = (datetime.now(timezone.utc) - timedelta(days=60)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        solutions = db.execute(
            """SELECT content, project FROM knowledge
               WHERE status = 'active' AND type = 'solution'
                 AND created_at > ?
               ORDER BY created_at DESC LIMIT 15""",
            (cutoff_60d,),
        ).fetchall()
        for sol in solutions:
            summary = sol["content"][:150].replace("\n", " ").strip()
            proj = sol["project"] or "general"
            profile["solutions"].append(f"[{proj}] {summary}")

        # Architectural patterns from decisions
        patterns = db.execute(
            """SELECT content FROM knowledge
               WHERE status = 'active' AND type = 'decision'
                 AND created_at > ?
               ORDER BY created_at DESC LIMIT 10""",
            (cutoff_90d,),
        ).fetchall()
        for p in patterns:
            summary = p["content"][:120].replace("\n", " ").strip()
            profile["patterns"].append(summary)

    except Exception as e:
        LOG(f"Error extracting user profile: {e}")
    finally:
        db.close()

    return profile


def _get_recent_trends(db_path: str) -> list[str]:
    """Collect recent trends from RSS articles, GitHub releases, and curiosity research.

    Returns list of trend summary strings.
    """
    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row
    trends: list[str] = []

    try:
        cutoff_14d = (datetime.now(timezone.utc) - timedelta(days=14)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )

        # RSS articles (recent news)
        rss = db.execute(
            """SELECT content FROM knowledge
               WHERE source = 'rss' AND status = 'active' AND created_at > ?
               ORDER BY created_at DESC LIMIT 10""",
            (cutoff_14d,),
        ).fetchall()
        for row in rss:
            trends.append(f"[News] {row['content'][:120].strip()}")

        # GitHub releases
        gh = db.execute(
            """SELECT content FROM knowledge
               WHERE source = 'github' AND status = 'active' AND created_at > ?
               ORDER BY created_at DESC LIMIT 10""",
            (cutoff_14d,),
        ).fetchall()
        for row in gh:
            trends.append(f"[Release] {row['content'][:120].strip()}")

        # Curiosity research
        curiosity = db.execute(
            """SELECT content FROM knowledge
               WHERE source = 'curiosity-engine' AND status = 'active'
                 AND created_at > ?
               ORDER BY created_at DESC LIMIT 5""",
            (cutoff_14d,),
        ).fetchall()
        for row in curiosity:
            trends.append(f"[Research] {row['content'][:120].strip()}")

    except Exception as e:
        LOG(f"Error fetching trends: {e}")
    finally:
        db.close()

    return trends


# ---------------------------------------------------------------------------
# 1. Idea Generator -- weekly, Friday 6:00 PM
# ---------------------------------------------------------------------------


def run_idea_generator(db_path: str) -> None:
    """Generate creative project/tool ideas based on user expertise and trends.

    Analyzes the knowledge base to understand expertise, cross-references with
    recent tech trends, and uses Ollama to generate actionable ideas.
    Saves each idea as type='decision' and sends a Telegram digest.
    """
    LOG("=== IDEA GENERATOR STARTING ===")
    start_time = time.time()

    # Step 1: Build user profile from knowledge base
    profile = _extract_user_profile(db_path)
    if profile["total_records"] < 10:
        LOG("Not enough knowledge records for idea generation (need 10+)")
        return

    tech_list = ", ".join(t for t, _ in profile["technologies"][:12])
    projects_list = ", ".join(f"{p}({c})" for p, c in profile["projects"][:8])
    solutions_summary = "\n".join(f"- {s}" for s in profile["solutions"][:8])
    patterns_summary = "\n".join(f"- {p}" for p in profile["patterns"][:5])

    # Step 2: Get recent trends
    trends = _get_recent_trends(db_path)
    trends_text = (
        "\n".join(f"- {t}" for t in trends[:10])
        if trends
        else "No recent trend data available."
    )

    # Step 3: Generate "What If" scenarios for extra context
    what_ifs = _generate_what_ifs(db_path)
    what_if_text = ""
    if what_ifs:
        what_if_text = "\n\nCross-pollination scenarios:\n" + "\n".join(
            f"- {wf['scenario']}" for wf in what_ifs[:3]
        )

    # Step 4: Ask Ollama for ideas
    prompt = f"""Based on this developer's expertise:
- Technologies: {tech_list}
- Active projects: {projects_list}
- Recent solutions:
{solutions_summary}
- Architectural patterns:
{patterns_summary}
- Current trends:
{trends_text}
{what_if_text}

Generate 3-5 creative ideas for:
1. A new tool/library that could be useful
2. An improvement to existing projects
3. A combination of their skills with current trends
4. Something they haven't tried but would benefit from

Each idea format:
IDEA: [title in 1 line]
DESCRIPTION: [2-3 sentences explaining the idea]
WHY: [why it matters for this developer]
DIFFICULTY: [easy/medium/hard]

Be specific and practical, not generic. Reference actual projects and technologies from the data."""

    LOG("Generating ideas with Ollama...")
    raw_output = _ollama_generate(prompt, timeout=300)
    if not raw_output:
        LOG("Ollama returned empty response, aborting")
        return

    # Step 5: Parse and save individual ideas
    ideas = _parse_ideas(raw_output)
    LOG(f"Parsed {len(ideas)} ideas from Ollama output")

    if not ideas:
        # Save raw output as single record if parsing failed
        _save_knowledge_record(
            db_path,
            content=f"[Weekly Ideas]\n\n{raw_output}",
            record_type="decision",
            tags=["idea-generator", "innovation", "weekly"],
            source="idea-engine",
            confidence=0.6,
        )
        ideas_count = 1
    else:
        ideas_count = 0
        for idea in ideas:
            _save_knowledge_record(
                db_path,
                content=(
                    f"[Idea] {idea['title']}\n\n"
                    f"{idea['description']}\n\n"
                    f"Why it matters: {idea['why']}\n"
                    f"Difficulty: {idea['difficulty']}"
                ),
                record_type="decision",
                tags=["idea-generator", "innovation", "weekly"],
                source="idea-engine",
                confidence=0.6,
            )
            ideas_count += 1

    # Step 6: Send Telegram notification
    tg_lines = [
        "<b>\U0001f4a1 Weekly Ideas</b>",
        f"<i>{datetime.now().strftime('%A, %B %d %Y')}</i>",
        "",
    ]

    if ideas:
        for i, idea in enumerate(ideas, 1):
            diff_emoji = {
                "easy": "\U0001f7e2",
                "medium": "\U0001f7e1",
                "hard": "\U0001f534",
            }.get(idea["difficulty"].lower(), "\u26aa")
            tg_lines.append(f"<b>{i}. {idea['title']}</b> {diff_emoji}")
            tg_lines.append(f"   {idea['description'][:200]}")
            tg_lines.append(f"   <i>Why: {idea['why'][:150]}</i>")
            tg_lines.append("")
    else:
        tg_lines.append(raw_output[:1500])

    if what_ifs:
        tg_lines.append("")
        tg_lines.append("<b>\U0001f914 What If...</b>")
        for wf in what_ifs[:2]:
            tg_lines.append(f"  {wf['scenario'][:200]}")

    message = "\n".join(tg_lines)
    sent = _telegram_broadcast(message)

    elapsed = time.time() - start_time
    LOG(
        f"=== IDEA GENERATOR COMPLETE === "
        f"({elapsed:.1f}s, {ideas_count} ideas saved, {sent} messages sent)"
    )


def _parse_ideas(raw: str) -> list[dict[str, str]]:
    """Parse structured ideas from Ollama output.

    Expects format with IDEA:, DESCRIPTION:, WHY:, DIFFICULTY: markers.
    Falls back to numbered lines if markers not found.
    """
    ideas: list[dict[str, str]] = []

    # Try structured parsing first
    current: dict[str, str] = {}
    for line in raw.split("\n"):
        line = line.strip()
        if not line:
            continue

        line_upper = line.upper()

        if line_upper.startswith("IDEA:"):
            if current.get("title"):
                ideas.append(_finalize_idea(current))
            current = {
                "title": line[5:].strip().strip("*#"),
                "description": "",
                "why": "",
                "difficulty": "medium",
            }
        elif line_upper.startswith("DESCRIPTION:"):
            current["description"] = line[12:].strip()
        elif line_upper.startswith("WHY:") or line_upper.startswith("WHY IT MATTERS:"):
            sep_idx = line.index(":") + 1
            current["why"] = line[sep_idx:].strip()
        elif line_upper.startswith("DIFFICULTY:"):
            diff = line[11:].strip().lower()
            if "easy" in diff:
                current["difficulty"] = "easy"
            elif "hard" in diff:
                current["difficulty"] = "hard"
            else:
                current["difficulty"] = "medium"
        elif current:
            # Continuation line -- append to the most recent open field
            if current.get("why"):
                # Already have WHY, this might be extra description text; skip
                pass
            elif current.get("description"):
                current["description"] += " " + line
            elif current.get("title") and not current.get("description"):
                # Text after IDEA before DESCRIPTION -- treat as description
                current["description"] = line

    if current.get("title"):
        ideas.append(_finalize_idea(current))

    # Fallback: try numbered lines (1. Title\n   Description...)
    if not ideas:
        import re

        blocks = re.split(r"\n(?=\d+[\.\)]\s)", raw)
        for block in blocks:
            block = block.strip()
            if not block:
                continue
            match = re.match(r"\d+[\.\)]\s*(.+)", block)
            if match:
                lines = block.split("\n")
                title = re.sub(r"^\d+[\.\)]\s*", "", lines[0]).strip().strip("*#")
                desc = " ".join(ln.strip() for ln in lines[1:] if ln.strip())[:300]
                if title and len(title) > 5:
                    ideas.append({
                        "title": title[:100],
                        "description": desc or title,
                        "why": "See description",
                        "difficulty": "medium",
                    })

    return ideas[:5]


def _finalize_idea(raw: dict[str, str]) -> dict[str, str]:
    """Clean up a parsed idea dict, ensuring all fields are populated."""
    return {
        "title": raw.get("title", "Untitled")[:100].strip(),
        "description": (
            raw.get("description", "") or raw.get("title", "")
        ).strip()[:500],
        "why": (
            raw.get("why", "") or "Potential value identified"
        ).strip()[:300],
        "difficulty": raw.get("difficulty", "medium"),
    }


# ---------------------------------------------------------------------------
# 2. "What If" Engine
# ---------------------------------------------------------------------------


def _generate_what_ifs(db_path: str) -> list[dict[str, str]]:
    """Generate 'what if' cross-pollination scenarios from project patterns.

    Finds architectural patterns used in different projects and generates
    scenarios like 'What if you applied X from project A to project B?'

    Returns list of dicts: {scenario, benefit, risk, effort}.
    """
    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row
    scenarios: list[dict[str, str]] = []

    try:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=180)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )

        rows = db.execute(
            """SELECT project, tags, content, type FROM knowledge
               WHERE status = 'active' AND project IS NOT NULL
                 AND project != 'general' AND created_at > ?
               ORDER BY project""",
            (cutoff,),
        ).fetchall()

        if not rows:
            db.close()
            return []

        # Build project -> technologies and project -> patterns maps
        project_techs: dict[str, Counter] = {}
        project_patterns: dict[str, list[str]] = {}

        for row in rows:
            proj = row["project"]
            if proj not in project_techs:
                project_techs[proj] = Counter()
                project_patterns[proj] = []

            try:
                tags = json.loads(row["tags"]) if row["tags"] else []
                for tag in tags:
                    if tag and len(tag) > 1:
                        project_techs[proj][tag.lower()] += 1
            except (json.JSONDecodeError, TypeError):
                pass

            if row["type"] in ("solution", "decision"):
                project_patterns[proj].append(row["content"][:150])

        projects = list(project_techs.keys())
        if len(projects) < 2:
            db.close()
            return []

        # Find technologies unique to one project that could benefit another
        cross_opportunities: list[str] = []

        for i, proj_a in enumerate(projects):
            techs_a = set(t for t, _ in project_techs[proj_a].most_common(10))
            for proj_b in projects[i + 1 :]:
                techs_b = set(t for t, _ in project_techs[proj_b].most_common(10))

                unique_to_a = techs_a - techs_b
                for tech in list(unique_to_a)[:2]:
                    cross_opportunities.append(
                        f"Project {proj_a} uses '{tech}' extensively. "
                        f"Project {proj_b} doesn't. Could {proj_b} benefit from {tech}?"
                    )

                unique_to_b = techs_b - techs_a
                for tech in list(unique_to_b)[:2]:
                    cross_opportunities.append(
                        f"Project {proj_b} uses '{tech}' extensively. "
                        f"Project {proj_a} doesn't. Could {proj_a} benefit from {tech}?"
                    )

        # Collect pattern descriptions for Ollama context
        all_patterns_text = ""
        for proj, pats in list(project_patterns.items())[:5]:
            if pats:
                all_patterns_text += f"\n{proj}:\n"
                for p in pats[:3]:
                    all_patterns_text += f"  - {p}\n"

        db.close()

        if not cross_opportunities and not all_patterns_text:
            return []

        # Ask Ollama to generate structured what-if scenarios
        prompt = f"""You are analyzing a developer's projects to find cross-pollination opportunities.

Cross-technology observations:
{chr(10).join(f'- {co}' for co in cross_opportunities[:6])}

Architectural patterns per project:
{all_patterns_text}

Generate 3-4 creative "What if" scenarios. Each MUST reference actual projects/technologies from above.
Format each as:
SCENARIO: What if you applied/used/replaced [specific thing] from/in [specific project]?
BENEFIT: [1 sentence about the potential gain]
RISK: [1 sentence about potential downsides]
EFFORT: [low/medium/high]"""

        raw = _ollama_generate(prompt, timeout=120)
        if not raw:
            # Fallback: return raw cross-opportunity observations as scenarios
            for co in cross_opportunities[:3]:
                scenarios.append({
                    "scenario": co,
                    "benefit": "Potential technology transfer",
                    "risk": "Migration effort required",
                    "effort": "medium",
                })
            return scenarios

        # Parse Ollama output
        current: dict[str, str] = {}
        for line in raw.split("\n"):
            line = line.strip()
            if not line:
                continue
            line_upper = line.upper()

            if line_upper.startswith("SCENARIO:"):
                if current.get("scenario"):
                    scenarios.append(current)
                current = {
                    "scenario": line[9:].strip(),
                    "benefit": "",
                    "risk": "",
                    "effort": "medium",
                }
            elif line_upper.startswith("BENEFIT:"):
                current["benefit"] = line[8:].strip()
            elif line_upper.startswith("RISK:"):
                current["risk"] = line[5:].strip()
            elif line_upper.startswith("EFFORT:"):
                eff = line[7:].strip().lower()
                if "low" in eff:
                    current["effort"] = "low"
                elif "high" in eff:
                    current["effort"] = "high"
                else:
                    current["effort"] = "medium"

        if current.get("scenario"):
            scenarios.append(current)

    except Exception as e:
        LOG(f"What-If generation error: {e}")
        try:
            db.close()
        except Exception:
            pass

    return scenarios[:4]


# ---------------------------------------------------------------------------
# 3. Innovation Digest -- bi-weekly, 1st and 15th of month
# ---------------------------------------------------------------------------


def run_innovation_digest(db_path: str) -> None:
    """Generate bi-weekly innovation digest connecting new information to projects.

    Collects RSS articles, curiosity research, GitHub releases from last
    2 weeks and finds connections to existing projects using Ollama.
    Saves as type='lesson' and sends rich Telegram report.
    """
    LOG("=== INNOVATION DIGEST STARTING ===")
    start_time = time.time()

    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row

    cutoff_14d = (datetime.now(timezone.utc) - timedelta(days=14)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )

    # Step 1: Collect new knowledge from last 2 weeks
    new_items: list[dict[str, str]] = []

    try:
        # RSS articles
        rss = db.execute(
            """SELECT content FROM knowledge
               WHERE source = 'rss' AND status = 'active' AND created_at > ?
               ORDER BY created_at DESC LIMIT 15""",
            (cutoff_14d,),
        ).fetchall()
        for row in rss:
            new_items.append({"source": "RSS", "content": row["content"][:200]})

        # Curiosity research
        curiosity = db.execute(
            """SELECT content FROM knowledge
               WHERE source = 'curiosity-engine' AND status = 'active'
                 AND created_at > ?
               ORDER BY created_at DESC LIMIT 10""",
            (cutoff_14d,),
        ).fetchall()
        for row in curiosity:
            new_items.append({"source": "Research", "content": row["content"][:200]})

        # GitHub releases
        github = db.execute(
            """SELECT content FROM knowledge
               WHERE source = 'github' AND status = 'active' AND created_at > ?
               ORDER BY created_at DESC LIMIT 10""",
            (cutoff_14d,),
        ).fetchall()
        for row in github:
            new_items.append({"source": "GitHub", "content": row["content"][:200]})

        # Dream mode insights
        dream = db.execute(
            """SELECT content FROM knowledge
               WHERE source = 'dream-mode' AND status = 'active' AND created_at > ?
               ORDER BY created_at DESC LIMIT 5""",
            (cutoff_14d,),
        ).fetchall()
        for row in dream:
            new_items.append({"source": "Insight", "content": row["content"][:200]})

    except Exception as e:
        LOG(f"Error collecting new items: {e}")
    finally:
        db.close()

    if not new_items:
        LOG("No new knowledge items in last 2 weeks, skipping digest")
        return

    # Step 2: Get active projects for context
    profile = _extract_user_profile(db_path)
    projects_text = ", ".join(p for p, _ in profile["projects"][:8])
    techs_text = ", ".join(t for t, _ in profile["technologies"][:12])

    # Step 3: Build new info summary
    new_info_text = "\n".join(
        f"- [{item['source']}] {item['content']}" for item in new_items[:20]
    )

    # Step 4: Ask Ollama to find connections
    prompt = f"""You are analyzing recent information for a software developer.

Developer's active projects: {projects_text}
Developer's technologies: {techs_text}

New information from the last 2 weeks:
{new_info_text}

Generate an "Innovation Opportunities" report:
1. For each relevant new item, explain HOW it connects to a specific project or technology
2. Suggest concrete actions the developer could take
3. Prioritize by impact (high/medium/low)

Format each opportunity as:
OPPORTUNITY: [title]
CONNECTION: [how this relates to their project/tech]
ACTION: [specific next step]
IMPACT: [high/medium/low]

Generate 3-5 opportunities. Be specific -- reference actual projects and technologies."""

    LOG("Generating innovation digest with Ollama...")
    raw_output = _ollama_generate(prompt, timeout=300)
    if not raw_output:
        LOG("Ollama returned empty, aborting innovation digest")
        return

    # Step 5: Parse opportunities
    opportunities = _parse_opportunities(raw_output)
    LOG(f"Parsed {len(opportunities)} innovation opportunities")

    # Step 6: Save to knowledge
    digest_content = f"[Innovation Digest] {datetime.now().strftime('%B %d, %Y')}\n\n"
    if opportunities:
        for opp in opportunities:
            digest_content += (
                f"## {opp['title']}\n"
                f"Connection: {opp['connection']}\n"
                f"Action: {opp['action']}\n"
                f"Impact: {opp['impact']}\n\n"
            )
    else:
        digest_content += raw_output

    _save_knowledge_record(
        db_path,
        content=digest_content,
        record_type="lesson",
        tags=["innovation-digest", "opportunities"],
        source="idea-engine",
        confidence=0.6,
    )

    # Step 7: Telegram notification
    tg_lines = [
        "<b>\U0001f680 Innovation Digest</b>",
        f"<i>{datetime.now().strftime('%B %d, %Y')} -- bi-weekly report</i>",
        f"<i>{len(new_items)} new items analyzed, {len(opportunities)} opportunities found</i>",
        "",
    ]

    if opportunities:
        for i, opp in enumerate(opportunities, 1):
            impact_emoji = {
                "high": "\U0001f525",
                "medium": "\U0001f7e1",
                "low": "\U0001f7e2",
            }.get(opp["impact"].lower(), "\u26aa")
            tg_lines.append(f"<b>{i}. {opp['title']}</b> {impact_emoji}")
            tg_lines.append(f"   {opp['connection'][:180]}")
            tg_lines.append(f"   \u27a1 {opp['action'][:150]}")
            tg_lines.append("")
    else:
        tg_lines.append(raw_output[:2000])

    message = "\n".join(tg_lines)
    sent = _telegram_broadcast(message)

    elapsed = time.time() - start_time
    LOG(
        f"=== INNOVATION DIGEST COMPLETE === "
        f"({elapsed:.1f}s, {len(opportunities)} opportunities, {sent} messages sent)"
    )


def _parse_opportunities(raw: str) -> list[dict[str, str]]:
    """Parse innovation opportunities from Ollama output."""
    opportunities: list[dict[str, str]] = []
    current: dict[str, str] = {}

    for line in raw.split("\n"):
        line = line.strip()
        if not line:
            continue
        line_upper = line.upper()

        if line_upper.startswith("OPPORTUNITY:"):
            if current.get("title"):
                opportunities.append(current)
            current = {
                "title": line[12:].strip().strip("*#"),
                "connection": "",
                "action": "",
                "impact": "medium",
            }
        elif line_upper.startswith("CONNECTION:"):
            current["connection"] = line[11:].strip()
        elif line_upper.startswith("ACTION:"):
            current["action"] = line[7:].strip()
        elif line_upper.startswith("IMPACT:"):
            imp = line[7:].strip().lower()
            if "high" in imp:
                current["impact"] = "high"
            elif "low" in imp:
                current["impact"] = "low"
            else:
                current["impact"] = "medium"

    if current.get("title"):
        opportunities.append(current)

    return opportunities[:5]


# ---------------------------------------------------------------------------
# 4. Blind Spot Challenger -- weekly, Monday 9:00 AM
# ---------------------------------------------------------------------------


def run_blind_spot_challenger(db_path: str) -> None:
    """Challenge the user's technology comfort zones.

    Analyzes which technologies the user ALWAYS picks across projects,
    and suggests alternatives that could be better for specific use cases.
    Saves challenges as type='lesson' and sends top 2 via Telegram.
    """
    LOG("=== BLIND SPOT CHALLENGER STARTING ===")
    start_time = time.time()

    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row

    # Step 1: Find comfort zone patterns
    comfort_zones: list[dict[str, Any]] = []

    try:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=180)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )

        rows = db.execute(
            """SELECT project, tags FROM knowledge
               WHERE status = 'active' AND project IS NOT NULL
                 AND project != 'general' AND created_at > ?""",
            (cutoff,),
        ).fetchall()

        # Count tech usage per project
        tech_per_project: dict[str, set[str]] = {}
        for row in rows:
            proj = row["project"]
            try:
                tags = json.loads(row["tags"]) if row["tags"] else []
                for tag in tags:
                    tag_lower = tag.lower()
                    if tag_lower and len(tag_lower) > 1 and tag_lower not in _SKIP_TAGS:
                        if tag_lower not in tech_per_project:
                            tech_per_project[tag_lower] = set()
                        tech_per_project[tag_lower].add(proj)
            except (json.JSONDecodeError, TypeError):
                pass

        total_projects = len(set(row["project"] for row in rows if row["project"]))

        for tech, projects in tech_per_project.items():
            # Comfort zone = used in 3+ projects, or 2+ when total projects <= 3
            if len(projects) >= 3 or (total_projects <= 3 and len(projects) >= 2):
                comfort_zones.append({
                    "tech": tech,
                    "count": len(projects),
                    "projects": sorted(projects)[:5],
                })

        comfort_zones.sort(key=lambda x: x["count"], reverse=True)

    except Exception as e:
        LOG(f"Error analyzing comfort zones: {e}")
    finally:
        db.close()

    if not comfort_zones:
        LOG("No clear comfort zone patterns found (need more project data)")
        return

    # Step 2: Build comfort zone text for Ollama
    comfort_text = "\n".join(
        f"- {cz['tech']}: used in {cz['count']} projects ({', '.join(cz['projects'])})"
        for cz in comfort_zones[:10]
    )

    # Step 3: Get user profile for context
    profile = _extract_user_profile(db_path)
    solutions_text = "\n".join(f"- {s}" for s in profile["solutions"][:5])

    # Step 4: Ask Ollama for challenges
    prompt = f"""You are a technology advisor challenging a developer's comfort zones.

Technologies this developer ALWAYS uses (comfort zones):
{comfort_text}

Recent solutions they've built:
{solutions_text}

For the top 3 most-used technologies, generate a challenge:
1. Identify a SPECIFIC use case from their projects where an alternative could be significantly better
2. Name the alternative technology and explain WHY it's better for that specific case
3. Be provocative but fair -- acknowledge the current choice's strengths

Format each challenge as:
CHALLENGE: You always use [tech]. For [specific use case], [alternative] could be [N]x better because [reason].
CURRENT_STRENGTH: [why their current choice isn't bad]
ALTERNATIVE: [specific technology name]
LEARN_TIME: [how long to learn the basics: hours/days/weeks]
EVIDENCE: [1 concrete example or benchmark]"""

    LOG("Generating challenges with Ollama...")
    raw_output = _ollama_generate(prompt, timeout=150)
    if not raw_output:
        LOG("Ollama returned empty, aborting blind spot challenger")
        return

    # Step 5: Parse challenges
    challenges = _parse_challenges(raw_output)
    LOG(f"Parsed {len(challenges)} challenges")

    # Step 6: Save to knowledge
    for challenge in challenges:
        _save_knowledge_record(
            db_path,
            content=(
                f"[Blind Spot Challenge] {challenge['challenge']}\n\n"
                f"Current strength: {challenge['current_strength']}\n"
                f"Alternative: {challenge['alternative']}\n"
                f"Learn time: {challenge['learn_time']}\n"
                f"Evidence: {challenge['evidence']}"
            ),
            record_type="lesson",
            tags=["blind-spot-challenge", "growth"],
            source="idea-engine",
            confidence=0.5,
        )

    # Step 7: Telegram -- send top 2 challenges
    tg_lines = [
        "<b>\U0001f52e Monday Challenge</b>",
        f"<i>{datetime.now().strftime('%B %d, %Y')}</i>",
        "",
        f"Comfort zones detected: {len(comfort_zones)} technologies you always pick.",
        "",
    ]

    for i, ch in enumerate(challenges[:2], 1):
        tg_lines.append(f"<b>{i}. {ch['challenge'][:250]}</b>")
        tg_lines.append(f"   \u2705 Current: {ch['current_strength'][:150]}")
        tg_lines.append(f"   \U0001f504 Try: {ch['alternative']}")
        tg_lines.append(f"   \u23f0 Learn: {ch['learn_time']}")
        if ch["evidence"]:
            tg_lines.append(f"   \U0001f4ca {ch['evidence'][:150]}")
        tg_lines.append("")

    if not challenges:
        tg_lines.append(raw_output[:1500])

    message = "\n".join(tg_lines)
    sent = _telegram_broadcast(message)

    elapsed = time.time() - start_time
    LOG(
        f"=== BLIND SPOT CHALLENGER COMPLETE === "
        f"({elapsed:.1f}s, {len(challenges)} challenges, {sent} messages sent)"
    )


def _parse_challenges(raw: str) -> list[dict[str, str]]:
    """Parse blind spot challenges from Ollama output."""
    challenges: list[dict[str, str]] = []
    current: dict[str, str] = {}

    for line in raw.split("\n"):
        line = line.strip()
        if not line:
            continue
        line_upper = line.upper()

        if line_upper.startswith("CHALLENGE:"):
            if current.get("challenge"):
                challenges.append(_finalize_challenge(current))
            current = {
                "challenge": line[10:].strip(),
                "current_strength": "",
                "alternative": "",
                "learn_time": "unknown",
                "evidence": "",
            }
        elif line_upper.startswith("CURRENT_STRENGTH:") or line_upper.startswith(
            "CURRENT STRENGTH:"
        ):
            sep_idx = line.index(":") + 1
            current["current_strength"] = line[sep_idx:].strip()
        elif line_upper.startswith("ALTERNATIVE:"):
            current["alternative"] = line[12:].strip()
        elif line_upper.startswith("LEARN_TIME:") or line_upper.startswith(
            "LEARN TIME:"
        ):
            sep_idx = line.index(":") + 1
            current["learn_time"] = line[sep_idx:].strip()
        elif line_upper.startswith("EVIDENCE:"):
            current["evidence"] = line[9:].strip()

    if current.get("challenge"):
        challenges.append(_finalize_challenge(current))

    return challenges[:3]


def _finalize_challenge(raw: dict[str, str]) -> dict[str, str]:
    """Clean up a parsed challenge dict, ensuring all fields are populated."""
    return {
        "challenge": raw.get("challenge", "")[:300].strip(),
        "current_strength": (
            raw.get("current_strength", "") or "Well-established choice"
        ).strip()[:200],
        "alternative": (
            raw.get("alternative", "") or "See challenge description"
        ).strip()[:100],
        "learn_time": (raw.get("learn_time", "") or "varies").strip()[:50],
        "evidence": (raw.get("evidence", "") or "").strip()[:200],
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Idea Generation and Innovation Engine"
    )
    parser.add_argument("--db", default=str(MEMORY_DIR / "memory.db"))
    parser.add_argument(
        "--run",
        choices=["ideas", "innovation", "challenge", "what-ifs"],
        required=True,
        help="Which module to run",
    )
    args = parser.parse_args()

    if args.run == "ideas":
        run_idea_generator(args.db)
    elif args.run == "innovation":
        run_innovation_digest(args.db)
    elif args.run == "challenge":
        run_blind_spot_challenger(args.db)
    elif args.run == "what-ifs":
        # Debug mode: just print what-if scenarios to stdout
        what_if_scenarios = _generate_what_ifs(args.db)
        if not what_if_scenarios:
            print("No what-if scenarios generated (need 2+ projects with data).")
        for s in what_if_scenarios:
            print(f"\nScenario: {s['scenario']}")
            print(f"  Benefit: {s['benefit']}")
            print(f"  Risk:    {s['risk']}")
            print(f"  Effort:  {s['effort']}")
