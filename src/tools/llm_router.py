#!/usr/bin/env python3
"""
LLM Router — smart routing for Vito Brain.

Architecture:
    CODE        → Claude Opus (subscription, in-session) — human writes code via Claude Code
    RESEARCH    → DeepSeek 671B Cloud (free via Ollama) — web search, summarization, analysis
    BRAIN       → DeepSeek 671B Cloud — reasoning, self-tasking, interest discovery
    QUICK Q&A   → Vito local (qwen 32b) — offline, knows projects
    ALL RESULTS → saved to Vito memory DB → Vito learns

Usage (background tasks — bot, scheduler, brain_autonomy):
    from tools.llm_router import generate_cheap, generate_cloud, generate_local

    answer = generate_cheap("Summarize this article: ...")   # DeepSeek → Claude fallback
    answer = generate_cloud("Analyze this pattern: ...")      # DeepSeek 671B direct
    answer = generate_local("What projects do I have?")       # Vito local, free
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.request
import ssl
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tool_support import make_ssl_context, ollama_generate  # noqa: E402

TELEGRAM_ENV_PATH = Path(__file__).parent.parent.parent / "telegram" / ".env"

LOG = lambda msg: sys.stderr.write(f"[llm-router] {msg}\n")

# Models
CLAUDE_CHEAP = "claude-haiku-4-5-20251001"   # $0.25/$1.25 per 1M tokens
CLAUDE_MID = "claude-sonnet-4-6-20250514"    # for complex tasks
OLLAMA_MODEL = os.environ.get("MEMORY_TOOLS_OLLAMA_MODEL", "")  # empty -> configured MEMORY_LLM_MODEL
OLLAMA_CLOUD = "deepseek-v3.1:671b-cloud"    # cloud 671B, free via Ollama Cloud


def _load_dotenv(path: Path) -> dict[str, str]:
    env = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def _make_ssl_context() -> ssl.SSLContext:
    """Verified TLS context for api.anthropic.com (the API key travels in the request)."""
    return make_ssl_context()


def _get_api_key() -> str | None:
    """Get Anthropic API key from env or .env file."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    env = _load_dotenv(TELEGRAM_ENV_PATH)
    return env.get("ANTHROPIC_API_KEY")


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*[A-Za-z]|\x1b\].*?\x07", "", text)


def _strip_thinking(text: str) -> str:
    """Remove DeepSeek thinking blocks from output."""
    # Remove <think>...</think> blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Remove "Thinking...\n...done thinking.\n" blocks
    text = re.sub(r"Thinking\.\.\.\n.*?\.\.\.\s*done thinking\.\s*\n?", "", text, flags=re.DOTALL)
    return text.strip()


# ---------------------------------------------------------------------------
# Claude API
# ---------------------------------------------------------------------------

def generate_claude(
    prompt: str,
    model: str = CLAUDE_CHEAP,
    max_tokens: int = 2048,
    system: str = "",
    timeout: int = 60,
) -> str:
    """Generate text via Claude API. Returns text or empty string."""
    api_key = _get_api_key()
    if not api_key:
        LOG("No ANTHROPIC_API_KEY found")
        return ""

    messages = [{"role": "user", "content": prompt}]
    payload: dict = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    if system:
        payload["system"] = system

    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=data,
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
    )

    ctx = _make_ssl_context()
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            result = json.loads(resp.read())
            # Extract text from content blocks
            content = result.get("content", [])
            texts = [b["text"] for b in content if b.get("type") == "text"]
            answer = "\n".join(texts).strip()
            # Log usage
            usage = result.get("usage", {})
            LOG(f"Claude [{model}]: {usage.get('input_tokens', '?')} in, "
                f"{usage.get('output_tokens', '?')} out")
            return answer
    except Exception as e:
        LOG(f"Claude API error: {e}")
        return ""


# ---------------------------------------------------------------------------
# Ollama (local)
# ---------------------------------------------------------------------------

def generate_local(
    prompt: str,
    model: str | None = None,
    timeout: int = 180,
) -> str:
    """Generate text via the Ollama CLI. Free, offline."""
    return ollama_generate(prompt, model or OLLAMA_MODEL or None, timeout, log=LOG)


# ---------------------------------------------------------------------------
# Ollama Cloud (DeepSeek 671B — free, powerful)
# ---------------------------------------------------------------------------

def generate_cloud(
    prompt: str,
    model: str = OLLAMA_CLOUD,
    timeout: int = 120,
) -> str:
    """Generate text via Ollama Cloud (DeepSeek 671B). Free, powerful."""
    answer = ollama_generate(
        prompt, model, timeout, log=LOG,
        postprocess=lambda text: _strip_thinking(text),
    )
    if answer:
        LOG(f"Cloud [{model}]: ~{len(answer)//3} tokens")
    return answer


# ---------------------------------------------------------------------------
# Smart Router
# ---------------------------------------------------------------------------

def generate_cheap(prompt: str, max_tokens: int = 2048, system: str = "") -> str:
    """Generate with DeepSeek Cloud (free) → Claude Haiku → Ollama local.

    For: research, analysis, summarization, reasoning — NOT code.
    """
    # Try free cloud first (671B, powerful)
    answer = generate_cloud(prompt)
    if answer:
        return answer
    # Fallback to Claude Haiku (paid but reliable)
    answer = generate_claude(prompt, model=CLAUDE_CHEAP, max_tokens=max_tokens, system=system)
    if answer:
        return answer
    # Last resort: local Ollama
    LOG("Cloud + Claude failed, falling back to local Ollama")
    return generate_local(prompt)


def generate_smart(
    prompt: str,
    complexity: str = "auto",
    max_tokens: int = 2048,
    system: str = "",
) -> str:
    """Smart routing based on task type.

    Role-based routing:
        "simple"   → Vito local (qwen 32b) — free, offline, knows projects
        "medium"   → DeepSeek 671B cloud — free, powerful reasoning
        "complex"  → DeepSeek 671B → Claude API fallback
        "auto"     → estimate from prompt

    NOTE: Code generation is done by Claude Opus in-session (subscription).
    This router is for background tasks (bot, scheduler, brain_autonomy).
    """
    if complexity == "auto":
        complexity = _estimate_complexity(prompt)

    if complexity == "simple":
        return generate_local(prompt)
    elif complexity == "complex":
        # Research/analysis → DeepSeek first, Claude fallback
        return generate_cheap(prompt, max_tokens, system)
    else:  # medium/research
        return generate_cheap(prompt, max_tokens, system)


def _estimate_complexity(prompt: str) -> str:
    """Estimate task type for routing."""
    prompt_lower = prompt.lower()
    length = len(prompt)

    # Research/analysis indicators → DeepSeek Cloud
    research_keywords = [
        "analyze", "compare", "research", "summarize", "explain",
        "анализ", "сравни", "исследуй", "объясни", "расскажи",
        "trade-off", "pros and cons", "best practice",
        "design", "architect", "спроектируй",
    ]

    # Simple indicators → Vito local
    simple_keywords = [
        "what is", "что такое", "list", "перечисли",
        "which", "какой", "when", "когда", "where", "где",
    ]

    if any(kw in prompt_lower for kw in research_keywords) or length > 3000:
        return "complex"
    elif any(kw in prompt_lower for kw in simple_keywords) and length < 500:
        return "simple"
    else:
        return "medium"


# ---------------------------------------------------------------------------
# Convenience: research-oriented generation
# ---------------------------------------------------------------------------

def summarize(text: str, topic: str = "", max_words: int = 300) -> str:
    """Summarize text using Claude Haiku."""
    prompt = (
        f"Summarize the following for a senior software developer.\n"
        f"{f'Topic: {topic}' if topic else ''}\n\n"
        f"Text:\n{text[:10000]}\n\n"
        f"Write a clear, technical summary (max {max_words} words). "
        f"Include key facts, code examples if relevant, and actionable advice."
    )
    return generate_cheap(prompt)


def _load_rules_from_files(stack: str = "") -> str:
    """Load rules from CLAUDE.md rule files as fallback."""
    rules_dir = Path.home() / ".claude" / "rules"
    rules_text = ""

    # Map stack to rule files
    file_map = {
        "go": "go.md",
        "php": "php.md",
        "vue": "vue.md",
        "sql": "database.md",
        "postgres": "database.md",
        "docker": "docker.md",
    }

    # Always load general architecture from CLAUDE.md (Architecture Standards section)
    claude_md = Path.home() / ".claude" / "CLAUDE.md"
    if claude_md.exists():
        content = claude_md.read_text(encoding="utf-8")
        # Extract Architecture Standards section
        match = re.search(
            r"## Architecture Standards.*?(?=\n## |\Z)",
            content, re.DOTALL
        )
        if match:
            rules_text += match.group()[:2000] + "\n\n"

    # Load stack-specific rules
    if stack:
        for key, filename in file_map.items():
            if key in stack.lower():
                filepath = rules_dir / filename
                if filepath.exists():
                    rules_text += filepath.read_text(encoding="utf-8")[:3000] + "\n\n"
                break

    return rules_text


# NOTE: Code generation is done by Claude Opus in-session (subscription).
# No API calls needed — user asks Claude Code directly.
# These rule-loading functions are kept for brain_autonomy context injection.


def research_and_summarize(topic: str, sources_text: str) -> str:
    """Research summarization with structured output."""
    prompt = (
        f"Based on these sources, create a comprehensive briefing about: {topic}\n\n"
        f"Include:\n"
        f"1. Overview (2-3 sentences)\n"
        f"2. Key concepts and technologies\n"
        f"3. Getting started steps\n"
        f"4. Important links from sources\n"
        f"5. Code examples if available\n\n"
        f"Sources:\n{sources_text[:10000]}\n\n"
        f"Write in clear, structured format. Use markdown headers."
    )
    return generate_cheap(prompt, max_tokens=3000)
