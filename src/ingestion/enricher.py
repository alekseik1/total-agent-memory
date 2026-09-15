"""
Metadata Enricher — auto-detect and add metadata to ingested content.

Detects language, project, content category, token count, and more.
Optionally generates summaries via Ollama. No external dependencies
beyond urllib for Ollama calls.
"""

from __future__ import annotations

import logging
import re
import sqlite3
import sys
import urllib.error

import config
from llm_provider import make_provider

SUMMARY_MAX_TOKENS = 256
SUMMARY_TEMPERATURE = 0.1
LOG = lambda msg: sys.stderr.write(f"[memory-enricher] {msg}\n")

# Known project names to match against content
_PROJECT_MARKERS: dict[str, list[str]] = {
    "claude-memory-server": ["memory_save", "memory_recall", "claude-memory", "mcp server", "chromadb"],
    "ImPatient": ["impatient", "patient", "appointment", "clinic"],
    "vrmw-companion": ["vrmw", "quest 3", "unity", "virtual display"],
}

# Code detection patterns
_CODE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"```\w*\n", re.MULTILINE),
    re.compile(r"^\s*(?:def |class |func |function |const |let |var |import |from |package )", re.MULTILINE),
    re.compile(r"^\s*(?:if\s*\(|for\s*\(|while\s*\(|switch\s*\(|try\s*\{)", re.MULTILINE),
    re.compile(r"(?:=>|->|\{\}|\[\]|::\s*\w+)", re.MULTILINE),
]

_CONFIG_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^\s*\[[\w.-]+\]\s*$", re.MULTILINE),  # INI sections
    re.compile(r"^\w+\s*[:=]\s*", re.MULTILINE),  # key: value / key = value
    re.compile(r"^\s*-\s+\w+:", re.MULTILINE),  # YAML lists
    re.compile(r'^\s*"[\w.-]+":\s*', re.MULTILINE),  # JSON keys
]

_DOC_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^#{1,6}\s+", re.MULTILINE),  # Markdown headers
    re.compile(r"^\*\*[^*]+\*\*", re.MULTILINE),  # Bold text
    re.compile(r"^>\s+", re.MULTILINE),  # Blockquotes
    re.compile(r"^\d+\.\s+", re.MULTILINE),  # Numbered lists
]


class MetadataEnricher:
    """Enrich ingested content with automatic metadata."""

    def __init__(self, db: sqlite3.Connection) -> None:
        self.db = db

    def enrich(self, content: str, metadata: dict | None = None) -> dict:
        """Auto-detect and add metadata to content.

        Detects: language, project, content_category, estimated_tokens,
        word_count, has_code. Merges with existing metadata if provided.

        Returns enriched metadata dict.
        """
        if not content:
            return metadata or {}

        base: dict = dict(metadata) if metadata else {}

        # Language detection
        if "language" not in base:
            base["language"] = self.detect_language(content)

        # Project detection
        if "project" not in base:
            file_path = base.get("file_path") or base.get("source_file")
            detected = self.detect_project(content, file_path=file_path)
            if detected:
                base["project"] = detected

        # Content category
        if "content_category" not in base:
            base["content_category"] = self.detect_content_type(content)

        # Token and word counts
        base["estimated_tokens"] = len(content) // 4 if content else 0
        base["word_count"] = len(content.split())
        base["char_count"] = len(content)

        # Code detection
        base["has_code"] = self._has_code(content)

        return base

    def detect_language(self, text: str) -> str:
        """Detect language: 'ru' or 'en'.

        Simple heuristic based on Cyrillic vs Latin character frequency.
        """
        if not text:
            return "en"

        # Count Cyrillic and Latin characters
        cyrillic = 0
        latin = 0
        for ch in text:
            cp = ord(ch)
            if 0x0400 <= cp <= 0x04FF or 0x0500 <= cp <= 0x052F:
                cyrillic += 1
            elif (0x0041 <= cp <= 0x005A) or (0x0061 <= cp <= 0x007A):
                latin += 1

        total = cyrillic + latin
        if total == 0:
            return "en"

        return "ru" if cyrillic / total > 0.3 else "en"

    def detect_project(self, text: str, file_path: str | None = None) -> str | None:
        """Guess project name from content or file path."""
        # Check file path first (most reliable)
        if file_path:
            path_lower = file_path.lower()
            for project, markers in _PROJECT_MARKERS.items():
                if project.lower() in path_lower:
                    return project

        # Check content for project markers
        if text:
            text_lower = text.lower()
            best_project: str | None = None
            best_score = 0

            for project, markers in _PROJECT_MARKERS.items():
                score = sum(1 for m in markers if m.lower() in text_lower)
                if score > best_score:
                    best_score = score
                    best_project = project

            if best_score >= 2:
                return best_project

        # Try to detect from DB — look for recently used projects
        if text:
            try:
                rows = self.db.execute(
                    """SELECT DISTINCT project FROM knowledge
                       WHERE project IS NOT NULL AND project != ''
                       ORDER BY updated_at DESC LIMIT 20"""
                ).fetchall()
                text_lower = text.lower()
                for row in rows:
                    project_name = row[0]
                    if project_name and project_name.lower() in text_lower:
                        return project_name
            except sqlite3.Error:
                pass

        return None

    def detect_content_type(self, text: str) -> str:
        """Classify content: code | documentation | discussion | config | other."""
        if not text:
            return "other"

        # Count pattern matches for each category
        code_score = sum(1 for p in _CODE_PATTERNS if p.search(text))
        config_score = sum(1 for p in _CONFIG_PATTERNS if p.search(text))
        doc_score = sum(1 for p in _DOC_PATTERNS if p.search(text))

        # Discussion indicators
        discussion_score = 0
        discussion_words = ["think", "should", "could", "maybe", "consider", "suggest", "agree", "disagree"]
        text_lower = text.lower()
        discussion_score += sum(1 for w in discussion_words if w in text_lower)
        # Question marks indicate discussion
        discussion_score += min(text.count("?"), 3)

        # Pick highest score
        scores = {
            "code": code_score * 2,  # Weight code higher — patterns are more specific
            "config": config_score,
            "documentation": doc_score,
            "discussion": discussion_score,
        }

        best = max(scores, key=lambda k: scores[k])
        if scores[best] == 0:
            return "other"

        return best

    def summarize_with_ollama(self, text: str, max_length: int = 200) -> str | None:
        """Generate short summary using Ollama.

        Returns summary string or None if Ollama unavailable.
        """
        if not text or len(text) < 50:
            return text[:max_length] if text else None
        if not config.has_llm():
            return None

        # Truncate input to avoid overwhelming the model
        truncated = text[:4000] if len(text) > 4000 else text

        prompt = (
            f"Summarize the following content in {max_length} characters or less. "
            f"Return ONLY the summary, no explanation or prefix.\n\n"
            f"Content:\n{truncated}"
        )

        try:
            summary = make_provider(config.get_llm_provider()).complete(
                prompt, model=config.get_llm_model_for_provider(), max_tokens=SUMMARY_MAX_TOKENS,
                temperature=SUMMARY_TEMPERATURE, timeout=config.get_llm_timeout_sec(),
            ).strip()
            return summary[:max_length] if summary else None
        except (ValueError, RuntimeError, urllib.error.URLError, OSError) as exc:
            logging.getLogger(__name__).warning("Summary generation failed", extra={"error": str(exc)})
            return None

    @staticmethod
    def _has_code(text: str) -> bool:
        """Check if text contains code blocks or code-like content."""
        # Markdown code fences
        if "```" in text:
            return True
        # Indented code blocks (4+ spaces at line start with code-like content)
        if re.search(r"^    (?:def |class |func |if |for |return )", text, re.MULTILINE):
            return True
        # Multiple code indicators in the text
        code_hits = sum(1 for p in _CODE_PATTERNS if p.search(text))
        return code_hits >= 2
