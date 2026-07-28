"""Tests for src/validator.py — LLM transformation safety net.

Validates that LLM-transformed text (summary, merge, compress) preserves:
- code blocks (byte-for-byte)
- URLs
- file paths
- heading count
- bullet count (within tolerance)
- inline code
"""

from __future__ import annotations

import pytest

from validator import ContentValidator, ValidationResult


# ──────────────────────────────────────────────
# Code block preservation
# ──────────────────────────────────────────────


def test_validator_ok_when_transformed_identical():
    v = ContentValidator()
    text = "Hello\n\n```python\nprint('x')\n```\n\nBye"
    r = v.validate(text, text)
    assert r.ok
    assert not r.errors


def test_code_block_byte_preserved():
    v = ContentValidator()
    orig = "Intro.\n\n```go\nfunc Hello() {}\n```\n\nEnd."
    # Transformed removed filler but kept code byte-for-byte
    trans = "Intro shortened.\n\n```go\nfunc Hello() {}\n```\n\nEnd."
    r = v.validate(orig, trans)
    assert r.ok, f"expected ok, got errors: {r.errors}"


def test_code_block_modified_fails():
    v = ContentValidator()
    orig = "Intro.\n\n```go\nfunc Hello() {}\n```\n\nEnd."
    trans = "Intro.\n\n```go\nfunc Hi() {}\n```\n\nEnd."  # renamed
    r = v.validate(orig, trans)
    assert not r.ok
    assert any("code" in e.lower() for e in r.errors)


def test_code_block_removed_fails():
    v = ContentValidator()
    orig = "Text\n\n```\nA = 1\n```\n"
    trans = "Text (code removed for brevity)"
    r = v.validate(orig, trans)
    assert not r.ok


def test_multiple_code_blocks_all_preserved():
    v = ContentValidator()
    orig = "A\n```\nx=1\n```\nB\n```\ny=2\n```"
    trans = "Shorter A\n```\nx=1\n```\nShorter B\n```\ny=2\n```"
    r = v.validate(orig, trans)
    assert r.ok


# ──────────────────────────────────────────────
# URL preservation
# ──────────────────────────────────────────────


def test_url_preserved():
    v = ContentValidator()
    orig = "See https://example.com/docs for details."
    trans = "Docs at https://example.com/docs."
    r = v.validate(orig, trans)
    assert r.ok


def test_url_lost_fails():
    v = ContentValidator()
    orig = "See https://example.com/docs"
    trans = "See the docs"
    r = v.validate(orig, trans)
    assert not r.ok
    assert any("url" in e.lower() for e in r.errors)


def test_multiple_urls_preserved():
    v = ContentValidator()
    orig = "Visit https://a.com and https://b.io/x?y=1"
    trans = "Visit https://a.com and https://b.io/x?y=1."
    r = v.validate(orig, trans)
    assert r.ok


# ──────────────────────────────────────────────
# File path preservation
# ──────────────────────────────────────────────


def test_absolute_path_preserved():
    v = ContentValidator()
    orig = "Edit /Users/alice/project/src/server.py"
    trans = "Edit /Users/alice/project/src/server.py now."
    r = v.validate(orig, trans, strict_paths=True)
    assert r.ok


def test_absolute_path_lost_fails():
    v = ContentValidator()
    path = "/Users/alice/project/src/server.py"
    orig = f"Edit {path}"
    trans = "Edit the server file."
    r = v.validate(orig, trans, strict_paths=True)
    assert r.errors == [f"path lost: {path}"]


def test_tilde_path_preserved():
    v = ContentValidator()
    orig = "Run from ~/.tam/memory.db"
    trans = "From ~/.tam/memory.db."
    r = v.validate(orig, trans, strict_paths=True)
    assert r.ok


def test_path_check_disabled_by_default():
    v = ContentValidator()
    orig = "Edit /Users/alice/project/src/server.py"
    trans = "Edit the server file."
    r = v.validate(orig, trans)
    assert r.ok


def test_instance_level_strict_paths_default_still_honoured():
    v = ContentValidator(strict_paths=True)
    path = "/Users/alice/project/src/server.py"
    orig = f"Edit {path}"
    trans = "Edit the server file."
    r = v.validate(orig, trans)
    assert r.errors == [f"path lost: {path}"]


@pytest.mark.parametrize(
    "path",
    [
        "/Users/alice/agecode/src/api/v1/chat.py",
        "/etc/hosts",
        "~/.tam/memory.db",
        "/src/fact_merger.py",
        "alembic/versions/485866bf9d39_update_checkin.py",
    ],
)
def test_strict_path_detected_when_lost(path):
    v = ContentValidator()
    orig = f"See {path} for details."
    trans = "See the relevant file for details."
    r = v.validate(orig, trans, strict_paths=True)
    assert r.errors == [f"path lost: {path}"]


def test_strict_path_detected_when_lost_at_line_start():
    v = ContentValidator()
    path = "/Users/alice/agecode/src/api/v1/chat.py"
    orig = f"Context above.\n{path} is the file to check.\nMore context below."
    trans = "Context above.\nThe relevant file to check.\nMore context below."
    r = v.validate(orig, trans, strict_paths=True)
    assert r.errors == [f"path lost: {path}"]


def test_strict_path_detected_when_lost_quoted_and_parenthesised():
    v = ContentValidator()
    path = "/Users/alice/agecode/src/api/v1/chat.py"
    orig = f'See "{path}" (also referenced) for details.'
    trans = "See the relevant file for details."
    r = v.validate(orig, trans, strict_paths=True)
    assert r.errors == [f"path lost: {path}"]


def test_strict_path_survives_backtick_wrapping():
    v = ContentValidator()
    path = "/Users/alice/project/src/server.py"
    orig = f"Edit {path} now"
    trans = f"Edit `{path}` now"
    r = v.validate(orig, trans, strict_paths=True)
    assert r.ok


def test_strict_path_survives_bold_wrapping():
    v = ContentValidator()
    path = "/Users/alice/project/src/server.py"
    orig = f"Edit {path} now"
    trans = f"Edit **{path}** now"
    r = v.validate(orig, trans, strict_paths=True)
    assert r.ok


@pytest.mark.parametrize(
    "phrase",
    [
        "EU/Russia",
        "1/6",
        "activity/nutrition/sleep",
        "SQLAlchemy/Postgres",
        "closes/rolls",
        "/compact",
        "/jira-task",
        "/loop",
        "/babysit-prs",
        "/memory",
        "python/3.12",
        "requests/2.31",
        "uvicorn/0.30.1",
        "1/6.5",
        "3.5/4.0",
        "27.07/28.07",
        "v1.2/2.0",
    ],
)
def test_strict_path_ignores_prose_slashes(phrase):
    v = ContentValidator()
    orig = f"Discussion of {phrase} came up."
    trans = "Discussion came up."
    r = v.validate(orig, trans, strict_paths=True)
    assert r.errors == []


# ──────────────────────────────────────────────
# Heading count
# ──────────────────────────────────────────────


def test_heading_count_equal_ok():
    v = ContentValidator()
    orig = "# A\nx\n## B\ny\n### C\nz"
    trans = "# A\nshort\n## B\nshort\n### C\nshort"
    r = v.validate(orig, trans)
    assert r.ok


def test_heading_count_differs_errors():
    v = ContentValidator()
    orig = "# A\nx\n## B\ny\n### C\nz"
    trans = "# A\nshort"  # lost 2 headings
    r = v.validate(orig, trans)
    assert not r.ok
    assert any("heading" in e.lower() for e in r.errors)


# ──────────────────────────────────────────────
# Bullet tolerance
# ──────────────────────────────────────────────


def test_bullets_within_tolerance_ok():
    v = ContentValidator()
    orig = "\n".join(f"- item {i}" for i in range(10))
    trans = "\n".join(f"- item {i}" for i in range(9))  # -10%
    r = v.validate(orig, trans)
    assert r.ok


def test_bullets_exceeding_tolerance_warns_or_errors():
    v = ContentValidator()
    orig = "\n".join(f"- item {i}" for i in range(10))
    trans = "- only one"  # -90%
    r = v.validate(orig, trans)
    # Lost content → errors or warnings must flag it
    assert not r.ok or r.warnings


# ──────────────────────────────────────────────
# Inline code preservation
# ──────────────────────────────────────────────


def test_inline_code_preserved():
    v = ContentValidator()
    orig = "Call `memory_save()` with proper args."
    trans = "Use `memory_save()`."
    r = v.validate(orig, trans)
    assert r.ok


def test_inline_code_lost_fails():
    v = ContentValidator()
    orig = "Call `memory_save()` to persist."
    trans = "Call the save function."
    r = v.validate(orig, trans)
    assert not r.ok


# ──────────────────────────────────────────────
# Retry loop
# ──────────────────────────────────────────────


def test_retry_succeeds_on_second_try():
    from validator import validate_and_retry

    attempts = {"n": 0}

    def transform(text: str, feedback: str | None = None) -> str:
        attempts["n"] += 1
        if attempts["n"] == 1:
            return "result without url"  # bad: missing URL
        return f"{text} but short with https://ex.com"  # good on retry

    orig = "long text with https://ex.com and more"
    final, result = validate_and_retry(orig, transform, max_retries=2)
    assert result.ok
    assert attempts["n"] == 2
    assert "https://ex.com" in final


def test_retry_gives_up_after_max():
    from validator import validate_and_retry

    def transform(text: str, feedback: str | None = None) -> str:
        return "always broken"  # never preserves

    orig = "text with https://ex.com"
    final, result = validate_and_retry(orig, transform, max_retries=2)
    assert not result.ok


# ──────────────────────────────────────────────
# Edge cases
# ──────────────────────────────────────────────


def test_empty_original_ok():
    v = ContentValidator()
    assert v.validate("", "").ok


def test_whitespace_normalization_ok():
    v = ContentValidator()
    orig = "Text  with\t\tspaces\n\n\n\nand newlines"
    trans = "Text with spaces and newlines"
    r = v.validate(orig, trans)
    assert r.ok


def test_url_trailing_punctuation_handled():
    """URL followed by period/comma should match."""
    v = ContentValidator()
    orig = "Docs: https://example.com."
    trans = "See https://example.com"
    r = v.validate(orig, trans)
    assert r.ok
