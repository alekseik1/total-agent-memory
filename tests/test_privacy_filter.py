"""Unit tests for src/privacy_filter.py — inline <private> tag redaction."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from privacy_filter import redact_private_sections


def test_empty_content():
    cleaned, n = redact_private_sections("")
    assert cleaned == ""
    assert n == 0


def test_no_tag_is_noop():
    text = "hello world, no secrets here"
    cleaned, n = redact_private_sections(text)
    assert cleaned == text
    assert n == 0


def test_single_tag_removed():
    cleaned, n = redact_private_sections("before <private>API_KEY=sk-xxx</private> after")
    assert cleaned == "before  after"
    assert n == 1
    assert "sk-xxx" not in cleaned


def test_multiple_tags_removed():
    text = "a <private>s1</private> b <private>s2</private> c <private>s3</private> d"
    cleaned, n = redact_private_sections(text)
    assert n == 3
    assert "s1" not in cleaned and "s2" not in cleaned and "s3" not in cleaned
    assert cleaned == "a  b  c  d"


def test_tag_across_lines_dotall():
    text = "start\n<private>line1\nline2\nSECRET=abc</private>\nend"
    cleaned, n = redact_private_sections(text)
    assert n == 1
    assert "SECRET" not in cleaned
    assert "line1" not in cleaned
    assert "start" in cleaned and "end" in cleaned


def test_tag_uppercase_case_insensitive():
    cleaned, n = redact_private_sections("x <PRIVATE>TOP</PRIVATE> y")
    assert n == 1
    assert "TOP" not in cleaned
    assert cleaned == "x  y"


def test_mixed_case_tag():
    cleaned, n = redact_private_sections("a <Private>hush</Private> b <pRiVaTe>zz</pRiVaTe> c")
    assert n == 2
    assert "hush" not in cleaned and "zz" not in cleaned


def test_idempotent_on_clean_input():
    text = "nothing to redact"
    once, n1 = redact_private_sections(text)
    twice, n2 = redact_private_sections(once)
    assert once == twice == text
    assert n1 == 0 and n2 == 0


def test_idempotent_after_redaction():
    text = "x <private>s</private> y"
    first, n1 = redact_private_sections(text)
    second, n2 = redact_private_sections(first)
    assert first == "x  y"
    assert second == first
    assert n1 == 1 and n2 == 0


def test_nested_private_sections_do_not_leak_outer_tail():
    text = "a <private>outer <private>inner</private> tail</private> b"
    cleaned, n = redact_private_sections(text)
    assert (cleaned, n) == ("a  b", 1)
    assert redact_private_sections(cleaned) == (cleaned, 0)


def test_unclosed_private_section_redacts_to_end():
    assert redact_private_sections("public <private>secret") == ("public ", 1)


def test_tag_at_string_boundaries():
    cleaned, n = redact_private_sections("<private>only</private>")
    assert n == 1
    assert cleaned == ""
