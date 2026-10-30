"""Inline private-section redaction, including nested and unfinished sections."""

from __future__ import annotations

import re
from typing import Any

PRIVATE_TAG = re.compile(r"<\s*(/?)private\s*>", re.IGNORECASE)


def redact_private_sections(content: str, placeholder: str = "") -> tuple[str, int]:
    """Remove every outermost private section (tags are case- and space-tolerant, nesting-safe).

    An unterminated section hides everything after it. `placeholder` replaces each removed section.
    """
    if not content:
        return content, 0
    public: list[str] = []
    depth = 0
    redactions = 0
    cursor = 0
    for tag in PRIVATE_TAG.finditer(content):
        if depth == 0:
            public.append(content[cursor:tag.start()])
        if tag.group(1):
            depth = max(0, depth - 1)
        else:
            if depth == 0:
                redactions += 1
                public.append(placeholder)
            depth += 1
        cursor = tag.end()
    if depth == 0:
        public.append(content[cursor:])
    return "".join(public), redactions


def redact_private_value(value: Any) -> Any:
    """`redact_private_sections` applied to every string inside a JSON-like value."""
    if isinstance(value, str):
        return redact_private_sections(value)[0]
    if isinstance(value, dict):
        return {key: redact_private_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        items = [redact_private_value(item) for item in value]
        return tuple(items) if isinstance(value, tuple) else items
    return value
