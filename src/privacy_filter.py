"""Inline private-section redaction, including nested and unfinished sections."""

from __future__ import annotations

import re

PRIVATE_TAG = re.compile(r"<\s*(/?)private\s*>", re.IGNORECASE)


def redact_private_sections(content: str) -> tuple[str, int]:
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
            depth += 1
        cursor = tag.end()
    if depth == 0:
        public.append(content[cursor:])
    return "".join(public), redactions
