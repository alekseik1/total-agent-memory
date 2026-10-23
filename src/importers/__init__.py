"""`tam import`: bring ChatGPT, claude.ai, mem0 and Markdown history into the store."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

from importers.chat_exports import ChatOptions, chatgpt_records, claude_records
from importers.markdown_tree import markdown_records
from importers.mem0_export import mem0_records
from importers.records import ImportFormatError, ImportRecord, ImportStats

RecordSource = Callable[[Path, ChatOptions, ImportStats], Iterator[ImportRecord]]

SOURCES: dict[str, RecordSource] = {
    "chatgpt": chatgpt_records,
    "claude": claude_records,
    "mem0": lambda path, _options, stats: mem0_records(path, stats),
    "markdown": lambda path, _options, stats: markdown_records(path, stats),
}
KINDS = tuple(SOURCES)

__all__ = ["KINDS", "SOURCES", "ChatOptions", "ImportFormatError", "ImportRecord", "ImportStats", "RecordSource"]
