"""Markdown directory reader (Obsidian vaults, docs folders, note dumps).

Walks the tree for ``*.md`` / ``*.markdown`` files, skipping hidden
directories (``.obsidian``, ``.git``, ``.trash``) and symlinks. A note becomes
one record when its body is at most ``SECTION_SPLIT_CHARS``; a longer note is
split into one record per heading section (``#`` to ``###``, headings inside
fenced code blocks ignored). YAML frontmatter ``tags`` / ``tag`` become tags;
``created`` / ``date`` / ``created_at`` give the original time, otherwise the
file modification time is used.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from importers.records import ImportFormatError, ImportRecord, ImportStats, normalise_tag, parse_timestamp

MARKDOWN_SUFFIXES = frozenset({".md", ".markdown"})
# Notes with a body above this many characters are stored per heading section.
SECTION_SPLIT_CHARS = 4000
# Files above this size are reported and skipped rather than read into memory.
MAX_NOTE_BYTES = 5 * 1024 * 1024
SPLIT_HEADING_MAX_LEVEL = 3
FRONTMATTER_DELIMITER = "---"
FRONTMATTER_TAG_KEYS = ("tags", "tag")
FRONTMATTER_TIME_KEYS = ("created", "created_at", "date")
FRONTMATTER_TITLE_KEY = "title"

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")
_KEY_VALUE = re.compile(r"^([A-Za-z_][\w-]*)\s*:\s*(.*)$")
_LIST_ITEM = re.compile(r"^\s+-\s+(.*)$")


@dataclass(frozen=True, slots=True)
class Frontmatter:
    fields: dict[str, str]
    lists: dict[str, list[str]]

    def tags(self) -> list[str]:
        values: list[str] = []
        for key in FRONTMATTER_TAG_KEYS:
            values.extend(self.lists.get(key, []))
            inline = self.fields.get(key, "")
            if inline:
                values.extend(_inline_list(inline))
        return values


def _unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


def _inline_list(value: str) -> list[str]:
    value = value.strip()
    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1]
    separator = "," if "," in value else " "
    return [_unquote(item) for item in value.split(separator) if _unquote(item)]


def split_frontmatter(text: str) -> tuple[Frontmatter, str]:
    """The flat key/value and list fields of a leading YAML block, and the body after it."""
    empty = Frontmatter({}, {})
    lines = text.splitlines()
    if not lines or lines[0].strip() != FRONTMATTER_DELIMITER:
        return empty, text
    for end in range(1, len(lines)):
        if lines[end].strip() in (FRONTMATTER_DELIMITER, "..."):
            break
    else:
        return empty, text
    fields: dict[str, str] = {}
    lists: dict[str, list[str]] = {}
    current: str | None = None
    for line in lines[1:end]:
        item = _LIST_ITEM.match(line)
        if item and current is not None:
            lists.setdefault(current, []).append(_unquote(item.group(1)))
            continue
        pair = _KEY_VALUE.match(line)
        if pair:
            current = pair.group(1).lower()
            value = pair.group(2).strip()
            if value:
                fields[current] = _unquote(value)
    return Frontmatter(fields, lists), "\n".join(lines[end + 1:])


@dataclass(frozen=True, slots=True)
class Section:
    heading_path: tuple[str, ...]
    text: str


def split_sections(body: str) -> list[Section]:
    """Sections at heading levels 1..SPLIT_HEADING_MAX_LEVEL; text before the first heading is its own section."""
    sections: list[Section] = []
    stack: list[tuple[int, str]] = []
    buffer: list[str] = []
    in_fence = False

    def flush() -> None:
        text = "\n".join(buffer).strip()
        if text:
            sections.append(Section(tuple(title for _, title in stack), text))
        buffer.clear()

    for line in body.splitlines():
        if _FENCE.match(line):
            in_fence = not in_fence
        heading = None if in_fence else _HEADING.match(line)
        if heading and len(heading.group(1)) <= SPLIT_HEADING_MAX_LEVEL:
            flush()
            level = len(heading.group(1))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, heading.group(2).strip()))
            buffer.append(line)
            continue
        buffer.append(line)
    flush()
    return sections


def _note_files(root: Path, stats: ImportStats) -> Iterator[Path]:
    for directory, subdirectories, files in os.walk(root, followlinks=False):
        subdirectories[:] = sorted(name for name in subdirectories
                                   if not name.startswith(".") and not (Path(directory) / name).is_symlink())
        for name in sorted(files):
            path = Path(directory) / name
            if path.suffix.lower() not in MARKDOWN_SUFFIXES or name.startswith("."):
                continue
            if path.is_symlink() or not path.is_file():
                stats.skipped_files += 1
                stats.warnings.append(f"skipped non-regular file {path.relative_to(root).as_posix()}")
                continue
            yield path


def _note_time(frontmatter: Frontmatter, path: Path) -> datetime:
    for key in FRONTMATTER_TIME_KEYS:
        moment = parse_timestamp(frontmatter.fields.get(key))
        if moment is not None:
            return moment
    return datetime.fromtimestamp(path.stat().st_mtime, UTC)


def _first_heading(body: str) -> str | None:
    for line in body.splitlines():
        match = _HEADING.match(line)
        if match and len(match.group(1)) == 1:
            return match.group(2).strip()
    return None


def markdown_records(path: Path, stats: ImportStats) -> Iterator[ImportRecord]:
    if path.is_file():
        root, files = path.parent, iter([path])
    elif path.is_dir():
        root, files = path, _note_files(path, stats)
    else:
        raise ImportFormatError(f"{path} is neither a directory nor a Markdown file")
    for note in files:
        relative = note.relative_to(root).as_posix()
        size = note.stat().st_size
        if size > MAX_NOTE_BYTES:
            stats.skipped_files += 1
            stats.warnings.append(f"skipped {relative}: {size} bytes exceeds {MAX_NOTE_BYTES}")
            continue
        try:
            raw = note.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            stats.skipped_files += 1
            stats.warnings.append(f"skipped {relative}: not UTF-8 text")
            continue
        frontmatter, body = split_frontmatter(raw)
        title = frontmatter.fields.get(FRONTMATTER_TITLE_KEY) or _first_heading(body) or note.stem
        tags = ("markdown", *(tag for tag in (normalise_tag(value) for value in frontmatter.tags()) if tag))
        occurred = _note_time(frontmatter, note)
        body = body.strip()
        if len(body) <= SECTION_SPLIT_CHARS:
            stats.units_seen += 1
            if not body:
                stats.skipped_empty += 1
                continue
            yield ImportRecord(text=body, occurred_at=occurred, source_ref=f"markdown:{relative}",
                               context=f'Markdown note "{title}" ({relative})', tags=tags)
            continue
        for index, section in enumerate(split_sections(body)):
            stats.units_seen += 1
            trail = " > ".join(section.heading_path)
            yield ImportRecord(
                text=section.text, occurred_at=occurred, source_ref=f"markdown:{relative}#{index}",
                context=f'Markdown note "{title}" ({relative})' + (f", section {trail}" if trail else ""),
                tags=tags,
            )
