"""Incremental reader for JSON exports whose payload is one (possibly huge) array.

ChatGPT and Claude exports are a single ``conversations.json`` array that can
run to hundreds of megabytes; mem0 exports are an array or an object holding
one. ``iter_json_array`` yields the array elements one at a time while holding
only the current element (plus one read chunk) in memory.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import IO, Any

from importers.records import ImportFormatError

READ_CHUNK_CHARS = 1 << 16
# Consumed text is dropped from the buffer once it exceeds this many characters.
COMPACT_AFTER_CHARS = 1 << 20
_WHITESPACE = " \t\n\r"
# A JSON literal or escape split at the buffer edge fails at most this far from the end.
TRUNCATION_MARGIN_CHARS = 8


class _Cursor:
    def __init__(self, handle: IO[str]) -> None:
        self._handle = handle
        self._decoder = json.JSONDecoder()
        self.buffer = ""
        self.pos = 0
        self.eof = False
        self._chunk = READ_CHUNK_CHARS

    def _fill(self) -> bool:
        if self.eof:
            return False
        if self.pos > COMPACT_AFTER_CHARS:
            self.buffer = self.buffer[self.pos:]
            self.pos = 0
        data = self._handle.read(self._chunk)
        if not data:
            self.eof = True
            return False
        self.buffer += data
        return True

    def skip_ws(self) -> None:
        while True:
            while self.pos < len(self.buffer) and self.buffer[self.pos] in _WHITESPACE:
                self.pos += 1
            if self.pos < len(self.buffer) or not self._fill():
                return

    def peek(self) -> str | None:
        self.skip_ws()
        return self.buffer[self.pos] if self.pos < len(self.buffer) else None

    def expect(self, char: str) -> None:
        found = self.peek()
        if found != char:
            raise ImportFormatError(f"malformed JSON: expected {char!r}, found {found!r} near offset {self.pos}")
        self.pos += 1

    def value(self) -> Any:
        self.skip_ws()
        self._chunk = READ_CHUNK_CHARS
        while True:
            try:
                result, end = self._decoder.raw_decode(self.buffer, self.pos)
            except json.JSONDecodeError as error:
                if _maybe_truncated(error, len(self.buffer)) and self._fill():
                    # Grow the read size so one large element costs linear, not quadratic, time.
                    self._chunk *= 2
                    continue
                raise ImportFormatError(f"malformed JSON: {error.msg} at offset {error.pos}") from error
            except RecursionError as error:
                raise ImportFormatError("malformed JSON: nested too deeply") from error
            # A number or literal ending exactly at the buffer edge may continue in the next chunk.
            if end == len(self.buffer) and not self.eof and self._fill():
                continue
            self.pos = end
            return result


def _maybe_truncated(error: json.JSONDecodeError, buffered: int) -> bool:
    """True when the decode failed because the element continues past the buffer.

    An unterminated string reports its start offset; every other truncation
    fails within a literal's length of the buffer end. Errors well inside the
    buffer are real syntax errors and are raised without reading further.
    """
    return error.msg.startswith("Unterminated string") or error.pos >= buffered - TRUNCATION_MARGIN_CHARS


def _elements(cursor: _Cursor) -> Iterator[Any]:
    cursor.expect("[")
    if cursor.peek() == "]":
        cursor.pos += 1
        return
    while True:
        yield cursor.value()
        separator = cursor.peek()
        if separator == ",":
            cursor.pos += 1
            continue
        if separator == "]":
            cursor.pos += 1
            return
        raise ImportFormatError(f"malformed JSON: expected ',' or ']', found {separator!r}")


def _iterate(cursor: _Cursor, container_keys: tuple[str, ...]) -> Iterator[Any]:
    first = cursor.peek()
    if first == "[":
        yield from _elements(cursor)
        if cursor.peek() is not None:
            raise ImportFormatError("malformed JSON: data after the top-level array")
        return
    if first != "{" or not container_keys:
        expected = "an array" + (f" or an object with {'/'.join(container_keys)}" if container_keys else "")
        raise ImportFormatError(f"expected {expected} at the top level, found {first!r}")
    cursor.expect("{")
    if cursor.peek() == "}":
        raise ImportFormatError(f"top-level object has no {'/'.join(container_keys)} array")
    while True:
        key = cursor.value()
        if not isinstance(key, str):
            raise ImportFormatError("malformed JSON: object key is not a string")
        cursor.expect(":")
        if key in container_keys and cursor.peek() == "[":
            yield from _elements(cursor)
            return
        cursor.value()
        separator = cursor.peek()
        if separator == ",":
            cursor.pos += 1
            continue
        if separator == "}":
            raise ImportFormatError(f"top-level object has no {'/'.join(container_keys)} array")
        raise ImportFormatError(f"malformed JSON: expected ',' or '}}', found {separator!r}")


def iter_json_array(path: Path, container_keys: tuple[str, ...] = ()) -> Iterator[Any]:
    """Yield the elements of the top-level array of ``path``.

    With ``container_keys`` a top-level object is accepted too; the first of
    those keys holding an array is streamed.
    """
    with path.open("r", encoding="utf-8-sig") as handle:
        yield from _iterate(_Cursor(handle), container_keys)
