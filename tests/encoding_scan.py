"""AST scan for text-mode file access that relies on the locale encoding.

Without ``encoding=`` Python decodes with the locale codec: cp1252 on a Windows
machine that has not enabled UTF-8 mode, UTF-8 on macOS and Linux. Code that
reads a UTF-8 file that way works on a developer laptop and raises
``UnicodeDecodeError`` for a Windows user, so every text-mode access names its
codec. The scan is syntactic: it finds the call, not what is behind the name.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

MODE_CHARS = re.compile(r"^[rwaxt+U]+$")
# ``.open()`` without arguments is a path only when the receiver reads like one.
PATH_RECEIVER = re.compile(r"(?i)path|file|\blog\b|\.result$|\.diff$")
# Receivers whose ``open`` takes a member name or opens something that is not a text file.
NON_TEXT_RECEIVERS = frozenset(
    {"os", "gzip", "bz2", "lzma", "zipfile", "tarfile", "webbrowser", "Image", "tempfile",
     "zf", "zip", "zipf", "tar", "tf", "archive", "sessions", "SESSIONS", "targets", "workdir",
     "self.targets", "setup.targets"}
)
TEMP_FILE_FACTORIES = frozenset({"NamedTemporaryFile", "TemporaryFile", "SpooledTemporaryFile"})


@dataclass(frozen=True)
class Finding:
    """A text-mode call without an explicit encoding, with its source span."""

    path: Path
    node: ast.Call
    kind: str

    @property
    def where(self) -> str:
        return f"{self.path}:{self.node.lineno}: {self.kind}"


def _keyword(call: ast.Call, name: str) -> ast.keyword | None:
    return next((kw for kw in call.keywords if kw.arg == name), None)


def _delegates_kwargs(call: ast.Call) -> bool:
    return any(kw.arg is None for kw in call.keywords)


def _names_encoding(call: ast.Call, positional: int) -> bool:
    return _keyword(call, "encoding") is not None or len(call.args) > positional or _delegates_kwargs(call)


def _mode(call: ast.Call, position: int, default: str) -> str | None:
    """The literal mode, ``default`` when absent, ``None`` when it is not a literal."""
    node: ast.expr | None = call.args[position] if len(call.args) > position else None
    keyword = _keyword(call, "mode")
    if keyword is not None:
        node = keyword.value
    if node is None:
        return default
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _is_text(mode: str | None) -> bool:
    return mode is None or "b" not in mode


def _classify(call: ast.Call) -> str | None:
    func = call.func
    if isinstance(func, ast.Attribute):
        if func.attr in ("read_text", "write_text"):
            position = 0 if func.attr == "read_text" else 1
            return None if _names_encoding(call, position) else func.attr
        receiver = ast.unparse(func.value)
        if func.attr == "fdopen" and receiver == "os":
            mode = _mode(call, 1, "r")
            return None if not _is_text(mode) or _names_encoding(call, 3) else "os.fdopen"
        if func.attr == "open" and receiver == "io":
            mode = _mode(call, 1, "r")
            return None if not _is_text(mode) or _names_encoding(call, 3) else "io.open"
        if func.attr == "open" and receiver not in NON_TEXT_RECEIVERS:
            mode = _mode(call, 0, "")
            if mode is None or mode == "":
                bare = not call.args and not call.keywords and PATH_RECEIVER.search(receiver)
                return "Path.open" if bare else None
            if not MODE_CHARS.match(mode) or "b" in mode or _names_encoding(call, 2):
                return None
            return "Path.open"
        if func.attr in TEMP_FILE_FACTORIES:
            mode = _mode(call, 0, "w+b")
            return None if not _is_text(mode) or _names_encoding(call, 3) else func.attr
        return None
    if isinstance(func, ast.Name):
        if func.id == "open":
            mode = _mode(call, 1, "r")
            return None if not _is_text(mode) or _names_encoding(call, 3) else "open"
        if func.id in TEMP_FILE_FACTORIES:
            mode = _mode(call, 0, "w+b")
            return None if not _is_text(mode) or _names_encoding(call, 3) else func.id
    return None


def scan_source(path: Path, source: str) -> Iterator[Finding]:
    """Yield every text-mode call in ``source`` that leaves the codec to the locale."""
    for node in ast.walk(ast.parse(source, filename=str(path))):
        if isinstance(node, ast.Call):
            kind = _classify(node)
            if kind is not None:
                yield Finding(path, node, kind)


def scan_tree(root: Path, subdirs: tuple[str, ...]) -> list[Finding]:
    """Scan every ``*.py`` file below ``root/<subdir>`` for each subdir that exists."""
    found: list[Finding] = []
    for subdir in subdirs:
        base = root / subdir
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            found.extend(scan_source(path, path.read_text(encoding="utf-8")))
    return found
