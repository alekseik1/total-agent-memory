"""UTF-8 standard streams for processes that exchange JSON with a parent.

Hook commands and helper children receive UTF-8 JSON on stdin and answer on
stdout. Python decodes those pipes with the locale codec, which is cp1252 on a
Windows machine without UTF-8 mode, so non-ASCII text arrives as mojibake or
raises ``UnicodeDecodeError``.
"""

from __future__ import annotations

import sys

ENCODING = "utf-8"
ERROR_HANDLER = "replace"


def use_utf8_stdio() -> None:
    """Switch stdin, stdout and stderr to UTF-8 when the streams support it."""
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding=ENCODING, errors=ERROR_HANDLER)


CONSOLE_ERROR_HANDLER = "backslashreplace"


def never_fail_on_output() -> None:
    """Keep the console's encoding for human-facing CLI output, but escape what it cannot show.

    A Windows console or pipe without UTF-8 mode is cp1252/cp866: printing an arrow or a Cyrillic path raised
    ``UnicodeEncodeError`` half-way through a command that had already changed files.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None and (getattr(stream, "encoding", "") or "").lower().replace("-", "") != "utf8":
            reconfigure(errors=CONSOLE_ERROR_HANDLER)
