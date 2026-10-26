#!/usr/bin/env python3
"""Fill install-time placeholders in a service template, escaping each value for the file format.

    render_template.py --mode systemd --var @INSTALL_DIR@=/opt/my&dir < unit.service > out

sed cannot do this safely: an ``&`` in a replacement expands to the matched text, ``|`` ends the
expression, and a plist value needs XML escaping. Values are inserted literally.

Modes:
  systemd  backslash, double quote and ``%`` escaped (templates quote the arguments)
  xml      ``&``, ``<`` and ``>`` escaped (launchd plists)
"""
from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence

EXIT_USAGE = 2


def _systemd(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%")


def _xml(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


ESCAPES: dict[str, Callable[[str], str]] = {"systemd": _systemd, "xml": _xml}


def render(template: str, variables: dict[str, str], mode: str) -> str:
    escape = ESCAPES[mode]
    for placeholder, value in variables.items():
        template = template.replace(placeholder, escape(value))
    return template


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=sorted(ESCAPES), required=True)
    parser.add_argument("--var", action="append", default=[], metavar="PLACEHOLDER=VALUE")
    args = parser.parse_args(argv)
    variables: dict[str, str] = {}
    for pair in args.var:
        placeholder, separator, value = pair.partition("=")
        if not separator or not placeholder:
            sys.stderr.write(f"--var {pair!r}: expected PLACEHOLDER=VALUE\n")
            return EXIT_USAGE
        variables[placeholder] = value
    sys.stdout.write(render(sys.stdin.read(), variables, args.mode))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
