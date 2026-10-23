"""``python -m ambient <command> [--memory-dir DIR]``: one ambient hook command.

Reads a :class:`ambient.hooks.HookCall` JSON on stdin and prints the command's
text (possibly empty) on stdout. This is how hooks on a Docker host reach the
store inside the TAM container (``docker exec -i <container> python -m ambient``).
Failures are logged on stderr and exit 1; the hook on the host then emits nothing.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from ambient.hooks import COMMANDS, HookCall, run

EXIT_OK = 0
EXIT_FAILURE = 1
LOG_FORMAT = "[%(asctime)s] [tam-ambient] %(name)s %(levelname)s %(message)s"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m ambient", description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=sorted(COMMANDS))
    parser.add_argument("--memory-dir", type=Path, default=None, help="default: TAM_MEMORY_DIR / ~/.tam")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, stream=sys.stderr)
    if args.memory_dir is None:
        from paths import memory_dir

        args.memory_dir = memory_dir()
    try:
        text = run(args.command, HookCall.from_json(sys.stdin.read()), args.memory_dir)
    except Exception as exc:  # noqa: BLE001 — process boundary of a hook command; reported with type
        sys.stderr.write(json.dumps({"event": "ambient_command_failed", "command": args.command,
                                     "error": f"{type(exc).__name__}: {exc}"}) + "\n")
        return EXIT_FAILURE
    sys.stdout.write(text)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
