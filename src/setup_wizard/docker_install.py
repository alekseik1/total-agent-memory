"""Claude Code registration for the Docker install: the MCP entry and the Docker-mode hook launchers.

Never destroys user configuration: an unparsable settings file stops the run before anything is written,
every file is backed up (timestamped ``.bak``) before it is replaced, hook entries are only added (entries of
a previous run of this installer are refreshed in place), and the MCP server goes to ``~/.claude.json``, the
file Claude Code reads user-scope servers from, with the access token the container requires.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from setup_wizard.clients import ConfigError, read_json
from setup_wizard.extras import POST_TOOL_USE_MATCHER, PROVENANCE_MATCHER, write_provenance_marker
from setup_wizard.files import PRIVATE_MODE, atomic_write

SERVER_NAME = "memory"
LEGACY_URL = "http://127.0.0.1:3737/mcp"
BACKUP_SUFFIX = ".bak-{stamp}"
BACKUP_STAMP_FORMAT = "%Y%m%d-%H%M%S"
EXIT_CONFIG_ERROR = 2

DOCKER_HOOKS: tuple[tuple[str, str, str], ...] = (
    ("SessionStart", "", "session-start.sh"),
    ("SessionEnd", "", "session-end.sh"),
    ("Stop", "", "on-stop.sh"),
    ("UserPromptSubmit", "", "user-prompt-submit.sh"),
    ("PreToolUse", "Bash|Edit|Write|MultiEdit|NotebookEdit", "rule-reminder.sh"),
    ("PreToolUse", PROVENANCE_MATCHER, "inject-provenance.sh"),
    ("PostToolUse", "Bash", "memory-trigger.sh"),
    ("PostToolUse", "Write|Edit", "auto-capture.sh"),
    ("PostToolUse", POST_TOOL_USE_MATCHER, "post-tool-use.sh"),
    # Claude Code reports a failed call through PostToolUseFailure, not PostToolUse.
    ("PostToolUseFailure", "Bash", "memory-trigger.sh"),
    ("PostToolUseFailure", POST_TOOL_USE_MATCHER, "post-tool-use.sh"),
)


def backup(path: Path) -> Path | None:
    """Copy ``path`` next to itself with a timestamp; None when there is nothing to keep."""
    if not path.is_file():
        return None
    target = path.with_name(path.name + BACKUP_SUFFIX.format(stamp=time.strftime(BACKUP_STAMP_FORMAT)))
    shutil.copy2(path, target)
    return target


def _is_ours(command: object, launcher_dir: Path) -> bool:
    return isinstance(command, str) and command.startswith(str(launcher_dir))


def merge_hooks(settings: dict, launcher_dir: Path) -> list[str]:
    """Add the launcher hooks to ``settings`` and return the events touched.

    Entries pointing into ``launcher_dir`` (an earlier run) are dropped first so a re-run refreshes matchers;
    every other hook, whatever its event, stays exactly as it was.
    """
    hooks = settings.setdefault("hooks", {})
    if not isinstance(hooks, dict):
        raise ConfigError("settings.json: 'hooks' must be an object")
    for event, blocks in list(hooks.items()):
        if not isinstance(blocks, list):
            continue
        for block in blocks:
            if isinstance(block, dict) and isinstance(block.get("hooks"), list):
                block["hooks"] = [h for h in block["hooks"]
                                  if not (isinstance(h, dict) and _is_ours(h.get("command"), launcher_dir))]
        hooks[event] = [b for b in blocks if not (isinstance(b, dict) and b.get("hooks") == [])]
    touched: list[str] = []
    for event, matcher, script in DOCKER_HOOKS:
        entries = hooks.setdefault(event, [])
        if not isinstance(entries, list):
            raise ConfigError(f"settings.json: hooks.{event} must be a list")
        entries.append({"matcher": matcher,
                        "hooks": [{"type": "command", "command": str(launcher_dir / script)}]})
        if event not in touched:
            touched.append(event)
    return touched


def drop_legacy_mcp(settings: dict) -> bool:
    """Remove the ``mcpServers.memory`` entry old versions of this installer wrote where Claude Code ignores it."""
    servers = settings.get("mcpServers")
    entry = servers.get(SERVER_NAME) if isinstance(servers, dict) else None
    if not (isinstance(entry, dict) and entry.get("url") == LEGACY_URL):
        return False
    del servers[SERVER_NAME]
    if not servers:
        del settings["mcpServers"]
    return True


def http_entry(url: str, token: str) -> dict:
    entry: dict = {"type": "http", "url": url}
    if token:
        entry["headers"] = {"Authorization": f"Bearer {token}"}
    return entry


def install(home: Path, launcher_dir: Path, url: str, token: str) -> dict[str, object]:
    """Write both Claude Code files; every parse happens before the first write."""
    settings_path = home / ".claude" / "settings.json"
    claude_json = home / ".claude.json"
    settings = read_json(settings_path)
    user_config = read_json(claude_json)
    touched = merge_hooks(settings, launcher_dir)
    dropped_legacy = drop_legacy_mcp(settings)
    servers = user_config.setdefault("mcpServers", {})
    if not isinstance(servers, dict):
        raise ConfigError(f"{claude_json}: 'mcpServers' must be an object")
    servers[SERVER_NAME] = http_entry(url, token)
    backups = [b for b in (backup(settings_path), backup(claude_json)) if b is not None]
    atomic_write(settings_path, json.dumps(settings, indent=2, ensure_ascii=False) + "\n")
    atomic_write(claude_json, json.dumps(user_config, indent=2, ensure_ascii=False) + "\n", PRIVATE_MODE)
    write_provenance_marker(settings_path.parent, url)
    return {"events": touched, "legacy_removed": dropped_legacy, "backups": [str(b) for b in backups]}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="docker_install", description=__doc__)
    parser.add_argument("--home", type=Path, default=Path.home())
    parser.add_argument("--launcher-dir", type=Path, required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--token-stdin", action="store_true", help="read the access token from stdin")
    args = parser.parse_args(argv)
    token = sys.stdin.read().strip() if args.token_stdin else ""
    try:
        result = install(args.home, args.launcher_dir, args.url, token)
    except ConfigError as error:
        sys.stderr.write(f"  ERROR: {error}\n  Nothing was changed.\n")
        return EXIT_CONFIG_ERROR
    for backup_path in result["backups"]:
        sys.stdout.write(f"  Backup: {backup_path}\n")
    sys.stdout.write(f"  OK: MCP '{SERVER_NAME}' -> {args.url} in {args.home / '.claude.json'}\n")
    sys.stdout.write(f"  OK: hooks added for {', '.join(result['events'])}\n")
    if result["legacy_removed"]:
        sys.stdout.write("  OK: removed the old mcpServers entry from settings.json (Claude Code ignores it)\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
