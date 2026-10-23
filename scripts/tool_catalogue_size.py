#!/usr/bin/env python3
"""Size of the tools/list catalogue under each TAM_TOOL_PROFILE.

Prints one line per profile: tool count, compact-JSON bytes and a rough token
estimate (bytes / 4). Run from the repository root:

    PYTHONPATH=src python scripts/tool_catalogue_size.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

BYTES_PER_TOKEN = 4


def catalogue_bytes(tools: list) -> int:
    """Compact JSON of the catalogue as a client receives it in tools/list."""
    payload = [tool.model_dump(mode="json", by_alias=True, exclude_none=True) for tool in tools]
    return len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def measure() -> dict[str, tuple[int, int]]:
    import server
    import tool_profiles

    sizes = {}
    previous = os.environ.get(tool_profiles.PROFILE_ENV)
    try:
        for profile in tool_profiles.PROFILE_NAMES:
            os.environ[tool_profiles.PROFILE_ENV] = profile
            tools = asyncio.run(server.list_tools())
            sizes[profile] = (len(tools), catalogue_bytes(tools))
    finally:
        if previous is None:
            os.environ.pop(tool_profiles.PROFILE_ENV, None)
        else:
            os.environ[tool_profiles.PROFILE_ENV] = previous
    return sizes


def main() -> int:
    for profile, (count, size) in measure().items():
        sys.stdout.write(f"{profile:8} {count:3} tools {size:7} bytes ~{size // BYTES_PER_TOKEN} tokens\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
