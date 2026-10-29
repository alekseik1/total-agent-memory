"""End-to-end stdio round trip through the installed MCP SDK (1.x and 2.x).

Spawns src/server.py as a real MCP server, lists the catalogue and calls tools over stdio.
It loads the embedding model, so it runs only when TAM_STDIO_SMOKE=1 (the CI job
`mcp-floor` sets it after prewarming the models and installing the oldest supported mcp).
"""

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest

from version import VERSION

ROOT = Path(__file__).resolve().parent.parent
EXPECTED_TOOLS = 79
SMOKE_ENV = "TAM_STDIO_SMOKE"
CALL_TIMEOUT_SEC = 120

pytestmark = pytest.mark.skipif(os.environ.get(SMOKE_ENV) != "1", reason=f"set {SMOKE_ENV}=1 to run")


def _attr(obj, *names):
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    raise AttributeError(f"{type(obj).__name__} has none of {names}")


async def _round_trip(memory_dir: Path) -> dict:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    env = {**os.environ, "TAM_MEMORY_DIR": str(memory_dir), "MEMORY_DIR": str(memory_dir),
           "PYTHONPATH": str(ROOT / "src"), "MEMORY_LLM_ENABLED": "false",
           "MEMORY_ASYNC_ENRICHMENT": "false", "MCP_TRANSPORT": "stdio"}
    params = StdioServerParameters(command=sys.executable, args=[str(ROOT / "src" / "server.py")], env=env)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            init = await session.initialize()
            tools = await session.list_tools()
            saved = await session.call_tool(
                "memory_save", {"content": "stdio smoke record about sdk floor", "type": "fact", "project": "smoke"})
            invalid = await session.call_tool("memory_save", {"type": "fact"})
            recalled = await session.call_tool("memory_recall", {"query": "sdk floor", "project": "smoke"})
            return {
                "version": _attr(init, "serverInfo", "server_info").version,
                "tools": len(tools.tools),
                "saved": json.loads(saved.content[0].text),
                "invalid": invalid.content[0].text,
                "invalid_is_error": bool(_attr(invalid, "isError", "is_error")),
                "recalled": recalled.content[0].text,
            }


def test_stdio_round_trip_lists_the_catalogue_and_saves_and_recalls(tmp_path):
    result = asyncio.run(asyncio.wait_for(_round_trip(tmp_path / "mem"), CALL_TIMEOUT_SEC))

    assert result["version"] == VERSION
    assert result["tools"] == EXPECTED_TOOLS
    assert result["saved"]["saved"] is True
    assert "content" in result["invalid"]
    assert result["invalid_is_error"] is True
    assert "stdio smoke record about sdk floor" in result["recalled"]
