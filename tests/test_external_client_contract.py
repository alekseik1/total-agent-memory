"""Contract for external clients that drive TAM over MCP stdio.

The MemoryAgentBench method (`methods/total_agent_memory.py` upstream) spawns
the server with a throwaway TAM_MEMORY_DIR, performs the 2025-06-18 handshake,
saves facts with `memory_save` and reads them back with `memory_recall`. It
relies on:

* stdout carrying JSON-RPC lines only — any other line fails its run;
* `memory_save` answering with `saved` and `deduplicated`;
* `memory_recall` answering `{"results": {type: [hit, ...]}}` in rank order,
  each hit with the full `content` and a `created_at` that grows with save order;
* an update that differs in one value being stored, not deduplicated.

A change to any of these breaks that adapter; this test fails first.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "src" / "server.py"
PASSED_ENV = ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "HF_HOME", "FASTEMBED_CACHE_PATH")

FACTS = [
    "The chief executive officer of Microsoft is Satya Nadella",
    "The capital of Tang Empire is Chang'an",
    "The chief executive officer of Microsoft is Steve Jobs",
]


class StdioClient:
    def __init__(self, memory_dir: Path):
        env = {name: os.environ[name] for name in PASSED_ENV if name in os.environ}
        env.update(TAM_MEMORY_DIR=str(memory_dir), MCP_TRANSPORT="stdio",
                   MEMORY_MODE="fast", MEMORY_LLM_ENABLED="false")
        self.proc = subprocess.Popen(
            [sys.executable, str(SERVER)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, env=env, cwd=str(ROOT), text=True, bufsize=1,
        )
        self.next_id = 0

    def request(self, method: str, params: dict) -> dict:
        self.next_id += 1
        self.proc.stdin.write(json.dumps({"jsonrpc": "2.0", "id": self.next_id,
                                          "method": method, "params": params}) + "\n")
        self.proc.stdin.flush()
        while True:
            line = self.proc.stdout.readline()
            assert line, "server closed stdout"
            message = json.loads(line)  # a non-JSON line on stdout fails here
            if "id" not in message:
                continue
            assert message["id"] == self.next_id
            assert "error" not in message, message["error"]
            return message["result"]

    def notify(self, method: str) -> None:
        self.proc.stdin.write(json.dumps({"jsonrpc": "2.0", "method": method}) + "\n")
        self.proc.stdin.flush()

    def call(self, tool: str, arguments: dict) -> dict:
        result = self.request("tools/call", {"name": tool, "arguments": arguments})
        assert result.get("isError") is not True, result
        return json.loads("".join(part["text"] for part in result["content"]))

    def close(self) -> None:
        self.proc.stdin.close()
        self.proc.terminate()
        self.proc.wait(timeout=30)


def test_memoryagentbench_adapter_contract(tmp_path):
    client = StdioClient(tmp_path)
    try:
        init = client.request("initialize", {
            "protocolVersion": "2025-06-18", "capabilities": {},
            "clientInfo": {"name": "memoryagentbench", "version": "1"},
        })
        assert init["protocolVersion"]
        client.notify("notifications/initialized")

        saved = [client.call("memory_save", {"content": fact, "type": "fact", "project": "mab"})
                 for fact in FACTS]
        assert all(s["saved"] is True and s["deduplicated"] is False for s in saved)

        found = client.call("memory_recall", {
            "query": "Who is the chief executive officer of Microsoft?",
            "project": "mab", "limit": 10, "detail": "full",
        })
        hits = [hit for group in found["results"].values() for hit in group]
        by_content = {hit["content"]: hit for hit in hits}
        assert FACTS[0] in by_content and FACTS[2] in by_content
        assert by_content[FACTS[0]]["created_at"] < by_content[FACTS[2]]["created_at"]
    finally:
        client.close()
