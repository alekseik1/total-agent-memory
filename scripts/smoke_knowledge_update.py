#!/usr/bin/env python3
"""End-to-end check that memory_answer returns the latest value of a changed fact.

Starts the installed `total-agent-memory` MCP server over stdio against a
throwaway TAM_MEMORY_DIR (live memory is never touched), saves an old and a
newer fact in Russian and English, and asks memory_answer. The LLM provider is
taken from the environment, falling back to the reflection launchd service.

    ~/.tam/.venv/bin/python scripts/smoke_knowledge_update.py

Exit code 0 only when both answers name green, the newer value, as the colour.
"""

from __future__ import annotations

import asyncio
import json
import os
import plistlib
import re
import sys
import tempfile
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVER = Path(os.environ.get('TAM_SERVER_BIN', Path.home() / '.tam/.venv/bin/total-agent-memory'))
SERVICE_PLIST = Path.home() / 'Library/LaunchAgents/com.total-agent-memory.reflection.plist'
PROVIDER_VARS = ('MEMORY_LLM_PROVIDER', 'MEMORY_LLM_MODEL', 'MEMORY_LLM_API_KEY')
# Two records must not share a recording instant, or "newer" is undefined.
SAVE_GAP_SECONDS = 1.1
CASES = (
    ('smoke_ru', ('Маша любит красный цвет.', 'Маше больше не нравится красный — она полюбила зелёный.'),
     'Какой цвет любит Маша?', re.compile(r'зел[её]н', re.IGNORECASE)),
    ('smoke_en', ('Mary loves the color red.', 'Mary no longer likes red; she has fallen for green.'),
     'What color does Mary love?', re.compile(r'green', re.IGNORECASE)),
)


def provider_env() -> dict[str, str]:
    values = {name: os.environ[name] for name in PROVIDER_VARS if os.environ.get(name)}
    if len(values) < len(PROVIDER_VARS) and SERVICE_PLIST.exists():
        service = plistlib.loads(SERVICE_PLIST.read_bytes()).get('EnvironmentVariables', {})
        values = {**{name: service[name] for name in PROVIDER_VARS if service.get(name)}, **values}
    missing = [name for name in PROVIDER_VARS if name not in values]
    if missing:
        raise SystemExit(f'Set {", ".join(missing)}: memory_answer needs an LLM provider')
    return values


async def run() -> bool:
    env = {**os.environ, **provider_env(), 'TAM_MEMORY_DIR': tempfile.mkdtemp(prefix='tam-smoke-'), 'MEMORY_QUIET': '1'}
    passed = True
    async with stdio_client(StdioServerParameters(command=str(SERVER), env=env)) as (read, write), \
            ClientSession(read, write) as session:
        info = await session.initialize()
        print(f'server {info.server_info.name} {info.server_info.version}, database {env["TAM_MEMORY_DIR"]}')
        for project, facts, question, expected in CASES:
            for fact in facts:
                await session.call_tool('memory_save', {'content': fact, 'type': 'fact', 'project': project})
                await asyncio.sleep(SAVE_GAP_SECONDS)
            response = await session.call_tool('memory_answer', {'query': question, 'project': project})
            result = json.loads(response.content[0].text)
            answer = result['answer'].split('\n\nCaveat:')[0]
            ok = bool(expected.search(answer))
            passed &= ok
            negative = result.get('negative') or {}
            print(f'\n{"PASS" if ok else "FAIL"}  {question}\n  answer:   {answer}')
            print(f'  contradiction check: {negative.get("decision")} at {negative.get("contradiction_score")} '
                  f'(scorer: {os.environ.get("MEMORY_CONTRADICTION_SCORER") or "llm"})')
            for hit in result['evidence']:
                print(f'  recorded: {hit.get("created_at")}  {hit["content"]}')
    return passed


if __name__ == '__main__':
    sys.exit(0 if asyncio.run(run()) else 1)
