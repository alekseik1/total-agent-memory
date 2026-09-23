"""Shared pieces of the QA benchmarks: an OpenAI client with a spend cap, and a result cache."""
from __future__ import annotations

import hashlib
import json
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

API_URL = "https://api.openai.com/v1/chat/completions"
PRICES = {  # USD per 1M tokens: input, output
    "gpt-4.1-mini-2025-04-14": (0.40, 1.60),
    "gpt-4.1-2025-04-14": (2.00, 8.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o-2024-08-06": (2.50, 10.00),
    "gpt-5-2025-08-07": (1.25, 10.00),
}
# Reasoning models take max_completion_tokens (reasoning included) and only the default temperature.
REASONING_MODEL_PREFIXES = ("gpt-5", "o1", "o3", "o4")
REASONING_MODEL_MAX_TOKENS = 4096
RETRIES = 5


class BudgetExceeded(RuntimeError):
    pass


class OpenAIClient:
    """Chat completions with retries and a hard spend cap shared across threads."""

    def __init__(self, api_key: str, budget_usd: float):
        self.api_key = api_key
        self.budget = budget_usd
        self.spent = 0.0
        self.lock = threading.Lock()

    def complete(self, model: str, messages: list[dict], *, max_tokens: int, json_mode: bool = False) -> tuple[str, dict]:
        if model.startswith(REASONING_MODEL_PREFIXES):
            body = {"model": model, "messages": messages,
                    "max_completion_tokens": max(max_tokens, REASONING_MODEL_MAX_TOKENS)}
        else:
            body = {"model": model, "messages": messages, "temperature": 0, "max_tokens": max_tokens}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        message, usage = self.chat(model, body)
        if not message.get("content") and "max_completion_tokens" in body:
            # The reasoning used the whole budget and left no answer; give it twice the room once.
            body["max_completion_tokens"] *= 2
            message, usage = self.chat(model, body)
        return message.get("content") or "", usage

    def chat(self, model: str, body: dict) -> tuple[dict, dict]:
        """One chat completion request; returns the assistant message and the usage."""
        with self.lock:
            if self.spent >= self.budget:
                raise BudgetExceeded(f"spent ${self.spent:.2f} of ${self.budget:.2f}")
        data = json.dumps(body).encode()
        delay = 2.0
        for attempt in range(RETRIES):
            request = urllib.request.Request(API_URL, data=data, headers={
                "Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    payload = json.loads(response.read())
                break
            except urllib.error.HTTPError as exc:
                if exc.code not in (408, 409, 429, 500, 502, 503, 504) or attempt == RETRIES - 1:
                    raise RuntimeError(f"{model} HTTP {exc.code}: {exc.read()[:300]!r}") from exc
            except OSError as exc:  # URLError, ssl.SSLError, timeouts and resets
                if attempt == RETRIES - 1:
                    raise RuntimeError(f"{model} unreachable: {exc}") from exc
            time.sleep(delay)
            delay *= 2
        usage = payload.get("usage", {})
        price_in, price_out = PRICES[model]
        cost = (usage.get("prompt_tokens", 0) * price_in + usage.get("completion_tokens", 0) * price_out) / 1e6
        with self.lock:
            self.spent += cost
        return payload["choices"][0]["message"], usage


class Cache:
    """Append-only JSONL keyed by content hash."""

    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        self.rows: dict[str, dict] = {}
        if path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    self.rows[row["key"]] = row

    def get(self, key: str) -> dict | None:
        return self.rows.get(key)

    def put(self, key: str, value: dict) -> None:
        row = {"key": key, **value}
        with self.lock:
            self.rows[key] = row
            with self.path.open("a") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def digest(*parts: object) -> str:
    return hashlib.sha256(json.dumps(parts, ensure_ascii=False, sort_keys=True).encode()).hexdigest()

