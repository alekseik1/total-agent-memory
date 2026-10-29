"""Shared plumbing for the standalone tools and the reflection scheduler.

* `make_ssl_context` - TLS client context that always verifies the peer.
* `ollama_generate` - run a prompt through the Ollama CLI. The prompt travels
  on stdin, never in argv (argv is visible to every user through `ps`, and is
  capped by ARG_MAX). The binary comes from the environment or PATH, the model
  from configuration.
"""
from __future__ import annotations

import os
import re
import shutil
import ssl
import subprocess
import sys
from collections.abc import Callable

OLLAMA_BIN_ENV = "MEMORY_OLLAMA_BIN"
OLLAMA_MODEL_ENV = "MEMORY_TOOLS_OLLAMA_MODEL"
# Where package managers put the binary when PATH is minimal (launchd, cron, systemd).
OLLAMA_FALLBACK_PATHS = ("/opt/homebrew/bin/ollama", "/usr/local/bin/ollama", "/usr/bin/ollama")
DEFAULT_TIMEOUT_SEC = 120

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]|\x1b\].*?\x07")

_default_log: Callable[[str], None] = lambda msg: sys.stderr.write(f"[tool-support] {msg}\n")


def strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences (colours, cursor moves)."""
    return _ANSI_RE.sub("", text)


def make_ssl_context() -> ssl.SSLContext:
    """Verified TLS context: certifi, else the platform store, never unverified."""
    from tls_context import verified_context

    return verified_context()

def ollama_binary() -> str | None:
    """Locate the ollama executable: env override, PATH, then well-known install paths."""
    explicit = os.environ.get(OLLAMA_BIN_ENV)
    if explicit:
        return explicit if os.access(explicit, os.X_OK) else None
    found = shutil.which("ollama")
    if found:
        return found
    return next((path for path in OLLAMA_FALLBACK_PATHS if os.access(path, os.X_OK)), None)


def default_model() -> str:
    """Model for CLI generation: explicit env override, else the configured LLM model."""
    override = os.environ.get(OLLAMA_MODEL_ENV)
    if override:
        return override
    import config

    return config.get_llm_model()


def ollama_generate(
    prompt: str,
    model: str | None = None,
    timeout: float = DEFAULT_TIMEOUT_SEC,
    *,
    log: Callable[[str], None] = _default_log,
    postprocess: Callable[[str], str] | None = None,
) -> str:
    """Run `ollama run <model>` with the prompt on stdin. Returns text, or "" after logging a failure."""
    binary = ollama_binary()
    if binary is None:
        log("Ollama binary not found (set MEMORY_OLLAMA_BIN or add ollama to PATH)")
        return ""
    env = os.environ.copy()
    env["NO_COLOR"] = "1"
    try:
        result = subprocess.run(  # noqa: S603 - fixed argv, prompt goes via stdin
            [binary, "run", model or default_model()],
            input=prompt, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout, env=env, check=False,
        )
    except subprocess.TimeoutExpired:
        log(f"Ollama timeout after {timeout}s")
        return ""
    except OSError as error:
        log(f"Ollama error: {error}")
        return ""
    if result.returncode != 0:
        log(f"Ollama exited with {result.returncode}: {result.stderr.strip()[:200]}")
    text = strip_ansi(result.stdout).strip()
    return postprocess(text).strip() if postprocess else text
