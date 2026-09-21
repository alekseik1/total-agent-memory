#!/usr/bin/env python3
"""Reflection daemon for Docker deployments.

Replaces the native LaunchAgent `com.claude.memory.reflection` (macOS).

Watches `$CLAUDE_MEMORY_DIR/.reflect-pending` for changes and drains the
reflection queues with debouncing. Also runs a safety-net reflection cycle
every `REFLECT_INTERVAL_SEC` seconds (default 1h).

Runs in the `reflection` compose service.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

MEMORY_DIR = Path(os.environ.get("CLAUDE_MEMORY_DIR", "/data"))
TRIGGER_FILE = MEMORY_DIR / ".reflect-pending"
DEBOUNCE_SEC = int(os.environ.get("REFLECT_DEBOUNCE_SEC", "5"))
INTERVAL_SEC = int(os.environ.get("REFLECT_INTERVAL_SEC", "3600"))
# Full digest+synthesize run, same 4x/day cadence as the macOS/systemd jobs
# (StartCalendarInterval / OnCalendar 02/08/14/20). Docker has no calendar
# scheduler, so this is interval-based: 6h apart covers the same cadence.
FULL_INTERVAL_SEC = int(os.environ.get("REFLECT_FULL_INTERVAL_SEC", "21600"))
SRC_DIR = Path(os.environ.get("CLAUDE_TOTAL_MEMORY_SRC", "/app/src"))


def _log(msg: str) -> None:
    sys.stderr.write(f"[reflection-daemon] {msg}\n")
    sys.stderr.flush()


def _run_reflection(scope: str = "auto") -> None:
    runner = SRC_DIR / "tools" / "run_reflection.py"
    if not runner.exists():
        _log(f"runner not found: {runner} — reflection skipped")
        return
    # --scope=full runs digest (merge_duplicates/decay/contradictions) +
    # synthesize and can legitimately take minutes; the 600s auto/drain
    # timeout would silently truncate it, so give full its own budget.
    timeout = 3600 if scope == "full" else 600
    try:
        subprocess.run(
            [sys.executable, str(runner), f"--scope={scope}"],
            check=False,
            cwd=str(SRC_DIR.parent),
            env={**os.environ, "PYTHONPATH": str(SRC_DIR)},
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        _log(f"reflection run (scope={scope}) timed out (>{timeout}s)")
    except Exception as e:
        _log(f"reflection run (scope={scope}) failed: {e}")


def main() -> None:
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    _log(
        f"watching {TRIGGER_FILE} (debounce={DEBOUNCE_SEC}s, "
        f"interval={INTERVAL_SEC}s, full_interval={FULL_INTERVAL_SEC}s)"
    )

    last_trigger_mtime = 0.0
    last_run_at = 0.0
    last_full_run_at = 0.0
    pending_since = 0.0

    while True:
        now = time.time()
        trigger_mtime = 0.0
        try:
            trigger_mtime = TRIGGER_FILE.stat().st_mtime
        except FileNotFoundError:
            pass

        # File was touched → mark pending
        if trigger_mtime > last_trigger_mtime:
            last_trigger_mtime = trigger_mtime
            pending_since = now

        # Debounced drain
        if pending_since and (now - pending_since) >= DEBOUNCE_SEC:
            _log("debounce elapsed — running reflection")
            _run_reflection(scope="auto")
            last_run_at = time.time()
            pending_since = 0.0
            continue

        # Scheduled full run (digest+synthesize) - its own independent
        # cadence, never triggered by on-save activity. An earlier attempt
        # escalated the auto/drain path to full based on DB state; that was
        # reverted because it let a routine save hold .reflect.lock for a
        # multi-minute LLM run. Cadence belongs to this timer, not to
        # pick_scope.
        if (now - last_full_run_at) >= FULL_INTERVAL_SEC:
            _log("periodic full run (digest+synthesize)")
            _run_reflection(scope="full")
            last_full_run_at = time.time()
            continue

        # Safety-net periodic run
        if (now - last_run_at) >= INTERVAL_SEC:
            _log("periodic run (safety net)")
            _run_reflection(scope="auto")
            last_run_at = time.time()

        time.sleep(1)


if __name__ == "__main__":
    main()
