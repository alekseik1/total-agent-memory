"""Daemon settings resolved from the environment, with named defaults."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass

# The daemon is a local service only: it never binds anything but loopback.
DAEMON_HOST = "127.0.0.1"

PORT_ENV = "TAM_DAEMON_PORT"
IDLE_EXIT_ENV = "TAM_DAEMON_IDLE_EXIT_MIN"
START_TIMEOUT_ENV = "TAM_DAEMON_START_TIMEOUT_SEC"
STOP_TIMEOUT_ENV = "TAM_DAEMON_STOP_TIMEOUT_SEC"

# 0 asks the OS for a free port; the chosen one is published in the registry.
DEFAULT_PORT = 0
# An idle daemon gives its memory back after an hour; 0 keeps it forever.
DEFAULT_IDLE_EXIT_MIN = 60.0
# Spawn -> bound socket answering /internal/health. The embedding model loads
# in the background afterwards, so this covers interpreter start and imports.
DEFAULT_START_TIMEOUT_SEC = 30.0
# Graceful shutdown: drain in-flight requests, flush background work.
DEFAULT_STOP_TIMEOUT_SEC = 20.0

MAX_PORT = 65535
SECONDS_PER_MINUTE = 60.0


class DaemonConfigError(ValueError):
    """An environment variable holds a value the daemon cannot use."""


@dataclass(frozen=True)
class DaemonSettings:
    host: str
    port: int
    idle_exit_sec: float
    start_timeout_sec: float
    stop_timeout_sec: float

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> DaemonSettings:
        env = os.environ if environ is None else environ
        port = _parse_int(env, PORT_ENV, DEFAULT_PORT)
        if not 0 <= port <= MAX_PORT:
            raise DaemonConfigError(f"{PORT_ENV} must be between 0 and {MAX_PORT}, got {port}")
        idle_min = _parse_float(env, IDLE_EXIT_ENV, DEFAULT_IDLE_EXIT_MIN, allow_zero=True)
        return cls(
            host=DAEMON_HOST,
            port=port,
            idle_exit_sec=idle_min * SECONDS_PER_MINUTE,
            start_timeout_sec=_parse_float(env, START_TIMEOUT_ENV, DEFAULT_START_TIMEOUT_SEC, allow_zero=False),
            stop_timeout_sec=_parse_float(env, STOP_TIMEOUT_ENV, DEFAULT_STOP_TIMEOUT_SEC, allow_zero=False),
        )


def _raw(env: Mapping[str, str], name: str) -> str | None:
    value = env.get(name, "").strip()
    return value or None


def _parse_int(env: Mapping[str, str], name: str, default: int) -> int:
    raw = _raw(env, name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        raise DaemonConfigError(f"{name} must be an integer, got {raw!r}") from None


def _parse_float(env: Mapping[str, str], name: str, default: float, *, allow_zero: bool) -> float:
    raw = _raw(env, name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        raise DaemonConfigError(f"{name} must be a number, got {raw!r}") from None
    if not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
        bound = "zero or positive" if allow_zero else "positive"
        raise DaemonConfigError(f"{name} must be {bound}, got {raw!r}")
    return value
