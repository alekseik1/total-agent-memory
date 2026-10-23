"""Talk to, start and stop the daemon of one memory directory."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from tam_daemon.logs import get_logger
from tam_daemon.registry import (
    DaemonPaths,
    DaemonRecord,
    FileLock,
    RegistryError,
    lock_is_held,
    log_tail,
    prepare_log,
    read_record,
    read_token,
    remove_record,
)
from tam_daemon.settings import DaemonSettings

LOGGER = get_logger("client")
HEALTH_TIMEOUT_SEC = 3.0
SHUTDOWN_REQUEST_TIMEOUT_SEC = 5.0
POLL_INTERVAL_SEC = 0.1
# After SIGTERM is ignored, SIGKILL; each phase waits this long.
FORCE_STOP_WAIT_SEC = 5.0
# The daemon's own exit code when another daemon already owns the memory dir.
EXIT_ALREADY_RUNNING = 3
ENTRY_DIR = Path(__file__).resolve().parent
STATUS_READY = "ready"
STATUS_STARTING = "starting"
HEALTHY_STATUSES = frozenset({STATUS_READY, STATUS_STARTING})


class DaemonUnavailable(RuntimeError):
    """No healthy daemon could be reached or started."""


@dataclass(frozen=True)
class DaemonEndpoint:
    record: DaemonRecord
    token: str

    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"}


@dataclass(frozen=True)
class DaemonStatus:
    running: bool
    record: DaemonRecord | None
    stale_removed: bool


def http_client(timeout: float) -> httpx.Client:
    # trust_env=False: an HTTP(S)_PROXY in the environment must never see loopback traffic or the token.
    return httpx.Client(timeout=timeout, trust_env=False, follow_redirects=False)


def health(endpoint: DaemonEndpoint, timeout: float = HEALTH_TIMEOUT_SEC) -> dict[str, Any] | None:
    """The daemon's /internal/health document, or None when it does not answer healthily."""
    try:
        with http_client(timeout) as client:
            response = client.get(f"{endpoint.record.base_url}/internal/health", headers=endpoint.headers)
    except httpx.HTTPError as exc:
        LOGGER.debug("health probe failed", extra={"error": str(exc)})
        return None
    if response.status_code != httpx.codes.OK:
        LOGGER.warning("health probe rejected", extra={"status": response.status_code})
        return None
    try:
        data = response.json()
    except ValueError:
        LOGGER.warning("health probe returned non-JSON")
        return None
    return data if isinstance(data, dict) else None


def request_shutdown(endpoint: DaemonEndpoint) -> bool:
    try:
        with http_client(SHUTDOWN_REQUEST_TIMEOUT_SEC) as client:
            response = client.post(f"{endpoint.record.base_url}/internal/shutdown", headers=endpoint.headers)
    except httpx.HTTPError as exc:
        LOGGER.warning("shutdown request failed", extra={"error": str(exc)})
        return False
    return response.status_code == httpx.codes.ACCEPTED


def probe(paths: DaemonPaths) -> DaemonStatus:
    """Running state from the lock; a registry left by a dead daemon is removed (stale pid recovery)."""
    running = lock_is_held(paths.lock)
    try:
        record = read_record(paths)
    except RegistryError as exc:
        LOGGER.warning("unreadable daemon registry", extra={"error": str(exc)})
        record = None
        if not running:
            remove_record(paths)
            return DaemonStatus(False, None, True)
    if not running and record is not None:
        LOGGER.info("removing stale daemon registry", extra={"pid": record.pid, "port": record.port})
        remove_record(paths, pid=record.pid)
        return DaemonStatus(False, None, True)
    return DaemonStatus(running, record, False)


def endpoint_for(paths: DaemonPaths, record: DaemonRecord) -> DaemonEndpoint:
    return DaemonEndpoint(record, read_token(paths))


def spawn_detached(paths: DaemonPaths, memory_dir: Path) -> subprocess.Popen[bytes]:
    """Start `tam daemon serve` in its own session, logging to daemon.log."""
    log_path = prepare_log(paths)
    env = dict(os.environ, TAM_MEMORY_DIR=str(memory_dir))
    command = [sys.executable, str(ENTRY_DIR), "daemon", "serve"]
    kwargs: dict[str, Any] = {}
    if os.name == "posix":
        kwargs["start_new_session"] = True
    else:
        kwargs["creationflags"] = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    with log_path.open("ab") as log:
        proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=log, env=env,
                                close_fds=True, **kwargs)
    LOGGER.info("daemon spawned", extra={"pid": proc.pid, "memory_dir": str(memory_dir)})
    return proc


def wait_until_healthy(paths: DaemonPaths, proc: subprocess.Popen[bytes] | None,
                       timeout_sec: float) -> tuple[DaemonEndpoint, dict[str, Any]]:
    """Block until the memory dir's daemon answers /internal/health, at most `timeout_sec`."""
    deadline = time.monotonic() + timeout_sec
    while True:
        if proc is not None and proc.poll() is not None and proc.returncode != EXIT_ALREADY_RUNNING:
            raise DaemonUnavailable(f"daemon exited with code {proc.returncode} during start-up; "
                                    f"log tail:\n{log_tail(paths)}")
        status = probe(paths)
        if status.record is not None:
            try:
                endpoint = endpoint_for(paths, status.record)
            except RegistryError as exc:
                raise DaemonUnavailable(str(exc)) from exc
            info = health(endpoint)
            if info is not None and info.get("status") in HEALTHY_STATUSES:
                return endpoint, info
        if time.monotonic() >= deadline:
            raise DaemonUnavailable(f"daemon did not become healthy within {timeout_sec:g}s; "
                                    f"log tail:\n{log_tail(paths)}")
        time.sleep(POLL_INTERVAL_SEC)


def wait_until_stopped(paths: DaemonPaths, timeout_sec: float) -> bool:
    deadline = time.monotonic() + timeout_sec
    while lock_is_held(paths.lock):
        if time.monotonic() >= deadline:
            return False
        time.sleep(POLL_INTERVAL_SEC)
    return True


def _signal(pid: int, sig: int) -> None:
    try:
        os.kill(pid, sig)
    except ProcessLookupError:
        LOGGER.info("daemon already gone", extra={"pid": pid})


def stop(paths: DaemonPaths, settings: DaemonSettings) -> bool:
    """Stop the running daemon; False when none was running. Raises DaemonUnavailable if it will not die."""
    status = probe(paths)
    if not status.running:
        return False
    record = status.record
    if record is None:
        # Lock held but registry not written yet: the daemon is still binding.
        deadline = time.monotonic() + settings.start_timeout_sec
        while record is None and time.monotonic() < deadline and lock_is_held(paths.lock):
            time.sleep(POLL_INTERVAL_SEC)
            record = probe(paths).record
        if record is None:
            if not lock_is_held(paths.lock):
                return True
            raise DaemonUnavailable("a daemon holds the lock but never published its registry")
    try:
        requested = request_shutdown(endpoint_for(paths, record))
    except RegistryError as exc:
        LOGGER.warning("cannot authenticate shutdown request", extra={"error": str(exc)})
        requested = False
    if requested and wait_until_stopped(paths, settings.stop_timeout_sec):
        remove_record(paths, pid=record.pid)
        return True
    LOGGER.warning("daemon ignored graceful shutdown; terminating", extra={"pid": record.pid})
    _signal(record.pid, signal.SIGTERM)
    if wait_until_stopped(paths, FORCE_STOP_WAIT_SEC):
        remove_record(paths, pid=record.pid)
        return True
    if os.name == "posix":
        _signal(record.pid, signal.SIGKILL)
        if wait_until_stopped(paths, FORCE_STOP_WAIT_SEC):
            remove_record(paths, pid=record.pid)
            return True
    raise DaemonUnavailable(f"daemon pid {record.pid} did not stop")


def ensure_running(paths: DaemonPaths, memory_dir: Path, settings: DaemonSettings, version: str, *,
                   restart_on_version_mismatch: bool) -> tuple[DaemonEndpoint, dict[str, Any], bool]:
    """A healthy daemon for `memory_dir`: the running one, or a freshly started one.

    Returns (endpoint, health, started). With `restart_on_version_mismatch`, a
    daemon running another TAM version is shut down and replaced.
    """
    paths.ensure_root()
    with FileLock(paths.start_lock) as start_lock:
        if not start_lock.acquire(settings.start_timeout_sec):
            raise DaemonUnavailable("another process has been starting the daemon for too long")
        if probe(paths).running:
            endpoint, info = wait_until_healthy(paths, None, settings.start_timeout_sec)
            if not restart_on_version_mismatch or info.get("version") == version:
                return endpoint, info, False
            LOGGER.info("daemon version differs; restarting",
                        extra={"daemon_version": info.get("version"), "client_version": version})
            stop(paths, settings)
        proc = spawn_detached(paths, memory_dir)
        endpoint, info = wait_until_healthy(paths, proc, settings.start_timeout_sec)
        return endpoint, info, True
