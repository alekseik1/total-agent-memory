"""On-disk state of the daemon under ``<memory dir>/daemon/``.

    daemon.lock   held (OS file lock) by the running daemon for its whole life.
                  The lock is the source of truth for "is a daemon running":
                  the OS drops it when the process dies, so a crashed daemon
                  can never leave a live-looking lock behind, and a recycled
                  pid can never be mistaken for the daemon.
    start.lock    serialises starters (`tam daemon start`, every `tam proxy`)
                  so concurrent clients spawn one daemon, not several.
    daemon.json   registry: pid, port, version, start time. Written by the
                  daemon once its socket is bound; a registry whose lock is
                  free is stale and is removed by the next reader.
    daemon.token  bearer token, 0600, generated on first start and reused.
    daemon.log    stderr of detached daemons.
"""

from __future__ import annotations

import json
import os
import secrets
import stat
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Self

DAEMON_SUBDIR = "daemon"
LOCK_NAME = "daemon.lock"
START_LOCK_NAME = "start.lock"
REGISTRY_NAME = "daemon.json"
TOKEN_NAME = "daemon.token"
LOG_NAME = "daemon.log"

PRIVATE_DIR_MODE = 0o700
PRIVATE_FILE_MODE = 0o600
TOKEN_BYTES = 32
LOCK_POLL_INTERVAL_SEC = 0.05
# A detached daemon's log is truncated at start once it grows past this.
MAX_LOG_BYTES = 5 * 1024 * 1024
LOG_TAIL_BYTES = 4096
REGISTRY_SCHEMA = 1
KIB = 1024


class RegistryError(RuntimeError):
    """The daemon state directory holds something unusable."""


@dataclass(frozen=True)
class DaemonPaths:
    root: Path
    lock: Path
    start_lock: Path
    registry: Path
    token: Path
    log: Path

    @classmethod
    def for_memory_dir(cls, memory_dir: Path) -> DaemonPaths:
        root = Path(memory_dir) / DAEMON_SUBDIR
        return cls(root=root, lock=root / LOCK_NAME, start_lock=root / START_LOCK_NAME,
                   registry=root / REGISTRY_NAME, token=root / TOKEN_NAME, log=root / LOG_NAME)

    def ensure_root(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        if os.name == "posix" and stat.S_IMODE(self.root.stat().st_mode) != PRIVATE_DIR_MODE:
            self.root.chmod(PRIVATE_DIR_MODE)


class FileLock:
    """Exclusive, non-reentrant OS lock on a file (flock on POSIX, msvcrt on Windows)."""

    def __init__(self, path: Path):
        self.path = path
        self._fd: int | None = None

    @property
    def held(self) -> bool:
        return self._fd is not None

    def try_acquire(self) -> bool:
        if self._fd is not None:
            raise RegistryError(f"{self.path} is already held by this object")
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, PRIVATE_FILE_MODE)
        try:
            locked = _lock_fd(fd)
        except BaseException:
            os.close(fd)
            raise
        if not locked:
            os.close(fd)
            return False
        self._fd = fd
        return True

    def acquire(self, timeout_sec: float) -> bool:
        deadline = time.monotonic() + timeout_sec
        while True:
            if self.try_acquire():
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(LOCK_POLL_INTERVAL_SEC)

    def release(self) -> None:
        if self._fd is None:
            return
        fd, self._fd = self._fd, None
        try:
            _unlock_fd(fd)
        finally:
            os.close(fd)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.release()


if os.name == "posix":
    import fcntl

    def _lock_fd(fd: int) -> bool:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        return True

    def _unlock_fd(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_UN)
else:
    import msvcrt

    def _lock_fd(fd: int) -> bool:
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        except OSError:
            return False
        return True

    def _unlock_fd(fd: int) -> None:
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)


def lock_is_held(path: Path) -> bool:
    """True while some process (a running daemon) holds the lock at `path`."""
    if not path.exists():
        return False
    probe = FileLock(path)
    if probe.try_acquire():
        probe.release()
        return False
    return True


@dataclass(frozen=True)
class DaemonRecord:
    pid: int
    host: str
    port: int
    version: str
    started_at: float
    memory_dir: str
    python: str
    schema: int = REGISTRY_SCHEMA

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def mcp_url(self) -> str:
        return f"{self.base_url}/mcp"

    @classmethod
    def from_json(cls, data: object) -> DaemonRecord:
        if not isinstance(data, dict):
            raise RegistryError("registry must be a JSON object")
        try:
            record = cls(pid=int(data["pid"]), host=str(data["host"]), port=int(data["port"]),
                         version=str(data["version"]), started_at=float(data["started_at"]),
                         memory_dir=str(data["memory_dir"]), python=str(data["python"]),
                         schema=int(data.get("schema", REGISTRY_SCHEMA)))
        except (KeyError, TypeError, ValueError) as exc:
            raise RegistryError(f"registry is malformed: {exc}") from exc
        if record.pid <= 0 or not 0 < record.port <= 65535:
            raise RegistryError("registry holds an invalid pid or port")
        return record


def _atomic_private_write(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, PRIVATE_FILE_MODE)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def write_record(paths: DaemonPaths, record: DaemonRecord) -> None:
    _atomic_private_write(paths.registry, json.dumps(asdict(record), indent=2) + "\n")


def read_record(paths: DaemonPaths) -> DaemonRecord | None:
    try:
        text = paths.registry.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    try:
        return DaemonRecord.from_json(json.loads(text))
    except (ValueError, RegistryError) as exc:
        raise RegistryError(f"{paths.registry}: {exc}") from exc


def remove_record(paths: DaemonPaths, pid: int | None = None) -> None:
    """Delete the registry; with `pid`, only when it still describes that process."""
    if pid is not None:
        try:
            current = read_record(paths)
        except RegistryError:
            current = None
        if current is not None and current.pid != pid:
            return
    paths.registry.unlink(missing_ok=True)


def load_or_create_token(paths: DaemonPaths) -> str:
    """The persisted bearer token, created (0600) on first use."""
    paths.ensure_root()
    try:
        fd = os.open(paths.token, os.O_WRONLY | os.O_CREAT | os.O_EXCL, PRIVATE_FILE_MODE)
    except FileExistsError:
        return read_token(paths, repair_mode=True)
    token = secrets.token_urlsafe(TOKEN_BYTES)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(token + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return token


def read_token(paths: DaemonPaths, *, repair_mode: bool = False) -> str:
    try:
        token = paths.token.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        raise RegistryError(f"{paths.token} does not exist; start the daemon first") from None
    if not token:
        raise RegistryError(f"{paths.token} is empty; delete it and restart the daemon")
    if repair_mode and os.name == "posix" and stat.S_IMODE(paths.token.stat().st_mode) != PRIVATE_FILE_MODE:
        paths.token.chmod(PRIVATE_FILE_MODE)
    return token


def prepare_log(paths: DaemonPaths) -> Path:
    """The daemon log, truncated first when it has grown past MAX_LOG_BYTES."""
    paths.ensure_root()
    if paths.log.exists() and paths.log.stat().st_size > MAX_LOG_BYTES:
        paths.log.write_text("", encoding="utf-8")
    if not paths.log.exists():
        os.close(os.open(paths.log, os.O_WRONLY | os.O_CREAT, PRIVATE_FILE_MODE))
    return paths.log


def log_tail(paths: DaemonPaths) -> str:
    try:
        with paths.log.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - LOG_TAIL_BYTES))
            return handle.read().decode("utf-8", errors="replace").strip()
    except FileNotFoundError:
        return ""


def process_rss_bytes(pid: int) -> int | None:
    """Resident set size of `pid`, or None where it cannot be measured without extra packages."""
    status = Path(f"/proc/{pid}/status")
    if status.exists():
        try:
            for line in status.read_text(encoding="utf-8").splitlines():
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * KIB
        except (OSError, ValueError, IndexError):
            return None
        return None
    if os.name != "posix":
        return None
    try:
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True,
                             timeout=5, check=False).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return int(out) * KIB if out.isdigit() else None
