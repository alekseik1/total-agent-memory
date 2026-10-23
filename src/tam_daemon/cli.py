"""``tam daemon start|stop|status|restart|serve`` and ``tam proxy``."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

from tam_daemon import client as daemon_client
from tam_daemon.logs import configure as configure_logging
from tam_daemon.registry import DaemonPaths, RegistryError, process_rss_bytes
from tam_daemon.settings import DaemonConfigError, DaemonSettings

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 64
# LSB init-script convention: `status` of a service that is not running.
EXIT_NOT_RUNNING = 3
MIB = 1024 * 1024


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tam daemon", description="Shared warm TAM daemon for this memory dir.")
    sub = p.add_subparsers(dest="command", required=True)
    start = sub.add_parser("start", help="start the daemon in the background (no-op when running)")
    start.add_argument("--foreground", action="store_true", help="run in this process (for launchd/systemd)")
    sub.add_parser("stop", help="stop the daemon gracefully")
    status = sub.add_parser("status", help="pid, port, version, uptime, RSS, memory dir")
    status.add_argument("--json", action="store_true", help="machine-readable output")
    sub.add_parser("restart", help="stop, then start")
    sub.add_parser("serve", help="run the daemon in the foreground (what `start` spawns)")
    return p


def _context() -> tuple[DaemonSettings, Path, DaemonPaths, str]:
    from paths import memory_dir as resolve_memory_dir
    from version import VERSION

    settings = DaemonSettings.from_env()
    memory_dir = resolve_memory_dir().resolve()
    return settings, memory_dir, DaemonPaths.for_memory_dir(memory_dir), VERSION


def _describe(endpoint: daemon_client.DaemonEndpoint, info: dict[str, Any] | None) -> dict[str, Any]:
    record = endpoint.record
    return {
        "running": True,
        "pid": record.pid,
        "port": record.port,
        "url": record.mcp_url,
        "version": record.version,
        "started_at": datetime.fromtimestamp(record.started_at, UTC).isoformat(timespec="seconds"),
        "uptime_sec": round(max(0.0, time.time() - record.started_at), 1),
        "rss_bytes": process_rss_bytes(record.pid),
        "memory_dir": record.memory_dir,
        "status": info.get("status") if info else "unreachable",
        "warm": info.get("warm") if info else None,
        "idle_exit_min": info.get("idle_exit_min") if info else None,
    }


def _format_uptime(seconds: float) -> str:
    minutes, sec = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    return (f"{days}d " if days else "") + f"{hours:02d}:{minutes:02d}:{sec:02d}"


def _print_human(desc: dict[str, Any], out: TextIO) -> None:
    rss = desc["rss_bytes"]
    lines = [
        f"TAM daemon running ({desc['status']})",
        f"  pid:        {desc['pid']}",
        f"  port:       {desc['port']}  ({desc['url']})",
        f"  version:    {desc['version']}",
        f"  uptime:     {_format_uptime(desc['uptime_sec'])}",
        f"  rss:        {f'{rss / MIB:.1f} MiB' if rss is not None else 'n/a'}",
        f"  memory dir: {desc['memory_dir']}",
    ]
    if desc.get("warm"):
        lines.append("  warm:       " + ", ".join(f"{k}={v}" for k, v in desc["warm"].items()))
    out.write("\n".join(lines) + "\n")


def cmd_status(paths: DaemonPaths, memory_dir: Path, as_json: bool, out: TextIO) -> int:
    status = daemon_client.probe(paths)
    if not status.running:
        payload = {"running": False, "memory_dir": str(memory_dir), "stale_registry_removed": status.stale_removed}
        out.write(json.dumps(payload) + "\n" if as_json else
                  f"TAM daemon not running (memory dir {memory_dir})"
                  + (" — removed a stale registry" if status.stale_removed else "") + "\n")
        return EXIT_NOT_RUNNING
    if status.record is None:
        payload = {"running": True, "status": "starting", "memory_dir": str(memory_dir)}
        out.write(json.dumps(payload) + "\n" if as_json else "TAM daemon is starting\n")
        return EXIT_OK
    endpoint = daemon_client.endpoint_for(paths, status.record)
    desc = _describe(endpoint, daemon_client.health(endpoint))
    if as_json:
        out.write(json.dumps(desc) + "\n")
    else:
        _print_human(desc, out)
    return EXIT_OK


def cmd_start(settings: DaemonSettings, paths: DaemonPaths, memory_dir: Path, version: str, out: TextIO) -> int:
    endpoint, _info, started = daemon_client.ensure_running(paths, memory_dir, settings, version,
                                                           restart_on_version_mismatch=False)
    record = endpoint.record
    if started:
        out.write(f"TAM daemon started: pid {record.pid}, port {record.port}, version {record.version}\n")
    else:
        out.write(f"TAM daemon already running: pid {record.pid}, port {record.port}, version {record.version}\n")
        if record.version != version:
            out.write(f"  note: installed version is {version}; run `tam daemon restart` to switch\n")
    return EXIT_OK


def cmd_stop(settings: DaemonSettings, paths: DaemonPaths, out: TextIO) -> int:
    if daemon_client.stop(paths, settings):
        out.write("TAM daemon stopped\n")
    else:
        out.write("TAM daemon not running\n")
    return EXIT_OK


def main(argv: Sequence[str] | None = None, out: TextIO | None = None) -> int:
    out = out or sys.stdout
    try:
        args = parser().parse_args(list(argv if argv is not None else sys.argv[1:]))
    except SystemExit as exc:
        return EXIT_USAGE if exc.code else EXIT_OK
    configure_logging()
    try:
        settings, memory_dir, paths, version = _context()
        if args.command == "serve" or (args.command == "start" and args.foreground):
            from tam_daemon.service import serve
            return serve(settings)
        if args.command == "status":
            return cmd_status(paths, memory_dir, args.json, out)
        if args.command == "start":
            return cmd_start(settings, paths, memory_dir, version, out)
        if args.command == "stop":
            return cmd_stop(settings, paths, out)
        cmd_stop(settings, paths, out)
        return cmd_start(settings, paths, memory_dir, version, out)
    except (DaemonConfigError, RegistryError, daemon_client.DaemonUnavailable) as exc:
        sys.stderr.write(f"tam daemon: {exc}\n")
        return EXIT_FAILED


def proxy_main(argv: Sequence[str] | None = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    if args:
        sys.stderr.write("usage: tam proxy   (no arguments; configure with TAM_MEMORY_DIR / TAM_DAEMON_* env)\n")
        return EXIT_USAGE
    from tam_daemon.proxy import main as run_proxy
    return run_proxy()


def dispatch(argv: Sequence[str]) -> int:
    """Entry for `python src/tam_daemon daemon ...` / `... proxy`."""
    if argv[:1] == ["daemon"]:
        return main(argv[1:])
    if argv[:1] == ["proxy"]:
        return proxy_main(argv[1:])
    sys.stderr.write("usage: tam_daemon daemon {start,stop,status,restart,serve} | tam_daemon proxy\n")
    return EXIT_USAGE
