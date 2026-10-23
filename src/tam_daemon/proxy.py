"""``tam proxy``: an MCP stdio server that relays every frame to the shared daemon.

The relay works on raw JSON-RPC frames, not on SDK objects, so tool lists,
results, ``structuredContent``, ``isError`` and protocol errors pass through
byte-for-byte whatever SDK version either side runs. Both wire eras are
handled: the initialize handshake (2024-11-05 .. 2025-11-25; the negotiated
version is sent as ``MCP-Protocol-Version`` afterwards) and the per-request
envelope of 2026-07-28 (routing headers derived from ``params._meta``).

Start-up: reuse the memory dir's daemon, start one when none runs, replace
one that runs another TAM version, and when no daemon can be had at all run
the ordinary in-process stdio server instead, so the client always works.
Mid-session: a daemon that went away (idle exit, upgrade by another client)
is restarted transparently; if that fails the proxy switches to a private
stdio server child and replays the handshake.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import sys
import threading
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import httpx

from tam_daemon import client as daemon_client
from tam_daemon.logs import configure as configure_logging
from tam_daemon.logs import get_logger
from tam_daemon.registry import DaemonPaths
from tam_daemon.settings import DaemonConfigError, DaemonSettings

LOGGER = get_logger("proxy")

PROTOCOL_VERSION_META_KEY = "io.modelcontextprotocol/protocolVersion"
HEADER_PROTOCOL_VERSION = "MCP-Protocol-Version"
HEADER_METHOD = "Mcp-Method"
HEADER_NAME = "Mcp-Name"
NAME_BEARING_PARAMS = {"tools/call": "name", "prompts/get": "name", "resources/read": "uri"}
_HEADER_SAFE = re.compile(r"^[\x20-\x7E]*$")
_B64_SENTINEL = re.compile(r"^=\?base64\?.*\?=$")

JSONRPC_PARSE_ERROR = -32700
JSONRPC_INTERNAL_ERROR = -32603
# stdio frames carry whole memory records; this only bounds a runaway line.
MAX_FRAME_BYTES = 64 * 1024 * 1024
# Tool calls may run long (consolidation, ingest); the relay itself never times out a read.
HTTP_CONNECT_TIMEOUT_SEC = 5.0
HTTP_WRITE_TIMEOUT_SEC = 30.0
HTTP_POOL_TIMEOUT_SEC = 30.0
# After stdin closes, in-flight requests get this long to finish.
EOF_DRAIN_TIMEOUT_SEC = 10.0
LOCAL_REPLAY_TIMEOUT_SEC = 120.0
LOCAL_STOP_TIMEOUT_SEC = 10.0
REPLAY_INIT_ID = "tam-proxy-replay-initialize"
SERVER_PATH = Path(__file__).resolve().parents[1] / "server.py"
SSE_CONTENT_TYPE = "text/event-stream"
JSON_CONTENT_TYPE = "application/json"


def encode_header_value(value: str) -> str:
    """Same rule as the MCP SDK: printable ASCII verbatim, anything else base64-wrapped."""
    if _HEADER_SAFE.fullmatch(value) and value == value.strip() and not _B64_SENTINEL.fullmatch(value):
        return value
    return f"=?base64?{base64.b64encode(value.encode('utf-8')).decode('ascii')}?="


def error_frame(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


class StdoutWriter:
    """Serialises whole frames onto stdout (writes never interleave: no await inside).

    When the client has gone (stdout closed) frames are dropped and
    `on_broken` fires once so the proxy shuts down instead of erroring per frame.
    """

    def __init__(self, stream: Any = None, on_broken: Callable[[], None] | None = None):
        self.stream = stream if stream is not None else sys.stdout.buffer
        self.on_broken = on_broken
        self.broken = False

    def message(self, message: dict[str, Any] | list[Any]) -> None:
        self.raw(json.dumps(message, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n")

    def raw(self, line: bytes) -> None:
        if self.broken:
            return
        try:
            self.stream.write(line if line.endswith(b"\n") else line + b"\n")
            self.stream.flush()
        except (BrokenPipeError, ValueError, OSError) as exc:
            self.broken = True
            LOGGER.warning("client stdout closed; stopping", extra={"error": str(exc)})
            if self.on_broken is not None:
                self.on_broken()


def stdin_frames(loop: asyncio.AbstractEventLoop, stream: Any = None) -> asyncio.Queue[bytes | None]:
    """Lines of stdin on a queue, read by a thread (portable: no pipe support needed from the loop)."""
    queue: asyncio.Queue[bytes | None] = asyncio.Queue()
    source = stream if stream is not None else sys.stdin.buffer

    def pump() -> None:
        try:
            while True:
                line = source.readline(MAX_FRAME_BYTES + 1)
                if not line:
                    break
                loop.call_soon_threadsafe(queue.put_nowait, line)
        except (OSError, ValueError) as exc:
            LOGGER.warning("stdin read failed", extra={"error": str(exc)})
        finally:
            try:
                loop.call_soon_threadsafe(queue.put_nowait, None)
            except RuntimeError:
                LOGGER.debug("event loop closed before stdin EOF was delivered")

    threading.Thread(target=pump, name="tam-proxy-stdin", daemon=True).start()
    return queue


async def sse_messages(lines: AsyncIterator[str]) -> AsyncIterator[dict[str, Any] | list[Any]]:
    """JSON-RPC messages carried by an SSE stream's `message` events."""
    data: list[str] = []
    event = "message"
    async for line in lines:
        if line == "":
            if data and event == "message":
                payload = "\n".join(data)
                try:
                    yield json.loads(payload)
                except ValueError:
                    LOGGER.warning("non-JSON SSE event from daemon", extra={"preview": payload[:200]})
            data, event = [], "message"
            continue
        if line.startswith(":"):
            continue
        name, _, value = line.partition(":")
        value = value.removeprefix(" ")
        if name == "data":
            data.append(value)
        elif name == "event":
            event = value or "message"
    if data and event == "message":
        try:
            yield json.loads("\n".join(data))
        except ValueError:
            LOGGER.warning("non-JSON trailing SSE event from daemon")


class LocalBackend:
    """A private stdio TAM server child, used when the daemon is gone for good."""

    def __init__(self, writer: StdoutWriter, on_exit: Callable[[], None]):
        self.writer = writer
        self.on_exit = on_exit
        self.proc: asyncio.subprocess.Process | None = None
        self._pump: asyncio.Task[None] | None = None

    async def start(self, init_frame: dict[str, Any] | None) -> None:
        env = dict(os.environ, MCP_TRANSPORT="stdio")
        env.pop("MCP_HTTP_LISTEN_FD", None)
        self.proc = await asyncio.create_subprocess_exec(
            sys.executable, str(SERVER_PATH), stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            env=env, limit=MAX_FRAME_BYTES)
        LOGGER.warning("switched to a local stdio server", extra={"pid": self.proc.pid})
        if init_frame is not None:
            await self._replay_handshake(init_frame)
        self._pump = asyncio.create_task(self._relay_stdout(), name="tam-proxy-local-stdout")

    async def _replay_handshake(self, init_frame: dict[str, Any]) -> None:
        assert self.proc is not None and self.proc.stdin is not None and self.proc.stdout is not None
        replay = dict(init_frame, id=REPLAY_INIT_ID)
        await self._write(json.dumps(replay).encode("utf-8") + b"\n")

        async def until_reply() -> None:
            assert self.proc is not None and self.proc.stdout is not None
            while True:
                line = await self.proc.stdout.readline()
                if not line:
                    raise ConnectionError("local server exited during the replayed handshake")
                try:
                    message = json.loads(line)
                except ValueError:
                    continue
                if isinstance(message, dict) and message.get("id") == REPLAY_INIT_ID:
                    return

        await asyncio.wait_for(until_reply(), LOCAL_REPLAY_TIMEOUT_SEC)
        await self._write(b'{"jsonrpc":"2.0","method":"notifications/initialized"}\n')

    async def _write(self, line: bytes) -> None:
        assert self.proc is not None and self.proc.stdin is not None
        self.proc.stdin.write(line if line.endswith(b"\n") else line + b"\n")
        await self.proc.stdin.drain()

    async def send(self, line: bytes) -> None:
        await self._write(line)

    async def _relay_stdout(self) -> None:
        assert self.proc is not None and self.proc.stdout is not None
        while True:
            line = await self.proc.stdout.readline()
            if not line:
                break
            self.writer.raw(line)
        code = await self.proc.wait()
        LOGGER.error("local stdio server exited", extra={"code": code})
        self.on_exit()

    async def kill(self) -> None:
        if self.proc is not None and self.proc.returncode is None:
            self.proc.kill()
            await self.proc.wait()

    async def close(self) -> None:
        if self.proc is None or self.proc.returncode is not None:
            return
        if self.proc.stdin is not None:
            self.proc.stdin.close()
        try:
            await asyncio.wait_for(self.proc.wait(), LOCAL_STOP_TIMEOUT_SEC)
        except TimeoutError:
            self.proc.terminate()
            await self.proc.wait()
        if self._pump is not None:
            await self._pump


class Bridge:
    """Relays stdin frames to the daemon's /mcp and the replies back to stdout."""

    def __init__(self, endpoint: daemon_client.DaemonEndpoint, reconnect: Callable[[], daemon_client.DaemonEndpoint],
                 writer: StdoutWriter):
        self.endpoint = endpoint
        self._reconnect_sync = reconnect
        self.writer = writer
        self.protocol_version: str | None = None
        self.init_frame: dict[str, Any] | None = None
        self.local: LocalBackend | None = None
        self.in_flight: dict[Any, asyncio.Task[None]] = {}
        self._tasks: set[asyncio.Task[None]] = set()
        self._reconnect_lock = asyncio.Lock()
        self._generation = 0
        self._local_lock = asyncio.Lock()
        self.done = asyncio.Event()
        if writer.on_broken is None:
            writer.on_broken = self.done.set
        self.client = httpx.AsyncClient(
            trust_env=False, follow_redirects=False,
            timeout=httpx.Timeout(connect=HTTP_CONNECT_TIMEOUT_SEC, read=None, write=HTTP_WRITE_TIMEOUT_SEC,
                                  pool=HTTP_POOL_TIMEOUT_SEC))

    # ── header derivation ────────────────────────────────────────────

    def headers_for(self, frame: dict[str, Any]) -> dict[str, str]:
        headers = {"Authorization": f"Bearer {self.endpoint.token}", "Content-Type": JSON_CONTENT_TYPE,
                   "Accept": f"{JSON_CONTENT_TYPE}, {SSE_CONTENT_TYPE}"}
        method = frame.get("method")
        params = frame.get("params")
        meta = params.get("_meta") if isinstance(params, dict) else None
        modern = meta.get(PROTOCOL_VERSION_META_KEY) if isinstance(meta, dict) else None
        if isinstance(modern, str):
            headers[HEADER_PROTOCOL_VERSION] = modern
            if isinstance(method, str):
                headers[HEADER_METHOD] = method
                name_key = NAME_BEARING_PARAMS.get(method)
                name = params.get(name_key) if name_key and isinstance(params, dict) else None
                if isinstance(name, str):
                    headers[HEADER_NAME] = encode_header_value(name)
        elif method != "initialize" and self.protocol_version:
            headers[HEADER_PROTOCOL_VERSION] = self.protocol_version
        return headers

    # ── frame dispatch ───────────────────────────────────────────────

    def _spawn(self, coro: Any, request_id: Any = None) -> None:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        if request_id is not None:
            self.in_flight[request_id] = task
            task.add_done_callback(lambda _t, rid=request_id: self.in_flight.pop(rid, None))

    async def dispatch(self, line: bytes) -> None:
        if not line.strip():
            return
        if len(line) > MAX_FRAME_BYTES:
            LOGGER.error("frame exceeds size limit; dropped", extra={"bytes": len(line)})
            self.writer.message(error_frame(None, JSONRPC_PARSE_ERROR, "frame exceeds the proxy size limit"))
            return
        if self.local is not None:
            await self.local.send(line)
            return
        try:
            frame = json.loads(line)
        except ValueError:
            self.writer.message(error_frame(None, JSONRPC_PARSE_ERROR, "Parse error"))
            return
        if isinstance(frame, dict) and frame.get("method") == "notifications/cancelled":
            # Stateless daemon: cancelling = dropping the HTTP exchange that carries the request.
            params = frame.get("params") if isinstance(frame.get("params"), dict) else {}
            task = self.in_flight.get(params.get("requestId"))
            if task is not None:
                task.cancel()
            return
        if isinstance(frame, dict) and frame.get("method") == "initialize":
            self.init_frame = frame
        request_id = frame.get("id") if isinstance(frame, dict) and "method" in frame else None
        self._spawn(self.forward(line, frame), request_id)

    async def forward(self, line: bytes, frame: Any) -> None:
        request_id = frame.get("id") if isinstance(frame, dict) and "method" in frame else None
        is_request = request_id is not None
        retried_auth = False
        reconnected = False
        while True:
            if self.local is not None:
                await self.local.send(line)
                return
            generation = self._generation
            headers = self.headers_for(frame if isinstance(frame, dict) else {})
            try:
                async with self.client.stream("POST", self.endpoint.record.mcp_url, content=line,
                                              headers=headers) as response:
                    if response.status_code == httpx.codes.UNAUTHORIZED and not retried_auth:
                        retried_auth = True
                        await response.aread()
                        await self.reconnect(generation)
                        continue
                    await self.relay(response, request_id, frame)
                    return
            except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                # Nothing reached the daemon, so a retry cannot duplicate a write.
                LOGGER.warning("daemon unreachable", extra={"error": str(exc), "reconnected": reconnected})
                if not reconnected:
                    reconnected = True
                    if await self.reconnect(generation):
                        continue
                try:
                    await self.activate_local()
                except (OSError, ConnectionError, TimeoutError) as local_exc:
                    LOGGER.error("local stdio server could not start", extra={"error": str(local_exc)})
                    if is_request:
                        self.writer.message(error_frame(request_id, JSONRPC_INTERNAL_ERROR,
                                                        f"TAM daemon unavailable and local fallback failed: {local_exc}"))
                    return
                continue
            except httpx.HTTPError as exc:
                LOGGER.error("daemon connection lost mid-request", extra={"error": str(exc)})
                if is_request:
                    self.writer.message(error_frame(request_id, JSONRPC_INTERNAL_ERROR,
                                                    f"TAM daemon connection lost: {exc}"))
                return

    async def relay(self, response: httpx.Response, request_id: Any, frame: Any) -> None:
        ctype = response.headers.get("content-type", "").split(";")[0].strip().lower()
        answered = False

        def emit(message: Any) -> None:
            nonlocal answered
            for item in message if isinstance(message, list) else [message]:
                if not isinstance(item, dict) or item.get("jsonrpc") != "2.0":
                    LOGGER.warning("non-JSON-RPC payload from daemon dropped", extra={"status": response.status_code})
                    continue
                if request_id is not None and item.get("id") == request_id and "method" not in item:
                    answered = True
                    self._observe_reply(frame, item)
                self.writer.message(item)

        if ctype == SSE_CONTENT_TYPE:
            async for message in sse_messages(response.aiter_lines()):
                emit(message)
        else:
            body = await response.aread()
            if body.strip() and ctype == JSON_CONTENT_TYPE:
                try:
                    emit(json.loads(body))
                except ValueError:
                    LOGGER.warning("non-JSON reply from daemon", extra={"status": response.status_code})
        if request_id is not None and not answered:
            LOGGER.warning("daemon reply carried no response", extra={"status": response.status_code})
            self.writer.message(error_frame(request_id, JSONRPC_INTERNAL_ERROR,
                                            f"TAM daemon returned HTTP {response.status_code} without a response"))

    def _observe_reply(self, frame: Any, reply: dict[str, Any]) -> None:
        if isinstance(frame, dict) and frame.get("method") == "initialize":
            result = reply.get("result")
            version = result.get("protocolVersion") if isinstance(result, dict) else None
            if isinstance(version, str):
                self.protocol_version = version

    # ── recovery ─────────────────────────────────────────────────────

    async def reconnect(self, seen_generation: int) -> bool:
        """Re-resolve (and if needed restart) the daemon once per failure generation."""
        async with self._reconnect_lock:
            if self._generation != seen_generation:
                return True
            try:
                self.endpoint = await asyncio.to_thread(self._reconnect_sync)
            except (daemon_client.DaemonUnavailable, OSError) as exc:
                LOGGER.error("daemon could not be restarted", extra={"error": str(exc)})
                return False
            self._generation += 1
            LOGGER.info("reconnected to daemon", extra={"port": self.endpoint.record.port})
            return True

    async def activate_local(self) -> None:
        async with self._local_lock:
            if self.local is not None:
                return
            local = LocalBackend(self.writer, self.done.set)
            try:
                await local.start(self.init_frame)
            except BaseException:
                await local.kill()
                raise
            self.local = local

    # ── main loop ────────────────────────────────────────────────────

    async def run(self, frames: asyncio.Queue[bytes | None]) -> None:
        try:
            while not self.done.is_set():
                getter = asyncio.ensure_future(frames.get())
                stopper = asyncio.ensure_future(self.done.wait())
                finished, _ = await asyncio.wait({getter, stopper}, return_when=asyncio.FIRST_COMPLETED)
                stopper.cancel()
                if getter not in finished:
                    getter.cancel()
                    break
                line = getter.result()
                if line is None:
                    break
                await self.dispatch(line)
            if self._tasks:
                _, pending = await asyncio.wait(set(self._tasks), timeout=EOF_DRAIN_TIMEOUT_SEC)
                for task in pending:
                    task.cancel()
        finally:
            await self.client.aclose()
            if self.local is not None:
                await self.local.close()


def run_in_process_server() -> None:
    """Fallback: the ordinary stdio server in this process (stdin has not been read yet)."""
    os.environ["MCP_TRANSPORT"] = "stdio"
    os.environ.pop("MCP_HTTP_LISTEN_FD", None)
    src = str(SERVER_PATH.parent)
    if src not in sys.path:
        sys.path.insert(0, src)
    import server
    server.run()


def main() -> int:
    configure_logging()
    from paths import memory_dir as resolve_memory_dir
    from version import VERSION

    try:
        settings = DaemonSettings.from_env()
    except DaemonConfigError as exc:
        LOGGER.error("invalid daemon settings; serving in-process", extra={"error": str(exc)})
        run_in_process_server()
        return 0
    memory_dir = resolve_memory_dir().resolve()
    paths = DaemonPaths.for_memory_dir(memory_dir)
    try:
        endpoint, info, started = daemon_client.ensure_running(
            paths, memory_dir, settings, VERSION, restart_on_version_mismatch=True)
    except (daemon_client.DaemonUnavailable, OSError) as exc:
        LOGGER.error("shared daemon unavailable; serving in-process", extra={"error": str(exc)})
        run_in_process_server()
        return 0
    LOGGER.info("proxy connected", extra={"port": endpoint.record.port, "daemon_version": info.get("version"),
                                          "started_daemon": started, "memory_dir": str(memory_dir)})

    def reconnect() -> daemon_client.DaemonEndpoint:
        # Mid-session: accept whichever version runs, so two installs never take turns restarting it.
        fresh, _info, _started = daemon_client.ensure_running(
            paths, memory_dir, settings, VERSION, restart_on_version_mismatch=False)
        return fresh

    async def bridge_main() -> None:
        bridge = Bridge(endpoint, reconnect, StdoutWriter())
        await bridge.run(stdin_frames(asyncio.get_running_loop()))

    asyncio.run(bridge_main())
    return 0
