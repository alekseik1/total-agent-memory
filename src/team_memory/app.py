import asyncio
import json
import time
from contextlib import asynccontextmanager
from contextvars import ContextVar

from mcp.server import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.types import CallToolResult, TextContent, Tool
from pydantic import ValidationError
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse
from starlette.routing import Mount, Route

from team_memory.contracts import DomainError, Unauthorized
from team_memory.service import TOOLS, MemoryService
from version import RELEASE_DATE, VERSION

MAX_REQUEST_BYTES = 1_000_000
MAX_PENDING_REQUESTS = 32
REQUEST_READ_TIMEOUT_SECONDS = 15
TOKEN: ContextVar[str] = ContextVar("team_memory_token", default="")


class Authentication:
    def __init__(self, app, registry):
        self.app, self.registry = app, registry
        self.pending = 0

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["path"] in ("/", "/healthz"):
            await self.app(scope, receive, send)
            return
        headers = dict(scope["headers"])
        authorization = headers.get(b"authorization", b"").decode("latin-1")
        if not authorization.startswith("Bearer "):
            await JSONResponse({"error": "Bearer token required"}, status_code=401,
                               headers={"WWW-Authenticate": "Bearer"})(scope, receive, send)
            return
        token = authorization[len("Bearer "):]
        try:
            self.registry.authenticate(token)
        except Unauthorized:
            await JSONResponse({"error": "Invalid or revoked token"}, status_code=401)(scope, receive, send)
            return
        # Token-based requests never use cookies; browser cross-origin requests are denied.
        if b"origin" in headers:
            from urllib.parse import urlsplit
            origin = urlsplit(headers[b"origin"].decode("latin-1"))
            if origin.netloc.encode() != headers.get(b"host"):
                await JSONResponse({"error": "Cross-origin request denied"}, status_code=403)(scope, receive, send)
                return
        if self.pending >= MAX_PENDING_REQUESTS:
            await JSONResponse({"error": "Server busy"}, status_code=429)(scope, receive, send)
            return
        self.pending += 1
        context = TOKEN.set(token)
        try:
            messages, size = [], 0
            deadline = time.monotonic() + REQUEST_READ_TIMEOUT_SECONDS
            while True:
                try:
                    message = await asyncio.wait_for(receive(), max(0, deadline - time.monotonic()))
                except asyncio.TimeoutError:
                    await JSONResponse({"error": "Request body timeout"}, status_code=408)(scope, receive, send)
                    return
                if message["type"] == "http.disconnect":
                    return
                size += len(message.get("body", b""))
                if size > MAX_REQUEST_BYTES:
                    await JSONResponse({"error": "Request too large"}, status_code=413)(scope, receive, send)
                    return
                messages.append(message)
                if not message.get("more_body", False):
                    break

            async def buffered_receive():
                return messages.pop(0) if messages else await receive()

            async def private_send(message):
                if message["type"] == "http.response.start":
                    message = {**message, "headers": [*message.get("headers", []), (b"cache-control", b"no-store")]}
                await send(message)

            await self.app(scope, buffered_receive, private_send)
        finally:
            TOKEN.reset(context)
            self.pending -= 1


def create_app(service: MemoryService):
    server = Server("total-agent-memory", version=VERSION)

    async def list_tools():
        return [Tool(name=name, description=description, inputSchema=model.model_json_schema())
                for name, (model, description) in TOOLS.items()]

    async def call_tool(name, arguments):
        try:
            data = await service.call(TOKEN.get(), name, arguments or {})
            return CallToolResult(content=[TextContent(type="text", text=json.dumps(data, ensure_ascii=False))])
        except (DomainError, ValidationError) as exc:
            return CallToolResult(content=[TextContent(type="text", text=json.dumps(
                {"error": str(exc), "code": getattr(exc, "code", "invalid_request")}))], isError=True)

    if hasattr(server, "list_tools"):
        server.list_tools()(list_tools)
        server.call_tool()(call_tool)
    else:
        from mcp.types import (
            CallToolRequestParams,
            ListToolsResult,
            PaginatedRequestParams,
        )

        async def handle_list(_context, _params):
            return ListToolsResult(tools=await list_tools())

        async def handle_call(_context, params):
            return await call_tool(params.name, params.arguments)

        server.add_request_handler("tools/list", PaginatedRequestParams, handle_list)
        server.add_request_handler("tools/call", CallToolRequestParams, handle_call)
    manager = StreamableHTTPSessionManager(app=server, event_store=None, json_response=True, stateless=True)

    async def mcp(scope, receive, send):
        await manager.handle_request(scope, receive, send)

    async def health(_request):
        return JSONResponse({"status": "ok", "name": "total-agent-memory", "version": VERSION,
                             "release_date": RELEASE_DATE})

    async def invoke(request: Request):
        try:
            payload = await request.json()
            result = await service.call(TOKEN.get(), payload["name"], payload.get("arguments", {}))
            return JSONResponse(result)
        except (DomainError, ValidationError, ValueError, KeyError, TypeError) as exc:
            return JSONResponse({"error": str(exc), "code": getattr(exc, "code", "invalid_request")}, status_code=400)

    async def index(_request):
        from team_memory.ui import PAGE
        return HTMLResponse(PAGE.replace("__VERSION__", VERSION).replace("__DATE__", RELEASE_DATE),
                            headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff",
                                     "X-Frame-Options": "DENY",
                                     "Content-Security-Policy": "default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; frame-ancestors 'none'; base-uri 'none'"})

    @asynccontextmanager
    async def lifespan(_app):
        from team_memory.lifecycle import ServerLease
        with ServerLease(service.registry.root):
            async with manager.run():
                try:
                    yield
                finally:
                    await asyncio.to_thread(service.pool.close)

    app = Starlette(routes=[Route("/", index), Route("/healthz", health),
                            Route("/api/call", invoke, methods=["POST"]), Mount("/mcp", app=mcp)], lifespan=lifespan)
    return Authentication(app, service.registry)
