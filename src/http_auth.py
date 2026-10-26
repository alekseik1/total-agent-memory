"""Bearer-token gate for the HTTP servers (MCP Streamable HTTP, dashboard).

A server bound to loopback is protected by its Host/Origin allow-list against browsers.
A server bound beyond loopback (a container's 0.0.0.0, a LAN address) is reachable by
clients that choose their own Host header, so it requires a token on every request.
The token comes from ``TAM_HTTP_TOKEN`` or, when unset, from ``<memory dir>/state/http-token``
(created 0600 on first use and shared by every process that mounts the same data volume).
"""

from __future__ import annotations

import hmac
import ipaddress
import logging
import os
import secrets
import time
from collections.abc import Mapping
from pathlib import Path

LOGGER = logging.getLogger(__name__)

TOKEN_ENV = "TAM_HTTP_TOKEN"
TOKEN_FILE_RELATIVE = Path("state") / "http-token"
TOKEN_BYTES = 32
MIN_TOKEN_LENGTH = 32
GROUP_OTHER_BITS = 0o077
TOKEN_FILE_MODE = 0o600
STATE_DIR_MODE = 0o700
BEARER_PREFIX = "bearer "
UNAUTHORIZED_STATUS = 401
WS_POLICY_VIOLATION = 1008
HEALTH_PATH = "/healthz"
LOOPBACK_NAMES = frozenset(("localhost",))


def is_loopback_bind(bind: str) -> bool:
    """True only for an explicit loopback address; wildcards and empty strings are not loopback."""
    name = bind.strip().strip("[]").lower()
    if name in LOOPBACK_NAMES:
        return True
    try:
        return ipaddress.ip_address(name).is_loopback
    except ValueError:
        return False


def token_file(root: Path) -> Path:
    return root / TOKEN_FILE_RELATIVE


def load_or_create_token(root: Path, environ: Mapping[str, str] | None = None) -> str:
    """The configured token, else the persisted one; created atomically (O_EXCL) when absent."""
    env = os.environ if environ is None else environ
    configured = env.get(TOKEN_ENV, "").strip()
    if configured:
        return _check_length(configured, f"{TOKEN_ENV} value")
    path = token_file(root)
    path.parent.mkdir(mode=STATE_DIR_MODE, parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, TOKEN_FILE_MODE)
    except FileExistsError:
        _restrict_mode(path)
        return _check_length(_read_token(path), f"token file {path}")
    token = secrets.token_urlsafe(TOKEN_BYTES)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(token)
    LOGGER.info("created HTTP access token at %s", path)
    return token


def _check_length(token: str, origin: str) -> str:
    if len(token) < MIN_TOKEN_LENGTH:
        raise RuntimeError(f"{origin} is {len(token)} characters; an HTTP access token needs at least "
                           f"{MIN_TOKEN_LENGTH} (generate one with: python -c 'import secrets; "
                           f"print(secrets.token_urlsafe({TOKEN_BYTES}))')")
    return token


def _restrict_mode(path: Path) -> None:
    """A token file readable by group or others is tightened to 0600 (and the fix is logged)."""
    if os.name == "nt":
        return
    mode = path.stat().st_mode & 0o777
    if mode & GROUP_OTHER_BITS:
        LOGGER.warning("HTTP token file %s had mode %o; restricting it to %o", path, mode, TOKEN_FILE_MODE)
        path.chmod(TOKEN_FILE_MODE)


def _read_token(path: Path) -> str:
    # A sibling process may have created the file but not written it yet.
    for _ in range(50):
        token = path.read_text(encoding="utf-8").strip()
        if token:
            return token
        time.sleep(0.05)
    raise RuntimeError(f"HTTP token file {path} is empty")


def required_token(bind: str, root: Path, environ: Mapping[str, str] | None = None) -> str | None:
    """The token every request must carry for this bind address, or None for loopback binds."""
    if is_loopback_bind(bind):
        return None
    return load_or_create_token(root, environ)


def bearer_matches(authorization: str | None, token: str) -> bool:
    """Constant-time check of an ``Authorization: Bearer <token>`` header value."""
    if not authorization or not authorization.lower().startswith(BEARER_PREFIX):
        return False
    return hmac.compare_digest(authorization[len(BEARER_PREFIX):].strip().encode(), token.encode())


class BearerTokenGuard:
    """ASGI wrapper answering 401 unless the request carries the bearer token."""

    def __init__(self, inner, token: str, exempt_paths: tuple[str, ...] = (HEALTH_PATH,)):
        self.inner, self.token, self.exempt_paths = inner, token, exempt_paths

    async def __call__(self, scope, receive, send):
        if scope["type"] == "websocket":
            # The MCP transport serves no websocket; refuse before the inner app can see the connection.
            await send({"type": "websocket.close", "code": WS_POLICY_VIOLATION})
            return
        if scope["type"] == "http" and scope.get("path") not in self.exempt_paths:
            headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}
            if not bearer_matches(headers.get("authorization"), self.token):
                from starlette.responses import Response
                await Response("Unauthorized", status_code=UNAUTHORIZED_STATUS,
                               headers={"WWW-Authenticate": "Bearer"})(scope, receive, send)
                return
        await self.inner(scope, receive, send)


def guard_asgi(app, bind: str, root: Path, environ: Mapping[str, str] | None = None):
    """``app`` wrapped in a token guard when ``bind`` is not loopback; unchanged otherwise."""
    token = required_token(bind, root, environ)
    if token is None:
        return app
    LOGGER.warning("MCP HTTP bound to %s: bearer token required (see %s or %s)", bind,
                   token_file(root), TOKEN_ENV)
    return BearerTokenGuard(app, token)
