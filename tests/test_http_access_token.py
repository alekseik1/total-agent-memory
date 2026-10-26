"""Servers bound beyond loopback require a token; endpoint settings need an explicit confirmation."""

from __future__ import annotations

import asyncio
import http.client
import json
import os
import socket
import sqlite3
import sys
import threading
from http.server import HTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import dashboard
import dashboard_login
import http_auth

TOKEN = "t0ken-for-tests-0123456789-abcdefgh"


def test_loopback_detection():
    for bind in ("127.0.0.1", "localhost", "::1", "[::1]", "127.0.0.5"):
        assert http_auth.is_loopback_bind(bind), bind
    for bind in ("0.0.0.0", "::", "", "192.168.1.5", "mem.lan"):
        assert not http_auth.is_loopback_bind(bind), bind


def test_token_is_generated_once_with_owner_only_permissions(tmp_path):
    first = http_auth.load_or_create_token(tmp_path, {})
    assert http_auth.load_or_create_token(tmp_path, {}) == first
    path = http_auth.token_file(tmp_path)
    assert path.read_text(encoding="utf-8") == first
    if os.name != "nt":
        assert path.stat().st_mode & 0o777 == 0o600


def test_env_token_wins_and_loopback_needs_none(tmp_path):
    assert http_auth.required_token("0.0.0.0", tmp_path, {http_auth.TOKEN_ENV: TOKEN}) == TOKEN
    assert http_auth.required_token("127.0.0.1", tmp_path, {}) is None
    assert not http_auth.token_file(tmp_path).exists()


def _call(app, headers: dict[str, str], path: str = "/mcp") -> int:
    status: list[int] = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        if message["type"] == "http.response.start":
            status.append(message["status"])

    scope = {"type": "http", "path": path, "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()]}
    asyncio.run(app(scope, receive, send))
    return status[0]


def test_asgi_guard_rejects_host_spoof_without_token(tmp_path):
    async def inner(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    guarded = http_auth.guard_asgi(inner, "0.0.0.0", tmp_path, {http_auth.TOKEN_ENV: TOKEN})
    assert _call(guarded, {"Host": "localhost:3737"}) == 401
    assert _call(guarded, {"Host": "localhost", "Authorization": "Bearer wrong"}) == 401
    assert _call(guarded, {"Authorization": f"Bearer {TOKEN}"}) == 200
    assert _call(guarded, {}, path="/healthz") == 200
    assert http_auth.guard_asgi(inner, "127.0.0.1", tmp_path, {}) is inner


# ── dashboard ────────────────────────────────────────────────────

def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def port(tmp_path, monkeypatch):
    db_path = tmp_path / "memory.db"
    with sqlite3.connect(db_path) as db:
        db.execute("CREATE TABLE knowledge (id INTEGER PRIMARY KEY, status TEXT)")
    monkeypatch.setattr(dashboard, "DB_PATH", db_path)

    class _Threaded(HTTPServer):
        daemon_threads = True

    server = _Threaded(("127.0.0.1", _free_port()), dashboard.DashboardHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server.server_address[1]
    server.shutdown()
    server.server_close()


def req(port, method, path, headers=None, body=None, host="localhost"):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        conn.putrequest(method, path, skip_host=True, skip_accept_encoding=True)
        conn.putheader("Host", f"{host}:{port}")
        for name, value in (headers or {}).items():
            conn.putheader(name, value)
        if body is not None:
            conn.putheader("Content-Length", str(len(body)))
        conn.endheaders(body)
        response = conn.getresponse()
        headers: dict[str, str] = {}
        for name, value in response.getheaders():
            headers[name.lower()] = f"{headers[name.lower()]}\n{value}" if name.lower() in headers else value
        return response.status, headers, response.read()
    finally:
        conn.close()


def _session_cookie(headers) -> str:
    return next(line.split(";")[0] for line in headers["set-cookie"].splitlines()
                if line.startswith(dashboard_login.SESSION_COOKIE + "="))


def test_dashboard_with_token_refuses_spoofed_localhost_host(port, monkeypatch):
    monkeypatch.setattr(dashboard, "ACCESS_TOKEN", TOKEN)
    assert req(port, "GET", "/api/stats")[0] == 401
    assert req(port, "GET", "/api/stats", {"Authorization": f"Bearer {TOKEN}"})[0] != 401
    assert req(port, "GET", "/healthz")[0] == 200
    assert req(port, "GET", "/login?token=bad")[0] == 401


def test_cookie_holds_an_opaque_session_not_the_token(port, monkeypatch):
    monkeypatch.setattr(dashboard, "ACCESS_TOKEN", TOKEN)
    status, headers, _ = req(port, "GET", f"/login?token={TOKEN}")
    assert status == 303 and headers["location"] == "/"
    assert "HttpOnly" in headers["set-cookie"] and "SameSite=Strict" in headers["set-cookie"]
    assert TOKEN not in "".join(headers.values())
    assert headers["referrer-policy"] == "no-referrer" and headers["cache-control"] == "no-store"
    cookie = _session_cookie(headers)
    assert cookie.startswith("tam_dashboard_session=") and TOKEN not in cookie
    assert req(port, "GET", "/api/stats", {"Cookie": cookie})[0] != 401
    assert all(TOKEN not in key for key in dashboard.SESSIONS._rows)
    assert cookie.split("=", 1)[1] not in dashboard.SESSIONS._rows


def test_the_old_raw_token_cookie_no_longer_authenticates(port, monkeypatch):
    monkeypatch.setattr(dashboard, "ACCESS_TOKEN", TOKEN)
    assert req(port, "GET", "/api/stats", {"Cookie": f"tam_dashboard_token={TOKEN}"})[0] == 401


def test_login_page_reads_the_fragment_and_posts_the_token(port, monkeypatch):
    monkeypatch.setattr(dashboard, "ACCESS_TOKEN", TOKEN)
    status, headers, body = req(port, "GET", "/login")
    page = body.decode()
    assert status == 200 and 'method="post" action="/login"' in page and "#token=" in page
    assert "history.replaceState" in page and TOKEN not in page
    assert "nonce-" in headers["content-security-policy"] and "no-referrer" == headers["referrer-policy"]
    form = {"Content-Type": "application/x-www-form-urlencoded", "Origin": f"http://localhost:{port}"}
    assert req(port, "POST", "/login", form, b"token=bad")[0] == 401
    status, headers, _ = req(port, "POST", "/login", form, f"token={TOKEN}".encode())
    assert status == 303 and headers["location"] == "/" and headers["referrer-policy"] == "no-referrer"
    assert req(port, "GET", "/api/stats", {"Cookie": _session_cookie(headers)})[0] != 401


def test_login_post_body_is_bounded(port, monkeypatch):
    monkeypatch.setattr(dashboard, "ACCESS_TOKEN", TOKEN)
    body = b"token=" + b"a" * (dashboard_login.LOGIN_BODY_MAX_BYTES + 1)
    assert req(port, "POST", "/login", {"Content-Type": "application/x-www-form-urlencoded"}, body)[0] == 413


def test_non_ascii_digits_in_content_length_are_a_400_not_a_crash(port, monkeypatch):
    monkeypatch.setattr(dashboard, "ACCESS_TOKEN", TOKEN)
    form = {"Content-Type": "application/x-www-form-urlencoded", "Content-Length": "\u00b2"}
    assert req(port, "POST", "/login", form)[0] == 400
    settings = {"Content-Type": "application/json", "Content-Length": "\u00b2", "Origin": f"http://localhost:{port}",
                "Authorization": f"Bearer {TOKEN}", dashboard.CSRF_HEADER: dashboard.CSRF_TOKEN}
    assert req(port, "POST", "/api/settings", settings)[0] == 400


def test_session_cookie_is_secure_only_on_https(port, monkeypatch):
    monkeypatch.setattr(dashboard, "ACCESS_TOKEN", TOKEN)
    monkeypatch.delenv(dashboard.TRUSTED_PROXIES_ENV, raising=False)
    plain = req(port, "GET", f"/login?token={TOKEN}")[1]["set-cookie"]
    assert "Secure" not in plain
    spoofed = req(port, "GET", f"/login?token={TOKEN}", {"X-Forwarded-Proto": "https"})[1]["set-cookie"]
    assert "Secure" not in spoofed, "a forwarded proto from an untrusted peer is ignored"
    monkeypatch.setenv(dashboard.TRUSTED_PROXIES_ENV, "192.0.2.1, 127.0.0.1")
    proxied = req(port, "GET", f"/login?token={TOKEN}", {"X-Forwarded-Proto": "https"})[1]["set-cookie"]
    assert "; Secure" in proxied.split("\n")[0]
    assert "Secure" not in req(port, "GET", f"/login?token={TOKEN}", {"X-Forwarded-Proto": "http"})[1]["set-cookie"]


def test_logout_ends_the_session(port, monkeypatch):
    monkeypatch.setattr(dashboard, "ACCESS_TOKEN", TOKEN)
    cookie = _session_cookie(req(port, "GET", f"/login?token={TOKEN}")[1])
    assert req(port, "GET", "/api/stats", {"Cookie": cookie})[0] != 401
    assert req(port, "POST", "/logout", {"Cookie": cookie})[0] == 303
    assert req(port, "GET", "/api/stats", {"Cookie": cookie})[0] == 401


def test_sessions_expire_and_die_with_a_token_change():
    now = [1000.0]
    sessions = dashboard_login.DashboardSessions(clock=lambda: now[0], ttl=60, capacity=2)
    first = sessions.open(TOKEN)
    assert sessions.valid(first, TOKEN) and not sessions.valid(first, TOKEN + "x") and not sessions.valid("", TOKEN)
    now[0] += 61
    assert not sessions.valid(first, TOKEN)
    older, newer = sessions.open(TOKEN), sessions.open(TOKEN)
    now[0] += 1
    sessions.open(TOKEN)
    assert not sessions.valid(older, TOKEN) and sessions.valid(newer, TOKEN)


def test_dashboard_without_token_unchanged_for_loopback(port, monkeypatch):
    monkeypatch.setattr(dashboard, "ACCESS_TOKEN", None)
    assert req(port, "GET", "/api/release")[0] == 200


def _settings_post(port, values, confirm=False):
    payload = {"values": values}
    if confirm:
        payload["confirm_endpoint"] = True
    body = json.dumps(payload).encode()
    return req(port, "POST", "/api/settings", {
        "Origin": f"http://localhost:{port}", "Content-Type": "application/json",
        dashboard.CSRF_HEADER: dashboard.CSRF_TOKEN}, body)


def test_endpoint_change_requires_confirmation(port, monkeypatch):
    saved: list[dict] = []
    monkeypatch.setattr(dashboard, "ACCESS_TOKEN", None)
    monkeypatch.setattr(dashboard.dashboard_settings, "save", lambda root, body: saved.append(body) or {"changed": 1})
    status, _, raw = _settings_post(port, {"MEMORY_LLM_API_BASE": "https://attacker.example/v1"})
    assert status == 409 and not saved
    assert json.loads(raw)["confirm_required"] == ["MEMORY_LLM_API_BASE"]
    assert _settings_post(port, {"MEMORY_LLM_MODEL": "m"})[0] == 200
    assert _settings_post(port, {"MEMORY_EMBED_API_BASE": "https://e.example"}, confirm=True)[0] == 200
    assert len(saved) == 2


def test_endpoint_keys_cover_llm_embed_and_ollama():
    assert {"MEMORY_LLM_API_BASE", "MEMORY_EMBED_API_BASE", "OLLAMA_URL"} <= dashboard.ENDPOINT_SETTING_KEYS


def test_short_env_token_is_refused_at_startup(tmp_path):
    with pytest.raises(RuntimeError, match="at least 32"):
        http_auth.required_token("0.0.0.0", tmp_path, {http_auth.TOKEN_ENV: "short-token"})


def test_short_token_file_is_refused(tmp_path):
    path = http_auth.token_file(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text("short", encoding="utf-8")
    path.chmod(0o600)
    with pytest.raises(RuntimeError, match="at least 32"):
        http_auth.load_or_create_token(tmp_path, {})


@pytest.mark.skipif(os.name == "nt", reason="POSIX modes")
def test_loose_token_file_mode_is_tightened(tmp_path, caplog):
    path = http_auth.token_file(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(TOKEN, encoding="utf-8")
    path.chmod(0o644)
    with caplog.at_level("WARNING", logger=http_auth.LOGGER.name):
        assert http_auth.load_or_create_token(tmp_path, {}) == TOKEN
    assert path.stat().st_mode & 0o777 == 0o600
    assert "restricting" in caplog.text


def test_guard_closes_websocket_connections_before_the_inner_app(tmp_path):
    reached: list[str] = []
    sent: list[dict] = []

    async def inner(scope, receive, send):
        reached.append(scope["type"])

    async def receive():
        return {"type": "websocket.connect"}

    async def send(message):
        sent.append(message)

    guarded = http_auth.guard_asgi(inner, "0.0.0.0", tmp_path, {http_auth.TOKEN_ENV: TOKEN})
    asyncio.run(guarded({"type": "websocket", "path": "/mcp", "headers": []}, receive, send))
    assert reached == []
    assert sent == [{"type": "websocket.close", "code": 1008}]
    asyncio.run(guarded({"type": "lifespan"}, receive, send))
    assert reached == ["lifespan"]
