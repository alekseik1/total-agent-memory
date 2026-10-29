"""llm_provider: bounded retry, typed unavailability, auth-failure caching, response cap."""

from __future__ import annotations

import io
import json
import sys
import urllib.error
from email.message import Message
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import llm_provider  # noqa: E402

SERVICE_UNAVAILABLE = 503
TOO_MANY_REQUESTS = 429
UNAUTHORIZED = 401
RETRY_AFTER_SEC = 7


class _Resp:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, size: int = -1) -> bytes:
        return self._payload if size < 0 else self._payload[:size]


def _http_error(code: int, retry_after: str | None = None) -> urllib.error.HTTPError:
    headers = Message()
    if retry_after:
        headers["Retry-After"] = retry_after
    return urllib.error.HTTPError("http://x", code, "err", headers, io.BytesIO(b""))


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr(llm_provider, "_sleep", sleeps.append)
    llm_provider._clear_available_cache()
    return sleeps


def _script(monkeypatch, outcomes):
    calls = {"n": 0}

    def fake(req, timeout=None, context=None):
        outcome = outcomes[min(calls["n"], len(outcomes) - 1)]
        calls["n"] += 1
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(llm_provider.urllib.request, "urlopen", fake)
    return calls


def test_retries_503_then_succeeds(monkeypatch, _no_sleep):
    calls = _script(monkeypatch, [_http_error(SERVICE_UNAVAILABLE), _Resp(b'{"response": "ok"}')])
    out = llm_provider._http_post_json("http://x/api", {}, {}, 1.0)
    assert out == {"response": "ok"}
    assert calls["n"] == 2
    assert len(_no_sleep) == 1


def test_retry_after_is_honoured(monkeypatch, _no_sleep):
    _script(monkeypatch, [_http_error(TOO_MANY_REQUESTS, str(RETRY_AFTER_SEC)), _Resp(b"{}")])
    llm_provider._http_post_json("http://x/api", {}, {}, 1.0)
    assert _no_sleep == [RETRY_AFTER_SEC]


def test_exhausted_retries_raise_typed_error(monkeypatch):
    calls = _script(monkeypatch, [urllib.error.URLError("refused")])
    with pytest.raises(llm_provider.ProviderUnavailable):
        llm_provider._http_post_json("http://x/api", {}, {}, 1.0)
    assert calls["n"] == llm_provider.DEFAULT_MAX_RETRIES + 1


def test_auth_error_is_not_retried(monkeypatch):
    calls = _script(monkeypatch, [_http_error(UNAUTHORIZED)])
    with pytest.raises(llm_provider.ProviderAuthError):
        llm_provider._http_post_json("http://x/api", {}, {}, 1.0)
    assert calls["n"] == 1


def test_anthropic_bad_key_becomes_unavailable(monkeypatch):
    provider = llm_provider.AnthropicProvider(api_key="sk-bad", api_base="http://x/v1")
    assert provider.available() is True
    _script(monkeypatch, [_http_error(UNAUTHORIZED)])
    with pytest.raises(llm_provider.ProviderAuthError):
        provider.complete("hi")
    assert provider.available() is False


def test_oversized_response_is_refused(monkeypatch):
    big = b"x" * (llm_provider.MAX_RESPONSE_BYTES + 10)
    _script(monkeypatch, [_Resp(big)])
    with pytest.raises(llm_provider.ProviderUnavailable):
        llm_provider._http_post_json("http://x/api", {}, {}, 1.0)


def test_non_retryable_client_error_propagates(monkeypatch):
    calls = _script(monkeypatch, [_http_error(400)])
    with pytest.raises(urllib.error.HTTPError):
        llm_provider._http_post_json("http://x/api", json.loads("{}"), {}, 1.0)
    assert calls["n"] == 1


def test_a_rejected_key_does_not_mark_another_key_unavailable():
    rejected = llm_provider._cache_key("openai", "https://api.example.test", "key-a")
    other = llm_provider._cache_key("openai", "https://api.example.test", "key-b")
    with pytest.raises(llm_provider.ProviderAuthError):
        with llm_provider._mark_unavailable_on_auth_error(rejected):
            raise llm_provider.ProviderAuthError("rejected")
    assert llm_provider._cache_get_available(rejected) is False
    assert llm_provider._cache_get_available(other) is None
    assert llm_provider._cache_key("openai", "https://api.example.test", "key-a") == rejected


def test_the_availability_cache_key_is_not_a_fast_hash_of_the_secret():
    import hashlib
    key = llm_provider._cache_key("anthropic", "", "secret-value")
    assert hashlib.sha1(b"secret-value").hexdigest()[:16] not in key
