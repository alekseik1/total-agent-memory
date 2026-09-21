"""Minimal client for TypeSafe's System One API (model family "Jev").

Jev answers typed questions (noul / choice / score) with calibrated
probabilities instead of generated text. Only ``POST /v1/systemone`` is
used here. Retry behaviour mirrors the official SDKs: up to two retries on
408, 429, 5xx and connection errors, exponential backoff from 0.5 s to 5 s
with up to 25 % jitter subtracted, and the server's ``retry-after-ms`` /
``Retry-After`` honoured up to 60 s. urllib only, like ``llm_provider``.
"""

from __future__ import annotations

import email.utils
import json
import random
import ssl
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from typing import Any

from memory_core.telemetry import counters, op_timer

RETRYABLE_STATUSES = frozenset({408, 429}) | frozenset(range(500, 600))
DEFAULT_TIMEOUT_SECONDS = 10.0
DEFAULT_MAX_RETRIES = 2
BACKOFF_INITIAL_SECONDS = 0.5
BACKOFF_MAX_SECONDS = 5.0
BACKOFF_JITTER = 0.25
MAX_RETRY_AFTER_SECONDS = 60.0
MAX_ERROR_BODY_BYTES = 4096
USER_AGENT = 'total-agent-memory-jev/1'


class JevConfigError(ValueError):
    """Invalid client configuration. Never carries the API key."""


class JevError(RuntimeError):
    def __init__(self, message: str, *, status: int | None = None, request_id: str | None = None):
        super().__init__(message)
        self.status, self.request_id = status, request_id


def _validated_key(api_key: str | None) -> str:
    key = (api_key or '').strip()
    if not key:
        raise JevConfigError('TypeSafe API key is empty; set TYPESAFE_API_KEY')
    if any(ord(char) < 0x20 or ord(char) == 0x7F for char in key):
        raise JevConfigError('TypeSafe API key contains control characters')
    return key


def _ssl_context() -> ssl.SSLContext:
    try:
        import certifi
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


def _retry_after_seconds(headers: Mapping[str, str]) -> float | None:
    lowered = {name.lower(): value for name, value in headers.items()}
    if (raw := lowered.get('retry-after-ms', '').strip()):
        try:
            return max(0.0, float(raw) / 1000.0)
        except ValueError:
            return None
    if not (raw := lowered.get('retry-after', '').strip()):
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        pass
    try:
        parsed = email.utils.parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return None
    return max(0.0, parsed.timestamp() - time.time())


def backoff_seconds(attempt: int, retry_after: float | None, rand: Callable[[], float] = random.random) -> float:
    """Delay before retry number ``attempt`` (1-based)."""
    if retry_after is not None and retry_after <= MAX_RETRY_AFTER_SECONDS:
        return retry_after
    base = min(BACKOFF_INITIAL_SECONDS * 2 ** (attempt - 1), BACKOFF_MAX_SECONDS)
    return base * (1.0 - BACKOFF_JITTER * rand())


def _error_message(body: bytes) -> str:
    try:
        detail = json.loads(body).get('detail')
    except (ValueError, AttributeError):
        return body[:200].decode('utf-8', 'replace')
    if isinstance(detail, dict):
        return str(detail.get('message') or detail.get('error_type') or detail)
    if isinstance(detail, list):
        return '; '.join(str(item.get('msg', item)) if isinstance(item, dict) else str(item) for item in detail)
    return str(detail)


class JevClient:
    def __init__(self, api_key: str | None, base_url: str, model: str, *,
                 timeout: float = DEFAULT_TIMEOUT_SECONDS, max_retries: int = DEFAULT_MAX_RETRIES,
                 sleep: Callable[[float], None] = time.sleep,
                 opener: Callable[..., Any] = urllib.request.urlopen):
        if not base_url.strip() or not model.strip():
            raise JevConfigError('TypeSafe base URL and model are required')
        if timeout <= 0 or max_retries < 0:
            raise JevConfigError('Timeout must be positive and max_retries non-negative')
        self._key = _validated_key(api_key)
        self.base_url, self.model = base_url.rstrip('/'), model
        self.timeout, self.max_retries = timeout, max_retries
        self._sleep, self._open = sleep, opener
        self._context = _ssl_context() if self.base_url.startswith('https://') else None

    def __repr__(self) -> str:
        return f'JevClient(base_url={self.base_url!r}, model={self.model!r})'

    def systemone(self, state: str | dict[str, Any] | list[Any], questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        """Evaluate ``questions`` against ``state``; returns the ``answers`` map."""
        if not questions:
            raise ValueError('At least one question is required')
        payload = json.dumps({'model': self.model, 'state': state, 'questions': questions},
                             ensure_ascii=False).encode('utf-8')
        with op_timer('jev_request_ms'):
            data = self._post(payload)
        answers = data.get('answers') if isinstance(data, dict) else None
        if not isinstance(answers, dict):
            raise JevError('TypeSafe response has no answers map')
        usage = data.get('usage') or {}
        counters.bump('jev_input_tokens', float(usage.get('input_tokens', 0)))
        counters.bump('jev_output_tokens', float(usage.get('output_tokens', 0)))
        return answers

    def _post(self, payload: bytes) -> dict[str, Any]:
        attempt = 0
        while True:
            attempt += 1
            headers = {'Authorization': f'Bearer {self._key}', 'Content-Type': 'application/json',
                       'Accept': 'application/json', 'User-Agent': USER_AGENT}
            if attempt > 1:
                headers['X-TypeSafe-Retry-Count'] = str(attempt - 1)
            request = urllib.request.Request(f'{self.base_url}/v1/systemone', data=payload, headers=headers, method='POST')
            counters.bump('jev_calls')
            counters.bump('network_calls')
            retry_after = None
            try:
                with self._open(request, timeout=self.timeout, context=self._context) as response:
                    return json.loads(response.read())
            except urllib.error.HTTPError as error:
                request_id = error.headers.get('x-typesafe-request-id') if error.headers else None
                message = _error_message(error.read(MAX_ERROR_BODY_BYTES))
                if error.code not in RETRYABLE_STATUSES or attempt > self.max_retries:
                    counters.bump('jev_errors')
                    raise JevError(f'TypeSafe API returned {error.code}: {message}', status=error.code,
                                   request_id=request_id) from None
                retry_after = _retry_after_seconds(dict(error.headers or {}))
            except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
                if attempt > self.max_retries:
                    counters.bump('jev_errors')
                    raise JevError(f'TypeSafe API unreachable: {type(error).__name__}') from None
            except ValueError as error:
                counters.bump('jev_errors')
                raise JevError('TypeSafe API returned invalid JSON') from error
            counters.bump('jev_retries')
            self._sleep(backoff_seconds(attempt, retry_after))
