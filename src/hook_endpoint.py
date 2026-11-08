"""Server side of remote hooks mode: ``POST /internal/hook/{cmd}``.

With ``TAM_REMOTE_URL`` set, hooks on the laptop stop opening ``memory.db``
and POST each hook command here instead (docs/remote-hooks-plan.md, slice S1).
The body is ``{"call": <HookCall JSON>, "payload": {...cmd-specific fields}}``;
the response is ``{"ok": true, "output": <text the local hook prints>}`` —
"" for the writers — or ``{"ok": false, "error": "<class>: <msg>"}``.

The server never reads laptop paths: ``session-end`` receives pre-extracted
text and message lists, and its LLM work runs on a background thread after
the 202 is sent. ``memory_dir`` is resolved per request so ``TAM_MEMORY_DIR``
is honoured the way the ambient commands honour it.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
import os
import sqlite3
import threading
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from ambient.hooks import HookCall, HookCallError, record_actions, run as run_ambient
from paths import memory_dir as resolve_memory_dir

LOGGER = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
TAM_HOOK_PATH = REPO_ROOT / "hooks" / "lib" / "tam_hook.py"

BODY_LIMIT_BYTES = 256 * 1024
TIMEOUT_ENV = "TAM_HOOK_TIMEOUT_SEC"
DEFAULT_TIMEOUT_SEC = 10.0
SQLITE_TIMEOUT_SEC = 5.0

# session-end payload bounds (the client clamps to the same values; the server
# re-clamps so a hand-rolled client cannot push unbounded text into the DB).
USER_TEXTS_MAX = 5
USER_TEXT_CHARS = 500
ASSISTANT_TEXT_CHARS = 2000
EPISODE_MESSAGES_MAX_BYTES = 64 * 1024
EPISODE_ROLES = frozenset({"user", "assistant", "tool"})

SESSION_START = "session-start"
SESSION_END = "session-end"

_BACKGROUND_LOCK = threading.Lock()
_BACKGROUND_THREADS: set[threading.Thread] = set()

_tam_hook_module: Any = None


def _load_tam_hook() -> Any:
    """hooks/lib/tam_hook.py via importlib: stdlib-only, safe inside the server."""
    global _tam_hook_module
    if _tam_hook_module is None:
        spec = importlib.util.spec_from_file_location("tam_hook_remote", TAM_HOOK_PATH)
        if spec is None or spec.loader is None:
            raise HookCallError(f"cannot load hook runtime from {TAM_HOOK_PATH}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _tam_hook_module = module
    return _tam_hook_module


def _ambient(command: str) -> Callable[[HookCall, Mapping[str, Any], Path], tuple[int, str]]:
    def handler(call: HookCall, payload: Mapping[str, Any], memory_dir: Path) -> tuple[int, str]:
        return 200, run_ambient(command, call, memory_dir)

    return handler


def _session_start(call: HookCall, payload: Mapping[str, Any], memory_dir: Path) -> tuple[int, str]:
    tam_hook = _load_tam_hook()
    text = tam_hook.session_start_context(
        db_path=memory_dir / "memory.db",
        project=call.project,
        cwd=call.cwd,
        session=call.session_id,
        source=str(call.payload.get("source") or ""),
        recovery_dir=memory_dir / "recovery",
    )
    return 200, text


def _save_intent(call: HookCall, payload: Mapping[str, Any], memory_dir: Path) -> tuple[int, str]:
    db_path = memory_dir / "memory.db"
    if db_path.is_file() and call.prompt:
        from intents import save_intent

        save_intent(db_path, call.prompt, call.session_id, call.project)
    return 200, ""


def _actions_batch(call: HookCall, payload: Mapping[str, Any], memory_dir: Path) -> tuple[int, str]:
    raw = payload.get("actions")
    actions = [item for item in raw if isinstance(item, Mapping)] if isinstance(raw, list) else []
    record_actions(call, actions, memory_dir)
    return 200, ""


def _error_capture(call: HookCall, payload: Mapping[str, Any], memory_dir: Path) -> tuple[int, str]:
    import auto_self_improve

    action = str(payload.get("action") or "")
    project = str(payload.get("project") or call.project or "general")
    db_path = memory_dir / "memory.db"
    db = sqlite3.connect(str(db_path), timeout=SQLITE_TIMEOUT_SEC)
    db.row_factory = sqlite3.Row
    try:
        db.execute("PRAGMA journal_mode=WAL")
        if action == "error":
            tags = payload.get("tags")
            auto_self_improve.log_error(
                db,
                description=str(payload.get("description") or ""),
                category=str(payload.get("category") or "general"),
                severity=str(payload.get("severity") or "medium"),
                fix=str(payload.get("fix") or ""),
                context=str(payload.get("context") or ""),
                project=project,
                tags=[str(t) for t in tags] if isinstance(tags, list) else None,
            )
        elif action == "fix":
            auto_self_improve.log_fix(
                db,
                description=str(payload.get("description") or ""),
                category=str(payload.get("category") or "general"),
                fix=str(payload.get("fix") or ""),
                project=project,
            )
        elif action == "check-patterns":
            try:
                days = int(payload.get("days") or 30)
            except (TypeError, ValueError):
                days = 30
            auto_self_improve.check_patterns(db, project=project, days=days)
        else:
            raise HookCallError(f"unknown error-capture action {action!r}")
    finally:
        db.close()
    return 200, ""


def _user_texts(payload: Mapping[str, Any]) -> list[str]:
    raw = payload.get("user_texts")
    if not isinstance(raw, list):
        return []
    return [text for text in
            (str(item)[:USER_TEXT_CHARS] for item in raw[:USER_TEXTS_MAX]) if text.strip()]


def _episode_messages(raw: Any) -> list[dict[str, str]]:
    """The pre-extracted {role, text} list, re-clamped to the 64 KB budget."""
    if not isinstance(raw, list):
        return []
    messages: list[dict[str, str]] = []
    size = 0
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        role = str(item.get("role") or "")
        text = str(item.get("text") or "")
        if role not in EPISODE_ROLES or not text.strip():
            continue
        entry = {"role": role, "text": text}
        cost = len(json.dumps(entry, ensure_ascii=False).encode("utf-8"))
        if size + cost > EPISODE_MESSAGES_MAX_BYTES:
            break
        size += cost
        messages.append(entry)
    return messages


def _session_end_background(session_id: str, db_path: str, project: str, assistant_text: str,
                            branch: str, reason: str, messages: list[dict[str, str]]) -> None:
    """LLM work of session-end, off the request path; failures are logged, never raised."""
    try:
        import auto_episode_capture

        auto_episode_capture.capture_episode(session_id, db_path, project, messages=messages)
    except Exception as exc:  # noqa: BLE001 — background hook work; the 202 is already sent
        LOGGER.warning("session-end episode capture failed for %s: %s: %s",
                       session_id, type(exc).__name__, exc)
    try:
        import auto_session_end

        auto_session_end.run_session_end(
            session_id=session_id, project=project, assistant_text=assistant_text,
            branch=branch, reason=reason, db_path=db_path)
    except Exception as exc:  # noqa: BLE001 — same boundary as above
        LOGGER.warning("session-end summary failed for %s: %s: %s",
                       session_id, type(exc).__name__, exc)
    finally:
        with _BACKGROUND_LOCK:
            _BACKGROUND_THREADS.discard(threading.current_thread())


def _session_end(call: HookCall, payload: Mapping[str, Any], memory_dir: Path) -> tuple[int, str]:
    """Cheap writes now (autosave fact + closed sessions row), LLM work after the 202."""
    import auto_session_save
    from session_continuity import SessionContinuity

    project = str(payload.get("project") or call.project or "general")
    reason = str(payload.get("reason") or "")
    branch = str(payload.get("branch") or "")
    assistant_text = str(payload.get("assistant_text") or "")[:ASSISTANT_TEXT_CHARS]
    user = "\n".join(_user_texts(payload))
    messages = _episode_messages(payload.get("episode_messages"))
    db_path = memory_dir / "memory.db"

    auto_session_save.save_session_context(
        project, call.cwd, reason,
        user_context=user, assistant_context=assistant_text[:1000],
        db_path=db_path)
    db = sqlite3.connect(str(db_path), timeout=SQLITE_TIMEOUT_SEC)
    db.row_factory = sqlite3.Row
    try:
        SessionContinuity(db).close_session_row(
            call.session_id, project=project, branch=branch or None)
        db.commit()
    finally:
        db.close()

    thread = threading.Thread(
        target=_session_end_background,
        args=(call.session_id, str(db_path), project, assistant_text[:1000],
              branch, reason, messages),
        name=f"tam-hook-session-end-{call.session_id[:32]}",
        daemon=True,
    )
    with _BACKGROUND_LOCK:
        _BACKGROUND_THREADS.add(thread)
    thread.start()
    return 202, ""


Handler = Callable[[HookCall, Mapping[str, Any], Path], tuple[int, str]]

_HANDLERS: dict[str, Handler] = {
    SESSION_START: _session_start,
    "prompt-context": _ambient("prompt-context"),
    "save-intent": _save_intent,
    "actions-batch": _actions_batch,
    "turn-end": _ambient("turn-end"),
    SESSION_END: _session_end,
    "error-capture": _error_capture,
}
COMMANDS = frozenset(_HANDLERS)


def dispatch(command: str, call: HookCall, payload: Mapping[str, Any], memory_dir: Path) -> tuple[int, str]:
    """Run one hook command against ``memory_dir``; returns (http_status, output)."""
    return _HANDLERS[command](call, payload, Path(memory_dir))


def join_background_threads(timeout: float | None = None) -> None:
    """Wait for in-flight session-end background work (tests, shutdown)."""
    with _BACKGROUND_LOCK:
        threads = list(_BACKGROUND_THREADS)
    deadline = None if timeout is None else time.monotonic() + timeout
    for thread in threads:
        remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
        thread.join(timeout=remaining)


def _timeout_seconds() -> float:
    raw = os.environ.get(TIMEOUT_ENV, "").strip()
    try:
        value = float(raw) if raw else DEFAULT_TIMEOUT_SEC
    except ValueError:
        return DEFAULT_TIMEOUT_SEC
    return value if value > 0 else DEFAULT_TIMEOUT_SEC


def _respond(status: int, error: str):
    from starlette.responses import JSONResponse

    return JSONResponse({"ok": False, "error": error}, status_code=status)


async def handle(request) -> Any:
    """Starlette endpoint for ``Route("/internal/hook/{cmd}", ..., methods=["POST"])``."""
    from starlette.responses import JSONResponse

    cmd = str(request.path_params.get("cmd") or "")
    if cmd not in _HANDLERS:
        return _respond(404, f"unknown hook command {cmd!r}")
    if not request.headers.get("content-type", "").lower().startswith("application/json"):
        return _respond(415, "expected an application/json body")
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > BODY_LIMIT_BYTES:
        return _respond(413, f"body exceeds {BODY_LIMIT_BYTES} bytes")
    body = await request.body()
    if len(body) > BODY_LIMIT_BYTES:
        return _respond(413, f"body exceeds {BODY_LIMIT_BYTES} bytes")
    try:
        data = json.loads(body)
    except ValueError as exc:
        return _respond(400, f"{type(exc).__name__}: {exc}")
    if not isinstance(data, dict):
        return _respond(400, "body must be a JSON object")
    call_raw = data.get("call")
    if not isinstance(call_raw, dict):
        return _respond(400, "missing 'call' object")
    payload = data.get("payload")
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        return _respond(400, "'payload' must be an object")
    try:
        call = HookCall.from_json(json.dumps(call_raw))
    except HookCallError as exc:
        return _respond(400, f"{type(exc).__name__}: {exc}")

    timeout = _timeout_seconds()
    try:
        status, output = await asyncio.wait_for(
            asyncio.to_thread(dispatch, cmd, call, payload, resolve_memory_dir()),
            timeout=timeout,
        )
    except TimeoutError:
        return _respond(500, f"TimeoutError: hook command {cmd!r} exceeded {TIMEOUT_ENV}={timeout:g}s")
    except Exception as exc:  # noqa: BLE001 — a hook failure is a 500, never a crashed server
        LOGGER.warning("hook command %s failed: %s: %s", cmd, type(exc).__name__, exc)
        return _respond(500, f"{type(exc).__name__}: {exc}")
    return JSONResponse({"ok": True, "output": output}, status_code=status)
