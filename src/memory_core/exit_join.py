"""Join background threads that run native inference before the interpreter finalizes.

Model loads and embedding passes run in daemon threads, mostly inside
onnxruntime with the GIL released. When the process exits while such a thread
is still there, Python 3.11 and 3.12 end it with pthread_exit as soon as it
tries to take the GIL back. That forced unwind runs through onnxruntime's
C++ frames, which cannot be unwound, and the C++ runtime aborts the whole
process ("terminate called without an active exception", SIGABRT) after the
server has already answered. Exit hooks run before finalization, so joining
the thread there lets it leave native code first.
"""
from __future__ import annotations

import atexit
import logging
import threading
from collections.abc import Callable

LOGGER = logging.getLogger(__name__)
EXIT_JOIN_TIMEOUT_SECONDS = 10.0


def join_at_exit(thread: threading.Thread, timeout: float = EXIT_JOIN_TIMEOUT_SECONDS,
                 stop: Callable[[], None] | None = None) -> None:
    """Wait up to ``timeout`` seconds for ``thread`` when the interpreter exits, calling ``stop`` first."""

    def _join() -> None:
        if not thread.is_alive():
            return
        if stop is not None:
            stop()
        thread.join(timeout)
        if thread.is_alive():
            LOGGER.warning("background thread still running at exit", extra={
                "thread": thread.name, "timeout_seconds": timeout})

    atexit.register(_join)
