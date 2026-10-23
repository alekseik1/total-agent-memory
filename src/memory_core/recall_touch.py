"""Usage statistics of recall, written off the read path.

Every `memory_recall` used to run one UPDATE per returned record plus a
COMMIT inside the search, taking the SQLite write lock on a read. The
`RecallTouchBuffer` collects the touched ids instead (`touch` does no I/O),
coalesces repeats into one row per record, and a background thread writes
them every `flush_interval` seconds with one `executemany` in one
transaction. The thread runs only while something is pending and exits
after a failed flush (the counts stay buffered; the next touch restarts
it). `close()` writes what is left; every open buffer is closed at
interpreter exit and at the exit of a multiprocessing worker.

The writes go through a connection of their own, opened by `connect()`:
the Store's connection is shared with the request thread, and two threads
issuing DML on one sqlite3 connection interleave their implicit
transactions. A connection never sees its own commits in `PRAGMA
data_version`, so `data_version()` read through it changes on every commit
except the buffer's flushes; the recall result cache keys on it so usage
statistics do not invalidate cached results.
"""

from __future__ import annotations

import atexit
import logging
import threading
import weakref
from collections.abc import Callable, Iterable
from multiprocessing import util as _mp_util
from typing import Any

from memory_core.telemetry import counters, op_timer
from memory_core.timestamps import utc_now

LOGGER = logging.getLogger(__name__)
# How long close() waits for an in-flight flush before writing the rest itself.
SHUTDOWN_JOIN_TIMEOUT_SECONDS = 5.0
# multiprocessing runs finalizers with a priority at worker exit (atexit does not run there).
MP_EXIT_PRIORITY = 10
TOUCH_SQL = "UPDATE knowledge SET recall_count=recall_count+?, last_recalled=? WHERE id=?"

_open_buffers: weakref.WeakSet[RecallTouchBuffer] = weakref.WeakSet()
_open_buffers_lock = threading.Lock()


def close_all() -> None:
    """Close every open buffer (process exit hook; safe to call more than once)."""
    with _open_buffers_lock:
        buffers = list(_open_buffers)
    for buffer in buffers:
        buffer.close()


atexit.register(close_all)
_mp_util.Finalize(None, close_all, exitpriority=MP_EXIT_PRIORITY)


class RecallTouchBuffer:
    """Thread-safe coalescing buffer of (record id -> recall count, last recall time)."""

    def __init__(self, connect: Callable[[], Any], *, flush_interval: float,
                 clock: Callable[[], str] = utc_now, thread_name: str = "recall-touch-flush"):
        if flush_interval <= 0:
            raise ValueError("flush_interval must be positive")
        self._connect = connect
        self._flush_interval = flush_interval
        self._clock = clock
        self._thread_name = thread_name
        self._pending: dict[int, list] = {}
        self._lock = threading.Lock()
        self._io_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._connection: Any = None
        self._closed = False
        with _open_buffers_lock:
            _open_buffers.add(self)

    # ── request path ────────────────────────────────────────────────────

    def touch(self, ids: Iterable[int]) -> None:
        """Record one recall of each id; no database access while the buffer is open."""
        now = self._clock()
        with self._lock:
            _merge(self._pending, ((int(kid), 1, now) for kid in ids))
            if not self._pending:
                return
            if not self._closed:
                self._start_locked()
                return
            batch, self._pending = self._pending, {}
        # After close() there is no flusher; write in the caller rather than lose the counts.
        try:
            self._write(batch)
        finally:
            self._release_connection()

    def pending(self) -> int:
        """Number of distinct records waiting for the next flush."""
        with self._lock:
            return len(self._pending)

    @property
    def closed(self) -> bool:
        return self._closed

    def data_version(self) -> int | None:
        """`PRAGMA data_version` of the flush connection: changes on every commit but the buffer's own.

        Never waits: None while a flush holds the connection (it may be
        waiting for the database write lock) and once the buffer is closed
        (a fresh connection would restart the counter). The caller then
        treats the data as changed or falls back to its own connection.
        """
        if not self._io_lock.acquire(blocking=False):
            return None
        try:
            if self._closed:
                return None
            return int(self._connection_locked().execute("PRAGMA data_version").fetchone()[0])
        finally:
            self._io_lock.release()

    # ── flushing ────────────────────────────────────────────────────────

    def flush(self) -> int:
        """Write everything pending now; returns the number of records written.

        On a database error the batch is merged back into the buffer (counts
        add up, the later timestamp wins) and the error propagates.
        """
        with self._lock:
            batch, self._pending = self._pending, {}
        if not batch:
            return 0
        try:
            self._write(batch)
        except Exception:
            with self._lock:
                _merge(self._pending, ((kid, count, last) for kid, (count, last) in batch.items()))
            raise
        return len(batch)

    def close(self) -> None:
        """Stop the flusher, write what is pending and close the connection (idempotent)."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            thread = self._thread
        self._stop.set()
        if thread is not None and thread is not threading.current_thread():
            thread.join(SHUTDOWN_JOIN_TIMEOUT_SECONDS)
        try:
            self.flush()
        except Exception:
            counters.bump("recall_touch_flush_errors")
            LOGGER.exception("Recall usage statistics lost at shutdown", extra={"pending": self.pending()})
        finally:
            self._release_connection()
            with _open_buffers_lock:
                _open_buffers.discard(self)

    def _start_locked(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name=self._thread_name, daemon=True)
            self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self._flush_interval):
            try:
                self.flush()
            except Exception:
                counters.bump("recall_touch_flush_errors")
                LOGGER.exception("Recall usage flush failed; counts kept until the next recall",
                                 extra={"pending": self.pending()})
                with self._lock:
                    self._thread = None
                return
            with self._lock:
                if not self._pending:
                    self._thread = None
                    return

    def _connection_locked(self):
        if self._connection is None:
            self._connection = self._connect()
        return self._connection

    def _release_connection(self) -> None:
        with self._io_lock:
            connection, self._connection = self._connection, None
        if connection is not None:
            connection.close()

    def _write(self, batch: dict[int, list]) -> None:
        rows = [(count, last, kid) for kid, (count, last) in sorted(batch.items())]
        with self._io_lock, op_timer("recall_touch_flush_ms"):
            connection = self._connection_locked()
            try:
                connection.executemany(TOUCH_SQL, rows)
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        counters.bump("recall_touch_flushes")
        counters.bump("recall_touch_rows", len(rows))


def _merge(pending: dict[int, list], items: Iterable[tuple[int, int, str]]) -> None:
    for kid, count, last in items:
        entry = pending.get(kid)
        if entry is None:
            pending[kid] = [count, last]
        else:
            entry[0] += count
            entry[1] = max(entry[1], last)
