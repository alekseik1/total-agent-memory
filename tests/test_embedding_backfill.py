"""Unit tests for the embedding_pending queue (src/embedding_backfill.py)."""

from __future__ import annotations

import sqlite3
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import embedding_backfill as eb
from embedding_backfill import BackfillOutcome, EmbeddingBackfill, EmbeddingUnavailable

MIGRATION = Path(__file__).parent.parent / "migrations" / "039_write_path_reliability.sql"
T0 = 1_800_000_000.0
LOCK_WAIT_SECONDS = 10
ROWS = 40


class Clock:
    def __init__(self, now: float = T0):
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _schema(db):
    # 039 also alters write_intents; give it the table it expects.
    db.executescript((Path(__file__).parent.parent / "migrations" / "017_outbox.sql").read_text())
    db.executescript(MIGRATION.read_text())


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def queue(clock):
    db = sqlite3.connect(":memory:")
    _schema(db)
    yield EmbeddingBackfill(db, clock=clock, max_attempts=3, base_backoff=60, max_backoff=600)
    db.close()


def _status(queue, kid):
    row = queue.db.execute("SELECT attempts, status FROM embedding_pending WHERE knowledge_id=?",
                           (kid,)).fetchone()
    return tuple(row) if row else None


def test_enqueue_is_idempotent(queue):
    assert queue.enqueue(7, "provider timeout") is True
    assert queue.enqueue(7, "again") is False
    assert queue.counts() == {"pending": 1, "failed": 0, "due": 1}


def test_backoff_is_exponential_and_capped():
    assert [eb.backoff_seconds(n, base=60, cap=600) for n in (1, 2, 3, 4, 5)] == [60, 120, 240, 480, 600]
    assert eb.backoff_seconds(10_000) == eb.MAX_BACKOFF_SECONDS


def test_claim_leases_rows_until_the_lease_expires(queue, clock):
    queue.enqueue(1)
    queue.enqueue(2)
    assert [item.knowledge_id for item in queue.claim(10)] == [1, 2]
    assert queue.claim(10) == []
    clock.advance(queue.lease_seconds + 1)
    assert [item.knowledge_id for item in queue.claim(10)] == [1, 2]


def test_failures_back_off_then_dead_letter(queue, clock):
    queue.enqueue(5)

    def failing(_kid):
        raise EmbeddingUnavailable("provider down")

    first = queue.drain(failing)
    assert first.retried == 1 and _status(queue, 5) == (1, "pending")
    assert queue.drain(failing).claimed == 0          # backing off: 60 s
    clock.advance(60)
    assert queue.drain(failing).retried == 1 and _status(queue, 5) == (2, "pending")
    clock.advance(119)
    assert queue.drain(failing).claimed == 0          # second delay is 120 s
    clock.advance(1)
    assert queue.drain(failing).dead_lettered == 1 and _status(queue, 5) == (3, "failed")
    clock.advance(10_000)
    assert queue.drain(failing).claimed == 0          # dead letters stay put
    assert queue.counts()["failed"] == 1

    assert queue.retry_dead_letter([5]) == 1
    assert _status(queue, 5) == (0, "pending")
    result = queue.drain(lambda _kid: BackfillOutcome.EMBEDDED)
    assert result.embedded == 1 and _status(queue, 5) is None


def test_retry_dead_letter_without_ids_resets_all_and_ignores_pending(queue):
    for kid in (1, 2, 3):
        queue.enqueue(kid)
    queue.db.execute("UPDATE embedding_pending SET status='failed', attempts=3 WHERE knowledge_id IN (1, 2)")
    queue.db.commit()
    assert queue.retry_dead_letter() == 2
    assert queue.retry_dead_letter([]) == 0
    assert queue.counts()["failed"] == 0


def test_first_failure_defers_the_rest_without_spending_attempts(queue):
    for kid in (1, 2, 3):
        queue.enqueue(kid)
    seen = []

    def down(kid):
        seen.append(kid)
        raise TimeoutError("read timed out")

    errors = []
    result = queue.drain(down, on_error=lambda *args: errors.append(args))
    assert seen == [1] and result.retried == 1 and result.deferred == 2
    assert [_status(queue, kid) for kid in (1, 2, 3)] == [(1, "pending"), (0, "pending"), (0, "pending")]
    assert errors[0][0] == "embedding_backfill" and isinstance(errors[0][1], TimeoutError)
    assert queue.counts()["due"] == 0


def test_drain_without_stop_on_failure_tries_every_row(queue):
    for kid in (1, 2):
        queue.enqueue(kid)
    result = queue.drain(lambda kid: (_ for _ in ()).throw(EmbeddingUnavailable("x")), stop_on_failure=False)
    assert result.retried == 2 and result.deferred == 0


def test_gone_records_leave_the_queue(queue):
    queue.enqueue(9)
    assert queue.drain(lambda _kid: BackfillOutcome.GONE).gone == 1
    assert _status(queue, 9) is None


def test_concurrent_claimers_never_get_the_same_row(tmp_path):
    path = tmp_path / "queue.db"
    setup = sqlite3.connect(path)
    _schema(setup)
    seeding = EmbeddingBackfill(setup)
    for kid in range(1, ROWS + 1):
        seeding.enqueue(kid)
    setup.close()

    claimed: list[list[int]] = [[], []]
    barrier = threading.Barrier(2)

    def worker(slot):
        conn = sqlite3.connect(path, timeout=LOCK_WAIT_SECONDS)
        try:
            queue = EmbeddingBackfill(conn)
            barrier.wait()
            while batch := queue.claim(3):
                claimed[slot].extend(item.knowledge_id for item in batch)
        finally:
            conn.close()

    threads = [threading.Thread(target=worker, args=(slot,)) for slot in (0, 1)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    everything = claimed[0] + claimed[1]
    assert sorted(everything) == list(range(1, ROWS + 1))
    assert not set(claimed[0]) & set(claimed[1])


def test_max_attempts_must_be_positive():
    with pytest.raises(ValueError):
        EmbeddingBackfill(sqlite3.connect(":memory:"), max_attempts=0)
