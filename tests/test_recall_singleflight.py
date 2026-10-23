"""Concurrent identical recalls share one execution; result cache returns private copies."""

from __future__ import annotations

import threading
import time

import pytest

import server
from memory_core.singleflight import SingleFlight

WAIT_SECONDS = 5.0
POLL_SECONDS = 0.005


def _wait_until(predicate) -> bool:
    deadline = time.monotonic() + WAIT_SECONDS
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(POLL_SECONDS)
    return predicate()


def test_waiter_reuses_leader_result():
    flight: SingleFlight[int] = SingleFlight("test")
    started, release = threading.Event(), threading.Event()
    calls = []

    def slow():
        calls.append(1)
        started.set()
        release.wait(WAIT_SECONDS)
        return 42

    results = []
    leader = threading.Thread(target=lambda: results.append(flight.run("k", slow, timeout=WAIT_SECONDS)))
    leader.start()
    assert started.wait(WAIT_SECONDS)
    waiter = threading.Thread(target=lambda: results.append(flight.run("k", slow, timeout=WAIT_SECONDS)))
    waiter.start()
    assert _wait_until(lambda: flight.waiting() == 1)
    release.set()
    leader.join(WAIT_SECONDS)
    waiter.join(WAIT_SECONDS)
    assert sorted(results) == [(42, False), (42, True)]
    assert len(calls) == 1
    assert flight.in_flight() == 0


def test_waiter_times_out_and_runs_itself():
    flight: SingleFlight[str] = SingleFlight("test")
    started, release = threading.Event(), threading.Event()

    def stuck():
        started.set()
        release.wait(WAIT_SECONDS)
        return "leader"

    leader = threading.Thread(target=lambda: flight.run("k", stuck, timeout=WAIT_SECONDS))
    leader.start()
    assert started.wait(WAIT_SECONDS)
    try:
        assert flight.run("k", lambda: "own", timeout=0.01) == ("own", False)
    finally:
        release.set()
        leader.join(WAIT_SECONDS)


def test_leader_failure_falls_back_for_waiters():
    flight: SingleFlight[str] = SingleFlight("test")
    started, release = threading.Event(), threading.Event()
    errors = []

    def broken():
        started.set()
        release.wait(WAIT_SECONDS)
        raise RuntimeError("boom")

    def lead():
        try:
            flight.run("k", broken, timeout=WAIT_SECONDS)
        except RuntimeError as error:
            errors.append(error)

    leader = threading.Thread(target=lead)
    leader.start()
    assert started.wait(WAIT_SECONDS)
    outcome = []
    waiter = threading.Thread(target=lambda: outcome.append(flight.run("k", lambda: "own", timeout=WAIT_SECONDS)))
    waiter.start()
    assert _wait_until(lambda: flight.waiting() == 1)
    release.set()
    leader.join(WAIT_SECONDS)
    waiter.join(WAIT_SECONDS)
    assert errors and outcome == [("own", False)]
    assert flight.in_flight() == 0


def test_different_keys_do_not_wait():
    flight: SingleFlight[str] = SingleFlight("test")
    assert flight.run("a", lambda: "a", timeout=0) == ("a", False)
    assert flight.run("b", lambda: "b", timeout=0) == ("b", False)


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "MEMORY_DIR", tmp_path)
    monkeypatch.setenv("MEMORY_CROSS_RERANK", "off")
    monkeypatch.setenv("MEMORY_RECALL_TOUCH_FLUSH_MS", "600000")
    instance = server.Store()
    instance._embed_mode = "none"
    for identity, content in enumerate(("billing database is postgres", "deploys use helm"), 1):
        instance.db.execute(
            "INSERT INTO knowledge(id,content,session_id,project,type,status,created_at,last_confirmed) "
            "VALUES (?,?,'s','p','fact','active','2026-01-01','2026-01-01')", (identity, content))
    instance.db.commit()
    yield instance
    instance.db.close()


def test_concurrent_identical_searches_run_once(store, monkeypatch):
    recall = server.Recall(store)
    started, release = threading.Event(), threading.Event()
    calls = []
    original = server.Recall._search_impl

    def slow_impl(self, *args, **kwargs):
        calls.append(kwargs.get("record_usage"))
        started.set()
        release.wait(WAIT_SECONDS)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(server.Recall, "_search_impl", slow_impl)
    results = []
    first = threading.Thread(target=lambda: results.append(recall.search("billing database", project="p", limit=5)))
    first.start()
    assert started.wait(WAIT_SECONDS)
    second = threading.Thread(target=lambda: results.append(
        recall.search("  Billing   DATABASE ", project="p", limit=5)))
    second.start()
    assert _wait_until(lambda: recall._flights.waiting() == 1), "the second caller must wait for the first"
    release.set()
    first.join(WAIT_SECONDS)
    second.join(WAIT_SECONDS)
    assert len(calls) == 1
    assert len(results) == 2 and results[0]["results"] == results[1]["results"]
    assert results[0] is not results[1]
    assert recall.usage_buffer().pending() >= 1
    pending_counts = dict(recall.usage_buffer()._pending)
    assert all(count == 2 for count, _ in pending_counts.values()), "both callers' recalls are counted"
    recall.close()


def test_singleflight_can_be_switched_off(store, monkeypatch):
    monkeypatch.setenv("MEMORY_RECALL_SINGLEFLIGHT", "off")
    recall = server.Recall(store)
    calls = []
    original = server.Recall._search_impl

    def counting(self, *args, **kwargs):
        calls.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(server.Recall, "_search_impl", counting)
    recall.search("billing database", project="p", limit=5, record_usage=False)
    assert len(calls) == 1
    assert recall._flights.in_flight() == 0


def test_repeated_search_hits_the_result_cache(store, monkeypatch):
    """The lookup and the store used different keys (tier_weights was missing on put)."""
    if store.cache is None:
        pytest.skip("query cache module unavailable")
    hits = []
    original_get = store.cache.get

    def spy(key):
        value = original_get(key)
        hits.append(value is not None)
        return value

    monkeypatch.setattr(store.cache, "get", spy)
    recall = server.Recall(store)
    recall.search("deploys helm", project="p", limit=5, record_usage=False)
    recall.search("deploys helm", project="p", limit=5, record_usage=False)
    assert hits == [False, True]
    monkeypatch.setenv("MEMORY_RECALL_INTENT_BOOSTS", "decision=2")
    recall.search("deploys helm", project="p", limit=5, record_usage=False)
    assert hits[-1] is False, "a changed ranking knob is part of the cache key"


def test_result_fused_before_the_encoder_loaded_is_not_served_after(store, monkeypatch):
    if store.cache is None:
        pytest.skip("query cache module unavailable")
    from memory_core import cross_rerank

    release = threading.Event()

    class Encoder:
        def rerank(self, query, documents):
            return [float(len(doc)) for doc in documents]

    def slow_loader(_model):
        release.wait(WAIT_SECONDS)
        return Encoder()

    reranker = cross_rerank.CrossReranker("fake", multilingual=True, window=10, weight=2.0, loader=slow_loader)
    monkeypatch.setattr(cross_rerank, "shared_reranker", lambda: reranker)
    hits = []
    original_get = store.cache.get

    def spy(key):
        value = original_get(key)
        hits.append(value is not None)
        return value

    monkeypatch.setattr(store.cache, "get", spy)
    recall = server.Recall(store)
    recall.search("billing database", project="p", limit=5, record_usage=False)
    recall.search("billing database", project="p", limit=5, record_usage=False)
    assert hits == [False, True]
    release.set()
    assert reranker.wait_ready(WAIT_SECONDS)
    recall.search("billing database", project="p", limit=5, record_usage=False)
    assert hits[-1] is False, "the re-ranked result replaces the one fused before the encoder loaded"


def test_cached_results_are_private_copies(store):
    if store.cache is None:
        pytest.skip("query cache module unavailable")
    recall = server.Recall(store)
    first = recall.search("billing database", project="p", limit=5, record_usage=False)
    first["results"].clear()
    first["injected"] = True
    second = recall.search("billing database", project="p", limit=5, record_usage=False)
    assert second["results"], "a caller editing its result must not change the cached one"
    assert "injected" not in second
    third = recall.search("billing database", project="p", limit=5, record_usage=False)
    assert third == second and third is not second
