import time

from ai_layer.bounded_reranker import BoundedReranker, RerankReply
from memory_core.telemetry import counters


def worker_scores(connection, model):
    connection.send(RerankReply("ready"))
    request = connection.recv()
    connection.send(RerankReply("scores", tuple(range(len(request.contents)))))
    connection.recv()


def worker_slow(connection, model):
    connection.send(RerankReply("ready"))
    connection.recv()
    time.sleep(10)


def worker_reads_middle(connection, model):
    connection.send(RerankReply("ready"))
    request = connection.recv()
    scores = tuple(float("Morgan owns the billing database" in text) for text in request.contents)
    connection.send(RerankReply("scores", scores))
    connection.recv()


def wait_ready(reranker):
    reranker._start()
    assert reranker.connection.poll(10)
    assert reranker.connection.recv().status == "ready"
    reranker.ready = True


def test_process_reranks_without_mutating_inputs():
    reranker = BoundedReranker("fixture", worker=worker_scores, deadline_ms=1000)
    hits = [{"id": 1, "content": "one"}, {"id": 2, "content": "two"}]
    try:
        wait_ready(reranker)
        assert [hit["id"] for hit in reranker.rank("query", hits)] == [2, 1]
        assert [hit["id"] for hit in hits] == [1, 2]
    finally:
        reranker.close()


def test_deadline_kills_inference_and_preserves_originals():
    reranker = BoundedReranker("fixture", worker=worker_slow, deadline_ms=30)
    hits = [{"id": 1}, {"id": 2}]
    try:
        wait_ready(reranker)
        before = counters.get("bounded_rerank_timeouts")
        started = time.monotonic()
        assert reranker.rank("query", hits) == hits
        assert time.monotonic() - started < 1
        assert reranker.process is None
        assert reranker.failed
        assert counters.get("bounded_rerank_timeouts") == before + 1
        assert reranker.rank("query", hits) == hits
    finally:
        reranker.close()


def test_long_source_middle_fact_reaches_reranker():
    content = "Record header\n" + "Unrelated weather notes.\n" * 200
    content += "Morgan owns the billing database.\n" + "Other notes.\n" * 200
    hits = [{"id": 1, "content": "An unrelated database."}, {"id": 2, "content": content}]
    reranker = BoundedReranker("fixture", worker=worker_reads_middle, deadline_ms=1000)
    try:
        wait_ready(reranker)
        assert [hit["id"] for hit in reranker.rank("Who owns the billing database?", hits)] == [2, 1]
        assert hits[1]["content"] == content
    finally:
        reranker.close()
