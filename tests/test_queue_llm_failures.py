"""Regression tests: LLM failures are retryable, never stored as successful empty results; stuck rows are reclaimed."""

from __future__ import annotations

import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

ROOT = Path(__file__).parent.parent
LONG_TEXT = "Long document about Go microservices and PostgreSQL tuning. " * 5
STALE_AGE = timedelta(hours=2)


@pytest.fixture
def qdb():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript((ROOT / "migrations" / "001_v5_schema.sql").read_text(encoding="utf-8"))
    conn.executescript(
        "CREATE TABLE IF NOT EXISTS knowledge (id INTEGER PRIMARY KEY AUTOINCREMENT, content TEXT, "
        "status TEXT DEFAULT 'active', created_at TEXT);"
    )
    conn.executescript((ROOT / "migrations" / "003_triple_extraction_queue.sql").read_text(encoding="utf-8"))
    conn.executescript((ROOT / "migrations" / "004_deep_enrichment.sql").read_text(encoding="utf-8"))
    yield conn
    conn.close()


def _kid(db) -> int:
    return db.execute("INSERT INTO knowledge (content, created_at) VALUES (?, 'now')", (LONG_TEXT,)).lastrowid


def _stamp(delta: timedelta) -> str:
    return (datetime.now(timezone.utc) - delta).strftime("%Y-%m-%dT%H:%M:%SZ")


def test_deep_enrich_flags_llm_failure_as_degraded(monkeypatch):
    import deep_enricher

    def boom(*args, **kwargs):
        raise ConnectionRefusedError("ollama down")

    monkeypatch.setattr(deep_enricher, "_llm_complete", boom)
    monkeypatch.setattr("config.has_llm", lambda *a, **k: True)
    out = deep_enricher.deep_enrich(LONG_TEXT)
    assert out["degraded"] is True
    assert out["entities"] == [] and out["intent"] == "unknown"


def test_deep_enrich_flags_missing_llm_as_degraded(monkeypatch):
    import deep_enricher

    monkeypatch.setattr("config.has_llm", lambda *a, **k: False)
    assert deep_enricher.deep_enrich(LONG_TEXT)["degraded"] is True


def test_deep_enrich_healthy_run_is_not_degraded(monkeypatch):
    import deep_enricher

    monkeypatch.setattr("config.has_llm", lambda *a, **k: True)
    monkeypatch.setattr(deep_enricher, "_llm_complete", lambda prompt, **kw: '{"entities": [], "intent": "fact"}')
    assert "degraded" not in deep_enricher.deep_enrich(LONG_TEXT)


def test_degraded_enrichment_is_deferred_not_stored_as_done(qdb):
    from deep_enrichment_queue import DeepEnrichmentQueue

    q = DeepEnrichmentQueue(qdb)
    kid = _kid(qdb)
    q.enqueue(kid)
    degraded = {"entities": [], "intent": "unknown", "topics": [], "degraded": True, "degraded_reason": "down"}
    stats = q.process_pending(lambda content, meta: degraded)
    assert stats["deferred"] == 1 and stats["processed"] == 0
    assert qdb.execute("SELECT COUNT(*) FROM knowledge_enrichment").fetchone()[0] == 0
    row = qdb.execute("SELECT status, attempts, next_attempt_at FROM deep_enrichment_queue").fetchone()
    assert row["status"] == "pending" and row["attempts"] == 0 and row["next_attempt_at"]
    # Backoff keeps it out of the very next drain ...
    assert q.process_pending(lambda c, m: degraded)["deferred"] == 0
    # ... and once due, a healthy LLM completes it.
    qdb.execute("UPDATE deep_enrichment_queue SET next_attempt_at=NULL")
    healthy = {"entities": [{"name": "Go", "type": "technology"}], "intent": "fact", "topics": ["go"]}
    assert q.process_pending(lambda c, m: healthy)["processed"] == 1
    assert qdb.execute("SELECT intent FROM knowledge_enrichment").fetchone()["intent"] == "fact"


def test_degraded_drain_stops_at_first_item(qdb):
    from deep_enrichment_queue import DeepEnrichmentQueue

    q = DeepEnrichmentQueue(qdb)
    for _ in range(3):
        q.enqueue(_kid(qdb))
    calls = []

    def enricher(content, meta):
        calls.append(1)
        return {"degraded": True}

    q.process_pending(enricher)
    assert len(calls) == 1
    assert qdb.execute("SELECT COUNT(*) FROM deep_enrichment_queue WHERE status='pending'").fetchone()[0] == 3


def test_deep_queue_reclaims_stuck_processing_rows(qdb):
    from deep_enrichment_queue import DeepEnrichmentQueue

    q = DeepEnrichmentQueue(qdb)
    kid = _kid(qdb)
    q.enqueue(kid)
    assert q.claim_next() is not None  # drain killed here: row stays 'processing'
    qdb.execute("UPDATE deep_enrichment_queue SET claimed_at=?", (_stamp(STALE_AGE),))
    item = q.claim_next()
    assert item is not None and item["knowledge_id"] == kid


def test_deep_queue_stuck_row_past_max_attempts_fails(qdb):
    from deep_enrichment_queue import DeepEnrichmentQueue

    q = DeepEnrichmentQueue(qdb, max_attempts=1)
    q.enqueue(_kid(qdb))
    q.claim_next()
    qdb.execute("UPDATE deep_enrichment_queue SET claimed_at=?", (_stamp(STALE_AGE),))
    assert q.claim_next() is None
    assert qdb.execute("SELECT status FROM deep_enrichment_queue").fetchone()["status"] == "failed"


def test_triple_queue_reclaim_resets_instead_of_deleting(qdb):
    from triple_extraction_queue import TripleExtractionQueue

    q = TripleExtractionQueue(qdb)
    kid = _kid(qdb)
    q.enqueue(kid)
    q.claim_next()
    qdb.execute("UPDATE triple_extraction_queue SET claimed_at=?", (_stamp(STALE_AGE),))
    item = q.claim_next()
    assert item is not None and item["knowledge_id"] == kid
    assert item["attempts"] == 1


def test_triple_queue_failure_is_backed_off(qdb):
    from triple_extraction_queue import TripleExtractionQueue

    q = TripleExtractionQueue(qdb)
    q.enqueue(_kid(qdb))

    def extractor(kid, content):
        raise RuntimeError("llm down")

    assert q.process_pending(extractor)["failed"] == 1
    assert q.process_pending(extractor)["failed"] == 0  # parked behind next_attempt_at


def test_quality_score_retryable_flags():
    from quality_gate import QualityScore

    base = dict(total=None, specificity=None, actionability=None, verifiability=None, reason="r", threshold=0.5)
    assert QualityScore(decision="error", **base).retryable is True
    assert QualityScore(decision="skip", llm_down=True, **base).retryable is True
    assert QualityScore(decision="skip", **base).retryable is False
    assert QualityScore(decision="pass", **base).retryable is False


def test_quality_gate_provider_unavailable_is_llm_down(monkeypatch):
    import quality_gate

    class Down:
        name = "down"

        def available(self):
            return False

    monkeypatch.setattr(quality_gate, "_get_provider", lambda: Down())
    monkeypatch.setattr(quality_gate, "should_score", lambda content, ktype: (True, ""))
    assert quality_gate.score_quality(LONG_TEXT, ktype="fact", project="p").retryable is True


def test_contradiction_llm_exception_yields_retryable_verdict():
    import contradiction_detector as cd

    def llm(prompt):
        raise ConnectionRefusedError("down")

    verdicts = cd.detect_contradictions(
        "limit is 500 rpm", ktype="fact", project="p",
        candidate_pool=[(7, 0.99)],
        fetch_candidates=lambda ids: [{"id": 7, "type": "fact", "content": "limit is 100 rpm"}],
        llm_fn=llm,
    )
    assert [v.retryable for v in verdicts] == [True]


def test_ingest_item_failure_is_retried_with_backoff(qdb, monkeypatch):
    from ingestion.gateway import MAX_INGEST_ATTEMPTS, IngestGateway

    gateway = IngestGateway(qdb)
    calls = {"n": 0}

    def explode(text, deep=False):
        calls["n"] += 1
        raise RuntimeError("extractor down")

    monkeypatch.setattr(gateway.extractor, "extract_and_link", explode)
    ingest_id = gateway.ingest("some substantial text to ingest", source="test")
    row = qdb.execute("SELECT status, attempts, next_attempt_at FROM ingest_queue WHERE id=?", (ingest_id,)).fetchone()
    assert row["status"] == "pending" and row["attempts"] == 1 and row["next_attempt_at"]
    assert gateway.process_pending() == 0 and calls["n"] == 1  # still parked
    for _ in range(MAX_INGEST_ATTEMPTS):
        qdb.execute("UPDATE ingest_queue SET next_attempt_at=NULL WHERE status='pending'")
        gateway.process_pending()
    assert qdb.execute("SELECT status FROM ingest_queue WHERE id=?", (ingest_id,)).fetchone()["status"] == "error"
    assert calls["n"] == MAX_INGEST_ATTEMPTS
