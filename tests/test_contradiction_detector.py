"""Tests for the v10 auto-contradiction detector."""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import contradiction_detector as cd
from base_schema import apply_full_schema


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    cd._reset_provider_cache()
    # Force enabled so 'auto' branch never depends on local Ollama.
    monkeypatch.setenv("MEMORY_CONTRADICTION_DETECT_ENABLED", "true")
    monkeypatch.delenv("MEMORY_CONTRADICTION_TYPES", raising=False)
    monkeypatch.delenv("MEMORY_CONTRADICTION_TOP_K", raising=False)
    monkeypatch.delenv("MEMORY_CONTRADICTION_MIN_COSINE", raising=False)
    monkeypatch.delenv("MEMORY_CONTRADICTION_LLM_THRESHOLD", raising=False)
    monkeypatch.delenv("MEMORY_CONTRADICTION_FLAG_THRESHOLD", raising=False)
    yield
    cd._reset_provider_cache()


@pytest.fixture
def cdb():
    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    apply_full_schema(db)
    yield db
    db.close()


def _seed_record(db, rid, content, ktype="decision", project="vito",
                 status="active"):
    db.execute(
        "INSERT INTO knowledge (id, session_id, type, content, project, status, created_at) "
        "VALUES (?, 's1', ?, ?, ?, ?, '2026-04-27T00:00:00Z')",
        (rid, ktype, content, project, status),
    )
    db.commit()


def _fake_fetcher_for(db, project, ktype):
    def fetch(ids):
        return cd.production_candidates_query(
            db, project=project, ktype=ktype, candidate_ids=ids
        )
    return fetch


# ──────────────────────────────────────────────
# Gating
# ──────────────────────────────────────────────


def test_should_run_disabled_via_env(monkeypatch):
    monkeypatch.setenv("MEMORY_CONTRADICTION_DETECT_ENABLED", "false")
    ok, reason = cd.should_run("decision")
    assert not ok and "disabled" in reason


def test_should_run_skips_unsupported_type(monkeypatch):
    monkeypatch.setenv("MEMORY_CONTRADICTION_TYPES", "decision,solution")
    ok, reason = cd.should_run("fact")
    assert not ok and "not in enabled types" in reason


def test_should_run_runs_for_enabled_type():
    ok, reason = cd.should_run("decision")
    assert ok and reason == ""


# ──────────────────────────────────────────────
# detect_contradictions — verdicts
# ──────────────────────────────────────────────


def _llm_response(contradicts: bool, confidence: float, reason: str = "x") -> str:
    return json.dumps(
        {"contradicts": contradicts, "confidence": confidence, "reason": reason}
    )


def test_detect_supersedes_high_confidence(cdb, monkeypatch):
    monkeypatch.setenv("MEMORY_CONTRADICTION_LLM_THRESHOLD", "0.8")
    _seed_record(cdb, 1, "We use Redis for session caching")
    fake_llm = lambda prompt: _llm_response(True, 0.92, "redis replaced by memcached")
    verdicts = cd.detect_contradictions(
        "We migrated from Redis to Memcached for sessions",
        ktype="decision", project="vito",
        candidate_pool=[(1, 0.85)],
        fetch_candidates=_fake_fetcher_for(cdb, "vito", "decision"),
        llm_fn=fake_llm,
    )
    assert len(verdicts) == 1
    v = verdicts[0]
    assert v.decision == "superseded"
    assert v.llm_confidence == 0.92
    assert v.candidate_id == 1


def test_detect_flags_medium_confidence(cdb, monkeypatch):
    monkeypatch.setenv("MEMORY_CONTRADICTION_LLM_THRESHOLD", "0.8")
    monkeypatch.setenv("MEMORY_CONTRADICTION_FLAG_THRESHOLD", "0.5")
    _seed_record(cdb, 1, "We use Redis for session caching")
    fake_llm = lambda prompt: _llm_response(True, 0.65, "ambiguous")
    verdicts = cd.detect_contradictions(
        "Looking into Memcached as an option for sessions",
        ktype="decision", project="vito",
        candidate_pool=[(1, 0.7)],
        fetch_candidates=_fake_fetcher_for(cdb, "vito", "decision"),
        llm_fn=fake_llm,
    )
    assert verdicts[0].decision == "flagged"


def test_detect_rejects_low_confidence(cdb):
    _seed_record(cdb, 1, "We use Redis for session caching")
    fake_llm = lambda prompt: _llm_response(True, 0.2, "weak signal")
    verdicts = cd.detect_contradictions(
        "Deployed new monitoring dashboard",
        ktype="decision", project="vito",
        candidate_pool=[(1, 0.6)],
        fetch_candidates=_fake_fetcher_for(cdb, "vito", "decision"),
        llm_fn=fake_llm,
    )
    assert verdicts[0].decision == "rejected"


def test_detect_rejects_when_llm_says_no_contradiction(cdb):
    _seed_record(cdb, 1, "Use Redis for caching")
    # contradicts=False overrides any high confidence number.
    fake_llm = lambda prompt: _llm_response(False, 0.99, "complementary, not conflicting")
    verdicts = cd.detect_contradictions(
        "Use Redis for rate limiting too",
        ktype="decision", project="vito",
        candidate_pool=[(1, 0.85)],
        fetch_candidates=_fake_fetcher_for(cdb, "vito", "decision"),
        llm_fn=fake_llm,
    )
    assert verdicts[0].decision == "rejected"


def test_detect_filters_by_min_cosine(cdb, monkeypatch):
    monkeypatch.setenv("MEMORY_CONTRADICTION_MIN_COSINE", "0.7")
    _seed_record(cdb, 1, "Use Redis for caching")
    calls = []
    fake_llm = lambda prompt: (calls.append(prompt) or _llm_response(True, 0.99))
    verdicts = cd.detect_contradictions(
        "Use Memcached", ktype="decision", project="vito",
        candidate_pool=[(1, 0.5)],  # below threshold
        fetch_candidates=_fake_fetcher_for(cdb, "vito", "decision"),
        llm_fn=fake_llm,
    )
    assert verdicts == []
    assert calls == []  # no LLM round-trip wasted


def test_detect_handles_llm_error(cdb):
    _seed_record(cdb, 1, "Use Redis for caching")
    def boom(prompt): raise RuntimeError("provider down")
    verdicts = cd.detect_contradictions(
        "Use Memcached", ktype="decision", project="vito",
        candidate_pool=[(1, 0.85)],
        fetch_candidates=_fake_fetcher_for(cdb, "vito", "decision"),
        llm_fn=boom,
    )
    assert verdicts[0].decision == "error"
    assert "provider down" in verdicts[0].reason


def test_detect_handles_unparsable_llm_response(cdb):
    _seed_record(cdb, 1, "Use Redis for caching")
    fake_llm = lambda prompt: "i am a banana"
    verdicts = cd.detect_contradictions(
        "Use Memcached", ktype="decision", project="vito",
        candidate_pool=[(1, 0.85)],
        fetch_candidates=_fake_fetcher_for(cdb, "vito", "decision"),
        llm_fn=fake_llm,
    )
    assert verdicts[0].decision == "error"
    assert "unparsable" in verdicts[0].reason


def test_detect_skips_candidates_filtered_out_by_fetcher(cdb):
    """Candidate exists but the fetcher returns it filtered out (e.g. wrong project)."""
    # Seed a record in a DIFFERENT project — fetcher won't find it.
    _seed_record(cdb, 1, "Use Redis", project="other-proj")
    fake_llm = lambda prompt: _llm_response(True, 0.99)
    verdicts = cd.detect_contradictions(
        "Use Memcached", ktype="decision", project="vito",
        candidate_pool=[(1, 0.9)],
        fetch_candidates=_fake_fetcher_for(cdb, "vito", "decision"),
        llm_fn=fake_llm,
    )
    assert verdicts == []


def test_detect_compares_each_candidate_separately(cdb):
    _seed_record(cdb, 1, "Use Redis for caching")
    _seed_record(cdb, 2, "MySQL is primary db")
    _seed_record(cdb, 3, "Postgres is read replica")
    seen = []
    def fake_llm(prompt):
        seen.append(prompt)
        # Distinguish candidates by the OLD-record marker the prompt embeds.
        if "id=1" in prompt:
            return _llm_response(True, 0.95, "redis replaced")
        return _llm_response(False, 0.0, "unrelated")
    verdicts = cd.detect_contradictions(
        "Memcached replaced Redis", ktype="decision", project="vito",
        candidate_pool=[(1, 0.9), (2, 0.7), (3, 0.65)],
        fetch_candidates=_fake_fetcher_for(cdb, "vito", "decision"),
        llm_fn=fake_llm,
    )
    assert len(verdicts) == 3
    by_id = {v.candidate_id: v.decision for v in verdicts}
    assert by_id[1] == "superseded"
    assert by_id[2] == "rejected"
    assert by_id[3] == "rejected"


# ──────────────────────────────────────────────
# Apply + log
# ──────────────────────────────────────────────


def test_apply_supersession_marks_old_record(cdb):
    _seed_record(cdb, 1, "Old fact")
    _seed_record(cdb, 2, "New fact")
    assert cd.apply_supersession(cdb, old_id=1, new_id=2) is True
    row = cdb.execute("SELECT status, superseded_by FROM knowledge WHERE id=1").fetchone()
    assert row["status"] == "superseded"
    assert row["superseded_by"] == 2


def test_apply_supersession_idempotent_when_already_superseded(cdb):
    _seed_record(cdb, 1, "Old fact", status="superseded")
    _seed_record(cdb, 2, "New fact")
    # Already non-active → returns False (no rows updated).
    assert cd.apply_supersession(cdb, old_id=1, new_id=2) is False


def test_apply_and_log_writes_audit_rows(cdb):
    _seed_record(cdb, 1, "Old")
    _seed_record(cdb, 2, "Newer")
    verdicts = [
        cd.ContradictionVerdict(
            candidate_id=1, cosine=0.9, llm_confidence=0.95,
            decision="superseded", reason="conflict",
            provider="fake", model="m1", latency_ms=42,
        ),
        cd.ContradictionVerdict(
            candidate_id=2, cosine=0.7, llm_confidence=0.6,
            decision="flagged", reason="ambiguous", provider="fake",
            model="m1", latency_ms=51,
        ),
    ]
    counts = cd.apply_and_log(cdb, verdicts, new_id=999, provider="fake", model="m1")
    assert counts == {"superseded": 1, "flagged": 1, "rejected": 0, "error": 0, "skip": 0}

    rows = cdb.execute(
        "SELECT decision, candidate_knowledge_id, llm_confidence FROM contradiction_log ORDER BY id"
    ).fetchall()
    assert len(rows) == 2
    assert rows[0]["decision"] == "superseded"
    assert rows[0]["candidate_knowledge_id"] == 1
    assert rows[1]["decision"] == "flagged"


def test_apply_and_log_downgrades_when_target_already_superseded(cdb):
    """If another writer already superseded the candidate between detect+apply,
    we must NOT double-write — re-classify as 'skip' so the audit log is
    truthful."""
    _seed_record(cdb, 1, "Old", status="superseded")  # already non-active
    verdicts = [
        cd.ContradictionVerdict(
            candidate_id=1, cosine=0.9, llm_confidence=0.95,
            decision="superseded", reason="conflict",
        ),
    ]
    counts = cd.apply_and_log(cdb, verdicts, new_id=2)
    assert counts["superseded"] == 0
    assert counts["skip"] == 1
    row = cdb.execute("SELECT decision FROM contradiction_log WHERE candidate_knowledge_id=1").fetchone()
    assert row["decision"] == "skip"
