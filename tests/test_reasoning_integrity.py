import json

import numpy as np
import pytest

from ai_layer import iterative_retriever as ir
from memory_core.evidence_pack import pack_evidence
from memory_core.vector_math import cosine_scores


@pytest.mark.parametrize("decision", [
    {"done": "false", "next_query": "next", "partial_answer": "x"},
    {"done": False, "next_query": None, "partial_answer": "x"},
    {"done": True, "next_query": "next", "partial_answer": "x"},
    {"done": True, "next_query": None, "partial_answer": ["x"]},
])
def test_invalid_decisions_are_not_evidence_of_success(decision):
    with pytest.raises(ValueError):
        ir._parse_planner_response(json.dumps(decision))


class Planner:
    def __init__(self, response):
        self.response = response
        self.calls = 0

    def complete(self, **kwargs):
        self.calls += 1
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


@pytest.mark.parametrize("response, reason", [
    (RuntimeError("offline"), "planner_error"),
    ("invalid JSON", "planner_error"),
    (json.dumps({"partial_answer": "", "next_query": " Q  ", "done": False}), "no_progress"),
])
def test_failure_and_repeated_query_stop_without_false_convergence(monkeypatch, response, reason):
    monkeypatch.setattr(ir, "_seed_sub_queries", lambda *a, **kw: (["q"], {}))
    client = Planner(response)
    result = ir.iterative_retrieve(
        "q", search_fn=lambda *a, **kw: [{"id": 1, "content": "Known fact"}],
        llm_client=client,
    )
    assert result.terminated_reason == reason
    assert result.iterations_used == 1
    assert result.final_evidence[0]["content"] == "Known fact"


def test_search_failure_does_not_call_planner(monkeypatch):
    monkeypatch.setattr(ir, "_seed_sub_queries", lambda *a, **kw: (["q"], {}))
    def failed_search(*args, **kwargs):
        raise RuntimeError("database unavailable")
    client = Planner("{}")
    result = ir.iterative_retrieve("q", search_fn=failed_search, llm_client=client)
    assert result.terminated_reason == "search_error"
    assert client.calls == 0


def test_early_bridge_and_late_fact_remain_visible():
    records = [{"id": i, "content": "irrelevant filler " * 20} for i in range(20)]
    records[0]["content"] += "Alice's mentor is Bea."
    records[-1]["content"] += "Bea works in Riga."
    prompt = ir._format_evidence(records)
    assert "Alice's mentor is Bea." in prompt
    assert "Bea works in Riga." in prompt
    assert len(prompt) <= 24000


def test_context_budget_preserves_short_facts_and_marks_long_excerpts():
    records = [{"id": 1, "content": "Alice knows Bea."},
               {"id": 2, "content": "START " + "x" * 9000 + " END"}]
    prompt = pack_evidence(records, max_chars=512)
    assert len(prompt) <= 512
    assert "Alice knows Bea." in prompt
    assert "START" in prompt and "END" in prompt
    assert "excerpt truncated" in prompt


def test_identity_uses_full_content():
    prefix = "x" * 300
    assert ir._hit_id({"content": prefix + "A"}) != ir._hit_id({"content": prefix + "B"})


def test_batched_cosine_matches_scalar_reference():
    rng = np.random.default_rng(42)
    vectors = rng.normal(size=(200, 384)).astype(np.float32)
    vectors[0] = 0
    query = rng.normal(size=384).astype(np.float32)
    expected = np.array([np.dot(query, row) / (
        np.linalg.norm(query) * np.linalg.norm(row) + 1e-10) for row in vectors])
    actual = cosine_scores(query, [row.tobytes() for row in vectors])
    np.testing.assert_allclose(actual, expected, atol=1e-7)
    np.testing.assert_array_equal(np.argsort(actual), np.argsort(expected))


def test_bad_vector_dimensions_rejected():
    with pytest.raises(ValueError):
        cosine_scores([1.0, 2.0], [np.ones(3, dtype=np.float32).tobytes()])


@pytest.mark.parametrize("values", [
    {"max_iters": 0}, {"max_iters": 13}, {"k_per_iter": 51}, {"k_per_iter": True},
])
def test_public_handler_rejects_invalid_limits_before_network(values):
    from v11_handlers import handle_recall_iterative
    with pytest.raises(ValueError):
        handle_recall_iterative({"query": "q", **values}, search_fn=lambda *a: [])


def test_original_question_survives_lossy_decomposition(monkeypatch):
    monkeypatch.setattr(ir, "_rewrite", lambda *a, **kw: {
        "canonical": "deployment", "decomposed": ["deployment owner"]})
    queries, metadata = ir._seed_sub_queries("Who owned deployment before May?", None)
    assert queries[0] == "Who owned deployment before May?"
    assert queries[1] == "deployment owner"
    assert metadata["used_decomposition"]
