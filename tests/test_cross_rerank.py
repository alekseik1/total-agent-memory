"""Cross-encoder re-ranking of the fused window (memory_core/cross_rerank.py)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import server
from memory_core import cross_rerank
from memory_core.cross_rerank import CrossReranker, latin_share
from memory_core.telemetry import counters


class KeywordEncoder:
    """Scores a record by how many of the query's words it contains."""

    def __init__(self):
        self.calls = 0

    def rerank(self, query, documents):
        self.calls += 1
        words = {w.strip("?.,").lower() for w in query.split()}
        return [float(sum(1 for w in doc.lower().split() if w.strip("?.,\"") in words)) for doc in documents]


def ready(encoder=None, **kwargs) -> CrossReranker:
    reranker = CrossReranker("fake", multilingual=kwargs.pop("multilingual", False),
                             window=kwargs.pop("window", 50), weight=kwargs.pop("weight", 2.0),
                             context_chars=kwargs.pop("context_chars", 0),
                             loader=lambda _model: encoder or KeywordEncoder())
    assert reranker.wait_ready(5)
    return reranker


def items(*texts):
    return [{"r": {"id": n, "content": text}} for n, text in enumerate(texts)]


def order(result):
    return [item["r"]["id"] for item in result]


def test_latin_share_counts_letters_only():
    assert latin_share("What book is Jon reading?") == 1.0
    assert latin_share("Какую книгу читает Джон?") == 0.0
    assert latin_share("2023-05-07 !!") == 1.0


def test_encoder_winner_moves_up_but_fused_rank_still_counts():
    """Weight 2: the encoder's clear winner climbs from 4th to 2nd, not straight to 1st."""
    candidates = items("Jon: Thanks!", "Jon: Cool", "Jon: nice", "Jon: I'm currently reading The Lean Startup book")
    result = ready().rerank("What book is Jon currently reading?", candidates, lambda i: i["r"]["content"], wait=False)
    assert order(result).index(3) == 1


def test_items_beyond_the_window_keep_their_order():
    candidates = items("a", "b", "reading book", "tail one", "tail two")
    result = ready(window=3).rerank("reading book", candidates, lambda i: i["r"]["content"], wait=False)
    assert order(result)[3:] == [3, 4]
    assert sorted(order(result)[:3]) == [0, 1, 2]


def test_english_model_skips_non_latin_queries():
    encoder = KeywordEncoder()
    candidates = items("первый", "второй")
    result = ready(encoder).rerank("какая книга", candidates, lambda i: i["r"]["content"], wait=False)
    assert order(result) == [0, 1]
    assert encoder.calls == 0


def test_multilingual_model_reranks_non_latin_queries():
    encoder = KeywordEncoder()
    ready(encoder, multilingual=True).rerank("какая книга", items("x", "книга"), lambda i: i["r"]["content"], wait=False)
    assert encoder.calls == 1


def test_auto_mode_keeps_order_until_the_model_is_loaded():
    gate = __import__("threading").Event()

    def slow_loader(_model):
        gate.wait(5)
        return KeywordEncoder()

    reranker = CrossReranker("fake", multilingual=False, window=50, weight=2.0, loader=slow_loader)
    before = counters.get("cross_rerank_not_ready")
    result = reranker.rerank("reading book", items("a", "reading book"), lambda i: i["r"]["content"], wait=False)
    assert order(result) == [0, 1]
    assert counters.get("cross_rerank_not_ready") == before + 1
    gate.set()
    assert reranker.wait_ready(5)


def test_load_failure_keeps_fused_order():
    def broken(_model):
        raise OSError("model download failed")

    reranker = CrossReranker("fake", multilingual=False, window=50, weight=2.0, loader=broken)
    before = counters.get("cross_rerank_load_failed")
    assert reranker.wait_ready(5) is False
    result = reranker.rerank("reading book", items("a", "reading book"), lambda i: i["r"]["content"], wait=True)
    assert order(result) == [0, 1]
    assert counters.get("cross_rerank_load_failed") == before + 1


def test_score_count_mismatch_is_an_error():
    class Short:
        def rerank(self, query, documents):
            return [1.0]

    with pytest.raises(RuntimeError):
        ready(Short()).rerank("q", items("a", "b"), lambda i: i["r"]["content"], wait=False)


def test_context_view_lifts_a_turn_whose_neighbour_matches_the_question():
    """"Self-acceptance and trans stories" shares no word with the question; the turn before it does."""
    candidates = items("Caroline: I love painting", "Melanie: nice weather", "Caroline: self-acceptance and trans stories",
                       "Jon: ok")
    around = {2: "Melanie: What was the poetry reading about?\nCaroline: self-acceptance and trans stories"}
    encoder = KeywordEncoder()
    plain = ready(encoder).rerank("What was the poetry reading about?", candidates,
                                  lambda i: i["r"]["content"], wait=False)
    with_context = ready(encoder).rerank(
        "What was the poetry reading about?", candidates, lambda i: i["r"]["content"], wait=False,
        context_of=lambda window: [around.get(i["r"]["id"], i["r"]["content"]) for i in window])
    assert order(with_context).index(2) < order(plain).index(2)
    assert encoder.calls == 3


def test_session_window_texts_stays_inside_the_session(store):
    store.session_start("s2", project="books")
    store.save_knowledge(sid="s2", content="other session turn", ktype="fact", project="books", skip_dedup=True)
    rows = [dict(r) for r in store.db.execute(
        "SELECT * FROM knowledge WHERE project='books' AND session_id='s1' ORDER BY created_at, id").fetchall()]
    texts = cross_rerank.session_window_texts(store.db, [rows[0], rows[5], rows[-1]], side_chars=12)
    assert texts[rows[5]["id"]] == "\n".join((rows[4]["content"][-12:], rows[5]["content"], rows[6]["content"][:12]))
    assert texts[rows[0]["id"]] == "\n".join((rows[0]["content"], rows[1]["content"][:12]))
    assert "other session" not in texts[rows[-1]["id"]]
    assert cross_rerank.session_window_texts(store.db, [], side_chars=12) == {}


def test_shared_reranker_follows_config(monkeypatch):
    monkeypatch.setattr(cross_rerank, "_shared", None)
    monkeypatch.setenv("MEMORY_CROSS_RERANK", "off")
    assert cross_rerank.shared_reranker() is None
    monkeypatch.setenv("MEMORY_CROSS_RERANK", "auto")
    monkeypatch.setenv("MEMORY_CROSS_RERANK_WINDOW", "30")
    first = cross_rerank.shared_reranker()
    assert first.window == 30
    monkeypatch.setenv("MEMORY_CROSS_RERANK_WINDOW", "40")
    assert cross_rerank.shared_reranker().window == 40
    assert cross_rerank.shared_reranker().context_chars == 400
    monkeypatch.setenv("MEMORY_CROSS_RERANK_CONTEXT", "0")
    assert cross_rerank.shared_reranker().context_chars == 0
    monkeypatch.setenv("MEMORY_CROSS_RERANK_CONTEXT", "-1")
    with pytest.raises(ValueError):
        cross_rerank.shared_reranker()
    monkeypatch.delenv("MEMORY_CROSS_RERANK_CONTEXT")
    monkeypatch.setenv("MEMORY_CROSS_RERANK", "sometimes")
    with pytest.raises(ValueError):
        cross_rerank.shared_reranker()


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "MEMORY_DIR", tmp_path)
    s = server.Store()
    s.session_start("s1", project="books")
    for n in range(30):
        s.save_knowledge(sid="s1", content=f"Jon: small talk number {n}, thanks Gina", ktype="fact",
                         project="books", skip_dedup=True)
    s.save_knowledge(sid="s1", content="Jon: I'm currently reading The Lean Startup, a book about building a business",
                     ktype="fact", project="books")
    try:
        yield s
    finally:
        s.db.close()


def test_search_widens_the_pool_and_returns_the_limit(store, monkeypatch):
    """Thirty short turns crowd the answer out of the top ten; the re-ranker brings it back."""

    def top_ten(reranker):
        monkeypatch.setattr(cross_rerank, "shared_reranker", lambda: reranker)
        result = server.Recall(store).search(query="What book is Jon currently reading?", project="books",
                                             limit=10, detail="full", record_usage=False)
        return [hit["content"] for group in result["results"].values() for hit in group]

    assert not any("Lean Startup" in text for text in top_ten(None))
    encoder = KeywordEncoder()
    reranked = top_ten(ready(encoder, window=40))
    assert len(reranked) == 10
    assert any("Lean Startup" in text for text in reranked)
    assert encoder.calls == 1
    encoder = KeywordEncoder()
    assert any("Lean Startup" in text for text in top_ten(ready(encoder, window=40, context_chars=200)))
    assert encoder.calls == 2
