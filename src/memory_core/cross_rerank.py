"""Cross-encoder re-ranking of the fused candidate window.

Hybrid retrieval fuses lexical and vector ranks with RRF. The vector tier
embeds short conversational turns poorly ("Jon: Thanks!" sits near any
question about Jon), and RRF lets that tier pull a lexical first place down
to 27th. A cross-encoder reads query and record together and fixes the order;
its rank is fused into the RRF score rather than replacing it, so a weak
cross-encoder verdict cannot bury a strong lexical match.

A conversational turn often means something only next to its neighbours
("What was it about?" / "It was about acceptance"), so the encoder also reads
each candidate together with the turns before and after it in its session, and
both verdicts join the fused rank. Reading the turn alone keeps a precise first
place; reading it in context finds the answer that the question's words only
match in the neighbouring turn.

The model loads in a background thread. Until it is ready, `mode="auto"`
leaves the order unchanged; `mode="on"` waits for it.
"""
from __future__ import annotations

import logging
import threading
import time
import unicodedata
from collections.abc import Callable, Sequence

from memory_core.telemetry import counters, op_timer

LOGGER = logging.getLogger(__name__)
RRF_K = 60
MAX_PAIR_CHARS = 2000
LATIN_SHARE_REQUIRED = 0.6
LOAD_TIMEOUT_SECONDS = 120.0


def latin_share(text: str) -> float:
    """Share of the letters in `text` that belong to the Latin script."""
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 1.0
    latin = sum(1 for ch in letters if unicodedata.name(ch, "").startswith("LATIN"))
    return latin / len(letters)


class CrossReranker:
    """Lazily loaded fastembed cross-encoder, shared by every search of one process."""

    def __init__(self, model: str, *, multilingual: bool, window: int, weight: float, context_chars: int = 0,
                 loader: Callable[[str], object] | None = None):
        self.model = model
        self.multilingual = multilingual
        self.window = window
        self.weight = weight
        self.context_chars = context_chars
        self._loader = loader or self._load_fastembed
        self._encoder = None
        self._error: Exception | None = None
        self._ready = threading.Event()
        self._started = False
        self._lock = threading.Lock()

    @staticmethod
    def _load_fastembed(model: str):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        from memory_core.fastembed_loader import load_model

        return load_model(TextCrossEncoder, model)

    def start(self) -> None:
        """Begin loading the model in the background (idempotent)."""
        with self._lock:
            if self._started:
                return
            self._started = True
        threading.Thread(target=self._load, name="cross-rerank-load", daemon=True).start()

    def _load(self) -> None:
        started = time.perf_counter()
        try:
            self._encoder = self._loader(self.model)
            counters.bump("cross_rerank_loaded")
            LOGGER.info("Cross-encoder ready", extra={
                "model": self.model, "seconds": round(time.perf_counter() - started, 2)})
        except Exception as error:  # noqa: BLE001 — any load failure disables re-ranking, search continues
            self._error = error
            counters.bump("cross_rerank_load_failed")
            LOGGER.warning("Cross-encoder unavailable; results keep fused order", extra={
                "model": self.model, "error": str(error)})
        finally:
            self._ready.set()

    def wait_ready(self, timeout: float = LOAD_TIMEOUT_SECONDS) -> bool:
        """Block until the model loaded or failed; True when it can re-rank."""
        self.start()
        self._ready.wait(timeout)
        return self._encoder is not None

    def applies_to(self, query: str) -> bool:
        return self.multilingual or latin_share(query) >= LATIN_SHARE_REQUIRED

    def rerank(self, query: str, items: Sequence[dict], text_of: Callable[[dict], str], *, wait: bool,
               context_of: Callable[[list[dict]], list[str]] | None = None) -> list[dict]:
        """`items` in fused order -> the window re-ordered, followed by the rest unchanged.

        `context_of(window)` gives each candidate's text with its neighbouring turns; when
        present, the encoder scores both views and the weight is split between them.
        """
        if len(items) < 2 or not self.applies_to(query):
            counters.bump("cross_rerank_skipped")
            return list(items)
        if wait:
            self.wait_ready()
        else:
            self.start()
        if self._encoder is None:
            counters.bump("cross_rerank_not_ready")
            return list(items)
        window = list(items[:self.window])
        views = [[text_of(item) for item in window]]
        if context_of is not None:
            views.append(context_of(window))
        ranks = []
        with op_timer("cross_rerank_ms"):
            for texts in views:
                scores = list(self._encoder.rerank(query, [text[:MAX_PAIR_CHARS] for text in texts]))
                if len(scores) != len(window):
                    raise RuntimeError(f"cross-encoder returned {len(scores)} scores for {len(window)} candidates")
                by_score = sorted(range(len(window)), key=lambda i: (-scores[i], i))
                ranks.append({index: rank for rank, index in enumerate(by_score)})
        share = self.weight / len(ranks)
        fused = sorted(range(len(window)), key=lambda i: (
            -(1 / (RRF_K + i + 1) + sum(share / (RRF_K + rank[i] + 1) for rank in ranks)), i))
        counters.bump("cross_rerank_applied")
        return [window[i] for i in fused] + list(items[self.window:])


def session_window_texts(db, rows: Sequence[dict], *, side_chars: int) -> dict[int, str]:
    """Record id -> previous turn + record + next turn of the same session."""
    anchors = [row for row in rows if row.get("session_id") and row.get("created_at")]
    if not anchors:
        return {}
    parts, params = [], []
    for row in anchors:
        for side, comparison, direction in ((0, "<", "DESC"), (1, ">", "ASC")):
            parts.append(
                "SELECT * FROM (SELECT ? AS a, ? AS s, content FROM knowledge "
                "WHERE session_id=? AND project IS ? AND status='active' "
                f"AND (created_at, id) {comparison} (?, ?) ORDER BY created_at {direction}, id {direction} LIMIT 1)")
            params += [row["id"], side, row["session_id"], row.get("project"), row["created_at"], row["id"]]
    around: dict[tuple[int, int], str] = {}
    for offset in range(0, len(parts), 100):
        for a, s, content in db.execute(" UNION ALL ".join(parts[offset:offset + 100]),
                                        params[offset * 6:(offset + 100) * 6]).fetchall():
            around[(a, s)] = content or ""
    out = {}
    for row in anchors:
        before = around.get((row["id"], 0), "")[-side_chars:]
        after = around.get((row["id"], 1), "")[:side_chars]
        out[row["id"]] = "\n".join(t for t in (before, row.get("content", ""), after) if t)
    return out


_shared: CrossReranker | None = None
_shared_lock = threading.Lock()


def shared_reranker() -> CrossReranker | None:
    """Process-wide re-ranker built from config, or None when disabled."""
    global _shared
    from config import (
        MULTILINGUAL_CROSS_RERANK_MODELS,
        get_cross_rerank_context,
        get_cross_rerank_mode,
        get_cross_rerank_model,
        get_cross_rerank_weight,
        get_cross_rerank_window,
    )

    if get_cross_rerank_mode() == "off":
        return None
    with _shared_lock:
        model, window, weight = get_cross_rerank_model(), get_cross_rerank_window(), get_cross_rerank_weight()
        context = get_cross_rerank_context()
        if _shared is None or (_shared.model, _shared.window, _shared.weight, _shared.context_chars) != (
                model, window, weight, context):
            _shared = CrossReranker(model, multilingual=model in MULTILINGUAL_CROSS_RERANK_MODELS,
                                    window=window, weight=weight, context_chars=context)
        return _shared
