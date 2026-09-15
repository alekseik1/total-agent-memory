from __future__ import annotations

import atexit
import logging
import math
import multiprocessing
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from multiprocessing.connection import Connection
from typing import Literal

from memory_core.evidence_excerpt import excerpt
from memory_core.retrieval import MemoryHit
from memory_core.telemetry import counters, op_timer

RERANK_EXCERPT_CHARS = 2000


@dataclass(frozen=True)
class RerankReply:
    status: Literal["ready", "scores", "error"]
    scores: tuple[float, ...] = ()
    error: str = ""


@dataclass(frozen=True)
class RerankRequest:
    query: str
    contents: tuple[str, ...]


def serve_local_model(connection: Connection, model_path: str) -> None:
    try:
        from cpu_budget import configure_torch_threads

        configure_torch_threads()
        from sentence_transformers import CrossEncoder

        model = CrossEncoder(
            model_path, device="cpu", local_files_only=True, trust_remote_code=False
        )
        connection.send(RerankReply("ready"))
        while True:
            request = connection.recv()
            if request is None:
                break
            scores = model.predict(
                [(request.query, text) for text in request.contents],
                show_progress_bar=False,
            )
            connection.send(
                RerankReply("scores", tuple(float(score) for score in scores))
            )
    except EOFError:
        logging.getLogger(__name__).debug("Reranker parent disconnected")
    except Exception as error:
        logging.getLogger(__name__).exception("Local reranker worker failed")
        try:
            connection.send(RerankReply("error", error=str(error)))
        except (BrokenPipeError, EOFError, OSError):
            logging.getLogger(__name__).debug("Reranker error could not reach parent")
    finally:
        connection.close()


class BoundedReranker:
    def __init__(
        self,
        model_path: str,
        deadline_ms: int = 250,
        candidates: int = 20,
        startup_seconds: float = 60,
        worker: Callable[[Connection, str], None] = serve_local_model,
    ):
        if (
            not model_path
            or not 1 <= deadline_ms <= 10000
            or not 2 <= candidates <= 100
        ):
            raise ValueError(
                "Reranker requires model path, deadline 1..10000ms and candidates 2..100"
            )
        if not 0 < startup_seconds <= 300:
            raise ValueError("Reranker startup must be between 0 and 300 seconds")
        self.model_path, self.deadline_ms, self.candidates = (
            model_path,
            deadline_ms,
            candidates,
        )
        self.startup_seconds, self.worker = startup_seconds, worker
        self.lock = threading.Lock()
        self.connection: Connection | None = None
        self.process: multiprocessing.Process | None = None
        self.ready = False
        self.started_at = 0.0
        self.failed = False
        self.in_flight = False
        atexit.register(self.close)

    def _start(self) -> None:
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe()
        self.process = context.Process(
            target=self.worker, args=(child, self.model_path), daemon=True
        )
        try:
            self.process.start()
        except (OSError, RuntimeError):
            parent.close()
            raise
        finally:
            child.close()
        self.connection = parent
        self.started_at = time.monotonic()

    def rank(self, query: str, hits: list[MemoryHit]) -> list[MemoryHit]:
        if len(hits) < 2 or self.failed or not self.lock.acquire(blocking=False):
            return hits
        try:
            with op_timer("bounded_rerank_ms"):
                return self._rank(query, hits)
        except (OSError, EOFError, RuntimeError, ValueError) as error:
            counters.bump("bounded_rerank_errors")
            logging.getLogger(__name__).warning(
                "Local reranking unavailable", extra={"error": str(error)}
            )
            self.failed = True
            self.close()
            return hits
        finally:
            self.lock.release()

    def _rank(self, query: str, hits: list[MemoryHit]) -> list[MemoryHit]:
        if self.process is None:
            self._start()
        if not self.ready:
            if not self.connection.poll(0):
                if time.monotonic() - self.started_at > self.startup_seconds:
                    raise RuntimeError("Local reranker startup deadline exceeded")
                counters.bump("bounded_rerank_warming")
                return hits
            reply = self.connection.recv()
            if not isinstance(reply, RerankReply) or reply.status != "ready":
                raise RuntimeError("Local reranker model could not be loaded")
            self.ready = True
        candidates = hits[: self.candidates]
        request = RerankRequest(
            query[:RERANK_EXCERPT_CHARS],
            tuple(excerpt(hit.get("content", ""), query, RERANK_EXCERPT_CHARS, len) for hit in candidates),
        )
        self.in_flight = True
        self.connection.send(request)
        if not self.connection.poll(self.deadline_ms / 1000):
            counters.bump("bounded_rerank_timeouts")
            self.failed = True
            self.close()
            return hits
        reply = self.connection.recv()
        self.in_flight = False
        if (
            not isinstance(reply, RerankReply)
            or reply.status != "scores"
            or len(reply.scores) != len(candidates)
            or not all(math.isfinite(v) for v in reply.scores)
        ):
            raise ValueError("Invalid local reranker scores")
        counters.bump("bounded_rerank_candidates", len(candidates))
        order = sorted(
            range(len(candidates)), key=lambda i: reply.scores[i], reverse=True
        )
        return [candidates[i] for i in order] + hits[len(candidates) :]

    def close(self) -> None:
        atexit.unregister(self.close)
        self.failed = True
        if self.connection and self.process and self.ready and not self.in_flight:
            try:
                self.connection.send(None)
                self.process.join(timeout=1)
            except (OSError, EOFError):
                logging.getLogger(__name__).debug(
                    "Reranker worker already disconnected"
                )
        if self.connection:
            self.connection.close()
            self.connection = None
        if self.process:
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=0.1)
                if self.process.is_alive():
                    self.process.kill()
            self.process.join(timeout=0.1)
            if not self.process.is_alive():
                self.process.close()
            self.process = None
        self.ready = False
