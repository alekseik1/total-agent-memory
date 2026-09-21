from __future__ import annotations

import heapq
import sqlite3
import threading
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from memory_core.telemetry import counters, op_timer
from memory_core.vector_math import COSINE_EPSILON, POPCOUNT, cosine_scores

DEFAULT_VECTOR_CACHE_BYTES = 64 * 1024 * 1024
VECTOR_BATCH_ROWS = 512
MAX_VECTOR_CACHE_ENTRIES = 128


@dataclass(frozen=True)
class VectorScope:
    dimension: int
    project: str | None = None
    spaces: tuple[str, ...] = ()
    model: str | None = None
    kind: str = "all"
    branch: str | None = None

    def sql(self) -> tuple[str, list[str | int]]:
        conditions, params = ["k.status='active'", "e.embed_dim=?"], [self.dimension]
        for column, value in (("k.project", self.project), ("e.embed_model", self.model),
                              ("k.type", self.kind if self.kind != "all" else None)):
            if value is not None:
                conditions.append(f"{column}=?")
                params.append(value)
        if self.branch:
            conditions.append("(k.branch=? OR k.branch='')")
            params.append(self.branch)
        if self.spaces:
            conditions.append("COALESCE(e.embedding_space,'text') IN (" + ",".join("?" for _ in self.spaces) + ")")
            params.extend(self.spaces)
        return " AND ".join(conditions), params


@dataclass(frozen=True)
class VectorPool:
    ids: np.ndarray
    values: np.ndarray
    norms: np.ndarray | None

    @property
    def nbytes(self) -> int:
        return self.ids.nbytes + self.values.nbytes + (0 if self.norms is None else self.norms.nbytes)


class VectorSearch:
    def __init__(self, db: sqlite3.Connection, max_cache_bytes: int = DEFAULT_VECTOR_CACHE_BYTES):
        if type(max_cache_bytes) is not int or max_cache_bytes < 0:
            raise ValueError("Vector cache budget must be a non-negative integer")
        self.db = db
        self.max_cache_bytes = max_cache_bytes
        self.cache_bytes = 0
        self.pools: OrderedDict[tuple[VectorScope, int], VectorPool] = OrderedDict()
        self.group_cache: OrderedDict[str | None, tuple[tuple[str, str, int], ...]] = OrderedDict()
        self.revision: int | None = None
        self.lock = threading.RLock()

    def clear(self) -> None:
        with self.lock:
            self.pools.clear()
            self.group_cache.clear()
            self.cache_bytes = 0
            self.revision = None

    def _revision(self) -> int:
        return self.db.execute("SELECT revision FROM vector_index_revision WHERE singleton=1").fetchone()[0]

    def _sync_revision(self) -> int:
        revision = self._revision()
        if revision != self.revision or self.db.in_transaction:
            self.clear()
            self.revision = revision
        return revision

    def groups(self, project: str | None) -> tuple[tuple[str, str, int], ...]:
        with self.lock, op_timer("vector_groups_ms"):
            revision = self._sync_revision()
            if project in self.group_cache:
                self.group_cache.move_to_end(project)
                counters.bump("vector_groups_cache_hits")
                return self.group_cache[project]
            params = (project,) if project is not None else ()
            rows = self.db.execute(
                "SELECT DISTINCT COALESCE(e.embedding_space,'text'),e.embed_model,e.embed_dim "
                "FROM embeddings e JOIN knowledge k ON k.id=e.knowledge_id WHERE k.status='active'"
                + (" AND k.project=?" if project is not None else ""), params,
            ).fetchall()
            result = tuple(tuple(row) for row in rows)
            if self.max_cache_bytes and not self.db.in_transaction and revision == self._revision():
                while len(self.group_cache) >= MAX_VECTOR_CACHE_ENTRIES:
                    self.group_cache.popitem(last=False)
                self.group_cache[project] = result
            return result

    def search(
        self, query: list[float], scope: VectorScope, *, candidates: int, limit: int,
        exact_limit: int = 0,
    ) -> list[tuple[int, float]]:
        if type(candidates) is not int or candidates < 1 or type(limit) is not int or limit < 1:
            raise ValueError("Candidate and result limits must be positive integers")
        if type(exact_limit) is not int or exact_limit < 0:
            raise ValueError("Exact scan limit must be a non-negative integer")
        q = np.asarray(query, dtype=np.float32)
        if q.ndim != 1 or q.size != scope.dimension or not q.size or not np.isfinite(q).all():
            raise ValueError("Query must be a finite vector matching its scope")
        with self.lock, op_timer("vector_lookup_ms"):
            revision = self._sync_revision()
            key = (scope, exact_limit)
            pool = self.pools.get(key)
            if pool is None:
                counters.bump("vector_pool_cache_misses")
                pool, count = self._load(scope, exact_limit)
                if pool is None:
                    return self._stream(q, scope, count, candidates, limit, exact_limit)
                if not self.db.in_transaction and revision == self._revision():
                    self._remember(key, pool)
            else:
                self.pools.move_to_end(key)
                counters.bump("vector_pool_cache_hits")
            return self._rank(q, pool, candidates, limit)

    def _remember(self, key: tuple[VectorScope, int], pool: VectorPool) -> None:
        if not self.max_cache_bytes or pool.nbytes > self.max_cache_bytes:
            return
        while self.pools and (
            self.cache_bytes + pool.nbytes > self.max_cache_bytes
            or len(self.pools) >= MAX_VECTOR_CACHE_ENTRIES
        ):
            _, removed = self.pools.popitem(last=False)
            self.cache_bytes -= removed.nbytes
            counters.bump("vector_pool_cache_evictions")
        self.pools[key] = pool
        self.cache_bytes += pool.nbytes

    def _load(self, scope: VectorScope, exact_limit: int) -> tuple[VectorPool | None, int]:
        where, params = scope.sql()
        joined = "FROM embeddings e JOIN knowledge k ON k.id=e.knowledge_id WHERE " + where
        count = self.db.execute("SELECT COUNT(*) " + joined, params).fetchone()[0]
        exact = count <= exact_limit
        width = scope.dimension * 4 + 4 if exact else (scope.dimension + 7) // 8
        if count * (width + 8) > self.max_cache_bytes:
            return None, count
        column = "float32_vector" if exact else "binary_vector"
        cursor = self.db.execute(f"SELECT e.knowledge_id,e.{column} " + joined + " ORDER BY e.knowledge_id", params)
        try:
            capacity = self.max_cache_bytes // (width + 8)
            rows = cursor.fetchmany(capacity + 1)
        finally:
            cursor.close()
        if len(rows) > capacity:
            return None, max(count, len(rows))
        ids = np.array([row[0] for row in rows], dtype=np.int64)
        dtype = np.float32 if exact else np.uint8
        elements = scope.dimension if exact else (scope.dimension + 7) // 8
        if any(len(row[1]) != elements * np.dtype(dtype).itemsize for row in rows):
            raise ValueError("Stored vector dimension differs from its metadata")
        values = np.frombuffer(b"".join(row[1] for row in rows), dtype=dtype).reshape(-1, elements)
        if exact and not np.isfinite(values).all():
            raise ValueError("Stored vectors must be finite")
        norms = np.sqrt(np.einsum("ij,ij->i", values, values)) if exact else None
        for array in (ids, values, norms):
            if array is not None:
                array.flags.writeable = False
        return VectorPool(ids, values, norms), count

    def _rank(self, query: np.ndarray, pool: VectorPool, candidates: int, limit: int) -> list[tuple[int, float]]:
        if not len(pool.ids):
            return []
        if pool.norms is not None:
            counters.bump("vector_exact_searches")
            scores = np.einsum("ij,j->i", pool.values, query) / (pool.norms * np.linalg.norm(query) + COSINE_EPSILON)
            counters.bump("vector_cosine_candidates", len(pool.ids))
            return self._top(pool.ids, scores, limit)
        counters.bump("vector_binary_searches")
        binary = np.packbits(query > 0)
        distances = POPCOUNT[np.bitwise_xor(pool.values, binary)].sum(axis=1)
        count = min(candidates, len(pool.ids))
        if count < len(pool.ids):
            boundary = np.partition(distances, count - 1)[count - 1]
            indexes = np.flatnonzero(distances < boundary)
            ties = np.flatnonzero(distances == boundary)[:count - len(indexes)]
            indexes = np.concatenate((indexes, ties))
        else:
            indexes = np.arange(len(pool.ids))
        return self._cosine_candidates(query, pool.ids[indexes].tolist(), limit)

    def _cosine_candidates(self, query: np.ndarray, identities: list[int], limit: int) -> list[tuple[int, float]]:
        scored = []
        for offset in range(0, len(identities), VECTOR_BATCH_ROWS):
            batch = identities[offset:offset + VECTOR_BATCH_ROWS]
            rows = self.db.execute(
                "SELECT knowledge_id,float32_vector FROM embeddings WHERE knowledge_id IN ("
                + ",".join("?" for _ in batch) + ")", batch,
            ).fetchall()
            scores = cosine_scores(query, [row[1] for row in rows])
            scored.extend((row[0], float(score)) for row, score in zip(rows, scores, strict=True))
        counters.bump("vector_cosine_candidates", len(scored))
        return heapq.nsmallest(limit, scored, key=lambda item: (-item[1], item[0]))

    @staticmethod
    def _top(ids: np.ndarray, scores: np.ndarray, limit: int) -> list[tuple[int, float]]:
        order = np.lexsort((ids, -scores))[:limit]
        return [(int(ids[i]), float(scores[i])) for i in order]

    def _stream(
        self, query: np.ndarray, scope: VectorScope, count: int,
        candidates: int, limit: int, exact_limit: int,
    ) -> list[tuple[int, float]]:
        counters.bump("vector_pool_streaming")
        exact = count <= exact_limit
        counters.bump("vector_exact_searches" if exact else "vector_binary_searches")
        where, params = scope.sql()
        column = "float32_vector" if exact else "binary_vector"
        cursor = self.db.execute(
            f"SELECT e.knowledge_id,e.{column} FROM embeddings e JOIN knowledge k "
            "ON k.id=e.knowledge_id WHERE " + where, params,
        )
        best: list[tuple[float, int]] = []
        keep = limit if exact else candidates
        binary = np.packbits(query > 0)
        while rows := cursor.fetchmany(VECTOR_BATCH_ROWS):
            if exact:
                scores = cosine_scores(query, [row[1] for row in rows])
                counters.bump("vector_cosine_candidates", len(rows))
            else:
                width = len(binary)
                if any(len(row[1]) != width for row in rows):
                    raise ValueError("Stored binary vector dimension differs from its metadata")
                values = np.frombuffer(b"".join(row[1] for row in rows), dtype=np.uint8).reshape(-1, width)
                scores = -POPCOUNT[np.bitwise_xor(values, binary)].sum(axis=1).astype(np.float64)
            for row, score in zip(rows, scores, strict=True):
                candidate = (float(score), -row[0])
                if len(best) < keep:
                    heapq.heappush(best, candidate)
                elif candidate > best[0]:
                    heapq.heapreplace(best, candidate)
        if exact:
            return [(-identity, score) for score, identity in sorted(best, reverse=True)]
        return self._cosine_candidates(query, [-identity for _, identity in best], limit)
