from __future__ import annotations

from collections.abc import Sequence

import numpy as np

POPCOUNT = np.array([value.bit_count() for value in range(256)], dtype=np.uint8)
POPCOUNT.flags.writeable = False
COSINE_EPSILON = 1e-10
EXACT_VECTOR_SCAN_LIMIT = 32768
BINARY_CANDIDATE_MULTIPLIER = 10
MIN_BINARY_CANDIDATES = 50
MAX_BINARY_CANDIDATES = 1000


def cosine_scores(query: Sequence[float], blobs: Sequence[bytes]) -> np.ndarray:
    q = np.asarray(query, dtype=np.float32)
    if q.ndim != 1 or not q.size or not np.isfinite(q).all():
        raise ValueError("query must be a finite non-empty vector")
    if not blobs:
        return np.empty(0, dtype=np.float32)
    if any(len(blob) != q.size * q.itemsize for blob in blobs):
        raise ValueError("candidate embedding dimension differs from query")
    matrix = np.frombuffer(b"".join(blobs), dtype=np.float32).reshape(-1, q.size)
    if not np.isfinite(matrix).all():
        raise ValueError("candidate embeddings must be finite")
    dots = np.einsum("ij,j->i", matrix, q)
    norms = np.sqrt(np.einsum("ij,ij->i", matrix, matrix))
    return dots / (np.linalg.norm(q) * norms + COSINE_EPSILON)
