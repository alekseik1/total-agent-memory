"""Semantic fact merger — consolidate related-but-distinct facts via LLM.

Complements `reflection.digest.merge_duplicates` (which handles near-duplicates
at Jaccard >=0.85). This module finds clusters of related records — cosine
similarity in the high-similarity band defined by DEFAULT_MIN_SIMILARITY and
DEFAULT_MAX_SIMILARITY below - and asks an LLM to synthesize them into a
single consolidated fact. Validator guards against LLM information loss.

Only records whose type is in MERGEABLE_TYPES are candidates. The merged
record the INSERT produces is always written as type='fact', so merging a
`solution`/`decision`/`lesson` row would silently relabel it. Those types are
episodic work-log entries (what was done, when, in what order) rather than
timeless claims: two similar entries are usually two distinct events - a
sequence of steps, two separate edits to the same thing, or a decision that
was reversed and re-reversed - and merging them would destroy the temporal
order that contradiction_detector and the temporal KG rely on. `fact` and
`convention` records carry no such sequence, so they are the only safe
candidates.

Example:
    "User uses Go for backend" + "User builds APIs in Go"
    → "User's primary backend language is Go (used for APIs)."

Source rows are archived (status='archived', superseded_by=<merged_id>) and
the merge event recorded in `knowledge_merges` for audit/rollback.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from typing import Any, Callable, Sequence

import numpy as np

from memory_core.timestamps import utc_now

try:
    from validator import ContentValidator
except ImportError:  # when imported as package
    from .validator import ContentValidator  # type: ignore[no-redef]

LOG = lambda msg: sys.stderr.write(f"[fact-merger] {msg}\n")


def _now() -> str:
    return utc_now()


SimilarityFn = Callable[[int, int], float]
LLMMergeFn = Callable[[list[str]], str]
MergedHookFn = Callable[[int, str], None]
VectorsFn = Callable[[list[int]], dict[int, Sequence[float]]]

# Origin marker for synthesized records: no real session produced them. The
# matching `sessions` row is seeded once in src/sql/base_schema.sql - which
# session a synthesized record belongs to is composition, not something this
# class (contract: "db: SQLite connection") should decide per merge.
MERGE_SESSION_ID = "fact-merge"

# Only these types are timeless claims that can legitimately be restated as
# one synthesized sentence. `solution`/`decision`/`lesson` rows are episodic
# work-log entries - see the module docstring for why merging those is unsafe.
MERGEABLE_TYPES = ("fact", "convention")

# Defaults for find_clusters' merge band. Referenced (not restated) by the
# `knowledge_merges` audit rationale in merge_cluster, so the two can't drift.
DEFAULT_MIN_SIMILARITY = 0.85
DEFAULT_MAX_SIMILARITY = 0.95


class FactMerger:
    """Find and merge semantically related facts via LLM consolidation."""

    def __init__(
        self,
        db: sqlite3.Connection,
        similarity_fn: SimilarityFn,
        llm_merge_fn: LLMMergeFn | None = None,
        on_merged: MergedHookFn | None = None,
        vectors_fn: VectorsFn | None = None,
    ) -> None:
        """
        Args:
            db: SQLite connection (row_factory = Row).
            similarity_fn: (id_a, id_b) -> cosine similarity in [0, 1].
                Typically wraps server._binary_search / float32 cosine. Used
                only when `vectors_fn` is None.
            vectors_fn: (ids) -> {id: vector}. When given, clustering compares
                everything in one vectorized pass instead of O(n²) calls back
                into `similarity_fn`.
            llm_merge_fn: (list[content]) -> merged_content. If None, no merges
                happen (useful for tests that only want clustering).
            on_merged: (merged_id, merged_content) -> None, called after the
                merge commits. A raw INSERT skips everything `memory_save` does
                downstream, so this is where the merged record gets embedded and
                queued for representations. Failures inside it must not undo a
                committed merge, so it is called outside the write path.
        """
        self.db = db
        self.similarity = similarity_fn
        self.llm_merge = llm_merge_fn or (lambda _c: "")
        self.on_merged = on_merged
        self.vectors_fn = vectors_fn
        self.validator = ContentValidator()

    # ──────────────────────────────────────────────
    # Cluster discovery
    # ──────────────────────────────────────────────

    def find_clusters(
        self,
        project: str | None = None,
        min_similarity: float = DEFAULT_MIN_SIMILARITY,
        max_similarity: float = DEFAULT_MAX_SIMILARITY,
        max_cluster_size: int = 5,
    ) -> list[list[int]]:
        """Find clusters of related (but not duplicate) knowledge records.

        Simple agglomerative: for each candidate pair with similarity in
        [min, max], union-find into a cluster. Caps each cluster at
        `max_cluster_size`.
        """
        rows = self._candidate_rows(project)
        ids = [r["id"] for r in rows]

        # Union-Find for grouping
        parent: dict[int, int] = {i: i for i in ids}

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        pairs = self._banded_pairs(ids, min_similarity, max_similarity)
        for a, b in pairs:
            union(a, b)

        # Collect clusters
        groups: dict[int, list[int]] = {}
        for i in ids:
            root = find(i)
            groups.setdefault(root, []).append(i)

        clusters = [sorted(g) for g in groups.values() if len(g) > 1]

        # Respect max_cluster_size by splitting oversized groups
        capped: list[list[int]] = []
        for cl in clusters:
            if len(cl) <= max_cluster_size:
                capped.append(cl)
            else:
                for start in range(0, len(cl), max_cluster_size):
                    chunk = cl[start : start + max_cluster_size]
                    if len(chunk) > 1:
                        capped.append(chunk)

        return capped

    def _banded_pairs(
        self, ids: list[int], min_similarity: float, max_similarity: float
    ) -> list[tuple[int, int]]:
        """Return every id pair whose similarity falls inside the merge band.

        Uses one vectorized cosine pass per embedding dimension when a
        `vectors_fn` is available, falling back to pairwise `similarity_fn`
        calls otherwise. The fallback is O(n²) Python-level calls: at 1.6k
        candidate records that is 1.3M calls and ~37s, and it grows quadratically.

        Vectors are grouped by dimension because cosine between different-width
        vectors is undefined. The pairwise path scored those pairs 0.0, silently
        making records embedded by a different model unmergeable rather than
        reporting them.
        """
        if self.vectors_fn is None:
            pairs: list[tuple[int, int]] = []
            for i, a in enumerate(ids):
                for b in ids[i + 1 :]:
                    try:
                        sim = float(self.similarity(a, b))
                    except Exception as e:  # noqa: BLE001
                        LOG(f"similarity({a},{b}) failed: {e}")
                        continue
                    if min_similarity <= sim <= max_similarity:
                        pairs.append((a, b))
            return pairs

        vectors = self.vectors_fn(ids)
        by_dim: dict[int, list[int]] = {}
        for kid, vec in vectors.items():
            # `if vec:` raises on a numpy array with >1 element ("truth value
            # of an array is ambiguous") - a plain length check works for
            # both a list and an ndarray.
            if len(vec):
                by_dim.setdefault(len(vec), []).append(kid)

        missing = len(ids) - sum(len(g) for g in by_dim.values())
        if missing:
            LOG(f"{missing}/{len(ids)} candidates have no embedding - not compared")
        if len(by_dim) > 1:
            LOG(f"embedding dims present: {sorted(by_dim)} - compared within each")

        pairs = []
        for dim, group in sorted(by_dim.items()):
            if len(group) < 2:
                continue
            pairs.extend(
                self._banded_pairs_in_group(group, vectors, min_similarity, max_similarity)
            )
        return pairs

    def _banded_pairs_in_group(
        self,
        group: list[int],
        vectors: dict[int, list[float]],
        min_similarity: float,
        max_similarity: float,
        chunk_size: int = 512,
    ) -> list[tuple[int, int]]:
        """Cosine-similarity band search within one embedding dimension.

        Chunks the matmul `chunk_size` rows at a time so peak memory is
        O(n * chunk_size) instead of O(n^2). A single `matrix @ matrix.T` plus
        `np.triu_indices(n)` holds two n^2/2 int64 index arrays and a fancy-index
        copy of the full similarity matrix - ~800MB at 10k candidates and ~3GB
        at 20k, a MemoryError risk in a background daemon. Chunking keeps every
        intermediate array bounded by `chunk_size * n` regardless of `n`.
        """
        matrix = np.array([vectors[kid] for kid in group], dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        # A zero vector has no direction: leave it unnormalized so every
        # similarity against it is 0 rather than NaN.
        norms[norms == 0] = 1.0
        matrix /= norms

        n = len(group)
        pairs: list[tuple[int, int]] = []
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            block = matrix[start:end] @ matrix.T  # (chunk, n)
            i_idx = np.arange(start, end)[:, None]
            j_idx = np.arange(n)[None, :]
            banded = (j_idx > i_idx) & (block >= min_similarity) & (block <= max_similarity)
            local_rows, cols = np.nonzero(banded)
            for local_i, j in zip(local_rows, cols):
                pairs.append((group[start + int(local_i)], group[int(j)]))
        return pairs

    def _candidate_rows(self, project: str | None) -> list[sqlite3.Row]:
        type_placeholders = ",".join("?" * len(MERGEABLE_TYPES))
        if project:
            return self.db.execute(
                "SELECT id, content FROM knowledge "
                "WHERE status='active' AND project=? AND superseded_by IS NULL "
                f"AND type IN ({type_placeholders}) ORDER BY id",
                (project, *MERGEABLE_TYPES),
            ).fetchall()
        return self.db.execute(
            "SELECT id, content FROM knowledge "
            "WHERE status='active' AND superseded_by IS NULL "
            f"AND type IN ({type_placeholders}) ORDER BY id",
            MERGEABLE_TYPES,
        ).fetchall()

    # ──────────────────────────────────────────────
    # Merge a single cluster
    # ──────────────────────────────────────────────

    def merge_cluster(self, ids: list[int]) -> dict[str, Any]:
        """Synthesize a consolidated knowledge record from a cluster.

        Returns {"merged_id": int|None, "reason": str}. If the LLM output
        fails validation against the concatenated source content (loses URLs,
        paths, inline code), abort: sources stay active, no merged record.
        """
        if len(ids) < 2:
            return {"merged_id": None, "reason": "cluster too small"}

        rows = self._fetch_rows(ids)
        if len(rows) < 2:
            return {"merged_id": None, "reason": "sources not found"}

        contents = [r["content"] for r in rows]

        try:
            merged_text = self.llm_merge(contents)
        except Exception as e:  # noqa: BLE001
            LOG(f"llm_merge failed: {e}")
            return {"merged_id": None, "reason": f"llm error: {e}"}

        if not merged_text or not merged_text.strip():
            return {"merged_id": None, "reason": "llm returned empty"}

        # Validate: merged must preserve critical elements from combined source
        combined = "\n\n".join(contents)
        v = self.validator.validate(combined, merged_text, strict_paths=True)
        if not v.ok:
            LOG(f"validator rejected merge: {v.errors}")
            return {
                "merged_id": None,
                "reason": f"validator rejected: {'; '.join(v.errors[:3])}",
            }

        # Insert merged record. session_id is NOT NULL with no default, so the
        # merge has to name itself as the origin - there is no user session
        # behind a record the reflection agent synthesized.
        first = rows[0]
        merged_id = self.db.execute(
            """INSERT INTO knowledge
                 (session_id, content, project, type, tags, status, source,
                  confidence, created_at, updated_at)
               VALUES (?, ?, ?, 'fact', ?, 'active', 'merged', ?, ?, ?)""",
            (
                MERGE_SESSION_ID,
                merged_text.strip(),
                first["project"] if "project" in first.keys() else "general",
                json.dumps(["merged", "consolidated"]),
                max((r["confidence"] for r in rows if r["confidence"] is not None), default=1.0),
                _now(),
                _now(),
            ),
        ).lastrowid

        # Archive sources
        for r in rows:
            self.db.execute(
                "UPDATE knowledge SET status='archived', superseded_by=?, updated_at=? "
                "WHERE id=?",
                (merged_id, _now(), r["id"]),
            )

        # Audit trail
        self.db.execute(
            "INSERT INTO knowledge_merges (merged_knowledge_id, source_ids, rationale, created_at) "
            "VALUES (?, ?, ?, ?)",
            (
                merged_id,
                json.dumps([r["id"] for r in rows]),
                f"semantic fact merge (cosine {DEFAULT_MIN_SIMILARITY}-{DEFAULT_MAX_SIMILARITY})",
                _now(),
            ),
        )
        self.db.commit()

        LOG(f"merged cluster {ids} -> knowledge_id={merged_id}")

        if self.on_merged is not None:
            try:
                self.on_merged(merged_id, merged_text.strip())
            except Exception as e:  # noqa: BLE001
                LOG(f"on_merged hook failed for {merged_id}: {e}")

        return {"merged_id": merged_id, "reason": "ok"}

    def _fetch_rows(self, ids: list[int]) -> list[sqlite3.Row]:
        placeholders = ",".join("?" * len(ids))
        return self.db.execute(
            f"SELECT id, content, project, confidence FROM knowledge "
            f"WHERE id IN ({placeholders}) AND status='active'",
            ids,
        ).fetchall()

    # ──────────────────────────────────────────────
    # Run (drive the full loop)
    # ──────────────────────────────────────────────

    def run(self, project: str | None = None) -> dict[str, int]:
        stats = {"clusters_found": 0, "merged": 0, "rejected": 0}

        clusters = self.find_clusters(project=project)
        stats["clusters_found"] = len(clusters)

        for cluster in clusters:
            result = self.merge_cluster(cluster)
            if result["merged_id"] is not None:
                stats["merged"] += 1
            else:
                stats["rejected"] += 1

        LOG(f"fact_merger.run: {stats}")
        return stats
