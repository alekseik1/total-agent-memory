#!/usr/bin/env python3
"""
Vito Retrieval Quality Benchmark

Measures search quality across all retrieval methods:
- FTS5 keyword search (BM25)
- Semantic search (binary quantization / ChromaDB)
- Combined 4-tier search (FTS5 + semantic + fuzzy + graph)
- Combined + Cognitive engine (spreading activation)

Metrics: Recall@K (R@1, R@5, R@10), MRR, average latency.

Usage:
    python src/tools/benchmark.py [--db PATH] [--queries 50] [--save]

Scheduler integration (not auto-modified):
    Run weekly (e.g., Sunday 04:00) via cron or scheduler.py:
        0 4 * * 0 cd ~/claude-memory-server && .venv/bin/python src/tools/benchmark.py --save
    Results are saved to memory with tags ['benchmark', 'retrieval-quality'].
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sqlite3
import struct
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

# Ensure src/ is importable
_src_dir = str(Path(__file__).resolve().parent.parent)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

# ── Optional imports (graceful degradation) ──

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import chromadb
    HAS_CHROMA = True
except ImportError:
    HAS_CHROMA = False

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

OLLAMA_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "qwen3-embedding:4b")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MEMORY_DIR = Path(os.environ.get("CLAUDE_MEMORY_DIR", os.path.expanduser("~/.claude-memory")))

LOG = lambda msg: sys.stderr.write(f"[benchmark] {msg}\n")


# ── Stop-words for keyword extraction ──

_STOP_WORDS = frozenset(
    "the a an is are was were be been being have has had do does did will would "
    "shall should may might can could must need dare ought i me my we our you your "
    "he she it they them his her its their this that these those what which who whom "
    "how when where why all each every both few more most other some such no nor not "
    "only same so than too very of in on at to for from by with as into through "
    "during before after above below between under again further then once and but or "
    "if because until while about against over также для что это как при или нет да "
    "если все есть был были будет быть его она они этот тот где когда".split()
)


# ═════════════════════════════════════════════════════════════
# Data structures
# ═════════════════════════════════════════════════════════════

@dataclass
class TestQuery:
    """A single test query with expected result."""
    query: str
    expected_id: int
    keywords: list[str]
    source_content: str = ""
    source_project: str = ""


@dataclass
class MethodResult:
    """Results for one search method across all queries."""
    name: str
    recall_at_1: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    mrr: float = 0.0
    avg_ms: float = 0.0
    total_queries: int = 0
    errors: int = 0


@dataclass
class BenchmarkReport:
    """Complete benchmark report."""
    methods: list[MethodResult] = field(default_factory=list)
    num_queries: int = 0
    db_records: int = 0
    generated_at: str = ""
    generation_time_s: float = 0.0


# ═════════════════════════════════════════════════════════════
# Query generation
# ═════════════════════════════════════════════════════════════

def _extract_keywords(content: str, n: int = 5) -> list[str]:
    """Extract top N meaningful keywords from content."""
    # Remove special chars, keep words
    words = re.findall(r'[a-zA-Z\u0400-\u04FF]{3,}', content)
    # Filter stop words and too-short words
    words = [w.lower() for w in words if w.lower() not in _STOP_WORDS and len(w) >= 3]

    # Frequency-based selection (prefer rarer, longer words)
    freq: dict[str, int] = defaultdict(int)
    for w in words:
        freq[w] += 1

    # Score: length bonus + moderate frequency (1-3 occurrences preferred)
    scored = []
    for w, count in freq.items():
        score = len(w) * 0.3 + min(count, 3) * 0.5
        scored.append((w, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    selected = [w for w, _ in scored[:n * 2]]

    # Pick N diverse keywords (not too similar to each other)
    result: list[str] = []
    for w in selected:
        if len(result) >= n:
            break
        is_dup = any(
            SequenceMatcher(None, w, existing).ratio() > 0.7
            for existing in result
        )
        if not is_dup:
            result.append(w)

    return result


def _form_natural_query(keywords: list[str]) -> str:
    """Form a natural-looking search query from keywords."""
    if len(keywords) <= 2:
        return " ".join(keywords)
    # Take 3-4 keywords, join naturally
    subset = random.sample(keywords, min(4, len(keywords)))
    return " ".join(subset)


def _paraphrase_query(content: str, keywords: list[str]) -> str:
    """Create a semantic paraphrase — rephrase meaning without using exact keywords."""
    # Extract first meaningful sentence
    sentences = [s.strip() for s in re.split(r'[.\n]', content) if len(s.strip()) > 20]
    if not sentences:
        return " ".join(keywords)

    first = sentences[0][:150]

    # Strategy: rephrase using synonyms and different wording
    replacements = {
        "implement": "build", "create": "make", "add": "introduce",
        "fix": "resolve", "update": "modify", "delete": "remove",
        "configuration": "setup", "integration": "connecting",
        "error": "problem", "bug": "issue", "feature": "capability",
        "deploy": "launch", "migrate": "move", "refactor": "restructure",
        "реализовать": "сделать", "создать": "построить", "добавить": "внедрить",
        "исправить": "решить", "обновить": "изменить", "удалить": "убрать",
        "настройка": "конфигурация", "интеграция": "подключение",
        "ошибка": "проблема", "баг": "дефект",
    }

    paraphrased = first.lower()
    for old, new in replacements.items():
        paraphrased = paraphrased.replace(old, new)

    # Remove exact keywords to force semantic matching
    for kw in keywords[:3]:
        paraphrased = paraphrased.replace(kw.lower(), "")

    paraphrased = re.sub(r'\s+', ' ', paraphrased).strip()

    # If too short after removal, use a question form
    if len(paraphrased) < 15:
        return f"how to {' '.join(keywords[:2])}"

    return paraphrased


def generate_test_queries(db_path: str, n: int = 50, mode: str = "keyword") -> list[TestQuery]:
    """Generate test queries from existing high-recall records.

    mode: "keyword" — extract keywords (tests FTS), "semantic" — paraphrase (tests embeddings)

    Selects records with recall_count >= 2 (previously useful),
    extracts keywords, and forms natural queries.

    Args:
        db_path: Path to memory.db
        n: Number of test queries to generate

    Returns:
        List of TestQuery objects
    """
    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row

    # Get high-recall records with enough content
    rows = db.execute("""
        SELECT id, content, project, tags, recall_count
        FROM knowledge
        WHERE status = 'active'
          AND recall_count >= 2
          AND length(content) >= 50
        ORDER BY recall_count DESC, last_recalled DESC
    """).fetchall()

    db.close()

    if not rows:
        LOG("WARNING: No records with recall_count >= 2 found")
        return []

    # Sample up to n records (prefer higher recall_count)
    candidates = list(rows)
    if len(candidates) > n:
        # Weighted sampling: higher recall_count = higher probability
        weights = [r["recall_count"] ** 0.5 for r in candidates]
        total_w = sum(weights)
        probs = [w / total_w for w in weights]
        indices = list(range(len(candidates)))
        selected_indices: set[int] = set()
        attempts = 0
        while len(selected_indices) < n and attempts < n * 10:
            idx = random.choices(indices, weights=probs, k=1)[0]
            selected_indices.add(idx)
            attempts += 1
        candidates = [candidates[i] for i in sorted(selected_indices)]
    else:
        random.shuffle(candidates)

    queries: list[TestQuery] = []
    for row in candidates[:n]:
        content = row["content"]
        keywords = _extract_keywords(content, n=5)
        if len(keywords) < 2:
            continue  # Not enough keywords to form a query

        if mode == "semantic":
            query = _paraphrase_query(content, keywords)
        else:
            query = _form_natural_query(keywords)
        queries.append(TestQuery(
            query=query,
            expected_id=row["id"],
            keywords=keywords,
            source_content=content[:200],
            source_project=row["project"] or "general",
        ))

    LOG(f"Generated {len(queries)} test queries from {len(rows)} high-recall records")
    return queries


# ═════════════════════════════════════════════════════════════
# Search methods
# ═════════════════════════════════════════════════════════════

def _fts_escape(word: str) -> str:
    """Escape a word for FTS5 MATCH."""
    return '"' + word.replace('"', '""') + '"'


def _search_fts(db: sqlite3.Connection, query: str, limit: int = 10) -> list[int]:
    """FTS5-only keyword search with BM25 scoring.

    Returns list of knowledge IDs sorted by relevance.
    """
    fts_q = " OR ".join(
        _fts_escape(w) for w in re.split(r'\s+', query) if len(w) > 2
    ) or _fts_escape(query)

    try:
        rows = db.execute("""
            SELECT k.id
            FROM knowledge_fts f
            JOIN knowledge k ON k.id = f.rowid
            WHERE knowledge_fts MATCH ? AND k.status = 'active'
            ORDER BY bm25(knowledge_fts)
            LIMIT ?
        """, (fts_q, limit)).fetchall()
        return [r[0] for r in rows]
    except Exception as e:
        LOG(f"FTS search error: {e}")
        return []


_fastembed_model = None

def _ollama_embed(text: str) -> list[float] | None:
    """Get embedding via qwen3-embedding:4b (2560-dim, matches re-embedded data)."""
    try:
        import urllib.request
        payload = json.dumps({"model": OLLAMA_EMBED_MODEL, "prompt": text}).encode()
        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/embeddings",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            return data["embedding"]
    except Exception as e:
        LOG(f"Ollama embed error: {e}")
        return None


def _search_semantic(db: sqlite3.Connection, query: str, limit: int = 10) -> list[int]:
    """Semantic search via binary quantization.

    Two-stage: Hamming distance pre-filter -> cosine re-rank.
    Returns list of knowledge IDs sorted by similarity.
    """
    if not HAS_NUMPY:
        LOG("NumPy not available, skipping semantic search")
        return []

    query_emb = _ollama_embed(query)
    if query_emb is None:
        return []

    query_arr = np.array(query_emb, dtype=np.float32)

    # Load binary vectors
    rows = db.execute("""
        SELECT e.knowledge_id, e.binary_vector, e.float32_vector
        FROM embeddings e
        JOIN knowledge k ON e.knowledge_id = k.id
        WHERE k.status = 'active'
    """).fetchall()

    if not rows:
        return []

    kid_list = [r[0] for r in rows]

    # For small DBs (<5000): full float32 cosine (accurate, still fast ~5ms)
    # For large DBs: binary pre-filter → cosine rerank
    if len(rows) < 5000:
        # Direct float32 cosine similarity (no binary quantization loss)
        results: list[tuple[int, float]] = []
        query_norm = np.linalg.norm(query_arr)
        for i, r in enumerate(rows):
            blob = r[2]
            if not blob:
                continue
            n_floats = len(blob) // 4
            vec = np.array(struct.unpack(f'{n_floats}f', blob), dtype=np.float32)
            cos_sim = float(np.dot(query_arr, vec)) / max(float(query_norm * np.linalg.norm(vec)), 1e-8)
            results.append((kid_list[i], cos_sim))
    else:
        # Binary pre-filter for large DBs
        bin_vecs = np.array([np.frombuffer(r[1], dtype=np.uint8) for r in rows])
        query_binary = np.packbits(np.where(query_arr > 0, 1, 0).astype(np.uint8))
        hamming = np.sum(np.unpackbits(bin_vecs ^ query_binary, axis=1), axis=1)
        n_candidates = min(200, len(rows))
        top_indices = np.argpartition(hamming, n_candidates)[:n_candidates]
        results = []
        query_norm = np.linalg.norm(query_arr)
        for idx in top_indices:
            blob = rows[idx][2]
            if not blob:
                continue
            n_floats = len(blob) // 4
            vec = np.array(struct.unpack(f'{n_floats}f', blob), dtype=np.float32)
            cos_sim = float(np.dot(query_arr, vec)) / max(float(query_norm * np.linalg.norm(vec)), 1e-8)
            results.append((kid_list[idx], cos_sim))

    results.sort(key=lambda x: x[1], reverse=True)
    return [kid for kid, _ in results[:limit]]


def _search_combined(db: sqlite3.Connection, query: str, limit: int = 10) -> list[int]:
    """Full 4-tier search: FTS5 + semantic + fuzzy + graph.

    Mirrors production search from server.py Recall.search() but
    without caching, reranking, or decay scoring.
    Returns list of knowledge IDs sorted by combined score.
    """
    results: dict[int, float] = {}

    # Tier 1: FTS5
    fts_q = " OR ".join(
        _fts_escape(w) for w in re.split(r'\s+', query) if len(w) > 2
    ) or _fts_escape(query)

    try:
        fts_rows = db.execute("""
            SELECT k.id, bm25(knowledge_fts) AS _bm25
            FROM knowledge_fts f
            JOIN knowledge k ON k.id = f.rowid
            WHERE knowledge_fts MATCH ? AND k.status = 'active'
            ORDER BY bm25(knowledge_fts)
            LIMIT ?
        """, (fts_q, limit * 3)).fetchall()

        raw_scores = [abs(r[1]) for r in fts_rows]
        max_bm25 = max(raw_scores) if raw_scores else 1.0
        for r in fts_rows:
            bm25_score = (abs(r[1]) / max(max_bm25, 0.01)) * 2.0
            results[r[0]] = max(0.5, bm25_score)
    except Exception:
        pass

    # Tier 2: Semantic
    if HAS_NUMPY:
        query_emb = _ollama_embed(query)
        if query_emb is not None:
            query_arr = np.array(query_emb, dtype=np.float32)
            emb_rows = db.execute("""
                SELECT e.knowledge_id, e.binary_vector, e.float32_vector
                FROM embeddings e
                JOIN knowledge k ON e.knowledge_id = k.id
                WHERE k.status = 'active'
            """).fetchall()

            if emb_rows:
                kid_list = [r[0] for r in emb_rows]
                query_norm = np.linalg.norm(query_arr)

                if len(emb_rows) < 5000:
                    # Full float32 cosine for small DBs
                    scan_range = range(len(emb_rows))
                else:
                    # Binary pre-filter for large DBs
                    bin_vecs = np.array([np.frombuffer(r[1], dtype=np.uint8) for r in emb_rows])
                    query_binary = np.packbits(np.where(query_arr > 0, 1, 0).astype(np.uint8))
                    hamming = np.sum(np.unpackbits(bin_vecs ^ query_binary, axis=1), axis=1)
                    n_cand = min(200, len(emb_rows))
                    scan_range = np.argpartition(hamming, n_cand)[:n_cand]

                for idx in scan_range:
                    kid = kid_list[idx]
                    blob = emb_rows[idx][2]
                    if not blob:
                        continue
                    n_floats = len(blob) // 4
                    vec = np.array(struct.unpack(f'{n_floats}f', blob), dtype=np.float32)
                    cos_sim = max(0.0, float(np.dot(query_arr, vec)) / max(float(query_norm * np.linalg.norm(vec)), 1e-8))
                    if kid in results:
                        results[kid] += cos_sim
                    else:
                        results[kid] = cos_sim

    # Tier 3: Fuzzy
    if len(results) < limit:
        try:
            candidates = db.execute("""
                SELECT id, content FROM knowledge
                WHERE status = 'active'
                ORDER BY last_confirmed DESC
                LIMIT ?
            """, (limit * 5,)).fetchall()

            ql = query.lower()
            for r in candidates:
                if r[0] in results:
                    continue
                ratio = SequenceMatcher(None, ql, r[1][:200].lower()).ratio()
                if ratio > 0.35:
                    results[r[0]] = ratio * 0.6
        except Exception:
            pass

    # Tier 4: Graph expansion (1-hop from top 5)
    top5 = sorted(results, key=results.get, reverse=True)[:5]
    for kid in top5:
        try:
            graph_rows = db.execute("""
                SELECT k.id FROM relations rel
                JOIN knowledge k ON k.id = CASE
                    WHEN rel.from_id = ? THEN rel.to_id ELSE rel.from_id END
                WHERE (rel.from_id = ? OR rel.to_id = ?) AND k.status = 'active'
            """, (kid, kid, kid)).fetchall()
            for gr in graph_rows:
                gid = gr[0]
                if gid not in results:
                    results[gid] = results[kid] * 0.4
        except Exception:
            pass

    # Sort by score, return top-limit IDs
    sorted_ids = sorted(results, key=results.get, reverse=True)
    return sorted_ids[:limit]


def _search_cognitive(db: sqlite3.Connection, query: str, limit: int = 10) -> list[int]:
    """Combined search + cognitive engine (spreading activation).

    Adds spreading activation on top of the 4-tier search.
    Flow: extract keywords -> find_seed_nodes -> spread -> get_activated_memories.
    Returns list of knowledge IDs.
    """
    # Start with combined results
    base_ids = _search_combined(db, query, limit=limit)

    # Try spreading activation via graph
    try:
        from associative.activation import SpreadingActivation
        sa = SpreadingActivation(db)

        # Extract keywords from query as concept names
        keywords = re.findall(r'[a-zA-Z\u0400-\u04FF]{3,}', query)
        keywords = [w for w in keywords if w.lower() not in _STOP_WORDS]

        if not keywords:
            return base_ids

        # Find graph nodes matching query keywords
        seed_nodes = sa.find_seed_nodes(keywords)
        if not seed_nodes:
            return base_ids

        # Spread activation through graph
        activation_map = sa.spread(seed_nodes, depth=2)
        if not activation_map:
            return base_ids

        # Get knowledge records connected to activated nodes
        activated_memories = sa.get_activated_memories(activation_map, top_k=limit)

        # Merge: base results first, then activated memories not already present
        result_set = set(base_ids)
        extended = list(base_ids)
        for knowledge_id, _score in activated_memories:
            if knowledge_id not in result_set and len(extended) < limit:
                extended.append(knowledge_id)
                result_set.add(knowledge_id)

        return extended[:limit]
    except Exception as e:
        LOG(f"Cognitive engine error: {e}")
        return base_ids


# ═════════════════════════════════════════════════════════════
# Metric calculations
# ═════════════════════════════════════════════════════════════

def _calc_recall_at_k(results: list[int], expected_id: int, k: int) -> float:
    """1.0 if expected_id appears in results[:k], else 0.0."""
    return 1.0 if expected_id in results[:k] else 0.0


def _calc_mrr(results: list[int], expected_id: int) -> float:
    """Mean Reciprocal Rank: 1/(rank of expected_id), or 0.0 if not found."""
    try:
        rank = results.index(expected_id) + 1
        return 1.0 / rank
    except ValueError:
        return 0.0


# ═════════════════════════════════════════════════════════════
# Benchmark runner
# ═════════════════════════════════════════════════════════════

def _benchmark_method(
    name: str,
    search_fn: Any,
    db: sqlite3.Connection,
    queries: list[TestQuery],
) -> MethodResult:
    """Run all queries through a single search method and collect metrics."""
    r1_total = 0.0
    r5_total = 0.0
    r10_total = 0.0
    mrr_total = 0.0
    time_total = 0.0
    errors = 0

    for tq in queries:
        try:
            t0 = time.perf_counter()
            result_ids = search_fn(db, tq.query, limit=10)
            elapsed = (time.perf_counter() - t0) * 1000  # ms

            r1_total += _calc_recall_at_k(result_ids, tq.expected_id, 1)
            r5_total += _calc_recall_at_k(result_ids, tq.expected_id, 5)
            r10_total += _calc_recall_at_k(result_ids, tq.expected_id, 10)
            mrr_total += _calc_mrr(result_ids, tq.expected_id)
            time_total += elapsed
        except Exception as e:
            LOG(f"  Error in {name} for query '{tq.query[:40]}': {e}")
            errors += 1

    n = len(queries)
    if n == 0:
        return MethodResult(name=name)

    return MethodResult(
        name=name,
        recall_at_1=r1_total / n,
        recall_at_5=r5_total / n,
        recall_at_10=r10_total / n,
        mrr=mrr_total / n,
        avg_ms=time_total / n,
        total_queries=n,
        errors=errors,
    )


def run_benchmark(db_path: str, num_queries: int = 50) -> BenchmarkReport:
    """Run full retrieval benchmark across all search methods.

    Args:
        db_path: Path to memory.db
        num_queries: Number of test queries to generate

    Returns:
        BenchmarkReport with metrics for each method
    """
    t_start = time.perf_counter()

    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row

    db_records = db.execute("SELECT COUNT(*) FROM knowledge WHERE status='active'").fetchone()[0]

    LOG(f"DB: {db_records} active records")
    LOG(f"Generating {num_queries} test queries (keyword + semantic)...")

    queries = generate_test_queries(db_path, num_queries, mode="keyword")
    if not queries:
        LOG("ERROR: Could not generate any test queries")
        db.close()
        return BenchmarkReport(
            num_queries=0,
            db_records=db_records,
            generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )

    actual_n = len(queries)
    LOG(f"Running benchmark with {actual_n} queries...\n")

    # Check available methods
    has_embeddings = False
    try:
        emb_count = db.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
        has_embeddings = emb_count > 0
        LOG(f"Embeddings: {emb_count} records")
    except Exception:
        LOG("Embeddings table not found, semantic search will be skipped")

    has_relations = False
    try:
        rel_count = db.execute("SELECT COUNT(*) FROM relations").fetchone()[0]
        has_relations = rel_count > 0
        LOG(f"Relations: {rel_count} edges")
    except Exception:
        LOG("Relations table not found, graph search will be limited")

    methods: list[MethodResult] = []

    # 1. FTS5 only
    LOG("  [1/4] FTS5 keyword search...")
    methods.append(_benchmark_method("FTS5 only", _search_fts, db, queries))
    LOG(f"         R@10={methods[-1].recall_at_10:.2f} MRR={methods[-1].mrr:.2f} "
        f"avg={methods[-1].avg_ms:.1f}ms")

    # 2. Semantic only
    if has_embeddings:
        LOG("  [2/4] Semantic search (binary quantization)...")
        methods.append(_benchmark_method("Semantic only", _search_semantic, db, queries))
        LOG(f"         R@10={methods[-1].recall_at_10:.2f} MRR={methods[-1].mrr:.2f} "
            f"avg={methods[-1].avg_ms:.1f}ms")
    else:
        LOG("  [2/4] Semantic search SKIPPED (no embeddings)")
        methods.append(MethodResult(name="Semantic only", total_queries=actual_n))

    # 3. Combined 4-tier
    LOG("  [3/4] Combined 4-tier search...")
    methods.append(_benchmark_method("Combined 4-tier", _search_combined, db, queries))
    LOG(f"         R@10={methods[-1].recall_at_10:.2f} MRR={methods[-1].mrr:.2f} "
        f"avg={methods[-1].avg_ms:.1f}ms")

    # 4. Combined + Cognitive
    LOG("  [4/4] Combined + Cognitive engine...")
    methods.append(_benchmark_method("+ Cognitive", _search_cognitive, db, queries))
    LOG(f"         R@10={methods[-1].recall_at_10:.2f} MRR={methods[-1].mrr:.2f} "
        f"avg={methods[-1].avg_ms:.1f}ms")

    # 5-6. Semantic queries (paraphrased — no keyword overlap)
    sem_queries = generate_test_queries(db_path, num_queries, mode="semantic")
    if sem_queries and has_embeddings:
        LOG(f"\n  --- Semantic queries ({len(sem_queries)} paraphrased) ---")
        LOG("  [5/6] Semantic (paraphrased queries)...")
        methods.append(_benchmark_method("Semantic (paraphrased)", _search_semantic, db, sem_queries))
        LOG(f"         R@10={methods[-1].recall_at_10:.2f} MRR={methods[-1].mrr:.2f} "
            f"avg={methods[-1].avg_ms:.1f}ms")

        LOG("  [6/6] Combined (paraphrased queries)...")
        methods.append(_benchmark_method("Combined (paraphrased)", _search_combined, db, sem_queries))
        LOG(f"         R@10={methods[-1].recall_at_10:.2f} MRR={methods[-1].mrr:.2f} "
            f"avg={methods[-1].avg_ms:.1f}ms")

    db.close()

    elapsed = time.perf_counter() - t_start
    LOG(f"\nBenchmark completed in {elapsed:.1f}s")

    return BenchmarkReport(
        methods=methods,
        num_queries=actual_n,
        db_records=db_records,
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        generation_time_s=round(elapsed, 2),
    )


# ═════════════════════════════════════════════════════════════
# Report formatting
# ═════════════════════════════════════════════════════════════

def format_benchmark_report(report: BenchmarkReport) -> str:
    """Format benchmark report as a readable table.

    Suitable for CLI output or Telegram messages.
    """
    lines: list[str] = []
    lines.append("")
    lines.append("=== Vito Retrieval Benchmark ===")
    lines.append(f"Queries: {report.num_queries} | DB: {report.db_records} records | "
                 f"Time: {report.generation_time_s}s")
    lines.append(f"Generated: {report.generated_at}")
    lines.append("")

    # Table header
    header = f"{'Method':<20} {'R@1':>6} {'R@5':>6} {'R@10':>6} {'MRR':>6} {'Avg ms':>8} {'Err':>4}"
    lines.append(header)
    lines.append("\u2500" * len(header))

    for m in report.methods:
        if m.total_queries == 0 and m.recall_at_10 == 0:
            line = f"{m.name:<20} {'N/A':>6} {'N/A':>6} {'N/A':>6} {'N/A':>6} {'N/A':>8} {'N/A':>4}"
        else:
            line = (
                f"{m.name:<20} "
                f"{m.recall_at_1:>6.2f} "
                f"{m.recall_at_5:>6.2f} "
                f"{m.recall_at_10:>6.2f} "
                f"{m.mrr:>6.2f} "
                f"{m.avg_ms:>7.1f} "
                f"{m.errors:>4}"
            )
        lines.append(line)

    lines.append("")

    # Best method highlight
    valid = [m for m in report.methods if m.total_queries > 0 and m.recall_at_10 > 0]
    if valid:
        best_r10 = max(valid, key=lambda m: m.recall_at_10)
        best_mrr = max(valid, key=lambda m: m.mrr)
        lines.append(f"Best R@10: {best_r10.name} ({best_r10.recall_at_10:.2f})")
        lines.append(f"Best MRR:  {best_mrr.name} ({best_mrr.mrr:.2f})")

    lines.append("")
    return "\n".join(lines)


def report_to_dict(report: BenchmarkReport) -> dict:
    """Convert report to serializable dict for saving to memory."""
    return {
        "num_queries": report.num_queries,
        "db_records": report.db_records,
        "generated_at": report.generated_at,
        "generation_time_s": report.generation_time_s,
        "methods": [
            {
                "name": m.name,
                "R@1": round(m.recall_at_1, 4),
                "R@5": round(m.recall_at_5, 4),
                "R@10": round(m.recall_at_10, 4),
                "MRR": round(m.mrr, 4),
                "avg_ms": round(m.avg_ms, 2),
                "errors": m.errors,
            }
            for m in report.methods
        ],
    }


# ═════════════════════════════════════════════════════════════
# Save to memory
# ═════════════════════════════════════════════════════════════

def save_to_memory(db_path: str, report: BenchmarkReport) -> None:
    """Save benchmark results to memory.db as a fact record.

    Tags: ['benchmark', 'retrieval-quality', YYYY-MM-DD]
    """
    db = sqlite3.connect(db_path)

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    report_dict = report_to_dict(report)
    content = f"Retrieval benchmark results ({today}): {json.dumps(report_dict, ensure_ascii=False)}"
    tags = json.dumps(["benchmark", "retrieval-quality", today])

    # Check for duplicate (same day)
    existing = db.execute(
        "SELECT id FROM knowledge WHERE type='fact' AND tags LIKE ? AND created_at LIKE ?",
        (f'%{today}%', f'{today}%')
    ).fetchone()

    if existing:
        db.execute(
            "UPDATE knowledge SET content = ?, tags = ?, last_confirmed = ? WHERE id = ?",
            (content, tags, now, existing[0])
        )
        LOG(f"Updated existing benchmark record id={existing[0]}")
    else:
        import uuid
        session_id = f"benchmark-{uuid.uuid4().hex[:8]}"
        db.execute("""
            INSERT INTO knowledge (session_id, type, content, context, project, tags,
                                   status, confidence, source, created_at, last_confirmed)
            VALUES (?, 'fact', ?, 'automated retrieval benchmark', 'claude-total-memory', ?,
                    'active', 0.9, 'auto', ?, ?)
        """, (session_id, content, tags, now, now))

        # The k_fts_i trigger indexes the row.
        new_id = db.execute("SELECT last_insert_rowid()").fetchone()[0]

        LOG(f"Saved benchmark to memory id={new_id}")

    db.commit()
    db.close()


# ═════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Vito Retrieval Quality Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/tools/benchmark.py
  python src/tools/benchmark.py --queries 100
  python src/tools/benchmark.py --save
  python src/tools/benchmark.py --db /path/to/memory.db --queries 30 --save

Scheduler (cron):
  0 4 * * 0 cd ~/claude-memory-server && .venv/bin/python src/tools/benchmark.py --save
        """,
    )
    parser.add_argument(
        "--db",
        default=str(MEMORY_DIR / "memory.db"),
        help="Path to memory.db (default: ~/.claude-memory/memory.db)",
    )
    parser.add_argument(
        "--queries",
        type=int,
        default=50,
        help="Number of test queries to generate (default: 50)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save results to memory with tags ['benchmark', 'retrieval-quality']",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON instead of table",
    )

    args = parser.parse_args()

    if not Path(args.db).exists():
        print(f"ERROR: Database not found: {args.db}", file=sys.stderr)
        sys.exit(1)

    report = run_benchmark(args.db, args.queries)

    if args.json:
        print(json.dumps(report_to_dict(report), indent=2, ensure_ascii=False))
    else:
        print(format_benchmark_report(report))

    if args.save and report.num_queries > 0:
        save_to_memory(args.db, report)
        LOG("Results saved to memory")


def run_and_save_benchmark(db_path: str) -> None:
    """Entry point for scheduler — run benchmark and save results."""
    LOG("=== WEEKLY BENCHMARK STARTING ===")
    report = run_benchmark(db_path, num_queries=50)
    save_to_memory(db_path, report)
    LOG(f"=== WEEKLY BENCHMARK COMPLETE === (Combined R@10={report.combined.recall_at_10:.2f})")


if __name__ == "__main__":
    main()
