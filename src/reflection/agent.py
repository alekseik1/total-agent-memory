"""
Reflection Agent -- background process for knowledge consolidation.

Like sleep for the brain: transfers short-term to long-term memory,
consolidates patterns, resolves contradictions, and evolves the knowledge graph.

Orchestrates DigestPhase (cleanup) and SynthesizePhase (pattern finding).
Runs in three modes: quick (post-session), full (periodic), weekly (deep analysis).
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import sys
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Relative imports via sys.path for flat project structure
_src = str(Path(__file__).parent.parent)
if _src not in sys.path:
    sys.path.insert(0, _src)

from reflection.digest import DigestPhase
from reflection.synthesize import SynthesizePhase

LOG = lambda msg: sys.stderr.write(f"[memory-reflection] {msg}\n")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _new_id() -> str:
    return uuid.uuid4().hex


# Report keys holding per-phase stat dicts rather than scalars or free text.
_PHASE_KEYS = (
    "digest",
    "synthesis",
    "triple_extraction",
    "fact_merge",
    "deep_enrichment",
    "representations",
    "graph_importance",
)


def phase_errors(report: dict) -> dict[str, str]:
    """Collect the failure/skip reasons the phases reported.

    Every phase swallows its own exceptions and returns them as an `error` key
    (or a `deferred`/`skipped` marker when a dependency is missing), so nothing
    raises out of run_full. Pulling them together is what makes a broken phase
    visible — without it the only trace is a stderr line nobody reads.

    `skipped` is a plain item COUNT in most phases (queue rows dropped because
    their knowledge record was deleted) — only `fact_merge` uses it as a
    failure reason string. A numeric `skipped` is healthy-run noise, not an
    error, so it is reported only when it is not a number. `skipped_reason`
    (used by `_run_representations` when it declines to run at all, e.g. no
    embedder) is checked ahead of `skipped` so that a phase reporting both a
    numeric `skipped` count and a `skipped_reason` still surfaces the reason.
    """
    found: dict[str, str] = {}
    for key in _PHASE_KEYS:
        stats = report.get(key)
        if not isinstance(stats, dict):
            continue
        for marker in ("error", "deferred", "skipped_reason", "skipped"):
            value = stats.get(marker)
            if marker == "skipped" and isinstance(value, (int, float)):
                continue
            if value:
                found[key] = f"{marker}: {value}"
                break
    return found


class ReflectionAgent:
    """Background process that consolidates, synthesizes, and evolves knowledge.

    Like sleep for the brain -- transfers short-term to long-term memory.
    Orchestrates two phases:
      1. Digest: cleanup, dedup, decay, contradiction resolution
      2. Synthesize: pattern finding, clustering, skill proposals
    """

    def __init__(
        self,
        db: sqlite3.Connection,
        embedder=None,
        embed_identity: tuple[str, str] | None = None,
    ) -> None:
        """
        Args:
            db: SQLite connection (row_factory=Row).
            embedder: Optional callable (text: str) -> list[float]. If None,
                a fresh server.Store() is instantiated lazily for multi-repr
                generation. Injected in tests to avoid heavy init.
            embed_identity: Optional (embed_model, embed_provider) naming what
                `embedder` actually produces. The production caller
                (`tools/run_reflection.py`) builds its own `server.Store()` to
                make the embedder and discards it afterwards, so this agent
                never sees a live Store to ask — without this, merged rows get
                stamped from `config`, which can disagree with the Store that
                really wrote the vector (e.g. `FASTEMBED_MODEL` set without
                the matching `MEMORY_EMBED_MODEL`).
        """
        self.db = db
        self.digest = DigestPhase(db)
        self.synthesize = SynthesizePhase(db)
        self._injected_embedder = embedder
        self._embed_identity = embed_identity
        self._store = None

    async def run(self, scope: str = "full") -> dict:
        """
        Run reflection pipeline.

        Args:
            scope: 'quick' (digest only), 'full' (digest + synthesize),
                   'weekly' (full + deep analysis + digest generation)

        Returns:
            ReflectionReport as dict.
        """
        if scope == "quick":
            return await self.run_quick()
        elif scope == "weekly":
            return await self.run_weekly()
        else:
            return await self.run_full()

    async def run_quick(self) -> dict:
        """Quick reflection: dedup + decay only. Runs after each session."""
        LOG("Starting quick reflection...")
        started_at = _now()

        # Run synchronously — SQLite connections are not thread-safe
        digest_stats = self._run_digest_quick()

        report = {
            "id": _new_id(),
            "type": "session",
            "scope": "quick",
            "started_at": started_at,
            "completed_at": _now(),
            "digest": digest_stats,
            "synthesis": None,
            "weekly_digest": None,
        }

        self._save_report(report)
        LOG(f"Quick reflection complete: {digest_stats}")
        return report

    async def run_full(self) -> dict:
        """Full reflection: digest + synthesize + triple extraction + fact merging.

        Runs every 6 hours. SQLite is not thread-safe, so all phases are sync.
        """
        LOG("Starting full reflection...")
        started_at = _now()

        # Phase 1: Digest (dedup, decay, contradictions)
        digest_stats = self.digest.run()

        # Phase 2: Synthesize (clusters, patterns — depends on clean data)
        synthesis_stats = self.synthesize.run(days=7)

        # Phase 3: Drain triple extraction queue (async pipeline from memory_save)
        triple_stats = self._run_triple_extraction()

        # Phase 4: Semantic fact merging (LLM consolidation of related facts)
        merge_stats = self._run_fact_merger()

        # Phase 5: Drain deep enrichment queue (entities/intent/topics)
        enrich_stats = self._run_deep_enrichment()

        # Phase 6: Generate multi-representation embeddings (GEM-RAG)
        repr_stats = self._run_representations()

        report = {
            "id": _new_id(),
            "type": "periodic",
            "scope": "full",
            "started_at": started_at,
            "completed_at": _now(),
            "digest": digest_stats,
            "synthesis": synthesis_stats,
            "triple_extraction": triple_stats,
            "fact_merge": merge_stats,
            "deep_enrichment": enrich_stats,
            "representations": repr_stats,
            "weekly_digest": None,
        }

        self._save_report(report)
        LOG(
            f"Full reflection complete: digest={digest_stats}, "
            f"synthesis={synthesis_stats}, triples={triple_stats}, "
            f"merge={merge_stats}, enrich={enrich_stats}, repr={repr_stats}"
        )
        return report

    # ──────────────────────────────────────────────
    # Phase 3 — drain triple extraction queue
    # ──────────────────────────────────────────────

    def _run_triple_extraction(self, limit: int = 500) -> dict[str, int]:
        """Pull pending knowledge_ids from the queue and run deep KG extraction.

        Skipped silently if Ollama / model unavailable — items remain in the
        queue and will be drained on a future run when LLM comes back online.
        """
        try:
            from config import has_llm
            if not has_llm("triple"):
                LOG("triple extraction skipped: LLM unavailable")
                return {"processed": 0, "failed": 0, "skipped": 0, "deferred": "no_llm"}
        except Exception:
            pass

        try:
            from triple_extraction_queue import TripleExtractionQueue
            from ingestion.extractor import ConceptExtractor
        except Exception as e:  # noqa: BLE001
            LOG(f"triple extraction imports failed: {e}")
            return {"processed": 0, "failed": 0, "skipped": 0, "error": str(e)}

        q = TripleExtractionQueue(self.db)
        extractor = ConceptExtractor(self.db)

        def extract(knowledge_id: int, content: str) -> dict:
            return extractor.extract_and_link(
                text=content, knowledge_id=knowledge_id, deep=True
            )

        try:
            return q.process_pending(extract, limit=limit)
        except Exception as e:  # noqa: BLE001
            LOG(f"triple queue processing error: {e}")
            return {"processed": 0, "failed": 0, "skipped": 0, "error": str(e)}

    # ──────────────────────────────────────────────
    # Phase 4 — semantic fact merging
    # ──────────────────────────────────────────────

    def _run_fact_merger(self) -> dict[str, int]:
        """Find clusters of related (non-duplicate) facts and synthesize via LLM."""
        try:
            from fact_merger import FactMerger
        except Exception as e:  # noqa: BLE001
            LOG(f"fact_merger import failed: {e}")
            return {"clusters_found": 0, "merged": 0, "rejected": 0, "error": str(e)}

        try:
            similarity = self._make_cosine_similarity_fn()
            llm_merge = self._make_llm_merge_fn()
            if similarity is None or llm_merge is None:
                # Dependencies unavailable; skip silently
                return {"clusters_found": 0, "merged": 0, "rejected": 0, "skipped": "deps"}

            merger = FactMerger(
                self.db,
                similarity_fn=similarity,
                llm_merge_fn=llm_merge,
                on_merged=self._make_merge_hook(),
                vectors_fn=self._make_vectors_fn(),
            )
            return merger.run()
        except Exception as e:  # noqa: BLE001
            LOG(f"fact_merger error: {e}")
            return {"clusters_found": 0, "merged": 0, "rejected": 0, "error": str(e)}

    def _make_vectors_fn(self):
        """Build a bulk vector loader for clustering, or None if unavailable.

        One query for every candidate instead of two per pair. Lets FactMerger
        compare the whole set with a single matrix multiply.
        """
        import numpy as np

        try:
            count = self.db.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
        except Exception:  # noqa: BLE001
            return None
        if not count:
            return None

        def load(ids: list[int]) -> dict[int, "np.ndarray"]:
            if not ids:
                return {}
            vectors: dict[int, "np.ndarray"] = {}
            # Chunked to stay under SQLITE_MAX_VARIABLE_NUMBER (999 by default).
            for start in range(0, len(ids), 500):
                chunk = ids[start : start + 500]
                placeholders = ",".join("?" * len(chunk))
                rows = self.db.execute(
                    "SELECT knowledge_id, float32_vector, embed_dim FROM embeddings "
                    f"WHERE knowledge_id IN ({placeholders})",
                    chunk,
                ).fetchall()
                for row in rows:
                    try:
                        # A materialized Python list of floats per record
                        # doubles memory at scale — _banded_pairs_in_group
                        # copies it into a float32 ndarray anyway. Load it as
                        # one already.
                        vectors[row["knowledge_id"]] = np.frombuffer(
                            row["float32_vector"], dtype=np.float32
                        )
                    except Exception as e:  # noqa: BLE001
                        LOG(f"vector unpack failed for {row['knowledge_id']}: {e}")
            return vectors

        return load

    def _make_cosine_similarity_fn(self):
        """Build a cosine-similarity closure over the embeddings table.

        Returns None if no embeddings are stored yet (cold start).
        """
        import struct

        try:
            count = self.db.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
        except Exception:
            return None
        if not count:
            return None

        cache: dict[int, list[float]] = {}

        def _get(kid: int) -> list[float] | None:
            if kid in cache:
                return cache[kid]
            row = self.db.execute(
                "SELECT float32_vector, embed_dim FROM embeddings WHERE knowledge_id=?",
                (kid,),
            ).fetchone()
            if row is None:
                cache[kid] = None  # type: ignore[assignment]
                return None
            vec = list(struct.unpack(f"{row['embed_dim']}f", row["float32_vector"]))
            cache[kid] = vec
            return vec

        def cosine(id_a: int, id_b: int) -> float:
            va, vb = _get(id_a), _get(id_b)
            if not va or not vb or len(va) != len(vb):
                return 0.0
            num = sum(x * y for x, y in zip(va, vb))
            da = sum(x * x for x in va) ** 0.5
            db2 = sum(y * y for y in vb) ** 0.5
            if da == 0 or db2 == 0:
                return 0.0
            return max(0.0, min(1.0, num / (da * db2)))

        return cosine

    # ──────────────────────────────────────────────
    # Phase 6 — generate multi-representation embeddings
    # ──────────────────────────────────────────────

    def _run_representations(self, limit: int = 500) -> dict[str, int]:
        """Drain representations_queue: LLM-generate views + embed each.

        When LLM is unavailable we still run — the queue worker will save the
        `raw` embedding (no LLM needed) and skip summary/keywords/etc. The
        gate inside `generate_representations` returns empty for LLM views.
        """
        try:
            from representations_queue import RepresentationsQueue
            from representations import generate_representations
        except Exception as e:  # noqa: BLE001
            LOG(f"representations imports failed: {e}")
            return {"processed": 0, "failed": 0, "skipped": 0, "error": str(e)}

        q = RepresentationsQueue(self.db)

        embedder = self._make_embedder()
        if embedder is None:
            return {"processed": 0, "failed": 0, "skipped": 0, "skipped_reason": "no embedder"}

        try:
            return q.process_pending(
                generator=lambda content, project=None: generate_representations(content, project),
                embedder=embedder,
                model_name="auto",
                limit=limit,
            )
        except Exception as e:  # noqa: BLE001
            LOG(f"representations processing error: {e}")
            return {"processed": 0, "failed": 0, "skipped": 0, "error": str(e)}

    def _get_store(self):
        """Return a lazily-built server.Store, or None if it cannot be built.

        Cached: Store init loads the embedding model, so the merge hook and the
        embedder must not each pay for their own — including a failed attempt.
        `self._store` is `False` once a build has been tried and failed (a
        real sentinel distinct from the "not tried yet" `None`), so a second
        caller in the same run doesn't retry the same expensive failure.
        """
        if self._store is False:
            return None
        if self._store is not None:
            return self._store
        try:
            import server as _srv

            self._store = _srv.Store()
        except Exception as e:  # noqa: BLE001
            LOG(f"store init failed: {e}")
            self._store = False
            return None
        return self._store

    def _make_embedder(self):
        """Return an embedder callable, or None if unavailable.

        Priority: injected via constructor → lazy server.Store init.
        """
        if self._injected_embedder is not None:
            return self._injected_embedder
        store = self._get_store()
        if store is None:
            return None

        def embed(text: str) -> list[float]:
            embs = store.embed([text])
            return embs[0] if embs else []

        return embed

    def _resolve_embed_identity(self) -> tuple[str, str]:
        """Return (embed_model, provider) for rows this agent writes.

        Priority: identity supplied by the composer at construction time (the
        real Store the production caller built) → a Store this agent lazily
        built itself → config as a last resort. Config alone reads
        `MEMORY_EMBED_MODEL`, not the `FASTEMBED_MODEL` env var `server.py`
        actually uses for the fastembed provider, so a deployment that sets
        only `FASTEMBED_MODEL` would otherwise stamp the wrong embed_model.
        """
        if self._embed_identity is not None:
            return self._embed_identity
        store = self._store
        if store:  # not None (never tried) and not False (tried, failed)
            try:
                return store.embed_identity()
            except Exception as e:  # noqa: BLE001
                LOG(f"embed identity from store failed: {e}")
        try:
            import config as _config

            provider = _config.get_embed_provider()
            if provider == "fastembed":
                import os as _os

                model = _os.environ.get("FASTEMBED_MODEL", _config.get_embed_model(provider))
            else:
                model = _config.get_embed_model(provider)
            return model, provider
        except Exception as e:  # noqa: BLE001
            LOG(f"embed identity from config failed: {e}")
            return "unknown", "unknown"

    def _make_merge_hook(self):
        """Return a post-merge callback that makes a merged fact retrievable.

        FactMerger INSERTs straight into `knowledge`, which bypasses everything
        `memory_save` normally does afterwards. Without this hook a merged fact
        gets an FTS row from the table trigger and nothing else: no vector in
        `embeddings`, so it is invisible to semantic recall, and — because
        clustering scores an absent vector as 0.0 — it can never be merged
        again. Returns None when there is no embedder, in which case merging
        still proceeds (FTS-only, same as before).
        """
        embedder = self._make_embedder()
        if embedder is None:
            return None

        # Not `reembed`: src/reembed.py and scripts/reembed.py share a module
        # name, so which one an import resolves to depends on sys.path order.
        # multi_repr_store is unambiguous and its blob format is documented as
        # matching server._quantize_binary.
        from multi_repr_store import _binary_blob, _float32_blob

        # Written on self.db, not store.db: they are separate connections, and
        # under an injected in-memory db they are separate databases entirely.
        #
        # The name matters — tooling groups by embed_model to detect model drift
        # and decide what to re-embed, so a placeholder here would quietly
        # exclude merged records from that. Prefer the live Store's answer, and
        # fall back to config when the embedder was injected without one.
        model_name, provider = self._resolve_embed_identity()

        try:
            from representations_queue import RepresentationsQueue

            queue = RepresentationsQueue(self.db)
        except Exception as e:  # noqa: BLE001
            LOG(f"merge hook: representations queue unavailable: {e}")
            queue = None

        def on_merged(knowledge_id: int, content: str) -> None:
            try:
                vector = embedder(content)
                if vector:
                    self.db.execute(
                        """INSERT OR REPLACE INTO embeddings (
                               knowledge_id, binary_vector, float32_vector,
                               embed_model, embed_dim, created_at,
                               embedding_provider, embedding_space,
                               content_type, language
                           ) VALUES (?, ?, ?, ?, ?, ?, ?, 'text', 'text', NULL)""",
                        (
                            knowledge_id,
                            _binary_blob(vector),
                            _float32_blob(vector),
                            model_name,
                            len(vector),
                            _now(),
                            provider,
                        ),
                    )
                    self.db.commit()
            except Exception as e:  # noqa: BLE001
                LOG(f"merge hook: embedding {knowledge_id} failed: {e}")
            if queue is not None:
                try:
                    queue.enqueue(knowledge_id)
                except Exception as e:  # noqa: BLE001
                    LOG(f"merge hook: enqueue {knowledge_id} failed: {e}")

        return on_merged

    # ──────────────────────────────────────────────
    # Phase 5 — drain deep enrichment queue
    # ──────────────────────────────────────────────

    def _run_deep_enrichment(self, limit: int = 500) -> dict[str, int]:
        """Drain queue with deep_enricher.deep_enrich (entities/intent/topics).

        Skipped if no LLM — items wait in queue.
        """
        try:
            from config import has_llm
            if not has_llm("enrich"):
                LOG("deep enrichment skipped: LLM unavailable")
                return {"processed": 0, "failed": 0, "skipped": 0, "deferred": "no_llm"}
        except Exception:
            pass

        try:
            from deep_enrichment_queue import DeepEnrichmentQueue
            from deep_enricher import deep_enrich
        except Exception as e:  # noqa: BLE001
            LOG(f"deep enrichment imports failed: {e}")
            return {"processed": 0, "failed": 0, "skipped": 0, "error": str(e)}

        q = DeepEnrichmentQueue(self.db)
        try:
            return q.process_pending(deep_enrich, limit=limit)
        except Exception as e:  # noqa: BLE001
            LOG(f"deep enrichment processing error: {e}")
            return {"processed": 0, "failed": 0, "skipped": 0, "error": str(e)}

    def _make_llm_merge_fn(self):
        """Build a provider-backed merger. Returns None if LLM unreachable.

        Uses the `llm_provider` abstraction: Ollama by default, OpenAI /
        Anthropic / any OpenAI-compatible endpoint when configured via
        MEMORY_LLM_PROVIDER. On provider/network errors during a merge the
        closure degrades to an empty string (FactMerger's validator rejects
        empty merges, so we simply skip the cluster).
        """
        # Don't even build the closure if no LLM configured
        try:
            from config import has_llm
            if not has_llm():
                return None
        except Exception:
            pass

        import config as _config
        import urllib.error as _urlerr

        prompt_tmpl = (
            "You are consolidating related facts into ONE concise sentence.\n"
            "Preserve any code blocks, URLs, and file paths EXACTLY as written.\n"
            "Do not invent details not in the sources.\n\n"
            "SOURCES:\n{sources}\n\nMERGED:"
        )

        # Build provider once; reuse across cluster merges in this run.
        try:
            from llm_provider import make_provider
            provider = make_provider(_config.get_llm_provider())
        except Exception as exc:  # noqa: BLE001 — bad provider name etc.
            LOG(f"fact_merger provider init failed: {exc}")
            return None
        if not provider.available():
            LOG(f"fact_merger provider '{getattr(provider, 'name', '?')}' unavailable")
            return None

        model_name = _config.get_llm_model_for_provider(_config.get_llm_provider())
        timeout = 60.0

        def merge(contents: list[str]) -> str:
            sources = "\n\n---\n\n".join(f"- {c}" for c in contents)
            prompt = prompt_tmpl.format(sources=sources)
            try:
                return provider.complete(
                    prompt,
                    model=model_name,
                    max_tokens=200,
                    temperature=0.1,
                    timeout=timeout,
                )
            except (_urlerr.URLError, RuntimeError, OSError) as exc:
                sys.stderr.write(f"[memory-reflection] LLM merge error: {exc}\n")
                return ""

        return merge

    async def run_drain(self) -> dict:
        """Fast path: drain only the v6 async queues (Phase 3, 5, 6).

        Skips expensive digest/synthesize. Use when you want new edges from a
        handful of fresh saves ASAP (file-watch trigger after memory_save).
        Typical latency: 10-30s for <10 pending items vs 2-3 min for run_full.
        """
        LOG("Starting drain reflection (queues only)...")
        started_at = _now()

        triple_stats = self._run_triple_extraction()
        enrich_stats = self._run_deep_enrichment()
        repr_stats   = self._run_representations()

        report = {
            "id": _new_id(),
            # CHECK constraint on reflection_reports.type only allows
            # session|periodic|weekly|manual — use 'manual' for drain runs.
            "type": "manual",
            "scope": "drain",
            "started_at": started_at,
            "completed_at": _now(),
            "triple_extraction": triple_stats,
            "deep_enrichment": enrich_stats,
            "representations": repr_stats,
        }
        self._save_report(report)
        LOG(
            f"Drain complete: triples={triple_stats}, "
            f"enrich={enrich_stats}, repr={repr_stats}"
        )
        return report

    async def run_weekly(self) -> dict:
        """Weekly deep reflection with digest generation."""
        LOG("Starting weekly reflection...")
        started_at = _now()

        # Run synchronously — SQLite connections are not thread-safe
        # Phase 1: Full digest
        digest_stats = self.digest.run()

        # Phase 2: Extended synthesis (30 days lookback)
        synthesis_stats = self.synthesize.run(days=30)

        # Phase 3: Generate weekly digest report
        weekly_digest = self.synthesize.generate_weekly_digest()

        # Phase 4: Update graph importance via PageRank
        graph_importance_stats = self._update_graph_importance()

        report = {
            "id": _new_id(),
            "type": "weekly",
            "scope": "weekly",
            "started_at": started_at,
            "completed_at": _now(),
            "digest": digest_stats,
            "synthesis": synthesis_stats,
            "weekly_digest": weekly_digest,
            "graph_importance": graph_importance_stats,
        }

        self._save_report(report)
        LOG(f"Weekly reflection complete")
        return report

    def _run_digest_quick(self) -> dict:
        """Run a lightweight digest: only dedup and decay, skip contradiction analysis."""
        stats: dict = {
            "duplicates_merged": 0,
            "decay": {"checked": 0, "archived": 0, "kept": 0},
        }

        try:
            stats["duplicates_merged"] = self.digest.merge_duplicates()
        except Exception as e:
            LOG(f"quick merge_duplicates error: {e}")

        try:
            stats["decay"] = self.digest.apply_intelligent_decay()
        except Exception as e:
            LOG(f"quick apply_intelligent_decay error: {e}")

        return stats

    def _update_graph_importance(self) -> dict:
        """Update graph node importance via PageRank.

        Returns {} on success, {"error": message} when graph modules are
        missing or PageRank fails, so `run_weekly` can fold the outcome into
        its report instead of the failure only reaching stderr.
        """
        try:
            from graph.query import GraphQuery
            from graph.store import GraphStore

            store = GraphStore(self.db)
            query = GraphQuery(store)
            query.update_importance()
            LOG("Graph importance updated via PageRank")
            return {}
        except ImportError as e:
            LOG("graph.query not available, skipping PageRank update")
            return {"error": str(e)}
        except Exception as e:
            LOG(f"PageRank update failed: {e}")
            return {"error": str(e)}

    def _has_phase_errors_column(self) -> bool:
        try:
            cols = {
                r[1]
                for r in self.db.execute("PRAGMA table_info(reflection_reports)").fetchall()
            }
        except Exception:  # noqa: BLE001
            return False
        return "phase_errors" in cols

    def _save_report(self, report: dict) -> str:
        """Save reflection report to DB. Returns report ID."""
        report_id = report.get("id", _new_id())
        report_type = report.get("type", "periodic")

        # Calculate period from report timing
        now = datetime.now(timezone.utc)
        if report_type == "weekly":
            period_start = (now - timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%SZ")
        elif report_type == "periodic":
            period_start = (now - timedelta(hours=6)).strftime("%Y-%m-%dT%H:%M:%SZ")
        else:
            period_start = report.get("started_at", _now())
        period_end = report.get("completed_at", _now())

        # Extract stats for report fields
        digest = report.get("digest") or {}
        synthesis = report.get("synthesis") or {}

        columns = [
            "id", "period_start", "period_end", "type",
            "new_nodes", "patterns_found", "skills_refined",
            "rules_proposed", "contradictions", "archived",
            "focus_areas", "key_findings", "proposed_changes", "created_at",
        ]
        values = [
            report_id,
            period_start,
            period_end,
            report_type,
            synthesis.get("edges_strengthened", 0),
            synthesis.get("clusters_found", 0),
            synthesis.get("skills_proposed", 0),
            0,  # rules_proposed
            digest.get("contradictions_found", 0),
            digest.get("decay", {}).get("archived", 0),
            json.dumps((report.get("weekly_digest") or {}).get("focus_areas", [])),
            json.dumps((report.get("weekly_digest") or {}).get("top_concepts", [])),
            json.dumps(synthesis),
            _now(),
        ]

        # Only appended when migration 029 has run: a database that predates the
        # column must still get its report saved, not lose the whole row. When
        # the column is missing AND there were real phase errors, log them —
        # otherwise a pre-029 database drops them with no trace at all, the
        # exact failure mode 029 exists to end.
        errors = phase_errors(report)
        if self._has_phase_errors_column():
            columns.append("phase_errors")
            values.append(json.dumps(errors))
        elif errors:
            LOG(f"phase_errors dropped (migration 029 not applied): {errors}")

        try:
            self.db.execute(
                f"""INSERT INTO reflection_reports ({", ".join(columns)})
                   VALUES ({", ".join("?" * len(columns))})""",
                values,
            )
            self.db.commit()
            LOG(f"Saved reflection report: {report_id} ({report_type})")
        except Exception as e:
            LOG(f"Failed to save reflection report: {e}")

        return report_id

    def _get_pending_proposals(self) -> list[dict]:
        """Get pending proposals for user review."""
        rows = self.db.execute(
            """SELECT id, type, content, evidence, confidence, created_at
               FROM pending_proposals
               WHERE status = 'pending'
               ORDER BY confidence DESC, created_at DESC"""
        ).fetchall()

        proposals: list[dict] = []
        for row in rows:
            proposal = dict(row)
            # Parse JSON fields
            if isinstance(proposal.get("content"), str):
                try:
                    proposal["content"] = json.loads(proposal["content"])
                except (json.JSONDecodeError, TypeError):
                    pass
            if isinstance(proposal.get("evidence"), str):
                try:
                    proposal["evidence"] = json.loads(proposal["evidence"])
                except (json.JSONDecodeError, TypeError):
                    pass
            proposals.append(proposal)

        return proposals

    def approve_proposal(self, proposal_id: str) -> bool:
        """Mark proposal as approved and apply it.

        For skill proposals: creates the skill in the skills table.
        Returns True if proposal was found and approved.
        """
        row = self.db.execute(
            "SELECT id, type, content, status FROM pending_proposals WHERE id = ?",
            (proposal_id,),
        ).fetchone()

        if not row:
            LOG(f"Proposal {proposal_id} not found")
            return False

        if row["status"] != "pending":
            LOG(f"Proposal {proposal_id} already {row['status']}")
            return False

        # Parse content
        content = row["content"]
        if isinstance(content, str):
            try:
                content = json.loads(content)
            except (json.JSONDecodeError, TypeError):
                content = {}

        proposal_type = row["type"]

        # Apply the proposal
        if proposal_type == "skill" and isinstance(content, dict):
            self._apply_skill_proposal(content)

        # Mark as approved
        self.db.execute(
            """UPDATE pending_proposals
               SET status = 'approved', reviewed_at = ?
               WHERE id = ?""",
            (_now(), proposal_id),
        )
        self.db.commit()
        LOG(f"Approved proposal {proposal_id} ({proposal_type})")
        return True

    def reject_proposal(self, proposal_id: str) -> bool:
        """Mark proposal as rejected."""
        row = self.db.execute(
            "SELECT id, status FROM pending_proposals WHERE id = ?",
            (proposal_id,),
        ).fetchone()

        if not row:
            LOG(f"Proposal {proposal_id} not found")
            return False

        if row["status"] != "pending":
            LOG(f"Proposal {proposal_id} already {row['status']}")
            return False

        self.db.execute(
            """UPDATE pending_proposals
               SET status = 'rejected', reviewed_at = ?
               WHERE id = ?""",
            (_now(), proposal_id),
        )
        self.db.commit()
        LOG(f"Rejected proposal {proposal_id}")
        return True

    def _apply_skill_proposal(self, content: dict) -> None:
        """Create a skill from an approved proposal."""
        skill_id = _new_id()
        name = content.get("skill_name", f"skill_{skill_id[:8]}")
        trigger = content.get("trigger_pattern", "")
        steps = content.get("steps", [])
        projects = content.get("projects", [])
        episode_ids = content.get("episode_ids", [])

        try:
            self.db.execute(
                """INSERT INTO skills
                   (id, name, trigger_pattern, steps, anti_patterns,
                    times_used, success_rate, projects, stack,
                    related_skills, learned_from, status, created_at)
                   VALUES (?, ?, ?, ?, ?, 0, ?, ?, ?, ?, ?, 'draft', ?)""",
                (
                    skill_id,
                    name,
                    trigger,
                    json.dumps(steps),
                    json.dumps([]),
                    content.get("success_rate", 0.0),
                    json.dumps(projects),
                    json.dumps([]),
                    json.dumps([]),
                    json.dumps(episode_ids),
                    _now(),
                ),
            )
            LOG(f"Created skill from proposal: {name} ({skill_id})")
        except sqlite3.IntegrityError:
            LOG(f"Skill {name} already exists, skipping")
