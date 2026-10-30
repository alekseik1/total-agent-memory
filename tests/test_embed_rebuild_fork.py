"""Fork behaviour on top of upstream `embed_identified`: memory_rebuild_embeddings
`stale_only`, its abort when a fallback answers, and one model per embedded batch."""

import json
import sqlite3

import pytest

import server


class _Embedder:
    """Stands in for the SentenceTransformer fallback (384 dims)."""

    def __init__(self, dim=384):
        self.dim = dim

    def encode(self, batch):
        import numpy as np

        return np.array([[0.1] * self.dim for _ in batch], dtype=np.float32)


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A Store with no real backends - each test wires the ones it needs."""
    s = server.Store.__new__(server.Store)
    s.db = sqlite3.connect(":memory:")
    s.db.row_factory = sqlite3.Row
    from base_schema import apply_full_schema

    apply_full_schema(s.db)
    s.v9_cache = None
    s._embedder = None       # `embedder` is a lazy property; back it directly
    s._embed_mode = "fastembed"
    s._binary_search_ready = None
    s._chroma_client = None
    return s


def _seed(store, kid, model, dim, provider):
    store.db.execute(
        "INSERT INTO knowledge (id, session_id, type, content, project, created_at) "
        "VALUES (?, 's1', 'fact', ?, 'p', '2026-08-30T00:00:00Z')",
        (kid, f"content {kid}"),
    )
    store._upsert_embedding(kid, [0.1] * dim, model, provider=provider)


def _rebuild(store, **args):
    """Drive the real tool handler, not a copy of its query."""
    import asyncio

    server.store = store
    # _do returns a JSON string (server.J), not an MCP content object.
    return json.loads(asyncio.run(server._do("memory_rebuild_embeddings", args)))


class _L2:
    enabled = True


# ──────────────────────────────────────────────
# memory_rebuild_embeddings(stale_only=True)
# ──────────────────────────────────────────────

def test_stale_only_rebuilds_rows_from_another_backend(store, monkeypatch):
    """`embedding_space` cannot express this: drifted rows are all `text` too."""
    store._embed_mode = "fastembed"
    monkeypatch.setattr(store, "_fastembed_embed", lambda batch: [[0.7] * 384 for _ in batch])
    _seed(store, 1, server.FASTEMBED_MODEL, 384, "fastembed")   # current
    _seed(store, 2, "nomic-embed-text", 768, "ollama")          # drifted
    _seed(store, 3, "nomic-embed-text", 384, "ollama")          # drifted + mislabelled
    store.db.commit()

    result = _rebuild(store, stale_only=True)

    assert result["rebuilt"] == 2, result
    models = dict(store.db.execute("SELECT knowledge_id, embed_model FROM embeddings").fetchall())
    assert models[2] == server.FASTEMBED_MODEL
    assert models[3] == server.FASTEMBED_MODEL


def test_without_stale_only_every_row_is_rebuilt(store, monkeypatch):
    """The flag must narrow the job, and its absence must not."""
    store._embed_mode = "fastembed"
    monkeypatch.setattr(store, "_fastembed_embed", lambda batch: [[0.7] * 384 for _ in batch])
    _seed(store, 1, server.FASTEMBED_MODEL, 384, "fastembed")
    _seed(store, 2, "nomic-embed-text", 768, "ollama")
    store.db.commit()

    assert _rebuild(store)["rebuilt"] == 2


def test_a_record_with_no_embedding_is_not_called_stale(store, monkeypatch):
    """Absent is a different job from stale, and a silent one to take on."""
    store._embed_mode = "fastembed"
    monkeypatch.setattr(store, "_fastembed_embed", lambda batch: [[0.7] * 384 for _ in batch])
    store.db.execute(
        "INSERT INTO knowledge (id, session_id, type, content, project, created_at) "
        "VALUES (9, 's1', 'fact', 'never embedded', 'p', '2026-08-30T00:00:00Z')"
    )
    store.db.commit()

    assert _rebuild(store, stale_only=True)["rebuilt"] == 0


def test_the_tool_exposes_stale_only():
    import asyncio

    tools = asyncio.run(server.list_tools())
    tool = {t.name: t for t in tools}["memory_rebuild_embeddings"]
    schema = getattr(tool, "input_schema", None) or tool.inputSchema
    assert "stale_only" in schema["properties"]


# ──────────────────────────────────────────────
# rebuild aborts when a fallback answers
# ──────────────────────────────────────────────

def test_a_diverging_rebuild_is_skipped_not_rewritten_under_a_fallback_label(store, monkeypatch):
    """`stale_only` compares a row's stored label to the CONFIGURED model;
    the rebuild writes the label of whoever actually answered. When Ollama
    is configured but unreachable, ST answers instead - those never agree,
    so writing the fallback vector anyway would both mislabel the row and
    leave it stale again next run. The tool must abort and skip instead of
    reporting a fallback write as a successful rebuild."""
    store._embed_mode = "ollama"
    store._embedder = _Embedder(384)
    monkeypatch.setattr(store, "_ollama_embed", lambda batch: None)
    _seed(store, 1, "some-old-model", 384, "st")
    store.db.commit()

    result = _rebuild(store, stale_only=True)
    assert result["rebuilt"] == 0
    assert result["skipped"] == 1
    assert "error" in result
    assert "nothing was rewritten" in result["error"]

    row = store.db.execute(
        "SELECT embed_model, embed_dim, embedding_provider FROM embeddings WHERE knowledge_id=1"
    ).fetchone()
    assert row["embed_model"] == "some-old-model"
    assert row["embed_dim"] == 384
    assert row["embedding_provider"] == "st"

    # A second run hits the same mismatch, not a second silent overwrite.
    result2 = _rebuild(store, stale_only=True)
    assert result2["rebuilt"] == 0
    assert result2["skipped"] == 1


def test_a_diverging_full_rebuild_aborts_too_not_only_stale_only(store, monkeypatch):
    """The abort is not gated on `stale_only`: a full rebuild with the
    configured backend down would otherwise rewrite every row with a
    fallback vector."""
    store._embed_mode = "ollama"
    store._embedder = _Embedder(384)
    monkeypatch.setattr(store, "_ollama_embed", lambda batch: None)
    _seed(store, 1, "nomic-embed-text", 768, "ollama")
    store.db.commit()

    result = _rebuild(store, stale_only=False)

    assert result["rebuilt"] == 0
    assert result["skipped"] == 1
    assert "error" in result

    row = store.db.execute(
        "SELECT embed_model, embed_dim FROM embeddings WHERE knowledge_id=1"
    ).fetchone()
    assert row["embed_model"] == "nomic-embed-text"
    assert row["embed_dim"] == 768


def test_a_mid_run_divergence_reports_rebuilt_and_skipped_separately(store, monkeypatch):
    """With batch_size=1 the backend answers correctly for the first row
    (rebuilt=1, committed) and only then diverges - `rebuilt` must survive
    the abort instead of being folded into `skipped`."""
    store._embed_mode = "fastembed"
    calls = {"n": 0}

    def _fastembed(batch):
        calls["n"] += 1
        if calls["n"] == 1:
            return [[0.7] * 384 for _ in batch]
        return None  # unavailable from the second call on

    monkeypatch.setattr(store, "_fastembed_embed", _fastembed)
    monkeypatch.setattr(store, "_check_ollama", lambda: False)
    monkeypatch.setenv("MEMORY_ALLOW_OLLAMA_IN_HOT_PATH", "true")
    store._embedder = _Embedder(384)

    _seed(store, 1, "nomic-embed-text", 768, "ollama")
    _seed(store, 2, "nomic-embed-text", 768, "ollama")
    _seed(store, 3, "nomic-embed-text", 768, "ollama")
    store.db.commit()

    result = _rebuild(store, stale_only=True, batch_size=1)

    assert result["rebuilt"] == 1
    assert result["skipped"] == 2
    assert "error" in result
    assert "nothing was rewritten" not in result["error"]
    assert "1 row(s) already rebuilt" in result["error"]

    models = dict(store.db.execute("SELECT knowledge_id, embed_model FROM embeddings").fetchall())
    assert models[1] == server.FASTEMBED_MODEL  # row 1 really was rebuilt...
    assert models[2] == "nomic-embed-text"       # ...rows 2 and 3 were left untouched
    assert models[3] == "nomic-embed-text"


# ──────────────────────────────────────────────
# one model per embedded batch
# ──────────────────────────────────────────────

def test_a_batch_with_a_partial_cache_hit_does_not_mix_backends(store, monkeypatch):
    """L2 holds 'alpha' under the active model; the rest of the batch falls
    back to a different backend. Every vector in the batch must come back
    under the one identity the call returns, and be re-cached under it."""
    store._embed_mode = "ollama"
    store._embedder = _Embedder(384)
    monkeypatch.setattr(store, "_ollama_embed", lambda batch: None)
    active = store._active_embed_model_name()

    class _PartialCache:
        l2 = _L2()

        def __init__(self):
            self.sets: list[tuple[str, int, str]] = []

        def embed_get(self, text, expected_model=None):
            return [0.5] * 768 if text == "alpha" else None

        def embed_set(self, text, vector, model):
            self.sets.append((text, len(vector), model))

    store.v9_cache = _PartialCache()

    vectors, identity = store.embed_identified(["alpha", "beta"])

    assert len(vectors[0]) == 384
    assert len(vectors[1]) == 384
    assert (identity.model, identity.provider) == (server.EMBEDDING_MODEL, "st")
    assert identity.model != active
    assert ("alpha", 384, server.EMBEDDING_MODEL) in store.v9_cache.sets
    assert ("beta", 384, server.EMBEDDING_MODEL) in store.v9_cache.sets


def test_a_flapping_backend_does_not_mix_backends_in_one_batch(store, monkeypatch):
    """Ollama is down for the fresh half and back by the redo of the whole
    batch: whichever backend answers the redo answers all of it, under its
    own identity."""
    store._embed_mode = "ollama"
    store._embedder = _Embedder(384)
    calls = {"n": 0}

    def _flapping_ollama(batch):
        calls["n"] += 1
        if calls["n"] == 1:
            return None  # down for the fresh half
        return [[0.9] * 768 for _ in batch]  # recovered for the redo

    monkeypatch.setattr(store, "_ollama_embed", _flapping_ollama)

    class _PartialCache:
        l2 = _L2()

        def embed_get(self, text, expected_model=None):
            return [0.5] * 768 if text == "alpha" else None

        def embed_set(self, text, vector, model):
            pass

    store.v9_cache = _PartialCache()

    vectors, identity = store.embed_identified(["alpha", "beta"])

    assert (identity.model, identity.provider) == (server.OLLAMA_EMBED_MODEL, "ollama")
    assert vectors == [[0.9] * 768, [0.9] * 768]
