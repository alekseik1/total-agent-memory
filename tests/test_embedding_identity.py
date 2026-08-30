"""A stored embedding is labelled with the backend that produced it.

`Store.embed` falls through FastEmbed → Ollama → SentenceTransformers and
returns only vectors, so callers read the model name off configuration. With
Ollama configured but unreachable that wrote rows saying `nomic-embed-text`
(768-dim) holding 384-dim SentenceTransformer vectors — 30 of them on this
machine — and cached those vectors under the same wrong name, so the cache's
`expected_model` guard matched and served them back as if they belonged.
"""

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
    """A Store with no real backends — each test wires the ones it needs."""
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


def test_a_fallback_is_recorded_as_itself_not_as_the_configured_model(store, monkeypatch):
    """The exact live failure: Ollama configured, tunnel down, ST answers."""
    store._embed_mode = "ollama"
    store._embedder = _Embedder(384)
    monkeypatch.setattr(store, "_ollama_embed", lambda batch: None)

    vectors, (model, provider) = store.embed_with_identity(["hello"])

    assert len(vectors[0]) == 384
    assert model == server.EMBEDDING_MODEL
    assert provider == "st"
    assert model != server.OLLAMA_EMBED_MODEL


def test_the_working_backend_is_reported_as_itself(store, monkeypatch):
    store._embed_mode = "ollama"
    monkeypatch.setattr(store, "_ollama_embed", lambda batch: [[0.5] * 768 for _ in batch])

    _vectors, (model, provider) = store.embed_with_identity(["hello"])

    assert model == server.OLLAMA_EMBED_MODEL
    assert provider == "ollama"


def test_fastembed_reports_its_own_model(store, monkeypatch):
    monkeypatch.setattr(store, "_fastembed_embed", lambda batch: [[0.2] * 384 for _ in batch])

    _vectors, (model, provider) = store.embed_with_identity(["hello"])

    assert model == server.FASTEMBED_MODEL
    assert provider == "fastembed"


def test_embed_keeps_its_old_shape_for_existing_callers(store, monkeypatch):
    """A dozen call sites still expect a bare list of vectors."""
    monkeypatch.setattr(store, "_fastembed_embed", lambda batch: [[0.2] * 384 for _ in batch])

    out = store.embed(["hello"])

    assert isinstance(out, list) and len(out[0]) == 384


def test_a_row_saved_through_the_fallback_says_what_made_it(store, monkeypatch):
    """End to end: the row in `embeddings`, not just the return value."""
    store._embed_mode = "ollama"
    store._embedder = _Embedder(384)
    monkeypatch.setattr(store, "_ollama_embed", lambda batch: None)

    vectors, (model, provider) = store.embed_with_identity(["some knowledge"])
    store.db.execute(
        "INSERT INTO knowledge (id, session_id, type, content, project, created_at) "
        "VALUES (1, 's1', 'fact', 'some knowledge', 'p', '2026-08-30T00:00:00Z')"
    )
    store._upsert_embedding(1, vectors[0], model, provider=provider)
    store.db.commit()

    row = store.db.execute(
        "SELECT embed_model, embed_dim, embedding_provider, length(float32_vector)/4 AS real_dim "
        "FROM embeddings WHERE knowledge_id=1"
    ).fetchone()
    assert row["embed_dim"] == row["real_dim"] == 384
    assert row["embed_model"] == server.EMBEDDING_MODEL
    assert row["embedding_provider"] == "st"


def test_nothing_available_returns_no_vectors_but_still_an_identity(store, monkeypatch):
    store._embed_mode = "ollama"
    monkeypatch.setattr(store, "_ollama_embed", lambda batch: None)
    # `embedder` is lazy: left alone it would load a real SentenceTransformer
    # and this test would measure the machine instead of the code.
    monkeypatch.setattr(server, "HAS_ST", False)
    store._embedder = None

    vectors, identity = store.embed_with_identity(["hello"])

    assert vectors is None
    assert isinstance(identity, tuple) and len(identity) == 2


# ──────────────────────────────────────────────
# memory_rebuild_embeddings(stale_only=True)
# ──────────────────────────────────────────────

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


def test_the_cache_stores_the_model_that_made_the_vector(store, monkeypatch):
    """`embed_get` guards on the stored name, so a wrong one serves a vector
    from another space as if it belonged to this one."""
    recorded = []

    class _L2:
        enabled = True

    class _Cache:
        l2 = _L2()

        def embed_get(self, text, expected_model=None):
            return None

        def embed_set(self, text, vector, model):
            recorded.append(model)

    store.v9_cache = _Cache()
    store._embed_mode = "ollama"
    store._embedder = _Embedder(384)
    monkeypatch.setattr(store, "_ollama_embed", lambda batch: None)

    store.embed_with_identity(["hello"])

    assert recorded == [server.EMBEDDING_MODEL]
    assert server.OLLAMA_EMBED_MODEL not in recorded


def test_the_tool_exposes_stale_only():
    import asyncio

    tools = asyncio.run(server.list_tools())
    tool = {t.name: t for t in tools}["memory_rebuild_embeddings"]
    schema = getattr(tool, "input_schema", None) or tool.inputSchema
    assert "stale_only" in schema["properties"]


def test_a_rebuilt_row_is_labelled_by_the_backend_that_rebuilt_it(store, monkeypatch):
    """The rebuild path had the same defect as the save path: it read the model
    name off configuration after embedding, so a rebuild run while Ollama was
    unreachable relabelled rows with a model that never saw them."""
    store._embed_mode = "ollama"
    store._embedder = _Embedder(384)
    monkeypatch.setattr(store, "_ollama_embed", lambda batch: None)
    _seed(store, 1, "some-old-model", 384, "st")
    store.db.commit()

    assert _rebuild(store, stale_only=True)["rebuilt"] == 1

    row = store.db.execute(
        "SELECT embed_model, embedding_provider FROM embeddings WHERE knowledge_id=1"
    ).fetchone()
    assert row["embed_model"] == server.EMBEDDING_MODEL
    assert row["embedding_provider"] == "st"
