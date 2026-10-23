"""Open the local SQLite store for a one-shot CLI command (`tam import`, `tam import-bundle`)."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

MEMORY_DIR_ENV = "TAM_MEMORY_DIR"


def resolve_memory_dir(explicit: Path | None) -> Path:
    """``--memory-dir`` when given, else TAM_MEMORY_DIR / the default location."""
    if explicit is not None:
        return explicit.expanduser().resolve()
    from paths import memory_dir

    return memory_dir()


def pin_memory_dir(explicit: Path | None) -> Path:
    """Resolve the memory directory and export it before ``server`` is imported, so every module agrees."""
    root = resolve_memory_dir(explicit)
    os.environ[MEMORY_DIR_ENV] = str(root)
    return root


@contextmanager
def open_store(memory_dir: Path) -> Iterator[Any]:
    """A ``server.Store`` on ``memory_dir`` (migrations applied), closed and unpinned afterwards."""
    import server

    previous = server.MEMORY_DIR
    server.MEMORY_DIR = memory_dir
    store = server.Store()
    try:
        yield store
    finally:
        worker = getattr(store, "_enrich_worker", None)
        if worker is not None:
            worker.stop()
        store.db.close()
        # chromadb keeps chroma.sqlite3 open until the client is closed; Windows refuses to move an open file
        close_chroma = getattr(store._chroma_client, "close", None)
        if close_chroma is not None:
            close_chroma()
        server.MEMORY_DIR = previous
