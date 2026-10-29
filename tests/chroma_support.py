"""Stop the Chroma systems a test started.

chromadb caches one System per persist directory and never stops it while a
client is not closed. A Store built on a fresh tmp MEMORY_DIR therefore leaves
a Rust runtime behind (about 15 threads and 6 descriptors). Hundreds of them
pile up over a suite run, and on the 3-core macOS runners chromadb then hung
inside its Rust start, holding the GIL, until the job timeout.
"""
import sys

SHARED_SYSTEM_MODULE = "chromadb.api.shared_system_client"


def _shared_system_client():
    """The chromadb class holding the System cache, or None when chromadb was never imported."""
    module = sys.modules.get(SHARED_SYSTEM_MODULE)
    return getattr(module, "SharedSystemClient", None)


def running_systems() -> frozenset[str]:
    """Identifiers (persist directories) of the Chroma systems currently cached."""
    client = _shared_system_client()
    if client is None:
        return frozenset()
    return frozenset(client._identifier_to_system)


def stop_systems_started_since(before: frozenset[str]) -> list[str]:
    """Stop and forget every cached Chroma system not in ``before``; returns their identifiers."""
    client = _shared_system_client()
    if client is None:
        return []
    refcounts = getattr(client, "_identifier_to_refcount", {})
    stopped = []
    for identifier in sorted(set(client._identifier_to_system) - before):
        system = client._identifier_to_system.pop(identifier)
        refcounts.pop(identifier, None)
        system.stop()
        stopped.append(identifier)
    return stopped
