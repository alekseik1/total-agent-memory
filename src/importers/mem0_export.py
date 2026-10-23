"""mem0 export reader.

Accepted shapes, validated strictly:

* the ``get_all`` / ``search`` result of the open-source ``Memory`` class and
  of ``MemoryClient`` (API v2): ``{"results": [<memory>, ...]}`` — the
  ``"memories"`` key is accepted as an alias;
* the legacy bare list ``[<memory>, ...]`` (OSS output format v1.0).

Each memory is an object with a non-empty string ``memory``; optional fields
used: ``id``, ``created_at`` / ``updated_at`` (ISO-8601, must parse when
present), ``categories`` (list of strings, become tags), ``user_id``,
``agent_id``, ``run_id`` (provenance in the context line). Platform exports
built from a custom ``create_memory_export`` schema have a user-defined
shape; convert them to this list first.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from importers.json_stream import iter_json_array
from importers.records import ImportFormatError, ImportRecord, ImportStats, normalise_tag, parse_timestamp

CONTAINER_KEYS = ("results", "memories")
PROVENANCE_FIELDS = ("user_id", "agent_id", "run_id")


def _memory(item: object, index: int) -> Mapping[str, Any]:
    if not isinstance(item, Mapping):
        raise ImportFormatError(f"mem0 memory #{index} is a {type(item).__name__}, expected an object")
    text = item.get("memory")
    if not isinstance(text, str):
        raise ImportFormatError(
            f"mem0 memory #{index} has no string 'memory' field (keys: {sorted(item)[:10]}); "
            "expected the get_all() result shape {\"results\": [{\"id\", \"memory\", \"created_at\", ...}]}"
        )
    categories = item.get("categories")
    if categories is not None and not (isinstance(categories, list)
                                       and all(isinstance(value, str) for value in categories)):
        raise ImportFormatError(f"mem0 memory #{index}: 'categories' must be a list of strings")
    return item


def mem0_records(path: Path, stats: ImportStats) -> Iterator[ImportRecord]:
    for index, item in enumerate(iter_json_array(path, CONTAINER_KEYS)):
        memory = _memory(item, index)
        stats.units_seen += 1
        text = memory["memory"].strip()
        if not text:
            stats.skipped_empty += 1
            continue
        occurred = (parse_timestamp(memory.get("created_at"), strict=True)
                    or parse_timestamp(memory.get("updated_at"), strict=True))
        provenance = ", ".join(f"{name}={memory[name]}" for name in PROVENANCE_FIELDS
                               if isinstance(memory.get(name), str) and memory[name])
        categories = tuple(tag for tag in (normalise_tag(value) for value in memory.get("categories") or [])
                           if tag)
        yield ImportRecord(
            text=text, occurred_at=occurred, source_ref=f"mem0:{memory.get('id') or index}",
            context="Imported mem0 memory" + (f" ({provenance})" if provenance else ""),
            tags=("mem0", *categories),
        )
