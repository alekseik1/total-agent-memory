"""ChatGPT and claude.ai data-export readers (``conversations.json``).

ChatGPT: an array of conversations, each with a ``mapping`` of message nodes
forming a tree (edits and regenerations are branches). The visible thread is
the path from ``current_node`` up to the root; messages carry
``author.role`` and ``content.parts``, times are epoch seconds.

claude.ai: an array of conversations with a flat ``chat_messages`` list;
``sender`` is ``human`` or ``assistant``, text lives in ``content`` blocks
(``type: text``) or the legacy ``text`` field, times are ISO-8601.

User turns only by default; ``include_assistant`` adds the assistant side.
System, tool and hidden messages are never imported.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from importers.json_stream import iter_json_array
from importers.records import ImportFormatError, ImportRecord, ImportStats, parse_timestamp

CHATGPT_TEXT_CONTENT_TYPES = frozenset({"text", "multimodal_text"})
IMPORTABLE_ROLES = ("user", "assistant")
CONVERSATION_SOURCE_FORMAT = "conversation"


@dataclass(frozen=True, slots=True)
class ChatOptions:
    include_assistant: bool = False

    def accepts(self, role: str) -> bool:
        return role == "user" or (self.include_assistant and role == "assistant")


def _title(raw: object) -> str:
    return raw.strip() if isinstance(raw, str) and raw.strip() else "untitled"


def _context(source: str, title: str, started: datetime | None, role: str) -> str:
    day = f" ({started.date().isoformat()})" if started is not None else ""
    return f'{source} conversation "{title}"{day}, {role} message'


def _require_object(item: object, index: int, source: str, required: str) -> Mapping[str, Any]:
    if not isinstance(item, Mapping) or required not in item:
        found = sorted(item)[:8] if isinstance(item, Mapping) else type(item).__name__
        raise ImportFormatError(
            f"conversation #{index} is not a {source} export conversation (no {required!r}; found {found})"
        )
    return item


# ─── ChatGPT ────────────────────────────────────────────────────────────────

def _chatgpt_thread(conversation: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Messages on the visible branch, oldest first."""
    mapping = conversation.get("mapping")
    if not isinstance(mapping, Mapping):
        raise ImportFormatError("ChatGPT conversation has no 'mapping' object")
    nodes: list[Mapping[str, Any]] = []
    node_id = conversation.get("current_node")
    if isinstance(node_id, str) and node_id in mapping:
        visited: set[str] = set()
        while isinstance(node_id, str) and node_id in mapping and node_id not in visited:
            visited.add(node_id)
            node = mapping[node_id]
            if not isinstance(node, Mapping):
                break
            nodes.append(node)
            node_id = node.get("parent")
        nodes.reverse()
    else:
        candidates = [node for node in mapping.values() if isinstance(node, Mapping)]
        nodes = sorted(candidates, key=lambda node: _message_time(node.get("message")) or 0.0)
    return [node["message"] for node in nodes if isinstance(node.get("message"), Mapping)]


def _message_time(message: object) -> float | None:
    if isinstance(message, Mapping):
        value = message.get("create_time")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def _chatgpt_text(message: Mapping[str, Any]) -> str | None:
    """Joined text parts, or None when the message is not plain text (code runs, browsing, images only)."""
    content = message.get("content")
    if not isinstance(content, Mapping) or content.get("content_type") not in CHATGPT_TEXT_CONTENT_TYPES:
        return None
    parts = content.get("parts")
    if not isinstance(parts, list):
        return None
    return "\n".join(part for part in parts if isinstance(part, str)).strip()


def _chatgpt_hidden(message: Mapping[str, Any]) -> bool:
    metadata = message.get("metadata")
    return isinstance(metadata, Mapping) and bool(metadata.get("is_visually_hidden_from_conversation"))


def chatgpt_records(path: Path, options: ChatOptions, stats: ImportStats) -> Iterator[ImportRecord]:
    for index, item in enumerate(iter_json_array(path)):
        conversation = _require_object(item, index, "ChatGPT", "mapping")
        conversation_id = str(conversation.get("conversation_id") or conversation.get("id") or index)
        title = _title(conversation.get("title"))
        started = parse_timestamp(conversation.get("create_time"))
        for position, message in enumerate(_chatgpt_thread(conversation)):
            author = message.get("author")
            role = author.get("role") if isinstance(author, Mapping) else None
            if role not in IMPORTABLE_ROLES or _chatgpt_hidden(message):
                continue
            stats.units_seen += 1
            if not options.accepts(role):
                stats.skipped_role += 1
                continue
            text = _chatgpt_text(message)
            if not text:
                stats.skipped_empty += 1
                continue
            occurred = parse_timestamp(message.get("create_time")) or started
            message_id = str(message.get("id") or position)
            yield ImportRecord(
                text=text, occurred_at=occurred, source_ref=f"chatgpt:{conversation_id}:{message_id}",
                context=_context("ChatGPT", title, started, role),
                tags=("chatgpt", f"role-{role}"), source_format=CONVERSATION_SOURCE_FORMAT,
            )


# ─── claude.ai ──────────────────────────────────────────────────────────────

_CLAUDE_ROLES = {"human": "user", "user": "user", "assistant": "assistant"}


def _claude_text(message: Mapping[str, Any]) -> str:
    blocks = message.get("content")
    if isinstance(blocks, list):
        texts = [block["text"] for block in blocks
                 if isinstance(block, Mapping) and block.get("type") == "text" and isinstance(block.get("text"), str)]
        joined = "\n\n".join(text.strip() for text in texts if text.strip())
        if joined:
            return joined
    legacy = message.get("text")
    return legacy.strip() if isinstance(legacy, str) else ""


def claude_records(path: Path, options: ChatOptions, stats: ImportStats) -> Iterator[ImportRecord]:
    for index, item in enumerate(iter_json_array(path)):
        conversation = _require_object(item, index, "claude.ai", "chat_messages")
        messages = conversation["chat_messages"]
        if not isinstance(messages, list):
            raise ImportFormatError(f"conversation #{index}: 'chat_messages' is not a list")
        conversation_id = str(conversation.get("uuid") or index)
        title = _title(conversation.get("name"))
        started = parse_timestamp(conversation.get("created_at"))
        for position, message in enumerate(messages):
            if not isinstance(message, Mapping):
                raise ImportFormatError(f"conversation #{index}, message #{position} is not an object")
            role = _CLAUDE_ROLES.get(str(message.get("sender", "")))
            if role is None:
                continue
            stats.units_seen += 1
            if not options.accepts(role):
                stats.skipped_role += 1
                continue
            text = _claude_text(message)
            if not text:
                stats.skipped_empty += 1
                continue
            occurred = parse_timestamp(message.get("created_at")) or started
            message_id = str(message.get("uuid") or position)
            yield ImportRecord(
                text=text, occurred_at=occurred, source_ref=f"claude:{conversation_id}:{message_id}",
                context=_context("Claude", title, started, role),
                tags=("claude-ai", f"role-{role}"), source_format=CONVERSATION_SOURCE_FORMAT,
            )
