"""`tam import`: ChatGPT, claude.ai, mem0 and Markdown importers through the normal save path."""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from importers import (
    ChatOptions,
    ImportFormatError,
    ImportStats,
    json_stream,
)
from importers import cli as import_cli
from importers.chat_exports import chatgpt_records, claude_records
from importers.markdown_tree import markdown_records, split_frontmatter, split_sections
from importers.mem0_export import mem0_records
from importers.records import content_hash, parse_timestamp, split_text

T0 = 1714557600.0  # 2024-05-01T10:00:00Z
API_KEY = "sk-proj-" + "Zx9Yw8Vu7Ts6Rq5Po4Nm3Lk2Ji1Hg0Fe"


def _node(node_id, parent, children, role=None, text=None, created=None, content_type="text", hidden=False):
    message = None
    if role is not None:
        message = {
            "id": f"msg-{node_id}", "author": {"role": role, "name": None, "metadata": {}},
            "create_time": created, "update_time": None,
            "content": {"content_type": content_type, "parts": text if isinstance(text, list) else [text]},
            "status": "finished_successfully", "end_turn": True, "weight": 1.0,
            "metadata": {"is_visually_hidden_from_conversation": True} if hidden else {},
            "recipient": "all",
        }
    return {"id": node_id, "message": message, "parent": parent, "children": children}


CHATGPT_EXPORT = [
    {
        "title": "Postgres tuning",
        "create_time": T0,
        "update_time": T0 + 600,
        "conversation_id": "conv-pg",
        "current_node": "a2",
        "mapping": {
            "root": _node("root", None, ["sys"]),
            "sys": _node("sys", "root", ["hid"], role="system", text="You are ChatGPT.", created=None),
            "hid": _node("hid", "sys", ["u1-old", "u1"], role="user", text="custom instructions: be terse",
                         created=T0, hidden=True),
            "u1-old": _node("u1-old", "hid", ["a1-old"], role="user",
                            text="Original question that was later edited away", created=T0 + 10),
            "a1-old": _node("a1-old", "u1-old", [], role="assistant", text="Old answer.", created=T0 + 20),
            "u1": _node("u1", "hid", ["a1"], role="user",
                        text="Our Postgres 16 primary uses shared_buffers=4GB on a 32GB box; should it go higher?",
                        created=T0 + 30),
            "a1": _node("a1", "u1", ["u2"], role="assistant",
                        text="Usually 25% of RAM is a sound starting point, so 8GB.", created=T0 + 40),
            "u2": _node("u2", "a1", ["t1"], role="user",
                        text=["Here is the explain plan for the slow report query.", {"asset_pointer": "file-1"}],
                        created=T0 + 50, content_type="multimodal_text"),
            "t1": _node("t1", "u2", ["a2"], role="tool", text="python output", created=T0 + 55),
            "a2": _node("a2", "t1", [], role="assistant", text="The seq scan on orders dominates.",
                        created=T0 + 60),
        },
    },
    {
        "title": "Short chat",
        "create_time": T0 + 86400,
        "id": "conv-short",
        "current_node": "u1",
        "mapping": {
            "root": _node("root", None, ["u1"]),
            "u1": _node("u1", "root", [], role="user", text="thanks!", created=T0 + 86400),
        },
    },
]

CLAUDE_EXPORT = [
    {
        "uuid": "c-1", "name": "Billing migration plan",
        "created_at": "2024-06-02T08:00:00.000000Z", "updated_at": "2024-06-02T09:00:00.000000Z",
        "account": {"uuid": "acc-1"},
        "chat_messages": [
            {"uuid": "m-1", "sender": "human", "created_at": "2024-06-02T08:00:05.123456Z",
             "text": "", "content": [{"type": "text", "text": "We bill through Stripe; invoices are generated nightly "
                                                              "by the billing-cron service at 02:00 UTC."}],
             "attachments": [], "files": []},
            {"uuid": "m-2", "sender": "assistant", "created_at": "2024-06-02T08:00:30Z",
             "text": "", "content": [{"type": "text", "text": "Then the migration window should avoid 02:00."},
                                     {"type": "tool_use", "name": "web_search", "input": {}}],
             "attachments": [], "files": []},
            {"uuid": "m-3", "sender": "human", "created_at": "2024-06-02T08:05:00Z",
             "text": f"The Stripe secret is STRIPE_API_KEY={API_KEY} for the staging account.",
             "attachments": [], "files": []},
        ],
    },
]

MEM0_EXPORT = {
    "results": [
        {"id": "mem-1", "memory": "Prefers PostgreSQL over MySQL for new services", "hash": "h1",
         "metadata": None, "categories": ["technology", "preferences"],
         "created_at": "2024-03-10T12:00:00-07:00", "updated_at": "2024-03-11T12:00:00-07:00", "user_id": "alice"},
        {"id": "mem-2", "memory": "Works in the Europe/Berlin timezone", "hash": "h2",
         "created_at": "2024-03-12T09:30:00Z", "updated_at": None, "user_id": "alice", "agent_id": "planner"},
    ],
}


@pytest.fixture
def exports(tmp_path):
    folder = tmp_path / "exports"
    folder.mkdir()
    (folder / "chatgpt.json").write_text(json.dumps(CHATGPT_EXPORT, indent=1), encoding="utf-8")
    (folder / "claude.json").write_text(json.dumps(CLAUDE_EXPORT), encoding="utf-8")
    (folder / "mem0.json").write_text(json.dumps(MEM0_EXPORT), encoding="utf-8")
    return folder


@pytest.fixture
def vault(tmp_path, make_symlink):
    root = tmp_path / "vault"
    (root / "projects").mkdir(parents=True)
    (root / ".obsidian").mkdir()
    (root / ".obsidian" / "workspace.md").write_text("# internal state that must not be imported\n", encoding="utf-8")
    (root / "projects" / "deploy.md").write_text(
        "---\ntitle: Deploy runbook\ntags: [ops, Deploy Notes]\ncreated: 2023-11-05\n---\n"
        "Deploys go out with `make release` from the main branch after CI is green.\n", encoding="utf-8")
    sections = "\n\n".join(
        f"## Section {index}\n\n" + (f"Paragraph about component {index}. " * 40) for index in range(1, 5))
    (root / "architecture.md").write_text(
        "---\ntags:\n  - architecture\n  - backend\n---\n# Architecture\n\nIntro text before sections.\n\n"
        "```bash\n# not a heading inside code\necho hi\n```\n\n" + sections, encoding="utf-8")
    (root / "empty.md").write_text("---\ntags: x\n---\n", encoding="utf-8")
    (root / "notes.txt").write_text("not markdown", encoding="utf-8")
    outside = tmp_path / "outside.md"
    outside.write_text("Outside the vault and reached through a symlink.", encoding="utf-8")
    make_symlink(root / "linked.md", outside)
    stamp = datetime(2022, 1, 2, 3, 4, 5, tzinfo=UTC).timestamp()
    os.utime(root / "architecture.md", (stamp, stamp))
    return root


def _stats(kind="chatgpt"):
    return ImportStats(kind=kind, project="p", dry_run=False)


# ─── parsers ────────────────────────────────────────────────────────────────

def test_chatgpt_follows_the_visible_branch_and_keeps_user_turns(exports):
    stats = _stats()
    records = list(chatgpt_records(exports / "chatgpt.json", ChatOptions(), stats))
    assert [record.text for record in records] == [
        "Our Postgres 16 primary uses shared_buffers=4GB on a 32GB box; should it go higher?",
        "Here is the explain plan for the slow report query.",
        "thanks!",
    ]
    assert records[0].occurred_at == datetime.fromtimestamp(T0 + 30, UTC)
    assert records[0].source_ref == "chatgpt:conv-pg:msg-u1"
    assert records[0].source_format == "conversation"
    assert 'ChatGPT conversation "Postgres tuning" (2024-05-01), user message' == records[0].context
    assert stats.skipped_role == 2  # the two assistant replies on the visible branch


def test_chatgpt_include_assistant(exports):
    records = list(chatgpt_records(exports / "chatgpt.json", ChatOptions(include_assistant=True), _stats()))
    assert "The seq scan on orders dominates." in [record.text for record in records]
    assert "Old answer." not in [record.text for record in records]
    assert all("python output" != record.text for record in records)


def test_claude_reads_content_blocks_and_legacy_text(exports):
    stats = _stats("claude")
    records = list(claude_records(exports / "claude.json", ChatOptions(), stats))
    assert len(records) == 2
    assert records[0].text.startswith("We bill through Stripe")
    assert records[0].occurred_at == datetime(2024, 6, 2, 8, 0, 5, 123456, tzinfo=UTC)
    assert records[1].text.startswith("The Stripe secret is")
    assert stats.skipped_role == 1
    with_assistant = list(claude_records(exports / "claude.json", ChatOptions(include_assistant=True), _stats()))
    assert with_assistant[1].text == "Then the migration window should avoid 02:00."


def test_wrong_export_kind_is_a_clear_error(exports):
    with pytest.raises(ImportFormatError, match="claude.ai export"):
        list(claude_records(exports / "chatgpt.json", ChatOptions(), _stats()))
    with pytest.raises(ImportFormatError, match="ChatGPT export"):
        list(chatgpt_records(exports / "claude.json", ChatOptions(), _stats()))


def test_mem0_results_shape_and_bare_list(exports, tmp_path):
    records = list(mem0_records(exports / "mem0.json", _stats("mem0")))
    assert [record.text for record in records] == ["Prefers PostgreSQL over MySQL for new services",
                                                   "Works in the Europe/Berlin timezone"]
    assert records[0].occurred_at == datetime(2024, 3, 10, 19, 0, tzinfo=UTC)
    assert records[0].tags == ("mem0", "technology", "preferences")
    assert "user_id=alice" in records[0].context and "agent_id=planner" in records[1].context
    bare = tmp_path / "bare.json"
    bare.write_text(json.dumps(MEM0_EXPORT["results"]), encoding="utf-8")
    assert len(list(mem0_records(bare, _stats("mem0")))) == 2


@pytest.mark.parametrize("payload, message", [
    ({"memories_list": []}, "no results/memories array"),
    ([{"text": "no memory field"}], "no string 'memory' field"),
    ([{"memory": "x", "created_at": "yesterday-ish"}], "invalid timestamp"),
    ([{"memory": "x", "categories": "tech"}], "categories"),
    ("just a string", "expected an array"),
])
def test_mem0_is_validated_strictly(tmp_path, payload, message):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ImportFormatError, match=message):
        list(mem0_records(path, _stats("mem0")))


def test_markdown_tree_notes_sections_frontmatter_and_skips(vault):
    stats = _stats("markdown")
    records = list(markdown_records(vault, stats))
    refs = [record.source_ref for record in records]
    assert "markdown:projects/deploy.md" in refs
    assert not any(".obsidian" in ref or "linked" in ref for ref in refs)
    deploy = next(record for record in records if record.source_ref == "markdown:projects/deploy.md")
    assert deploy.tags == ("markdown", "ops", "deploy-notes")
    assert deploy.occurred_at == datetime(2023, 11, 5, tzinfo=UTC)
    assert deploy.context == 'Markdown note "Deploy runbook" (projects/deploy.md)'
    sections = [record for record in records if record.source_ref.startswith("markdown:architecture.md#")]
    assert len(sections) == 5  # intro under "# Architecture" + four "##" sections
    assert sections[0].text.startswith("# Architecture") and "not a heading inside code" in sections[0].text
    assert sections[2].context.endswith("section Architecture > Section 2")
    assert sections[0].tags == ("markdown", "architecture", "backend")
    assert sections[0].occurred_at == datetime(2022, 1, 2, 3, 4, 5, tzinfo=UTC)
    assert stats.skipped_empty == 1 and stats.skipped_files == 1


def test_frontmatter_and_section_helpers():
    frontmatter, body = split_frontmatter("---\ntags: a, b\ndate: '2024-01-01'\n---\nBody")
    assert frontmatter.tags() == ["a", "b"] and frontmatter.fields["date"] == "2024-01-01" and body == "Body"
    assert split_frontmatter("--- not frontmatter\ntext")[1] == "--- not frontmatter\ntext"
    assert [section.heading_path for section in split_sections("pre\n# A\nx\n## B\ny\n# C\nz")] == [
        (), ("A",), ("A", "B"), ("C",)]


# ─── shared helpers ─────────────────────────────────────────────────────────

def test_streaming_reader_handles_tiny_chunks_and_rejects_garbage(tmp_path, monkeypatch):
    monkeypatch.setattr(json_stream, "READ_CHUNK_CHARS", 7)
    path = tmp_path / "items.json"
    items = [{"n": index, "text": "é" * index, "nested": [1.5, True, None]} for index in range(50)] + [12345, "s"]
    path.write_text(json.dumps(items), encoding="utf-8")
    assert list(json_stream.iter_json_array(path)) == items
    wrapped = tmp_path / "wrapped.json"
    wrapped.write_text(json.dumps({"meta": {"x": [1, 2]}, "results": items[:3]}), encoding="utf-8")
    assert list(json_stream.iter_json_array(wrapped, ("results",))) == items[:3]
    for bad in ('[{"a": 1} {"b": 2}]', '[1, 2', '[1] trailing', '{"results": 5}', '[{"a": tru}]'):
        path.write_text(bad, encoding="utf-8")
        with pytest.raises(ImportFormatError):
            list(json_stream.iter_json_array(path, ("results",)))


def test_split_text_and_timestamps():
    pieces = split_text("para one\n\n" + "x" * 25 + "\n\npara three", 20)
    assert all(len(piece) <= 20 for piece in pieces) and pieces[0] == "para one"
    assert "".join(pieces).count("x") == 25
    assert parse_timestamp(T0 * 1000) == parse_timestamp(T0)
    assert parse_timestamp("2024-01-01T00:00:00") == datetime(2024, 1, 1, tzinfo=UTC)
    assert parse_timestamp("garbage") is None
    assert content_hash("k", "p", "t", None) != content_hash("k", "p", "t", parse_timestamp(T0))


# ─── end to end through the store ───────────────────────────────────────────

@pytest.fixture
def memory_dir(tmp_path, monkeypatch):
    root = tmp_path / "memory"
    root.mkdir()
    monkeypatch.setenv("TAM_MEMORY_DIR", str(root))
    return root


def _rows(memory_dir, project):
    with sqlite3.connect(memory_dir / "memory.db") as db:
        db.row_factory = sqlite3.Row
        return [dict(row) for row in db.execute(
            "SELECT id, content, context, tags, created_at, last_confirmed, source_format FROM knowledge "
            "WHERE project=? AND status='active' ORDER BY created_at", (project,))]


def test_chatgpt_import_preserves_time_and_is_idempotent(exports, memory_dir):
    stats = import_cli.run("chatgpt", exports / "chatgpt.json", project="pg", dry_run=False,
                           include_assistant=False, memory_dir=memory_dir)
    assert (stats.imported, stats.skipped_short, stats.skipped_role) == (2, 1, 2)
    rows = _rows(memory_dir, "pg")
    assert [row["created_at"] for row in rows] == ["2024-05-01T10:00:30.000000Z", "2024-05-01T10:00:50.000000Z"]
    assert rows[0]["last_confirmed"] == rows[0]["created_at"]
    assert {"imported", "import-chatgpt", "chatgpt", "role-user"} <= set(json.loads(rows[0]["tags"]))
    assert rows[0]["source_format"] == "conversation"
    with sqlite3.connect(memory_dir / "memory.db") as db:
        assert db.execute("SELECT COUNT(*) FROM import_ledger WHERE kind='chatgpt'").fetchone()[0] == 2

    again = import_cli.run("chatgpt", exports / "chatgpt.json", project="pg", dry_run=False,
                           include_assistant=False, memory_dir=memory_dir)
    assert (again.imported, again.already_imported) == (0, 2)
    assert len(_rows(memory_dir, "pg")) == 2


def test_dry_run_stores_nothing_and_sees_the_ledger(exports, memory_dir):
    dry = import_cli.run("claude", exports / "claude.json", project="billing", dry_run=True,
                         include_assistant=True, memory_dir=memory_dir)
    assert (dry.would_import, dry.imported) == (3, 0)
    assert not (memory_dir / "memory.db").exists()
    import_cli.run("claude", exports / "claude.json", project="billing", dry_run=False, include_assistant=True,
                   memory_dir=memory_dir)
    after = import_cli.run("claude", exports / "claude.json", project="billing", dry_run=True,
                           include_assistant=True, memory_dir=memory_dir)
    assert (after.would_import, after.already_imported) == (0, 3)


def test_imported_credentials_are_redacted(exports, memory_dir):
    stats = import_cli.run("claude", exports / "claude.json", project="billing", dry_run=False,
                           include_assistant=False, memory_dir=memory_dir)
    assert stats.redacted == 1
    assert all(API_KEY not in row["content"] for row in _rows(memory_dir, "billing"))


def test_duplicates_of_existing_records_are_not_backdated(exports, memory_dir):
    import_cli.run("mem0", exports / "mem0.json", project="prefs", dry_run=False, include_assistant=False,
                   memory_dir=memory_dir)
    created = {row["content"]: row["created_at"] for row in _rows(memory_dir, "prefs")}
    assert created["Prefers PostgreSQL over MySQL for new services"] == "2024-03-10T19:00:00.000000Z"
    copy = exports / "mem0-copy.json"
    shifted = {"results": [dict(MEM0_EXPORT["results"][0], created_at="2020-01-01T00:00:00Z")]}
    copy.write_text(json.dumps(shifted), encoding="utf-8")
    stats = import_cli.run("mem0", copy, project="prefs", dry_run=False, include_assistant=False,
                           memory_dir=memory_dir)
    assert (stats.imported, stats.deduplicated) == (0, 1)
    after = {row["content"]: row["created_at"] for row in _rows(memory_dir, "prefs")}
    assert after == created


def test_markdown_import_and_recall_by_time(vault, memory_dir):
    stats = import_cli.run("markdown", vault, project="notes", dry_run=False, include_assistant=False,
                           memory_dir=memory_dir)
    assert stats.imported == 6
    rows = _rows(memory_dir, "notes")
    assert rows[0]["created_at"].startswith("2022-01-02T03:04:05")
    assert rows[-1]["created_at"] == "2023-11-05T00:00:00.000000Z"
    assert stats.first_timestamp == "2022-01-02T03:04:05.000000Z"


def test_long_turns_are_split_into_parts(tmp_path, memory_dir):
    export = [{"title": "Long", "create_time": T0, "conversation_id": "long", "current_node": "u1",
               "mapping": {"u1": _node("u1", None, [], role="user",
                                       text="\n\n".join(f"Paragraph {i}: " + "detail " * 300 for i in range(6)),
                                       created=T0)}}]
    path = tmp_path / "long.json"
    path.write_text(json.dumps(export), encoding="utf-8")
    stats = import_cli.run("chatgpt", path, project="long", dry_run=False, include_assistant=False,
                           memory_dir=memory_dir)
    rows = _rows(memory_dir, "long")
    assert stats.split_parts == len(rows) > 1
    assert all("(part " in row["context"] for row in rows)


def test_cli_main_prints_summary_and_reports_format_errors(exports, memory_dir, capsys):
    assert import_cli.main(["mem0", str(exports / "mem0.json"), "--project", "prefs", "--dry-run",
                            "--memory-dir", str(memory_dir), "--quiet"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["would_import"] == 2 and summary["dry_run"] is True
    assert import_cli.main(["claude", str(exports / "mem0.json"), "--memory-dir", str(memory_dir)]) == 2
    assert "tam import:" in capsys.readouterr().err
    assert import_cli.main(["markdown", str(exports / "missing"), "--memory-dir", str(memory_dir)]) == 2


def test_tam_entry_point_routes_import(monkeypatch):
    import total_agent_memory.server as entry

    assert entry._portable_cli("import") is import_cli.main


def test_deeply_nested_json_exits_with_a_usage_error_not_a_traceback(tmp_path, memory_dir, capsys):
    depth = 200_000
    nested = tmp_path / "nested.json"
    nested.write_text("[" * depth + "]" * depth, encoding="utf-8")
    code = import_cli.main(["mem0", str(nested), "--dry-run", "--project", "p", "--memory-dir", str(memory_dir)])
    assert code == import_cli.EXIT_USAGE
    assert "nested too deeply" in capsys.readouterr().err
