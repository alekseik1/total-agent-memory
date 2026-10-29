"""Exploration memos: conclusions keyed to source-file hashes, fresh/stale on recall."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

from tests.pg_store_support import store_backend  # noqa: F401 — fixture

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import exploration_memo


@pytest.fixture
def live_server(store_backend, monkeypatch, tmp_path):  # noqa: F811 — pytest fixture injection
    memory = tmp_path / "memory"
    for sub in ("raw", "blobs", "chroma"):
        (memory / sub).mkdir(parents=True, exist_ok=True)
    import server
    monkeypatch.setattr(server, "MEMORY_DIR", memory)
    store = server.Store(database=store_backend)
    monkeypatch.setattr(server, "store", store)
    monkeypatch.setattr(server, "recall", server.Recall(store))
    monkeypatch.setattr(server, "SID", "sess-exploration")
    monkeypatch.setattr(server, "_v5_modules", {})
    yield server, store
    store.db.close()


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    (root / "src" / "auth").mkdir(parents=True)
    (root / "src" / "auth" / "middleware.py").write_text("def check_token(request):\n    return True\n", encoding="utf-8")
    (root / "src" / "auth" / "routes.py").write_text("ROUTES = ['/login', '/logout']\n", encoding="utf-8")
    (root / "README.md").write_text("# Demo service\n", encoding="utf-8")
    return root


def call(server, tool, **args):
    content, is_error = asyncio.run(server._call_tool_impl(tool, args))
    text = content[0].text
    try:
        return json.loads(text), is_error
    except ValueError:
        return text, is_error


def record(server, repo, question="How is authentication wired?", files=None, **extra):
    return call(server, "memory_exploration_record", question=question,
                conclusion="Auth runs in src/auth/middleware.py check_token(); routes.py lists public routes.",
                files=files or ["src/auth/middleware.py", "src/auth/routes.py"], project="demo", root=str(repo),
                **extra)


def test_record_then_recall_is_fresh_with_relative_paths(live_server, repo):
    server, store = live_server
    saved, is_error = record(server, repo, files=["src/auth/middleware.py", str(repo / "src" / "auth" / "routes.py")])
    assert not is_error, saved
    assert saved["recorded"] is True and saved["replaced"] is False
    assert [item["path"] for item in saved["files"]] == ["src/auth/middleware.py", "src/auth/routes.py"]
    stored = store.db.execute("SELECT path FROM exploration_memo_files ORDER BY path").fetchall()
    assert [row[0] for row in stored] == ["src/auth/middleware.py", "src/auth/routes.py"]

    found, is_error = call(server, "memory_exploration_recall", question="how is authentication wired",
                           project="demo", root=str(repo))
    assert not is_error, found
    assert found["fresh"] == 1 and found["stale"] == 0
    hit = found["results"][0]
    assert hit["status"] == "fresh"
    assert hit["score"] == pytest.approx(1.0)
    assert hit["conclusion"].startswith("Auth runs in")
    assert "changed_files" not in hit


def test_changed_and_missing_files_make_the_memo_stale(live_server, repo):
    server, _ = live_server
    record(server, repo, files=["src/auth/middleware.py", "src/auth/routes.py", "README.md"])
    (repo / "src" / "auth" / "middleware.py").write_text("def check_token(request):\n    return False\n", encoding="utf-8")
    (repo / "README.md").unlink()
    found, _ = call(server, "memory_exploration_recall", question="How is authentication wired?",
                    project="demo", root=str(repo))
    hit = found["results"][0]
    assert hit["status"] == "stale"
    assert hit["changed_files"] == ["src/auth/middleware.py"]
    assert hit["missing_files"] == ["README.md"]

    only_fresh, _ = call(server, "memory_exploration_recall", question="How is authentication wired?",
                         project="demo", root=str(repo), include_stale=False)
    assert only_fresh["results"] == []


def test_moved_checkout_is_verified_against_the_recall_root(live_server, repo, tmp_path):
    server, _ = live_server
    record(server, repo)
    moved = tmp_path / "moved"
    repo.rename(moved)
    found, _ = call(server, "memory_exploration_recall", question="How is authentication wired?",
                    project="demo", root=str(moved))
    assert found["results"][0]["status"] == "fresh"


@pytest.mark.parametrize("outside", ["../secret.txt", "{tmp}/secret.txt", "link.txt"])
def test_files_outside_the_root_are_refused(live_server, repo, tmp_path, outside, make_symlink):
    server, store = live_server
    secret = tmp_path / "secret.txt"
    secret.write_text("not part of the repo", encoding="utf-8")
    if outside == "link.txt":
        make_symlink(repo / "link.txt", secret)
    result, is_error = record(server, repo, files=[outside.format(tmp=tmp_path)])
    assert is_error
    assert "outside the project root" in result
    assert store.db.execute("SELECT COUNT(*) FROM exploration_memos").fetchone()[0] == 0


def test_symlink_swapped_in_after_recording_is_not_followed(live_server, repo, tmp_path, make_symlink):
    server, _ = live_server
    record(server, repo)
    target = repo / "src" / "auth" / "routes.py"
    outside = tmp_path / "elsewhere.py"
    outside.write_text(target.read_text(encoding="utf-8"), encoding="utf-8")
    target.unlink()
    make_symlink(target, outside)
    found, _ = call(server, "memory_exploration_recall", question="How is authentication wired?",
                    project="demo", root=str(repo))
    hit = found["results"][0]
    assert hit["status"] == "stale"
    assert hit["unverifiable_files"] == [{"path": "src/auth/routes.py", "reason": "outside_root"}]


def test_missing_file_directory_and_relative_root_are_refused(live_server, repo):
    server, _ = live_server
    assert record(server, repo, files=["src/auth/nope.py"])[1]
    assert record(server, repo, files=["src/auth"])[1]
    result, is_error = call(server, "memory_exploration_record", question="q?", conclusion="c", files=["README.md"],
                            project="demo", root="relative/root")
    assert is_error and "absolute" in result


def test_size_and_count_limits(live_server, repo, monkeypatch):
    server, _ = live_server
    monkeypatch.setattr(exploration_memo, "MAX_FILE_BYTES", 10)
    result, is_error = record(server, repo, files=["src/auth/middleware.py"])
    assert is_error and "limit" in result
    too_many = ["README.md"] * (exploration_memo.MAX_FILES_PER_MEMO + 1)
    _, is_error = record(server, repo, files=too_many)
    assert is_error


def test_rerecording_a_question_replaces_the_memo(live_server, repo):
    server, store = live_server
    first, _ = record(server, repo)
    second, _ = call(server, "memory_exploration_record", question="how is AUTHENTICATION wired",
                     conclusion="Auth moved to a gateway.", files=["README.md"], project="demo", root=str(repo))
    assert second["replaced"] is True and second["id"] == first["id"]
    assert store.db.execute("SELECT COUNT(*) FROM exploration_memos").fetchone()[0] == 1
    paths = [row[0] for row in store.db.execute("SELECT path FROM exploration_memo_files")]
    assert paths == ["README.md"]


def test_ranking_prefers_the_similar_question_and_projects_are_isolated(live_server, repo):
    server, _ = live_server
    record(server, repo, question="How is authentication wired?")
    call(server, "memory_exploration_record", question="Where are database migrations applied?",
         conclusion="Migrations run from Store._apply_sql_migrations.", files=["README.md"], project="demo",
         root=str(repo))
    call(server, "memory_exploration_record", question="How is authentication wired?",
         conclusion="Other project.", files=["README.md"], project="other", root=str(repo))
    found, _ = call(server, "memory_exploration_recall", question="where are the database migrations applied",
                    project="demo", root=str(repo))
    assert found["results"][0]["question"] == "Where are database migrations applied?"
    assert all(hit["conclusion"] != "Other project." for hit in found["results"])


def test_credentials_in_conclusions_are_redacted(live_server, repo):
    server, store = live_server
    key = "sk-proj-" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6"
    _, is_error = call(server, "memory_exploration_record", question="Which API key does the client use?",
                       conclusion=f"The client reads OPENAI_API_KEY={key} from .env", files=["README.md"],
                       project="demo", root=str(repo))
    assert not is_error
    conclusion = store.db.execute("SELECT conclusion FROM exploration_memos").fetchone()[0]
    assert key not in conclusion


@pytest.mark.parametrize("arguments", [
    {"question": "  ", "conclusion": "c", "files": ["README.md"], "project": "demo"},
    {"question": "q", "conclusion": "c", "files": [], "project": "demo"},
    {"question": "q", "conclusion": "c", "files": ["README.md"], "project": "demo", "unknown": 1},
])
def test_invalid_arguments_are_errors(live_server, repo, arguments):
    server, _ = live_server
    _, is_error = call(server, "memory_exploration_record", root=str(repo), **arguments)
    assert is_error


def test_tools_are_annotated():
    assert "memory_exploration_recall" in __import__("server")._READ_ONLY_TOOLS
    assert "memory_exploration_record" in __import__("server")._IDEMPOTENT_TOOLS


class _Db:
    """Minimal in-memory store connection with the 042 tables."""

    @staticmethod
    def connect():
        import sqlite3
        db = sqlite3.connect(":memory:")
        sql = (Path(__file__).parent.parent / "migrations" / "042_import_ledger_exploration_memo.sql").read_text(encoding="utf-8")
        db.executescript(sql)
        return db


def _word_jaccard(a: str, b: str) -> float:
    left, right = set(a.lower().split()), set(b.lower().split())
    return len(left & right) / len(left | right) if left and right else 0.0


def test_semantic_score_is_used_only_for_the_same_model(repo):
    db = _Db.connect()
    vectors = {"alpha question": [1.0, 0.0], "totally different words": [0.9, 0.1]}
    memos = exploration_memo.ExplorationMemos(db, lexical=_word_jaccard, embed=lambda texts: [vectors[texts[0]]],
                                              model_name="m1")
    memos.record(exploration_memo.RecordArguments(question="alpha question", conclusion="c",
                                                  files=["README.md"], project="p", root=str(repo)))
    query = exploration_memo.RecallArguments(question="totally different words", project="p", root=str(repo))
    hit = memos.recall(query)["results"][0]
    assert hit["score"] > 0.7
    other_model = exploration_memo.ExplorationMemos(db, lexical=_word_jaccard,
                                                    embed=lambda texts: [vectors[texts[0]]], model_name="m2")
    assert other_model.recall(query)["results"] == []


def test_embedding_failure_falls_back_to_lexical(repo):
    db = _Db.connect()

    def broken(_texts):
        raise RuntimeError("model unavailable")

    memos = exploration_memo.ExplorationMemos(db, lexical=_word_jaccard, embed=broken, model_name="m1")
    saved = memos.record(exploration_memo.RecordArguments(question="where is caching done", conclusion="c",
                                                          files=["README.md"], project="p", root=str(repo)))
    assert saved["semantic_index"] is False
    found = memos.recall(exploration_memo.RecallArguments(question="where is caching configured", project="p",
                                                          root=str(repo)))
    assert found["results"][0]["score"] == pytest.approx(_word_jaccard("where is caching configured",
                                                                       "where is caching done"), abs=1e-4)
