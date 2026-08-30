"""One human session, one identity.

The server minted `mcp_<ts>_<pid>` while the SessionEnd hook could only name
the Claude Code session, so knowledge saved through the tools and the summary
written at the end belonged to two different ids — and `sessions` grew a row
per identity space rather than per session. The host publishes its id in the
environment it hands the server; nothing read it.
"""

import os
import re

import server


def test_the_host_session_id_is_used_when_published(monkeypatch):
    monkeypatch.setenv("CLAUDE_CODE_SESSION_ID", "1c1596bb-9e67-49bf-909a-eabab5bcd90d")

    assert server._session_id() == "1c1596bb-9e67-49bf-909a-eabab5bcd90d"


def test_it_matches_the_id_the_hook_reports():
    """The hook names the transcript's basename; the env var is the same id.

    Both halves must agree or the summary lands on a session row the tools
    never wrote to — which is exactly what happened.
    """
    transcript_basename = "7733731d-4bba-4fa2-95df-7e4dbf51474f"
    os.environ["CLAUDE_CODE_SESSION_ID"] = transcript_basename
    try:
        assert server._session_id() == transcript_basename
    finally:
        del os.environ["CLAUDE_CODE_SESSION_ID"]


def test_a_host_that_publishes_nothing_still_gets_an_id(monkeypatch):
    """Codex, Cursor and the Docker image publish no session id."""
    monkeypatch.delenv("CLAUDE_CODE_SESSION_ID", raising=False)

    sid = server._session_id()

    assert re.fullmatch(r"mcp_\d{8}_\d{6}_\d+", sid), sid


def test_a_blank_value_is_not_taken_as_an_id(monkeypatch):
    monkeypatch.setenv("CLAUDE_CODE_SESSION_ID", "   ")

    assert server._session_id().startswith("mcp_")


def test_bootstrap_takes_the_id_from_the_resolver():
    """Pin the wiring — the resolver is useless if bootstrap mints its own."""
    import inspect

    src = inspect.getsource(server._bootstrap_session)
    assert "SID = _session_id()" in src, src
