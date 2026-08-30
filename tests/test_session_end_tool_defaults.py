"""`session_end` must not force a caller to name a session it cannot know.

No tool hands out this process's SID, so a caller obliged to pass `session_id`
invents one — which is how 231 of 256 stored summaries came to name sessions
that never existed. Omitting it now means "this server's own session".
"""

import asyncio
import inspect

import server


def _schema(tool):
    """The field is `input_schema` on the 2.x SDK and `inputSchema` on 1.x."""
    return getattr(tool, "input_schema", None) or tool.inputSchema


def _session_end_tool():
    tools = asyncio.run(server.list_tools())
    by_name = {t.name: t for t in tools}
    assert "session_end" in by_name, sorted(by_name)
    return by_name["session_end"]


def test_session_id_is_not_required():
    schema = _schema(_session_end_tool())
    assert schema.get("required", []) == [], schema.get("required")
    assert "session_id" in schema["properties"]


def test_the_description_tells_a_caller_to_omit_it():
    """A caller that reads only the schema must learn the default exists."""
    desc = _schema(_session_end_tool())["properties"]["session_id"].get("description", "")
    assert "omit" in desc.lower(), desc


def test_the_handler_falls_back_to_the_live_sid():
    """Pin the wiring: the fallback is the one value a caller could not pass.

    Matched together with the `summary` argument that follows it: `save_intent`
    carries the same `or SID` idiom, and a bare match on it passes while this
    handler is broken.
    """
    src = inspect.getsource(server._do)
    assert 'a.get("session_id") or SID, a.get("summary")' in src


def test_branch_reaches_session_end():
    """The row's branch came out empty because the tool never forwarded one."""
    schema = _schema(_session_end_tool())
    assert "branch" in schema["properties"]
    assert 'branch=a.get("branch", BRANCH)' in inspect.getsource(server._do)
