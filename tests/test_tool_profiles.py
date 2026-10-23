"""TAM_TOOL_PROFILE: opt-in reduced tool catalogues with one-line descriptions."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import server
import tool_profiles
from settings_catalog import LOCAL_CATALOG, validate_value

# The reduced catalogues exist to save context; these are the floor of that saving.
MAX_LEAN_SHARE_OF_FULL = 0.4
MAX_MINIMAL_SHARE_OF_FULL = 0.2


def _schema(tool) -> dict:
    schema = getattr(tool, "input_schema", None)
    return tool.inputSchema if schema is None else schema


def _dump(tools) -> str:
    return json.dumps([tool.model_dump(mode="json", by_alias=True) for tool in tools], sort_keys=True)


def _tools_under(monkeypatch, profile: str | None):
    if profile is None:
        monkeypatch.delenv(tool_profiles.PROFILE_ENV, raising=False)
    else:
        monkeypatch.setenv(tool_profiles.PROFILE_ENV, profile)
    return asyncio.run(server.list_tools())


@pytest.fixture
def full_catalogue(monkeypatch):
    return {tool.name: tool for tool in _tools_under(monkeypatch, None)}


def test_every_profile_name_is_a_registered_tool(full_catalogue):
    tool_profiles.validate_against(full_catalogue)
    for name, tools in tool_profiles.PROFILES.items():
        assert tools is None or tools <= set(full_catalogue), name


def test_validation_names_the_unknown_tools():
    with pytest.raises(tool_profiles.ToolProfileError, match="memory_recall"):
        tool_profiles.validate_against({"memory_save"})


def test_minimal_is_inside_lean():
    assert tool_profiles.MINIMAL_TOOLS < tool_profiles.LEAN_TOOLS


@pytest.mark.parametrize("value", [None, "", "full", " FULL "])
def test_default_and_full_profile_serve_the_unchanged_catalogue(monkeypatch, value):
    raw = [server._annotate(tool) for tool in asyncio.run(server._tool_catalogue())]
    assert _dump(_tools_under(monkeypatch, value)) == _dump(raw)


@pytest.mark.parametrize("profile", [tool_profiles.LEAN, tool_profiles.MINIMAL])
def test_reduced_profiles_list_only_their_tools(monkeypatch, profile):
    tools = _tools_under(monkeypatch, profile)
    assert {tool.name for tool in tools} == tool_profiles.PROFILES[profile]


@pytest.mark.parametrize("profile", [tool_profiles.LEAN, tool_profiles.MINIMAL])
def test_reduced_profiles_use_one_line_signatures(monkeypatch, profile):
    for tool in _tools_under(monkeypatch, profile):
        assert tool.description == tool_profiles.SHORT_DESCRIPTIONS[tool.name]
        assert "\n" not in tool.description
        assert tool.description.startswith(f"{tool.name}(")


@pytest.mark.parametrize("name", sorted(tool_profiles.SHORT_DESCRIPTIONS))
def test_signatures_match_the_live_schema(name, full_catalogue):
    schema = _schema(full_catalogue[name])
    params = tool_profiles.signature_params(tool_profiles.SHORT_DESCRIPTIONS[name])
    assert set(params) <= set(schema.get("properties", {})), f"{name}: signature names unknown parameters"
    missing = set(schema.get("required", [])) - set(params)
    assert not missing, f"{name}: required parameters missing from the signature: {sorted(missing)}"
    assert len(params) == len(set(params)), name


@pytest.mark.parametrize("profile", [tool_profiles.LEAN, tool_profiles.MINIMAL])
def test_reduced_schemas_keep_structure_and_drop_prose(monkeypatch, profile, full_catalogue):
    for tool in _tools_under(monkeypatch, profile):
        reduced, full = _schema(tool), _schema(full_catalogue[tool.name])
        assert "description" not in reduced
        assert all("description" not in spec for spec in reduced.get("properties", {}).values())
        assert reduced == tool_profiles.strip_descriptions(full)
        assert set(reduced.get("properties", {})) == set(full.get("properties", {}))
        assert reduced.get("required") == full.get("required")
        assert tool.annotations == full_catalogue[tool.name].annotations


def test_strip_descriptions_keeps_a_parameter_named_description():
    schema = {"type": "object", "description": "x", "required": ["description"], "properties": {
        "description": {"type": "string", "description": "What happened"},
        "tags": {"type": "array", "items": {"type": "string", "description": "a tag"}},
    }}
    assert tool_profiles.strip_descriptions(schema) == {"type": "object", "required": ["description"], "properties": {
        "description": {"type": "string"}, "tags": {"type": "array", "items": {"type": "string"}}}}


def test_self_error_log_keeps_its_description_parameter(monkeypatch):
    tool = next(t for t in _tools_under(monkeypatch, "lean") if t.name == "self_error_log")
    assert "description" in _schema(tool)["properties"]
    assert "description" in _schema(tool)["required"]


@pytest.mark.parametrize("value", ["tiny", "everything", "1"])
def test_unknown_profile_is_an_error(value):
    with pytest.raises(tool_profiles.ToolProfileError, match="full, lean, minimal"):
        tool_profiles.resolve_profile({tool_profiles.PROFILE_ENV: value})


def test_unknown_profile_fails_list_tools_instead_of_exposing_everything(monkeypatch):
    monkeypatch.setenv(tool_profiles.PROFILE_ENV, "tiny")
    with pytest.raises(tool_profiles.ToolProfileError):
        asyncio.run(server.list_tools())


def test_unknown_profile_stops_server_startup(monkeypatch):
    monkeypatch.setenv(tool_profiles.PROFILE_ENV, "tiny")
    with pytest.raises(SystemExit, match="TAM_TOOL_PROFILE='tiny'"):
        server.run()


@pytest.fixture
def live_store(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "MEMORY_DIR", tmp_path)
    store = server.Store()
    monkeypatch.setattr(server, "store", store)
    monkeypatch.setattr(server, "recall", server.Recall(store))
    monkeypatch.setattr(server, "SID", "sess-profiles")
    monkeypatch.setattr(server, "TELEMETRY", None)
    yield store
    store.db.close()


def test_hidden_tool_stays_callable_and_is_traced(monkeypatch, live_store, caplog):
    monkeypatch.setenv(tool_profiles.PROFILE_ENV, "minimal")
    assert "memory_stats" not in {tool.name for tool in asyncio.run(server.list_tools())}
    with caplog.at_level(logging.DEBUG, logger="tool_profiles"):
        content, is_error = asyncio.run(server._call_tool_impl("memory_stats", {}))
    assert not is_error, content[0].text
    assert isinstance(json.loads(content[0].text), dict)
    assert any("hidden tool called: memory_stats (profile=minimal)" in r.getMessage() for r in caplog.records)


def test_listed_tool_is_not_traced_as_hidden(monkeypatch, live_store, caplog):
    monkeypatch.setenv(tool_profiles.PROFILE_ENV, "minimal")
    with caplog.at_level(logging.DEBUG, logger="tool_profiles"):
        asyncio.run(server._call_tool_impl("memory_get", {"ids": []}))
    assert not any("hidden tool called" in r.getMessage() for r in caplog.records)


def test_reduced_catalogues_are_much_smaller():
    spec = importlib.util.spec_from_file_location("tool_catalogue_size", ROOT / "scripts" / "tool_catalogue_size.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sizes = module.measure()
    full, lean, minimal = (sizes[name][1] for name in ("full", "lean", "minimal"))
    assert sizes["full"][0] == len(json.loads((ROOT / "tests" / "fixtures" / "tool_contract.json").read_text()))
    assert lean <= full * MAX_LEAN_SHARE_OF_FULL, sizes
    assert minimal <= full * MAX_MINIMAL_SHARE_OF_FULL, sizes


def test_settings_catalogue_offers_exactly_the_profiles():
    spec = next(s for s in LOCAL_CATALOG if s.key == tool_profiles.PROFILE_ENV)
    assert spec.choices == tool_profiles.PROFILE_NAMES
    assert f"lean: the {len(tool_profiles.LEAN_TOOLS)} " in spec.help
    assert f"minimal: the {len(tool_profiles.MINIMAL_TOOLS)} " in spec.help
    assert validate_value(spec, "lean") == "lean"
    with pytest.raises(ValueError):
        validate_value(spec, "tiny")
