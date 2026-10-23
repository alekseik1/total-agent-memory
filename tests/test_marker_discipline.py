"""Quarantine discipline: a test may leave the blocking CI path only with a root cause on record.

``quarantined`` moves a test out of the required CI jobs into a non-blocking
one, and ``platform_sensitive`` flags a platform-dependent result. Both are
easy to reach for and easy to forget, so every use must carry a
``reason=`` string literal long enough to name the root cause or the issue
that tracks it. The markers themselves must stay registered, otherwise the
``-m "not quarantined"`` selection in CI silently selects nothing away.
"""

from __future__ import annotations

import ast
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
TESTS = ROOT / "tests"
DISCIPLINED = ("quarantined", "platform_sensitive")
MIN_REASON_CHARS = 25
REQUIRED_MARKERS = ("eval", "quarantined", "platform_sensitive", "slow", "postgres")


def registered_markers() -> dict[str, str]:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    out = {}
    for line in data["tool"]["pytest"]["ini_options"]["markers"]:
        name, _, description = line.partition(":")
        out[name.split("(")[0].strip()] = description.strip()
    return out


def marker_name(node: ast.AST) -> str | None:
    """`pytest.mark.<name>` or `pytest.mark.<name>(...)` -> name."""
    target = node.func if isinstance(node, ast.Call) else node
    if (
        isinstance(target, ast.Attribute)
        and isinstance(target.value, ast.Attribute)
        and target.value.attr == "mark"
        and isinstance(target.value.value, ast.Name)
        and target.value.value.id == "pytest"
    ):
        return target.attr
    return None


def disciplined_uses() -> list[tuple[str, int, str, ast.AST]]:
    uses = []
    for path in sorted(TESTS.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            name = marker_name(node)
            if name in DISCIPLINED:
                uses.append((str(path.relative_to(ROOT)), node.lineno, name, node))
    # A bare `pytest.mark.quarantined` attribute is also matched as the
    # func of its Call; keep only the outermost node per location.
    seen: dict[tuple[str, int, str], ast.AST] = {}
    for rel, line, name, node in uses:
        key = (rel, line, name)
        if key not in seen or isinstance(node, ast.Call):
            seen[key] = node
    return [(rel, line, name, node) for (rel, line, name), node in sorted(seen.items())]


def reason_of(node: ast.AST) -> str | None:
    if not isinstance(node, ast.Call):
        return None
    for keyword in node.keywords:
        if keyword.arg == "reason" and isinstance(keyword.value, ast.Constant) and isinstance(keyword.value.value, str):
            return keyword.value.value
    return None


@pytest.mark.parametrize("marker", REQUIRED_MARKERS)
def test_marker_is_registered_with_a_description(marker):
    markers = registered_markers()
    assert marker in markers, f"pytest marker {marker!r} is not registered in pyproject.toml"
    assert len(markers[marker]) >= 20, f"marker {marker!r} needs a real description"


@pytest.mark.parametrize("marker", DISCIPLINED)
def test_disciplined_marker_description_demands_a_root_cause(marker):
    assert "MUST" in registered_markers()[marker]


def test_every_quarantine_names_its_root_cause():
    bad = []
    for rel, line, name, node in disciplined_uses():
        reason = reason_of(node)
        if reason is None or len(reason.strip()) < MIN_REASON_CHARS:
            bad.append(f"{rel}:{line} pytest.mark.{name} needs reason='<root cause / issue, date>' "
                       f"(string literal, >= {MIN_REASON_CHARS} chars), got {reason!r}")
    assert not bad, "\n".join(bad)


def test_the_checker_recognises_a_missing_reason():
    tree = ast.parse("import pytest\n@pytest.mark.quarantined\ndef test_x(): pass\n"
                     "@pytest.mark.platform_sensitive(reason='NTFS has no POSIX modes; os.chmod is a no-op')\n"
                     "def test_y(): pass\n")
    found: dict[str, ast.AST] = {}
    for node in ast.walk(tree):
        name = marker_name(node)
        if name and (name not in found or isinstance(node, ast.Call)):
            found[name] = node
    assert reason_of(found["quarantined"]) is None
    assert reason_of(found["platform_sensitive"]).startswith("NTFS")
