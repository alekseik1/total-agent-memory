"""Text-mode file access names its encoding.

A Windows machine without UTF-8 mode decodes with cp1252, so a bare
``Path.read_text()`` on a UTF-8 file raised ``UnicodeDecodeError`` for real
users while macOS and CI (which forced ``PYTHONUTF8=1``) stayed green.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from tests.encoding_scan import scan_source, scan_tree

ROOT = Path(__file__).resolve().parent.parent
SCANNED = ("src", "scripts", "hooks", "tests", "benchmarks")


def _kinds(code: str) -> list[str]:
    return [finding.kind for finding in scan_source(Path("snippet.py"), textwrap.dedent(code))]


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("p.read_text()", ["read_text"]),
        ("p.write_text('x')", ["write_text"]),
        ("open(p)", ["open"]),
        ("open(p, 'w')", ["open"]),
        ("p.open('a')", ["Path.open"]),
        ("config_path.open()", ["Path.open"]),
        ("os.fdopen(fd, 'w')", ["os.fdopen"]),
        ("io.open(p)", ["io.open"]),
        ("tempfile.NamedTemporaryFile('w')", ["NamedTemporaryFile"]),
    ],
)
def test_scanner_flags_a_missing_encoding(code, expected):
    assert _kinds(code) == expected


@pytest.mark.parametrize(
    "code",
    [
        "p.read_text(encoding='utf-8')",
        "p.read_text('utf-8')",
        "p.write_text('x', encoding='utf-8')",
        "p.write_text('x', 'utf-8')",
        "open(p, 'rb')",
        "open(p, 'w', encoding='utf-8')",
        "open(p, 'r', -1, 'utf-8')",
        "open(p, **options)",
        "p.open('rb')",
        "p.open('w', encoding='utf-8')",
        "zf.open('a')",
        "sessions.open(token)",
        "os.fdopen(fd, 'wb')",
        "os.open(p, os.O_RDONLY)",
        "tempfile.NamedTemporaryFile(delete=False)",
        "tempfile.NamedTemporaryFile('w', encoding='utf-8')",
    ],
)
def test_scanner_accepts_an_explicit_codec_or_binary_mode(code):
    assert _kinds(code) == []


def test_no_text_mode_access_relies_on_the_locale_encoding():
    findings = scan_tree(ROOT, SCANNED)
    assert not findings, "text-mode access without encoding= (cp1252 on Windows):\n" + "\n".join(
        finding.where for finding in findings
    )


CI_WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"


@pytest.mark.skipif(not CI_WORKFLOW.is_file(), reason=".github/ is not shipped in the source distribution")
def test_windows_ci_cells_run_without_utf8_mode():
    import yaml

    env = yaml.safe_load(CI_WORKFLOW.read_text(encoding="utf-8"))["jobs"]["test"]["env"]
    assert "windows-2022" in env["PYTHONUTF8"] and "'0'" in env["PYTHONUTF8"]
    assert "windows-2022" in env["PYTHONIOENCODING"] and env["PYTHONIOENCODING"].endswith("|| '' }}")
