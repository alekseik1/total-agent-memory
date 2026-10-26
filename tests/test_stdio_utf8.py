"""Helper processes read and write UTF-8 even when the console codec is not."""

from __future__ import annotations

import io
import os
import subprocess
import sys
from pathlib import Path

import pytest

import stdio_utf8

SRC = Path(__file__).resolve().parent.parent / "src"
TEXT = "память — 日本語"


def _run(code: str, payload: bytes) -> subprocess.CompletedProcess[bytes]:
    # PYTHONIOENCODING pins the pipes to cp1252 on every OS, as on a stock Windows install.
    env = {**os.environ, "PYTHONPATH": str(SRC), "PYTHONIOENCODING": "cp1252", "PYTHONUTF8": "0"}
    return subprocess.run([sys.executable, "-c", code], input=payload, capture_output=True, env=env, timeout=60, check=False)


def test_use_utf8_stdio_round_trips_non_ascii_through_pipes():
    code = "import sys; from stdio_utf8 import use_utf8_stdio; use_utf8_stdio(); sys.stdout.write(sys.stdin.read())"
    result = _run(code, TEXT.encode("utf-8"))
    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")
    assert result.stdout.decode("utf-8") == TEXT


def test_without_the_helper_the_pipe_codec_mangles_the_text():
    result = _run("import sys; sys.stdout.write(sys.stdin.read())", TEXT.encode("utf-8"))
    assert result.stdout.decode("utf-8", "replace") != TEXT


def test_description_from_stdin_is_decoded_as_utf8():
    code = (
        "import argparse, sys; import auto_self_improve as m; "
        "ns = argparse.Namespace(description_stdin=True, description=None); "
        "sys.stdout.buffer.write(m._resolve_description(ns).encode('utf-8'))"
    )
    result = _run(code, TEXT.encode("utf-8"))
    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")
    assert result.stdout.decode("utf-8") == TEXT


def _console(encoding: str) -> tuple[io.BytesIO, io.TextIOWrapper]:
    raw = io.BytesIO()
    return raw, io.TextIOWrapper(raw, encoding=encoding, errors="strict", newline="\n", write_through=True)


def test_cli_output_escapes_what_a_cp1252_console_cannot_show(monkeypatch):
    raw, console = _console("cp1252")
    monkeypatch.setattr(sys, "stdout", console)
    monkeypatch.setattr(sys, "stderr", console)
    stdio_utf8.never_fail_on_output()
    print("OK: registered -> C:\\Users\\Вася → memory")
    assert raw.getvalue().decode("cp1252") == "OK: registered -> C:\\Users\\\\u0412\\u0430\\u0441\\u044f \\u2192 memory\n"


def test_without_the_guard_a_cp1252_console_raises(monkeypatch):
    _raw, console = _console("cp1252")
    monkeypatch.setattr(sys, "stdout", console)
    with pytest.raises(UnicodeEncodeError):
        print("→")


def test_a_utf8_console_is_left_alone(monkeypatch):
    raw, console = _console("utf-8")
    monkeypatch.setattr(sys, "stdout", console)
    monkeypatch.setattr(sys, "stderr", console)
    stdio_utf8.never_fail_on_output()
    print("Вася →")
    assert raw.getvalue().decode("utf-8") == "Вася →\n"
    assert console.errors == "strict"
