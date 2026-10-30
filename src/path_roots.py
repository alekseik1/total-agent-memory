"""Implicit filesystem roots for caller-supplied paths."""

from __future__ import annotations

from pathlib import Path


def cwd_root(cwd: Path | None = None, home: Path | None = None) -> Path | None:
    """The working directory as an allowed root, or None when it would expose too much.

    A process started from `/` or from the user's home directory would otherwise let a tool read the whole
    filesystem or home; such a launch must list its roots explicitly (TAM_ALLOWED_PATH_ROOTS).
    """
    resolved = (cwd or Path.cwd()).resolve()
    if resolved == Path(resolved.anchor) or resolved == (home or Path.home()).resolve():
        return None
    return resolved
