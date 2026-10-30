"""Owner-only file creation for exports and other files that hold memory content."""

from __future__ import annotations

import os
from pathlib import Path
from typing import IO

PRIVATE_FILE_MODE = 0o600


def open_private_text(path: Path) -> IO[str]:
    """Open `path` for writing as UTF-8 text with mode 0600, also when the file already exists wider."""
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, PRIVATE_FILE_MODE)
    try:
        if os.name != "nt":
            os.fchmod(descriptor, PRIVATE_FILE_MODE)
        return os.fdopen(descriptor, "w", encoding="utf-8")
    except BaseException:
        os.close(descriptor)
        raise
