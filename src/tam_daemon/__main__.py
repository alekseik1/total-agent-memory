"""``python <src>/tam_daemon daemon ...|proxy`` — also what detached daemons are spawned with."""

import sys
from pathlib import Path

# Run as a directory, Python puts this package dir first on sys.path; TAM's
# modules live one level up, which is what server.py itself expects.
_src = str(Path(__file__).resolve().parent.parent)
if sys.path and Path(sys.path[0]).resolve() == Path(__file__).resolve().parent:
    sys.path[0] = _src
elif _src not in sys.path:
    sys.path.insert(0, _src)

from tam_daemon.cli import dispatch

if __name__ == "__main__":
    sys.exit(dispatch(sys.argv[1:]))
