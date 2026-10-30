import sys
from pathlib import Path


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from stdio_utf8 import never_fail_on_output

    never_fail_on_output()
    from aml_adapter.cli import main as run
    sys.exit(run())
