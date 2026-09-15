import sys
from pathlib import Path


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from team_memory.remote import main as run
    run()
