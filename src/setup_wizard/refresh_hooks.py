"""`python -m setup_wizard.refresh_hooks`: refresh the hook copies in ~/.claude/hooks after a source update."""
import os
import sys
from collections.abc import Sequence

from setup_wizard import clients, extras
from setup_wizard.contracts import WizardError
from setup_wizard.locations import install_root, locate_memory_dir

FAILED = 1


def main(argv: Sequence[str] | None = None) -> int:
    del argv
    host = clients.Host.current()
    try:
        written = extras.refresh_installed_hooks(install_root(), host, locate_memory_dir(os.environ))
    except WizardError as exc:
        sys.stderr.write(f"  ERROR: {exc}\n")
        return FAILED
    except OSError as exc:
        sys.stderr.write(f"  ERROR: could not write {exc.filename or ''}: {exc.strerror}\n")
        return FAILED
    sys.stdout.write(f"hooks refreshed: {len(written)} file(s)\n" if written else "hooks: none installed, nothing to refresh\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
