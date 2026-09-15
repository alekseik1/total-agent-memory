import hashlib
import json
import os
import sqlite3
from contextlib import ExitStack, closing
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from pydantic import Field

from team_memory.contracts import DTO, Conflict
from version import VERSION

LOCK_BYTE = b'0'


class ServerLease:
    def __init__(self, root: Path):
        self.root = root
        self.file = None

    def __enter__(self):
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.file = (self.root / '.server.lock').open('a+b')
        self.file.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.file.seek(0)
            if not self.file.read(1):
                self.file.write(LOCK_BYTE)
                self.file.flush()
        except OSError as exc:
            self.file.close()
            self.file = None
            raise Conflict('Server data is in use; stop the server before offline maintenance') from exc
        return self

    def __exit__(self, exc_type, exc, traceback):
        if self.file is not None:
            self.file.close()
            self.file = None


class DatabaseSnapshot(DTO):
    path: str
    sha256: str = Field(pattern=r'^[a-f0-9]{64}$')


class Snapshot(DTO):
    format_version: int = 1
    package_version: str
    created_at: str
    databases: list[DatabaseSnapshot]


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open('rb') as source:
        while chunk := source.read(1024 * 1024):
            result.update(chunk)
    return result.hexdigest()


def verify_database(path: Path) -> None:
    with closing(sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)) as db:
        if db.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
            raise Conflict('Snapshot database integrity check failed')
        if db.execute('PRAGMA foreign_key_check').fetchone() is not None:
            raise Conflict('Snapshot database foreign-key check failed')


def backup(root: Path, destination: Path) -> Snapshot:
    root, destination = root.resolve(), destination.resolve()
    if not (root / 'identity.db').is_file():
        raise Conflict('Server identity database does not exist')
    if destination == root or root in destination.parents:
        raise Conflict('Backup must be outside server data')
    with ServerLease(root), ExitStack() as leases:
        sources = [root / 'identity.db', *sorted((root / 'workspaces').glob('*/memory.db'))]
        for directory in sorted({source.parent for source in sources if source.name == 'memory.db'}):
            leases.enter_context(ServerLease(directory))
        destination.mkdir(parents=True, exist_ok=False, mode=0o700)
        entries = []
        for source in sources:
            if source.is_symlink() or root not in source.resolve().parents:
                raise Conflict('Database path must remain inside server data')
            relative = source.relative_to(root)
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            with closing(sqlite3.connect(source.as_uri() + '?mode=ro', uri=True)) as src, closing(sqlite3.connect(target)) as dst:
                src.backup(dst)
            target.chmod(0o600)
            verify_database(target)
            entries.append(DatabaseSnapshot(path=relative.as_posix(), sha256=digest(target)))
        manifest = Snapshot(package_version=VERSION, created_at=datetime.now(timezone.utc).isoformat(), databases=entries)
        (destination / 'manifest.json').write_text(manifest.model_dump_json(indent=2))
        return manifest


def restore(snapshot: Path, destination: Path) -> None:
    import re
    import shutil

    snapshot, destination = snapshot.resolve(), destination.resolve()
    manifest = Snapshot.model_validate_json((snapshot / 'manifest.json').read_text())
    if manifest.format_version != 1:
        raise Conflict('Unsupported snapshot format')
    paths = [entry.path for entry in manifest.databases]
    if paths.count('identity.db') != 1 or len(paths) != len(set(paths)):
        raise Conflict('Invalid snapshot database list')
    for entry in manifest.databases:
        if not re.fullmatch(r'identity\.db|workspaces/(shared|(?:personal|team)_[a-f0-9]{64})/memory\.db', entry.path):
            raise Conflict('Invalid snapshot path')
        path = snapshot / entry.path
        if path.is_symlink() or snapshot not in path.resolve().parents or digest(path) != entry.sha256:
            raise Conflict('Snapshot checksum or path validation failed')
        verify_database(path)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with TemporaryDirectory(prefix='.tam-restore-', dir=destination.parent) as temporary:
        staging = Path(temporary) / 'data'
        staging.mkdir(mode=0o700)
        for entry in manifest.databases:
            target = staging / entry.path
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            shutil.copyfile(snapshot / entry.path, target)
            target.chmod(0o600)
            if digest(target) != entry.sha256:
                raise Conflict('Snapshot changed during restore')
            verify_database(target)
        (staging / 'restored-from.json').write_text(json.dumps({'snapshot': manifest.created_at, 'version': manifest.package_version}))
        if destination.exists():
            raise FileExistsError(destination)
        staging.rename(destination)
