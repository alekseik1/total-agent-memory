import sqlite3
from uuid import uuid4

import pytest

from team_memory.contracts import Conflict
from team_memory.lifecycle import ServerLease, backup, restore
from team_memory.registry import Registry


def test_backup_restore_authentication_and_checksum(tmp_path):
    root = tmp_path / 'original'
    registry = Registry(root)
    registry.add_user('vasya', 'Вася')
    token = registry.issue_token('vasya', 'client')
    snapshot = tmp_path / 'snapshot'
    with ServerLease(root), pytest.raises(Conflict):
        backup(root, snapshot)
    manifest = backup(root, snapshot)
    assert len(manifest.databases) == 1
    recovered = tmp_path / 'recovered'
    restore(snapshot, recovered)
    assert Registry(recovered).authenticate(token).user_id == 'vasya'
    with pytest.raises(FileExistsError):
        restore(snapshot, recovered)
    with (snapshot / 'identity.db').open('ab') as target:
        target.write(b'corrupted')
    with pytest.raises(Conflict):
        restore(snapshot, tmp_path / 'invalid')
    assert not (tmp_path / 'invalid').exists()


def test_lease_released_after_error(tmp_path):
    with pytest.raises(RuntimeError), ServerLease(tmp_path):
        with pytest.raises(Conflict), ServerLease(tmp_path):
            pytest.fail('Second server acquired the same data')
        raise RuntimeError('Crash')
    with ServerLease(tmp_path):
        assert (tmp_path / '.server.lock').is_file()


def test_restore_copy_failure_never_publishes_partial_server(tmp_path, monkeypatch):
    import shutil

    root = tmp_path / 'original'
    Registry(root).add_user('vasya', 'Вася')
    snapshot = tmp_path / 'snapshot'
    backup(root, snapshot)

    def failed_copy(source, target):
        target.write_bytes(b'partial')
        raise OSError('Disk full')

    monkeypatch.setattr(shutil, 'copyfile', failed_copy)
    destination = tmp_path / 'restored'
    with pytest.raises(OSError, match='Disk full'):
        restore(snapshot, destination)
    assert not destination.exists()


def test_backup_refuses_orphan_worker(tmp_path):
    root = tmp_path / 'server'
    Registry(root).add_user('vasya', 'Вася')
    workspace = root / 'workspaces' / 'shared'
    workspace.mkdir(parents=True)
    with sqlite3.connect(workspace / 'memory.db') as db:
        db.execute('CREATE TABLE example(id INTEGER PRIMARY KEY)')
    with ServerLease(workspace), pytest.raises(Conflict):
        backup(root, tmp_path / 'snapshot')
    assert not (tmp_path / 'snapshot').exists()


def test_idempotency_uses_same_transaction_as_write(tmp_path, monkeypatch):
    import server
    from team_memory.contracts import Save, Scope, Update, Work
    from team_memory.worker import Runtime

    root = tmp_path / 'data'
    monkeypatch.setattr(server, 'MEMORY_DIR', root)
    for name, value in {'TAM_MEMORY_DIR': str(root), 'CLAUDE_MEMORY_DIR': str(root),
                        'MEMORY_ASYNC_ENRICHMENT': 'false', 'USE_BINARY_SEARCH': 'true',
                        'MEMORY_QUALITY_GATE_ENABLED': 'false'}.items():
        monkeypatch.setenv(name, value)
    registry = Registry(tmp_path / 'identity')
    registry.add_user('vasya', 'Вася')
    actor = registry.authenticate(registry.issue_token('vasya', 'client'))
    workspace = registry.authorize(actor, Scope(), True)
    runtime = Runtime(str(root))
    request = Save(content='The release database uses SQLite WAL mode.', request_id=uuid4())
    work = Work(actor=actor, workspace=workspace, operation='memory_save',
                arguments=request.model_dump(mode='json', exclude={'scope'}))
    try:
        first = runtime.execute(work)
        assert runtime.execute(work) == first
        assert runtime.store.db.execute('SELECT count(*) FROM tam_requests').fetchone()[0] == 1
        changed = work.model_copy(update={'arguments': {**work.arguments, 'content': 'Different payload'}})
        with pytest.raises(Conflict):
            runtime.execute(changed)
        update = Update(id=first['id'], expected_revision=first['revision'], content='The release database uses WAL checkpoints.',
                        reason='Updated plan', request_id=uuid4())
        work = Work(actor=actor, workspace=workspace, operation='memory_update',
                    arguments=update.model_dump(mode='json', exclude={'scope'}))
        second = runtime.execute(work)
        assert runtime.execute(work) == second
        assert runtime.store.db.execute('SELECT count(*) FROM knowledge').fetchone()[0] == 2
    finally:
        runtime.store.db.close()
    with sqlite3.connect(root / 'memory.db') as db:
        assert db.execute('SELECT count(*) FROM tam_requests').fetchone()[0] == 2
