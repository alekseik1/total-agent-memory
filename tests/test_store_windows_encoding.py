import builtins
import json
from pathlib import Path


def test_migrations_and_raw_journal_ignore_windows_legacy_encoding(tmp_path, monkeypatch):
    import server

    read_text = Path.read_text
    open_file = builtins.open

    def windows_read(path, *args, **kwargs):
        kwargs.setdefault('encoding', 'cp1252')
        return read_text(path, *args, **kwargs)

    def windows_open(path, mode='r', *args, **kwargs):
        if 'b' not in mode and not args:
            kwargs.setdefault('encoding', 'cp1252')
        return open_file(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, 'read_text', windows_read)
    monkeypatch.setattr(builtins, 'open', windows_open)
    monkeypatch.setattr(server, 'MEMORY_DIR', tmp_path)
    monkeypatch.setenv('TAM_MEMORY_DIR', str(tmp_path))
    monkeypatch.setenv('USE_BINARY_SEARCH', 'true')
    store = server.Store()
    try:
        store.raw_append('encoding-check', {'content': 'Вася исправил запись Пети — запуск готов.'})
        data = json.loads((tmp_path / 'raw/encoding-check.jsonl').read_bytes().decode('utf-8'))
        assert data['content'] == 'Вася исправил запись Пети — запуск готов.'
        assert store.db.execute('SELECT count(*) FROM migrations').fetchone()[0] > 0
    finally:
        store.db.close()
