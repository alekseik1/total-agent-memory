import asyncio
import json

import pytest

from memory_core.classifier import classify


@pytest.fixture
def server_store(tmp_path, monkeypatch):
    import server
    monkeypatch.setattr(server, 'MEMORY_DIR', tmp_path)
    monkeypatch.setattr(server.Store, 'embed', lambda self, texts: [[1.0, 0.0, 0.5] for _ in texts])
    store = server.Store()
    store.session_start('conversation-test', project='p')
    monkeypatch.setattr(server, 'store', store)
    monkeypatch.setattr(server, 'recall', server.Recall(store))
    monkeypatch.setattr(server, 'SID', 'conversation-test')
    yield server, store
    store.db.close()


@pytest.mark.parametrize('tool', ['memory_save', 'memory_save_fast'])
def test_conversation_contract_preserves_code_and_privacy(server_store, tool):
    server, store = server_store
    text = ('user: The launch owner is Morgan.\nassistant: Traceback (most recent call last):\n'
            '  File "app.py", line 1\nValueError: bad input\n<private>confidential</private>')
    response = asyncio.run(server.call_tool(tool, {'content': text, 'type': 'fact', 'project': 'p', 'source_format': 'conversation'}))
    payload = json.loads(response[0].text)
    row = store.db.execute('SELECT * FROM knowledge WHERE id=?', (payload['id'],)).fetchone()
    assert 'launch owner is Morgan' in row['content']
    assert 'Traceback' in row['content'] and 'confidential' not in row['content']
    assert row['source_format'] == 'conversation'
    assert store.db.execute('SELECT embedding_space FROM embeddings WHERE knowledge_id=?', (row['id'],)).fetchone()[0] == 'text'
    schema = next(t for t in asyncio.run(server.list_tools()) if t.name == tool).input_schema
    assert 'conversation' in schema['properties']['source_format']['enum']


def test_invalid_source_format_and_conflicting_filter_write_nothing(server_store):
    _, store = server_store
    before = store.db.execute('SELECT count(*) FROM knowledge').fetchone()[0]
    for kwargs in ({'source_format': 'invalid'}, {'source_format': 'conversation', 'filter_name': 'generic_logs'}):
        with pytest.raises(ValueError):
            store.save_knowledge('conversation-test', 'text', 'fact', **kwargs)
    assert store.db.execute('SELECT count(*) FROM knowledge').fetchone()[0] == before


def test_explicit_conversation_handles_two_turns_and_long_first_turn():
    for text in ('user: hello\nassistant: Traceback (most recent call last):',
                 'user: ' + 'x' * 10000 + '\nassistant: SELECT key FROM records WHERE key=1'):
        assert classify(text, source_format='conversation').type == 'text'


def test_update_preserves_format_and_removes_old_passages(server_store, monkeypatch):
    server, store = server_store
    from memory_core.passage_index import PassageIndex
    kid, *_ = store.save_knowledge('conversation-test', 'user: Morgan owns launch.\nassistant: Noted.', 'fact', project='p', source_format='conversation', skip_quality=True)
    row = dict(store.db.execute('SELECT * FROM knowledge WHERE id=?', (kid,)).fetchone())
    PassageIndex(store.db, lambda texts: [[1.0, 0.5] for _ in texts], 'test').ensure([row])
    monkeypatch.setattr(server.recall, 'search', lambda *args, **kwargs: {'results': {'fact': [{'id': kid}]}})
    asyncio.run(server.call_tool('memory_update', {'find': 'Morgan', 'new_content': 'user: Riley owns launch.\nassistant: Traceback (most recent call last):'}))
    updated = store.db.execute("SELECT * FROM knowledge WHERE status='active' ORDER BY id DESC LIMIT 1").fetchone()
    assert updated['source_format'] == 'conversation' and 'Riley owns launch' in updated['content']
    assert store.db.execute('SELECT count(*) FROM evidence_passages WHERE knowledge_id=?', (kid,)).fetchone()[0] == 0


def test_index_cursor_and_evidence_mcp_are_operational(server_store, monkeypatch):
    server, store = server_store
    for name in ('Morgan', 'Riley'):
        store.save_knowledge('conversation-test', f'user: {name} owns a launch.\nassistant: Noted.', 'fact', project='p', source_format='conversation', skip_quality=True, skip_dedup=True)
    monkeypatch.setattr(store.evidence_embedder, 'embed_texts', lambda texts: [[1.0, 0.5] for _ in texts])
    first = json.loads(asyncio.run(server.call_tool('memory_index_passages', {'project': 'p', 'limit': 1}))[0].text)
    assert first['processed'] == first['remaining'] == 1
    second = json.loads(asyncio.run(server.call_tool('memory_index_passages', {'project': 'p', 'after_id': first['next_after_id']}))[0].text)
    assert second['remaining'] == 0
    records = [dict(r) for r in store.db.execute('SELECT * FROM knowledge')]
    monkeypatch.setattr(server.recall, 'search', lambda *args, **kwargs: {'results': {'fact': records}})
    response = json.loads(asyncio.run(server.call_tool('memory_recall', {'query': 'launch', 'project': 'p', 'mode': 'evidence', 'evidence_followup': False}))[0].text)
    assert response['mode'] == 'evidence' and response['search_calls'] == 1
    assert response['results'] and 'Morgan' in response['context']
    assert all(hit['project'] == 'p' and hit['passages'] for hit in response['results'])
