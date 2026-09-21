import sqlite3
from pathlib import Path

import pytest

from memory_core.evidence_pack import pack_evidence_records
from memory_core.evidence_search import EvidenceSearch
from memory_core.grounding import MissingRelation
from memory_core.passage_index import PassageIndex, split_passages
from memory_core.retrieval import SearchScope


def embed(texts):
    return [[float('golden retriever' in text.lower() or 'breed' in text.lower()),
             float('morgan' in text.lower()), 0.1] for text in texts]


@pytest.fixture(params=[False, True])
def db(request):
    connection = sqlite3.connect(':memory:')
    connection.row_factory = sqlite3.Row
    connection.execute(f'PRAGMA foreign_keys={int(request.param)}')
    connection.executescript("CREATE TABLE knowledge(id INTEGER PRIMARY KEY,content TEXT,project TEXT DEFAULT 'p',"
        "type TEXT DEFAULT 'fact',branch TEXT DEFAULT '',status TEXT DEFAULT 'active',created_at TEXT,session_id TEXT,tags TEXT DEFAULT '[]');")
    connection.executescript((Path(__file__).parents[1] / 'migrations/033_evidence_passages.sql').read_text())
    yield connection
    connection.close()


def save(db, content, project='p', branch=''):
    cursor = db.execute('INSERT INTO knowledge(content,project,branch) VALUES(?,?,?)', (content, project, branch))
    db.commit()
    return dict(db.execute('SELECT * FROM knowledge WHERE id=?', (cursor.lastrowid,)).fetchone())


def test_split_preserves_offsets_roles_and_every_character():
    text = '[2026-01-02]\nuser: ' + 'x' * 2000 + '\nassistant: Morgan owns it.\nuser: Thanks.'
    passages = split_passages(text)
    assert ''.join(p.content for p in passages) == text
    assert all(text[p.start:p.end] == p.content for p in passages)
    assert any(p.speaker == 'assistant' and 'Morgan' in p.content for p in passages)
    assert max(len(p.content) for p in passages) <= 900


def test_index_reuses_embeddings_and_invalidates_on_model_or_content(db):
    record = save(db, 'user: Golden Retriever Max.\nassistant: Morgan knows Max.')
    calls = []
    def encoder(texts):
        calls.append(texts)
        return embed(texts)
    index = PassageIndex(db, encoder, 'first')
    index.ensure([record])
    count = len(calls)
    index.ensure([record])
    assert len(calls) == count
    PassageIndex(db, encoder, 'second').ensure([record])
    assert len(calls) > count
    db.execute("UPDATE knowledge SET content='new fact' WHERE id=?", (record['id'],))
    assert db.execute('SELECT count(*) FROM evidence_passages').fetchone()[0] == 0
    assert db.execute("SELECT count(*) FROM evidence_passages_fts WHERE evidence_passages_fts MATCH 'Retriever'").fetchone()[0] == 0


@pytest.mark.parametrize('action', ["DELETE FROM knowledge", "UPDATE knowledge SET status='superseded'", "UPDATE knowledge SET project='other'", "UPDATE knowledge SET branch='other'"])
def test_source_lifecycle_removes_cached_passages(db, action):
    record = save(db, 'Golden Retriever')
    PassageIndex(db, embed, 'test').ensure([record])
    db.execute(action)
    assert db.execute('SELECT count(*) FROM passage_sources').fetchone()[0] == 0
    assert db.execute('SELECT count(*) FROM evidence_passages').fetchone()[0] == 0
    assert db.execute("SELECT count(*) FROM evidence_passages_fts WHERE evidence_passages_fts MATCH 'Retriever'").fetchone()[0] == 0


def test_index_build_does_not_commit_callers_transaction(db):
    record = save(db, 'Golden Retriever')
    db.execute('BEGIN')
    PassageIndex(db, embed, 'test').ensure([record])
    assert db.in_transaction
    db.rollback()
    assert db.execute('SELECT count(*) FROM passage_sources').fetchone()[0] == 0


def test_failed_embedding_keeps_previous_index(db):
    record = save(db, 'Golden Retriever')
    PassageIndex(db, embed, 'first').ensure([record])
    with pytest.raises(ValueError, match='incomplete'):
        PassageIndex(db, lambda texts: [], 'second').ensure([record])
    assert db.execute('SELECT model FROM passage_sources').fetchone()[0].startswith('first:')


def test_semantic_passage_retains_middle_fact_and_scope(db):
    text = 'user: Weather is pleasant.\n' * 100 + 'user: Max is a Golden Retriever.\n' + 'assistant: Garden notes.\n' * 100
    valid = save(db, text)
    hidden = save(db, 'Golden Retriever confidential', project='other')
    branch = save(db, 'Golden Retriever secret', branch='other')
    service = EvidenceSearch(db, lambda query, limit: [valid, hidden, branch], embed, 'test', neighbor_radius=0)
    result = service.run('What breed?', SearchScope('p', branch='main'), limit=1, max_bytes=2000, followup=False)
    assert 'Golden Retriever' in result.context
    assert len(result.context.encode()) <= 2000
    assert [r['id'] for r in result.evidence] == [valid['id']]
    assert 'confidential' not in result.context and 'secret' not in result.context
    assert result.evidence[0]['passages']


def test_followup_is_bounded_and_can_discover_missing_relation(db):
    first = save(db, 'Morgan coordinates the launch.')
    second = save(db, 'Morgan approved the launch date: April 7.')
    calls = []
    def search(query, limit):
        calls.append(query)
        return [first] if len(calls) == 1 else [first, second]
    result = EvidenceSearch(db, search, embed, 'test', neighbor_radius=0).run(
        'When is Morgan launch date?', SearchScope('p'), limit=2,
        missing_relation=MissingRelation('Morgan', 'launch date'))
    assert len(calls) == result.search_calls == 2
    assert calls[0] != calls[1] and 'Morgan' in calls[1]
    assert 'April 7' in result.context
    assert result.followup.missing_terms == ('launch date',)
    assert result.followup.reason == 'missing_relation'


def test_weighted_budget_favors_useful_source_without_losing_others():
    records = [{'id': 1, 'content': 'alpha ' * 1000, 'evidence_weight': 4.0},
               {'id': 2, 'content': 'beta ' * 1000, 'evidence_weight': 1.0}]
    packed = pack_evidence_records(records, max_bytes=2000)
    assert len(packed[0]['content']) > 3 * len(packed[1]['content'])
    assert [p['id'] for p in packed] == [1, 2]
    with pytest.raises(ValueError, match='positive'):
        pack_evidence_records([{**records[0], 'evidence_weight': float('nan')}])


def test_short_records_keep_parent_ranking_without_reembedding_or_neighbors(db):
    first = save(db, 'Morgan owns launch.')
    second = save(db, 'Golden Retriever unrelated.')
    def unexpected_encoder(texts):
        raise AssertionError('Short records already have a first-level index')
    result = EvidenceSearch(db, lambda query, limit: [first], unexpected_encoder, 'test').run(
        'launch', SearchScope('p'), followup=False)
    assert [hit['id'] for hit in result.evidence] == [first['id']]
    assert second['id'] not in [hit['id'] for hit in result.evidence]


def test_focused_evidence_is_not_discarded_when_first_pass_fills_limit(db):
    first = save(db, 'Morgan coordinates launch.')
    noise = save(db, 'Morgan takes meeting notes.')
    answer = save(db, 'Morgan approved April 7 as the launch date.')
    calls = []
    def search(query, limit):
        calls.append(query)
        return [first, noise] if len(calls) == 1 else [answer, first]
    result = EvidenceSearch(db, search, embed, 'test').run('When is Morgan launch date?', SearchScope('p'), limit=2,
                                                       missing_relation=MissingRelation('Morgan', 'launch date'))
    assert [hit['id'] for hit in result.evidence] == [first['id'], answer['id']]
    assert 'April 7' in result.context
