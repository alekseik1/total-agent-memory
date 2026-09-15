import asyncio
import json

import pytest

from ai_layer.atomic_fact_extractor import parse_facts
from ai_layer.grounded_answer import GroundedAnswerService
from ai_layer.grounded_reader import GroundedReader, parse_draft
from memory_core.atomic_facts import InvalidExtraction, Source
from memory_core.embeddings import EmbeddingProvider
from memory_core.grounding import InvalidGrounding, MissingRelation
from memory_core.retrieval import SearchScope


def supported(kid, quote, subject='Dave'):
    return {'status': 'supported', 'answer': 'California Love', 'missing': None,
            'claims': [{'subject': subject, 'event': 'childhood road trip song', 'time': '', 'modality': 'asserted',
                        'citations': [{'source_id': kid, 'quote': quote}]}]}


class Provider:
    def __init__(self, responses, hook=None):
        self.responses, self.hook, self.calls = list(responses), hook, []

    def complete(self, prompt, **kwargs):
        self.calls.append(prompt)
        if self.hook:
            self.hook()
        response = self.responses.pop(0)
        return response if isinstance(response, str) else json.dumps(response)


@pytest.fixture
def store(tmp_path, monkeypatch):
    import server
    monkeypatch.setattr(server, 'MEMORY_DIR', tmp_path)
    monkeypatch.setattr(server.Store, 'embed', lambda self, texts: [[1.0] + [0.0] * 383 for _ in texts])
    current = server.Store()
    current.session_start('grounded', project='p')
    monkeypatch.setattr(server, 'store', current)
    monkeypatch.setattr(server, 'recall', server.Recall(current))
    yield current
    current.db.close()


def save(store, text, project='p'):
    kid, *_ = store.save_knowledge('grounded', text, 'fact', project=project, source_format='conversation',
                                   skip_quality=True, skip_dedup=True)
    return dict(store.db.execute('SELECT * FROM knowledge WHERE id=?', (kid,)).fetchone())


def test_atomic_subject_cannot_be_invented_or_swapped():
    source = Source(1, 'Calvin: I remember California Love.', 'p', 's', '')
    fact = {'subject': 'Dave', 'predicate': 'remembers', 'object': 'California Love', 'temporal_text': '',
            'sources': [{'id': 1, 'quote': source.content}]}
    with pytest.raises(InvalidExtraction, match='Subject'):
        parse_facts(json.dumps({'facts': [fact]}), source, (source,))
    fact['subject'] = 'Calvin'
    assert parse_facts(json.dumps({'facts': [fact]}), source, (source,))[0].subject == 'Calvin'


@pytest.mark.parametrize('corruption', ['subject', 'quote', 'source', 'modality', 'missing'])
def test_grounded_draft_rejects_invalid_evidence(corruption):
    quote = 'Dave remembers California Love.'
    data = supported(1, quote)
    claim = data['claims'][0]
    if corruption == 'subject':
        claim['subject'] = 'Calvin'
    elif corruption == 'quote':
        claim['citations'][0]['quote'] = 'Invented quote'
    elif corruption == 'source':
        claim['citations'][0]['source_id'] = 2
    elif corruption == 'modality':
        claim['modality'] = 'maybe'
    else:
        data['missing'] = {'subject': 'May', 'relation': 'song', 'time': ''}
    with pytest.raises(InvalidGrounding):
        parse_draft(json.dumps(data), 'Dave song?', [{'id': 1, 'content': quote}])


def test_missing_relation_rejects_incidental_entity():
    with pytest.raises(InvalidGrounding, match='subject'):
        MissingRelation('May', 'song').query('Which song does Dave remember?')
    assert MissingRelation('Dave', 'road trip song').query('Dave song?') == 'Dave road trip song'


@pytest.mark.parametrize('status', ['supported', 'inferred'])
def test_contradictory_answer_with_missing_premise_fails_closed(store, status):
    quote = 'Dave remembers California Love.'
    hit = save(store, quote)
    data = supported(hit['id'], quote)
    data.update(status=status, missing={'subject': 'Dave', 'relation': 'road trip song', 'time': ''})
    provider = Provider([data, data])
    result = GroundedAnswerService(store.db, lambda query, limit: [hit], GroundedReader(provider, None)).answer(
        'Which road trip song does Dave remember?', SearchScope('p'))
    assert result.answer == 'Not enough information'
    assert result.draft.rejection == 'Answer claims support while declaring a missing premise'
    assert result.verification.supported is False
    assert len(provider.calls) == 2


def test_malformed_missing_relation_remains_a_protocol_error():
    quote = 'Dave remembers California Love.'
    data = supported(1, quote)
    data['missing'] = ['Dave', 'song']
    provider = Provider([data, data])
    with pytest.raises(InvalidGrounding, match='Reader returned invalid grounded evidence'):
        GroundedReader(provider, None).read('Dave song?', [{'id': 1, 'content': quote}])
    assert len(provider.calls) == 2


def test_speaker_outside_quote_binds_first_person_without_name_invention():
    content = '[7 August 2023] Evan: I got this because it symbolizes strength.'
    data = supported(1, 'I got this because it symbolizes strength.', subject='Evan')
    assert parse_draft(json.dumps(data), 'Evan tree?', [{'id': 1, 'content': content}]).claims[0].subject == 'Evan'
    data['claims'][0]['subject'] = 'Dave'
    with pytest.raises(InvalidGrounding, match='subject'):
        parse_draft(json.dumps(data), 'Dave tree?', [{'id': 1, 'content': content}])


def test_structured_transport_preserves_schema_and_rejects_provider_refusal(monkeypatch):
    import llm_provider
    from ai_layer.grounded_schema import DRAFT_SCHEMA
    calls = []
    def response(url, body, headers, timeout):
        calls.append(body)
        return {'choices': [{'message': {'content': '{}'}}]}
    monkeypatch.setattr(llm_provider, '_http_post_json', response)
    provider = llm_provider.OpenAIProvider(api_key='test-key')
    assert provider.complete_structured('question', DRAFT_SCHEMA) == '{}'
    assert calls[0]['response_format']['json_schema']['strict'] is True
    monkeypatch.setattr(llm_provider, '_http_post_json', lambda *args, **kwargs: {'choices': [{'message': {'refusal': 'declined', 'content': None}}]})
    with pytest.raises(RuntimeError, match='declined'):
        provider.complete_structured('question', DRAFT_SCHEMA)


def test_answer_verified_and_verifier_rejection_is_not_hidden(store):
    row = save(store, 'Dave remembers California Love.')
    for accepted in (True, False):
        provider = Provider([supported(row['id'], row['content']), {'supported': accepted, 'reason': 'checked person and event'}])
        result = GroundedAnswerService(store.db, lambda query, limit: [row], GroundedReader(provider, 'test')).answer('Dave song?', SearchScope('p'))
        assert result.answer == ('California Love' if accepted else 'Not enough information')
        assert len(provider.calls) == 2 and result.search_calls == 1
        ref = result.draft.claims[0].citations[0]
        assert row['content'][ref.start:ref.end] == ref.quote


def test_followup_runs_once_for_a_specific_gap(store):
    first = save(store, 'Dave remembers a road trip with his father.')
    second = save(store, 'Dave remembers California Love from the road trip.')
    missing = {'status': 'partial', 'answer': 'Not enough information', 'claims': [],
               'missing': {'subject': 'Dave', 'relation': 'road trip song', 'time': ''}}
    provider = Provider([missing, supported(second['id'], second['content']), {'supported': True, 'reason': 'all premises present'}])
    calls = []
    def search(query, limit):
        calls.append(query)
        return [first] if len(calls) == 1 else [second]
    result = GroundedAnswerService(store.db, search, GroundedReader(provider, 'test')).answer('Dave song?', SearchScope('p'), limit=2)
    assert calls == ['Dave song?', 'Dave road trip song']
    assert len(provider.calls) == 3 and result.search_calls == 2
    assert result.answer == 'California Love'


def test_scope_and_source_mutation_fail_closed(store):
    secret = save(store, 'Dave remembers secret song.', project='private')
    provider = Provider([])
    result = GroundedAnswerService(store.db, lambda query, limit: [secret], GroundedReader(provider, 'test')).answer('Dave song?', SearchScope('p'))
    assert not provider.calls and result.answer == 'Not enough information'
    row = save(store, 'Dave remembers California Love.')
    provider = Provider([supported(row['id'], row['content']), {'supported': True, 'reason': 'supported'}],
                        hook=lambda: store.db.execute("UPDATE knowledge SET status='deleted' WHERE id=?", (row['id'],)))
    with pytest.raises(InvalidGrounding, match='changed'):
        GroundedAnswerService(store.db, lambda query, limit: [row], GroundedReader(provider, 'test')).answer('Dave song?', SearchScope('p'))


def test_followup_can_recover_another_passage_of_the_same_source(store, monkeypatch):
    first = 'Dave remembers a road trip with his father.'
    second = 'Dave remembers California Love from the road trip.'
    row = save(store, first + '\n' + second)
    missing = {'status': 'partial', 'answer': 'Not enough information', 'claims': [],
               'missing': {'subject': 'Dave', 'relation': 'road trip song', 'time': ''}}
    provider = Provider([missing, supported(row['id'], second), {'supported': True, 'reason': 'source supports answer'}])
    service = GroundedAnswerService(store.db, lambda query, limit: [row], GroundedReader(provider, 'test'))
    def context(hits, *, query, **kwargs):
        return [{**row, 'content': second if query == 'Dave road trip song' else first}]
    monkeypatch.setattr(service.context, 'build', context)
    result = service.answer('Dave song?', SearchScope('p'))
    assert result.answer == 'California Love'
    assert result.search_calls == 2 and len(provider.calls) == 3
    assert len(result.evidence) == 1
    assert first in result.evidence[0]['content']
    assert second in result.evidence[0]['content']


def test_invalid_verdict_cannot_be_coerced_to_true(store):
    row = save(store, 'Dave remembers California Love.')
    provider = Provider([supported(row['id'], row['content']), {'supported': 'false', 'reason': 'unsupported'}])
    with pytest.raises(InvalidGrounding, match='Verifier'):
        GroundedAnswerService(store.db, lambda query, limit: [row], GroundedReader(provider, 'test')).answer('Dave song?', SearchScope('p'))


def test_quote_repair_is_bounded_and_failed_grounding_is_reported(store):
    row = save(store, 'Dave remembers California Love.')
    bad = supported(row['id'], 'Dave remembers a fabricated song.')
    provider = Provider([bad, supported(row['id'], row['content']), {'supported': True, 'reason': 'repaired literal quote'}])
    result = GroundedAnswerService(store.db, lambda query, limit: [row], GroundedReader(provider, 'test')).answer('Dave song?', SearchScope('p'))
    assert result.answer == 'California Love' and len(provider.calls) == 3
    provider = Provider([bad, bad])
    result = GroundedAnswerService(store.db, lambda query, limit: [row], GroundedReader(provider, 'test')).answer('Dave song?', SearchScope('p'))
    assert result.answer == 'Not enough information' and not result.verification.supported
    assert result.draft.rejection and len(provider.calls) == 2


def test_mcp_answer_contract_and_configured_provider(store, monkeypatch):
    import answer_endpoint
    import server
    row = save(store, 'Dave remembers California Love.')
    provider = Provider([supported(row['id'], row['content']), {'supported': True, 'reason': 'supported'}])
    monkeypatch.setattr(answer_endpoint, 'make_provider', lambda name: provider)
    monkeypatch.setattr(server.recall, 'search', lambda *args, **kwargs: {'results': {'fact': [row]}})
    response = json.loads(asyncio.run(server.call_tool('memory_answer', {'query': 'Dave song?', 'project': 'p'}))[0].text)
    assert response['answer'] == 'California Love' and response['verification']['supported']
    schema = next(tool for tool in asyncio.run(server.list_tools()) if tool.name == 'memory_answer').input_schema
    assert schema['required'] == ['query', 'project']


def test_query_cache_is_bounded_model_aware_and_copies_results(monkeypatch):
    provider = EmbeddingProvider()
    calls = []
    monkeypatch.setattr(provider, 'embed_texts', lambda texts, **kwargs: calls.append(texts) or [[1.0, 0.5]])
    monkeypatch.setattr(provider, 'active_model', lambda space='text': 'first')
    result = provider.embed_query('query')
    result[0] = 99
    assert provider.embed_query('query') == [1.0, 0.5] and len(calls) == 1
    monkeypatch.setattr(provider, 'active_model', lambda space='text': 'second')
    provider.embed_query('query')
    assert len(calls) == 2
    for i in range(140):
        provider.embed_query(str(i))
    assert provider._cached_query.cache_info().currsize == 128


def test_worker_indexes_saved_source_and_reports_embedding_failure(store, monkeypatch):
    from enrichment_worker import EnrichmentTask, _run_passage_index
    row = save(store, 'Dave: ' + 'We discussed the road trip. ' * 60)
    task = EnrichmentTask(1, row['id'], 'grounded', 'p', 'fact', row['content'], [], 'medium', True, 0)
    monkeypatch.setenv('MEMORY_PASSAGE_INDEX_ENABLED', 'true')
    monkeypatch.setattr(store.evidence_embedder, 'embed_texts', lambda texts: [[1.0, 0.5] for _ in texts])
    _run_passage_index(store.db, task, store)
    assert store.db.execute('SELECT count(*) FROM evidence_passages').fetchone()[0] > 0
    store.db.execute("UPDATE knowledge SET content=content||' extra' WHERE id=?", (row['id'],))
    monkeypatch.setattr(store.evidence_embedder, 'embed_texts', lambda texts: [])
    with pytest.raises(ValueError, match='incomplete'):
        _run_passage_index(store.db, task, store)


def test_saved_dialogue_is_indexed_through_durable_worker_queue(store, monkeypatch):
    import enrichment_worker as worker
    monkeypatch.setenv('MEMORY_ASYNC_ENRICHMENT', 'true')
    monkeypatch.setenv('MEMORY_PASSAGE_INDEX_ENABLED', 'true')
    monkeypatch.setattr(store.evidence_embedder, 'embed_texts', lambda texts: [[1.0, 0.5] for _ in texts])
    monkeypatch.setattr(worker, '_STAGES', [('passage_index', worker._run_passage_index)])
    row = save(store, 'Dave: ' + 'We discussed the road trip. ' * 60)
    assert store.db.execute("SELECT count(*) FROM enrichment_queue WHERE knowledge_id=? AND status='pending'", (row['id'],)).fetchone()[0] == 1
    worker.run_pending(store.db, max_rows=1, store=store)
    assert store.db.execute('SELECT count(*) FROM evidence_passages WHERE knowledge_id=?', (row['id'],)).fetchone()[0] > 0
    assert store.db.execute("SELECT status FROM enrichment_queue WHERE knowledge_id=?", (row['id'],)).fetchone()[0] == 'done'
