import email.message
import io
import json
import urllib.error

import pytest

from ai_layer.jev_client import (
    BACKOFF_JITTER,
    MAX_RETRY_AFTER_SECONDS,
    JevClient,
    JevConfigError,
    JevError,
    backoff_seconds,
)
from ai_layer.negative_evidence import JEV_CRITERIA, JevContradictionScorer
from memory_core.telemetry import counters

KEY = 'ts_secret_key_value'


class Response:
    def __init__(self, body):
        self.body = json.dumps(body).encode()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self.body


def http_error(status, body, headers=None):
    message = email.message.Message()
    for name, value in (headers or {}).items():
        message[name] = value
    raw = body if isinstance(body, bytes) else json.dumps(body).encode()
    return urllib.error.HTTPError('https://api.typesafe.ai/v1/systemone', status, 'error', message, io.BytesIO(raw))


class Opener:
    def __init__(self, outcomes):
        self.outcomes, self.requests = list(outcomes), []

    def __call__(self, request, timeout, context):
        self.requests.append((request, timeout))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return Response(outcome)


def client(outcomes, sleeps=None, **kwargs):
    opener = Opener(outcomes)
    recorded = sleeps if sleeps is not None else []
    return JevClient(KEY, 'https://api.typesafe.ai/', 'jev-latest', sleep=recorded.append, opener=opener, **kwargs), opener


OK = {'model': 'jev-1.13.0', 'answers': {'q': {'type': 'noul', 'noul': 0.9}}, 'usage': {'input_tokens': 10, 'output_tokens': 2}}


@pytest.mark.parametrize('bad', ['', '   ', None, 'ts_a\nbc', 'ts_\x00abc'])
def test_invalid_keys_are_rejected_without_echoing_the_key(bad):
    with pytest.raises(JevConfigError) as caught:
        JevClient(bad, 'https://api.typesafe.ai', 'jev-latest')
    assert 'ts_' not in str(caught.value)


def test_surrounding_whitespace_is_stripped_and_key_never_in_repr():
    jev, opener = client([OK])
    assert KEY not in repr(jev)
    JevClient(f'  {KEY}\n', 'https://api.typesafe.ai', 'jev-latest', opener=opener)


def test_request_shape_answers_and_token_counters():
    before = counters.get('jev_input_tokens')
    jev, opener = client([OK])
    answers = jev.systemone({'question': 'x'}, {'q': {'type': 'noul', 'instructions': 'Маша?'}})
    assert answers == {'q': {'type': 'noul', 'noul': 0.9}}
    request, timeout = opener.requests[0]
    assert request.full_url == 'https://api.typesafe.ai/v1/systemone' and request.get_method() == 'POST'
    assert request.get_header('Authorization') == f'Bearer {KEY}' and timeout == 10.0
    assert request.get_header('X-typesafe-retry-count') is None
    assert json.loads(request.data) == {'model': 'jev-latest', 'state': {'question': 'x'},
                                        'questions': {'q': {'type': 'noul', 'instructions': 'Маша?'}}}
    assert 'Маша' in request.data.decode()
    assert counters.get('jev_input_tokens') - before == 10


def test_rate_limit_honours_retry_after_ms_and_marks_retries():
    sleeps = []
    jev, opener = client([http_error(429, {'detail': 'slow down'}, {'retry-after-ms': '1500'}), OK], sleeps)
    jev.systemone('s', {'q': {'type': 'noul'}})
    assert sleeps == [1.5]
    assert opener.requests[1][0].get_header('X-typesafe-retry-count') == '1'


def test_server_errors_back_off_and_give_up_after_max_retries():
    sleeps = []
    jev, opener = client([http_error(503, {'detail': 'busy'}) for _ in range(3)], sleeps)
    with pytest.raises(JevError) as caught:
        jev.systemone('s', {'q': {'type': 'noul'}})
    assert caught.value.status == 503 and 'busy' in str(caught.value)
    assert len(opener.requests) == 3 and len(sleeps) == 2
    assert 0.5 * (1 - BACKOFF_JITTER) <= sleeps[0] <= 0.5 and 1.0 * (1 - BACKOFF_JITTER) <= sleeps[1] <= 1.0


@pytest.mark.parametrize(('status', 'body', 'expected'), [
    (400, {'detail': 'Noul question must have criteria or instructions: q'}, 'must have criteria'),
    (401, {'detail': {'error_type': 'authentication_error', 'message': 'Invalid API key'}}, 'Invalid API key'),
    (422, {'detail': [{'type': 'missing', 'loc': ['body', 'state'], 'msg': 'Field required'}]}, 'Field required'),
    (403, b'not json', 'not json'),
])
def test_client_errors_are_not_retried_and_keep_the_server_message(status, body, expected):
    sleeps = []
    jev, opener = client([http_error(status, body, {'x-typesafe-request-id': 'req_1'})], sleeps)
    with pytest.raises(JevError) as caught:
        jev.systemone('s', {'q': {'type': 'noul'}})
    assert caught.value.status == status and caught.value.request_id == 'req_1'
    assert expected in str(caught.value) and KEY not in str(caught.value)
    assert sleeps == [] and len(opener.requests) == 1


def test_connection_errors_are_retried_then_reported_without_details():
    sleeps = []
    jev, _ = client([urllib.error.URLError('refused'), TimeoutError(), OK], sleeps)
    assert jev.systemone('s', {'q': {'type': 'noul'}})['q']['noul'] == 0.9
    jev, _ = client([urllib.error.URLError('refused') for _ in range(3)], [])
    with pytest.raises(JevError, match='unreachable: URLError'):
        jev.systemone('s', {'q': {'type': 'noul'}})


def test_invalid_payloads_and_arguments_fail_clearly():
    jev, _ = client([{'model': 'jev-1.13.0'}])
    with pytest.raises(JevError, match='no answers map'):
        jev.systemone('s', {'q': {'type': 'noul'}})
    with pytest.raises(ValueError, match='At least one question'):
        jev.systemone('s', {})
    with pytest.raises(JevConfigError):
        JevClient(KEY, 'https://api.typesafe.ai', 'jev-latest', timeout=0)


def test_backoff_uses_server_delay_until_the_cap():
    assert backoff_seconds(1, 2.0) == 2.0
    assert backoff_seconds(1, MAX_RETRY_AFTER_SECONDS + 1, rand=lambda: 0.0) == 0.5
    assert backoff_seconds(5, None, rand=lambda: 0.0) == 5.0


def test_retry_after_http_date_is_parsed():
    sleeps = []
    jev, _ = client([http_error(529, {'detail': 'overloaded'}, {'Retry-After': 'Wed, 21 Oct 2015 07:28:00 GMT'}), OK], sleeps)
    jev.systemone('s', {'q': {'type': 'noul'}})
    assert sleeps == [0.0]


def test_jev_scorer_sends_one_noul_question_per_pair_with_the_question():
    jev, opener = client([{'answers': {'p1': {'type': 'noul', 'noul': 0.87}, 'p2': {'type': 'noul', 'noul': 0.15}}}])
    scores = JevContradictionScorer(jev)([('Маша любит красный.', 'Маше нравится зелёный.'), ('Фёдор: бордовый', 'Фёдор: белый')],
                                         question='Какой цвет любит Маша?')
    assert scores == [0.87, 0.15]
    body = json.loads(opener.requests[0][0].data)
    assert body['questions']['p1'] == {'type': 'noul', 'criteria': JEV_CRITERIA, 'instructions': {
        'question': 'Какой цвет любит Маша?', 'fact_a': 'Маша любит красный.', 'fact_b': 'Маше нравится зелёный.'}}
    unscoped, unscoped_opener = client([{'answers': {'p1': {'type': 'noul', 'noul': 1}}}])
    assert JevContradictionScorer(unscoped)([('a', 'b')]) == [1.0]
    assert 'question' not in json.loads(unscoped_opener.requests[0][0].data)['questions']['p1']['instructions']


def test_jev_scorer_rejects_missing_answers_and_skips_empty_input():
    jev, opener = client([{'answers': {'p1': {'type': 'noul', 'noul': True}}}])
    with pytest.raises(TypeError, match='pair 1'):
        JevContradictionScorer(jev)([('a', 'b')])
    assert JevContradictionScorer(jev)([]) == [] and len(opener.requests) == 1


def test_answer_endpoint_selects_the_configured_scorer(monkeypatch):
    import answer_endpoint
    from ai_layer.negative_evidence import LLMContradictionScorer
    monkeypatch.delenv('MEMORY_CONTRADICTION_SCORER', raising=False)
    assert isinstance(answer_endpoint.contradiction_scorer(object(), 'm'), LLMContradictionScorer)
    monkeypatch.setenv('MEMORY_CONTRADICTION_SCORER', 'jev')
    monkeypatch.setenv('TYPESAFE_API_KEY', KEY)
    monkeypatch.setenv('TYPESAFE_DEFAULT_MODEL', 'jev-preview')
    scorer = answer_endpoint.contradiction_scorer(object(), 'm')
    assert isinstance(scorer, JevContradictionScorer) and scorer.client.model == 'jev-preview'
    monkeypatch.delenv('TYPESAFE_API_KEY')
    with pytest.raises(JevConfigError, match='TYPESAFE_API_KEY'):
        answer_endpoint.contradiction_scorer(object(), 'm')
