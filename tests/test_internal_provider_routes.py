import json
import sqlite3

import pytest

import auto_episode_capture
import config
import llm_provider
import reranker
from ingestion.enricher import MetadataEnricher


@pytest.mark.parametrize('provider', ['openai', 'openai-compatible', 'anthropic', 'ollama'])
def test_internal_tasks_use_selected_provider(monkeypatch, provider):
    monkeypatch.setenv('MEMORY_LLM_ENABLED', 'true')
    monkeypatch.setenv('MEMORY_LLM_PROVIDER', provider)
    monkeypatch.setenv('MEMORY_LLM_MODEL', 'selected-model')
    monkeypatch.setenv('MEMORY_LLM_API_BASE', 'http://127.0.0.1:8080/v1')
    monkeypatch.setenv('MEMORY_LLM_API_KEY', 'test-token')
    calls = []
    narrative = json.dumps({'narrative': 'Configured provider generated the episode.'})

    def post(url, body, headers, timeout):
        calls.append((url, body))
        return {'response': narrative, 'choices': [{'message': {'content': narrative}}],
                'content': [{'type': 'text', 'text': narrative}]}

    monkeypatch.setattr(llm_provider, '_http_post_json', post)
    result = auto_episode_capture.generate_narrative_ollama([{'role': 'user', 'text': 'Fixed the regression.'}], 'p')
    assert result['narrative'] == 'Configured provider generated the episode.'
    with sqlite3.connect(':memory:') as db:
        assert MetadataEnricher(db).summarize_with_ollama('Useful information. ' * 10) == narrative
    assert reranker._ollama_generate('Rank this evidence', 'legacy-model') == narrative
    assert len(calls) == 3
    assert all(body['model'] == 'selected-model' for _, body in calls[:2])
    assert calls[2][1]['model'] == ('legacy-model' if provider == 'ollama' else 'selected-model')
    if provider != 'ollama':
        assert all('/api/generate' not in url for url, _ in calls)


def test_disabled_llm_never_runs_internal_generation(monkeypatch):
    monkeypatch.setenv('MEMORY_LLM_ENABLED', 'false')

    def forbidden(*args, **kwargs):
        pytest.fail('Disabled internal task attempted network access')

    monkeypatch.setattr(llm_provider, '_http_post_json', forbidden)
    assert auto_episode_capture.generate_narrative_ollama([], 'p') is None
    with sqlite3.connect(':memory:') as db:
        assert MetadataEnricher(db).summarize_with_ollama('Useful information. ' * 10) is None
    assert reranker._ollama_generate('Rank', 'unused') is None


@pytest.mark.parametrize('value', ['0', '-1', 'invalid'])
def test_invalid_thread_budget_is_rejected(monkeypatch, value):
    monkeypatch.setenv('MEMORY_EMBED_THREADS', value)
    with pytest.raises(ValueError):
        config.get_embed_threads()


def test_empty_compose_model_uses_provider_default(monkeypatch):
    monkeypatch.setenv('MEMORY_LLM_MODEL', '')
    assert config.get_llm_model() == config.get_llm_model_for_provider('ollama')
    assert config.get_llm_model()
