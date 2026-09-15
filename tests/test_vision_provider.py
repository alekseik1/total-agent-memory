import base64

import pytest

import llm_provider
from ingestion.ocr import OCREngine
from ingestion.vision_provider import ImageInput

PNG = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aBYQAAAAASUVORK5CYII=')


@pytest.mark.parametrize('provider', ['ollama', 'openai', 'openai-compatible', 'anthropic'])
def test_ocr_vision_routes_selected_provider(monkeypatch, tmp_path, provider):
    monkeypatch.setenv('MEMORY_LLM_ENABLED', 'true')
    monkeypatch.setenv('MEMORY_LLM_PROVIDER', provider)
    monkeypatch.setenv('MEMORY_LLM_API_BASE', 'http://127.0.0.1:8080/v1')
    monkeypatch.setenv('MEMORY_LLM_MODEL', 'text-model')
    monkeypatch.setenv('MEMORY_VISION_MODEL', 'vision-model')
    monkeypatch.setenv('MEMORY_LLM_API_KEY', 'test-key')
    captured = []

    def post(url, body, headers, timeout):
        captured.append((url, body, headers))
        return {'response': 'image description', 'choices': [{'message': {'content': 'image description'}}],
                'content': [{'type': 'text', 'text': 'image description'}]}

    monkeypatch.setattr(llm_provider, '_http_post_json', post)
    monkeypatch.setattr(OCREngine, '_detect_method', lambda _: 'none')
    path = tmp_path / 'image.png'
    path.write_bytes(PNG)
    assert OCREngine().describe_image(str(path)) == 'image description'
    assert captured[0][1]['model'] == 'vision-model'
    if provider == 'ollama':
        assert base64.b64decode(captured[0][1]['images'][0]) == PNG
    elif provider == 'anthropic':
        assert captured[0][1]['messages'][0]['content'][0]['source']['media_type'] == 'image/png'
    else:
        assert captured[0][1]['messages'][0]['content'][1]['image_url']['url'].startswith('data:image/png;base64,')


def test_disabled_vision_does_not_read_file_or_call_provider(monkeypatch):
    monkeypatch.setenv('MEMORY_LLM_ENABLED', 'false')
    monkeypatch.setenv('MEMORY_VISION_MODEL', 'vision-model')
    monkeypatch.setattr(OCREngine, '_detect_method', lambda _: 'none')

    def forbidden(*args, **kwargs):
        pytest.fail('Disabled vision attempted file or network access')

    monkeypatch.setattr(ImageInput, 'read', forbidden)
    monkeypatch.setattr(llm_provider, '_http_post_json', forbidden)
    assert OCREngine().describe_image('unused.png') is None


def test_image_type_comes_from_bytes_not_filename(tmp_path):
    path = tmp_path / 'pretend.png'
    path.write_text('not an image')
    with pytest.raises(ValueError):
        ImageInput.read(path)
