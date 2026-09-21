import base64
from dataclasses import dataclass
from pathlib import Path

import config
import llm_provider

MAX_IMAGE_BYTES = 6 * 1024 * 1024
VISION_MAX_TOKENS = 500
VISION_TEMPERATURE = 0.3
VISION_PROMPT = "Describe this image in detail. Transcribe visible code and describe the structure of diagrams."


@dataclass(frozen=True)
class ImageInput:
    media_type: str
    base64_data: str

    @classmethod
    def read(cls, path: Path) -> "ImageInput":
        with path.open('rb') as source:
            data = source.read(MAX_IMAGE_BYTES + 1)
        if len(data) > MAX_IMAGE_BYTES:
            raise ValueError("Vision image exceeds configured transport size limit")
        if data.startswith(b'\x89PNG\r\n\x1a\n'):
            media_type = 'image/png'
        elif data.startswith(b'\xff\xd8\xff'):
            media_type = 'image/jpeg'
        elif data.startswith((b'GIF87a', b'GIF89a')):
            media_type = 'image/gif'
        elif data.startswith(b'RIFF') and data[8:12] == b'WEBP':
            media_type = 'image/webp'
        else:
            raise ValueError("Vision supports PNG, JPEG, GIF and WebP; convert other formats first")
        return cls(media_type, base64.b64encode(data).decode('ascii'))


def describe(image: ImageInput, model: str) -> str:
    provider = llm_provider.make_provider(config.get_llm_provider(), model=model)
    timeout = config.get_llm_timeout_sec()
    if isinstance(provider, llm_provider.OllamaProvider):
        data = llm_provider._http_post_json(
            f'{provider.api_base}/api/generate',
            body={'model': model, 'prompt': VISION_PROMPT, 'images': [image.base64_data], 'stream': False,
                  'options': {'num_predict': VISION_MAX_TOKENS, 'temperature': VISION_TEMPERATURE}},
            headers={}, timeout=timeout)
        result = data['response']
    elif isinstance(provider, llm_provider.OpenAIProvider):
        data = llm_provider._http_post_json(
            f'{provider.api_base}/chat/completions',
            body={'model': model, 'max_tokens': VISION_MAX_TOKENS, 'temperature': VISION_TEMPERATURE,
                  'messages': [{'role': 'user', 'content': [
                      {'type': 'text', 'text': VISION_PROMPT},
                      {'type': 'image_url', 'image_url': {'url': f'data:{image.media_type};base64,{image.base64_data}'}}]}]},
            headers=provider._authorization_headers(), timeout=timeout)
        result = data['choices'][0]['message']['content']
    elif isinstance(provider, llm_provider.AnthropicProvider):
        if not provider.api_key:
            raise ValueError('Anthropic vision requires an API key')
        data = llm_provider._http_post_json(
            f'{provider.api_base}/messages',
            body={'model': model, 'max_tokens': VISION_MAX_TOKENS, 'temperature': VISION_TEMPERATURE,
                  'messages': [{'role': 'user', 'content': [
                      {'type': 'image', 'source': {'type': 'base64', 'media_type': image.media_type, 'data': image.base64_data}},
                      {'type': 'text', 'text': VISION_PROMPT}]}]},
            headers={'x-api-key': provider.api_key, 'anthropic-version': provider.API_VERSION}, timeout=timeout)
        result = '\n'.join(block['text'] for block in data['content'] if block.get('type') == 'text')
    else:
        raise TypeError('Selected provider does not implement image descriptions')
    if not isinstance(result, str) or not result.strip():
        raise ValueError('Vision provider returned no text')
    return result.strip()
