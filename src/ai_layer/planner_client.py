from __future__ import annotations

import config
from llm_provider import make_provider


class PlannerClient:
    def complete(self, *, model: str, system: str, user: str,
                 max_tokens: int = 256, temperature: float = 0.0) -> str:
        provider = make_provider(config.get_llm_provider())
        selected = config.get_llm_model_for_provider() if model == "configured" else model
        return provider.complete(f"{system}\n\n{user}", model=selected,
                                 max_tokens=max_tokens, temperature=temperature, timeout=config.get_llm_timeout_sec())
