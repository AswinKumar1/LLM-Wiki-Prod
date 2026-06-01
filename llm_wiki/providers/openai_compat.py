"""
OpenAI-compatible provider — covers any server that speaks the OpenAI
Chat Completions API format:

  - Hermes 3 via Ollama (use OllamaProvider instead for native support)
  - LM Studio         base_url: http://localhost:1234/v1
  - vLLM              base_url: http://localhost:8000/v1
  - Together AI       base_url: https://api.together.xyz/v1
  - Groq              base_url: https://api.groq.com/openai/v1
  - Fireworks AI      base_url: https://api.fireworks.ai/inference/v1
  - Any local server that exposes /v1/chat/completions

Config example (config.yaml):
  provider: openai_compat
  base_url: http://localhost:1234/v1
  model: hermes-3-llama-3.1-8b
  api_key: "lm-studio"   # LM Studio accepts any non-empty key
"""

from __future__ import annotations

import os
from .base import ProviderConfig
from .openai_provider import OpenAIProvider


class OpenAICompatProvider(OpenAIProvider):
    """
    Thin subclass of OpenAIProvider that changes the default base_url
    and relaxes the API key requirement (many local servers don't need one).
    """

    def __init__(self, config: ProviderConfig):
        # Local servers often don't require a real key
        if not config.api_key:
            config.api_key = os.environ.get("OPENAI_COMPAT_API_KEY", "local")
        super().__init__(config)

    @property
    def provider_name(self) -> str:
        return "openai_compat"

    def health_check(self) -> bool:
        # Many local servers don't expose /models — just do a cheap chat ping
        try:
            self.chat("", "ping", max_tokens=5)
            return True
        except Exception:
            return False
