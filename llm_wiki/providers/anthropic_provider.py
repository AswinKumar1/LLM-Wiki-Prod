"""
Anthropic provider — Claude Haiku, Sonnet, Opus.

Requires: ANTHROPIC_API_KEY environment variable or api_key in config.yaml
Models:   claude-haiku-4-5-20251001    (fastest, cheapest — good wiki default)
          claude-sonnet-4-6             (balanced)
          claude-opus-4-6               (most capable)
"""

from __future__ import annotations

import json
import os
import urllib.request
import urllib.error
from typing import Iterator, Optional

from .base import LLMProvider, LLMResponse, ProviderConfig

_DEFAULT_BASE_URL = "https://api.anthropic.com"
_DEFAULT_MODEL = "claude-haiku-4-5-20251001"
_ANTHROPIC_VERSION = "2023-06-01"


class AnthropicProvider(LLMProvider):
    """
    Adapter for the Anthropic Messages API.
    """

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self._base_url = (config.base_url or _DEFAULT_BASE_URL).rstrip("/")
        if not config.model:
            config.model = _DEFAULT_MODEL
        self._api_key = config.api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not self._api_key:
            raise ValueError(
                "Anthropic provider requires an API key. "
                "Set ANTHROPIC_API_KEY env var or add api_key to config.yaml."
            )

    @property
    def provider_name(self) -> str:
        return "anthropic"

    def _headers(self) -> dict:
        return {
            "Content-Type": "application/json",
            "x-api-key": self._api_key,
            "anthropic-version": _ANTHROPIC_VERSION,
        }

    def _post(self, endpoint: str, payload: dict) -> dict:
        url = f"{self._base_url}{endpoint}"
        data = json.dumps(payload).encode()
        req = urllib.request.Request(url, data=data, headers=self._headers(), method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            body = exc.read().decode()
            raise RuntimeError(f"Anthropic API error {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise ConnectionError(f"Cannot reach Anthropic API: {exc.reason}") from exc

    def chat(
        self,
        system: str,
        user: str,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        payload: dict = {
            "model": self.config.model,
            "max_tokens": max_tokens or self.config.max_tokens,
            "messages": [{"role": "user", "content": user}],
            "temperature": temperature if temperature is not None else self.config.temperature,
        }
        if system:
            payload["system"] = system

        result = self._post("/v1/messages", payload)
        content = result["content"][0]["text"]
        usage = result.get("usage", {})

        return LLMResponse(
            content=content,
            model=self.config.model,
            provider=self.provider_name,
            prompt_tokens=usage.get("input_tokens", 0),
            completion_tokens=usage.get("output_tokens", 0),
        )

    def stream(
        self,
        system: str,
        user: str,
        *,
        temperature: Optional[float] = None,
    ) -> Iterator[str]:
        payload: dict = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "messages": [{"role": "user", "content": user}],
            "temperature": temperature if temperature is not None else self.config.temperature,
            "stream": True,
        }
        if system:
            payload["system"] = system

        url = f"{self._base_url}/v1/messages"
        data = json.dumps(payload).encode()
        req = urllib.request.Request(url, data=data, headers=self._headers(), method="POST")
        with urllib.request.urlopen(req, timeout=self.config.timeout) as resp:
            for raw_line in resp:
                line = raw_line.decode().strip()
                if not line.startswith("data: "):
                    continue
                payload_str = line[6:]
                try:
                    event = json.loads(payload_str)
                    if event.get("type") == "content_block_delta":
                        delta = event.get("delta", {}).get("text", "")
                        if delta:
                            yield delta
                except json.JSONDecodeError:
                    continue

    def health_check(self) -> bool:
        try:
            # Minimal ping: send a tiny message
            self.chat("", "ping", max_tokens=5)
            return True
        except Exception:
            return False
