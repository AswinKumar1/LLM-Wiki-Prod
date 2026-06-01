"""
OpenAI provider — GPT-4o, GPT-4o-mini, o1-mini, Codex, etc.

Requires: OPENAI_API_KEY environment variable or api_key in config.yaml
Models:   gpt-4o-mini  (fast, cheap — good default for wiki tasks)
          gpt-4o       (best quality)
          o1-mini      (reasoning)
"""

from __future__ import annotations

import json
import os
import urllib.request
import urllib.error
from typing import Iterator, Optional

from .base import LLMProvider, LLMResponse, ProviderConfig

_DEFAULT_BASE_URL = "https://api.openai.com/v1"
_DEFAULT_MODEL = "gpt-4o-mini"


class OpenAIProvider(LLMProvider):
    """
    Adapter for the OpenAI Chat Completions API.
    Also works for any OpenAI-compatible endpoint — see OpenAICompatProvider
    for a convenience subclass that pre-fills the base_url.
    """

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self._base_url = (config.base_url or _DEFAULT_BASE_URL).rstrip("/")
        if not config.model:
            config.model = _DEFAULT_MODEL
        self._api_key = config.api_key or os.environ.get("OPENAI_API_KEY", "")
        if not self._api_key:
            raise ValueError(
                "OpenAI provider requires an API key. "
                "Set OPENAI_API_KEY env var or add api_key to config.yaml."
            )

    @property
    def provider_name(self) -> str:
        return "openai"

    def _headers(self) -> dict:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
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
            raise RuntimeError(f"OpenAI API error {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise ConnectionError(f"Cannot reach OpenAI API: {exc.reason}") from exc

    def chat(
        self,
        system: str,
        user: str,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})

        payload: dict = {
            "model": self.config.model,
            "messages": messages,
        }
        # o1 models don't support temperature or max_tokens in the same way
        if not self.config.model.startswith("o1"):
            payload["temperature"] = temperature if temperature is not None else self.config.temperature
            payload["max_tokens"] = max_tokens or self.config.max_tokens

        result = self._post("/chat/completions", payload)
        choice = result["choices"][0]["message"]["content"]
        usage = result.get("usage", {})

        return LLMResponse(
            content=choice,
            model=self.config.model,
            provider=self.provider_name,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
        )

    def stream(
        self,
        system: str,
        user: str,
        *,
        temperature: Optional[float] = None,
    ) -> Iterator[str]:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})

        payload = {
            "model": self.config.model,
            "messages": messages,
            "stream": True,
            "temperature": temperature if temperature is not None else self.config.temperature,
        }
        url = f"{self._base_url}/chat/completions"
        data = json.dumps(payload).encode()
        req = urllib.request.Request(url, data=data, headers=self._headers(), method="POST")
        with urllib.request.urlopen(req, timeout=self.config.timeout) as resp:
            for raw_line in resp:
                line = raw_line.decode().strip()
                if not line.startswith("data: "):
                    continue
                payload_str = line[6:]
                if payload_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload_str)
                    delta = chunk["choices"][0]["delta"].get("content", "")
                    if delta:
                        yield delta
                except (json.JSONDecodeError, KeyError):
                    continue

    def health_check(self) -> bool:
        try:
            result = self._post("/models", {}) if False else None
            # Lightweight check: just hit the models list endpoint via GET
            url = f"{self._base_url}/models"
            req = urllib.request.Request(url, headers=self._headers(), method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
                ids = [m["id"] for m in data.get("data", [])]
                return self.config.model in ids
        except Exception:
            return False
