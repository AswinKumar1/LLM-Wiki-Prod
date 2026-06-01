"""
Ollama provider — runs 100% locally, no API key required.

Default model: qwen2.5:3b  (fast, good for wiki tasks, ~2GB RAM)
Also works with: mistral:7b, llama3.2:3b, phi3.5, hermes3, and anything
you've pulled via `ollama pull <model>`.

Install Ollama: https://ollama.com
Pull a model:   ollama pull qwen2.5:3b
"""

from __future__ import annotations

import json
from typing import Iterator, Optional

import urllib.request
import urllib.error

from .base import LLMProvider, LLMResponse, ProviderConfig

_DEFAULT_BASE_URL = "http://localhost:11434"
_DEFAULT_MODEL = "qwen2.5:3b"


class OllamaProvider(LLMProvider):
    """
    Adapter for the Ollama local inference server.
    Uses the /api/chat endpoint (not /api/generate) for proper
    system/user role separation.
    """

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self._base_url = (config.base_url or _DEFAULT_BASE_URL).rstrip("/")
        if not config.model:
            config.model = _DEFAULT_MODEL

    @property
    def provider_name(self) -> str:
        return "ollama"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _post(self, endpoint: str, payload: dict) -> dict:
        url = f"{self._base_url}{endpoint}"
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.URLError as exc:
            raise ConnectionError(
                f"Cannot reach Ollama at {self._base_url}. "
                "Is Ollama running? Try: ollama serve"
            ) from exc

    def _build_messages(self, system: str, user: str) -> list[dict]:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        return messages

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def chat(
        self,
        system: str,
        user: str,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        payload = {
            "model": self.config.model,
            "messages": self._build_messages(system, user),
            "stream": False,
            "options": {
                "temperature": temperature if temperature is not None else self.config.temperature,
                "num_predict": max_tokens or self.config.max_tokens,
            },
        }
        result = self._post("/api/chat", payload)
        content = result.get("message", {}).get("content", "")

        # Strip <think>...</think> blocks emitted by reasoning models (Qwen3, etc.)
        content = _strip_think_blocks(content)

        return LLMResponse(
            content=content,
            model=self.config.model,
            provider=self.provider_name,
            prompt_tokens=result.get("prompt_eval_count", 0),
            completion_tokens=result.get("eval_count", 0),
        )

    def stream(
        self,
        system: str,
        user: str,
        *,
        temperature: Optional[float] = None,
    ) -> Iterator[str]:
        payload = {
            "model": self.config.model,
            "messages": self._build_messages(system, user),
            "stream": True,
            "options": {
                "temperature": temperature if temperature is not None else self.config.temperature,
            },
        }
        url = f"{self._base_url}/api/chat"
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout) as resp:
                for raw_line in resp:
                    line = raw_line.decode().strip()
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                        token = chunk.get("message", {}).get("content", "")
                        if token:
                            yield token
                    except json.JSONDecodeError:
                        continue
        except urllib.error.URLError as exc:
            raise ConnectionError(
                f"Cannot reach Ollama at {self._base_url}."
            ) from exc

    def embed(self, text: str) -> list[float]:
        payload = {"model": self.config.model, "input": text}
        result = self._post("/api/embed", payload)
        embeddings = result.get("embeddings", [[]])
        return embeddings[0] if embeddings else []

    def health_check(self) -> bool:
        try:
            url = f"{self._base_url}/api/tags"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
                models = [m["name"] for m in data.get("models", [])]
                model_base = self.config.model.split(":")[0]
                return any(m.startswith(model_base) for m in models)
        except Exception:
            return False

    def list_local_models(self) -> list[str]:
        """Helper: return all models currently pulled in Ollama."""
        try:
            url = f"{self._base_url}/api/tags"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
                return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []


# ------------------------------------------------------------------
# Utility
# ------------------------------------------------------------------

def _strip_think_blocks(text: str) -> str:
    """Remove <think>...</think> blocks emitted by Qwen3 and similar models."""
    import re
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
