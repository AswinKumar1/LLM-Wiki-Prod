"""
Base provider interface for llm-wiki-universal.

Every LLM provider (Ollama, OpenAI, Anthropic, custom) implements this
interface. The wiki engine never imports a concrete provider directly —
it only talks to this interface, keeping operations fully provider-agnostic.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Iterator, Optional


@dataclass
class LLMResponse:
    """Normalised response from any provider."""
    content: str
    model: str
    provider: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached: bool = False

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class ProviderConfig:
    """
    Provider configuration loaded from config.yaml or env vars.

    All fields are optional — each provider fills in its own defaults.
    """
    provider: str = "ollama"
    model: str = ""
    base_url: str = ""
    api_key: str = ""
    temperature: float = 0.2
    max_tokens: int = 4096
    timeout: int = 120
    extra: dict = field(default_factory=dict)


class LLMProvider(abc.ABC):
    """
    Abstract base class that every provider must implement.

    Minimal surface: chat() is the only required method.
    stream() and embed() are optional — providers that don't support
    them return sensible no-ops so the wiki engine doesn't break.
    """

    def __init__(self, config: ProviderConfig):
        self.config = config

    # ------------------------------------------------------------------
    # Required
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def chat(
        self,
        system: str,
        user: str,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """
        Send a system + user prompt, return a normalised LLMResponse.
        This is the only method the wiki engine strictly depends on.
        """

    # ------------------------------------------------------------------
    # Optional — default no-ops
    # ------------------------------------------------------------------

    def stream(
        self,
        system: str,
        user: str,
        *,
        temperature: Optional[float] = None,
    ) -> Iterator[str]:
        """
        Yield response tokens as they arrive.
        Default: falls back to chat() and yields the full response at once.
        """
        response = self.chat(system, user, temperature=temperature)
        yield response.content

    def embed(self, text: str) -> list[float]:
        """
        Return an embedding vector for text.
        Default: raises NotImplementedError — only used by optional search.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support embeddings. "
            "Use a provider that does, or disable semantic search."
        )

    @abc.abstractmethod
    def health_check(self) -> bool:
        """Return True if the provider is reachable and the model is available."""

    @property
    @abc.abstractmethod
    def provider_name(self) -> str:
        """Human-readable name, e.g. 'ollama', 'openai', 'anthropic'."""

    @property
    def model_name(self) -> str:
        return self.config.model

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_name!r})"
