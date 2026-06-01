"""
Provider factory.

Usage:
    provider = get_provider(config)   # config.provider = "ollama" | "openai" | ...

Adding a new provider:
    1. Implement LLMProvider in a new file under llm_wiki/providers/
    2. Add it to _REGISTRY below
    3. That's it — the CLI and wiki engine pick it up automatically
"""

from __future__ import annotations

from .base import LLMProvider, ProviderConfig

# Registry maps config string → provider class (imported lazily to keep
# startup fast and avoid import errors for providers whose deps aren't installed)
_REGISTRY: dict[str, str] = {
    "ollama":        "llm_wiki.providers.ollama:OllamaProvider",
    "openai":        "llm_wiki.providers.openai_provider:OpenAIProvider",
    "anthropic":     "llm_wiki.providers.anthropic_provider:AnthropicProvider",
    "openai_compat": "llm_wiki.providers.openai_compat:OpenAICompatProvider",
    # Aliases
    "lm_studio":     "llm_wiki.providers.openai_compat:OpenAICompatProvider",
    "vllm":          "llm_wiki.providers.openai_compat:OpenAICompatProvider",
    "groq":          "llm_wiki.providers.openai_compat:OpenAICompatProvider",
    "together":      "llm_wiki.providers.openai_compat:OpenAICompatProvider",
    "hermes":        "llm_wiki.providers.openai_compat:OpenAICompatProvider",
}


def get_provider(config: ProviderConfig) -> LLMProvider:
    """
    Resolve a ProviderConfig to a concrete LLMProvider instance.

    Raises:
        ValueError  if config.provider is not registered
        ImportError if the provider's module can't be imported
    """
    key = config.provider.lower()
    if key not in _REGISTRY:
        known = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(
            f"Unknown provider {config.provider!r}. "
            f"Known providers: {known}"
        )
    module_path, class_name = _REGISTRY[key].rsplit(":", 1)
    import importlib
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(config)


def list_providers() -> list[str]:
    """Return the canonical provider names (no aliases)."""
    canonical = {"ollama", "openai", "anthropic", "openai_compat"}
    return sorted(canonical)
