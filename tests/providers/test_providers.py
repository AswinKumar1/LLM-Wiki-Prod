"""Tests for the provider layer."""

import pytest
from llm_wiki.providers.base import LLMProvider, LLMResponse, ProviderConfig
from llm_wiki.providers.factory import get_provider, list_providers


class MockProvider(LLMProvider):
    """Minimal provider for testing."""

    def chat(self, system, user, *, temperature=None, max_tokens=None):
        return LLMResponse(
            content=f"mock response to: {user[:30]}",
            model=self.config.model,
            provider=self.provider_name,
            prompt_tokens=10,
            completion_tokens=20,
        )

    def health_check(self):
        return True

    @property
    def provider_name(self):
        return "mock"


def test_llm_response_total_tokens():
    r = LLMResponse(content="hi", model="m", provider="p", prompt_tokens=10, completion_tokens=5)
    assert r.total_tokens == 15


def test_mock_provider_chat():
    cfg = ProviderConfig(provider="mock", model="test-model")
    p = MockProvider(cfg)
    resp = p.chat("system", "hello world")
    assert "mock response" in resp.content
    assert resp.model == "test-model"
    assert resp.total_tokens == 30


def test_mock_provider_stream_fallback():
    cfg = ProviderConfig(provider="mock", model="test-model")
    p = MockProvider(cfg)
    chunks = list(p.stream("system", "hello"))
    assert len(chunks) == 1
    assert "mock response" in chunks[0]


def test_list_providers():
    providers = list_providers()
    assert "ollama" in providers
    assert "openai" in providers
    assert "anthropic" in providers
    assert "openai_compat" in providers


def test_get_provider_unknown():
    cfg = ProviderConfig(provider="nonexistent_xyz")
    with pytest.raises(ValueError, match="Unknown provider"):
        get_provider(cfg)


def test_provider_config_defaults():
    cfg = ProviderConfig()
    assert cfg.provider == "ollama"
    assert cfg.temperature == 0.2
    assert cfg.max_tokens == 4096
