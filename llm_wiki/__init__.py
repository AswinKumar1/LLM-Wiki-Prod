"""
llm-wiki-universal

A provider-agnostic implementation of Andrej Karpathy's LLM Wiki pattern.
Works with Ollama (local/free), OpenAI, Anthropic, or any OpenAI-compatible endpoint.
"""

__version__ = "0.1.0"
__author__ = "Alwin Kumar"
__repo__ = "https://github.com/AswinKumar1/LLM-Wiki-Prod"

from .providers import LLMProvider, LLMResponse, ProviderConfig, get_provider
from .wiki_fs import WikiFS
from .config import load_config
from .operations import IngestOperation, QueryOperation, LintOperation

__all__ = [
    "LLMProvider",
    "LLMResponse",
    "ProviderConfig",
    "get_provider",
    "WikiFS",
    "load_config",
    "IngestOperation",
    "QueryOperation",
    "LintOperation",
]
