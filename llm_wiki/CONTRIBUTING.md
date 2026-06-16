# Contributing to llm-wiki-universal

Thanks for wanting to contribute. The codebase is intentionally simple — no frameworks, no heavy dependencies — so it's easy to get started.

---

## Quick start

```bash
git clone https://github.com/AswinKumar1/LLM-Wiki-Prod.git
cd LLM-Wiki-Prod
pip install -e ".[dev]"
pytest tests/
```

All tests run offline — no API key, no Ollama, no internet connection required.

---

## How to add a new LLM provider

Adding a provider is three steps:

### 1. Create the adapter

Create `llm_wiki/providers/your_provider.py`:

```python
from .base import LLMProvider, LLMResponse, ProviderConfig
from typing import Optional

class YourProvider(LLMProvider):
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        # Set up any auth/client here

    def chat(
        self,
        system: str,
        user: str,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        # Call your API/model here
        response_text = "..."   # your API call
        return LLMResponse(
            content=response_text,
            model=self.config.model,
            provider=self.provider_name,
            prompt_tokens=0,        # fill from API response if available
            completion_tokens=0,
        )

    def health_check(self) -> bool:
        try:
            self.chat("", "ping", max_tokens=5)
            return True
        except Exception:
            return False

    @property
    def provider_name(self) -> str:
        return "your_provider"
```

### 2. Register it in the factory

Add one line to `llm_wiki/providers/factory.py`:

```python
_REGISTRY = {
    ...
    "your_provider": "llm_wiki.providers.your_provider:YourProvider",
}
```

### 3. Add a test

Create `tests/providers/test_your_provider.py` with a `MockYourProvider` that doesn't make real API calls. See `tests/providers/test_providers.py` for the pattern.

That's it. The CLI, config loader, doctor, and usage tracker all pick it up automatically.

---

## Project structure

```
llm_wiki/
├── providers/          # LLM adapters (one file per provider)
│   ├── base.py         # LLMProvider abstract class — the core interface
│   ├── factory.py      # Provider registry and get_provider()
│   ├── ollama.py
│   ├── openai_provider.py
│   ├── anthropic_provider.py
│   └── openai_compat.py
├── operations/         # The three core operations
│   ├── ingest.py       # Raw sources → wiki pages
│   ├── query.py        # Questions → answers from wiki
│   ├── lint.py         # Health checks + NLI contradiction detection
│   └── search.py       # BM25 search wrapper
├── utils/              # Shared utilities
│   ├── search.py       # BM25 index (pure stdlib)
│   ├── chunker.py      # Large source splitting
│   ├── usage.py        # Token/cost tracking
│   ├── doctor.py       # Pre-flight checks
│   ├── nli.py          # NLI contradiction detection
│   ├── pdf_reader.py   # PDF text extraction
│   ├── url_fetcher.py  # URL fetch + HTML cleaning
│   ├── source_reader.py# Unified source router
│   ├── watcher.py      # Watch mode (Day 5)
│   └── exporter.py     # HTML export (Day 5)
├── wiki_fs.py          # All filesystem I/O (single source of truth)
├── config.py           # Config loading (yaml → env vars → CLI flags)
├── prompts.py          # All LLM prompt templates
└── cli.py              # CLI entry point
```

---

## Design principles

**Zero required dependencies.** The core path (ingest → query → search) works with only Python stdlib. Optional deps (`pyyaml`, `pdfminer.six`, `sentence-transformers`) are always opt-in.

**Provider-agnostic.** Operations only talk to `LLMProvider`. No operation imports a concrete provider class directly.

**Testable without API keys.** Every test uses `MockProvider`. If a test requires a live API call, mark it `@pytest.mark.skipif` with a clear reason.

**Flat files only.** No databases, no vector stores, no message queues. The wiki is just a directory of markdown files.

---

## Running tests

```bash
# All tests
pytest tests/

# Just one module
pytest tests/utils/test_nli.py -v

# With coverage
pytest tests/ --cov=llm_wiki --cov-report=term-missing
```

---

## PR checklist

- [ ] New code has tests (all passing offline)
- [ ] No new required dependencies (use optional extras in `pyproject.toml`)
- [ ] New provider: adapter + factory registration + test
- [ ] New CLI command: added to `cli.py` + documented in README
- [ ] `pytest tests/` passes locally

---

## Code style

We use `ruff` for linting. Run `ruff check llm_wiki/` before pushing. Line length is 100.

No type stubs required — `from __future__ import annotations` + inline hints is enough.

---

## Opening an issue

- Bug: include the exact command you ran, the error output, your provider, and Python version
- Feature: describe the use case first, then the proposed solution
- Provider request: name the provider and link to its API docs

---

## License

By contributing, you agree your contributions will be licensed under the MIT License.
