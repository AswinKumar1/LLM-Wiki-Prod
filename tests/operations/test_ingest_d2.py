"""Tests for the upgraded Day 2 ingest operation."""

import pytest
from pathlib import Path
from llm_wiki.providers.base import LLMProvider, LLMResponse, ProviderConfig
from llm_wiki.wiki_fs import WikiFS
from llm_wiki.operations.ingest import IngestOperation


class MockProvider(LLMProvider):
    def __init__(self, config=None, response_text=None, fail_times=0):
        super().__init__(config or ProviderConfig())
        self._response = response_text or _DEFAULT_RESPONSE
        self._fail_times = fail_times
        self._call_count = 0

    def chat(self, system, user, *, temperature=None, max_tokens=None):
        self._call_count += 1
        if self._call_count <= self._fail_times:
            raise ConnectionError("simulated failure")
        return LLMResponse(
            content=self._response,
            model="mock",
            provider="mock",
            prompt_tokens=100,
            completion_tokens=50,
        )

    def health_check(self):
        return True

    @property
    def provider_name(self):
        return "mock"


_DEFAULT_RESPONSE = """### TAKEAWAYS
- Key insight from the source

### SOURCE_SUMMARY_PAGE
---
title: test-article
type: source-summary
created: TODAY
updated: TODAY
confidence: high
---
# Test Article Summary

Content here.

### PAGES_TO_UPDATE

### UPDATED_PAGES

### NEW_PAGES

### INDEX_UPDATE
# Wiki Index

## Sources
- [[test-article]]

### LOG_ENTRY
ingest | test-article.md | 1 pages created, 0 pages updated
"""


@pytest.fixture
def wiki_root(tmp_path):
    fs = WikiFS(tmp_path)
    fs.ensure_structure()
    fs.init_index()
    return tmp_path


def test_ingest_records_token_usage(wiki_root):
    import json

    source = wiki_root / "raw" / "articles" / "test-article.md"
    source.write_text("# Article\n" + "word " * 100)

    fs = WikiFS(wiki_root)
    op = IngestOperation(MockProvider(), fs)
    results = op.run()

    assert len(results) == 1
    assert results[0].success
    assert results[0].tokens_used == 150  # 100 prompt + 50 completion

    usage_file = wiki_root / "wiki" / "usage.json"
    assert usage_file.exists()
    data = json.loads(usage_file.read_text())
    assert len(data) >= 1
    assert data[0]["op"] == "ingest"


def test_ingest_retry_succeeds_after_failures(wiki_root):
    """Provider fails twice then succeeds — should still produce a result."""
    source = wiki_root / "raw" / "articles" / "retry-test.md"
    source.write_text("# Retry Article\nContent.")

    fs = WikiFS(wiki_root)
    # fail_times=2: first two calls fail, third succeeds
    provider = MockProvider(fail_times=2)
    op = IngestOperation(provider, fs, verbose=False)

    import unittest.mock as mock
    import time

    with mock.patch("time.sleep"):  # skip actual sleep in tests
        results = op.run()

    assert len(results) == 1
    assert results[0].success
    assert provider._call_count == 3  # 2 failures + 1 success


def test_ingest_fails_after_max_retries(wiki_root):
    """All retries exhausted — result should have error."""
    source = wiki_root / "raw" / "articles" / "fail-test.md"
    source.write_text("# Always Fails\nContent.")

    fs = WikiFS(wiki_root)
    provider = MockProvider(fail_times=999)  # always fails
    op = IngestOperation(provider, fs, verbose=False)

    import unittest.mock as mock

    with mock.patch("time.sleep"):
        results = op.run()

    assert len(results) == 1
    assert not results[0].success
    assert results[0].error is not None


def test_ingest_large_source_is_chunked(wiki_root):
    """A file over the chunk threshold should be split into multiple chunks."""
    # Create a file that's ~25k chars (>6000 tokens)
    paragraph = "This is a test paragraph with many words. " * 20
    text = "\n\n".join([paragraph] * 30)  # ~25k chars
    source = wiki_root / "raw" / "articles" / "large-doc.md"
    source.write_text(text)

    fs = WikiFS(wiki_root)
    call_count_tracker = {"n": 0}

    class CountingProvider(MockProvider):
        def chat(self, *a, **kw):
            call_count_tracker["n"] += 1
            return super().chat(*a, **kw)

    op = IngestOperation(
        CountingProvider(),
        fs,
        verbose=False,
        max_tokens_per_chunk=2000,  # low threshold to force chunking
    )
    results = op.run()

    assert len(results) == 1
    assert results[0].success
    assert results[0].chunks_processed > 1
    assert call_count_tracker["n"] > 1  # multiple LLM calls were made


def test_ingest_cost_usd_tracked(wiki_root):
    """Cost should be 0 for Ollama but > 0 for a priced model."""
    source = wiki_root / "raw" / "articles" / "cost-test.md"
    source.write_text("# Cost Test\nContent.")

    class OpenAIMockProvider(MockProvider):
        @property
        def provider_name(self):
            return "openai"

        @property
        def model_name(self):
            return "gpt-4o-mini"

    fs = WikiFS(wiki_root)
    op = IngestOperation(OpenAIMockProvider(), fs)
    results = op.run()

    # With gpt-4o-mini pricing, 150 tokens should cost something
    assert results[0].cost_usd > 0.0
