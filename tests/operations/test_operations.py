"""Tests for ingest, query, and lint operations using a mock provider."""

import pytest
import tempfile
from pathlib import Path

from llm_wiki.providers.base import LLMProvider, LLMResponse, ProviderConfig
from llm_wiki.wiki_fs import WikiFS
from llm_wiki.operations import IngestOperation, QueryOperation, LintOperation


# ------------------------------------------------------------------
# Mock provider
# ------------------------------------------------------------------

class MockProvider(LLMProvider):
    def __init__(self, config, responses=None):
        super().__init__(config)
        self._responses = responses or {}
        self._default = "### TAKEAWAYS\n- Key point\n\n### SOURCE_SUMMARY_PAGE\n---\ntitle: test\ntype: source-summary\ncreated: TODAY\nupdated: TODAY\nconfidence: high\n---\n# Test\nContent.\n\n### PAGES_TO_UPDATE\n\n### UPDATED_PAGES\n\n### NEW_PAGES\n\n### INDEX_UPDATE\n# Wiki Index\n\n## Sources\n- [[test]]\n\n### LOG_ENTRY\ningest | test.md | 1 pages created, 0 pages updated"

    def chat(self, system, user, *, temperature=None, max_tokens=None):
        content = self._responses.get("chat", self._default)
        return LLMResponse(content=content, model="mock", provider="mock", prompt_tokens=5, completion_tokens=10)

    def health_check(self): return True

    @property
    def provider_name(self): return "mock"


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def wiki_root(tmp_path):
    fs = WikiFS(tmp_path)
    fs.ensure_structure()
    fs.init_index()
    return tmp_path


@pytest.fixture
def provider():
    return MockProvider(ProviderConfig(provider="mock", model="mock-model"))


# ------------------------------------------------------------------
# WikiFS tests
# ------------------------------------------------------------------

def test_ensure_structure_creates_dirs(wiki_root):
    assert (wiki_root / "raw" / "articles").exists()
    assert (wiki_root / "wiki" / "concepts").exists()
    assert (wiki_root / "outputs").exists()


def test_init_index_creates_file(wiki_root):
    fs = WikiFS(wiki_root)
    assert fs.index_path.exists()
    content = fs.read_index()
    assert "Wiki Index" in content


def test_write_and_read_wiki_page(wiki_root):
    fs = WikiFS(wiki_root)
    fs.write_wiki_page("wiki/concepts/test.md", "---\ntitle: Test\ntype: concept\n---\n# Test\n")
    content = fs.read_wiki_page("wiki/concepts/test.md")
    assert "# Test" in content


def test_append_log(wiki_root):
    fs = WikiFS(wiki_root)
    fs.append_log("ingest | test.md | 2 created")
    log = fs.read_log()
    assert "test.md" in log


def test_new_raw_sources(wiki_root):
    fs = WikiFS(wiki_root)
    # Create a raw source
    source = wiki_root / "raw" / "articles" / "my-article.md"
    source.write_text("# Article\nSome content.")
    sources = fs.new_raw_sources()
    assert len(sources) == 1
    assert sources[0].name == "my-article.md"


def test_sha256_stability(wiki_root):
    fs = WikiFS(wiki_root)
    source = wiki_root / "raw" / "articles" / "stable.md"
    source.write_text("content")
    sha1 = fs.source_sha256(source)
    sha2 = fs.source_sha256(source)
    assert sha1 == sha2
    assert len(sha1) == 64


# ------------------------------------------------------------------
# Ingest tests
# ------------------------------------------------------------------

def test_ingest_creates_source_summary(wiki_root, provider):
    # Create a raw source
    source = wiki_root / "raw" / "articles" / "test.md"
    source.write_text("# Test Article\nThis is a test.")

    fs = WikiFS(wiki_root)
    op = IngestOperation(provider, fs)
    results = op.run()

    assert len(results) == 1
    assert results[0].success
    assert any("sources" in p for p in results[0].pages_created)


def test_ingest_no_sources_returns_empty(wiki_root, provider):
    fs = WikiFS(wiki_root)
    op = IngestOperation(provider, fs)
    results = op.run()
    assert results == []


def test_ingest_error_captured(wiki_root):
    class FailProvider(MockProvider):
        def chat(self, *a, **kw):
            raise ConnectionError("cannot connect")

    source = wiki_root / "raw" / "articles" / "fail.md"
    source.write_text("content")

    fs = WikiFS(wiki_root)
    op = IngestOperation(FailProvider(ProviderConfig()), fs)
    results = op.run()

    assert len(results) == 1
    assert not results[0].success
    assert "cannot connect" in results[0].error


# ------------------------------------------------------------------
# Query tests
# ------------------------------------------------------------------

def test_query_empty_wiki(wiki_root, provider):
    # WikiFS.init_index() writes a template, so overwrite with truly empty
    fs = WikiFS(wiki_root)
    fs.index_path.write_text("")   # blank out the index
    op = QueryOperation(provider, fs)
    result = op.run("what is this about?")
    assert not result.success
    assert "empty" in result.error.lower()


def test_query_with_content(wiki_root):
    # Populate wiki with a page
    fs = WikiFS(wiki_root)
    fs.write_wiki_page("wiki/concepts/rag.md", "---\ntitle: RAG\ntype: concept\n---\n# RAG\nRetrieval Augmented Generation.\n")
    fs.write_index("# Wiki Index\n\n## Concepts\n- [[rag]] — Retrieval Augmented Generation\n")

    answer_text = "RAG stands for Retrieval Augmented Generation."
    responses = {
        "chat": '["wiki/concepts/rag.md"]'  # first call: page selection
    }

    class TwoStepProvider(MockProvider):
        def __init__(self):
            super().__init__(ProviderConfig(), responses)
            self._call_count = 0

        def chat(self, system, user, **kw):
            self._call_count += 1
            if self._call_count == 1:
                return LLMResponse(content='["wiki/concepts/rag.md"]', model="mock", provider="mock")
            return LLMResponse(content=answer_text, model="mock", provider="mock")

    op = QueryOperation(TwoStepProvider(), fs)
    result = op.run("What is RAG?")
    assert result.success
    assert result.answer == answer_text


# ------------------------------------------------------------------
# Lint tests
# ------------------------------------------------------------------

def test_lint_no_pages(wiki_root, provider):
    fs = WikiFS(wiki_root)
    op = LintOperation(provider, fs)
    result = op.run()
    assert not result.success
    assert "No wiki pages" in result.error


def test_lint_finds_orphan(wiki_root):
    fs = WikiFS(wiki_root)
    # Page with no incoming links
    fs.write_wiki_page("wiki/concepts/lonely.md", "---\ntitle: Lonely\ntype: concept\n---\n# Lonely\nNo one links here.\n")
    fs.write_index("# Index\n")

    lint_response = """### CONTRADICTIONS
(none)

### ORPHAN_PAGES
- wiki/concepts/lonely.md

### MISSING_PAGES

### LOW_CONFIDENCE_PAGES

### STRUCTURAL_ISSUES

### SUMMARY
Wiki looks mostly healthy.
"""

    class LintProvider(MockProvider):
        def chat(self, *a, **kw):
            return LLMResponse(content=lint_response, model="mock", provider="mock")

    op = LintOperation(LintProvider(ProviderConfig()), fs)
    result = op.run()
    assert result.success
    assert "wiki/concepts/lonely.md" in result.orphan_pages


def test_lint_missing_frontmatter(wiki_root, provider):
    fs = WikiFS(wiki_root)
    # Page without frontmatter
    fs.write_wiki_page("wiki/concepts/no-fm.md", "# No Frontmatter\nJust content.\n")

    op = LintOperation(provider, fs)
    # Even without calling LLM, structural check should flag it
    issues = op._check_frontmatter(fs.list_wiki_pages())
    assert any("no-fm" in i for i in issues)
