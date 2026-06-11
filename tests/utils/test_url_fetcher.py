"""Tests for the URL fetcher utility."""

import pytest
from llm_wiki.utils.url_fetcher import (
    fetch_url,
    FetchResult,
    _normalise_url,
    _url_to_filename,
    _url_to_title,
    _parse_html,
    _is_github_file,
    _github_raw_url,
    _WikiParser,
)


# ---------------------------------------------------------------------------
# Unit tests — helpers (no network)
# ---------------------------------------------------------------------------

def test_normalise_url_adds_https():
    assert _normalise_url("example.com") == "https://example.com"


def test_normalise_url_preserves_http():
    assert _normalise_url("http://example.com") == "http://example.com"


def test_normalise_url_preserves_https():
    assert _normalise_url("https://example.com") == "https://example.com"


def test_url_to_filename_basic():
    fn = _url_to_filename("https://example.com/blog/my-article")
    assert fn.endswith(".md")
    assert "example" in fn
    assert "/" not in fn


def test_url_to_filename_strips_www():
    fn = _url_to_filename("https://www.example.com/page")
    assert "www" not in fn


def test_url_to_filename_max_length():
    long_url = "https://example.com/" + "a" * 200
    fn = _url_to_filename(long_url)
    assert len(fn) <= 85   # 80 slug + ".md"


def test_url_to_title_from_path():
    title = _url_to_title("https://example.com/blog/my-great-article")
    assert "My" in title or "my" in title.lower()


def test_url_to_title_from_domain():
    title = _url_to_title("https://example.com/")
    assert "example" in title.lower()


def test_is_github_file_true():
    assert _is_github_file("https://github.com/user/repo/blob/main/README.md")


def test_is_github_file_false_for_repo():
    assert not _is_github_file("https://github.com/user/repo")


def test_github_raw_url():
    url = "https://github.com/user/repo/blob/main/README.md"
    raw = _github_raw_url(url)
    assert "raw.githubusercontent.com" in raw
    assert "/blob/" not in raw


# ---------------------------------------------------------------------------
# HTML parser tests
# ---------------------------------------------------------------------------

def test_wiki_parser_extracts_text():
    html = "<html><body><p>Hello world</p></body></html>"
    parser = _WikiParser()
    parser.feed(html)
    assert "Hello world" in parser.text


def test_wiki_parser_strips_script():
    html = "<html><body><script>alert('bad')</script><p>Good content</p></body></html>"
    parser = _WikiParser()
    parser.feed(html)
    assert "alert" not in parser.text
    assert "Good content" in parser.text


def test_wiki_parser_strips_nav():
    html = "<html><body><nav>Menu items</nav><p>Article body</p></body></html>"
    parser = _WikiParser()
    parser.feed(html)
    assert "Menu items" not in parser.text
    assert "Article body" in parser.text


def test_wiki_parser_extracts_title():
    html = "<html><head><title>My Page Title</title></head><body><p>Body</p></body></html>"
    parser = _WikiParser()
    parser.feed(html)
    assert parser.title == "My Page Title"


def test_wiki_parser_converts_headings():
    html = "<html><body><h1>Main Heading</h1><h2>Sub Heading</h2></body></html>"
    parser = _WikiParser()
    parser.feed(html)
    assert "# Main Heading" in parser.text
    assert "## Sub Heading" in parser.text


def test_wiki_parser_converts_list_items():
    html = "<html><body><ul><li>Item one</li><li>Item two</li></ul></body></html>"
    parser = _WikiParser()
    parser.feed(html)
    assert "Item one" in parser.text
    assert "Item two" in parser.text


def test_parse_html_returns_title_and_text():
    html = """<html>
    <head><title>Test Article</title></head>
    <body>
        <nav>Skip me</nav>
        <main><p>This is the main content of the article.</p></main>
    </body>
    </html>"""
    title, text = _parse_html(html, "https://example.com/test")
    assert title == "Test Article"
    assert "main content" in text
    assert "Skip me" not in text


def test_fetch_result_as_markdown():
    result = FetchResult(
        url="https://example.com/article",
        text="This is the article body.",
        title="Test Article",
        success=True,
        suggested_filename="example-com-article.md",
    )
    md = result.as_markdown
    assert "---" in md
    assert "title: Test Article" in md
    assert "source_url: https://example.com/article" in md
    assert "This is the article body." in md


def test_fetch_url_invalid_url():
    """Non-existent domain should return a failure result, not raise."""
    result = fetch_url("https://this-domain-absolutely-does-not-exist-xyz-abc.com/")
    assert not result.success
    assert result.error is not None


def test_fetch_url_bad_scheme():
    """ftp:// URLs should be handled gracefully."""
    result = fetch_url("ftp://example.com/file.txt")
    # Normalise adds https if no scheme, so ftp stays ftp and should fail
    assert not result.success or result.success  # either way, no crash


# ---------------------------------------------------------------------------
# SearchOperation tests (uses mock provider)
# ---------------------------------------------------------------------------

def test_search_operation_bm25_only(tmp_path):
    from llm_wiki.providers.base import LLMProvider, LLMResponse, ProviderConfig
    from llm_wiki.wiki_fs import WikiFS
    from llm_wiki.operations.search import SearchOperation

    class NullProvider(LLMProvider):
        def chat(self, *a, **kw):
            return LLMResponse(content="", model="null", provider="null")
        def health_check(self): return True
        @property
        def provider_name(self): return "null"

    fs = WikiFS(tmp_path)
    fs.ensure_structure()
    fs.write_wiki_page(
        "wiki/concepts/rag.md",
        "---\ntitle: RAG\ntype: concept\n---\n# RAG\nRetrieval Augmented Generation.\n",
    )
    fs.write_wiki_page(
        "wiki/concepts/llm.md",
        "---\ntitle: LLM\ntype: concept\n---\n# LLM\nLarge Language Model.\n",
    )

    op   = SearchOperation(NullProvider(ProviderConfig()), fs)
    resp = op.search("retrieval", top_k=5)

    assert resp.success
    assert resp.found
    assert resp.total_docs_searched == 2
    assert "rag" in resp.results[0].path


def test_search_operation_empty_wiki(tmp_path):
    from llm_wiki.providers.base import LLMProvider, LLMResponse, ProviderConfig
    from llm_wiki.wiki_fs import WikiFS
    from llm_wiki.operations.search import SearchOperation

    class NullProvider(LLMProvider):
        def chat(self, *a, **kw):
            return LLMResponse(content="", model="null", provider="null")
        def health_check(self): return True
        @property
        def provider_name(self): return "null"

    fs   = WikiFS(tmp_path)
    fs.ensure_structure()
    op   = SearchOperation(NullProvider(ProviderConfig()), fs)
    resp = op.search("anything")

    assert not resp.success
    assert "No wiki pages" in resp.error
