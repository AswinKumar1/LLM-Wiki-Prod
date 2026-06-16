"""Tests for WikiWatcher (watch mode) and WikiExporter (HTML export)."""

import pytest
import time
from pathlib import Path
from llm_wiki.providers.base import LLMProvider, LLMResponse, ProviderConfig
from llm_wiki.wiki_fs import WikiFS
from llm_wiki.utils.watcher import WikiWatcher
from llm_wiki.utils.exporter import (
    WikiExporter,
    _md_to_html,
    _parse_frontmatter,
    _path_to_slug,
    _path_to_group,
)


# ---------------------------------------------------------------------------
# Mock provider
# ---------------------------------------------------------------------------

_INGEST_RESPONSE = """### TAKEAWAYS
- Key point from source

### SOURCE_SUMMARY_PAGE
---
title: test-source
type: source-summary
created: TODAY
updated: TODAY
confidence: high
---
# Test Source Summary
Content here.

### PAGES_TO_UPDATE

### UPDATED_PAGES

### NEW_PAGES

### INDEX_UPDATE
# Wiki Index

## Sources
- [[test-source]]

### LOG_ENTRY
ingest | test.md | 1 pages created, 0 pages updated
"""


class MockProvider(LLMProvider):
    def __init__(self):
        super().__init__(ProviderConfig())
        self.call_count = 0

    def chat(self, system, user, *, temperature=None, max_tokens=None):
        self.call_count += 1
        return LLMResponse(
            content=_INGEST_RESPONSE, model="mock", provider="mock",
            prompt_tokens=50, completion_tokens=100,
        )

    def health_check(self): return True

    @property
    def provider_name(self): return "mock"


# ---------------------------------------------------------------------------
# WikiWatcher tests
# ---------------------------------------------------------------------------

@pytest.fixture
def wiki_root(tmp_path):
    fs = WikiFS(tmp_path)
    fs.ensure_structure()
    fs.init_index()
    return tmp_path


def test_watcher_run_once_no_files(wiki_root):
    fs       = WikiFS(wiki_root)
    provider = MockProvider()
    watcher  = WikiWatcher(provider, fs, interval=1)
    events   = watcher.run_once()
    assert events == []
    assert provider.call_count == 0


def test_watcher_run_once_detects_new_file(wiki_root):
    fs   = WikiFS(wiki_root)
    source = wiki_root / "raw" / "articles" / "new-article.md"
    source.write_text("# New Article\n\nThis is fresh content for the wiki.")

    provider = MockProvider()
    watcher  = WikiWatcher(provider, fs, interval=1)
    events   = watcher.run_once()

    assert len(events) == 1
    assert events[0].success
    assert events[0].path == source
    assert provider.call_count == 1


def test_watcher_does_not_reingest_processed_files(wiki_root):
    fs   = WikiFS(wiki_root)
    source = wiki_root / "raw" / "articles" / "already-done.md"
    source.write_text("# Already Done\n\nAlready processed content.")

    # Pre-create the source summary to simulate already-processed state
    fs.write_wiki_page(
        "wiki/sources/already-done.md",
        "---\ntitle: already-done\ntype: source-summary\n---\n# Done\n",
    )

    provider = MockProvider()
    watcher  = WikiWatcher(provider, fs, interval=1)
    events   = watcher.run_once()

    assert events == []
    assert provider.call_count == 0


def test_watcher_processes_multiple_new_files(wiki_root):
    fs = WikiFS(wiki_root)
    for i in range(3):
        (wiki_root / "raw" / "articles" / f"doc-{i}.md").write_text(
            f"# Document {i}\n\nContent for document {i}."
        )

    provider = MockProvider()
    watcher  = WikiWatcher(provider, fs, interval=1)
    events   = watcher.run_once()

    assert len(events) == 3
    assert all(e.success for e in events)
    assert provider.call_count == 3


def test_watcher_event_has_correct_fields(wiki_root):
    fs     = WikiFS(wiki_root)
    source = wiki_root / "raw" / "articles" / "event-test.md"
    source.write_text("# Event Test\n\nContent.")

    provider = MockProvider()
    watcher  = WikiWatcher(provider, fs)
    events   = watcher.run_once()

    assert len(events) == 1
    e = events[0]
    assert e.path == source
    assert e.success
    assert e.tokens_used == 150   # 50 + 100 from mock
    assert e.error is None
    assert isinstance(e.timestamp, str)


def test_watcher_on_event_callback(wiki_root):
    fs     = WikiFS(wiki_root)
    source = wiki_root / "raw" / "articles" / "callback-test.md"
    source.write_text("# Callback\n\nContent.")

    received = []
    def callback(event):
        received.append(event)

    provider = MockProvider()
    watcher  = WikiWatcher(provider, fs, on_event=callback)
    watcher.run_once()

    assert len(received) == 1
    assert received[0].success


def test_watcher_stop_flag(wiki_root):
    fs       = WikiFS(wiki_root)
    provider = MockProvider()
    watcher  = WikiWatcher(provider, fs, interval=1)
    watcher.stop()
    assert not watcher._running


def test_watcher_seen_set_grows(wiki_root):
    fs     = WikiFS(wiki_root)
    source = wiki_root / "raw" / "articles" / "seen-test.md"
    source.write_text("# Seen\n\nContent.")

    provider = MockProvider()
    watcher  = WikiWatcher(provider, fs)

    events1 = watcher.run_once()
    assert len(events1) == 1

    # Second run — file is now in _seen, should not re-ingest
    events2 = watcher.run_once()
    assert len(events2) == 0


# ---------------------------------------------------------------------------
# WikiExporter tests
# ---------------------------------------------------------------------------

@pytest.fixture
def wiki_with_pages(tmp_path):
    fs = WikiFS(tmp_path)
    fs.ensure_structure()

    pages = {
        "wiki/concepts/rag.md": """---
title: RAG
type: concept
confidence: high
created: 2026-01-01
---
# Retrieval Augmented Generation

RAG uses a [[vector-database]] for retrieval.

## How it works

The system retrieves documents and passes them to the LLM.
""",
        "wiki/concepts/llm-wiki.md": """---
title: LLM Wiki Pattern
type: concept
confidence: high
created: 2026-01-01
---
# LLM Wiki Pattern

The [[llm-wiki-pattern]] compounds knowledge over time.

It is an alternative to [[rag]].
""",
        "wiki/entities/karpathy.md": """---
title: Andrej Karpathy
type: entity
confidence: high
---
# Andrej Karpathy

Creator of the [[llm-wiki-pattern]].
""",
    }
    for path, content in pages.items():
        fs.write_wiki_page(path, content)

    (tmp_path / "AGENTS.md").write_text("# Test Wiki\nAgent schema.\n")
    return tmp_path, fs


def test_exporter_renders_html(wiki_with_pages):
    _, fs = wiki_with_pages
    exporter = WikiExporter(fs)
    html = exporter.render()
    assert html.startswith("<!DOCTYPE html>")
    assert "<html" in html
    assert "</html>" in html


def test_exporter_includes_all_pages(wiki_with_pages):
    _, fs = wiki_with_pages
    exporter = WikiExporter(fs)
    html = exporter.render()
    assert "Retrieval Augmented Generation" in html
    assert "LLM Wiki Pattern" in html
    assert "Andrej Karpathy" in html


def test_exporter_converts_wikilinks(wiki_with_pages):
    _, fs = wiki_with_pages
    exporter = WikiExporter(fs)
    html = exporter.render()
    # [[rag]] should become a clickable link
    assert 'class="wikilink"' in html
    assert 'href="#' in html


def test_exporter_has_navigation(wiki_with_pages):
    _, fs = wiki_with_pages
    exporter = WikiExporter(fs)
    html = exporter.render()
    assert 'class="nav-link"' in html
    assert "Concepts" in html
    assert "Entities" in html


def test_exporter_saves_file(wiki_with_pages, tmp_path):
    _, fs = wiki_with_pages
    exporter = WikiExporter(fs)
    out_path = exporter.save()
    assert out_path.exists()
    assert out_path.suffix == ".html"
    content = out_path.read_text()
    assert "<!DOCTYPE html>" in content


def test_exporter_empty_wiki(tmp_path):
    fs = WikiFS(tmp_path)
    fs.ensure_structure()
    exporter = WikiExporter(fs)
    html = exporter.render()
    assert "No wiki pages" in html


# ---------------------------------------------------------------------------
# Markdown → HTML unit tests
# ---------------------------------------------------------------------------

def test_md_to_html_heading():
    html = _md_to_html("# Main Heading", {}, {})
    assert "<h1" in html
    assert "Main Heading" in html


def test_md_to_html_bold():
    html = _md_to_html("This is **bold** text.", {}, {})
    assert "<strong>bold</strong>" in html


def test_md_to_html_italic():
    html = _md_to_html("This is *italic* text.", {}, {})
    assert "<em>italic</em>" in html


def test_md_to_html_inline_code():
    html = _md_to_html("Use `wiki ingest` to run.", {}, {})
    assert "<code>wiki ingest</code>" in html


def test_md_to_html_code_block():
    md = "```python\nprint('hello')\n```"
    html = _md_to_html(md, {}, {})
    assert "<pre>" in html
    assert "<code" in html
    assert "print" in html


def test_md_to_html_unordered_list():
    md = "- Item one\n- Item two\n- Item three"
    html = _md_to_html(md, {}, {})
    assert "<ul>" in html
    assert "<li>Item one</li>" in html


def test_md_to_html_ordered_list():
    md = "1. First\n2. Second\n3. Third"
    html = _md_to_html(md, {}, {})
    assert "<ol>" in html
    assert "<li>First</li>" in html


def test_md_to_html_wikilink_resolves():
    slug_map  = {"wiki-concepts-rag-md": "RAG"}
    title_map = {"rag": "wiki-concepts-rag-md"}
    html = _md_to_html("See [[rag]] for details.", slug_map, title_map)
    assert 'class="wikilink"' in html
    assert "rag" in html.lower()


def test_md_to_html_external_link():
    html = _md_to_html("Visit [Ollama](https://ollama.com) today.", {}, {})
    assert 'href="https://ollama.com"' in html
    assert "Ollama" in html


def test_md_to_html_table():
    md = "| A | B |\n|---|---|\n| 1 | 2 |"
    html = _md_to_html(md, {}, {})
    assert "<table>" in html
    assert "<th>" in html
    assert "<td>" in html


# ---------------------------------------------------------------------------
# Frontmatter parser
# ---------------------------------------------------------------------------

def test_parse_frontmatter_extracts_title():
    text = "---\ntitle: My Page\ntype: concept\n---\n\n# Body"
    title, meta, body = _parse_frontmatter(text)
    assert title == "My Page"
    assert meta["type"] == "concept"
    assert "# Body" in body


def test_parse_frontmatter_no_frontmatter():
    text = "# Just a heading\n\nBody content."
    title, meta, body = _parse_frontmatter(text)
    assert title == ""
    assert meta == {}
    assert "# Just a heading" in body


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def test_path_to_group_concepts():
    assert _path_to_group("wiki/concepts/rag.md") == "concepts"


def test_path_to_group_entities():
    assert _path_to_group("wiki/entities/karpathy.md") == "entities"


def test_path_to_group_sources():
    assert _path_to_group("wiki/sources/source-gist.md") == "sources"


def test_path_to_slug_safe():
    slug = _path_to_slug("wiki/concepts/llm-wiki.md")
    assert " " not in slug
    assert "/" not in slug
    assert slug.islower() or slug == slug.lower()
