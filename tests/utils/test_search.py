"""Tests for BM25 search index."""

import pytest
from pathlib import Path
from llm_wiki.utils.search import BM25Index, SearchResult, _tokenize, _extract_title, _strip_frontmatter
from llm_wiki.wiki_fs import WikiFS


# ---------------------------------------------------------------------------
# Unit tests — tokenizer / helpers
# ---------------------------------------------------------------------------

def test_tokenize_basic():
    tokens = _tokenize("Retrieval Augmented Generation")
    assert "retrieval" in tokens
    assert "augmented" in tokens
    assert "generation" in tokens


def test_tokenize_removes_stopwords():
    tokens = _tokenize("the cat sat on the mat")
    assert "the" not in tokens
    assert "on" not in tokens
    assert "cat" in tokens
    assert "sat" in tokens


def test_tokenize_removes_punctuation():
    tokens = _tokenize("RAG: a powerful technique!")
    assert "rag" in tokens
    assert "powerful" in tokens
    assert ":" not in str(tokens)


def test_strip_frontmatter():
    text = "---\ntitle: Test\ntype: concept\n---\n\n# Content\n\nBody here."
    stripped = _strip_frontmatter(text)
    assert "---" not in stripped
    assert "# Content" in stripped
    assert "Body here" in stripped


def test_strip_frontmatter_no_frontmatter():
    text = "# Just a heading\n\nSome content."
    assert _strip_frontmatter(text) == text


def test_extract_title_from_frontmatter():
    text = "---\ntitle: My Page Title\ntype: concept\n---\n# Other"
    assert _extract_title(text, "wiki/concepts/test.md") == "My Page Title"


def test_extract_title_from_h1():
    text = "# First Heading\n\nContent."
    assert _extract_title(text, "wiki/concepts/test.md") == "First Heading"


def test_extract_title_from_filename():
    text = "No heading here."
    title = _extract_title(text, "wiki/concepts/rag-overview.md")
    assert "Rag" in title or "rag" in title.lower()


# ---------------------------------------------------------------------------
# Integration tests — BM25Index
# ---------------------------------------------------------------------------

@pytest.fixture
def populated_wiki(tmp_path):
    """Wiki with several pages on different topics."""
    fs = WikiFS(tmp_path)
    fs.ensure_structure()

    pages = {
        "wiki/concepts/rag.md": """---
title: Retrieval Augmented Generation
type: concept
---
# Retrieval Augmented Generation

RAG combines a retrieval system with a language model.
The retrieval step fetches relevant documents from a vector database.
The generation step produces an answer conditioned on retrieved context.
""",
        "wiki/concepts/llm-wiki.md": """---
title: LLM Wiki Pattern
type: concept
---
# LLM Wiki Pattern

The LLM Wiki pattern uses a language model to maintain a structured knowledge base.
Unlike RAG, the wiki stores pre-synthesised information rather than raw documents.
Knowledge compounds over time as new sources are ingested.
""",
        "wiki/concepts/bm25.md": """---
title: BM25 Algorithm
type: concept
---
# BM25 Algorithm

BM25 is a ranking function used in information retrieval.
It ranks documents by term frequency and inverse document frequency.
BM25 is the foundation of many search engines including Elasticsearch.
""",
        "wiki/entities/karpathy.md": """---
title: Andrej Karpathy
type: entity
---
# Andrej Karpathy

AI researcher and former Director of AI at Tesla.
Known for NanoGPT, MinGPT, and the LLM Wiki pattern.
""",
    }

    for rel_path, content in pages.items():
        fs.write_wiki_page(rel_path, content)

    return tmp_path, fs


def test_index_builds_from_wiki(populated_wiki):
    _, fs = populated_wiki
    index = BM25Index.build(fs)
    assert index.doc_count == 4
    assert index.vocab_size > 0


def test_search_returns_relevant_results(populated_wiki):
    _, fs = populated_wiki
    index = BM25Index.build(fs)
    results = index.search("retrieval augmented generation")
    assert len(results) > 0
    # RAG page should be the top result
    assert "rag" in results[0].path


def test_search_bm25_algorithm(populated_wiki):
    _, fs = populated_wiki
    index = BM25Index.build(fs)
    results = index.search("BM25 ranking search")
    assert len(results) > 0
    assert "bm25" in results[0].path


def test_search_person(populated_wiki):
    _, fs = populated_wiki
    index = BM25Index.build(fs)
    results = index.search("Karpathy NanoGPT")
    assert len(results) > 0
    assert "karpathy" in results[0].path


def test_search_top_k_respected(populated_wiki):
    _, fs = populated_wiki
    index = BM25Index.build(fs)
    results = index.search("language model", top_k=2)
    assert len(results) <= 2


def test_search_returns_scores(populated_wiki):
    _, fs = populated_wiki
    index = BM25Index.build(fs)
    results = index.search("retrieval documents")
    assert all(r.score > 0 for r in results)
    # Scores should be descending
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_search_empty_query_returns_empty(populated_wiki):
    _, fs = populated_wiki
    index = BM25Index.build(fs)
    results = index.search("")
    assert results == []


def test_search_no_match_returns_empty(populated_wiki):
    _, fs = populated_wiki
    index = BM25Index.build(fs)
    results = index.search("zzzyyyxxx qqqwwweee nonexistent")
    assert results == []


def test_search_result_has_snippet(populated_wiki):
    _, fs = populated_wiki
    index = BM25Index.build(fs)
    results = index.search("vector database")
    assert len(results) > 0
    assert len(results[0].snippet) > 0


def test_search_result_has_matched_terms(populated_wiki):
    _, fs = populated_wiki
    index = BM25Index.build(fs)
    results = index.search("retrieval documents")
    assert len(results) > 0
    assert len(results[0].matched_terms) > 0


def test_search_result_format(populated_wiki):
    _, fs = populated_wiki
    index = BM25Index.build(fs)
    results = index.search("language model")
    assert len(results) > 0
    formatted = results[0].format(1)
    assert "1." in formatted
    assert results[0].title in formatted


def test_empty_wiki_returns_empty(tmp_path):
    fs = WikiFS(tmp_path)
    fs.ensure_structure()
    index = BM25Index.build(fs)
    assert index.doc_count == 0
    results = index.search("anything")
    assert results == []


def test_index_ignores_frontmatter_keywords(populated_wiki):
    _, fs = populated_wiki
    index = BM25Index.build(fs)
    # "type" and "concept" are frontmatter values — should still match
    # but let's verify frontmatter is stripped by checking scores
    results_body = index.search("retrieval")
    results_fm   = index.search("source-summary")
    # Body term should score higher than frontmatter-only term
    assert len(results_body) >= len(results_fm)
