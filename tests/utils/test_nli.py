"""Tests for the NLI contradiction detection engine."""

import pytest
from llm_wiki.providers.base import LLMProvider, LLMResponse, ProviderConfig
from llm_wiki.utils.nli import (
    NLIEngine,
    NLIResult,
    ContradictionPair,
    downgrade_confidence,
    _extract_claims,
    _find_related_pairs,
    _cross_claims,
    _cross_encoder_available,
)


# ---------------------------------------------------------------------------
# Mock provider
# ---------------------------------------------------------------------------

class MockNLIProvider(LLMProvider):
    """
    Provider that simulates NLI responses.
    Returns CONTRADICTION for pairs containing trigger words,
    NEUTRAL otherwise.
    """
    def __init__(self, trigger_words=None):
        super().__init__(ProviderConfig())
        self._triggers = trigger_words or ["latency", "speed", "fast", "slow"]
        self.call_count = 0

    def chat(self, system, user, *, temperature=None, max_tokens=None):
        self.call_count += 1
        # Check if both claims contain conflicting trigger words
        content = user.lower()
        if "contradiction" in system.lower():
            # NLI classification request
            if any(t in content for t in self._triggers):
                response = "CONTRADICTION"
            else:
                response = "NEUTRAL"
        else:
            response = "NEUTRAL"
        return LLMResponse(
            content=response, model="mock", provider="mock",
            prompt_tokens=20, completion_tokens=5,
        )

    def health_check(self): return True

    @property
    def provider_name(self): return "mock"


# ---------------------------------------------------------------------------
# Unit tests — claim extraction
# ---------------------------------------------------------------------------

def test_extract_claims_basic():
    content = """---
title: RAG Overview
type: concept
---
# RAG Overview

RAG uses a retrieval system to fetch documents.
The system returns results in under 500ms.
Knowledge compounds over multiple ingests.
"""
    claims = _extract_claims(content)
    assert len(claims) > 0
    # Should not include frontmatter content as claims
    assert not any("title:" in c for c in claims)


def test_extract_claims_strips_frontmatter():
    content = "---\ntitle: Test\ntype: concept\n---\n\nThis is a factual statement that uses the verb is."
    claims = _extract_claims(content)
    assert not any("title:" in c for c in claims)
    assert any("factual" in c.lower() for c in claims)


def test_extract_claims_filters_short_sentences():
    content = "---\ntitle: T\ntype: concept\n---\n\nShort.\nThis is a proper factual claim that uses the verb is and has enough length."
    claims = _extract_claims(content)
    assert not any(c == "Short." for c in claims)


def test_extract_claims_removes_headings():
    content = "---\ntitle: T\ntype: concept\n---\n\n# Big Heading\n## Sub Heading\n\nThis sentence is a real claim with enough content."
    claims = _extract_claims(content)
    assert not any("#" in c for c in claims)


def test_extract_claims_caps_at_50():
    # Create a page with many sentences
    sentences = [f"This is factual sentence number {i} which uses the verb is." for i in range(100)]
    content = "---\ntitle: T\ntype: concept\n---\n\n" + " ".join(sentences)
    claims = _extract_claims(content)
    assert len(claims) <= 50


def test_extract_claims_requires_verb():
    content = "---\ntitle: T\ntype: concept\n---\n\nJust a noun phrase with no verb whatsoever in the entire thing here shown.\nThis claim uses the verb is and should be included in the results."
    claims = _extract_claims(content)
    # The second sentence has a verb, should be included
    assert any("uses" in c for c in claims)


# ---------------------------------------------------------------------------
# Unit tests — related pairs
# ---------------------------------------------------------------------------

def test_find_related_pairs_via_wikilinks():
    pages = {
        "wiki/concepts/rag.md":    "RAG uses [[vector-database]] for retrieval.",
        "wiki/concepts/search.md": "Search also uses [[vector-database]] indexes.",
        "wiki/concepts/llm.md":    "LLMs generate text without retrieval.",
    }
    paths   = list(pages.keys())
    related = _find_related_pairs(paths, pages)
    # rag and search share [[vector-database]]
    assert any(
        ("rag" in a and "search" in b) or ("search" in a and "rag" in b)
        for a, b in related
    )
    # llm has no shared links with the others
    assert not any("llm" in a or "llm" in b for a, b in related)


def test_find_related_pairs_fallback_empty():
    """When no wikilinks exist, returns empty list (caller handles fallback)."""
    pages = {
        "wiki/a.md": "No wikilinks here at all.",
        "wiki/b.md": "Also no wikilinks.",
    }
    related = _find_related_pairs(list(pages.keys()), pages)
    assert related == []


# ---------------------------------------------------------------------------
# Unit tests — cross claims
# ---------------------------------------------------------------------------

def test_cross_claims_basic():
    claims_a = ["RAG latency is typically 1 to 3 seconds for retrieval."]
    claims_b = ["Our system responds in under 100 milliseconds consistently."]
    pairs    = _cross_claims(claims_a, claims_b)
    assert len(pairs) == 1
    assert pairs[0] == (claims_a[0], claims_b[0])


def test_cross_claims_caps_at_max():
    claims_a = [f"Claim A{i} uses the verb is." for i in range(15)]
    claims_b = [f"Claim B{i} uses the verb is." for i in range(15)]
    pairs    = _cross_claims(claims_a, claims_b, max_pairs=20)
    assert len(pairs) <= 20


# ---------------------------------------------------------------------------
# Unit tests — NLI engine (LLM backend)
# ---------------------------------------------------------------------------

def test_nli_engine_uses_llm_backend_when_no_cross_encoder():
    provider = MockNLIProvider()
    engine   = NLIEngine(provider, backend="llm")
    assert engine.backend_name == "llm"


def test_nli_engine_score_single_contradiction():
    provider = MockNLIProvider(trigger_words=["latency"])
    engine   = NLIEngine(provider, backend="llm")
    score    = engine.score_single(
        "RAG latency is 1-3 seconds.",
        "The system has sub-100ms latency.",
    )
    assert score == 1.0


def test_nli_engine_score_single_neutral():
    provider = MockNLIProvider(trigger_words=["xyz_never_matches"])
    engine   = NLIEngine(provider, backend="llm")
    score    = engine.score_single(
        "RAG uses a vector database.",
        "The sky is blue.",
    )
    assert score == 0.0


def test_nli_scan_pages_finds_contradiction():
    pages = {
        "wiki/concepts/rag.md": """---
title: RAG
type: concept
---
# RAG

RAG uses [[vector-database]] retrieval to fetch documents.
The latency for RAG retrieval is typically between 1 and 3 seconds.
""",
        "wiki/concepts/search.md": """---
title: Search
type: concept
---
# Search

Search uses [[vector-database]] indexes for fast lookups.
Our search system has latency under 50 milliseconds for all queries.
""",
    }
    provider = MockNLIProvider(trigger_words=["latency"])
    engine   = NLIEngine(provider, backend="llm")
    result   = engine.scan_pages(pages)

    assert result.success
    assert result.pages_checked == 2
    assert result.contradiction_count > 0
    c = result.contradictions[0]
    assert c.backend == "llm"
    assert c.score == 1.0


def test_nli_scan_pages_no_contradiction():
    pages = {
        "wiki/concepts/rag.md": """---
title: RAG
type: concept
---
# RAG

RAG uses [[vector-database]] retrieval to find documents.
The system supports multiple embedding models for indexing.
""",
        "wiki/concepts/search.md": """---
title: Search
type: concept
---
# Search

Search uses [[vector-database]] indexes for retrieval operations.
The index supports incremental updates without full rebuilds.
""",
    }
    provider = MockNLIProvider(trigger_words=["xyz_never_matches"])
    engine   = NLIEngine(provider, backend="llm")
    result   = engine.scan_pages(pages)

    assert result.success
    assert result.contradiction_count == 0


def test_nli_scan_pages_needs_two_pages():
    pages  = {"wiki/only.md": "---\ntitle: T\ntype: concept\n---\nSingle page uses the verb is."}
    engine = NLIEngine(MockNLIProvider(), backend="llm")
    result = engine.scan_pages(pages)
    assert not result.success
    assert "2 pages" in result.error or "least" in result.error


def test_nli_result_pages_with_contradictions():
    result = NLIResult()
    result.contradictions = [
        ContradictionPair(
            page_a="wiki/a.md", page_b="wiki/b.md",
            claim_a="A is fast.", claim_b="A is slow.",
            score=0.9, backend="llm",
        )
    ]
    affected = result.pages_with_contradictions()
    assert "wiki/a.md" in affected
    assert "wiki/b.md" in affected


# ---------------------------------------------------------------------------
# Unit tests — confidence downgrader
# ---------------------------------------------------------------------------

def test_downgrade_high_to_medium():
    content = "---\ntitle: Test\ntype: concept\nconfidence: high\n---\n\n# Test\n"
    new, changed = downgrade_confidence(content)
    assert changed
    assert "confidence: medium" in new


def test_downgrade_medium_to_low():
    content = "---\ntitle: Test\ntype: concept\nconfidence: medium\n---\n\n# Test\n"
    new, changed = downgrade_confidence(content)
    assert changed
    assert "confidence: low" in new


def test_downgrade_low_stays_low():
    content = "---\ntitle: Test\ntype: concept\nconfidence: low\n---\n\n# Test\n"
    new, changed = downgrade_confidence(content)
    # low stays low — changed should be False since value didn't change
    assert "confidence: low" in new


def test_downgrade_no_confidence_field():
    content = "---\ntitle: Test\ntype: concept\n---\n\n# Test\n"
    new, changed = downgrade_confidence(content)
    assert not changed
    assert new == content


def test_downgrade_preserves_rest_of_content():
    content = (
        "---\ntitle: My Page\ntype: concept\nconfidence: high\n"
        "created: 2025-06-01\n---\n\n# My Page\n\nImportant content here.\n"
    )
    new, changed = downgrade_confidence(content)
    assert changed
    assert "title: My Page" in new
    assert "created: 2025-06-01" in new
    assert "Important content here." in new


# ---------------------------------------------------------------------------
# ContradictionPair formatting
# ---------------------------------------------------------------------------

def test_contradiction_pair_format():
    c = ContradictionPair(
        page_a="wiki/concepts/rag.md",
        page_b="wiki/concepts/llm-wiki.md",
        claim_a="RAG latency is 1-3 seconds.",
        claim_b="The system responds in under 100ms.",
        score=0.92,
        backend="llm",
    )
    formatted = c.format()
    assert "wiki/concepts/rag.md" in formatted
    assert "wiki/concepts/llm-wiki.md" in formatted
    assert "0.92" in formatted
    assert "RAG latency" in formatted


# ---------------------------------------------------------------------------
# Cross-encoder availability
# ---------------------------------------------------------------------------

def test_cross_encoder_available_returns_bool():
    result = _cross_encoder_available()
    assert isinstance(result, bool)


def test_nli_engine_cross_encoder_raises_if_unavailable():
    if _cross_encoder_available():
        pytest.skip("sentence-transformers is installed — skip unavailability test")
    provider = MockNLIProvider()
    with pytest.raises(ImportError, match="sentence-transformers"):
        NLIEngine(provider, backend="cross_encoder")


def test_nli_engine_auto_falls_back_to_llm():
    """Auto backend should use llm when cross_encoder is not available."""
    if _cross_encoder_available():
        pytest.skip("sentence-transformers installed — auto picks cross_encoder")
    provider = MockNLIProvider()
    engine   = NLIEngine(provider, backend="auto")
    assert engine.backend_name == "llm"
