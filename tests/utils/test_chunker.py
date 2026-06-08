"""Tests for the source chunker."""

import pytest
from llm_wiki.utils.chunker import (
    estimate_tokens,
    needs_chunking,
    chunk_text,
    merge_ingest_responses,
    Chunk,
)


def test_estimate_tokens_basic():
    text = "a" * 400  # 400 chars → 100 tokens
    assert estimate_tokens(text) == 100


def test_needs_chunking_false_for_small():
    text = "short text " * 50  # ~550 chars → ~137 tokens
    assert not needs_chunking(text, max_tokens=6000)


def test_needs_chunking_true_for_large():
    text = "word " * 10000  # 50k chars → ~12500 tokens
    assert needs_chunking(text, max_tokens=6000)


def test_chunk_text_single_chunk_for_small():
    text = "This is a small document.\n\nIt has two paragraphs."
    chunks = chunk_text(text, max_tokens=6000)
    assert len(chunks) == 1
    assert chunks[0].text == text
    assert chunks[0].is_first
    assert chunks[0].is_last
    assert chunks[0].total == 1


def test_chunk_text_splits_large_text():
    # Create text larger than 6000 tokens (>24000 chars)
    paragraph = "This is a paragraph with enough words to fill space. " * 10
    text = "\n\n".join([paragraph] * 60)  # ~60 paragraphs
    chunks = chunk_text(text, max_tokens=6000, overlap_tokens=400)
    assert len(chunks) > 1
    # All chunks should have some content
    for chunk in chunks:
        assert len(chunk.text) > 0
        assert chunk.total == len(chunks)


def test_chunk_indices_are_correct():
    paragraph = "word " * 200
    text = "\n\n".join([paragraph] * 40)
    chunks = chunk_text(text, max_tokens=2000, overlap_tokens=200)
    for i, chunk in enumerate(chunks):
        assert chunk.index == i
        assert chunk.total == len(chunks)


def test_chunk_header():
    paragraph = "word " * 200
    text = "\n\n".join([paragraph] * 40)
    chunks = chunk_text(text, max_tokens=2000, overlap_tokens=200)
    if len(chunks) > 1:
        assert "[Part 1 of" in chunks[0].header
        assert chunks[0].text_with_header.startswith("[Part 1 of")


def test_single_chunk_has_no_header():
    text = "small text"
    chunks = chunk_text(text)
    assert chunks[0].header == ""
    assert chunks[0].text_with_header == "small text"


def test_merge_single_response_passthrough():
    response = "### TAKEAWAYS\n- point\n\n### LOG_ENTRY\ntest"
    merged = merge_ingest_responses([response], "test.md")
    assert merged == response


def test_merge_multiple_responses():
    r1 = """### TAKEAWAYS
- Point from part 1

### SOURCE_SUMMARY_PAGE
---
title: test
type: source-summary
---
# Test

### PAGES_TO_UPDATE

### UPDATED_PAGES

### NEW_PAGES

### INDEX_UPDATE
# Index

### LOG_ENTRY
ingest | test | 1 created"""

    r2 = """### TAKEAWAYS
- Point from part 2

### SOURCE_SUMMARY_PAGE
(ignored)

### PAGES_TO_UPDATE

### UPDATED_PAGES

### NEW_PAGES
<path: wiki/concepts/new-concept.md>
---
title: New concept
---
Content.

### INDEX_UPDATE
# Index (updated)

### LOG_ENTRY
ingest | test | 0 created"""

    merged = merge_ingest_responses([r1, r2], "test.md")
    assert "Part 1" in merged
    assert "Part 2" in merged
    assert "SOURCE_SUMMARY_PAGE" in merged
    assert "new-concept.md" in merged
    # INDEX_UPDATE should be from last chunk
    assert "# Index (updated)" in merged
