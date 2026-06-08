"""
Source chunker.

Splits large source documents into overlapping text windows so the ingest
operation can safely handle files that would otherwise exceed the LLM's
context window.

Strategy:
  - Estimate token count (rough: 1 token ≈ 4 chars)
  - If under threshold → return as-is (single chunk)
  - If over threshold → split on paragraph boundaries with overlap

Zero dependencies — stdlib only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


# Rough chars-per-token estimate (conservative — works across models)
_CHARS_PER_TOKEN = 4

# Default safe payload size: leave room for system prompt + response
_DEFAULT_MAX_TOKENS = 6000
_DEFAULT_OVERLAP_TOKENS = 400


@dataclass
class Chunk:
    index: int  # 0-based chunk number
    total: int  # total number of chunks
    text: str  # chunk content
    char_start: int  # character offset in original text
    char_end: int  # character offset in original text

    @property
    def is_first(self) -> bool:
        return self.index == 0

    @property
    def is_last(self) -> bool:
        return self.index == self.total - 1

    @property
    def header(self) -> str:
        if self.total == 1:
            return ""
        return f"[Part {self.index + 1} of {self.total}]\n\n"

    @property
    def text_with_header(self) -> str:
        return self.header + self.text


def estimate_tokens(text: str) -> int:
    """Fast, dependency-free token estimate."""
    return max(1, len(text) // _CHARS_PER_TOKEN)


def needs_chunking(text: str, max_tokens: int = _DEFAULT_MAX_TOKENS) -> bool:
    return estimate_tokens(text) > max_tokens


def chunk_text(
    text: str,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
    overlap_tokens: int = _DEFAULT_OVERLAP_TOKENS,
) -> list[Chunk]:
    """
    Split text into overlapping chunks, preferring paragraph boundaries.

    Returns a list of Chunk objects. If no chunking is needed, returns
    a single-element list.
    """
    if not needs_chunking(text, max_tokens):
        return [Chunk(index=0, total=1, text=text, char_start=0, char_end=len(text))]

    max_chars = max_tokens * _CHARS_PER_TOKEN
    overlap_chars = overlap_tokens * _CHARS_PER_TOKEN

    # Split into paragraphs (blank-line separated)
    paragraphs = _split_paragraphs(text)

    chunks: list[Chunk] = []
    current_chars: list[str] = []
    current_len = 0
    char_cursor = 0
    chunk_start = 0

    for para in paragraphs:
        para_len = len(para)

        # If a single paragraph is bigger than the window, split it by sentences
        if para_len > max_chars:
            sub_chunks = _split_by_sentences(para, max_chars, overlap_chars)
            for sub in sub_chunks:
                chunks.append(
                    Chunk(
                        index=len(chunks),
                        total=0,  # filled in after
                        text=sub,
                        char_start=char_cursor,
                        char_end=char_cursor + len(sub),
                    )
                )
            char_cursor += para_len
            current_chars = []
            current_len = 0
            chunk_start = char_cursor
            continue

        if current_len + para_len > max_chars and current_chars:
            # Flush current chunk
            chunk_text_str = "\n\n".join(current_chars)
            chunks.append(
                Chunk(
                    index=len(chunks),
                    total=0,
                    text=chunk_text_str,
                    char_start=chunk_start,
                    char_end=char_cursor,
                )
            )
            # Start next chunk with overlap — carry last N chars worth of paragraphs
            overlap_paras = _take_tail(current_chars, overlap_chars)
            current_chars = overlap_paras
            current_len = sum(len(p) for p in current_chars)
            chunk_start = char_cursor - sum(len(p) for p in overlap_paras)

        current_chars.append(para)
        current_len += para_len
        char_cursor += para_len + 2  # +2 for the \n\n separator

    # Flush remaining
    if current_chars:
        chunk_text_str = "\n\n".join(current_chars)
        chunks.append(
            Chunk(
                index=len(chunks),
                total=0,
                text=chunk_text_str,
                char_start=chunk_start,
                char_end=char_cursor,
            )
        )

    # Fill in total
    total = len(chunks)
    for i, c in enumerate(chunks):
        c.index = i
        c.total = total

    return chunks


def merge_ingest_responses(responses: list[str], source_name: str) -> str:
    """
    Merge multiple ingest LLM responses (one per chunk) into a single
    response that the ingest parser can handle.

    Strategy: take TAKEAWAYS from all chunks, SOURCE_SUMMARY_PAGE from
    the first chunk, NEW_PAGES and UPDATED_PAGES accumulated across all,
    INDEX_UPDATE and LOG_ENTRY from the last chunk.
    """
    if len(responses) == 1:
        return responses[0]

    all_takeaways: list[str] = []
    source_summary = ""
    all_new_pages: list[str] = []
    all_updated_pages: list[str] = []
    index_update = ""
    log_entry = ""

    for i, resp in enumerate(responses):
        sections = _split_sections(resp)
        if sections.get("TAKEAWAYS"):
            all_takeaways.append(f"[Part {i + 1}]\n{sections['TAKEAWAYS'].strip()}")
        if i == 0 and sections.get("SOURCE_SUMMARY_PAGE"):
            source_summary = sections["SOURCE_SUMMARY_PAGE"].strip()
        if sections.get("NEW_PAGES"):
            all_new_pages.append(sections["NEW_PAGES"].strip())
        if sections.get("UPDATED_PAGES"):
            all_updated_pages.append(sections["UPDATED_PAGES"].strip())
        if sections.get("INDEX_UPDATE"):
            index_update = sections["INDEX_UPDATE"].strip()
        if sections.get("LOG_ENTRY"):
            log_entry = sections["LOG_ENTRY"].strip()

    parts = [
        "### TAKEAWAYS",
        "\n".join(all_takeaways),
        "",
        "### SOURCE_SUMMARY_PAGE",
        source_summary,
        "",
        "### PAGES_TO_UPDATE",
        "",
        "### UPDATED_PAGES",
        "\n\n".join(all_updated_pages),
        "",
        "### NEW_PAGES",
        "\n\n".join(all_new_pages),
        "",
        "### INDEX_UPDATE",
        index_update,
        "",
        "### LOG_ENTRY",
        log_entry or f"ingest | {source_name} | multi-chunk ({len(responses)} parts)",
    ]
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _split_paragraphs(text: str) -> list[str]:
    """Split on one or more blank lines."""
    paras = re.split(r"\n{2,}", text.strip())
    return [p.strip() for p in paras if p.strip()]


def _split_by_sentences(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    """Fallback: split a very long paragraph by sentence boundaries."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for sent in sentences:
        if current_len + len(sent) > max_chars and current:
            chunks.append(" ".join(current))
            overlap = _take_tail(current, overlap_chars)
            current = overlap
            current_len = sum(len(s) for s in current)
        current.append(sent)
        current_len += len(sent)
    if current:
        chunks.append(" ".join(current))
    return chunks


def _take_tail(items: list[str], max_chars: int) -> list[str]:
    """Return items from the end of the list that fit within max_chars."""
    result: list[str] = []
    total = 0
    for item in reversed(items):
        if total + len(item) > max_chars:
            break
        result.insert(0, item)
        total += len(item)
    return result


def _split_sections(text: str) -> dict[str, str]:
    sections: dict[str, str] = {}
    current_key = None
    current_lines: list[str] = []
    for line in text.splitlines():
        header = re.match(r"^###\s+([A-Z_]+)\s*$", line.strip())
        if header:
            if current_key:
                sections[current_key] = "\n".join(current_lines).strip()
            current_key = header.group(1)
            current_lines = []
        elif current_key:
            current_lines.append(line)
    if current_key:
        sections[current_key] = "\n".join(current_lines).strip()
    return sections
