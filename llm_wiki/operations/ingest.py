"""
Ingest operation — Day 2 upgrade.

New in Day 2:
  - Automatic chunking for large source files (>6k tokens)
  - Retry with exponential backoff on LLM errors
  - Usage tracking (token counts written to wiki/usage.json)
  - Chunk merge so the wiki gets consistent pages regardless of file size
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..providers.base import LLMProvider
from ..wiki_fs import WikiFS
from ..prompts import ingest_source
from ..utils.chunker import chunk_text, needs_chunking, merge_ingest_responses
from ..utils.usage import UsageTracker

# Retry config
_MAX_RETRIES = 3
_RETRY_DELAYS = [2, 5, 10]  # seconds between retries


@dataclass
class IngestResult:
    source_name: str
    pages_created: list[str] = field(default_factory=list)
    pages_updated: list[str] = field(default_factory=list)
    takeaways: str = ""
    tokens_used: int = 0
    cost_usd: float = 0.0
    chunks_processed: int = 1
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None


class IngestOperation:
    """
    Process one or more raw source files into wiki pages.

    Usage:
        op = IngestOperation(provider, wiki_fs)
        results = op.run()                      # all new sources
        result  = op.run_one(path_to_source)    # specific file
    """

    def __init__(
        self,
        provider: LLMProvider,
        wiki_fs: WikiFS,
        verbose: bool = False,
        max_tokens_per_chunk: int = 6000,
    ):
        self.provider = provider
        self.fs = wiki_fs
        self.verbose = verbose
        self.max_tokens_per_chunk = max_tokens_per_chunk
        self._tracker = UsageTracker(wiki_fs.root)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, max_sources: Optional[int] = None) -> list[IngestResult]:
        sources = self.fs.new_raw_sources()
        if not sources:
            return []
        if max_sources:
            sources = sources[:max_sources]
        return [self.run_one(src) for src in sources]

    def run_one(self, source_path: Path) -> IngestResult:
        source_name = source_path.name
        result = IngestResult(source_name=source_name)

        if self.verbose:
            print(f"  ingesting: {source_name} ...")

        source_text = self.fs.read_source(source_path)
        source_sha = self.fs.source_sha256(source_path)
        index_content = self.fs.read_index()

        # ----- Chunking ------------------------------------------------
        chunks = chunk_text(source_text, max_tokens=self.max_tokens_per_chunk)
        result.chunks_processed = len(chunks)

        if self.verbose and len(chunks) > 1:
            print(f"    → splitting into {len(chunks)} chunks")

        # ----- LLM calls (one per chunk, with retry) -------------------
        chunk_responses: list[str] = []
        for chunk in chunks:
            if self.verbose and len(chunks) > 1:
                print(f"    → chunk {chunk.index + 1}/{chunk.total} ...")

            system, user = ingest_source(chunk.text_with_header, source_name, index_content)
            response, tokens = self._call_with_retry(system, user, source_name)

            if response is None:
                result.error = f"LLM call failed after {_MAX_RETRIES} retries"
                return result

            chunk_responses.append(response)
            result.tokens_used += tokens

            # Track usage per chunk
            cost = self._tracker.record(
                op="ingest",
                provider=self.provider.provider_name,
                model=self.provider.model_name,
                prompt_tokens=tokens // 2,  # rough split (exact not available per-chunk)
                completion_tokens=tokens // 2,
                source=source_name,
            )
            result.cost_usd += cost

        # ----- Merge multi-chunk responses ----------------------------
        merged = merge_ingest_responses(chunk_responses, source_name)
        self._parse_and_write(merged, source_name, source_sha, result)
        return result

    # ------------------------------------------------------------------
    # Retry wrapper
    # ------------------------------------------------------------------

    def _call_with_retry(
        self, system: str, user: str, source_name: str
    ) -> tuple[Optional[str], int]:
        """
        Call provider.chat() with exponential backoff.
        Returns (content, total_tokens) or (None, 0) on permanent failure.
        """
        last_exc = None
        for attempt in range(_MAX_RETRIES):
            try:
                response = self.provider.chat(system, user)
                return response.content, response.total_tokens
            except Exception as exc:
                last_exc = exc
                if attempt < _MAX_RETRIES - 1:
                    delay = _RETRY_DELAYS[attempt]
                    if self.verbose:
                        print(
                            f"    ⚠ attempt {attempt + 1} failed ({exc}), retrying in {delay}s ..."
                        )
                    time.sleep(delay)

        if self.verbose:
            print(f"    ✗ all retries exhausted for {source_name}: {last_exc}")
        return None, 0

    # ------------------------------------------------------------------
    # Parse & write (unchanged from Day 1, kept here for self-containment)
    # ------------------------------------------------------------------

    def _parse_and_write(
        self,
        llm_output: str,
        source_name: str,
        source_sha: str,
        result: IngestResult,
    ) -> None:
        import re

        sections = _split_sections(llm_output)

        result.takeaways = sections.get("TAKEAWAYS", "").strip()

        summary = sections.get("SOURCE_SUMMARY_PAGE", "").strip()
        if summary:
            summary = _inject_sha(summary, source_sha)
            slug = _slugify(source_name)
            path = f"wiki/sources/{slug}.md"
            self.fs.write_wiki_page(path, summary)
            result.pages_created.append(path)

        for path, content in _parse_multipage_block(sections.get("UPDATED_PAGES", "")):
            self.fs.write_wiki_page(path, content)
            result.pages_updated.append(path)

        for path, content in _parse_multipage_block(sections.get("NEW_PAGES", "")):
            self.fs.write_wiki_page(path, content)
            result.pages_created.append(path)

        index_update = sections.get("INDEX_UPDATE", "").strip()
        if index_update:
            self.fs.write_index(index_update)

        log_entry = sections.get("LOG_ENTRY", "").strip()
        if log_entry:
            self.fs.append_log(log_entry)
        else:
            self.fs.append_log(
                f"ingest | {source_name} | "
                f"{len(result.pages_created)} created, "
                f"{len(result.pages_updated)} updated | "
                f"{result.tokens_used} tokens"
            )


# ---------------------------------------------------------------------------
# Parsing utilities (duplicated from Day 1 to keep this file standalone)
# ---------------------------------------------------------------------------

import re as _re


def _split_sections(text: str) -> dict[str, str]:
    sections: dict[str, str] = {}
    current_key = None
    current_lines: list[str] = []
    for line in text.splitlines():
        header = _re.match(r"^###\s+([A-Z_]+)\s*$", line.strip())
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


def _parse_multipage_block(text: str) -> list[tuple[str, str]]:
    pages = []
    current_path = None
    current_lines: list[str] = []
    for line in text.splitlines():
        path_match = _re.match(r"^<path:\s*(.+?)>\s*$", line.strip())
        if path_match:
            if current_path and current_lines:
                pages.append((current_path, "\n".join(current_lines).strip()))
            current_path = path_match.group(1).strip()
            current_lines = []
        elif current_path:
            current_lines.append(line)
    if current_path and current_lines:
        pages.append((current_path, "\n".join(current_lines).strip()))
    return pages


def _inject_sha(frontmatter_page: str, sha: str) -> str:
    if "source_sha:" not in frontmatter_page and "---" in frontmatter_page:
        frontmatter_page = frontmatter_page.replace("---\n", f"---\nsource_sha: {sha}\n", 1)
    return frontmatter_page


def _slugify(name: str) -> str:
    stem = Path(name).stem
    return _re.sub(r"[^a-z0-9-]", "-", stem.lower()).strip("-")
