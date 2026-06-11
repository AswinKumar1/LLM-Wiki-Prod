"""
Ingest operation — Day 3 upgrade.

New in Day 3:
  - PDF ingestion: wiki ingest raw/paper.pdf
  - URL ingestion: wiki ingest --url https://...
  - Source reader routing (text / pdf / url → unified text pipeline)

Everything else (chunking, retry, usage tracking) carried forward from Day 2.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..providers.base import LLMProvider
from ..wiki_fs import WikiFS
from ..prompts import ingest_source
from ..utils.chunker import chunk_text, merge_ingest_responses
from ..utils.usage import UsageTracker
from ..utils.source_reader import read_source, is_url

_MAX_RETRIES = 3
_RETRY_DELAYS = [2, 5, 10]


@dataclass
class IngestResult:
    source_name: str
    pages_created: list[str] = field(default_factory=list)
    pages_updated: list[str] = field(default_factory=list)
    takeaways: str = ""
    tokens_used: int = 0
    cost_usd: float = 0.0
    chunks_processed: int = 1
    source_type: str = "text"  # "text" | "pdf" | "url"
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None


class IngestOperation:
    """
    Process one or more raw source files (or URLs) into wiki pages.

    Usage:
        op = IngestOperation(provider, wiki_fs)
        results = op.run()                           # all new raw/ sources
        result  = op.run_one(path)                   # specific file
        result  = op.run_url("https://arxiv.org/…") # fetch + ingest URL
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
        """Ingest all new un-processed files from raw/."""
        sources = self.fs.new_raw_sources()
        if not sources:
            return []
        if max_sources:
            sources = sources[:max_sources]
        return [self.run_one(src) for src in sources]

    def run_one(self, source_path: Path) -> IngestResult:
        """Ingest a single file — auto-detects PDF vs text."""
        source_name = source_path.name
        result = IngestResult(source_name=source_name)

        if self.verbose:
            print(f"  ingesting: {source_name} ...")

        # Route to correct reader
        text, _, method = read_source(source_path)
        result.source_type = method

        if text is None:
            result.error = method  # method holds the error message for failures
            return result

        return self._ingest_text(text, source_name, result)

    def run_url(self, url: str) -> IngestResult:
        """
        Fetch a URL, save to raw/articles/, then ingest.
        Returns an IngestResult with source_type='url'.
        """
        if self.verbose:
            print(f"  fetching: {url} ...")

        text, filename, method = read_source(url)
        if text is None:
            return IngestResult(
                source_name=url,
                source_type="url",
                error=method,
            )

        # Save fetched content to raw/articles/ so it's tracked
        raw_path = self.fs.raw_dir / "articles" / filename
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(text, encoding="utf-8")

        if self.verbose:
            print(f"  saved to: raw/articles/{filename}")

        result = IngestResult(source_name=filename, source_type="url")
        return self._ingest_text(text, filename, result)

    # ------------------------------------------------------------------
    # Shared ingest pipeline
    # ------------------------------------------------------------------

    def _ingest_text(self, text: str, source_name: str, result: IngestResult) -> IngestResult:
        source_sha = _sha256_text(text)
        index_content = self.fs.read_index()

        # Chunk if needed
        chunks = chunk_text(text, max_tokens=self.max_tokens_per_chunk)
        result.chunks_processed = len(chunks)

        if self.verbose and len(chunks) > 1:
            print(f"    → {len(chunks)} chunks")

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

            cost = self._tracker.record(
                op="ingest",
                provider=self.provider.provider_name,
                model=self.provider.model_name,
                prompt_tokens=tokens // 2,
                completion_tokens=tokens // 2,
                source=source_name,
            )
            result.cost_usd += cost

        merged = merge_ingest_responses(chunk_responses, source_name)
        self._parse_and_write(merged, source_name, source_sha, result)
        return result

    # ------------------------------------------------------------------
    # Retry wrapper
    # ------------------------------------------------------------------

    def _call_with_retry(
        self, system: str, user: str, source_name: str
    ) -> tuple[Optional[str], int]:
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
                        print(f"    ⚠ retry {attempt + 1} ({exc}), waiting {delay}s ...")
                    time.sleep(delay)
        if self.verbose:
            print(f"    ✗ retries exhausted for {source_name}: {last_exc}")
        return None, 0

    # ------------------------------------------------------------------
    # Parse & write
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
        self.fs.append_log(
            log_entry
            or (
                f"ingest | {source_name} | "
                f"{len(result.pages_created)} created, "
                f"{len(result.pages_updated)} updated | "
                f"{result.tokens_used} tokens"
            )
        )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

import re as _re
import hashlib as _hashlib


def _sha256_text(text: str) -> str:
    return _hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


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


def _inject_sha(page: str, sha: str) -> str:
    if "source_sha:" not in page and "---" in page:
        page = page.replace("---\n", f"---\nsource_sha: {sha}\n", 1)
    return page


def _slugify(name: str) -> str:
    stem = Path(name).stem
    return _re.sub(r"[^a-z0-9-]", "-", stem.lower()).strip("-")
