"""
Ingest operation.

Reads unprocessed files from raw/, calls the LLM to synthesise wiki pages,
and writes the results back to wiki/.

The operation is provider-agnostic — it only uses the LLMProvider interface.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..providers.base import LLMProvider
from ..wiki_fs import WikiFS
from ..prompts import ingest_source


@dataclass
class IngestResult:
    source_name: str
    pages_created: list[str] = field(default_factory=list)
    pages_updated: list[str] = field(default_factory=list)
    takeaways: str = ""
    tokens_used: int = 0
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None


class IngestOperation:
    """
    Process one or more raw source files into wiki pages.

    Usage:
        op = IngestOperation(provider, wiki_fs)
        results = op.run()                        # ingest all new sources
        result  = op.run_one(path_to_source)      # ingest a specific file
    """

    def __init__(self, provider: LLMProvider, wiki_fs: WikiFS, verbose: bool = False):
        self.provider = provider
        self.fs = wiki_fs
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, max_sources: Optional[int] = None) -> list[IngestResult]:
        """Ingest all new (un-processed) raw sources."""
        sources = self.fs.new_raw_sources()
        if not sources:
            return []
        if max_sources:
            sources = sources[:max_sources]
        return [self.run_one(src) for src in sources]

    def run_one(self, source_path: Path) -> IngestResult:
        """Ingest a single source file."""
        source_name = source_path.name
        result = IngestResult(source_name=source_name)

        if self.verbose:
            print(f"  ingesting: {source_name} ...")

        source_text = self.fs.read_source(source_path)
        source_sha = self.fs.source_sha256(source_path)
        index_content = self.fs.read_index()

        system, user = ingest_source(source_text, source_name, index_content)

        try:
            response = self.provider.chat(system, user)
        except Exception as exc:
            result.error = str(exc)
            return result

        result.tokens_used = response.total_tokens
        self._parse_and_write(response.content, source_name, source_sha, result)
        return result

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    def _parse_and_write(
        self,
        llm_output: str,
        source_name: str,
        source_sha: str,
        result: IngestResult,
    ) -> None:
        sections = _split_sections(llm_output)

        # Takeaways (informational only)
        result.takeaways = sections.get("TAKEAWAYS", "").strip()

        # Source summary page
        summary = sections.get("SOURCE_SUMMARY_PAGE", "").strip()
        if summary:
            summary = _inject_sha(summary, source_sha)
            slug = _slugify(source_name)
            path = f"wiki/sources/{slug}.md"
            self.fs.write_wiki_page(path, summary)
            result.pages_created.append(path)

        # Updated existing pages
        for path, content in _parse_multipage_block(sections.get("UPDATED_PAGES", "")):
            self.fs.write_wiki_page(path, content)
            result.pages_updated.append(path)

        # New concept/entity pages
        for path, content in _parse_multipage_block(sections.get("NEW_PAGES", "")):
            self.fs.write_wiki_page(path, content)
            result.pages_created.append(path)

        # Index update
        index_update = sections.get("INDEX_UPDATE", "").strip()
        if index_update:
            self.fs.write_index(index_update)

        # Log entry
        log_entry = sections.get("LOG_ENTRY", "").strip()
        if log_entry:
            self.fs.append_log(log_entry)
        else:
            n_created = len(result.pages_created)
            n_updated = len(result.pages_updated)
            self.fs.append_log(
                f"ingest | {source_name} | "
                f"{n_created} pages created, {n_updated} pages updated"
            )


# ------------------------------------------------------------------
# Parsing utilities
# ------------------------------------------------------------------

def _split_sections(text: str) -> dict[str, str]:
    """
    Split LLM output into named sections.
    Looks for lines like:  ### SECTION_NAME
    """
    sections: dict[str, str] = {}
    current_key: Optional[str] = None
    current_lines: list[str] = []

    for line in text.splitlines():
        header = re.match(r"^###\s+([A-Z_]+)\s*$", line.strip())
        if header:
            if current_key:
                sections[current_key] = "\n".join(current_lines).strip()
            current_key = header.group(1)
            current_lines = []
        else:
            if current_key:
                current_lines.append(line)

    if current_key:
        sections[current_key] = "\n".join(current_lines).strip()

    return sections


def _parse_multipage_block(text: str) -> list[tuple[str, str]]:
    """
    Parse a block that contains multiple pages in the format:
        <path: wiki/...>
        <full markdown>
        <path: wiki/...>
        <full markdown>
    """
    pages = []
    current_path: Optional[str] = None
    current_lines: list[str] = []

    for line in text.splitlines():
        path_match = re.match(r"^<path:\s*(.+?)>\s*$", line.strip())
        if path_match:
            if current_path and current_lines:
                pages.append((current_path, "\n".join(current_lines).strip()))
            current_path = path_match.group(1).strip()
            current_lines = []
        else:
            if current_path:
                current_lines.append(line)

    if current_path and current_lines:
        pages.append((current_path, "\n".join(current_lines).strip()))

    return pages


def _inject_sha(frontmatter_page: str, sha: str) -> str:
    """Add source_sha field to YAML frontmatter if not present."""
    if "source_sha:" not in frontmatter_page and "---" in frontmatter_page:
        frontmatter_page = frontmatter_page.replace(
            "---\n", f"---\nsource_sha: {sha}\n", 1
        )
    return frontmatter_page


def _slugify(name: str) -> str:
    import re
    stem = Path(name).stem
    return re.sub(r"[^a-z0-9-]", "-", stem.lower()).strip("-")
