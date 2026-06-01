"""
Lint operation.

Scans all wiki pages and reports:
  - Contradictions between pages
  - Orphan pages (no incoming wikilinks)
  - Missing pages (wikilinks that point nowhere)
  - Low-confidence pages needing review
  - Structural issues (missing frontmatter)

Saves results to outputs/lint-YYYY-MM-DD.md
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

from ..providers.base import LLMProvider
from ..wiki_fs import WikiFS
from ..prompts import lint_scan


@dataclass
class LintResult:
    contradictions: list[str] = field(default_factory=list)
    orphan_pages: list[str] = field(default_factory=list)
    missing_pages: list[str] = field(default_factory=list)
    low_confidence_pages: list[str] = field(default_factory=list)
    structural_issues: list[str] = field(default_factory=list)
    summary: str = ""
    output_path: Optional[str] = None
    tokens_used: int = 0
    error: Optional[str] = None

    @property
    def issue_count(self) -> int:
        return (
            len(self.contradictions)
            + len(self.orphan_pages)
            + len(self.missing_pages)
            + len(self.low_confidence_pages)
            + len(self.structural_issues)
        )

    @property
    def success(self) -> bool:
        return self.error is None


class LintOperation:
    """
    Health-check the wiki.

    Usage:
        op = LintOperation(provider, wiki_fs)
        result = op.run()
    """

    # Max number of pages to send to LLM in one pass (keep context manageable)
    _MAX_PAGES_PER_BATCH = 20

    def __init__(self, provider: LLMProvider, wiki_fs: WikiFS, verbose: bool = False):
        self.provider = provider
        self.fs = wiki_fs
        self.verbose = verbose

    def run(self) -> LintResult:
        result = LintResult()

        all_page_paths = self.fs.list_wiki_pages()
        if not all_page_paths:
            result.error = "No wiki pages found. Run `wiki ingest` first."
            return result

        if self.verbose:
            print(f"  scanning {len(all_page_paths)} pages ...")

        # --- Structural checks (no LLM needed) ---
        result.orphan_pages = self._find_orphans(all_page_paths)
        result.missing_pages = self._find_missing_pages()
        result.structural_issues = self._check_frontmatter(all_page_paths)

        # --- LLM-powered checks ---
        # Batch pages to avoid context overflow
        pages = self._load_pages(all_page_paths[: self._MAX_PAGES_PER_BATCH])
        system, user = lint_scan(pages)

        try:
            response = self.provider.chat(system, user)
        except Exception as exc:
            result.error = str(exc)
            return result

        result.tokens_used = response.total_tokens
        self._parse_lint_response(response.content, result)

        # Save report
        output_path = self._save_report(result)
        result.output_path = output_path

        self.fs.append_log(
            f"lint | {result.issue_count} issues found | report: {output_path}"
        )
        return result

    # ------------------------------------------------------------------
    # Structural checks (deterministic — no LLM)
    # ------------------------------------------------------------------

    def _find_orphans(self, all_pages: list[Path]) -> list[str]:
        """Pages that no other page links to via [[wikilinks]]."""
        all_links = self.fs.all_wikilinks()
        orphans = []
        for page in all_pages:
            stem = page.stem
            rel = self.fs.relative_to_root(page)
            if stem not in all_links and rel not in all_links:
                orphans.append(rel)
        return orphans

    def _find_missing_pages(self) -> list[str]:
        """[[wikilinks]] that reference non-existent pages."""
        all_links = self.fs.all_wikilinks()
        existing_stems = {p.stem for p in self.fs.list_wiki_pages()}
        missing = []
        for link in all_links:
            # Normalize: "Some Concept" → "some-concept"
            slug = re.sub(r"[^a-z0-9-]", "-", link.lower()).strip("-")
            if slug not in existing_stems and link not in existing_stems:
                missing.append(link)
        return sorted(set(missing))

    def _check_frontmatter(self, pages: list[Path]) -> list[str]:
        """Pages missing required YAML frontmatter fields."""
        required = {"title", "type"}
        issues = []
        for page in pages:
            text = page.read_text(encoding="utf-8", errors="replace")
            if not text.startswith("---"):
                issues.append(f"{self.fs.relative_to_root(page)}: missing frontmatter")
                continue
            fm_match = re.match(r"^---\n(.*?)\n---", text, re.DOTALL)
            if not fm_match:
                issues.append(f"{self.fs.relative_to_root(page)}: malformed frontmatter")
                continue
            fm_keys = set(re.findall(r"^(\w+):", fm_match.group(1), re.MULTILINE))
            missing_keys = required - fm_keys
            if missing_keys:
                issues.append(
                    f"{self.fs.relative_to_root(page)}: missing fields: {', '.join(missing_keys)}"
                )
        return issues

    # ------------------------------------------------------------------
    # LLM response parsing
    # ------------------------------------------------------------------

    def _parse_lint_response(self, text: str, result: LintResult) -> None:
        sections = _split_sections(text)
        result.contradictions = _parse_list(sections.get("CONTRADICTIONS", ""))
        # Merge LLM orphan findings with our deterministic ones
        llm_orphans = _parse_list(sections.get("ORPHAN_PAGES", ""))
        result.orphan_pages = sorted(set(result.orphan_pages) | set(llm_orphans))
        llm_missing = _parse_list(sections.get("MISSING_PAGES", ""))
        result.missing_pages = sorted(set(result.missing_pages) | set(llm_missing))
        result.low_confidence_pages = _parse_list(sections.get("LOW_CONFIDENCE_PAGES", ""))
        result.summary = sections.get("SUMMARY", "").strip()

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def _save_report(self, result: LintResult) -> str:
        today = date.today()
        lines = [
            f"# Wiki Lint Report — {today}",
            "",
            f"**Total issues:** {result.issue_count}",
            "",
        ]

        def section(title: str, items: list[str]) -> None:
            lines.append(f"## {title}")
            if items:
                for item in items:
                    lines.append(f"- {item}")
            else:
                lines.append("_None found ✓_")
            lines.append("")

        section("Contradictions", result.contradictions)
        section("Orphan pages", result.orphan_pages)
        section("Missing pages (broken wikilinks)", result.missing_pages)
        section("Low-confidence pages", result.low_confidence_pages)
        section("Structural issues", result.structural_issues)

        if result.summary:
            lines += ["## Summary", "", result.summary, ""]

        content = "\n".join(lines)
        filename = f"lint-{today}.md"
        out_path = self.fs.write_output(filename, content)
        return str(self.fs.relative_to_root(out_path))

    def _load_pages(self, paths: list[Path]) -> dict[str, str]:
        pages = {}
        for p in paths:
            rel = self.fs.relative_to_root(p)
            pages[rel] = p.read_text(encoding="utf-8", errors="replace")
        return pages


# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------

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
        else:
            if current_key:
                current_lines.append(line)
    if current_key:
        sections[current_key] = "\n".join(current_lines).strip()
    return sections


def _parse_list(text: str) -> list[str]:
    """Parse a bullet/numbered list into a Python list."""
    items = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith(("- ", "* ", "+ ")):
            items.append(line[2:].strip())
        elif re.match(r"^\d+\.\s", line):
            items.append(re.sub(r"^\d+\.\s+", "", line))
        elif line and not line.startswith("#"):
            items.append(line)
    return [i for i in items if i]
