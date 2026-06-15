"""
Lint operation — Day 4 upgrade.

New in Day 4:
  - NLI contradiction detection pass (sentence-level, cited, scored)
  - Confidence auto-downgrade for pages with confirmed contradictions
  - Structural checks unchanged from Day 3

The NLI pass runs after the structural checks. It loads all wiki pages,
extracts factual claims, finds semantically related page pairs via wikilinks,
and scores each claim pair using the configured NLI backend (LLM or
cross-encoder).
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
from ..utils.nli import NLIEngine, NLIResult, ContradictionPair, downgrade_confidence


@dataclass
class LintResult:
    # Structural (Day 1–3)
    orphan_pages: list[str] = field(default_factory=list)
    missing_pages: list[str] = field(default_factory=list)
    structural_issues: list[str] = field(default_factory=list)
    low_confidence_pages: list[str] = field(default_factory=list)
    # LLM summary (Day 1–3)
    summary: str = ""
    # NLI contradictions (Day 4)
    nli_contradictions: list[ContradictionPair] = field(default_factory=list)
    nli_pages_checked: int = 0
    nli_pairs_checked: int = 0
    nli_backend: str = "none"
    pages_downgraded: list[str] = field(default_factory=list)
    # Meta
    output_path: Optional[str] = None
    tokens_used: int = 0
    error: Optional[str] = None

    @property
    def contradictions(self) -> list[str]:
        """Legacy string list — includes both LLM and NLI contradictions."""
        return [
            f"{c.page_a} ↔ {c.page_b} (score {c.score:.2f}): "
            f'"{c.claim_a[:60]}…" vs "{c.claim_b[:60]}…"'
            for c in self.nli_contradictions
        ]

    @property
    def issue_count(self) -> int:
        return (
            len(self.nli_contradictions)
            + len(self.orphan_pages)
            + len(self.missing_pages)
            + len(self.structural_issues)
            + len(self.low_confidence_pages)
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
        result = op.run(skip_nli=True)   # structural checks only
    """

    _MAX_PAGES_PER_LLM_BATCH = 20

    def __init__(
        self,
        provider: LLMProvider,
        wiki_fs: WikiFS,
        verbose: bool = False,
        nli_backend: str = "auto",
    ):
        self.provider = provider
        self.fs = wiki_fs
        self.verbose = verbose
        self.nli_backend = nli_backend

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, skip_nli: bool = False) -> LintResult:
        result = LintResult()

        all_pages = self.fs.list_wiki_pages()
        if not all_pages:
            result.error = "No wiki pages found. Run `wiki ingest` first."
            return result

        if self.verbose:
            print(f"  Scanning {len(all_pages)} pages ...")

        # --- Structural checks (deterministic, no LLM) ---
        result.orphan_pages = self._find_orphans(all_pages)
        result.missing_pages = self._find_missing_pages()
        result.structural_issues = self._check_frontmatter(all_pages)

        # --- LLM summary pass (fast, one prompt) ---
        pages_dict = self._load_pages(all_pages[: self._MAX_PAGES_PER_LLM_BATCH])
        system, user = lint_scan(pages_dict)
        try:
            response = self.provider.chat(system, user)
            result.tokens_used = response.total_tokens
            self._parse_llm_response(response.content, result)
        except Exception as exc:
            if self.verbose:
                print(f"  LLM lint pass failed (non-fatal): {exc}")

        # --- NLI contradiction detection (Day 4) ---
        if not skip_nli:
            nli_result = self._run_nli(pages_dict)
            result.nli_contradictions = nli_result.contradictions
            result.nli_pages_checked = nli_result.pages_checked
            result.nli_pairs_checked = nli_result.pairs_checked
            result.nli_backend = nli_result.backend_used
            result.tokens_used += nli_result.tokens_used

            # Auto-downgrade confidence of affected pages
            affected = nli_result.pages_with_contradictions()
            for rel_path in affected:
                self._downgrade_page_confidence(rel_path, result)

        # --- Save report ---
        output_path = self._save_report(result)
        result.output_path = output_path

        self.fs.append_log(
            f"lint | {result.issue_count} issues "
            f"({len(result.nli_contradictions)} NLI contradictions) | "
            f"report: {output_path}"
        )
        return result

    # ------------------------------------------------------------------
    # NLI pass
    # ------------------------------------------------------------------

    def _run_nli(self, pages: dict[str, str]) -> NLIResult:
        try:
            engine = NLIEngine(
                self.provider,
                backend=self.nli_backend,
                verbose=self.verbose,
            )
            if self.verbose:
                print(f"  NLI backend: {engine.backend_name}")
            return engine.scan_pages(pages)
        except Exception as exc:
            if self.verbose:
                print(f"  NLI scan failed (non-fatal): {exc}")
            nli = NLIResult()
            nli.error = str(exc)
            return nli

    # ------------------------------------------------------------------
    # Confidence downgrader
    # ------------------------------------------------------------------

    def _downgrade_page_confidence(self, rel_path: str, result: LintResult) -> None:
        content = self.fs.read_wiki_page(rel_path)
        if not content:
            return
        new_content, changed = downgrade_confidence(content)
        if changed:
            self.fs.write_wiki_page(rel_path, new_content)
            result.pages_downgraded.append(rel_path)
            if self.verbose:
                print(f"  ↓ confidence downgraded: {rel_path}")

    # ------------------------------------------------------------------
    # Structural checks (unchanged from Day 3)
    # ------------------------------------------------------------------

    def _find_orphans(self, all_pages: list[Path]) -> list[str]:
        all_links = self.fs.all_wikilinks()
        orphans = []
        for page in all_pages:
            stem = page.stem
            rel = self.fs.relative_to_root(page)
            if stem not in all_links and rel not in all_links:
                orphans.append(rel)
        return orphans

    def _find_missing_pages(self) -> list[str]:
        all_links = self.fs.all_wikilinks()
        existing_stems = {p.stem for p in self.fs.list_wiki_pages()}
        missing = []
        for link in all_links:
            slug = re.sub(r"[^a-z0-9-]", "-", link.lower()).strip("-")
            if slug not in existing_stems and link not in existing_stems:
                missing.append(link)
        return sorted(set(missing))

    def _check_frontmatter(self, pages: list[Path]) -> list[str]:
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

    def _parse_llm_response(self, text: str, result: LintResult) -> None:
        sections = _split_sections(text)
        result.low_confidence_pages = _parse_list(sections.get("LOW_CONFIDENCE_PAGES", ""))
        # Merge LLM orphan/missing findings with deterministic ones
        llm_orphans = _parse_list(sections.get("ORPHAN_PAGES", ""))
        result.orphan_pages = sorted(set(result.orphan_pages) | set(llm_orphans))
        llm_missing = _parse_list(sections.get("MISSING_PAGES", ""))
        result.missing_pages = sorted(set(result.missing_pages) | set(llm_missing))
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
            f"**NLI backend:** {result.nli_backend}",
            f"**Pages NLI-checked:** {result.nli_pages_checked}  "
            f"| **Claim pairs checked:** {result.nli_pairs_checked}",
            "",
        ]

        def section(title: str, items: list) -> None:
            lines.append(f"## {title}")
            if items:
                for item in items:
                    if hasattr(item, "format"):
                        lines.append(item.format())
                    else:
                        lines.append(f"- {item}")
            else:
                lines.append("_None found ✓_")
            lines.append("")

        # NLI contradictions get their own detailed section
        lines.append("## NLI Contradictions (sentence-level)")
        if result.nli_contradictions:
            for c in result.nli_contradictions:
                lines.append(c.format())
        else:
            lines.append("_None found ✓_")
        lines.append("")

        if result.pages_downgraded:
            lines.append("## Confidence Downgraded")
            for p in result.pages_downgraded:
                lines.append(f"- {p}")
            lines.append("")

        section("Orphan Pages", result.orphan_pages)
        section("Missing Pages (broken wikilinks)", result.missing_pages)
        section("Low-Confidence Pages", result.low_confidence_pages)
        section("Structural Issues", result.structural_issues)

        if result.summary:
            lines += ["## LLM Summary", "", result.summary, ""]

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


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


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


def _parse_list(text: str) -> list[str]:
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
