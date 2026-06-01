"""
Query operation.

1. Read wiki/index.md
2. Ask the LLM which pages are relevant to the question
3. Load those pages
4. Ask the LLM to synthesise an answer with [[wikilink]] citations
5. Optionally save the answer as a new wiki page
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional

from ..providers.base import LLMProvider
from ..wiki_fs import WikiFS
from ..prompts import query_find_relevant_pages, query_question


@dataclass
class QueryResult:
    question: str
    answer: str
    pages_read: list[str] = field(default_factory=list)
    tokens_used: int = 0
    error: Optional[str] = None
    saved_to: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None


class QueryOperation:
    """
    Answer a question by reading relevant wiki pages.

    Usage:
        op = QueryOperation(provider, wiki_fs)
        result = op.run("What is the difference between RAG and LLM Wiki?")
    """

    def __init__(self, provider: LLMProvider, wiki_fs: WikiFS, verbose: bool = False):
        self.provider = provider
        self.fs = wiki_fs
        self.verbose = verbose

    def run(self, question: str, save_answer: bool = False) -> QueryResult:
        result = QueryResult(question=question, answer="")

        index_content = self.fs.read_index()
        if not index_content.strip():
            result.error = (
                "Wiki index is empty. Run `wiki ingest` first to add content."
            )
            return result

        # Step 1: find relevant pages
        relevant_paths = self._find_relevant_pages(question, index_content)
        result.pages_read = relevant_paths

        if self.verbose:
            print(f"  reading {len(relevant_paths)} pages: {relevant_paths}")

        # Step 2: load page content
        pages: dict[str, str] = {}
        for path in relevant_paths:
            content = self.fs.read_wiki_page(path)
            if content:
                pages[path] = content

        if not pages:
            # Fallback: just use the index
            pages = {"wiki/index.md": index_content}

        # Step 3: generate answer
        system, user = query_question(question, pages)
        try:
            response = self.provider.chat(system, user)
        except Exception as exc:
            result.error = str(exc)
            return result

        result.answer = response.content
        result.tokens_used = response.total_tokens

        # Step 4: optionally save the answer
        if save_answer:
            saved_path = self._save_answer(question, result.answer)
            result.saved_to = saved_path
            self.fs.append_log(
                f"query | {question[:60]}... | saved to {saved_path}"
            )

        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_relevant_pages(self, question: str, index_content: str) -> list[str]:
        """Ask the LLM to identify which wiki pages are relevant."""
        system, user = query_find_relevant_pages(question, index_content)
        try:
            response = self.provider.chat(system, user, max_tokens=512)
            # Parse JSON array from response
            text = response.content.strip()
            # Extract JSON array even if wrapped in markdown code fences
            match = re.search(r"\[.*?\]", text, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception:
            pass

        # Fallback: return all pages (works fine for small wikis)
        all_pages = self.fs.list_wiki_pages()
        return [str(self.fs.relative_to_root(p)) for p in all_pages[:10]]

    def _save_answer(self, question: str, answer: str) -> str:
        """Save the answer as a wiki page in wiki/queries/."""
        from datetime import date
        import re
        slug = re.sub(r"[^a-z0-9-]", "-", question.lower())[:50].strip("-")
        path = f"wiki/queries/{date.today()}-{slug}.md"
        content = (
            f"---\n"
            f"title: {question}\n"
            f"type: query-answer\n"
            f"created: {date.today()}\n"
            f"---\n\n"
            f"# {question}\n\n"
            f"{answer}\n"
        )
        self.fs.write_wiki_page(path, content)
        return path
