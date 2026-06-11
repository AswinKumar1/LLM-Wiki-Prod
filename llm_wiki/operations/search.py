"""
Search operation.

Two modes:
  1. BM25-only (default) — pure keyword search, no LLM call, <10ms
  2. BM25 + LLM rerank  (--rerank) — uses LLM to pick the most relevant
     result and generate a synthesised answer, like a mini RAG

Usage:
    op = SearchOperation(provider, wiki_fs)
    results = op.search("retrieval augmented generation")
    for r in results:
        print(r.format(i))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..providers.base import LLMProvider
from ..wiki_fs import WikiFS
from ..utils.search import BM25Index, SearchResult


@dataclass
class SearchResponse:
    query: str
    results: list[SearchResult] = field(default_factory=list)
    total_docs_searched: int = 0
    answer: Optional[str] = None       # set when rerank=True
    tokens_used: int = 0
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None

    @property
    def found(self) -> bool:
        return len(self.results) > 0


class SearchOperation:
    """
    Search the wiki using BM25 ranking.

    Usage:
        op = SearchOperation(provider, wiki_fs)
        resp = op.search("what is rag?", top_k=5)
        resp = op.search("chunking", top_k=5, rerank=True)  # LLM reranking
    """

    def __init__(self, provider: LLMProvider, wiki_fs: WikiFS, verbose: bool = False):
        self.provider = provider
        self.fs = wiki_fs
        self.verbose = verbose

    def search(
        self,
        query: str,
        top_k: int = 10,
        rerank: bool = False,
    ) -> SearchResponse:
        resp = SearchResponse(query=query)

        if not query.strip():
            resp.error = "Empty query"
            return resp

        # Build BM25 index
        index = BM25Index.build(self.fs)
        resp.total_docs_searched = index.doc_count

        if index.doc_count == 0:
            resp.error = "No wiki pages found. Run `wiki ingest` first."
            return resp

        if self.verbose:
            print(f"  Searching {index.doc_count} pages ({index.vocab_size} unique terms) ...")

        resp.results = index.search(query, top_k=top_k)

        if not resp.results:
            return resp

        # Optional: LLM reranking + answer synthesis
        if rerank and resp.results:
            resp = self._rerank_and_answer(query, resp)

        return resp

    # ------------------------------------------------------------------
    # LLM reranking
    # ------------------------------------------------------------------

    def _rerank_and_answer(self, query: str, resp: SearchResponse) -> SearchResponse:
        """
        Load the top BM25 results and ask the LLM to:
        1. Pick the most relevant pages
        2. Synthesise a short answer
        """
        # Load content of top results (up to 5 for context budget)
        pages: dict[str, str] = {}
        for r in resp.results[:5]:
            content = self.fs.read_wiki_page(r.path)
            if content:
                pages[r.path] = content[:2000]  # cap per page

        if not pages:
            return resp

        pages_text = "\n\n---\n\n".join(
            f"## {path}\n\n{content}" for path, content in pages.items()
        )

        system = (
            "You are a precise search assistant for a markdown wiki. "
            "Answer questions using only the provided pages. "
            "Be concise (3-5 sentences max). Use [[wikilinks]] for citations."
        )
        user = (
            f"Query: {query}\n\n"
            f"Wiki pages:\n\n{pages_text}\n\n"
            "Give a short, precise answer based on these pages."
        )

        try:
            response = self.provider.chat(system, user, max_tokens=512)
            resp.answer = response.content
            resp.tokens_used = response.total_tokens
        except Exception as exc:
            # Reranking failure is non-fatal — BM25 results still returned
            if self.verbose:
                print(f"  Reranking failed (non-fatal): {exc}")

        return resp
