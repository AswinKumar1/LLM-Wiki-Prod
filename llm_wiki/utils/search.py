"""
BM25 keyword search over wiki pages.

Pure Python stdlib — no numpy, no vector DB, no embeddings.
Uses the BM25 Okapi algorithm (same core as Elasticsearch/Lucene).

BM25 ranks documents by term frequency (how often the query term
appears in the page) penalised by document length, so a short focused
page scores higher than a long page with one passing mention.

Usage:
    index = BM25Index.build(wiki_fs)
    results = index.search("retrieval augmented generation", top_k=5)
    for r in results:
        print(r.score, r.path, r.snippet)

The index is built in-memory and takes ~1ms per wiki page — fast enough
to rebuild on every search for wikis up to ~10k pages.
"""

from __future__ import annotations

import math
import re
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..wiki_fs import WikiFS

# BM25 hyperparameters (standard defaults)
_K1 = 1.5   # term frequency saturation
_B  = 0.75  # length normalisation


# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    path: str           # relative path, e.g. "wiki/concepts/rag.md"
    title: str          # from frontmatter or first heading
    score: float        # BM25 relevance score (higher = more relevant)
    snippet: str        # best matching excerpt (~160 chars)
    matched_terms: list[str] = field(default_factory=list)

    def format(self, index: int) -> str:
        bar = "█" * min(10, max(1, int(self.score * 2)))
        return (
            f"  {index}. [{bar}] {self.title}\n"
            f"     {self.path}\n"
            f"     {self.snippet}\n"
        )


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------

class BM25Index:
    """
    In-memory BM25 index over all wiki pages.

    Build once per search call — cheap enough that persistence isn't needed
    for wikis under ~5k pages.
    """

    def __init__(self) -> None:
        self._docs:   list[_Doc]  = []
        self._avgdl:  float       = 0.0
        self._df:     dict[str, int] = {}   # term → document frequency

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def build(cls, wiki_fs: "WikiFS") -> "BM25Index":
        """Build an index from all pages in the wiki."""
        index = cls()
        pages = wiki_fs.list_wiki_pages()
        for page_path in pages:
            try:
                text = page_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            rel = wiki_fs.relative_to_root(page_path)
            doc = _Doc.from_text(rel, text)
            index._docs.append(doc)

        if not index._docs:
            return index

        # Compute average document length
        index._avgdl = sum(d.length for d in index._docs) / len(index._docs)

        # Compute document frequencies
        for doc in index._docs:
            for term in set(doc.term_freqs.keys()):
                index._df[term] = index._df.get(term, 0) + 1

        return index

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """
        Return up to top_k results ranked by BM25 score.
        Returns empty list if index has no documents.
        """
        if not self._docs:
            return []

        query_terms = _tokenize(query)
        if not query_terms:
            return []

        n = len(self._docs)
        scores: list[tuple[float, _Doc]] = []

        for doc in self._docs:
            score = 0.0
            for term in query_terms:
                if term not in doc.term_freqs:
                    continue
                tf  = doc.term_freqs[term]
                df  = self._df.get(term, 0)
                idf = math.log((n - df + 0.5) / (df + 0.5) + 1)
                tf_norm = (tf * (_K1 + 1)) / (
                    tf + _K1 * (1 - _B + _B * doc.length / max(self._avgdl, 1))
                )
                score += idf * tf_norm
            if score > 0:
                scores.append((score, doc))

        scores.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, doc in scores[:top_k]:
            matched = [t for t in query_terms if t in doc.term_freqs]
            results.append(SearchResult(
                path=doc.path,
                title=doc.title,
                score=round(score, 3),
                snippet=_extract_snippet(doc.raw_text, query_terms),
                matched_terms=matched,
            ))
        return results

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def doc_count(self) -> int:
        return len(self._docs)

    @property
    def vocab_size(self) -> int:
        return len(self._df)


# ---------------------------------------------------------------------------
# Internal document representation
# ---------------------------------------------------------------------------

@dataclass
class _Doc:
    path:       str
    title:      str
    raw_text:   str
    term_freqs: dict[str, int]
    length:     int              # number of tokens

    @classmethod
    def from_text(cls, path: str, text: str) -> "_Doc":
        title   = _extract_title(text, path)
        # Strip YAML frontmatter before indexing
        body    = _strip_frontmatter(text)
        tokens  = _tokenize(body)
        freqs:  dict[str, int] = {}
        for t in tokens:
            freqs[t] = freqs.get(t, 0) + 1
        return cls(
            path=path,
            title=title,
            raw_text=body,
            term_freqs=freqs,
            length=len(tokens),
        )


# ---------------------------------------------------------------------------
# Text processing helpers
# ---------------------------------------------------------------------------

# Common English stop words — skipping these improves precision
_STOP_WORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "this", "that",
    "these", "those", "it", "its", "as", "not", "no", "if", "so", "we",
    "i", "you", "he", "she", "they", "what", "which", "who", "how",
    "when", "where", "why", "all", "each", "more", "also", "than", "then",
    "into", "about", "up", "out", "their", "our", "your", "my", "his", "her",
})

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def _tokenize(text: str) -> list[str]:
    """
    Lowercase, strip punctuation, split on whitespace, remove stop words.
    Returns a list of tokens suitable for BM25 indexing.
    """
    text = text.lower().translate(_PUNCT_TABLE)
    tokens = text.split()
    return [t for t in tokens if t and t not in _STOP_WORDS and len(t) > 1]


def _strip_frontmatter(text: str) -> str:
    """Remove YAML frontmatter block from markdown text."""
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            return text[end + 4:].strip()
    return text


def _extract_title(text: str, path: str) -> str:
    """Extract title from frontmatter, first H1, or filename."""
    # Try frontmatter title:
    fm_match = re.search(r"^title:\s*(.+)$", text, re.MULTILINE)
    if fm_match:
        return fm_match.group(1).strip().strip('"').strip("'")
    # Try first H1:
    h1_match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    if h1_match:
        return h1_match.group(1).strip()
    # Fallback: filename stem
    return Path(path).stem.replace("-", " ").replace("_", " ").title()


def _extract_snippet(text: str, query_terms: list[str], length: int = 200) -> str:
    """
    Find the best excerpt from text that contains query terms.
    Returns a ~length-char window centred on the first match.
    """
    text_lower = text.lower()
    best_pos = -1

    # Find the earliest position where any query term appears
    for term in query_terms:
        pos = text_lower.find(term)
        if pos != -1 and (best_pos == -1 or pos < best_pos):
            best_pos = pos

    if best_pos == -1:
        # No match found — return start of text
        snippet = text[:length].strip()
    else:
        start = max(0, best_pos - length // 4)
        end   = min(len(text), start + length)
        snippet = text[start:end].strip()
        if start > 0:
            snippet = "…" + snippet
        if end < len(text):
            snippet = snippet + "…"

    # Collapse whitespace and newlines for clean display
    return re.sub(r"\s+", " ", snippet).strip()
