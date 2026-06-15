"""
NLI (Natural Language Inference) engine for contradiction detection.

Scores sentence pairs as:
  ENTAILMENT   — claim B is supported by claim A
  NEUTRAL      — claims are unrelated
  CONTRADICTION — claims conflict with each other

Two backends, auto-selected:

  Backend A — LLM scorer (default, zero new deps)
    Uses your already-configured LLM provider.
    Asks it to classify pairs as entail/neutral/contradict.
    Works with Ollama, OpenAI, Anthropic, anything.

  Backend B — Cross-encoder (optional, more accurate)
    Uses sentence-transformers MiniLM NLI cross-encoder.
    ~80MB download, runs locally, ~0.84 F1 on NLI benchmarks.
    Install: pip install sentence-transformers

The engine automatically picks Backend B if sentence-transformers is
installed, falling back to Backend A otherwise. You can force a backend
via the `backend` parameter.

Usage:
    engine = NLIEngine(provider)
    results = engine.scan_pages(pages_dict)
    for c in results.contradictions:
        print(c.format())
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional, Literal

from ..providers.base import LLMProvider

# ---------------------------------------------------------------------------
# Threshold — pairs scoring below this are flagged as contradictions
# ---------------------------------------------------------------------------
_CONTRADICTION_THRESHOLD = 0.65   # for cross-encoder (logit → sigmoid)
_LLM_LABELS = {"contradiction", "contradict", "conflicts", "conflict"}

Backend = Literal["auto", "llm", "cross_encoder"]


# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------

@dataclass
class ContradictionPair:
    page_a:    str    # relative path
    page_b:    str    # relative path
    claim_a:   str    # sentence from page A
    claim_b:   str    # sentence from page B
    score:     float  # contradiction confidence 0–1
    backend:   str    # "llm" or "cross_encoder"

    def format(self) -> str:
        bar = "█" * min(10, max(1, int(self.score * 10)))
        return (
            f"  [{bar}] {self.score:.2f}\n"
            f"  Page A: {self.page_a}\n"
            f"    \"{self.claim_a}\"\n"
            f"  Page B: {self.page_b}\n"
            f"    \"{self.claim_b}\"\n"
        )


@dataclass
class NLIResult:
    contradictions: list[ContradictionPair] = field(default_factory=list)
    pairs_checked:  int = 0
    pages_checked:  int = 0
    backend_used:   str = "none"
    tokens_used:    int = 0
    error:          Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None

    @property
    def contradiction_count(self) -> int:
        return len(self.contradictions)

    def pages_with_contradictions(self) -> set[str]:
        affected: set[str] = set()
        for c in self.contradictions:
            affected.add(c.page_a)
            affected.add(c.page_b)
        return affected


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

class NLIEngine:
    """
    Scan wiki pages for semantic contradictions.

    Usage:
        engine = NLIEngine(provider, backend="auto")
        pages  = {"wiki/concepts/rag.md": "...", "wiki/concepts/llm.md": "..."}
        result = engine.scan_pages(pages, max_pairs=200)
    """

    def __init__(
        self,
        provider:  LLMProvider,
        backend:   Backend = "auto",
        verbose:   bool    = False,
    ):
        self.provider = provider
        self.verbose  = verbose
        self._backend = self._resolve_backend(backend)

    @property
    def backend_name(self) -> str:
        return self._backend

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scan_pages(
        self,
        pages:      dict[str, str],
        max_pairs:  int = 300,
        min_claims: int = 2,
    ) -> NLIResult:
        """
        Scan a dict of {path: content} pages for contradictions.

        Strategy:
          1. Extract factual claims (sentences) from each page
          2. Build candidate pairs — only pages that share wikilinks
             (i.e. are semantically related) to keep pairs manageable
          3. Score each pair with the chosen NLI backend
          4. Return pairs scoring above the contradiction threshold
        """
        result = NLIResult(backend_used=self._backend)

        if len(pages) < 2:
            result.error = "Need at least 2 pages to check for contradictions"
            return result

        # Step 1: extract claims per page
        claims: dict[str, list[str]] = {}
        for path, content in pages.items():
            extracted = _extract_claims(content)
            if len(extracted) >= min_claims:
                claims[path] = extracted

        result.pages_checked = len(claims)

        if result.pages_checked < 2:
            result.error = "Not enough pages with extractable claims"
            return result

        # Step 2: build candidate page pairs (related pages only)
        page_paths   = list(claims.keys())
        related_pairs = _find_related_pairs(page_paths, pages)

        if not related_pairs:
            # Fallback: check all pairs if no wikilinks found
            related_pairs = [
                (page_paths[i], page_paths[j])
                for i in range(len(page_paths))
                for j in range(i + 1, len(page_paths))
            ]

        if self.verbose:
            print(f"  NLI: checking {len(related_pairs)} page pairs "
                  f"({self._backend} backend) ...")

        # Step 3: score pairs — cap at max_pairs
        checked = 0
        for path_a, path_b in related_pairs[:max_pairs]:
            if path_a not in claims or path_b not in claims:
                continue
            pairs = _cross_claims(claims[path_a], claims[path_b])
            for claim_a, claim_b in pairs:
                checked += 1
                scored = self._score_pair(claim_a, claim_b, path_a, path_b)
                if scored is not None:
                    result.contradictions.append(scored)
                    if self.verbose:
                        print(f"    ⚠ contradiction ({scored.score:.2f}): "
                              f"{path_a} ↔ {path_b}")

        result.pairs_checked = checked
        return result

    def score_single(self, claim_a: str, claim_b: str) -> float:
        """
        Score a single pair. Returns contradiction probability 0–1.
        Useful for testing or one-off checks.
        """
        if self._backend == "cross_encoder":
            return self._score_cross_encoder(claim_a, claim_b)
        return self._score_llm_single(claim_a, claim_b)

    # ------------------------------------------------------------------
    # Backend resolution
    # ------------------------------------------------------------------

    def _resolve_backend(self, requested: Backend) -> str:
        if requested == "cross_encoder":
            if not _cross_encoder_available():
                raise ImportError(
                    "sentence-transformers is not installed.\n"
                    "Install it with: pip install sentence-transformers"
                )
            return "cross_encoder"
        if requested == "llm":
            return "llm"
        # auto: prefer cross_encoder if available
        if _cross_encoder_available():
            return "cross_encoder"
        return "llm"

    # ------------------------------------------------------------------
    # Scoring — cross-encoder backend
    # ------------------------------------------------------------------

    _ce_model = None   # lazy-loaded class attribute

    def _score_cross_encoder(self, claim_a: str, claim_b: str) -> float:
        """Score using sentence-transformers cross-encoder NLI model."""
        if NLIEngine._ce_model is None:
            from sentence_transformers import CrossEncoder  # type: ignore
            NLIEngine._ce_model = CrossEncoder(
                "cross-encoder/nli-MiniLM2-L6-H4",
                max_length=256,
            )
        scores = NLIEngine._ce_model.predict(
            [(claim_a, claim_b)],
            apply_softmax=True,
        )[0]
        # Model output order: contradiction, entailment, neutral
        # (verify with model card — MiniLM2 uses this order)
        contradiction_score = float(scores[0])
        return contradiction_score

    # ------------------------------------------------------------------
    # Scoring — LLM backend
    # ------------------------------------------------------------------

    def _score_llm_single(self, claim_a: str, claim_b: str) -> float:
        """
        Ask the LLM to classify a pair.
        Returns 1.0 for CONTRADICTION, 0.0 otherwise.
        """
        system = (
            "You are a precise fact-checker. Given two statements, classify their relationship.\n"
            "Reply with exactly one word: ENTAILMENT, NEUTRAL, or CONTRADICTION.\n"
            "CONTRADICTION means the statements make conflicting factual claims.\n"
            "Do not explain. Reply with the single word only."
        )
        user = f'Statement A: "{claim_a}"\nStatement B: "{claim_b}"'
        try:
            response = self.provider.chat(system, user, max_tokens=10, temperature=0.0)
            label = response.content.strip().upper()
            if any(word in label for word in ("CONTRADICTION", "CONTRADICT", "CONFLICT")):
                return 1.0
            return 0.0
        except Exception:
            return 0.0

    def _score_pair(
        self,
        claim_a: str,
        claim_b: str,
        path_a:  str,
        path_b:  str,
    ) -> Optional[ContradictionPair]:
        """Score a pair and return a ContradictionPair if it's a contradiction."""
        if self._backend == "cross_encoder":
            score = self._score_cross_encoder(claim_a, claim_b)
            is_contradiction = score >= _CONTRADICTION_THRESHOLD
        else:
            score = self._score_llm_single(claim_a, claim_b)
            is_contradiction = score >= 1.0

        if not is_contradiction:
            return None

        return ContradictionPair(
            page_a=path_a,
            page_b=path_b,
            claim_a=claim_a,
            claim_b=claim_b,
            score=round(score, 3),
            backend=self._backend,
        )


# ---------------------------------------------------------------------------
# Confidence downgrader
# ---------------------------------------------------------------------------

def downgrade_confidence(content: str) -> tuple[str, bool]:
    """
    Downgrade the confidence field in a wiki page's frontmatter.
    high → medium → low
    Returns (updated_content, was_changed).
    """
    _LADDER = {"high": "medium", "medium": "low", "low": "low"}

    def replacer(match: re.Match) -> str:
        current = match.group(1).strip().lower()
        new     = _LADDER.get(current, current)
        return match.group(0).replace(match.group(1), new)

    pattern = r"^(confidence:\s*)(\w+)$"

    new_content, n = re.subn(
        r"^confidence:\s*(\w+)$",
        lambda m: f"confidence: {_LADDER.get(m.group(1).lower(), m.group(1))}",
        content,
        flags=re.MULTILINE,
    )
    changed = n > 0 and new_content != content
    return new_content, changed


# ---------------------------------------------------------------------------
# Text processing helpers
# ---------------------------------------------------------------------------

def _extract_claims(page_content: str) -> list[str]:
    """
    Extract factual sentences from a wiki page.

    Filters:
      - strips frontmatter
      - skips headings, wikilinks-only lines, very short sentences
      - keeps sentences with concrete factual content
    """
    # Strip YAML frontmatter
    text = page_content
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            text = text[end + 4:]

    # Remove markdown headings, wikilinks, URLs, code blocks
    text = re.sub(r"^#{1,6}\s+.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    text = re.sub(r"`[^`]+`", "", text)
    text = re.sub(r"\[\[.*?\]\]", "", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)   # images
    text = re.sub(r"\[.*?\]\(.*?\)", "", text)    # links

    # Split into sentences
    sentences = re.split(r"(?<=[.!?])\s+", text)

    claims = []
    for sent in sentences:
        sent = sent.strip()
        # Filter: min 20 chars, max 300, no pure list markers
        if len(sent) < 20 or len(sent) > 300:
            continue
        if sent.startswith(("-", "*", "•", ">")):
            sent = sent.lstrip("-*•> ").strip()
        if len(sent) < 20:
            continue
        # Must contain at least one verb-like word (basic factual check)
        if not re.search(r"\b(is|are|was|were|has|have|can|does|do|will|"
                         r"uses|use|provides|provide|requires|require|"
                         r"reduces|increases|supports|enables|contains|"
                         r"returns|stores|produces|generates)\b",
                         sent, re.IGNORECASE):
            continue
        claims.append(sent)

    return claims[:50]   # cap per page to keep pairs manageable


def _find_related_pairs(
    paths: list[str],
    pages: dict[str, str],
) -> list[tuple[str, str]]:
    """
    Find pairs of pages that are semantically related via shared wikilinks.
    Pages that share a [[wikilink]] target are likely to discuss the same topic
    and are the most likely candidates for contradictions.
    """
    # Build wikilink sets per page
    wikilinks: dict[str, set[str]] = {}
    pattern = re.compile(r"\[\[([^\]]+)\]\]")
    for path in paths:
        content = pages.get(path, "")
        wikilinks[path] = set(pattern.findall(content))

    related: list[tuple[str, str]] = []
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            a, b = paths[i], paths[j]
            # Related if they share any wikilinks, or one links to the other
            shared = wikilinks.get(a, set()) & wikilinks.get(b, set())
            a_links_b = any(b.endswith(link + ".md") for link in wikilinks.get(a, set()))
            b_links_a = any(a.endswith(link + ".md") for link in wikilinks.get(b, set()))
            if shared or a_links_b or b_links_a:
                related.append((a, b))

    return related


def _cross_claims(
    claims_a: list[str],
    claims_b: list[str],
    max_pairs: int = 20,
) -> list[tuple[str, str]]:
    """
    Generate claim pairs to check between two pages.
    Caps at max_pairs to keep LLM call count manageable.
    Prioritises shorter sentences (more likely to be atomic facts).
    """
    sorted_a = sorted(claims_a, key=len)[:10]
    sorted_b = sorted(claims_b, key=len)[:10]
    pairs = [(a, b) for a in sorted_a for b in sorted_b]
    return pairs[:max_pairs]


def _cross_encoder_available() -> bool:
    try:
        import sentence_transformers  # type: ignore
        return True
    except ImportError:
        return False
