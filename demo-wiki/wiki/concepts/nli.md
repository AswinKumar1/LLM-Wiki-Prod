---
title: Natural Language Inference (NLI)
type: concept
created: 2026-04-15
updated: 2026-04-15
confidence: high
related:
  - "[[llm-wiki-pattern]]"
  - "[[rag-vs-llm-wiki]]"
---

# Natural Language Inference (NLI)

NLI is the task of determining the logical relationship between two text fragments (a **premise** and a **hypothesis**):

- **Entailment** — the hypothesis follows from the premise
- **Neutral** — the hypothesis is unrelated or neither follows nor conflicts
- **Contradiction** — the hypothesis conflicts with the premise

In `llm-wiki-universal`, NLI powers `wiki nli` and the contradiction detection pass in `wiki lint`.

## How contradiction detection works here

1. **Extract claims** — factual sentences are extracted from each wiki page (headings, frontmatter, and very short sentences are filtered out)
2. **Find related pairs** — only pages that share `[[wikilinks]]` are checked against each other (semantically related pages are the most likely source of contradictions)
3. **Score pairs** — each (claim_A, claim_B) pair is scored by the NLI backend
4. **Flag contradictions** — pairs scoring above the threshold are reported with exact sentences cited

## Two backends

### LLM backend (default)
Uses your already-configured LLM provider to classify pairs:

```
System: You are a fact-checker. Reply with exactly one word: ENTAILMENT, NEUTRAL, or CONTRADICTION.
User: Statement A: "RAG latency is 1-3 seconds."
      Statement B: "Our system responds in under 100ms."
→ CONTRADICTION
```

Works with Ollama, OpenAI, Anthropic — zero new dependencies.

### Cross-encoder backend (optional, more accurate)
Uses `sentence-transformers` MiniLM NLI cross-encoder:
- ~80MB download, runs locally on CPU
- ~0.84 F1 on NLI benchmarks
- Returns a probability score (0–1) rather than a binary label

```bash
pip install sentence-transformers
wiki nli --backend cross_encoder
```

## Confidence auto-downgrade

When a page has a confirmed contradiction flagged against it, its frontmatter `confidence:` field is automatically downgraded:

```
high → medium → low
```

This surfaces the issue in future `wiki search` results and `wiki query` answers, where low-confidence pages can be treated with appropriate skepticism.

## Limitations

- NLI works best on short, atomic factual claims (one fact per sentence)
- Long, complex sentences may be misclassified
- The LLM backend is sensitive to phrasing — different models may disagree
- Neither backend achieves perfect precision on domain-specific technical claims

## See also

- [[llm-wiki-pattern]] for context on where NLI fits
- [[rag-vs-llm-wiki]] for comparison with RAG's approach to grounding
