---
title: RAG vs LLM Wiki Pattern
type: comparison
created: 2026-04-15
updated: 2026-04-15
confidence: high
related:
  - "[[rag]]"
  - "[[llm-wiki-pattern]]"
  - "[[bm25]]"
  - "[[vector-database]]"
---

# RAG vs LLM Wiki Pattern

A direct comparison of the two dominant approaches to LLM-powered knowledge retrieval.

## Architecture comparison

| Dimension | RAG | LLM Wiki |
|-----------|-----|----------|
| Storage | Raw chunks + embedding vectors | Synthesised markdown pages |
| Query path | Embed → ANN search → LLM over chunks | BM25 search → LLM over wiki pages |
| Infrastructure | [[vector-database]] required | None (flat files) |
| Query latency | 1–3 seconds | 0.5–1.5 seconds (smaller context) |
| Knowledge synthesis | No — LLM sees raw chunks | Yes — LLM sees pre-digested pages |
| Contradiction handling | None | [[nli]] scan detects conflicts |
| Cost per query | Higher (more tokens per query) | Lower (focused context) |
| Ingest cost | Lower (just embed) | Higher (LLM writes pages) |
| Handles updates | Easy (re-embed the chunk) | Requires re-ingest |
| Good for large corpora | Yes (millions of docs) | Best under ~10k sources |

## When to choose RAG

- Data changes continuously (logs, news, live databases)
- Corpus is very large (millions of documents)
- You need verbatim passage retrieval
- You already have vector infrastructure

## When to choose LLM Wiki

- Building a personal or team knowledge base from a curated set of sources
- You want answers that synthesise across multiple documents
- You want zero infrastructure (no database to run)
- You want knowledge to compound over time
- You're resource-constrained (local Ollama, no cloud budget)

## Hybrid approach

The two patterns are not mutually exclusive. A production system might:

1. Use the LLM Wiki for synthesised, high-value knowledge (architecture decisions, research summaries)
2. Use RAG over raw documents for verbatim retrieval and freshness

`llm-wiki-universal` is designed to be the wiki layer — it can coexist with a RAG system.

## See also

- [[rag]] for detailed RAG architecture
- [[llm-wiki-pattern]] for detailed LLM Wiki architecture
- [[bm25]] for the search layer in this project
- [[vector-database]] for RAG's storage layer
