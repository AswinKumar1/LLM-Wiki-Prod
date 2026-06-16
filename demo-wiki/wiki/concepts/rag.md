---
title: Retrieval Augmented Generation (RAG)
type: concept
created: 2026-04-15
updated: 2026-04-15
confidence: high
sources:
  - raw/articles/rag-overview.md
related:
  - "[[llm-wiki-pattern]]"
  - "[[vector-database]]"
  - "[[rag-vs-llm-wiki]]"
---

# Retrieval Augmented Generation (RAG)

RAG is a technique that combines a retrieval system with a language model to answer questions grounded in a specific document corpus.

## How it works

1. **Index** — Source documents are split into chunks and embedded into a [[vector-database]]
2. **Retrieve** — At query time, the question is embedded and the nearest chunk vectors are found
3. **Generate** — The retrieved chunks are stuffed into the LLM's context window alongside the question

## Strengths

- Works well on fresh, rapidly-changing data (no re-ingestion needed — just re-embed)
- Handles very large corpora (millions of chunks) via ANN search
- Grounding is traceable to specific chunk sources

## Weaknesses

- **High query latency** — embedding the query + vector search + LLM call adds up to 1–3 seconds
- **Token cost scales with context** — each query sends chunks + question to the LLM
- **No knowledge synthesis** — retrieved chunks are raw text; the LLM must reason over noise
- **Chunking artifacts** — splitting documents by token count often cuts mid-sentence or mid-argument
- **Requires infrastructure** — a running vector database (Pinecone, Chroma, pgvector, FAISS)

## Typical latency breakdown

| Step | Typical latency |
|------|----------------|
| Query embedding | 50–200ms |
| Vector ANN search | 10–100ms |
| LLM generation (over chunks) | 800ms–2s |
| **Total** | **~1–3 seconds** |

## When RAG is the right choice

RAG beats [[llm-wiki-pattern]] when:
- Data changes frequently (daily or hourly) and re-ingestion is too slow
- The corpus is enormous (millions of documents)
- You need verbatim passage retrieval rather than synthesised knowledge

## See also

- [[rag-vs-llm-wiki]] for a direct comparison
- [[vector-database]] for the storage layer
- [[llm-wiki-pattern]] for the alternative approach
