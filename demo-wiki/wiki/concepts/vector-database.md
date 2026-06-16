---
title: Vector Database
type: concept
created: 2026-04-15
updated: 2026-04-15
confidence: high
related:
  - "[[rag]]"
  - "[[bm25]]"
  - "[[llm-wiki-pattern]]"
---

# Vector Database

A vector database stores high-dimensional embedding vectors and supports Approximate Nearest Neighbour (ANN) search — the core retrieval primitive in [[rag]] systems.

## How embeddings work

A text embedding model (e.g. `text-embedding-3-small`, `nomic-embed-text`, `all-MiniLM-L6`) converts text into a dense vector of 384–3072 floats. Semantically similar texts produce vectors that are close together in this high-dimensional space.

## Common vector databases

| System | Type | Notes |
|--------|------|-------|
| FAISS | Library | Facebook AI, in-memory, no persistence |
| Chroma | Embedded DB | Easy to run locally, good for prototypes |
| Pinecone | Managed cloud | Fully hosted, expensive at scale |
| Weaviate | Self-hosted | GraphQL API, hybrid BM25+vector |
| pgvector | Postgres extension | Best if you already run Postgres |
| Qdrant | Self-hosted | Fast, written in Rust |

## Why llm-wiki-universal doesn't use one

The [[llm-wiki-pattern]] approach avoids vector databases entirely:

- **No infrastructure to run** — flat markdown files only
- **No embedding cost** — no API calls to generate embeddings
- **No staleness** — wiki pages are the synthesised truth; no index to sync
- **BM25 is sufficient** — for structured wiki pages (not raw chunks), keyword search works well

For cases where semantic search matters (e.g. "find pages similar to this concept"), the optional `wiki search --rerank` flow uses the LLM directly rather than embeddings.

## When you do need a vector database

If you outgrow the wiki pattern — e.g. your raw corpus has millions of documents and you need sub-100ms semantic search at scale — consider a hybrid approach: maintain the wiki for knowledge synthesis, and add pgvector for raw document retrieval.

## See also

- [[rag]] for the system that uses vector databases
- [[bm25]] for the keyword-search alternative used here
- [[llm-wiki-pattern]] for why this project avoids vector databases
