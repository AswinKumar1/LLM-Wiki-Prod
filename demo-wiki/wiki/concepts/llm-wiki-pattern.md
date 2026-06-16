---
title: LLM Wiki Pattern
type: concept
created: 2026-04-15
updated: 2026-04-15
confidence: high
sources:
  - raw/articles/karpathy-gist.md
related:
  - "[[rag]]"
  - "[[rag-vs-llm-wiki]]"
  - "[[andrej-karpathy]]"
  - "[[bm25]]"
---

# LLM Wiki Pattern

The LLM Wiki pattern is an approach to knowledge base management where an LLM **incrementally synthesises and maintains** a structured wiki from source documents — shifting heavy computation from query time to ingest time.

Originally described by [[andrej-karpathy]] in April 2026.

## Core idea

Instead of storing raw documents and retrieving chunks at query time ([[rag]]), the LLM reads each new source and:

1. Extracts key concepts, entities, and facts
2. Creates or updates structured wiki pages
3. Maintains cross-references ([[wikilinks]]) between related pages
4. Updates an index for navigation

The result is a **pre-digested, interlinked knowledge base** that answers queries much faster because the LLM only needs to read a few focused wiki pages — not raw chunks of source material.

## Why knowledge compounds

Each new ingest doesn't just add a page — it updates existing pages with new context. A concept page about "transformer attention" gets richer every time a new paper on the topic is ingested. This compounding effect is the key advantage over [[rag]].

## The AGENTS.md schema

The pattern uses a schema file (`AGENTS.md` in this implementation, `CLAUDE.md` in Karpathy's original) that instructs the LLM agent how to maintain the wiki:
- Page types and their frontmatter conventions
- Naming conventions (kebab-case files, `[[wikilinks]]`)
- The three workflow descriptions: Ingest, Query, Lint

## Tradeoffs vs RAG

| | LLM Wiki | RAG |
|--|--|--|
| Query speed | Fast (pre-synthesised) | Slow (runtime retrieval) |
| Ingest cost | Higher (LLM writes pages) | Lower (just embed) |
| Knowledge synthesis | Yes — LLM summarises | No — raw chunks |
| Contradiction detection | Possible via [[nli]] | Not built-in |
| Infrastructure | None (flat files) | Vector database required |

## See also

- [[rag-vs-llm-wiki]] for detailed comparison
- [[andrej-karpathy]] for background on the creator
- [[bm25]] for the search layer used in this implementation
