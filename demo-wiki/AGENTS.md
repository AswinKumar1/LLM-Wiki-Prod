# LLM Wiki Universal — Demo Wiki Agent Schema
# Provider-agnostic. Works with Ollama, OpenAI, Anthropic, or any compatible LLM.
# This is the demo wiki shipped with the repo. Add your own sources to raw/ and run wiki ingest.

## Purpose

This wiki covers knowledge base patterns, retrieval techniques, and LLM tooling —
the technical domain of the llm-wiki-universal project itself.

## Project Structure

- `raw/`          — Immutable source documents
- `wiki/`         — LLM-generated and maintained markdown pages
- `wiki/index.md` — Master content catalog
- `wiki/log.md`   — Append-only operation log
- `outputs/`      — Lint and NLI reports

## Page Types

```yaml
---
title: Page Title
type: concept | entity | source-summary | comparison | query-answer
sources:
  - raw/articles/filename.md
related:
  - "[[related-concept]]"
created: YYYY-MM-DD
updated: YYYY-MM-DD
confidence: high | medium | low
---
```

## Naming Conventions

- Filenames: kebab-case (e.g. `attention-mechanism.md`)
- Cross-references: `[[wikilinks]]` for all internal links
- Source references: always link back to `raw/` file paths

## Ingest Workflow

1. Read source document
2. Identify key concepts and entities
3. Create `wiki/sources/<source-name>.md` summary
4. Update or create concept/entity pages
5. Update `wiki/index.md`
6. Append to `wiki/log.md`

## Query Workflow

1. Read `wiki/index.md` to find relevant pages
2. Read those pages and synthesise an answer
3. Cite with `[[wikilinks]]`
4. Offer to save valuable answers as new pages

## Lint Workflow

1. Scan for contradictions (NLI)
2. Find orphan pages (no incoming links)
3. Find missing pages (broken wikilinks)
4. Flag low-confidence pages
5. Save report to `outputs/lint-YYYY-MM-DD.md`
