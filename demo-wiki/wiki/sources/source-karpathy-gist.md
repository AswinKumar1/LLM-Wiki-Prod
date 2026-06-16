---
title: Karpathy LLM Wiki Gist (April 2026)
type: source-summary
source: raw/articles/karpathy-gist.md
created: 2026-04-15
updated: 2026-04-15
confidence: high
related:
  - "[[llm-wiki-pattern]]"
  - "[[andrej-karpathy]]"
---

# Karpathy LLM Wiki Gist (April 2026)

**Source:** GitHub Gist by Andrej Karpathy, April 2026
**URL:** https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f

## Key takeaways

- The LLM Wiki pattern describes using an LLM agent to incrementally build and maintain a structured knowledge base from source documents
- The Gist is intentionally abstract — it defines the concept and conventions, not a specific implementation
- The approach is explicitly designed to be agent-agnostic: "designed to be copy pasted to your own LLM Agent (e.g. OpenAI Codex, Claude Code, OpenCode / Pi, or etc.)"
- Three core operations: Ingest (add new source), Query (answer questions), Lint (health-check)
- The wiki uses a schema file (`CLAUDE.md` in the original, `AGENTS.md` in this implementation) to instruct the agent on conventions
- Pages use YAML frontmatter with fields: title, type, sources, related, created, updated, confidence

## What this project adds

This implementation (`llm-wiki-universal`) extends the pattern with:

- Provider-agnostic adapter layer (Ollama, OpenAI, Anthropic, any OpenAI-compatible endpoint)
- BM25 keyword search over wiki pages (zero deps, instant)
- PDF and URL ingestion
- NLI contradiction detection (sentence-level, with confidence auto-downgrade)
- Token usage tracking and cost estimation
- Pre-flight health checks (`wiki doctor`)
- Auto-chunking for large source files
- Retry with exponential backoff

## See also

- [[llm-wiki-pattern]] for the pattern summary
- [[andrej-karpathy]] for background on the author
