---
title: LLM Providers for Wiki Workloads
type: comparison
created: 2026-04-15
updated: 2026-04-15
confidence: high
related:
  - "[[llm-wiki-pattern]]"
  - "[[ollama]]"
---

# LLM Providers for Wiki Workloads

A guide to choosing between the providers supported by `llm-wiki-universal` for ingest, query, and NLI tasks.

## Provider comparison

| Provider | Cost | Speed | Quality | Best for |
|----------|------|-------|---------|----------|
| [[ollama]] (qwen2.5:3b) | Free | Fast on CPU | Good | Personal wikis, offline use |
| Ollama (mistral:7b) | Free | Medium | Better | Richer concept pages |
| OpenAI gpt-4o-mini | ~$0.001/ingest | Fast | Excellent | Team wikis, production |
| OpenAI gpt-4o | ~$0.01/ingest | Medium | Best | High-stakes knowledge bases |
| Anthropic Haiku | ~$0.001/ingest | Fast | Excellent | Good default for paid tier |
| Anthropic Sonnet | ~$0.005/ingest | Medium | Superior | Best instruction following |
| Groq (via openai_compat) | Free tier | Very fast | Good | Fast experimentation |
| LM Studio (local) | Free | Hardware-dependent | Model-dependent | Privacy-sensitive corpora |

## Recommended defaults

**Zero budget:** Ollama with `qwen2.5:3b` — 2GB RAM, fast on CPU, good enough for most wikis.

**Small budget:** OpenAI `gpt-4o-mini` — best quality/cost ratio for ingest. Use Haiku for NLI (cheaper per pair).

**Best quality:** Anthropic `claude-sonnet-4-6` — best instruction following for structured wiki page generation.

**Privacy-sensitive:** LM Studio or Ollama — data never leaves your machine.

## Per-operation recommendations

| Operation | Recommended model | Why |
|-----------|------------------|-----|
| `wiki ingest` | gpt-4o-mini or Sonnet | Best structured output |
| `wiki query` | qwen2.5:3b or Haiku | Fast, low cost for Q&A |
| `wiki nli` | qwen2.5:3b (LLM backend) | Cheap per pair, classification is simple |
| `wiki lint` | any | Mostly structural, LLM for summary only |

## Setting per-operation providers

Currently `llm-wiki-universal` uses one provider for all operations (set in `config.yaml`). Per-operation provider overrides are planned for Day 6+.

## See also

- [[ollama]] for local inference setup
- [[llm-wiki-pattern]] for how providers fit into the ingest pipeline
