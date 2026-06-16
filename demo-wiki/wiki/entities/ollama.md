---
title: Ollama
type: entity
created: 2026-04-15
updated: 2026-04-15
confidence: high
related:
  - "[[llm-wiki-pattern]]"
  - "[[llm-providers]]"
---

# Ollama

Local LLM inference server. Runs open-weight models on your own hardware — no internet connection, no API key, no cost.

Website: https://ollama.com

## Why Ollama is the default for this project

`llm-wiki-universal` defaults to Ollama with `qwen2.5:3b` because:

1. **Zero cost** — no API subscription needed
2. **Privacy** — data never leaves your machine
3. **Works offline** — no internet required after model download
4. **Low hardware requirements** — `qwen2.5:3b` runs on 4GB RAM

## Setup

```bash
# Install (macOS/Linux/Windows)
# See https://ollama.com for platform-specific instructions

# Pull a model
ollama pull qwen2.5:3b    # ~2GB — fast, good for wiki tasks
ollama pull mistral:7b    # ~4GB — better quality
ollama pull llama3.2:3b   # ~2GB — Meta's model

# Start the server (usually auto-starts)
ollama serve

# Verify
ollama list
```

## Recommended models for wiki workloads

| Model | Size | RAM | Notes |
|-------|------|-----|-------|
| `qwen2.5:3b` | 2GB | 4GB | Default — fast, good instruction following |
| `mistral:7b` | 4GB | 8GB | Better structured output for complex pages |
| `llama3.2:3b` | 2GB | 4GB | Good alternative to qwen |
| `phi3.5` | 2GB | 4GB | Microsoft, strong reasoning |
| `hermes3` | 4GB | 8GB | Good for agent-style tasks |
| `qwen2.5:7b` | 4GB | 8GB | Best local quality for wikis |

## Using with this project

```yaml
# config.yaml
provider: ollama
model: qwen2.5:3b
base_url: http://localhost:11434   # default, can omit
```

```bash
wiki doctor    # checks Ollama is running and model is available
wiki ingest    # uses Ollama automatically
```

## See also

- [[llm-providers]] for comparison with other providers
- [[llm-wiki-pattern]] for how Ollama fits into the ingest pipeline
