# LLM Wiki Universal

> **Andrej Karpathy's LLM Wiki pattern — provider-agnostic, zero infrastructure, free for everyone.**

[![CI](https://github.com/AswinKumar1/LLM-Wiki-Prod/actions/workflows/ci.yml/badge.svg)](https://github.com/AswinKumar1/LLM-Wiki-Prod/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Works with Ollama](https://img.shields.io/badge/works%20with-Ollama-orange)](https://ollama.com)

Build and maintain a personal or team knowledge base where the LLM does all the bookkeeping — without locking you into any single AI provider or requiring a paid subscription.

---

## What is this?

In April 2026, [Andrej Karpathy posted a pattern](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f) for using LLMs to build self-maintaining knowledge bases. Instead of querying raw documents every time (RAG), the LLM **incrementally builds and maintains a structured wiki** from your sources. Every ingest updates the wiki. Knowledge compounds. Cross-references are always there.

**The gap:** Every community implementation locked users to Claude Code or OpenAI. This repo fixes that.

**This project:** A clean Python implementation of the pattern that works with any LLM — local or cloud, free or paid — through a single provider interface.

```
┌─────────────────────────────────────────────────────┐
│              your sources (raw/)                    │
└──────────────────────┬──────────────────────────────┘
                       │  ingest
                       ▼
┌─────────────────────────────────────────────────────┐
│           wiki/ (structured markdown)               │
│   index.md · concepts/ · entities/ · sources/      │
└──────────────────────┬──────────────────────────────┘
                       │  query / lint
                       ▼
┌─────────────────────────────────────────────────────┐
│          any LLM (Ollama · OpenAI · Anthropic)      │
│          or any OpenAI-compatible endpoint           │
└─────────────────────────────────────────────────────┘
```

---

## Quick Start

### Option A — Free, fully local (Ollama)

```bash
# 1. Install Ollama (https://ollama.com)
ollama pull qwen2.5:3b          # ~2GB, fast on CPU

# 2. Install llm-wiki-universal
pip install -e .

# 3. Initialise a wiki
mkdir my-wiki && cd my-wiki
wiki init                       # creates raw/, wiki/, AGENTS.md, config.yaml

# 4. Add a source
cp ~/Downloads/some-article.md raw/articles/

# 5. Ingest
wiki ingest

# 6. Ask questions
wiki query "What are the main ideas in this article?"
```

### Option B — OpenAI

```bash
export OPENAI_API_KEY=sk-...
wiki init --provider openai --model gpt-4o-mini
wiki ingest
wiki query "Summarise what I've learned so far"
```

### Option C — Anthropic

```bash
export ANTHROPIC_API_KEY=sk-ant-...
wiki init --provider anthropic --model claude-haiku-4-5-20251001
wiki ingest
```

### Option D — LM Studio / vLLM / Groq / any OpenAI-compatible server

```yaml
# config.yaml
provider: openai_compat
base_url: http://localhost:1234/v1   # LM Studio
model: hermes-3-llama-3.1-8b
api_key: lm-studio                   # any non-empty string for LM Studio
```

```bash
wiki ingest
```

---

## Installation

```bash
git clone https://github.com/AswinKumar1/LLM-Wiki-Prod.git
cd LLM-Wiki-Prod
pip install -e .            # installs the `wiki` CLI command
```

**No required dependencies beyond the Python standard library.**  
Optional: `pip install pyyaml` for cleaner config parsing.

---

## Commands

| Command | Description |
|---------|-------------|
| `wiki init` | Initialise a new wiki in the current directory |
| `wiki ingest` | Process all new files in `raw/` into wiki pages |
| `wiki ingest --source path/to/file.md` | Ingest a specific file |
| `wiki query "your question"` | Answer a question from the wiki |
| `wiki query "..." --save` | Answer and save as a new wiki page |
| `wiki lint` | Health-check: contradictions, orphans, missing pages |
| `wiki status` | Show wiki stats and provider health |
| `wiki providers` | List all supported providers |

**Global flags:**
```
--root PATH      Wiki root directory (default: current dir)
--provider NAME  Override provider from config
--model NAME     Override model from config
--verbose / -v   Verbose output
```

---

## Directory Structure

```
my-wiki/
├── raw/                    # Your source documents (immutable)
│   ├── articles/
│   ├── papers/
│   ├── repos/
│   └── data/
├── wiki/                   # LLM-generated knowledge base
│   ├── index.md            # Master catalog — updated every ingest
│   ├── log.md              # Append-only operations log
│   ├── concepts/           # Concept pages
│   ├── entities/           # Entity pages (people, orgs, tools)
│   ├── sources/            # One summary per ingested source
│   └── comparisons/        # Side-by-side analysis pages
├── outputs/                # Lint reports, generated presentations
├── AGENTS.md               # Schema: tells the LLM how to maintain the wiki
└── config.yaml             # Provider and model config
```

---

## Supported Providers

| Provider | Key Required | Default Model | Notes |
|----------|-------------|---------------|-------|
| `ollama` | No | `qwen2.5:3b` | Free, fully local. Install Ollama first. |
| `openai` | `OPENAI_API_KEY` | `gpt-4o-mini` | GPT-4o, o1-mini, etc. |
| `anthropic` | `ANTHROPIC_API_KEY` | `claude-haiku-4-5-20251001` | All Claude models |
| `openai_compat` | Optional | Server default | LM Studio, vLLM, Groq, Together, Fireworks, Hermes |

**Adding a custom provider:** subclass `LLMProvider`, implement `chat()` and `health_check()`, register in `providers/factory.py`. That's it.

---

## How it works

Karpathy's pattern has three operations:

### Ingest
Drop a file into `raw/` and run `wiki ingest`. The LLM:
1. Reads the source
2. Creates a summary in `wiki/sources/`
3. Updates or creates concept and entity pages
4. Updates `wiki/index.md`
5. Appends to `wiki/log.md`

A single ingest typically touches 5–15 wiki pages. Every source file is SHA-256 fingerprinted so re-ingesting is safe.

### Query
`wiki query "your question"` navigates the index, reads relevant pages, and synthesises an answer with `[[wikilink]]` citations. Good answers can be saved back into the wiki with `--save`, so your explorations compound.

### Lint
`wiki lint` runs a health check:
- Contradictions between pages
- Orphan pages with no incoming links
- Broken `[[wikilinks]]` pointing nowhere
- Low-confidence pages needing review
- Missing frontmatter

Results are saved to `outputs/lint-YYYY-MM-DD.md`.

---

## vs RAG vs Karpathy's original

| | Standard RAG | Karpathy's Base Wiki | **This project** |
|--|--|--|--|
| Provider | Any | Claude Code only | **Any (Ollama, OpenAI, Anthropic, custom)** |
| Infrastructure | Vector DB + embeddings | None | **None** |
| Query latency | 1–3s | LLM-dependent | LLM-dependent |
| Knowledge compounds | No | Yes | **Yes** |
| Citation tracing | Chunk-level | File-level | **File-level + SHA-256** |
| Contradiction detection | No | Manual lint | **Automated lint** |
| Free tier | Vendor-dependent | No | **Yes (Ollama)** |
| Installable as package | No | No | **`pip install -e .`** |

---

## GitHub Codespaces

This repo includes a `.devcontainer` config. Click **Code → Codespaces → Create codespace** and you'll have a working environment in 60 seconds with the `wiki` CLI ready.

```bash
# In Codespace — set your provider via the Codespace secrets UI, then:
wiki init
wiki ingest
wiki query "your question"
```

---

## Roadmap

This is an iterative project — contributions committed daily.

**Day 1 (this commit):** Provider adapter layer, core wiki engine, CLI skeleton, tests  
**Day 2:** Streaming output, better ingest parsing, token usage tracking  
**Day 3:** Semantic search over wiki pages (optional, no external deps)  
**Day 4:** Web clipper integration, PDF ingestion  
**Day 5+:** NLI-based contradiction detection, multi-wiki support, graph visualisation

---

## Contributing

PRs welcome. The codebase is intentionally simple — no frameworks, no heavy deps.

1. Fork and clone
2. `pip install -e ".[dev]"`
3. `pytest tests/`
4. Open a PR

The `tests/` directory uses a `MockProvider` that requires no API key — all tests run offline.

---

## Philosophy

> "The wiki stays maintained because the cost of maintenance is near zero."  
> — Andrej Karpathy

The goal of this project is to make that true for everyone — not just people with Claude Pro subscriptions or OpenAI API budgets. A student with a laptop running Ollama should get the same compounding knowledge system as an enterprise with a cloud budget.

Knowledge bases shouldn't be locked behind paywalls.

---

## License

MIT — do whatever you want with this.

---

## Acknowledgements

- [Andrej Karpathy](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f) for the original pattern
- The community implementations that validated the idea: `llmwiki`, `obsidian-wiki`, `second-brain`, `wiki-skills`
- [Tobi Lutke's QMD](https://github.com/tobi/qmd) for the search layer reference
