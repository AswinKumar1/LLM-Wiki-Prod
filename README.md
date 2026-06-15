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

# 4. Add a source (file, PDF, or URL)
cp ~/Downloads/some-article.md raw/articles/
wiki ingest --url https://arxiv.org/abs/2005.11401
wiki ingest raw/papers/attention.pdf

# 5. Ingest everything in raw/
wiki ingest

# 6. Search or ask questions
wiki search "retrieval augmented generation"
wiki query "What are the main ideas in this article?"

# 7. Check for contradictions
wiki nli
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
Optional: 
```bash
pip install pyyaml                  # cleaner config parsing
pip install pdfminer.six            # PDF ingestion (recommended)
pip install sentence-transformers   # NLI cross-encoder (more accurate contradiction detection)
```

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

### Quality & integrity
 
| Command | Description |
|---------|-------------|
| `wiki lint` | Full health-check including NLI contradiction scan |
| `wiki lint --skip-nli` | Structural checks only (fast) |
| `wiki lint --nli-backend cross_encoder` | Force cross-encoder NLI |
| `wiki nli` | Standalone NLI contradiction scan |
| `wiki nli --backend llm` | Force LLM backend for NLI |
| `wiki nli --backend cross_encoder` | Force cross-encoder backend |

### Diagnostics & monitoring
 
| Command | Description |
|---------|-------------|
| `wiki doctor` | Pre-flight check: config, dirs, API keys, provider health |
| `wiki status` | Wiki stats, search index size, recent ops |
| `wiki usage` | Token usage and estimated cost summary |
| `wiki usage --since 2025-06-01` | Filter usage to a date range |
| `wiki providers` | List all supported providers |
 
**Global flags:** `--root PATH`, `--provider NAME`, `--model NAME`, `--verbose / -v`

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
 
### Ingest
Drop any file into `raw/` or use `--url`. Supports `.md`, `.txt`, `.pdf`, and any URL. Large files (> 6k tokens) are auto-chunked. Failed calls retry 3× with backoff. Every source is SHA-256 fingerprinted.
 
### Search (BM25 — no LLM)
`wiki search "terms"` builds an in-memory BM25 index and returns ranked results in under 10ms. No API call, no embeddings, no vector database. Add `--rerank` for a synthesised LLM answer.
 
### Query (LLM)
Navigates the index, reads relevant pages, synthesises an answer with `[[wikilink]]` citations. Use `--stream` for live output. Save answers with `--save`.
 
### NLI Contradiction Detection (Day 4)
`wiki nli` scans all wiki pages for semantic contradictions at the sentence level.
 
**Two backends:**
- **LLM backend** (default, zero new deps): asks your configured LLM to classify sentence pairs as `ENTAILMENT / NEUTRAL / CONTRADICTION`. Works with Ollama, OpenAI, anything.
- **Cross-encoder backend** (optional, more accurate): uses `sentence-transformers` MiniLM, ~80MB, runs locally. ~0.84 F1 on NLI benchmarks.
```bash
# Install for better accuracy
pip install sentence-transformers
 
# Run standalone
wiki nli
 
# Or as part of full lint
wiki lint
```
 
When a contradiction is confirmed, the `confidence:` field in the affected page's frontmatter is automatically downgraded (`high → medium → low`), surfacing the issue in future search results and queries.
 
### Lint
Full health-check: NLI contradictions (sentence-level, cited, scored), orphan pages, broken wikilinks, low-confidence pages, missing frontmatter. Report saved to `outputs/lint-YYYY-MM-DD.md`.
 
---
 
## Large file handling
 
Files over ~6,000 tokens are automatically split into overlapping chunks:
 
```bash
wiki ingest --chunk-size 4000    # smaller chunks for 3B models
wiki ingest --chunk-size 8000    # larger chunks for GPT-4o
```
 
---
 
## Token usage tracking
 
```bash
wiki usage                       # all-time
wiki usage --since 2025-06-01    # from a date
```
 
Cost estimates for OpenAI and Anthropic. Ollama always shows $0.00.
 
---
 
## vs RAG vs Karpathy's original
 
| | Standard RAG | Karpathy's Base Wiki | **This project** |
|--|--|--|--|
| Provider | Any | Claude Code only | **Any (Ollama, OpenAI, Anthropic, custom)** |
| Infrastructure | Vector DB + embeddings | None | **None** |
| Keyword search | Vector similarity | None | **BM25 (stdlib, no deps)** |
| Contradiction detection | No | Manual | **NLI sentence-level, auto-cited (Day 4)** |
| Confidence tracking | No | Manual | **Auto-downgraded on contradiction** |
| PDF ingestion | Requires setup | Manual | **Built-in** |
| URL ingestion | Manual | Manual | **Built-in** |
| Large file handling | Chunked at query | Manual | **Auto-chunked at ingest** |
| Knowledge compounds | No | Yes | **Yes** |
| Retry on failure | No | No | **3× exponential backoff** |
| Cost visibility | No | No | **wiki usage** |
| Free tier | Vendor-dependent | No | **Yes (Ollama)** |
 
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

---

## License

MIT

---

## Acknowledgements

- [Andrej Karpathy](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f) for the original pattern
- The community implementations that validated the idea: `llmwiki`, `obsidian-wiki`, `second-brain`, `wiki-skills`
- [Tobi Lutke's QMD](https://github.com/tobi/qmd) for the search layer reference
