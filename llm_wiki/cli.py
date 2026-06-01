#!/usr/bin/env python3
"""
llm-wiki-universal CLI

Commands:
  wiki init    [--provider ollama] [--model qwen2.5:3b]
  wiki ingest  [--source path/to/file]
  wiki query   "your question here"  [--save]
  wiki lint
  wiki status
  wiki providers

Run `wiki <command> --help` for options.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="wiki",
        description="LLM-Wiki-Universal — provider-agnostic knowledge base",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Wiki root directory (default: current directory)",
    )
    parser.add_argument(
        "--provider",
        help="Override provider from config (ollama | openai | anthropic | openai_compat)",
    )
    parser.add_argument(
        "--model",
        help="Override model from config",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # init
    p_init = subparsers.add_parser("init", help="Initialise a new wiki in the current directory")
    p_init.add_argument("--topic", default="", help="Wiki topic (used to generate AGENTS.md)")
    p_init.add_argument("--provider", default="ollama")
    p_init.add_argument("--model", default="")

    # ingest
    p_ingest = subparsers.add_parser("ingest", help="Ingest raw sources into the wiki")
    p_ingest.add_argument("--source", default=None, help="Ingest a specific file (default: all new)")
    p_ingest.add_argument("--max", type=int, default=None, help="Max number of sources to process")

    # query
    p_query = subparsers.add_parser("query", help="Ask a question against the wiki")
    p_query.add_argument("question", help="The question to answer")
    p_query.add_argument("--save", action="store_true", help="Save the answer as a wiki page")

    # lint
    subparsers.add_parser("lint", help="Health-check the wiki")

    # status
    subparsers.add_parser("status", help="Show wiki stats and provider health")

    # providers
    subparsers.add_parser("providers", help="List supported providers")

    args = parser.parse_args()

    # Route to command handlers
    dispatch = {
        "init":      cmd_init,
        "ingest":    cmd_ingest,
        "query":     cmd_query,
        "lint":      cmd_lint,
        "status":    cmd_status,
        "providers": cmd_providers,
    }
    dispatch[args.command](args)


# ------------------------------------------------------------------
# Command handlers
# ------------------------------------------------------------------

def cmd_init(args) -> None:
    from .config import load_config, config_to_yaml
    from .wiki_fs import WikiFS
    from .providers.base import ProviderConfig

    root = Path(args.root).resolve()

    cfg = ProviderConfig(
        provider=getattr(args, "provider", "ollama") or "ollama",
        model=getattr(args, "model", "") or "",
    )
    if not cfg.model:
        _set_default_model(cfg)

    fs = WikiFS(root)
    fs.ensure_structure()
    fs.init_index()

    # Write config.yaml
    if not fs.config_path.exists():
        fs.config_path.write_text(config_to_yaml(cfg))
        _print_ok(f"Created config.yaml  (provider: {cfg.provider}, model: {cfg.model})")
    else:
        _print_info("config.yaml already exists — skipped")

    # Write AGENTS.md
    if not fs.agents_md_path.exists():
        topic = getattr(args, "topic", "") or "General"
        agents_content = _default_agents_md(topic)
        fs.agents_md_path.write_text(agents_content)
        _print_ok("Created AGENTS.md")
    else:
        _print_info("AGENTS.md already exists — skipped")

    # Write .gitignore
    gitignore = root / ".gitignore"
    if not gitignore.exists():
        gitignore.write_text("outputs/*.pdf\n__pycache__/\n*.pyc\n.env\n")
        _print_ok("Created .gitignore")

    _print_ok(f"\nWiki initialised at: {root}")
    print()
    print("Next steps:")
    print(f"  1. Drop files into {root}/raw/articles/")
    print(f"  2. Run: wiki ingest")
    print(f"  3. Run: wiki query \"your question here\"")


def cmd_ingest(args) -> None:
    provider = _get_provider(args)
    from .wiki_fs import WikiFS
    from .operations import IngestOperation

    fs = WikiFS(Path(args.root).resolve())
    op = IngestOperation(provider, fs, verbose=args.verbose)

    if args.source:
        source_path = Path(args.source)
        if not source_path.exists():
            _print_err(f"Source file not found: {args.source}")
            sys.exit(1)
        results = [op.run_one(source_path)]
    else:
        results = op.run(max_sources=getattr(args, "max", None))

    if not results:
        _print_info("No new sources to ingest. Drop files into raw/ first.")
        return

    for r in results:
        if r.success:
            _print_ok(
                f"{r.source_name}: "
                f"{len(r.pages_created)} created, "
                f"{len(r.pages_updated)} updated "
                f"({r.tokens_used} tokens)"
            )
            if args.verbose and r.takeaways:
                print(f"\n  Takeaways:\n{r.takeaways}\n")
        else:
            _print_err(f"{r.source_name}: FAILED — {r.error}")


def cmd_query(args) -> None:
    provider = _get_provider(args)
    from .wiki_fs import WikiFS
    from .operations import QueryOperation

    fs = WikiFS(Path(args.root).resolve())
    op = QueryOperation(provider, fs, verbose=args.verbose)
    result = op.run(args.question, save_answer=args.save)

    if result.success:
        print()
        print(result.answer)
        print()
        if args.verbose:
            print(f"  Pages read: {result.pages_read}")
            print(f"  Tokens used: {result.tokens_used}")
        if result.saved_to:
            _print_ok(f"Answer saved to: {result.saved_to}")
    else:
        _print_err(f"Query failed: {result.error}")
        sys.exit(1)


def cmd_lint(args) -> None:
    provider = _get_provider(args)
    from .wiki_fs import WikiFS
    from .operations import LintOperation

    fs = WikiFS(Path(args.root).resolve())
    op = LintOperation(provider, fs, verbose=args.verbose)
    result = op.run()

    if result.success:
        _print_ok(f"Lint complete — {result.issue_count} issues found")
        if result.contradictions:
            print(f"\n  Contradictions ({len(result.contradictions)}):")
            for c in result.contradictions:
                print(f"    • {c}")
        if result.orphan_pages:
            print(f"\n  Orphan pages ({len(result.orphan_pages)}):")
            for p in result.orphan_pages:
                print(f"    • {p}")
        if result.missing_pages:
            print(f"\n  Missing pages ({len(result.missing_pages)}):")
            for p in result.missing_pages:
                print(f"    • [[{p}]]")
        if result.summary:
            print(f"\n  Summary: {result.summary}")
        if result.output_path:
            _print_info(f"Full report: {result.output_path}")
    else:
        _print_err(f"Lint failed: {result.error}")
        sys.exit(1)


def cmd_status(args) -> None:
    from .config import load_config
    from .wiki_fs import WikiFS
    from .providers.factory import get_provider

    root = Path(args.root).resolve()
    cfg = load_config(root)
    if getattr(args, "provider", None):
        cfg.provider = args.provider
    if getattr(args, "model", None):
        cfg.model = args.model

    fs = WikiFS(root)

    print(f"\n{'─'*40}")
    print(f"  Wiki root:   {root}")
    print(f"  Provider:    {cfg.provider}")
    print(f"  Model:       {cfg.model}")
    print(f"{'─'*40}")

    pages = fs.list_wiki_pages()
    sources = fs.list_raw_sources()
    new_sources = fs.new_raw_sources()
    print(f"  Wiki pages:  {len(pages)}")
    print(f"  Raw sources: {len(sources)} ({len(new_sources)} unprocessed)")
    print(f"  Last 3 ops:  {fs.read_log(3) or '(none)'}")
    print(f"{'─'*40}")

    # Provider health
    try:
        provider = get_provider(cfg)
        ok = provider.health_check()
        status = "✓ reachable" if ok else "✗ not reachable"
        print(f"  Provider health: {status}")
    except Exception as exc:
        print(f"  Provider health: ✗ error — {exc}")
    print()


def cmd_providers(args) -> None:
    from .providers.factory import list_providers
    print("\nSupported providers:")
    descriptions = {
        "ollama":        "Local models via Ollama (free, no API key) — default",
        "openai":        "OpenAI GPT-4o, GPT-4o-mini, o1 — requires OPENAI_API_KEY",
        "anthropic":     "Anthropic Claude — requires ANTHROPIC_API_KEY",
        "openai_compat": "Any OpenAI-compatible endpoint (LM Studio, vLLM, Groq, etc.)",
    }
    for p in list_providers():
        print(f"  {p:<18} {descriptions.get(p, '')}")
    print()
    print("Set provider in config.yaml or WIKI_PROVIDER env var.")
    print("Model aliases: hermes, lm_studio, vllm, groq, together → openai_compat")
    print()


# ------------------------------------------------------------------
# Shared helpers
# ------------------------------------------------------------------

def _get_provider(args):
    from .config import load_config
    from .providers.factory import get_provider

    root = Path(args.root).resolve()
    cfg = load_config(root)
    if getattr(args, "provider", None):
        cfg.provider = args.provider
    if getattr(args, "model", None):
        cfg.model = args.model

    try:
        provider = get_provider(cfg)
    except Exception as exc:
        _print_err(f"Could not initialise provider: {exc}")
        sys.exit(1)
    return provider


def _set_default_model(cfg) -> None:
    defaults = {
        "ollama": "qwen2.5:3b",
        "openai": "gpt-4o-mini",
        "anthropic": "claude-haiku-4-5-20251001",
        "openai_compat": "default",
    }
    cfg.model = defaults.get(cfg.provider, "")


def _default_agents_md(topic: str) -> str:
    return f"""# {topic} Wiki — Agent Schema
# Provider-agnostic. Works with Ollama, OpenAI, Anthropic, or any compatible LLM.
# Generated by llm-wiki-universal — edit to suit your domain.

## Project Structure

- `raw/`          — Immutable source documents. Never modify.
- `wiki/`         — LLM-generated and maintained markdown pages.
- `wiki/index.md` — Master content catalog. Updated on every ingest.
- `wiki/log.md`   — Append-only operation log.
- `outputs/`      — Generated reports, lint results.

## Page Types

Every wiki page must have YAML frontmatter:

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

## Naming

- Filenames: kebab-case (e.g., `attention-mechanism.md`)
- Cross-references: `[[wikilinks]]` for all internal links
- Source references: always link back to `raw/` file paths

## Ingest Workflow

1. Read source document
2. Identify key concepts and entities
3. Create `wiki/sources/<source-name>.md` summary
4. Update or create concept/entity pages with new information
5. Update `wiki/index.md`
6. Append to `wiki/log.md`

## Query Workflow

1. Read `wiki/index.md` to find relevant pages
2. Read those pages and synthesise an answer
3. Cite with `[[wikilinks]]`
4. Offer to save valuable answers as new pages

## Lint Workflow

1. Scan for contradictions between pages
2. Find orphan pages (no incoming links)
3. Find missing pages (broken wikilinks)
4. Flag low-confidence pages
5. Save report to `outputs/lint-YYYY-MM-DD.md`
"""


def _print_ok(msg: str) -> None:
    print(f"\033[32m✓\033[0m {msg}")


def _print_err(msg: str) -> None:
    print(f"\033[31m✗\033[0m {msg}", file=sys.stderr)


def _print_info(msg: str) -> None:
    print(f"\033[34mℹ\033[0m {msg}")


if __name__ == "__main__":
    main()
