#!/usr/bin/env python3
"""
llm-wiki-universal CLI — Day 4 additions

New commands:
  wiki nli                 — standalone NLI contradiction scan (no full lint)
  wiki nli --backend llm   — force LLM backend
  wiki nli --backend cross_encoder — force cross-encoder backend

Updated commands:
  wiki lint                — now runs NLI pass automatically
  wiki lint --skip-nli     — structural checks only (fast)
  wiki lint --nli-backend cross_encoder  — force a specific backend
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
    parser.add_argument("--root", default=".", help="Wiki root directory")
    parser.add_argument("--provider", help="Override provider from config")
    parser.add_argument("--model", help="Override model from config")
    parser.add_argument("--verbose", "-v", action="store_true")

    sub = parser.add_subparsers(dest="command", required=True)

    # init
    p_init = sub.add_parser("init", help="Initialise a new wiki")
    p_init.add_argument("--topic", default="")
    p_init.add_argument("--provider", default="ollama")
    p_init.add_argument("--model", default="")

    # ingest
    p_ingest = sub.add_parser("ingest", help="Ingest sources into the wiki")
    p_ingest.add_argument("--source", default=None)
    p_ingest.add_argument("--url", default=None)
    p_ingest.add_argument("--max", type=int, default=None)
    p_ingest.add_argument("--chunk-size", type=int, default=6000)

    # query
    p_query = sub.add_parser("query", help="Ask a question against the wiki")
    p_query.add_argument("question")
    p_query.add_argument("--save", action="store_true")
    p_query.add_argument("--stream", action="store_true")

    # search
    p_search = sub.add_parser("search", help="BM25 keyword search")
    p_search.add_argument("query")
    p_search.add_argument("--top", "-n", type=int, default=10)
    p_search.add_argument("--rerank", action="store_true")

    # lint  ← updated
    p_lint = sub.add_parser("lint", help="Health-check the wiki (includes NLI)")
    p_lint.add_argument(
        "--skip-nli",
        action="store_true",
        help="Skip NLI contradiction detection (structural checks only)",
    )
    p_lint.add_argument(
        "--nli-backend",
        default="auto",
        choices=["auto", "llm", "cross_encoder"],
        help="NLI backend to use (default: auto)",
    )

    # nli  ← NEW
    p_nli = sub.add_parser("nli", help="Standalone NLI contradiction scan")
    p_nli.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "llm", "cross_encoder"],
        help="NLI backend (default: auto — uses cross_encoder if installed)",
    )
    p_nli.add_argument(
        "--max-pairs",
        type=int,
        default=300,
        help="Max claim pairs to check (default: 300)",
    )

    # status
    sub.add_parser("status", help="Show wiki stats and provider health")

    # providers
    sub.add_parser("providers", help="List supported providers")

    # doctor
    p_doctor = sub.add_parser("doctor", help="Pre-flight check")
    p_doctor.add_argument("--no-health-check", action="store_true")

    # usage
    p_usage = sub.add_parser("usage", help="Token usage and cost summary")
    p_usage.add_argument("--since", default=None)

    args = parser.parse_args()

    dispatch = {
        "init": cmd_init,
        "ingest": cmd_ingest,
        "query": cmd_query,
        "search": cmd_search,
        "lint": cmd_lint,
        "nli": cmd_nli,
        "status": cmd_status,
        "providers": cmd_providers,
        "doctor": cmd_doctor,
        "usage": cmd_usage,
    }
    dispatch[args.command](args)


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------


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

    if not fs.config_path.exists():
        fs.config_path.write_text(config_to_yaml(cfg))
        _ok(f"Created config.yaml  (provider: {cfg.provider}, model: {cfg.model})")
    else:
        _info("config.yaml already exists — skipped")

    if not fs.agents_md_path.exists():
        topic = getattr(args, "topic", "") or "General"
        fs.agents_md_path.write_text(_default_agents_md(topic))
        _ok("Created AGENTS.md")

    gitignore = root / ".gitignore"
    if not gitignore.exists():
        gitignore.write_text("outputs/\n__pycache__/\n*.pyc\n.env\n")
        _ok("Created .gitignore")

    _ok(f"\nWiki initialised at: {root}")
    print(f"\n  Next: wiki ingest --url https://...  or  drop files into raw/")


def cmd_ingest(args) -> None:
    provider = _get_provider(args)
    from .wiki_fs import WikiFS
    from .operations.ingest import IngestOperation

    fs = WikiFS(Path(args.root).resolve())
    op = IngestOperation(
        provider,
        fs,
        verbose=args.verbose,
        max_tokens_per_chunk=getattr(args, "chunk_size", 6000),
    )

    url = getattr(args, "url", None)
    if url:
        _info(f"Fetching: {url}")
        result = op.run_url(url)
        _print_ingest_result(result, args.verbose)
        return

    if args.source:
        source_path = Path(args.source)
        if not source_path.exists():
            _err(f"File not found: {args.source}")
            sys.exit(1)
        results = [op.run_one(source_path)]
    else:
        results = op.run(max_sources=getattr(args, "max", None))

    if not results:
        _info("No new sources to ingest. Drop files into raw/ or use --url.")
        return

    total_tokens = 0
    total_cost = 0.0
    for r in results:
        _print_ingest_result(r, args.verbose)
        total_tokens += r.tokens_used
        total_cost += r.cost_usd

    if len(results) > 1:
        cost_str = f"  total cost: ${total_cost:.4f}" if total_cost > 0 else ""
        print(f"\n  Total: {total_tokens:,} tokens{cost_str}")


def cmd_query(args) -> None:
    provider = _get_provider(args)
    from .wiki_fs import WikiFS
    from .operations.query import QueryOperation
    from .utils.usage import UsageTracker

    fs = WikiFS(Path(args.root).resolve())
    tracker = UsageTracker(fs.root)
    op = QueryOperation(provider, fs, verbose=args.verbose)

    if getattr(args, "stream", False):
        _run_streaming_query(provider, fs, op, args, tracker)
        return

    result = op.run(args.question, save_answer=args.save)
    if result.success:
        print()
        print(result.answer)
        print()
        if result.tokens_used:
            cost = tracker.record(
                op="query",
                provider=provider.provider_name,
                model=provider.model_name,
                prompt_tokens=result.tokens_used // 2,
                completion_tokens=result.tokens_used // 2,
            )
            if args.verbose:
                cost_str = f"  ${cost:.4f}" if cost > 0 else ""
                print(f"  Pages: {result.pages_read}")
                print(f"  Tokens: {result.tokens_used:,}{cost_str}")
        if result.saved_to:
            _ok(f"Saved to: {result.saved_to}")
    else:
        _err(f"Query failed: {result.error}")
        sys.exit(1)


def cmd_search(args) -> None:
    from .wiki_fs import WikiFS
    from .operations.search import SearchOperation

    fs = WikiFS(Path(args.root).resolve())
    if getattr(args, "rerank", False):
        provider = _get_provider(args)
    else:
        provider = _null_provider()

    op = SearchOperation(provider, fs, verbose=args.verbose)
    resp = op.search(
        args.query,
        top_k=getattr(args, "top", 10),
        rerank=getattr(args, "rerank", False),
    )

    if not resp.success:
        _err(resp.error or "Search failed")
        sys.exit(1)

    print(f'\n  Search: "{args.query}"  ({resp.total_docs_searched} pages indexed)\n')

    if not resp.found:
        _info("No results found. Try broader search terms.")
        return

    for i, result in enumerate(resp.results, 1):
        print(result.format(i))

    if resp.answer:
        print("  ─" * 20)
        print(f"\n  Answer:\n\n  {resp.answer}\n")


def cmd_lint(args) -> None:
    """Run full lint including NLI contradiction detection."""
    provider = _get_provider(args)
    from .wiki_fs import WikiFS
    from .operations.lint import LintOperation

    fs = WikiFS(Path(args.root).resolve())
    op = LintOperation(
        provider,
        fs,
        verbose=args.verbose,
        nli_backend=getattr(args, "nli_backend", "auto"),
    )
    skip_nli = getattr(args, "skip_nli", False)
    result = op.run(skip_nli=skip_nli)

    if not result.success:
        _err(f"Lint failed: {result.error}")
        sys.exit(1)

    _ok(f"Lint complete — {result.issue_count} total issues")

    # NLI results
    if not skip_nli:
        print(
            f"\n  NLI scan ({result.nli_backend} backend): "
            f"{result.nli_pages_checked} pages, "
            f"{result.nli_pairs_checked} pairs checked"
        )
        if result.nli_contradictions:
            print(f"\n  Contradictions ({len(result.nli_contradictions)}):")
            for c in result.nli_contradictions:
                print(c.format())
        else:
            print("  Contradictions: none found ✓")
        if result.pages_downgraded:
            print(f"\n  Confidence downgraded: {len(result.pages_downgraded)} page(s)")
            for p in result.pages_downgraded:
                print(f"    • {p}")

    # Structural results
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
        _info(f"\n  Full report: {result.output_path}")


def cmd_nli(args) -> None:
    """Standalone NLI contradiction scan — faster than full lint."""
    provider = _get_provider(args)
    from .wiki_fs import WikiFS
    from ..llm_wiki.utils.nli import NLIEngine

    fs = WikiFS(Path(args.root).resolve())
    pages_paths = fs.list_wiki_pages()

    if not pages_paths:
        _err("No wiki pages found. Run `wiki ingest` first.")
        sys.exit(1)

    # Load all pages
    pages: dict[str, str] = {}
    for p in pages_paths:
        rel = fs.relative_to_root(p)
        pages[rel] = p.read_text(encoding="utf-8", errors="replace")

    backend = getattr(args, "backend", "auto")
    max_pairs = getattr(args, "max_pairs", 300)

    print(f"\n  NLI scan: {len(pages)} pages, backend={backend}\n")

    try:
        engine = NLIEngine(provider, backend=backend, verbose=args.verbose)
        _info(f"Using backend: {engine.backend_name}")
    except ImportError as exc:
        _err(str(exc))
        sys.exit(1)

    result = engine.scan_pages(pages, max_pairs=max_pairs)

    if not result.success:
        _err(f"NLI scan failed: {result.error}")
        sys.exit(1)

    print(f"\n  Pages checked:      {result.pages_checked}")
    print(f"  Claim pairs scored: {result.pairs_checked}")
    print(f"  Contradictions:     {result.contradiction_count}\n")

    if result.contradictions:
        print("  Contradictions found:\n")
        for c in result.contradictions:
            print(c.format())

        # Offer to downgrade confidence
        affected = result.pages_with_contradictions()
        print(f"  Affected pages: {len(affected)}")
        for p in sorted(affected):
            print(f"    • {p}")
        print()
        _info("Run `wiki lint` to downgrade confidence and save a full report.")
    else:
        _ok("No contradictions detected.")


def cmd_status(args) -> None:
    from .config import load_config
    from .wiki_fs import WikiFS
    from .providers.factory import get_provider
    from .utils.usage import UsageTracker
    from .utils.search import BM25Index

    root = Path(args.root).resolve()
    cfg = load_config(root)
    if getattr(args, "provider", None):
        cfg.provider = args.provider
    if getattr(args, "model", None):
        cfg.model = args.model

    fs = WikiFS(root)
    tracker = UsageTracker(root)

    print(f"\n{'─' * 44}")
    print(f"  Wiki root:   {root}")
    print(f"  Provider:    {cfg.provider}  ({cfg.model})")
    print(f"{'─' * 44}")

    pages = fs.list_wiki_pages()
    sources = fs.list_raw_sources()
    new_sources = fs.new_raw_sources()
    print(f"  Wiki pages:  {len(pages)}")
    print(f"  Raw sources: {len(sources)}  ({len(new_sources)} unprocessed)")

    if pages:
        index = BM25Index.build(fs)
        print(f"  Search idx:  {index.doc_count} docs, {index.vocab_size} terms")

    log = fs.read_log(3)
    if log:
        print(f"  Recent ops:")
        for line in log.splitlines():
            print(f"    {line}")

    summary = tracker.summary()
    if summary["total_calls"] > 0:
        print(f"{'─' * 44}")
        total_tok = summary["total_prompt_tokens"] + summary["total_completion_tokens"]
        print(f"  Tokens:      {total_tok:,} total")
        print(f"  Est. cost:   ${summary['total_cost_usd']:.4f} USD")

    print(f"{'─' * 44}")
    try:
        provider = get_provider(cfg)
        ok = provider.health_check()
        print(f"  Provider:    {'✓ reachable' if ok else '✗ not reachable'}")
    except Exception as exc:
        print(f"  Provider:    ✗ {exc}")
    print()


def cmd_providers(args) -> None:
    from .providers.factory import list_providers

    print("\nSupported providers:")
    descriptions = {
        "ollama": "Local models via Ollama — free, no API key (default)",
        "openai": "OpenAI GPT-4o / o1 — requires OPENAI_API_KEY",
        "anthropic": "Anthropic Claude — requires ANTHROPIC_API_KEY",
        "openai_compat": "Any OpenAI-compatible endpoint (LM Studio, vLLM, Groq…)",
    }
    for p in list_providers():
        print(f"  {p:<18} {descriptions.get(p, '')}")
    print()


def cmd_doctor(args) -> None:
    from .config import load_config
    from .wiki_fs import WikiFS
    from .utils.doctor import WikiDoctor

    root = Path(args.root).resolve()
    cfg = load_config(root)
    if getattr(args, "provider", None):
        cfg.provider = args.provider
    if getattr(args, "model", None):
        cfg.model = args.model

    check_provider = not getattr(args, "no_health_check", False)
    print(f"\n  Running wiki doctor on: {root}\n")
    doctor = WikiDoctor(root, cfg)
    report = doctor.run(check_provider=check_provider)
    print(report.format(verbose=args.verbose))
    print()
    if report.passed:
        _ok("All checks passed — wiki is ready.")
    else:
        _err(f"{len(report.failures)} check(s) failed.")
        sys.exit(1)


def cmd_usage(args) -> None:
    from .utils.usage import UsageTracker
    from datetime import date

    root = Path(args.root).resolve()
    tracker = UsageTracker(root)
    since = None
    since_str = getattr(args, "since", None)
    if since_str:
        try:
            since = date.fromisoformat(since_str)
        except ValueError:
            _err(f"Invalid date: {since_str!r}  (use YYYY-MM-DD)")
            sys.exit(1)
    print()
    print(tracker.format_summary(since=since))
    print()


# ---------------------------------------------------------------------------
# Streaming query (carried from Day 2/3)
# ---------------------------------------------------------------------------


def _run_streaming_query(provider, fs, op, args, tracker) -> None:
    from .prompts import query_find_relevant_pages, query_question
    import json, re

    index_content = fs.read_index()
    if not index_content.strip():
        _err("Wiki index is empty. Run `wiki ingest` first.")
        sys.exit(1)

    sys_fp, usr_fp = query_find_relevant_pages(args.question, index_content)
    try:
        page_resp = provider.chat(sys_fp, usr_fp, max_tokens=512)
    except Exception as exc:
        _err(f"Failed to find pages: {exc}")
        sys.exit(1)

    paths: list[str] = []
    try:
        match = re.search(r"\[.*?\]", page_resp.content, re.DOTALL)
        if match:
            paths = json.loads(match.group())
    except Exception:
        paths = [str(fs.relative_to_root(p)) for p in fs.list_wiki_pages()[:8]]

    pages: dict[str, str] = {}
    for path in paths:
        content = fs.read_wiki_page(path)
        if content:
            pages[path] = content
    if not pages:
        pages = {"wiki/index.md": index_content}

    system, user = query_question(args.question, pages)
    print()
    total_chars = 0
    try:
        for token in provider.stream(system, user):
            print(token, end="", flush=True)
            total_chars += len(token)
    except Exception as exc:
        print()
        _err(f"Stream error: {exc}")
        sys.exit(1)
    print("\n")
    tracker.record(
        op="query",
        provider=provider.provider_name,
        model=provider.model_name,
        prompt_tokens=total_chars // 4,
        completion_tokens=total_chars // 4,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
        return get_provider(cfg)
    except Exception as exc:
        _err(f"Provider init failed: {exc}")
        sys.exit(1)


def _null_provider():
    from .providers.base import LLMProvider, LLMResponse, ProviderConfig

    class NullProvider(LLMProvider):
        def chat(self, *a, **kw):
            return LLMResponse(content="", model="null", provider="null")

        def health_check(self):
            return True

        @property
        def provider_name(self):
            return "null"

    return NullProvider(ProviderConfig())


def _print_ingest_result(r, verbose: bool) -> None:
    if r.success:
        type_tag = f" [{r.source_type}]" if r.source_type != "text" else ""
        chunk_note = f" ({r.chunks_processed} chunks)" if r.chunks_processed > 1 else ""
        cost_note = f"  ${r.cost_usd:.4f}" if r.cost_usd > 0 else ""
        _ok(
            f"{r.source_name}{type_tag}{chunk_note}: "
            f"{len(r.pages_created)} created, "
            f"{len(r.pages_updated)} updated  "
            f"[{r.tokens_used} tokens{cost_note}]"
        )
        if verbose and r.takeaways:
            print(f"\n  Takeaways:\n{r.takeaways}\n")
    else:
        _err(f"{r.source_name}: FAILED — {r.error}")


def _set_default_model(cfg) -> None:
    defaults = {
        "ollama": "qwen2.5:3b",
        "openai": "gpt-4o-mini",
        "anthropic": "claude-haiku-4-5-20251001",
        "openai_compat": "default",
    }
    cfg.model = defaults.get(cfg.provider, "")


def _default_agents_md(topic: str) -> str:
    return (
        f"# {topic} Wiki — Agent Schema\n"
        "# Provider-agnostic. Works with Ollama, OpenAI, Anthropic, or any compatible LLM.\n\n"
        "## Structure\n"
        "- raw/       Immutable sources (.md .txt .pdf or fetched URLs)\n"
        "- wiki/      LLM-generated markdown pages\n"
        "- outputs/   Lint reports, NLI reports\n\n"
        "## Page frontmatter\n"
        "---\ntitle: Page Title\ntype: concept | entity | source-summary\n"
        "created: YYYY-MM-DD\nupdated: YYYY-MM-DD\nconfidence: high | medium | low\n---\n\n"
        "## Naming: kebab-case files, [[wikilinks]] for cross-references\n"
    )


def _ok(msg: str) -> None:
    print(f"\033[32m✓\033[0m {msg}")


def _err(msg: str) -> None:
    print(f"\033[31m✗\033[0m {msg}", file=sys.stderr)


def _info(msg: str) -> None:
    print(f"\033[34mℹ\033[0m {msg}")


if __name__ == "__main__":
    main()
