"""
Token usage tracker.

Writes to wiki/usage.json — an append-only ledger of every LLM call made.
Provides cost estimates for known models and a summary CLI command.

Zero dependencies — uses stdlib json only.
"""

from __future__ import annotations

import json
from datetime import datetime, date
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Cost table (USD per 1k tokens, prompt / completion)
# Update as pricing changes — these are approximate as of mid-2025
# ---------------------------------------------------------------------------
_COST_TABLE: dict[str, tuple[float, float]] = {
    # Anthropic
    "claude-haiku-4-5-20251001": (0.00025, 0.00125),
    "claude-sonnet-4-6": (0.003, 0.015),
    "claude-opus-4-6": (0.015, 0.075),
    # OpenAI
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-4o": (0.005, 0.015),
    "o1-mini": (0.003, 0.012),
    # Ollama / local — zero cost
    "qwen2.5:3b": (0.0, 0.0),
    "mistral:7b": (0.0, 0.0),
    "llama3.2:3b": (0.0, 0.0),
    "phi3.5": (0.0, 0.0),
    "hermes3": (0.0, 0.0),
}

_USAGE_FILENAME = "usage.json"


class UsageTracker:
    """
    Records token usage to wiki/usage.json.

    Each entry:
        {
          "ts":         "2025-06-01T14:22:01",
          "op":         "ingest",
          "provider":   "ollama",
          "model":      "qwen2.5:3b",
          "source":     "my-article.md",   # optional
          "prompt_tokens":    812,
          "completion_tokens": 341,
          "cost_usd":   0.0
        }
    """

    def __init__(self, wiki_root: Path):
        self._path = wiki_root / "wiki" / _USAGE_FILENAME

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def record(
        self,
        op: str,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        source: Optional[str] = None,
    ) -> float:
        """Append a usage entry. Returns estimated cost in USD."""
        cost = _estimate_cost(model, prompt_tokens, completion_tokens)
        entry = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "op": op,
            "provider": provider,
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost_usd": round(cost, 6),
        }
        if source:
            entry["source"] = source

        self._path.parent.mkdir(parents=True, exist_ok=True)
        records = self._load()
        records.append(entry)
        self._path.write_text(json.dumps(records, indent=2), encoding="utf-8")
        return cost

    # ------------------------------------------------------------------
    # Read / summarise
    # ------------------------------------------------------------------

    def summary(self, since: Optional[date] = None) -> dict:
        """
        Return aggregated stats, optionally filtered by date.

        Returns:
            {
              "total_calls": int,
              "total_prompt_tokens": int,
              "total_completion_tokens": int,
              "total_cost_usd": float,
              "by_provider": { "ollama": {...}, ... },
              "by_op":       { "ingest": {...}, ... },
              "by_day":      { "2025-06-01": {...}, ... },
            }
        """
        records = self._load()
        if since:
            records = [r for r in records if r["ts"][:10] >= str(since)]

        summary: dict = {
            "total_calls": len(records),
            "total_prompt_tokens": sum(r["prompt_tokens"] for r in records),
            "total_completion_tokens": sum(r["completion_tokens"] for r in records),
            "total_cost_usd": round(sum(r["cost_usd"] for r in records), 6),
            "by_provider": {},
            "by_op": {},
            "by_day": {},
        }

        for r in records:
            _add(summary["by_provider"], r["provider"], r)
            _add(summary["by_op"], r["op"], r)
            _add(summary["by_day"], r["ts"][:10], r)

        return summary

    def format_summary(self, since: Optional[date] = None) -> str:
        s = self.summary(since=since)
        lines = [
            "Token usage summary",
            "─" * 38,
            f"  Total calls:       {s['total_calls']}",
            f"  Prompt tokens:     {s['total_prompt_tokens']:,}",
            f"  Completion tokens: {s['total_completion_tokens']:,}",
            f"  Total tokens:      {s['total_prompt_tokens'] + s['total_completion_tokens']:,}",
            f"  Est. cost (USD):   ${s['total_cost_usd']:.4f}",
            "",
            "  By provider:",
        ]
        for prov, d in sorted(s["by_provider"].items()):
            lines.append(
                f"    {prov:<18} {d['calls']} calls  "
                f"{d['prompt_tokens'] + d['completion_tokens']:,} tokens  "
                f"${d['cost_usd']:.4f}"
            )
        lines += ["", "  By operation:"]
        for op, d in sorted(s["by_op"].items()):
            lines.append(
                f"    {op:<18} {d['calls']} calls  "
                f"{d['prompt_tokens'] + d['completion_tokens']:,} tokens"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load(self) -> list[dict]:
        if not self._path.exists():
            return []
        try:
            return json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    # Exact match first, then prefix match (handles "qwen2.5:3b-instruct" etc.)
    rates = _COST_TABLE.get(model)
    if rates is None:
        for key, val in _COST_TABLE.items():
            if model.startswith(key.split(":")[0]):
                rates = val
                break
    if rates is None:
        return 0.0
    prompt_cost = (prompt_tokens / 1000) * rates[0]
    completion_cost = (completion_tokens / 1000) * rates[1]
    return prompt_cost + completion_cost


def _add(bucket: dict, key: str, record: dict) -> None:
    if key not in bucket:
        bucket[key] = {
            "calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cost_usd": 0.0,
        }
    b = bucket[key]
    b["calls"] += 1
    b["prompt_tokens"] += record["prompt_tokens"]
    b["completion_tokens"] += record["completion_tokens"]
    b["cost_usd"] = round(b["cost_usd"] + record["cost_usd"], 6)
