"""Tests for the token usage tracker."""

import json
import pytest
from pathlib import Path
from llm_wiki.utils.usage import UsageTracker, _estimate_cost


@pytest.fixture
def tracker(tmp_path):
    (tmp_path / "wiki").mkdir()
    return UsageTracker(tmp_path)


def test_record_creates_file(tracker, tmp_path):
    tracker.record("ingest", "ollama", "qwen2.5:3b", 100, 50)
    usage_file = tmp_path / "wiki" / "usage.json"
    assert usage_file.exists()


def test_record_entry_structure(tracker, tmp_path):
    tracker.record("ingest", "ollama", "qwen2.5:3b", 100, 50, source="test.md")
    data = json.loads((tmp_path / "wiki" / "usage.json").read_text())
    assert len(data) == 1
    entry = data[0]
    assert entry["op"] == "ingest"
    assert entry["provider"] == "ollama"
    assert entry["model"] == "qwen2.5:3b"
    assert entry["prompt_tokens"] == 100
    assert entry["completion_tokens"] == 50
    assert entry["source"] == "test.md"
    assert "ts" in entry
    assert "cost_usd" in entry


def test_record_multiple_entries(tracker):
    tracker.record("ingest", "ollama", "qwen2.5:3b", 100, 50)
    tracker.record("query", "ollama", "qwen2.5:3b", 200, 80)
    tracker.record("lint", "ollama", "qwen2.5:3b", 150, 60)
    summary = tracker.summary()
    assert summary["total_calls"] == 3


def test_summary_totals(tracker):
    tracker.record("ingest", "openai", "gpt-4o-mini", 1000, 500)
    tracker.record("query", "openai", "gpt-4o-mini", 800, 300)
    s = tracker.summary()
    assert s["total_prompt_tokens"] == 1800
    assert s["total_completion_tokens"] == 800
    assert s["total_calls"] == 2


def test_summary_by_provider(tracker):
    tracker.record("ingest", "ollama", "qwen2.5:3b", 100, 50)
    tracker.record("query", "anthropic", "claude-haiku-4-5-20251001", 200, 80)
    s = tracker.summary()
    assert "ollama" in s["by_provider"]
    assert "anthropic" in s["by_provider"]
    assert s["by_provider"]["ollama"]["calls"] == 1
    assert s["by_provider"]["anthropic"]["calls"] == 1


def test_summary_by_op(tracker):
    tracker.record("ingest", "ollama", "qwen2.5:3b", 100, 50)
    tracker.record("ingest", "ollama", "qwen2.5:3b", 120, 60)
    tracker.record("query", "ollama", "qwen2.5:3b", 80, 40)
    s = tracker.summary()
    assert s["by_op"]["ingest"]["calls"] == 2
    assert s["by_op"]["query"]["calls"] == 1


def test_ollama_cost_is_zero(tracker):
    cost = tracker.record("ingest", "ollama", "qwen2.5:3b", 10000, 5000)
    assert cost == 0.0


def test_openai_cost_nonzero(tracker):
    cost = tracker.record("ingest", "openai", "gpt-4o-mini", 1000, 500)
    assert cost > 0.0


def test_estimate_cost_known_model():
    cost = _estimate_cost("gpt-4o-mini", 1000, 500)
    # 1000 * 0.00015/1000 + 500 * 0.0006/1000 = 0.00015 + 0.0003 = 0.00045
    assert abs(cost - 0.00045) < 1e-6


def test_estimate_cost_unknown_model_zero():
    cost = _estimate_cost("some-unknown-model-xyz", 1000, 500)
    assert cost == 0.0


def test_empty_summary(tracker):
    s = tracker.summary()
    assert s["total_calls"] == 0
    assert s["total_cost_usd"] == 0.0


def test_format_summary_no_crash(tracker):
    tracker.record("ingest", "ollama", "qwen2.5:3b", 500, 200)
    output = tracker.format_summary()
    assert "Token usage summary" in output
    assert "ollama" in output


def test_summary_since_filter(tracker):
    from datetime import date, timedelta

    tracker.record("ingest", "ollama", "qwen2.5:3b", 100, 50)
    # Filter to tomorrow — should exclude today's entries
    tomorrow = date.today() + timedelta(days=1)
    s = tracker.summary(since=tomorrow)
    assert s["total_calls"] == 0
