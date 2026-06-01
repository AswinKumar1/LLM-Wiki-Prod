"""
Configuration loader for llm-wiki-universal.

Priority order (highest wins):
  1. CLI flags (handled in cli.py)
  2. Environment variables  (WIKI_PROVIDER, WIKI_MODEL, etc.)
  3. config.yaml in the wiki root
  4. Built-in defaults

This keeps the system usable with zero config (just run `wiki init`
and it picks Ollama + qwen2.5:3b) while allowing full override for
power users or CI environments.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from .providers.base import ProviderConfig

# Environment variable names
_ENV_PREFIX = "WIKI_"
_ENV_MAP = {
    "WIKI_PROVIDER":    "provider",
    "WIKI_MODEL":       "model",
    "WIKI_BASE_URL":    "base_url",
    "WIKI_API_KEY":     "api_key",
    "WIKI_TEMPERATURE": "temperature",
    "WIKI_MAX_TOKENS":  "max_tokens",
    # Provider-specific keys are also accepted
    "OPENAI_API_KEY":   "api_key",
    "ANTHROPIC_API_KEY": "api_key",
    "OPENAI_COMPAT_API_KEY": "api_key",
}

_DEFAULTS = ProviderConfig(
    provider="ollama",
    model="qwen2.5:3b",
    base_url="",
    api_key="",
    temperature=0.2,
    max_tokens=4096,
    timeout=120,
)


def load_config(wiki_root: Optional[Path] = None) -> ProviderConfig:
    """
    Load ProviderConfig from config.yaml (if present) then overlay env vars.

    Args:
        wiki_root: path to the wiki directory. Defaults to cwd.
    """
    root = Path(wiki_root) if wiki_root else Path.cwd()
    config_path = root / "config.yaml"

    cfg = _load_yaml(config_path) if config_path.exists() else {}
    return _apply_env(_build_config(cfg))


def _load_yaml(path: Path) -> dict:
    """Parse config.yaml without requiring PyYAML as a hard dependency."""
    try:
        import yaml  # type: ignore
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        # Fallback: naive key: value parser (covers 95% of config files)
        result = {}
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or ":" not in line:
                    continue
                key, _, value = line.partition(":")
                result[key.strip()] = value.strip().strip('"').strip("'")
        return result


def _build_config(raw: dict) -> ProviderConfig:
    def _get(key: str, default):
        return raw.get(key, default)

    return ProviderConfig(
        provider=_get("provider", _DEFAULTS.provider),
        model=_get("model", _DEFAULTS.model),
        base_url=_get("base_url", _DEFAULTS.base_url),
        api_key=_get("api_key", _DEFAULTS.api_key),
        temperature=float(_get("temperature", _DEFAULTS.temperature)),
        max_tokens=int(_get("max_tokens", _DEFAULTS.max_tokens)),
        timeout=int(_get("timeout", _DEFAULTS.timeout)),
        extra={k: v for k, v in raw.items()
               if k not in {"provider", "model", "base_url", "api_key",
                            "temperature", "max_tokens", "timeout"}},
    )


def _apply_env(cfg: ProviderConfig) -> ProviderConfig:
    """Overlay environment variables onto a config object."""
    applied: dict = {}

    # Provider-specific API keys (lower priority than WIKI_API_KEY)
    for env_key, field in _ENV_MAP.items():
        val = os.environ.get(env_key)
        if val and field not in applied:
            applied[field] = val

    # WIKI_* vars take highest priority
    for env_key, field in _ENV_MAP.items():
        if env_key.startswith(_ENV_PREFIX):
            val = os.environ.get(env_key)
            if val:
                applied[field] = val

    for field, val in applied.items():
        if field == "temperature":
            val = float(val)
        elif field == "max_tokens":
            val = int(val)
        setattr(cfg, field, val)

    return cfg


def config_to_yaml(cfg: ProviderConfig) -> str:
    """Serialize a ProviderConfig back to YAML text for writing config.yaml."""
    lines = [
        "# llm-wiki-universal configuration",
        "# See: https://github.com/AswinKumar1/LLM-Wiki-Prod",
        "",
        f"provider: {cfg.provider}",
        f"model: {cfg.model}",
    ]
    if cfg.base_url:
        lines.append(f"base_url: {cfg.base_url}")
    if cfg.api_key:
        lines.append(f"api_key: {cfg.api_key}   # consider using env vars instead")
    lines += [
        f"temperature: {cfg.temperature}",
        f"max_tokens: {cfg.max_tokens}",
        f"timeout: {cfg.timeout}",
    ]
    return "\n".join(lines) + "\n"
