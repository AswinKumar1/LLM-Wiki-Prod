"""
Config validator and wiki doctor.

Runs pre-flight checks before any operation so users get clear, actionable
error messages instead of cryptic failures mid-run.

Checks:
  1. config.yaml is readable and has valid fields
  2. Provider is reachable (optional — skipped with --no-health-check)
  3. Wiki directory structure is intact
  4. AGENTS.md exists
  5. raw/ has at least one file (warns, doesn't error)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..providers.base import ProviderConfig


@dataclass
class CheckResult:
    name: str
    passed: bool
    message: str
    fix: Optional[str] = None  # actionable fix hint


@dataclass
class DoctorReport:
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def failures(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed]

    @property
    def warnings(self) -> list[CheckResult]:
        # Checks that didn't pass but have a fix hint (soft failures)
        return [c for c in self.checks if not c.passed and c.fix]

    def format(self, verbose: bool = False) -> str:
        lines = []
        for c in self.checks:
            icon = "✓" if c.passed else "✗"
            color_open = "\033[32m" if c.passed else "\033[31m"
            color_close = "\033[0m"
            lines.append(f"  {color_open}{icon}{color_close} {c.name}")
            if not c.passed:
                lines.append(f"      {c.message}")
                if c.fix:
                    lines.append(f"      → {c.fix}")
            elif verbose:
                lines.append(f"      {c.message}")
        return "\n".join(lines)


class WikiDoctor:
    """
    Run pre-flight checks on a wiki root and its configuration.

    Usage:
        doctor = WikiDoctor(wiki_root, config)
        report = doctor.run(check_provider=True)
        if not report.passed:
            print(report.format())
            sys.exit(1)
    """

    _REQUIRED_DIRS = [
        "raw",
        "wiki",
        "wiki/concepts",
        "wiki/entities",
        "wiki/sources",
    ]
    _REQUIRED_FILES = [
        "AGENTS.md",
        "config.yaml",
    ]

    def __init__(self, wiki_root: Path, config: ProviderConfig):
        self.root = Path(wiki_root).resolve()
        self.config = config

    def run(self, check_provider: bool = True) -> DoctorReport:
        report = DoctorReport()

        report.checks.append(self._check_root_exists())
        report.checks.extend(self._check_dirs())
        report.checks.extend(self._check_files())
        report.checks.append(self._check_config_fields())
        report.checks.append(self._check_raw_has_sources())

        if check_provider:
            report.checks.append(self._check_provider_reachable())

        return report

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _check_root_exists(self) -> CheckResult:
        if self.root.exists() and self.root.is_dir():
            return CheckResult("Wiki root exists", True, str(self.root))
        return CheckResult(
            "Wiki root exists",
            False,
            f"Directory not found: {self.root}",
            fix="Run: wiki init",
        )

    def _check_dirs(self) -> list[CheckResult]:
        results = []
        for d in self._REQUIRED_DIRS:
            full = self.root / d
            if full.exists():
                results.append(CheckResult(f"Directory: {d}/", True, str(full)))
            else:
                results.append(
                    CheckResult(
                        f"Directory: {d}/",
                        False,
                        f"Missing: {full}",
                        fix="Run: wiki init",
                    )
                )
        return results

    def _check_files(self) -> list[CheckResult]:
        results = []
        for f in self._REQUIRED_FILES:
            full = self.root / f
            if full.exists():
                results.append(CheckResult(f"File: {f}", True, str(full)))
            else:
                results.append(
                    CheckResult(
                        f"File: {f}",
                        False,
                        f"Missing: {full}",
                        fix=f"Run: wiki init  (recreates {f})",
                    )
                )
        return results

    def _check_config_fields(self) -> CheckResult:
        issues = []
        if not self.config.provider:
            issues.append("provider is empty")
        if not self.config.model:
            issues.append("model is empty")
        if self.config.temperature < 0 or self.config.temperature > 2:
            issues.append(f"temperature {self.config.temperature} out of range [0, 2]")
        if self.config.max_tokens < 256:
            issues.append(f"max_tokens {self.config.max_tokens} too small (min 256)")

        # API key checks for cloud providers
        if self.config.provider == "openai" and not self.config.api_key:
            issues.append("provider=openai but api_key is not set (set OPENAI_API_KEY)")
        if self.config.provider == "anthropic" and not self.config.api_key:
            issues.append("provider=anthropic but api_key is not set (set ANTHROPIC_API_KEY)")

        if issues:
            return CheckResult(
                "Config fields valid",
                False,
                "; ".join(issues),
                fix="Edit config.yaml or set the relevant env vars",
            )
        return CheckResult(
            "Config fields valid",
            True,
            f"provider={self.config.provider}  model={self.config.model}",
        )

    def _check_raw_has_sources(self) -> CheckResult:
        raw_dir = self.root / "raw"
        if not raw_dir.exists():
            return CheckResult(
                "Raw sources present",
                False,
                "raw/ directory missing",
                fix="Run: wiki init",
            )
        files = [p for p in raw_dir.rglob("*") if p.is_file() and not p.name.startswith(".")]
        if not files:
            return CheckResult(
                "Raw sources present",
                False,
                "No files found in raw/ — nothing to ingest yet",
                fix="Drop source files into raw/articles/ then run: wiki ingest",
            )
        return CheckResult(
            "Raw sources present",
            True,
            f"{len(files)} file(s) found in raw/",
        )

    def _check_provider_reachable(self) -> CheckResult:
        try:
            from ..providers.factory import get_provider

            provider = get_provider(self.config)
            ok = provider.health_check()
            if ok:
                return CheckResult(
                    f"Provider reachable ({self.config.provider})",
                    True,
                    f"Model {self.config.model!r} is available",
                )
            else:
                return CheckResult(
                    f"Provider reachable ({self.config.provider})",
                    False,
                    f"Health check failed — model {self.config.model!r} may not be loaded",
                    fix=_provider_fix(self.config),
                )
        except ConnectionError as exc:
            return CheckResult(
                f"Provider reachable ({self.config.provider})",
                False,
                str(exc),
                fix=_provider_fix(self.config),
            )
        except Exception as exc:
            return CheckResult(
                f"Provider reachable ({self.config.provider})",
                False,
                f"Unexpected error: {exc}",
                fix=_provider_fix(self.config),
            )


def _provider_fix(config: ProviderConfig) -> str:
    if config.provider == "ollama":
        return f"Run: ollama serve  then  ollama pull {config.model}"
    if config.provider == "openai":
        return "Check OPENAI_API_KEY is set and valid"
    if config.provider == "anthropic":
        return "Check ANTHROPIC_API_KEY is set and valid"
    return f"Check that your server at {config.base_url!r} is running"
