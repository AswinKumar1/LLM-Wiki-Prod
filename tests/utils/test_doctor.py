"""Tests for the wiki doctor pre-flight checker."""

import pytest
from pathlib import Path
from llm_wiki.utils.doctor import WikiDoctor
from llm_wiki.providers.base import ProviderConfig
from llm_wiki.wiki_fs import WikiFS


@pytest.fixture
def healthy_wiki(tmp_path):
    """A fully initialised wiki with all required files."""
    fs = WikiFS(tmp_path)
    fs.ensure_structure()
    fs.init_index()
    (tmp_path / "AGENTS.md").write_text("# Wiki schema\n")
    (tmp_path / "config.yaml").write_text("provider: ollama\nmodel: qwen2.5:3b\n")
    # Add a raw source so the raw-has-sources check passes
    (tmp_path / "raw" / "articles" / "test.md").write_text("# Article\nContent.")
    return tmp_path


@pytest.fixture
def valid_config():
    return ProviderConfig(provider="ollama", model="qwen2.5:3b")


def test_doctor_passes_on_healthy_wiki(healthy_wiki, valid_config):
    doctor = WikiDoctor(healthy_wiki, valid_config)
    report = doctor.run(check_provider=False)
    failures = [c for c in report.checks if not c.passed]
    # Only the provider check is skipped — everything else should pass
    assert len(failures) == 0, f"Unexpected failures: {failures}"


def test_doctor_fails_missing_root(tmp_path, valid_config):
    missing = tmp_path / "nonexistent"
    doctor = WikiDoctor(missing, valid_config)
    report = doctor.run(check_provider=False)
    root_check = next(c for c in report.checks if "root" in c.name.lower())
    assert not root_check.passed
    assert root_check.fix is not None


def test_doctor_fails_missing_agents_md(healthy_wiki, valid_config):
    (healthy_wiki / "AGENTS.md").unlink()
    doctor = WikiDoctor(healthy_wiki, valid_config)
    report = doctor.run(check_provider=False)
    agents_check = next(c for c in report.checks if "AGENTS.md" in c.name)
    assert not agents_check.passed


def test_doctor_fails_missing_config(healthy_wiki, valid_config):
    (healthy_wiki / "config.yaml").unlink()
    doctor = WikiDoctor(healthy_wiki, valid_config)
    report = doctor.run(check_provider=False)
    cfg_check = next(c for c in report.checks if "config.yaml" in c.name)
    assert not cfg_check.passed


def test_doctor_warns_no_raw_sources(healthy_wiki, valid_config):
    # Remove the raw source
    (healthy_wiki / "raw" / "articles" / "test.md").unlink()
    doctor = WikiDoctor(healthy_wiki, valid_config)
    report = doctor.run(check_provider=False)
    raw_check = next(c for c in report.checks if "Raw sources" in c.name)
    assert not raw_check.passed
    assert raw_check.fix is not None


def test_doctor_fails_invalid_config_fields(healthy_wiki):
    bad_config = ProviderConfig(
        provider="openai",
        model="gpt-4o",
        api_key="",  # missing key for openai
        temperature=3.5,  # out of range
    )
    doctor = WikiDoctor(healthy_wiki, bad_config)
    report = doctor.run(check_provider=False)
    cfg_check = next(c for c in report.checks if "Config fields" in c.name)
    assert not cfg_check.passed
    assert "api_key" in cfg_check.message.lower() or "temperature" in cfg_check.message.lower()


def test_doctor_report_format_contains_check_names(healthy_wiki, valid_config):
    doctor = WikiDoctor(healthy_wiki, valid_config)
    report = doctor.run(check_provider=False)
    fmt = report.format()
    assert "Wiki root" in fmt
    assert "Config fields" in fmt


def test_doctor_report_passed_property(healthy_wiki, valid_config):
    doctor = WikiDoctor(healthy_wiki, valid_config)
    report = doctor.run(check_provider=False)
    assert report.passed == all(c.passed for c in report.checks)


def test_doctor_report_failures_list(healthy_wiki, valid_config):
    (healthy_wiki / "AGENTS.md").unlink()
    doctor = WikiDoctor(healthy_wiki, valid_config)
    report = doctor.run(check_provider=False)
    assert len(report.failures) >= 1
    assert all(not c.passed for c in report.failures)
