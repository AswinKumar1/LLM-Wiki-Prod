"""
Source reader for Day 3 — routes ingest to the right reader based on
file type or URL, then returns plain text for the ingest pipeline.

Supported:
  .md / .txt / .rst / .json / .csv / .yaml  — read directly as text
  .pdf                                        — extract via pdf_reader
  http:// / https://                         — fetch via url_fetcher

This module is imported by IngestOperation and by the CLI --url flag.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def read_source(path_or_url: str | Path) -> tuple[Optional[str], str, str]:
    """
    Read a source — file path or URL — and return its text content.

    Returns:
        (text, suggested_filename, method)
        text is None on failure; method describes what was used.

    The returned (text, filename) can be passed directly to the ingest
    pipeline without any further processing.
    """
    s = str(path_or_url)

    if s.startswith(("http://", "https://")):
        return _read_url(s)

    path = Path(s)
    if path.suffix.lower() == ".pdf":
        return _read_pdf(path)

    return _read_text(path)


def is_url(s: str) -> bool:
    return str(s).startswith(("http://", "https://"))


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------

def _read_text(path: Path) -> tuple[Optional[str], str, str]:
    _TEXT_EXTENSIONS = {
        ".md", ".txt", ".rst", ".text",
        ".json", ".yaml", ".yml",
        ".csv", ".tsv",
        ".py", ".js", ".ts", ".java", ".go", ".rs",  # code files
        ".html", ".htm", ".xml",
    }
    if path.suffix.lower() not in _TEXT_EXTENSIONS:
        # Unknown extension — try anyway
        pass
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        return text, path.name, "text"
    except OSError as exc:
        return None, path.name, f"error: {exc}"


def _read_pdf(path: Path) -> tuple[Optional[str], str, str]:
    from .pdf_reader import extract_pdf_text
    text, method = extract_pdf_text(path)
    if text is None:
        return None, path.name, method   # method contains error message
    # Wrap in markdown with metadata
    md = (
        f"---\n"
        f"title: {path.stem}\n"
        f"source_file: {path.name}\n"
        f"extracted_by: {method}\n"
        f"---\n\n"
        f"# {path.stem.replace('-', ' ').replace('_', ' ').title()}\n\n"
        f"{text}\n"
    )
    return md, f"{path.stem}.md", method


def _read_url(url: str) -> tuple[Optional[str], str, str]:
    from .url_fetcher import fetch_url
    result = fetch_url(url)
    if not result.success:
        return None, "", result.error or "fetch failed"
    filename = result.suggested_filename or "fetched-page.md"
    return result.as_markdown, filename, "url"
