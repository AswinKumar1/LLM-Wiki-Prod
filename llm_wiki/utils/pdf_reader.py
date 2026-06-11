"""
PDF text extractor for wiki ingestion.

Strategy (tries each in order, uses first that works):
  1. pdfminer.six   — best quality, handles complex layouts
  2. pypdf          — lightweight, good for text-heavy PDFs
  3. pdfplumber     — good for table-heavy PDFs

All three are optional dependencies. If none are installed, returns a
clear error with install instructions rather than crashing.

Install (pick one):
    pip install pdfminer.six      # recommended
    pip install pypdf
    pip install pdfplumber

Usage:
    text, method = extract_pdf_text(Path("paper.pdf"))
    if text:
        # feed into ingest pipeline as normal
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_pdf_text(path: Path) -> tuple[Optional[str], str]:
    """
    Extract text from a PDF file.

    Returns:
        (text, method)  where method is "pdfminer" | "pypdf" | "pdfplumber"
        (None, error_message)  if extraction failed or no library available
    """
    path = Path(path)
    if not path.exists():
        return None, f"File not found: {path}"
    if not path.suffix.lower() == ".pdf":
        return None, f"Not a PDF file: {path.name}"

    # Try each extractor in preference order
    extractors = [
        (_try_pdfminer,   "pdfminer"),
        (_try_pypdf,      "pypdf"),
        (_try_pdfplumber, "pdfplumber"),
    ]

    last_error = ""
    for extractor_fn, name in extractors:
        try:
            text = extractor_fn(path)
            if text and text.strip():
                # Basic quality check — at least 100 chars of real text
                if len(text.strip()) >= 100:
                    return _clean_pdf_text(text), name
        except ImportError:
            continue
        except Exception as exc:
            last_error = str(exc)
            continue

    # Nothing worked
    if last_error:
        return None, f"PDF extraction failed: {last_error}"

    return None, _no_library_message()


def is_pdf_supported() -> tuple[bool, str]:
    """
    Check if any PDF library is available.
    Returns (supported: bool, library_name_or_install_hint: str)
    """
    for lib, name in [("pdfminer", "pdfminer.six"), ("pypdf", "pypdf"), ("pdfplumber", "pdfplumber")]:
        try:
            __import__(lib if lib != "pdfminer" else "pdfminer.high_level")
            return True, name
        except ImportError:
            continue
    return False, _no_library_message()


# ---------------------------------------------------------------------------
# Extractor implementations
# ---------------------------------------------------------------------------

def _try_pdfminer(path: Path) -> str:
    """Extract using pdfminer.six — best quality."""
    from pdfminer.high_level import extract_text  # type: ignore
    return extract_text(str(path))


def _try_pypdf(path: Path) -> str:
    """Extract using pypdf."""
    try:
        import pypdf  # type: ignore
        reader = pypdf.PdfReader(str(path))
    except ImportError:
        import PyPDF2 as pypdf  # type: ignore  # older name
        reader = pypdf.PdfReader(str(path))

    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n\n".join(pages)


def _try_pdfplumber(path: Path) -> str:
    """Extract using pdfplumber — good for tables."""
    import pdfplumber  # type: ignore
    pages = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
    return "\n\n".join(pages)


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def _clean_pdf_text(text: str) -> str:
    """
    Clean extracted PDF text:
    - Remove excessive blank lines (PDF extractors often produce many)
    - Fix common ligature issues (fi, fl, etc.)
    - Remove page headers/footers that repeat
    - Normalise unicode
    """
    import re
    import unicodedata

    # Normalise unicode (handles ligatures like ﬁ → fi)
    text = unicodedata.normalize("NFKC", text)

    # Fix common PDF extraction artefacts
    replacements = {
        "\x0c": "\n\n",   # form feed → paragraph break
        "\xa0": " ",       # non-breaking space
        "\u2019": "'",     # right single quotation
        "\u201c": '"',     # left double quotation
        "\u201d": '"',     # right double quotation
        "\u2013": "-",     # en dash
        "\u2014": "--",    # em dash
        "\ufb01": "fi",    # fi ligature
        "\ufb02": "fl",    # fl ligature
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)

    # Collapse 3+ consecutive blank lines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove lines that look like page numbers (lone numbers)
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)

    return text.strip()


# ---------------------------------------------------------------------------
# Install hint
# ---------------------------------------------------------------------------

def _no_library_message() -> str:
    return (
        "No PDF library found. Install one of:\n"
        "  pip install pdfminer.six   # recommended\n"
        "  pip install pypdf\n"
        "  pip install pdfplumber\n"
        "Then re-run: wiki ingest"
    )
