"""Tests for the PDF reader utility."""

import pytest
from pathlib import Path
from llm_wiki.utils.pdf_reader import (
    is_pdf_supported,
    extract_pdf_text,
    _clean_pdf_text,
)


def test_is_pdf_supported_returns_tuple():
    supported, info = is_pdf_supported()
    assert isinstance(supported, bool)
    assert isinstance(info, str)
    # info is either a library name or an install message
    if supported:
        assert any(lib in info for lib in ("pdfminer", "pypdf", "pdfplumber"))
    else:
        assert "pip install" in info


def test_extract_nonexistent_file():
    text, method = extract_pdf_text(Path("/nonexistent/path/fake.pdf"))
    assert text is None
    assert "not found" in method.lower() or "error" in method.lower()


def test_extract_non_pdf_file(tmp_path):
    not_a_pdf = tmp_path / "test.txt"
    not_a_pdf.write_text("not a pdf")
    text, method = extract_pdf_text(not_a_pdf)
    assert text is None
    assert "Not a PDF" in method


def test_clean_pdf_text_removes_form_feeds():
    text = "Page one\x0cPage two"
    cleaned = _clean_pdf_text(text)
    assert "\x0c" not in cleaned
    assert "Page one" in cleaned
    assert "Page two" in cleaned


def test_clean_pdf_text_normalises_ligatures():
    text = "The \ufb01rst \ufb02oor of the building"
    cleaned = _clean_pdf_text(text)
    assert "fi" in cleaned
    assert "fl" in cleaned
    assert "\ufb01" not in cleaned


def test_clean_pdf_text_collapses_blank_lines():
    text = "Line one\n\n\n\n\nLine two"
    cleaned = _clean_pdf_text(text)
    assert "\n\n\n" not in cleaned


def test_clean_pdf_text_removes_lone_page_numbers():
    text = "Some content\n\n42\n\nMore content"
    cleaned = _clean_pdf_text(text)
    # Page number line should be gone
    lines = cleaned.splitlines()
    page_number_lines = [l for l in lines if l.strip() == "42"]
    assert len(page_number_lines) == 0


def test_clean_pdf_text_fixes_smart_quotes():
    text = "\u201cHello\u201d and \u2018world\u2019"
    cleaned = _clean_pdf_text(text)
    assert '"Hello"' in cleaned or "'Hello'" in cleaned


@pytest.mark.xfail(reason="requires reportlab + a pdf library", strict=False)
@pytest.mark.skipif(
    not is_pdf_supported()[0],
    reason="No PDF library installed — install pdfminer.six to run this test"
)
def test_extract_real_pdf_if_library_available(tmp_path):
    """
    This test only runs if a PDF library is installed.
    It creates a minimal valid PDF in memory and extracts text from it.
    """
    # Create a tiny valid PDF with reportlab if available
    try:
        from reportlab.pdfgen import canvas
        pdf_path = tmp_path / "test.pdf"
        c = canvas.Canvas(str(pdf_path))
        c.drawString(100, 750, "Hello from test PDF")
        c.save()
        text, method = extract_pdf_text(pdf_path)
        assert text is not None
        assert "Hello" in text
    except ImportError:
        pytest.skip("reportlab not available for PDF generation in test")
