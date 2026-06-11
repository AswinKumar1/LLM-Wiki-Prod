"""
URL fetcher for wiki ingestion.

Fetches a web page, strips HTML boilerplate, and returns clean text
suitable for feeding into the ingest pipeline.

Zero dependencies — uses only Python stdlib (urllib, html.parser).

Features:
  - Follows redirects
  - Handles common encodings
  - Strips nav, header, footer, script, style elements
  - Preserves headings, paragraphs, lists as plain text
  - Handles arXiv abstract pages specially (extracts title + abstract)
  - Returns metadata (title, url, fetch_date) for frontmatter injection

Usage:
    result = fetch_url("https://arxiv.org/abs/2005.11401")
    if result.success:
        print(result.text)
        print(result.title)
"""

from __future__ import annotations

import html
import re
import urllib.request
import urllib.error
from dataclasses import dataclass
from datetime import date
from html.parser import HTMLParser
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class FetchResult:
    url: str
    text: str
    title: str
    success: bool
    error: Optional[str] = None
    suggested_filename: Optional[str] = None

    @property
    def as_markdown(self) -> str:
        """Wrap fetched content in a markdown document with metadata."""
        return (
            f"---\n"
            f"title: {self.title}\n"
            f"source_url: {self.url}\n"
            f"fetched: {date.today()}\n"
            f"type: web-article\n"
            f"---\n\n"
            f"# {self.title}\n\n"
            f"> Source: {self.url}\n\n"
            f"{self.text}\n"
        )


def fetch_url(url: str, timeout: int = 30) -> FetchResult:
    """
    Fetch a URL and return cleaned text content.

    Handles:
      - HTML pages (strips boilerplate)
      - arXiv abstract pages (extracts abstract cleanly)
      - GitHub READMEs (fetches raw content)
      - Plain text / markdown files
    """
    url = _normalise_url(url)

    # GitHub: redirect to raw content for markdown files
    if _is_github_file(url):
        url = _github_raw_url(url)

    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (compatible; llm-wiki-universal/0.1; "
                    "+https://github.com/AswinKumar1/LLM-Wiki-Prod)"
                ),
                "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9",
            },
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw_bytes = resp.read()
            content_type = resp.headers.get("Content-Type", "text/html")
            final_url = resp.url  # after redirects
    except urllib.error.HTTPError as exc:
        return FetchResult(url=url, text="", title="", success=False,
                           error=f"HTTP {exc.code}: {exc.reason}")
    except urllib.error.URLError as exc:
        return FetchResult(url=url, text="", title="", success=False,
                           error=f"Cannot reach URL: {exc.reason}")
    except Exception as exc:
        return FetchResult(url=url, text="", title="", success=False,
                           error=str(exc))

    # Decode bytes
    encoding = _detect_encoding(content_type, raw_bytes)
    try:
        raw_text = raw_bytes.decode(encoding, errors="replace")
    except (LookupError, UnicodeDecodeError):
        raw_text = raw_bytes.decode("utf-8", errors="replace")

    # Route to appropriate parser
    if "text/html" in content_type or raw_text.lstrip().startswith("<"):
        title, text = _parse_html(raw_text, final_url)
    else:
        # Plain text / markdown
        title = _url_to_title(final_url)
        text  = raw_text

    if not text.strip():
        return FetchResult(url=final_url, text="", title=title, success=False,
                           error="No readable content extracted from page")

    filename = _url_to_filename(final_url)

    return FetchResult(
        url=final_url,
        text=text.strip(),
        title=title,
        success=True,
        suggested_filename=filename,
    )


# ---------------------------------------------------------------------------
# HTML parser
# ---------------------------------------------------------------------------

class _WikiParser(HTMLParser):
    """
    Minimal HTML → text parser that discards boilerplate elements.

    Keeps: headings, paragraphs, list items, blockquotes, code blocks
    Discards: nav, header, footer, aside, script, style, form, ads
    """

    _SKIP_TAGS = frozenset({
        "script", "style", "noscript", "nav", "header", "footer",
        "aside", "form", "button", "input", "select", "textarea",
        "iframe", "embed", "object", "svg", "canvas", "figure",
        "figcaption", "advertisement", "ads",
    })
    _BLOCK_TAGS = frozenset({
        "p", "div", "section", "article", "main", "blockquote",
        "pre", "li", "dt", "dd", "tr", "td", "th",
    })
    _HEADING_TAGS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})

    def __init__(self):
        super().__init__()
        self._parts:    list[str] = []
        self._skip_depth = 0
        self._title:    str = ""
        self._in_title: bool = False
        self._current_heading: Optional[str] = None

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag == "title":
            self._in_title = True
        elif tag in self._HEADING_TAGS:
            self._current_heading = tag
            self._parts.append("\n\n")
            level = int(tag[1])
            self._parts.append("#" * level + " ")
        elif tag in self._BLOCK_TAGS:
            self._parts.append("\n")
        elif tag == "br":
            self._parts.append("\n")
        elif tag == "li":
            self._parts.append("\n- ")
        elif tag == "a":
            pass  # ignore links, keep text

    def handle_endtag(self, tag):
        if tag in self._SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if tag == "title":
            self._in_title = False
        elif tag in self._HEADING_TAGS:
            self._current_heading = None
            self._parts.append("\n")
        elif tag in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data):
        if self._skip_depth:
            return
        if self._in_title:
            self._title += data
            return
        text = html.unescape(data)
        self._parts.append(text)

    @property
    def text(self) -> str:
        raw = "".join(self._parts)
        # Collapse 3+ newlines
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        # Remove lines that are just whitespace
        lines = [l.rstrip() for l in raw.splitlines()]
        return "\n".join(lines).strip()

    @property
    def title(self) -> str:
        return self._title.strip()


def _parse_html(html_text: str, url: str) -> tuple[str, str]:
    """Parse HTML and return (title, cleaned_text)."""
    # Special handling for arXiv
    if "arxiv.org/abs/" in url:
        return _parse_arxiv(html_text)

    parser = _WikiParser()
    try:
        parser.feed(html_text)
    except Exception:
        pass

    title = parser.title or _url_to_title(url)
    text  = parser.text

    # If very little text extracted, the page might be JS-rendered
    if len(text) < 200:
        text = (
            f"{text}\n\n"
            f"[Note: limited text extracted — page may require JavaScript. "
            f"Consider saving the page content manually to raw/articles/.]"
        )

    return title, text


def _parse_arxiv(html_text: str) -> tuple[str, str]:
    """Specialised extractor for arXiv abstract pages."""
    title_match = re.search(
        r'<h1[^>]*class="[^"]*title[^"]*"[^>]*>(.*?)</h1>', html_text, re.DOTALL
    )
    abstract_match = re.search(
        r'<blockquote[^>]*class="[^"]*abstract[^"]*"[^>]*>(.*?)</blockquote>',
        html_text, re.DOTALL,
    )
    authors_match = re.search(
        r'<div[^>]*class="[^"]*authors[^"]*"[^>]*>(.*?)</div>', html_text, re.DOTALL
    )

    def clean_tag(s: str) -> str:
        return html.unescape(re.sub(r"<[^>]+>", "", s)).strip()

    title    = clean_tag(title_match.group(1))    if title_match    else "arXiv Paper"
    abstract = clean_tag(abstract_match.group(1)) if abstract_match else ""
    authors  = clean_tag(authors_match.group(1))  if authors_match  else ""

    # Remove the "Abstract:" prefix arXiv adds
    abstract = re.sub(r"^Abstract:\s*", "", abstract, flags=re.IGNORECASE)

    parts = []
    if authors:
        parts.append(f"**Authors:** {authors}")
    if abstract:
        parts.append(f"## Abstract\n\n{abstract}")

    return title, "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise_url(url: str) -> str:
    if not url.startswith(("http://", "https://")):
        return "https://" + url
    return url


def _detect_encoding(content_type: str, raw: bytes) -> str:
    # Try Content-Type header first
    match = re.search(r"charset=([^\s;]+)", content_type, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # Sniff BOM
    if raw[:3] == b"\xef\xbb\xbf":
        return "utf-8-sig"
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return "utf-16"
    return "utf-8"


def _url_to_title(url: str) -> str:
    parsed = urlparse(url)
    path   = parsed.path.rstrip("/")
    if path:
        stem = Path(path).stem
        return stem.replace("-", " ").replace("_", " ").title()
    return parsed.netloc


def _url_to_filename(url: str) -> str:
    """Convert a URL to a safe filename for raw/articles/."""
    parsed = urlparse(url)
    # Use domain + path
    domain = parsed.netloc.replace("www.", "").replace(".", "-")
    path   = parsed.path.strip("/").replace("/", "-")
    slug   = re.sub(r"[^a-z0-9-]", "-", f"{domain}-{path}".lower())
    slug   = re.sub(r"-{2,}", "-", slug).strip("-")[:80]
    return f"{slug}.md"


def _is_github_file(url: str) -> bool:
    return "github.com" in url and "/blob/" in url


def _github_raw_url(url: str) -> str:
    return url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
