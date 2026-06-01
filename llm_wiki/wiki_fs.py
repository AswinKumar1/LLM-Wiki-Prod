"""
WikiFS — filesystem abstraction for the wiki directory structure.

All file I/O goes through this class so operations stay testable
and the path conventions are in one place.
"""

from __future__ import annotations

import hashlib
import re
from datetime import date
from pathlib import Path
from typing import Optional


class WikiFS:
    """Read/write interface to a wiki root directory."""

    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.raw_dir = self.root / "raw"
        self.wiki_dir = self.root / "wiki"
        self.outputs_dir = self.root / "outputs"
        self.index_path = self.wiki_dir / "index.md"
        self.log_path = self.wiki_dir / "log.md"
        self.agents_md_path = self.root / "AGENTS.md"
        self.config_path = self.root / "config.yaml"

    # ------------------------------------------------------------------
    # Directory structure
    # ------------------------------------------------------------------

    def ensure_structure(self) -> None:
        """Create the standard directory tree if it doesn't exist."""
        dirs = [
            self.raw_dir / "articles",
            self.raw_dir / "papers",
            self.raw_dir / "repos",
            self.raw_dir / "data",
            self.raw_dir / "images",
            self.wiki_dir / "concepts",
            self.wiki_dir / "entities",
            self.wiki_dir / "sources",
            self.wiki_dir / "comparisons",
            self.outputs_dir,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Raw sources
    # ------------------------------------------------------------------

    def list_raw_sources(self) -> list[Path]:
        """Return all files under raw/ (excluding subdirs)."""
        if not self.raw_dir.exists():
            return []
        return sorted(
            p for p in self.raw_dir.rglob("*")
            if p.is_file() and not p.name.startswith(".")
        )

    def new_raw_sources(self) -> list[Path]:
        """Return raw sources that don't yet have a wiki/sources/ summary."""
        sources_dir = self.wiki_dir / "sources"
        existing_slugs = {p.stem for p in sources_dir.glob("*.md")} if sources_dir.exists() else set()
        return [
            p for p in self.list_raw_sources()
            if _slugify(p.stem) not in existing_slugs
        ]

    def read_source(self, path: Path) -> str:
        """Read a raw source file. Returns text for text files."""
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            return f"[Could not read {path.name}: {exc}]"

    def source_sha256(self, path: Path) -> str:
        """Return SHA-256 hex digest of a raw source file."""
        h = hashlib.sha256()
        h.update(path.read_bytes())
        return h.hexdigest()

    # ------------------------------------------------------------------
    # Wiki pages
    # ------------------------------------------------------------------

    def list_wiki_pages(self) -> list[Path]:
        if not self.wiki_dir.exists():
            return []
        return sorted(
            p for p in self.wiki_dir.rglob("*.md")
            if p.is_file()
            and p.name not in {"index.md", "log.md"}
        )

    def read_wiki_page(self, rel_path: str) -> Optional[str]:
        full = self.root / rel_path
        if full.exists():
            return full.read_text(encoding="utf-8")
        return None

    def write_wiki_page(self, rel_path: str, content: str) -> None:
        full = self.root / rel_path
        full.parent.mkdir(parents=True, exist_ok=True)
        # Inject today's date into any TODAY placeholders
        content = content.replace("TODAY", str(date.today()))
        full.write_text(content, encoding="utf-8")

    # ------------------------------------------------------------------
    # Index
    # ------------------------------------------------------------------

    def read_index(self) -> str:
        if self.index_path.exists():
            return self.index_path.read_text(encoding="utf-8")
        return ""

    def write_index(self, content: str) -> None:
        self.wiki_dir.mkdir(parents=True, exist_ok=True)
        self.index_path.write_text(content.replace("TODAY", str(date.today())), encoding="utf-8")

    def init_index(self) -> None:
        if not self.index_path.exists():
            self.write_index(
                "# Wiki Index\n\n"
                "_Updated automatically on every ingest._\n\n"
                "## Concepts\n\n"
                "## Entities\n\n"
                "## Sources\n\n"
                "## Comparisons\n\n"
            )

    # ------------------------------------------------------------------
    # Log
    # ------------------------------------------------------------------

    def append_log(self, entry: str) -> None:
        self.wiki_dir.mkdir(parents=True, exist_ok=True)
        entry = entry.replace("TODAY", str(date.today()))
        if not entry.startswith("## "):
            entry = f"## [{date.today()}] {entry}"
        if self.log_path.exists():
            existing = self.log_path.read_text(encoding="utf-8")
            self.log_path.write_text(entry + "\n\n" + existing, encoding="utf-8")
        else:
            self.log_path.write_text("# Wiki Log\n\n" + entry + "\n", encoding="utf-8")

    def read_log(self, last_n: int = 10) -> str:
        if not self.log_path.exists():
            return ""
        lines = self.log_path.read_text(encoding="utf-8").splitlines()
        entries = [l for l in lines if l.startswith("## [")]
        return "\n".join(entries[:last_n])

    # ------------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------------

    def write_output(self, filename: str, content: str) -> Path:
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        out = self.outputs_dir / filename
        out.write_text(content, encoding="utf-8")
        return out

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def relative_to_root(self, path: Path) -> str:
        return str(path.relative_to(self.root))

    def all_wikilinks(self) -> set[str]:
        """Collect all [[wikilink]] targets across all wiki pages."""
        links: set[str] = set()
        pattern = re.compile(r"\[\[([^\]]+)\]\]")
        for page in self.list_wiki_pages():
            text = page.read_text(encoding="utf-8", errors="replace")
            for match in pattern.finditer(text):
                links.add(match.group(1))
        return links


def _slugify(name: str) -> str:
    """Convert a filename stem to a wiki slug."""
    return re.sub(r"[^a-z0-9-]", "-", name.lower()).strip("-")
