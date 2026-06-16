"""
Wiki HTML exporter.

Renders all wiki pages into a single self-contained HTML file:
  - Sidebar navigation (grouped by type: concepts, entities, sources, etc.)
  - [[wikilinks]] → clickable anchor links
  - Markdown → HTML (headings, bold, italic, code, lists, tables)
  - Dark/light mode via CSS variables
  - Zero JavaScript dependencies

Usage:
    exporter = WikiExporter(wiki_fs)
    html = exporter.render()
    out_path = exporter.save()   # saves to outputs/wiki-export-YYYY-MM-DD.html
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import Optional

from ..wiki_fs import WikiFS


class WikiExporter:
    """
    Export all wiki pages to a single self-contained HTML file.

    Usage:
        exporter = WikiExporter(wiki_fs)
        path = exporter.save()
    """

    def __init__(self, wiki_fs: WikiFS):
        self.fs = wiki_fs

    def save(self) -> Path:
        """Render and write to outputs/wiki-export-YYYY-MM-DD.html."""
        html    = self.render()
        filename= f"wiki-export-{date.today()}.html"
        out     = self.fs.write_output(filename, html)
        return out

    def render(self) -> str:
        """Render the entire wiki to HTML string."""
        pages  = self._load_pages()
        if not pages:
            return "<html><body><p>No wiki pages found.</p></body></html>"

        nav_html   = self._render_nav(pages)
        body_html  = self._render_body(pages)
        title      = self._wiki_title()

        return _HTML_TEMPLATE.format(
            title=title,
            nav=nav_html,
            body=body_html,
            export_date=str(date.today()),
            page_count=len(pages),
        )

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def _render_nav(self, pages: list[dict]) -> str:
        groups: dict[str, list[dict]] = {}
        for p in pages:
            g = p["group"]
            groups.setdefault(g, []).append(p)

        parts = []
        for group in ["concepts", "entities", "sources", "comparisons", "queries", "other"]:
            if group not in groups:
                continue
            parts.append(f'<div class="nav-group">')
            parts.append(f'<div class="nav-group-title">{group.title()}</div>')
            for p in sorted(groups[group], key=lambda x: x["title"]):
                slug = p["slug"]
                title = p["title"]
                parts.append(
                    f'<a class="nav-link" href="#{slug}">{title}</a>'
                )
            parts.append("</div>")

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Body
    # ------------------------------------------------------------------

    def _render_body(self, pages: list[dict]) -> str:
        parts = []
        # Build slug→title map for wikilink resolution
        slug_map = {p["slug"]: p["title"] for p in pages}
        title_map = {p["title"].lower(): p["slug"] for p in pages}

        for p in pages:
            slug    = p["slug"]
            title   = p["title"]
            content = p["content"]
            meta    = p["meta"]

            html_content = _md_to_html(content, slug_map, title_map)

            meta_html = ""
            if meta:
                meta_items = " · ".join(
                    f'<span class="meta-item"><span class="meta-key">{k}</span> {v}</span>'
                    for k, v in meta.items()
                    if k not in {"title"}
                )
                if meta_items:
                    meta_html = f'<div class="page-meta">{meta_items}</div>'

            parts.append(
                f'<article id="{slug}" class="wiki-page">\n'
                f'  <h1 class="page-title">{title}</h1>\n'
                f'  {meta_html}\n'
                f'  <div class="page-content">{html_content}</div>\n'
                f'</article>\n'
            )

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Page loading
    # ------------------------------------------------------------------

    def _load_pages(self) -> list[dict]:
        pages = []
        for path in self.fs.list_wiki_pages():
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue

            title, meta, body = _parse_frontmatter(text)
            rel   = self.fs.relative_to_root(path)
            slug  = _path_to_slug(rel)
            group = _path_to_group(rel)

            if not title:
                # Fall back to first H1 or filename
                h1 = re.search(r"^#\s+(.+)$", body, re.MULTILINE)
                title = h1.group(1) if h1 else path.stem.replace("-", " ").title()

            pages.append({
                "path":    rel,
                "slug":    slug,
                "title":   title,
                "group":   group,
                "meta":    meta,
                "content": body,
            })

        return pages

    def _wiki_title(self) -> str:
        agents = self.fs.agents_md_path
        if agents.exists():
            first_line = agents.read_text().splitlines()[0]
            return re.sub(r"^#\s*", "", first_line).strip() or "Wiki"
        return "LLM Wiki"


# ---------------------------------------------------------------------------
# Markdown → HTML converter (stdlib only)
# ---------------------------------------------------------------------------

def _md_to_html(md: str, slug_map: dict, title_map: dict) -> str:
    """
    Convert a subset of Markdown to HTML.
    Handles: headings, bold, italic, inline code, code blocks,
             unordered/ordered lists, tables, paragraphs, [[wikilinks]], hrules.
    """
    lines   = md.splitlines()
    output  = []
    i       = 0
    in_list = None   # "ul" or "ol"
    in_table= False

    def close_list():
        nonlocal in_list
        if in_list:
            output.append(f"</{in_list}>")
            in_list = None

    def close_table():
        nonlocal in_table
        if in_table:
            output.append("</tbody></table>")
            in_table = False

    while i < len(lines):
        line = lines[i]

        # Code block
        if line.strip().startswith("```"):
            close_list(); close_table()
            lang = line.strip()[3:].strip()
            code_lines = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            code = "\n".join(code_lines)
            output.append(
                f'<pre><code class="language-{lang}">{_esc(code)}</code></pre>'
            )
            i += 1
            continue

        # Horizontal rule
        if re.match(r"^---+$|^\*\*\*+$", line.strip()):
            close_list(); close_table()
            output.append("<hr>")
            i += 1
            continue

        # Table
        if "|" in line and i + 1 < len(lines) and re.match(r"[\|\-\s]+", lines[i + 1]):
            close_list(); close_table()
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            output.append('<table><thead><tr>')
            for c in cells:
                output.append(f"<th>{_inline(c, slug_map, title_map)}</th>")
            output.append("</tr></thead><tbody>")
            in_table = True
            i += 2   # skip separator row
            continue

        if in_table and "|" in line:
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            output.append("<tr>")
            for c in cells:
                output.append(f"<td>{_inline(c, slug_map, title_map)}</td>")
            output.append("</tr>")
            i += 1
            continue

        close_table()

        # Headings
        h = re.match(r"^(#{1,6})\s+(.+)$", line)
        if h:
            close_list()
            level = len(h.group(1))
            text  = _inline(h.group(2), slug_map, title_map)
            slug  = re.sub(r"[^a-z0-9-]", "-", h.group(2).lower()).strip("-")
            output.append(f'<h{level} id="{slug}">{text}</h{level}>')
            i += 1
            continue

        # Unordered list
        if re.match(r"^[-*+]\s+", line):
            if in_list != "ul":
                close_list()
                output.append("<ul>")
                in_list = "ul"
            text = _inline(re.sub(r"^[-*+]\s+", "", line), slug_map, title_map)
            output.append(f"<li>{text}</li>")
            i += 1
            continue

        # Ordered list
        if re.match(r"^\d+\.\s+", line):
            if in_list != "ol":
                close_list()
                output.append("<ol>")
                in_list = "ol"
            text = _inline(re.sub(r"^\d+\.\s+", "", line), slug_map, title_map)
            output.append(f"<li>{text}</li>")
            i += 1
            continue

        close_list()

        # Blank line
        if not line.strip():
            i += 1
            continue

        # Paragraph
        output.append(f"<p>{_inline(line, slug_map, title_map)}</p>")
        i += 1

    close_list()
    close_table()
    return "\n".join(output)


def _inline(text: str, slug_map: dict, title_map: dict) -> str:
    """Apply inline formatting: wikilinks, bold, italic, code, links."""
    # [[wikilinks]]
    def wikilink(m):
        target = m.group(1)
        slug   = title_map.get(target.lower())
        if not slug:
            slug = re.sub(r"[^a-z0-9-]", "-", target.lower()).strip("-")
        return f'<a class="wikilink" href="#{slug}">{target}</a>'
    text = re.sub(r"\[\[([^\]]+)\]\]", wikilink, text)

    # Markdown links [text](url)
    text = re.sub(r"\[([^\]]+)\]\((https?://[^\)]+)\)",
                  r'<a href="\2" target="_blank" rel="noopener">\1</a>', text)

    # Inline code
    text = re.sub(r"`([^`]+)`", lambda m: f"<code>{_esc(m.group(1))}</code>", text)

    # Bold **text** or __text__
    text = re.sub(r"\*\*(.+?)\*\*|__(.+?)__",
                  lambda m: f"<strong>{m.group(1) or m.group(2)}</strong>", text)

    # Italic *text* or _text_
    text = re.sub(r"\*(.+?)\*|_(.+?)_",
                  lambda m: f"<em>{m.group(1) or m.group(2)}</em>", text)

    return text


def _esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ---------------------------------------------------------------------------
# Frontmatter parser
# ---------------------------------------------------------------------------

def _parse_frontmatter(text: str) -> tuple[str, dict, str]:
    """Return (title, meta_dict, body)."""
    title = ""
    meta  = {}
    body  = text

    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            fm_text = text[3:end]
            body    = text[end + 4:].strip()
            for line in fm_text.splitlines():
                if ":" in line:
                    k, _, v = line.partition(":")
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    meta[k] = v
                    if k == "title":
                        title = v

    return title, meta, body


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _path_to_slug(rel_path: str) -> str:
    return re.sub(r"[^a-z0-9-]", "-", rel_path.lower().replace("/", "-")).strip("-")


def _path_to_group(rel_path: str) -> str:
    parts = Path(rel_path).parts
    if len(parts) >= 2:
        return parts[-2]   # e.g. "concepts", "entities", "sources"
    return "other"


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
  :root {{
    --bg: #ffffff; --fg: #1a1a1a; --border: #e5e7eb;
    --accent: #2563eb; --accent-hover: #1d4ed8;
    --code-bg: #f3f4f6; --nav-bg: #f9fafb;
    --meta-fg: #6b7280; --sidebar-w: 240px;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      --bg: #0f1117; --fg: #e5e7eb; --border: #374151;
      --accent: #60a5fa; --accent-hover: #93c5fd;
      --code-bg: #1f2937; --nav-bg: #111827; --meta-fg: #9ca3af;
    }}
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    background: var(--bg); color: var(--fg); display: flex; min-height: 100vh; }}
  nav {{ width: var(--sidebar-w); min-width: var(--sidebar-w); background: var(--nav-bg);
    border-right: 1px solid var(--border); padding: 24px 0; position: sticky;
    top: 0; height: 100vh; overflow-y: auto; }}
  .nav-header {{ padding: 0 16px 16px; font-weight: 700; font-size: 14px;
    color: var(--fg); border-bottom: 1px solid var(--border); margin-bottom: 8px; }}
  .nav-group {{ margin-bottom: 8px; }}
  .nav-group-title {{ padding: 6px 16px; font-size: 11px; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.08em; color: var(--meta-fg); }}
  .nav-link {{ display: block; padding: 4px 16px; font-size: 13px; color: var(--fg);
    text-decoration: none; white-space: nowrap; overflow: hidden;
    text-overflow: ellipsis; transition: color 0.15s; }}
  .nav-link:hover {{ color: var(--accent); }}
  main {{ flex: 1; padding: 40px 48px; max-width: 860px; }}
  .wiki-page {{ margin-bottom: 64px; padding-bottom: 48px;
    border-bottom: 1px solid var(--border); }}
  .wiki-page:last-child {{ border-bottom: none; }}
  .page-title {{ font-size: 28px; font-weight: 700; margin-bottom: 8px; }}
  .page-meta {{ font-size: 12px; color: var(--meta-fg); margin-bottom: 20px;
    display: flex; flex-wrap: wrap; gap: 12px; }}
  .meta-key {{ font-weight: 600; }}
  .page-content {{ line-height: 1.7; }}
  .page-content h1,.page-content h2,.page-content h3,
  .page-content h4,.page-content h5,.page-content h6 {{
    margin: 24px 0 12px; font-weight: 600; }}
  .page-content h2 {{ font-size: 20px; }}
  .page-content h3 {{ font-size: 17px; }}
  .page-content p {{ margin-bottom: 14px; }}
  .page-content ul, .page-content ol {{ margin: 0 0 14px 24px; }}
  .page-content li {{ margin-bottom: 4px; }}
  .page-content pre {{ background: var(--code-bg); border: 1px solid var(--border);
    border-radius: 6px; padding: 16px; overflow-x: auto; margin: 16px 0; }}
  .page-content code {{ background: var(--code-bg); padding: 2px 5px;
    border-radius: 3px; font-size: 0.875em; font-family: "SF Mono", Consolas, monospace; }}
  .page-content pre code {{ background: none; padding: 0; }}
  .page-content table {{ border-collapse: collapse; width: 100%; margin: 16px 0; font-size: 14px; }}
  .page-content th, .page-content td {{ border: 1px solid var(--border);
    padding: 8px 12px; text-align: left; }}
  .page-content th {{ background: var(--code-bg); font-weight: 600; }}
  .page-content hr {{ border: none; border-top: 1px solid var(--border); margin: 24px 0; }}
  a.wikilink {{ color: var(--accent); text-decoration: none; }}
  a.wikilink:hover {{ text-decoration: underline; color: var(--accent-hover); }}
  .footer {{ margin-top: 64px; padding-top: 24px; border-top: 1px solid var(--border);
    font-size: 12px; color: var(--meta-fg); }}
  @media (max-width: 768px) {{
    nav {{ display: none; }}
    main {{ padding: 24px 20px; }}
  }}
</style>
</head>
<body>
<nav>
  <div class="nav-header">{title}</div>
  {nav}
</nav>
<main>
  {body}
  <div class="footer">
    Exported {export_date} · {page_count} pages ·
    <a href="https://github.com/AswinKumar1/LLM-Wiki-Prod">llm-wiki-universal</a>
  </div>
</main>
</body>
</html>"""
