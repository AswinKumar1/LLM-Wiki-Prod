"""
All prompt templates for wiki operations.

Keeping prompts here (not scattered in operation files) makes them
easy to audit, swap out, or fine-tune for specific models.

Each function returns (system_prompt, user_prompt) tuples.
"""

from __future__ import annotations

from pathlib import Path


# ------------------------------------------------------------------
# Ingest prompts
# ------------------------------------------------------------------

def ingest_system() -> str:
    return """You are a disciplined knowledge base compiler. Your job is to read
source documents and produce structured, interlinked markdown wiki pages.

Rules you always follow:
- Write every page in clean markdown with YAML frontmatter
- Use [[wikilinks]] for all cross-references to other wiki pages
- Every factual claim must trace to the source document you just read
- Be specific and concrete — avoid vague summaries
- Prefer updating existing pages over creating duplicate pages
- Keep confidence levels honest: high/medium/low"""


def ingest_source(source_text: str, source_name: str, index_content: str) -> tuple[str, str]:
    system = ingest_system()
    user = f"""You are ingesting a new source document into the wiki.

## Source document: {source_name}

{source_text}

---

## Current wiki index (for context on what already exists):

{index_content or "(wiki is empty — this is the first source)"}

---

## Your task:

1. Summarize the key takeaways from this source in 3-5 bullet points
2. Create a source summary page with this exact frontmatter:
   ```
   ---
   title: {source_name}
   type: source-summary
   source: raw/{source_name}
   created: TODAY
   updated: TODAY
   confidence: high
   ---
   ```
3. List every existing wiki page that should be updated based on this source
4. For each page to update: provide the full updated markdown content
5. List any NEW concept or entity pages that should be created
6. For each new page: provide the full markdown content with correct frontmatter
7. Update wiki/index.md — provide the full updated index
8. Provide a one-line log entry for wiki/log.md in this format:
   ## [TODAY] ingest | {source_name} | N pages updated, M pages created

Respond in this exact structure:
### TAKEAWAYS
<bullet points>

### SOURCE_SUMMARY_PAGE
<full markdown for wiki/sources/{source_name}.md>

### PAGES_TO_UPDATE
<list of existing page paths>

### UPDATED_PAGES
<path: wiki/...>
<full markdown>
<path: wiki/...>
<full markdown>

### NEW_PAGES
<path: wiki/...>
<full markdown>

### INDEX_UPDATE
<full updated wiki/index.md>

### LOG_ENTRY
<single line>
"""
    return system, user


# ------------------------------------------------------------------
# Query prompts
# ------------------------------------------------------------------

def query_system() -> str:
    return """You are a precise knowledge base assistant. You answer questions
by reading the wiki pages provided to you. You:
- Cite sources using [[wikilinks]] and raw/ file paths
- Acknowledge when information is incomplete or uncertain
- Never fabricate information not present in the wiki pages
- Suggest follow-up questions when relevant"""


def query_question(question: str, relevant_pages: dict[str, str]) -> tuple[str, str]:
    system = query_system()
    pages_text = "\n\n---\n\n".join(
        f"## {path}\n\n{content}" for path, content in relevant_pages.items()
    )
    user = f"""Question: {question}

## Relevant wiki pages:

{pages_text}

Answer the question based only on these pages. Use [[wikilinks]] for references.
If the answer requires information not present here, say so clearly."""
    return system, user


def query_find_relevant_pages(question: str, index_content: str) -> tuple[str, str]:
    system = "You are a search assistant for a markdown wiki."
    user = f"""Given this question:
"{question}"

And this wiki index:
{index_content}

List the 3-7 most relevant wiki page paths to read to answer this question.
Return ONLY a JSON array of paths, e.g.:
["wiki/concepts/rag.md", "wiki/comparisons/rag-vs-llm-wiki.md"]
No other text."""
    return system, user


# ------------------------------------------------------------------
# Lint prompts
# ------------------------------------------------------------------

def lint_system() -> str:
    return """You are a wiki health inspector. You scan wiki pages for quality
issues and report them precisely and actionably."""


def lint_scan(all_pages: dict[str, str]) -> tuple[str, str]:
    system = lint_system()
    pages_text = "\n\n---\n\n".join(
        f"## {path}\n\n{content}" for path, content in all_pages.items()
    )
    user = f"""Scan these wiki pages for the following issues:

1. **Contradictions** — claims that conflict between pages
2. **Orphan pages** — pages with no [[wikilinks]] pointing to them
3. **Missing pages** — concepts referenced with [[wikilinks]] that have no page
4. **Stale frontmatter** — pages with confidence: low that haven't been updated
5. **Broken structure** — pages missing required frontmatter fields

## Wiki pages:

{pages_text}

Report findings in this exact structure:

### CONTRADICTIONS
<list: page_a vs page_b — description of conflict>

### ORPHAN_PAGES
<list of page paths with no incoming links>

### MISSING_PAGES
<list of [[wikilinks]] that have no corresponding page file>

### LOW_CONFIDENCE_PAGES
<list of pages flagged for review>

### STRUCTURAL_ISSUES
<list of pages missing frontmatter fields>

### SUMMARY
<one paragraph overall health assessment>
"""
    return system, user


# ------------------------------------------------------------------
# Init prompt — generates the AGENTS.md schema for a new wiki
# ------------------------------------------------------------------

def generate_agents_md(topic: str, domain_notes: str = "") -> tuple[str, str]:
    system = "You are helping set up a new LLM wiki knowledge base."
    user = f"""Generate an AGENTS.md schema file for a wiki about: {topic}

{f"Additional context: {domain_notes}" if domain_notes else ""}

The schema should define:
- Wiki purpose and scope
- Page types and their frontmatter conventions
- Naming conventions (kebab-case filenames, etc.)
- The three workflow descriptions: Ingest, Query, Lint
- Domain-specific conventions if relevant

Use this as a template but customise for the topic:
- Replace [Your Topic] with the actual topic
- Add domain-specific page types if needed
- Keep it concise (under 300 lines)

Start the file with:
# [Topic] Wiki — Agent Schema
# Provider-agnostic. Works with Ollama, OpenAI, Anthropic, or any compatible LLM.
"""
    return system, user
