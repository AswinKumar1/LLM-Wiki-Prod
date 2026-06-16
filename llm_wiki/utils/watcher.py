"""
Watch mode for llm-wiki-universal.

Polls raw/ on a configurable interval and ingests any new files
that appear. Ctrl+C stops cleanly.

Usage:
    watcher = WikiWatcher(provider, wiki_fs, interval=10)
    watcher.run()   # blocks until Ctrl+C

Or via CLI:
    wiki ingest --watch
    wiki ingest --watch --interval 30
    wiki ingest --watch --interval 5 --provider ollama
"""

from __future__ import annotations

import signal
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable

from ..providers.base import LLMProvider
from ..wiki_fs import WikiFS


@dataclass
class WatchEvent:
    path:       Path
    timestamp:  str
    success:    bool
    pages_created: list[str] = field(default_factory=list)
    pages_updated: list[str] = field(default_factory=list)
    tokens_used:   int = 0
    error:         Optional[str] = None


class WikiWatcher:
    """
    File system watcher that auto-ingests new files dropped into raw/.

    Poll-based (no inotify/FSEvents dependency) so it works everywhere
    including GitHub Codespaces, Docker containers, and remote filesystems.

    Args:
        provider:   LLM provider to use for ingest
        wiki_fs:    WikiFS instance pointing at the wiki root
        interval:   Poll interval in seconds (default: 10)
        verbose:    Print detailed output per file
        on_event:   Optional callback(WatchEvent) after each ingest
    """

    def __init__(
        self,
        provider:   LLMProvider,
        wiki_fs:    WikiFS,
        interval:   int = 10,
        verbose:    bool = False,
        on_event:   Optional[Callable[[WatchEvent], None]] = None,
    ):
        self.provider  = provider
        self.fs        = wiki_fs
        self.interval  = interval
        self.verbose   = verbose
        self.on_event  = on_event
        self._running  = False
        self._seen:    set[str] = set()   # paths already processed

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> None:
        """
        Start watching. Blocks until Ctrl+C or stop() is called.
        Registers SIGINT handler for clean shutdown.
        """
        self._running = True
        self._seen    = self._current_processed_set()

        _print_info(
            f"Watching {self.fs.raw_dir} "
            f"(polling every {self.interval}s) — Ctrl+C to stop"
        )

        # Register clean shutdown on Ctrl+C
        original_sigint = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._handle_sigint)

        try:
            while self._running:
                self._poll()
                for _ in range(self.interval):
                    if not self._running:
                        break
                    time.sleep(1)
        finally:
            signal.signal(signal.SIGINT, original_sigint)
            _print_info("Watch stopped.")

    def stop(self) -> None:
        """Stop the watch loop (useful in tests)."""
        self._running = False

    def run_once(self) -> list[WatchEvent]:
        """
        Check for new files once and ingest them.
        Returns a list of WatchEvents. Useful for testing.
        """
        self._seen = self._current_processed_set()
        return self._poll()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _poll(self) -> list[WatchEvent]:
        """Check for new files and ingest any found."""
        from ..operations.ingest import IngestOperation

        new_files = self._find_new_files()
        events    = []

        for path in new_files:
            ts = datetime.now().strftime("%H:%M:%S")
            _print_info(f"[{ts}] New file detected: {path.name}")

            op     = IngestOperation(self.provider, self.fs, verbose=self.verbose)
            result = op.run_one(path)

            event = WatchEvent(
                path=path,
                timestamp=ts,
                success=result.success,
                pages_created=result.pages_created,
                pages_updated=result.pages_updated,
                tokens_used=result.tokens_used,
                error=result.error,
            )

            if result.success:
                _print_ok(
                    f"[{ts}] ✓ {path.name}: "
                    f"{len(result.pages_created)} created, "
                    f"{len(result.pages_updated)} updated "
                    f"({result.tokens_used} tokens)"
                )
            else:
                _print_err(f"[{ts}] ✗ {path.name}: {result.error}")

            # Mark as seen regardless of success (avoid infinite retry loops)
            self._seen.add(str(path))
            events.append(event)

            if self.on_event:
                try:
                    self.on_event(event)
                except Exception:
                    pass

        return events

    def _find_new_files(self) -> list[Path]:
        """Return files in raw/ that haven't been processed yet."""
        all_raw = self.fs.list_raw_sources()
        return [
            p for p in all_raw
            if str(p) not in self._seen
        ]

    def _current_processed_set(self) -> set[str]:
        """
        Build initial set of already-processed paths.
        A file is considered processed if its slug exists in wiki/sources/.
        """
        import re
        sources_dir = self.fs.wiki_dir / "sources"
        existing_slugs = (
            {p.stem for p in sources_dir.glob("*.md")}
            if sources_dir.exists() else set()
        )
        processed: set[str] = set()
        for raw_path in self.fs.list_raw_sources():
            slug = re.sub(r"[^a-z0-9-]", "-", raw_path.stem.lower()).strip("-")
            if slug in existing_slugs:
                processed.add(str(raw_path))
        return processed

    def _handle_sigint(self, signum, frame) -> None:
        print()   # newline after ^C
        _print_info("Stopping watcher...")
        self._running = False


# ---------------------------------------------------------------------------
# Output helpers (local to avoid circular import)
# ---------------------------------------------------------------------------

def _print_ok(msg):   print(f"\033[32m✓\033[0m {msg}")
def _print_err(msg):  print(f"\033[31m✗\033[0m {msg}")
def _print_info(msg): print(f"\033[34mℹ\033[0m {msg}")
