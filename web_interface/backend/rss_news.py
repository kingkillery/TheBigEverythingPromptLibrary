from __future__ import annotations

"""RSS / Atom AI news aggregator for PromptGarden website.

Pulls headlines from multiple public feeds (Google News, Flipboard magazines,
RSSHub proxies) and caches them in-memory to avoid hitting rate limits.

External dependency: `feedparser` (add to requirements). If unavailable, we
fallback to simple requests + regex title extraction.
"""

import time
import threading
from typing import List, Dict, Any
import feedparser
import requests

# --- Temporary compatibility shim for Python ≥ 3.13 ---------------------------
# The 'cgi' module was removed in Python 3.13. Some third-party libraries such
# as "feedparser" still import it (primarily for the simple helper
# `cgi.parse_header`).  To avoid crashing the application under the latest
# interpreter we create a minimal stub that implements *just* the functionality
# required by Feedparser.  Once feedparser removes its dependency we can delete
# this shim.

import sys
import types

if sys.version_info >= (3, 13) and 'cgi' not in sys.modules:  # pragma: no cover
    def _parse_header(value: str):  # type: ignore
        """Lightweight replacement for `cgi.parse_header`.

        Returns a tuple of `(main_value, params_dict)`.  This naive
        implementation is good enough for Feedparser's needs (parsing the
        HTTP `Content-Type` header).  It intentionally avoids pulling in heavy
        dependencies such as the standalone `email` package.
        """
        if not value:
            return "", {}
        parts = [p.strip() for p in value.split(';')]
        main_value = parts[0]
        params = {}
        for part in parts[1:]:
            if '=' in part:
                k, v = part.split('=', 1)
                params[k.strip().lower()] = v.strip().strip('"')
        return main_value, params

    cgi_stub = types.ModuleType('cgi')
    cgi_stub.parse_header = _parse_header  # type: ignore[attr-defined]
    sys.modules['cgi'] = cgi_stub

# -----------------------------------------------------------------------------

AI_FEEDS = {
    "google_news": "https://news.google.com/rss/search?q=artificial+intelligence&hl=en-US&gl=US&ceid=US:en",
    "flipboard_ai": "https://flipboard.com/topic/artificialintelligence.rss",
    # Bluesky via rsshub proxy (public, no auth)
    "bluesky_ai": "https://rsshub.app/bluesky/tag/ai",
    # Medium tag feed for AI-related stories
    "medium_ai": "https://medium.com/feed/tag/artificial-intelligence",
    # OpenAI official blog (Ghost platform)
    "openai_blog": "https://openai.com/blog/rss",
    # Anthropic blog (Ghost platform) – if 404s, simply ignored by parser
    "anthropic_blog": "https://www.anthropic.com/news/rss",
    # Hugging Face blog (Ghost platform)
    "huggingface_blog": "https://huggingface.co/blog/rss",
}

CACHE_TTL = 60 * 30  # 30 minutes

class _CacheEntry:
    def __init__(self, items: List[Dict[str, Any]], ts: float):
        self.items = items
        self.ts = ts

class RSSFetcher:
    """Thread-safe RSS fetcher with TTL caching."""
    def __init__(self):
        self._cache: Dict[str, _CacheEntry] = {}
        self._lock = threading.Lock()

    def _parse_feed(self, url: str) -> List[Dict[str, Any]]:
        """Parse a single RSS/Atom feed and return up-to-date entries sorted by
        publish date (most recent first). We restrict to 25 items per feed to
        keep things light.
        """
        entries: List[Dict[str, Any]] = []
        d = feedparser.parse(url)
        if d.bozo:  # if parser complains, try fetching raw text first
            try:
                resp = requests.get(url, timeout=10)
                d = feedparser.parse(resp.text)
            except Exception:
                return []

        for e in d.entries[:25]:
            published_ts = 0
            if "published_parsed" in e and e.published_parsed:
                try:
                    published_ts = time.mktime(e.published_parsed)
                except Exception:
                    published_ts = 0

            entries.append({
                "title": e.get("title", ""),
                "link": e.get("link", ""),
                "published": e.get("published", ""),
                "published_ts": published_ts,
                "source": url,
            })

        # newest first within each feed
        entries.sort(key=lambda x: x.get("published_ts", 0), reverse=True)
        return entries

    def get_ai_news(self) -> List[Dict[str, Any]]:
        """Return a *diverse* list of AI news items.

        We merge cached feeds but mix them round-robin so that no single
        publication dominates the results. This guarantees a spread of sources
        irrespective of which ones publish most frequently (e.g. Medium).
        """
        now = time.time()
        with self._lock:
            # Ensure cache is up-to-date
            for name, url in AI_FEEDS.items():
                entry = self._cache.get(name)
                if not entry or now - entry.ts > CACHE_TTL:
                    parsed = self._parse_feed(url)
                    self._cache[name] = _CacheEntry(parsed, now)

            # Create per-source queues (copy lists so we can pop without
            # mutating cache)
            feed_queues: Dict[str, List[Dict[str, Any]]] = {
                name: list(self._cache[name].items) for name in AI_FEEDS.keys()
            }

        # Round-robin combine to promote diversity
        combined: List[Dict[str, Any]] = []
        limit = 50
        while len(combined) < limit:
            made_progress = False
            for name in AI_FEEDS.keys():
                q = feed_queues.get(name, [])
                if q:
                    combined.append(q.pop(0))
                    made_progress = True
                    if len(combined) >= limit:
                        break
            if not made_progress:
                break  # all queues empty

        return combined

rss_fetcher = RSSFetcher() 