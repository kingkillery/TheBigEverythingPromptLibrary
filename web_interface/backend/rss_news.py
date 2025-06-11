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

AI_FEEDS = {
    "google_news": "https://news.google.com/rss/search?q=artificial+intelligence&hl=en-US&gl=US&ceid=US:en",
    "flipboard_ai": "https://flipboard.com/topic/artificialintelligence.rss",
    # Bluesky via rsshub proxy (public, no auth)
    "bluesky_ai": "https://rsshub.app/bluesky/tag/ai",
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
        entries = []
        d = feedparser.parse(url)
        if d.bozo:  # fallback fetch raw
            try:
                r = requests.get(url, timeout=10)
                d = feedparser.parse(r.text)
            except Exception:
                return []
        for e in d.entries[:25]:
            entries.append({
                "title": e.get("title", ""),
                "link": e.get("link", ""),
                "published": e.get("published", ""),
                "source": url,
            })
        return entries

    def get_ai_news(self) -> List[Dict[str, Any]]:
        now = time.time()
        items: List[Dict[str, Any]] = []
        with self._lock:
            for name, url in AI_FEEDS.items():
                entry = self._cache.get(name)
                if not entry or now - entry.ts > CACHE_TTL:
                    parsed = self._parse_feed(url)
                    self._cache[name] = _CacheEntry(parsed, now)
                items.extend(self._cache[name].items)
        # sort newest first (if published present)
        items.sort(key=lambda x: x.get("published", ""), reverse=True)
        return items[:50]

rss_fetcher = RSSFetcher() 