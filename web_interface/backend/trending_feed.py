from datetime import datetime, timedelta
from typing import List, Dict

try:
    from pygooglenews import GoogleNews
except ImportError as e:
    raise ImportError("pygooglenews is required for trending feed support") from e

_cache: Dict[str, any] = {"timestamp": None, "items": []}


def _fetch(limit: int = 10) -> List[Dict[str, str]]:
    """Fetch trending articles from Google News RSS"""
    gn = GoogleNews(lang='en', country='US')
    feed = gn.top_news()
    results = []
    for entry in feed.get('entries', [])[:limit]:
        results.append({
            'title': entry.get('title'),
            'link': entry.get('link'),
            'published': entry.get('published'),
            'source': entry.get('source', {}).get('title') if isinstance(entry.get('source'), dict) else None,
        })
    return results


def get_trending_feed(limit: int = 10) -> List[Dict[str, str]]:
    """Return cached trending feed, refreshing every 15 minutes"""
    now = datetime.utcnow()
    ts = _cache.get('timestamp')
    if not ts or now - ts > timedelta(minutes=15):
        _cache['items'] = _fetch(limit)
        _cache['timestamp'] = now
    return _cache['items'][:limit]
