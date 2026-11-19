"""Web search for dimensions nobody stated. Standard library only.

Keys (Render -> Environment): BRAVE_SEARCH_API_KEY or TAVILY_API_KEY.
Without either key `configured()` is False and nothing is searched.
IDRIS_SEARCH_URL overrides the endpoint (the tests point it at a local mock).
"""
from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request


def configured() -> bool:
    return bool(os.environ.get("BRAVE_SEARCH_API_KEY") or os.environ.get("TAVILY_API_KEY"))


def provider() -> str | None:
    """Which provider the configured key selects: 'brave', 'tavily' or None. The name only, never the key."""
    if os.environ.get("BRAVE_SEARCH_API_KEY"):
        return "brave"
    if os.environ.get("TAVILY_API_KEY"):
        return "tavily"
    return None


def search(query: str, count: int = 5, timeout: float = 12.0) -> list[dict]:
    """-> [{title, url, snippet}]. Empty when nothing is configured, or nothing was found, or the call failed."""
    query = (query or "").strip()
    if not query or not configured():
        return []
    try:
        out = _brave(query, count, timeout) if os.environ.get("BRAVE_SEARCH_API_KEY") else _tavily(query, count, timeout)
    except Exception:
        return []                       # a search outage must never break a design request
    return [r for r in out if str(r.get("url", "")).startswith("http")][:count]


def _json(url: str, headers: dict, data: bytes | None = None, timeout: float = 12.0):
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def _brave(query: str, count: int, timeout: float) -> list[dict]:
    base = os.environ.get("IDRIS_SEARCH_URL") or "https://api.search.brave.com/res/v1/web/search"
    url = base + ("&" if "?" in base else "?") + urllib.parse.urlencode({"q": query, "count": count})
    d = _json(url, {"Accept": "application/json",
                    "X-Subscription-Token": os.environ["BRAVE_SEARCH_API_KEY"]}, timeout=timeout)
    rows = ((d.get("web") or {}).get("results")) or []
    return [{"title": str(r.get("title") or "")[:200], "url": str(r.get("url") or ""),
             "snippet": str(r.get("description") or "")[:600]} for r in rows]


def _tavily(query: str, count: int, timeout: float) -> list[dict]:
    url = os.environ.get("IDRIS_SEARCH_URL") or "https://api.tavily.com/search"
    body = json.dumps({"api_key": os.environ["TAVILY_API_KEY"], "query": query,
                       "max_results": count, "search_depth": "basic"}).encode()
    d = _json(url, {"Content-Type": "application/json"}, body, timeout=timeout)
    return [{"title": str(r.get("title") or "")[:200], "url": str(r.get("url") or ""),
             "snippet": str(r.get("content") or "")[:600]} for r in (d.get("results") or [])]
