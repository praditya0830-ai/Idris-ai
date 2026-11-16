"""Sourced lookups for dimensions the engineer did not state.

Principle: a number may only enter a design together with the URL and the sentence it came from.
No verified quote -> no number -> the engineer is asked, exactly as before.

The layer is off unless a search key is configured (BRAVE_SEARCH_API_KEY or TAVILY_API_KEY), so
nothing changes on a deployment that has not opted in.
"""
from __future__ import annotations

from .lookup import enrich, notes_for
from .search import configured as available

__all__ = ["available", "notes_for", "enrich"]
