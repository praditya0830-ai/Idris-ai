"""Stage 2: design references for a new product (v0.41, doc chat rekan engineer A8).

Arya, 21 Nov 17:21: a new product deserves the step Zookeeper does — look at real designs of the same thing before
drawing. The product here is a set of 3-6 cards an engineer can compare, built only from what the search provider
really returned:

  * `what_it_is` and `key_features` come from the snippet text of that source, never from the model;
  * every number is a CANDIDATE, labelled "web — not verified", and it may not enter any calculation until the
    engineer confirms it (the Studio's "Use this number" button is that confirmation);
  * the pattern summary counts what appears in the cards ("miter + TIG in 3 of 5 sources"), so it is a statement
    about the sources, not advice invented here.

Cost: no AI call at all — the cards are cut from the search snippets — so this step spends nothing from ai_budget.
Search results are cached per query for 24 h (IDRIS_REF_CACHE_S), at most three queries per reference stage. When the
provider is missing or answers 402, the reply says so plainly and the designer goes on with the solver library.
"""
from __future__ import annotations

import re
import time

from . import search as _search

CACHE_S = 24 * 3600
QUERY_MAX = 3
CARDS_MIN, CARDS_MAX = 3, 6
READ_MAX = 4
_CACHE = {}                      # query -> (stamp, rows); cleared by reset_cache()

_UNIT = re.compile(r"(?<![\w.])(\d+(?:[.,]\d+)?)\s*(mm|cm|m|kg|g|kW|W|Nm|N|MPa|GPa|deg|°|%|t|in|inch)\b", re.I)
_FEATURES = (
    ("miter cut", r"miter|mitre"), ("TIG weld", r"\btig\b"), ("MIG weld", r"\bmig\b"),
    ("bolted joint", r"\bbolt"), ("press-brake bend", r"press[- ]brake|bent|bend"),
    ("laser cut", r"laser"), ("cast", r"\bcast\b"), ("extruded profile", r"extrud"),
    ("sheet metal", r"sheet[- ]metal|sheet steel"), ("gusset", r"gusset"), ("rib", r"\brib"),
    ("tube frame", r"\btube\b"), ("chainstay", r"chainstay"), ("hydraulic", r"hydraulic"),
    ("standard part", r"standard (part|component)|off[- ]the[- ]shelf"),
)


def reset_cache() -> None:
    _CACHE.clear()


def _ttl() -> float:
    import os
    try:
        return max(60.0, float(os.environ.get("IDRIS_REF_CACHE_S", str(CACHE_S))))
    except ValueError:
        return CACHE_S


def queries_from(brief: dict, message: str = "") -> list:
    """At most three queries built from the brief: what it is, what it is made of, how it is made."""
    b = brief if isinstance(brief, dict) else {}
    obj = str(b.get("object") or "").strip()[:60]
    base = obj or str(message or "").strip()[:60]
    if not base:
        return []
    out = []
    material = str(b.get("material") or "").strip()[:30]
    purpose = str(b.get("purpose") or "").strip()[:40]
    process = str(b.get("process") or "").strip()[:30]
    out.append(_sp(" ".join(x for x in (material, purpose, base) if x)))
    if material and process:
        out.append(_sp(f"{material} {base} {process} construction"))
    elif material:
        out.append(_sp(f"{material} {base} fabrication"))
    if str(b.get("key_sizes") or "").strip():
        out.append(_sp(f"{base} dimensions key sizes"))
    if not out[1:]:
        out.append(_sp(f"{base} design details"))
    seen, keep = set(), []
    for q in out:
        if q and q.lower() not in seen and len(keep) < QUERY_MAX:
            seen.add(q.lower())
            keep.append(q)
    return keep


def _sp(s: str) -> str:
    return re.sub(r"\s{2,}", " ", str(s or "")).strip()[:120]


def _first_sentence(text: str, limit: int = 220) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if not text:
        return ""
    m = re.search(r"(?<=[.!?])\s", text)
    return (text[:m.start()] if m else text)[:limit]


def _features(card_text: str) -> list:
    low = card_text.lower()
    return [name for name, pattern in _FEATURES if re.search(pattern, low)][:6]


def _numbers(text: str, url: str) -> list:
    """Every number a source states, labelled a candidate: it may not enter a calculation before the engineer agrees."""
    out, seen = [], set()
    for value, unit in _UNIT.findall(str(text or "")):
        key = (value, unit.lower())
        if key in seen:
            continue
        seen.add(key)
        try:
            v = float(value.replace(",", "."))
        except ValueError:
            continue
        out.append({"value": v, "unit": unit, "source_url": url, "label": "web — not verified"})
    return out[:6]


def _cards(rows: list) -> list:
    cards, seen = [], set()
    for r in rows:
        url = str(r.get("url") or "")
        if not url.startswith("http") or url in seen:
            continue
        seen.add(url)                    # three queries return the same source more than once: one card
        snippet = str(r.get("snippet") or "")
        card = {"title": _first_sentence(r.get("title"), 120) or url,
                "source_url": url,
                "thumbnail_url": str(r.get("thumbnail") or "") or None,
                "what_it_is": _first_sentence(snippet, 220),
                "key_features": _features(f"{r.get('title') or ''} {snippet}"),
                "numbers_found": _numbers(f"{r.get('title') or ''} {snippet}", url)}
        if card["what_it_is"] or card["key_features"]:
            cards.append(card)
        if len(cards) >= CARDS_MAX:
            break
    return cards


def patterns_of(cards: list) -> list:
    """What the cards agree on: a count, never an opinion."""
    counts = {}
    for c in cards:
        for f in c.get("key_features") or []:
            counts[f] = counts.get(f, 0) + 1
    n = len(cards)
    return [f"{name} appears in {k} of {n} sources" for name, k in
            sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])) if k >= 2][:4]


def proposal_for(patterns: list, cards: list) -> str:
    if not cards:
        return ""
    if not patterns:
        return ("The sources do not agree on one construction, so I would compare two layouts before drawing. "
                "Nothing is designed yet.")
    return (f"I would start from the pattern the sources share: {patterns[0]}. "
            "Which one do you want to use as the base for the concept? (Patterns only — no branded product is copied, "
            "and no number from the web enters a calculation before you confirm it.)")


def _web_search(query: str, count: int) -> tuple:
    """-> (rows, error). An outage is reported as one honest line, never as an empty success."""
    if not _search.configured():
        return [], "web search unavailable: no search key on this server (TAVILY_API_KEY / BRAVE_SEARCH_API_KEY)"
    try:
        rows = _search.search(query, count)
    except Exception as e:
        return [], f"web search unavailable: {type(e).__name__}: {e}"[:200]
    if not rows:
        return [], "web search returned nothing for this query"
    return rows, None


def run(brief: dict, project_id=None, searcher=None, now: float | None = None) -> dict:
    """The reference stage: -> {ok, queries, cards, patterns, proposal, steps, mode, error}.

    `searcher(query, count) -> rows` is injectable so the tests never touch the network.
    """
    now = time.time() if now is None else now
    queries = queries_from(brief)
    if not queries:
        return {"ok": False, "queries": [], "cards": [], "patterns": [], "proposal": "", "steps": [],
                "mode": "none", "error": "the brief says nothing yet, so there is nothing to search for",
                "project_id": project_id}
    search_fn = searcher or _web_search
    steps, rows, errors, cached = [], [], [], 0
    for q in queries:
        hit = _CACHE.get(q.lower())
        if hit and now - hit[0] < _ttl():
            cached += 1
            steps.append(f"Cached: {q}")
            rows += hit[1]
            continue
        steps.append(f"Searching: {q}")
        got, err = search_fn(q, CARDS_MAX)
        if err:
            errors.append(err)
            steps.append(f"No result: {err}")
            continue
        steps.append(f"Found {len(got)}, reading {min(len(got), READ_MAX)}")
        rows += got[:READ_MAX]
        _CACHE[q.lower()] = (now, got[:READ_MAX])
    cards = _cards(rows)
    patterns = patterns_of(cards)
    err = None
    if not cards:
        err = errors[0] if errors else "no usable reference was found"
    elif errors and len(cards) < CARDS_MIN:
        err = errors[0]
    return {"ok": bool(cards), "queries": queries, "cards": cards, "patterns": patterns,
            "proposal": proposal_for(patterns, cards), "steps": steps,
            "mode": "web" if cards else "none", "error": err, "cached_queries": cached,
            "project_id": project_id,
            "note": ("Numbers from the web are candidates, labelled \"web — not verified\"; they enter no "
                     "calculation until the engineer confirms them. Cards are cut from the search snippets, so this "
                     "step spends nothing from ai_budget.")}
