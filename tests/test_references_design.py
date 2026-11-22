"""Stage 2 design references (v0.41, doc chat rekan engineer A8): cards from real sources, honest when the web is away."""
import pytest

import design_ext
from research import design_refs


BRIEF = {"object": "snow plow", "purpose": "clear a pickup truck bed", "material": "S355 steel",
         "process": "laser cut and welded", "key_sizes": "2.4 m blade"}

ROWS = [
    {"title": "Truck snow plow blade mount", "url": "https://example.com/a",
     "snippet": "The blade is laser cut from 6 mm S355 steel and TIG welded to a tube frame. "
                "The mount uses four M16 bolts and a gusset at each corner."},
    {"title": "Home-built plow frame", "url": "https://example.com/b",
     "snippet": "Most builds miter cut the tube and TIG weld the joints, with a bolted mount to the chassis."},
    {"title": "Hydraulic plow lift", "url": "https://example.com/c",
     "snippet": "A hydraulic ram lifts the 2.4 m blade; the frame is a tube frame with a gusset."},
]


def _searcher(rows=ROWS, error=None, calls=None):
    def run(query, count):
        if calls is not None:
            calls.append(query)
        return (list(rows), error)
    return run


def test_a_brief_gives_cards_with_their_source_and_labelled_numbers():
    design_refs.reset_cache()
    r = design_refs.run(BRIEF, "p1", searcher=_searcher())
    assert r["ok"] is True and len(r["cards"]) >= 3
    first = r["cards"][0]
    assert first["source_url"] == "https://example.com/a"
    assert first["what_it_is"] and first["key_features"]
    assert any(n["label"] == "web — not verified" for n in first["numbers_found"])
    assert all(n["source_url"].startswith("http") for n in first["numbers_found"])
    assert r["patterns"] and "of 3 sources" in r["patterns"][0]
    assert "sources share" in r["proposal"] or "do not agree" in r["proposal"]
    assert any(s.startswith("Searching:") for s in r["steps"])


def test_at_most_three_queries_per_reference_stage():
    design_refs.reset_cache()
    q = design_refs.queries_from(BRIEF)
    assert 1 <= len(q) <= 3
    assert q[0].startswith("S355 steel")


def test_the_same_query_is_cached_and_does_not_search_twice():
    design_refs.reset_cache()
    calls = []
    design_refs.run(BRIEF, "p1", searcher=_searcher(calls=calls))
    n = len(calls)
    r2 = design_refs.run(BRIEF, "p1", searcher=_searcher(calls=calls))
    assert len(calls) == n                      # every query came from the cache
    assert r2["cached_queries"] == len(r2["queries"])
    assert r2["ok"] is True


def test_a_provider_that_fails_says_so_and_does_not_stall_the_flow():
    design_refs.reset_cache()
    r = design_refs.run(BRIEF, "p1", searcher=_searcher(rows=[], error="web search unavailable: HTTP 402"))
    assert r["ok"] is False and r["cards"] == []
    assert "402" in (r["error"] or "")
    assert r["proposal"] == "" and r["mode"] == "none"


def test_no_search_key_is_reported_honestly(monkeypatch):
    design_refs.reset_cache()
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
    r = design_refs.run(BRIEF, "p1")
    assert r["ok"] is False
    assert "no search key" in (r["error"] or "")


def test_the_endpoint_answers_with_success_and_cards(monkeypatch):
    design_refs.reset_cache()
    monkeypatch.setattr(design_refs, "_web_search", lambda q, n: (ROWS, None))
    code, body = design_ext.handle_post("/references/design", {"brief": BRIEF, "project_id": "p9"})
    assert code == 200 and body["success"] is True
    assert body["project_id"] == "p9" and len(body["cards"]) >= 3
    assert body["cards"][0]["thumbnail_url"] is None or body["cards"][0]["thumbnail_url"].startswith("http")


def test_a_brief_that_says_nothing_is_refused_with_a_reason():
    code, body = design_ext.handle_post("/references/design", {"brief": {}})
    assert code == 200 and body["ok"] is False
    assert "nothing to search" in body["error"]
