"""Research layer: a number is accepted only together with a quote that really is on the page."""
import json

import llm
import research
from research import extract, fetch, lookup, search

PAGE = ("Phone X datasheet. Dimensions: height 160.8 mm, width 78.1 mm, thickness 7.8 mm. "
        "Mass 221 g. The bezel is 0.21 cm wide and the camera bump is 1.35 mm.")
KEY = "BRAVE_SEARCH_API_KEY"


def _on(monkeypatch):
    monkeypatch.setenv(KEY, "test")
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)


def _hits(url="https://ex.test/x"):
    return lambda q, count=5: [{"title": "Datasheet", "url": url, "snippet": ""}]


def _model(value, unit, quote):
    def fake(provider, system, user, max_tokens=500, timeout=None):
        return json.dumps({"value": value, "unit": unit, "quote": quote}), "fake-model"
    return fake


def _wire(monkeypatch, reply, page=PAGE, url="https://ex.test/x"):
    _on(monkeypatch)
    monkeypatch.setattr(search, "search", _hits(url))
    monkeypatch.setattr(fetch, "fetch", lambda u, **k: page)
    monkeypatch.setattr(llm, "call_provider", reply)


def test_off_without_a_search_key(monkeypatch):
    monkeypatch.delenv(KEY, raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    assert research.available() is False
    assert search.search("thickness") == []
    assert lookup.notes_for([{"k": "T", "question": "thickness?"}]) == ("", [])


def test_value_with_a_real_quote_is_accepted(monkeypatch):
    _wire(monkeypatch, _model(7.8, "mm", "thickness 7.8 mm"))
    note, srcs = lookup.notes_for([{"k": "T", "question": "How thick is the phone (mm)?"}],
                                  "holder hp", provider="claude")
    assert len(srcs) == 1
    assert srcs[0]["value_mm"] == 7.8 and srcs[0]["url"] == "https://ex.test/x"
    assert srcs[0]["quote"].startswith("thickness 7.8 mm")
    assert "https://ex.test/x" in note and "T = 7.8 mm" in note


def test_invented_quote_is_rejected(monkeypatch):
    """The model reports a number whose sentence is not on the page: it must not be used."""
    _wire(monkeypatch, _model(4.2, "mm", "thickness 4.2 mm"))
    assert lookup.notes_for([{"k": "T", "question": "How thick is the phone (mm)?"}],
                            "holder hp", provider="claude") == ("", [])


def test_unit_is_converted_to_mm(monkeypatch):
    _wire(monkeypatch, _model(0.21, "cm", "bezel is 0.21 cm wide"))
    _note, srcs = lookup.notes_for([{"k": "B", "question": "bezel width"}], provider="claude")
    assert srcs and srcs[0]["value_mm"] == 2.1 and srcs[0]["unit_in_source"] == "cm"


def test_unknown_unit_or_null_value_gives_nothing(monkeypatch):
    _wire(monkeypatch, _model(None, "mm", "thickness 7.8 mm"))
    assert lookup.notes_for([{"k": "T", "question": "thickness"}], provider="claude") == ("", [])
    _wire(monkeypatch, _model(7.8, "furlong", "thickness 7.8 mm"))
    assert lookup.notes_for([{"k": "T", "question": "thickness"}], provider="claude") == ("", [])


def test_enrich_attaches_sources_and_marks_the_param(monkeypatch):
    srcs = [{"k": "T", "value_mm": 7.8, "unit_in_source": "mm", "url": "https://ex.test/x",
             "quote": "thickness 7.8 mm", "title": "Datasheet"}]
    cad = {"title": "Holder", "params": [{"k": "T", "value": 7.8, "source": "assumed"}]}
    out = research.enrich(cad, srcs)
    assert out["sources"][0]["url"] == "https://ex.test/x"
    assert out["params"][0]["source"] == "research" and out["params"][0]["source_url"] == "https://ex.test/x"
    assert "sources" not in cad                      # the input is not mutated


def test_fetch_refuses_anything_but_http():
    assert fetch.fetch("file:///etc/passwd") == ""
    assert fetch.fetch("javascript:alert(1)") == ""


def test_quote_check_ignores_punctuation_but_not_meaning():
    assert extract.quote_is_real("thickness 7.8 mm", PAGE)
    assert extract.quote_is_real("Thickness: 7,8 mm", PAGE) is False     # the page says 7.8
    assert extract.quote_is_real("7.8", PAGE) is False                   # too short to be evidence
