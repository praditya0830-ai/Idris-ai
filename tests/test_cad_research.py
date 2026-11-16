"""The research layer inside cad_codegen: a size nobody stated is looked up once, with a source.
Off by default (no search key -> no note), so the engineer is asked exactly as before."""
import json

import cad_codegen as cg
import llm
import research

GOOD = {"title": "L bracket", "kind": "functional", "process": "fdm", "material": "pla",
        "params": [{"k": "L", "value": 40, "min": 10, "max": 200, "source": "user"},
                   {"k": "t", "value": 3, "min": 1.2, "max": 10, "source": "user", "wall": True},
                   {"k": "W", "value": 20, "min": 10, "max": 100, "source": "user"}],
        "code": "const { L, t, W } = P;\nconst body = makeBox([0,0,0],[L,W,t]).fuse(makeBox([0,0,0],[t,W,L]));\nreturn { body };",
        "expect": [{"label": "Leg X", "axis": "x", "value": 40}, {"label": "Width", "axis": "y", "value": 20}],
        "missing": [], "variants": [{"name": "Sturdy", "params": {"t": 5, "W": 999}}]}
MISSING = {"title": "Phone stand", "params": [], "code": "return { body };",
           "missing": [{"k": "T", "question": "Berapa tebal HP?"}]}
MSG = "bracket L 40 mm tebal 3 mm"
SRCS = [{"k": "T", "value_mm": 8.0, "unit_in_source": "mm", "url": "https://example.test/spec",
         "quote": "thickness 8 mm", "title": "Phone case spec"}]


def _fake(monkeypatch, answers):
    calls = []
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    monkeypatch.setenv("IDRIS_CAD_ORDER", "gemini")

    def fake(prov, system, user, max_tokens=500, timeout=None):
        calls.append(user)
        a = answers[min(len(calls) - 1, len(answers) - 1)]
        return (a if isinstance(a, str) else json.dumps(a)), "gemini-test"
    monkeypatch.setattr(llm, "call_provider", fake)
    return calls


def test_no_search_key_asks_the_engineer_as_before(monkeypatch):
    monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    calls = _fake(monkeypatch, [MISSING])
    r = cg.generate(MSG)
    assert r["ok"] and len(calls) == 1                     # no lookup, no second call
    assert r["cad"]["missing"] and "tebal HP" in r["answer"]
    assert not (r["cad"].get("sources") or [])


def test_looked_up_size_comes_back_with_its_source(monkeypatch):
    calls = _fake(monkeypatch, [MISSING, GOOD])
    monkeypatch.setattr(research, "notes_for",
                        lambda missing, request, provider: ("\n\nNOTE T = 8 mm", SRCS))
    r = cg.generate(MSG)
    assert len(calls) == 2 and "NOTE T = 8 mm" in calls[1]  # the note goes to the same model once
    assert not r["cad"]["missing"]                          # the question is gone
    assert r["cad"]["sources"][0]["url"] == SRCS[0]["url"]  # and the number keeps its source
    assert r["cad"]["sources"][0]["quote"] == SRCS[0]["quote"]
    assert "Sources: T 8 mm (https://example.test/spec)" in r["answer"]


def test_a_failing_lookup_does_not_break_the_design(monkeypatch):
    calls = _fake(monkeypatch, [MISSING])

    def boom(*a, **k):
        raise RuntimeError("search is down")
    monkeypatch.setattr(research, "notes_for", boom)
    r = cg.generate(MSG)
    assert r["ok"] and len(calls) == 1 and r["cad"]["missing"]


def test_a_size_from_a_source_is_marked_on_the_param(monkeypatch):
    cad = {"params": [{"k": "T", "value": 8.0, "source": "default"}]}
    out = cg._research_enrich(cad, SRCS)
    assert out["params"][0]["source"] == "research"
    assert out["params"][0]["source_url"] == SRCS[0]["url"]
    assert cg._sources_text(out) == "Sources: T 8 mm (https://example.test/spec)"


def test_cad_order_offers_the_new_providers(monkeypatch):
    for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "NVIDIA_API_KEY",
              "DEEPSEEK_API_KEY", "DASHSCOPE_API_KEY", "MOONSHOT_API_KEY", "IDRIS_CAD_ORDER"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
    assert cg.cad_order() == ["deepseek"]
    monkeypatch.setenv("MOONSHOT_API_KEY", "k")
    monkeypatch.setenv("IDRIS_CAD_ORDER", "moonshot,claude,kimi")
    assert cg.cad_order() == ["kimi"]                      # unknown names are ignored
