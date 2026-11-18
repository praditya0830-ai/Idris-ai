"""AI-first routing (v0.34). The AI is faked: these tests pin what IDRIS does with its answer, and that keyword
rules still decide when the AI is away."""
import json

import cad_router as rt
import llm

TABLE = ("Design a welded steel table frame 1200 × 600 mm, height 750 mm, from 40 × 40 × 2 mm square tube, with a lower "
         "shelf frame 150 mm above the floor and levelling feet M10")
PLATE = ("Design a modular fixture plate 250 × 250 × 20 mm, M6 threaded holes on a 25 mm grid, two 8 mm dowel holes in "
         "opposite corners, aluminium 6061")
JAW = ("Design a 3-jaw style soft jaw set for a 100 mm lathe chuck, each jaw 50 × 20 × 25 mm with a stepped grip for a "
       "40 mm part, aluminium")


def _ai(monkeypatch, answer):
    seen = {}
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.setenv("IDRIS_LLM_ORDER", "claude")
    monkeypatch.delenv("IDRIS_AI_ROUTER", raising=False)

    def fake(prov, system, user, max_tokens=500, timeout=None):
        seen.update(prov=prov, system=system, user=user, max_tokens=max_tokens, timeout=timeout)
        if isinstance(answer, Exception):
            raise answer
        return (answer if isinstance(answer, str) else json.dumps(answer)), "claude-haiku-5-5"
    monkeypatch.setattr(llm, "call_provider", fake)
    return seen


def test_the_three_sentences_that_failed_on_14_nov_go_to_the_designer(monkeypatch):
    for s in (TABLE, PLATE, JAW):
        seen = _ai(monkeypatch, {"route": "free_design", "library_id": None, "query": {}, "reason": "custom part"})
        r = rt.route(s)
        assert r["route"] == "free_design" and r["read_by"] == "claude (claude-haiku-5-5)"
        assert s[:40] in seen["user"] and seen["max_tokens"] <= 500 and seen["timeout"] <= 15      # a short, cheap call
        assert "spur_gear" in seen["system"] and "W = Working width" in seen["system"]


def test_library_values_must_be_in_the_sentence(monkeypatch):
    _ai(monkeypatch, {"route": "library", "library_id": "spur_gear",
                      "query": {"m": 2, "z": 32, "b": 15, "bore": 12, "alpha": 20, "hub_d": 99, "keyway": True}})
    r = rt.route("spur gear module 2, 32 teeth, face width 15 mm, bore 12 mm with keyway")
    assert r["route"] == "library" and r["library_id"] == "spur_gear"
    assert r["query"]["m"] == "2" and r["query"]["z"] == "32" and r["query"]["keyway"] == "true"
    assert "hub_d" not in r["query"] and "alpha" not in r["query"]            # 99 and 20 were never said: dropped, not guessed


def test_text_values_and_unknown_parts(monkeypatch):
    _ai(monkeypatch, {"route": "library", "library_id": "socket_head_cap_screw", "query": {"size": "M6", "length": 20}})
    r = rt.route("I need an M6 x 20 socket head cap screw")
    assert r["query"].get("size", "").upper() == "M6" and r["query"]["length"] == "20"
    _ai(monkeypatch, {"route": "library", "library_id": "circlip", "query": {"d": 20}})
    assert rt.route("circlip for a 20 mm shaft")["route"] == "free_design"     # not in the library: IDRIS designs it


def test_ai_away_or_confused_means_the_rules_decide(monkeypatch):
    _ai(monkeypatch, llm.LLMError("could not reach the AI provider (TimeoutError)"))
    assert rt.route(TABLE) is None
    _ai(monkeypatch, "Sure! I think this is a table.")
    assert rt.route(TABLE) is None
    _ai(monkeypatch, {"route": "teleport"})
    assert rt.route(TABLE) is None
    _ai(monkeypatch, {"route": "free_design"})
    monkeypatch.setenv("IDRIS_AI_ROUTER", "0")
    assert rt.route(TABLE) is None
    monkeypatch.delenv("IDRIS_AI_ROUTER")
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    assert rt.route(TABLE) is None                                            # no key configured


def test_main_uses_the_router_before_the_keyword_rules():
    import os
    src = open(os.path.join(os.path.dirname(__file__), "..", "main.py"), encoding="utf-8").read()
    chat = src[src.index('if base == "/chat":'):]
    # the AI router is asked first; the keyword rules only answer once it has nothing usable to say
    assert chat.index("cad_router.route(") < chat.index("plib.match_sentence(msg)") < chat.index("self._chat_cad(msg, ctx")
    # a follow-up or a repair (pending_cad, cad_error, current_cad) never goes through the router, and neither
    # does a message the bench already computed: the guard sits between `routed = None` and the router call
    guard = chat[chat.index("routed = None"):chat.index("cad_router.route(")]
    assert "if not (gear_ctx or cad_ctx or eng_ctx):" in guard
    assert "gear_ctx" in guard and "cad_ctx" in guard and "eng_ctx" in guard


def test_free_design_brings_three_concepts_without_numbers(monkeypatch):
    _ai(monkeypatch, {"route": "free_design", "concepts": [
        {"name": "L-bracket with gusset", "idea": "bent plate with a 45 deg gusset"},
        {"name": "U-channel", "idea": "channel section, 3 mm walls"},
        {"name": "Ribbed plate 2", "idea": "flat plate with ribs"}, {"name": "u-channel", "idea": "dup"}, "junk"]})
    r = rt.route("Design a shelf bracket 150 x 100 mm, 3 mm steel, two M6 holes")
    assert [c["name"] for c in r["concepts"]] == ["L-bracket with gusset", "U-channel", "Ribbed plate"]
    assert not any(ch.isdigit() for c in r["concepts"] for ch in c["idea"])    # no size can travel from the AI's idea
