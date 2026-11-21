"""Design brief (v0.40): IDRIS asks like an engineer before it writes any code.

The AI is faked here: these tests pin what IDRIS does with the reader's answer — the route, the questions, the
suggestions, the conversation memory and the numbers that may never travel from the AI into the brief.
"""
import json
import re

import brief as B
import cad_codegen
import cad_router
import llm
import main

DRONE = "buatkan aku design konsep drone"
MOTOR = "design concept electric motor that is quiet"
BRACKET = "L bracket 80 x 60 mm, 5 mm, aluminium, 500 N at the tip"


def _ai(monkeypatch, answer):
    """Fake the AI door; returns what was sent (for the prompt-shape test)."""
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


def _no_code(monkeypatch):
    """Any codegen/planner call during a brief is a failure of the whole feature."""
    def boom(*a, **k):
        raise AssertionError("no code may be written while the brief is open")
    for name in ("generate", "generate_concepts", "review"):
        if hasattr(cad_codegen, name):
            monkeypatch.setattr(cad_codegen, name, boom)
    monkeypatch.setattr(main.Handler, "_chat_assembly", boom, raising=False)
    monkeypatch.setattr(main.Handler, "_chat_cad", boom, raising=False)
    monkeypatch.setattr(main.Handler, "_chat_concepts", boom, raising=False)


def _routed(kind="part", brief=None):
    return {"route": "brief", "kind": kind, "brief": brief or {}, "library_id": None, "query": {},
            "concepts": [], "reason": "not enough yet", "read_by": "fake"}


DRONE_ANSWER = {
    "understanding": "You want a drone concept, not a finished airframe yet.",
    "brief": {"object": "drone", "purpose": "aerial camera platform"},
    "questions": [
        {"id": "q1", "text": "What size class should it be?", "options": [
            {"label": "FPV 5-inch", "note": "fast and agile"},
            {"label": "450 mm camera", "note": "stable, carries a camera"},
            {"label": "heavy-lift 8-10 inch", "note": "more payload, more cost"}]},
        {"id": "q2", "text": "What must it carry?", "options": [
            {"label": "action camera", "note": "light mount"},
            {"label": "gimbal camera", "note": "vibration damping needed"}]}],
    "suggestions": [
        {"id": "s1", "text": "Folding landing gear", "why": "fits in a backpack"},
        {"id": "s2", "text": "Vibration-damped gimbal mount", "why": "cleaner footage"},
        {"id": "s3", "text": "Quick battery hatch", "why": "faster field changes"}],
}


def test_a_general_drone_request_becomes_a_brief_with_questions_and_never_writes_code(monkeypatch):
    _no_code(monkeypatch)
    seen = _ai(monkeypatch, DRONE_ANSWER)
    out = main.Handler._chat_brief(None, DRONE, None, _routed(), None, None)
    assert not out["continue"], "a vague request must not fall through to design"
    r = out["out"]
    assert r["route"]["route"] == "brief" and r["route"]["of"] == "free_design"
    assert r["ready"] is False
    assert len(r["questions"]) >= 1 and all(len(q["options"]) >= 2 for q in r["questions"])
    assert all(q["allow_free_text"] is True for q in r["questions"])
    assert len(r["suggestions"]) >= 1
    assert seen["max_tokens"] == 1800 and seen["timeout"] == 30         # room for questions + options + suggestions
    assert "one to three feature suggestions" in seen["system"].lower() or "feature suggestions" in seen["system"]


def test_the_motor_brief_asks_power_size_and_use_and_sells_no_performance(monkeypatch):
    _no_code(monkeypatch)
    _ai(monkeypatch, {
        "understanding": "A quiet electric motor concept.",
        "brief": {"object": "electric motor", "purpose": "quiet drive"},
        "questions": [
            {"id": "q1", "text": "How much power?", "options": [
                {"label": "300 W", "note": "small fan or pump"},
                {"label": "750 W", "note": "light vehicle"},
                {"label": "2.2 kW", "note": "more torque, more heat"}]},
            {"id": "q2", "text": "What size fits?", "options": [
                {"label": "NEMA 34", "note": "common frame"},
                {"label": "custom 120 mm", "note": "free envelope"}]},
            {"id": "q3", "text": "Where is it used?", "options": [
                {"label": "indoor appliance", "note": "sound matters most"},
                {"label": "outdoor drive", "note": "IP rating matters"}]}],
        "suggestions": [
            {"id": "s1", "text": "Cooling fins", "why": "more surface without a fan", "note": "flies 25 minutes"},
            {"id": "s2", "text": "Magnet skew", "why": "lower torque ripple and noise"},
            {"id": "s3", "text": "IP54 sealing", "why": "dust and splash protection"}],
    })
    out = main.Handler._chat_brief(None, MOTOR, None, _routed(), None, None)
    r = out["out"]
    text = " ".join([r["answer"]] + [o["note"] for q in r["questions"] for o in q["options"]]
                    + [s["text"] + " " + s["why"] for s in r["suggestions"]])
    assert "power" in text.lower() and "size" in text.lower() and "used" in text.lower()
    assert any("cooling" in s["text"].lower() for s in r["suggestions"])
    assert any("magnet skew" in s["text"].lower() for s in r["suggestions"])
    # a performance claim the AI made up never reaches the engineer
    assert not re.search(r"\d+\s*(min|menit|jam|hours?|km|wh|db)\b", text, re.I), text


def test_a_complete_bracket_skips_the_brief_entirely(monkeypatch):
    _no_code(monkeypatch)
    _ai(monkeypatch, AssertionError("the brief writer must not be called for a complete request"))
    complete = {"object": "L bracket", "purpose": "holds a 500 N tip load",
                "key_sizes": "80 x 60 mm, 5 mm thick", "loads": "500 N at the tip", "material": "aluminium"}
    out = main.Handler._chat_brief(None, BRACKET, None, _routed("part", complete), None, None)
    assert out["continue"] is True, "a request that already says enough goes straight to design"
    assert out["message"] == BRACKET


def test_turn_two_fills_the_brief_and_asks_how_many_concepts(monkeypatch):
    _no_code(monkeypatch)
    _ai(monkeypatch, DRONE_ANSWER)
    first = main.Handler._chat_brief(None, DRONE, None, _routed(), None, None)
    assert not first["continue"]
    ctx = {"brief": first["out"]["brief"]}
    _ai(monkeypatch, {"understanding": "450 mm camera class, 1 kg camera.",
                      "brief": {"envelope": "450 mm", "loads": "1 kg camera"},
                      "questions": [{"id": "q1", "text": "Battery or tethered?", "options": [
                          {"label": "battery", "note": "free flight"}, {"label": "tethered", "note": "longer runs"}]}],
                      "suggestions": [{"id": "s1", "text": "Folding arms", "why": "transport"}]})
    out = main.Handler._chat_brief(None, "yang 450 mm, buat kamera 1 kg", ctx, _routed(), None, None)
    assert not out["continue"]
    r = out["out"]
    assert r["ready"] is True and r["brief"]["ready"] is True
    assert r["ask"] == {"k": "concept_count", "options": [3, 5, 7, 10], "default": 3}
    assert r["brief"]["envelope"] == "450 mm" and "1 kg" in r["brief"]["loads"]
    assert "450 mm" in r["answer"]
    assert "konsep" in r["answer"].lower()          # the engineer wrote Indonesian, so does IDRIS


def test_just_design_it_continues_with_written_assumptions(monkeypatch):
    _no_code(monkeypatch)
    _ai(monkeypatch, DRONE_ANSWER)
    first = main.Handler._chat_brief(None, DRONE, None, _routed(), None, None)
    ctx = {"brief": first["out"]["brief"]}
    out = main.Handler._chat_brief(None, "langsung aja desain", ctx, _routed(), None, None)
    assert out["continue"] is True
    a = out["brief"]["assumptions"]
    assert a and all(x.get("assumed") is True for x in a)
    assert out["brief"]["ready"] is True and out["brief"]["assumed"] is True
    assert "assumptions" in out["message"].lower() and "assumed" in out["message"].lower()
    assert out["message"].startswith("Design ")


def test_an_indonesian_message_is_answered_in_indonesian(monkeypatch):
    _no_code(monkeypatch)
    _ai(monkeypatch, {**DRONE_ANSWER, "understanding": "Kamu mau konsep drone dulu, belum airframe final."})
    out = main.Handler._chat_brief(None, DRONE, None, _routed(), None, None)
    r = out["out"]
    assert r["language"] == "id"
    assert "Atau bilang" in r["answer"]          # the fixed line, not only the AI text
    assert B.language("design concept electric motor that is quiet") == "en"


def test_the_ai_cannot_smuggle_a_size_into_the_brief():
    b = B.merge(B.blank("part"), {"object": "drone", "key_sizes": "600 mm", "envelope": "450 mm"}, "drone 450 mm")
    assert b["object"] == "drone"
    assert b["envelope"] == "450 mm"             # the engineer said 450
    assert b["key_sizes"] == ""                  # 600 was never said
    b2 = B.merge(b, {"key_sizes": "600 mm"}, "make it 600 mm long")
    assert b2["key_sizes"] == "600 mm"           # now it was


def test_inch_and_cm_written_with_their_unit_are_grounded():
    b = B.merge(B.blank("part"), {"key_sizes": "5 inch", "envelope": "300 mm"}, "a 5 inch frame, 30 cm across")
    assert b["key_sizes"] == "5 inch" and b["envelope"] == "300 mm"


def test_readiness_needs_an_object_a_purpose_and_a_size_and_a_load_only_when_it_matters():
    assert B.missing(B.blank()) == ["object", "purpose", "sizes"]
    b = B.merge(B.blank(), {"object": "bracket", "purpose": "hold a shelf", "key_sizes": "200 mm"},
                "bracket 200 mm to hold a shelf")
    assert B.missing(b) == []
    c = B.merge(B.blank(), {"object": "bracket", "purpose": "carry a 500 N load", "key_sizes": "200 mm"},
                "bracket 200 mm that must carry a 500 N load")
    assert B.missing(c) == ["loads"]


def test_the_resume_needs_an_explicit_start_and_keeps_the_brief():
    ready = {**B.blank("part"), "object": "drone", "purpose": "camera", "key_sizes": "450 mm", "ready": True}
    assert B.resume({"brief": ready}, "kenapa carbon lebih baik dari aluminium?") is None
    assert B.resume({"brief": ready}, "3")["message"].startswith("Design ")
    assert B.resume({"brief": ready}, "3")["concept_count"] == 3
    assert B.resume({"brief": ready}, "ya lanjut")["kind"] == "part"
    assert B.resume({"brief": {**ready, "ready": False}}, "3") is None


def test_the_router_can_return_brief_and_only_the_sentence_supplies_numbers(monkeypatch):
    seen = _ai(monkeypatch, {"route": "brief", "kind": "part", "reason": "no size",
                             "brief": {"object": "drone", "key_sizes": "600 mm"}})
    r = cad_router.route(DRONE)
    assert r["route"] == "brief" and r["kind"] == "part"
    assert r["brief"] == {"object": "drone", "key_sizes": "600 mm"}      # merge, not the router, drops the digit
    assert "brief" in seen["system"] and "kind" in seen["system"]
    assert '"brief"' in seen["system"]


def test_a_reader_answer_with_no_questions_still_completes_the_brief(monkeypatch):
    """The answer itself can finish the brief; the reader does not have to ask anything."""
    _no_code(monkeypatch)
    _ai(monkeypatch, DRONE_ANSWER)
    first = main.Handler._chat_brief(None, DRONE, None, _routed(), None, None)
    ctx = {"brief": first["out"]["brief"]}
    _ai(monkeypatch, {"understanding": "450 mm class, a 1 kg camera.",
                      "brief": {"envelope": "450 mm", "loads": "1 kg camera"}, "questions": [], "suggestions": []})
    out = main.Handler._chat_brief(None, "yang 450 mm, buat kamera 1 kg", ctx, _routed(), None, None)
    assert not out["continue"], "a brief that is now complete goes to the start question, not back to design"
    r = out["out"]
    assert r["ready"] is True and r["questions"] == [] and r["suggestions"] == []
    assert r["ask"]["k"] == "concept_count" and "450 mm" in r["answer"]


def test_a_reader_answer_with_no_questions_and_a_still_open_brief_asks_the_fixed_questions(monkeypatch):
    _no_code(monkeypatch)
    _ai(monkeypatch, DRONE_ANSWER)
    first = main.Handler._chat_brief(None, DRONE, None, _routed(), None, None)
    ctx = {"brief": first["out"]["brief"]}
    _ai(monkeypatch, {"understanding": "Sizes still unknown.", "brief": {}, "questions": [], "suggestions": []})
    out = main.Handler._chat_brief(None, "yang kedua aja", ctx, _routed(), None, None)
    assert out["continue"] is False, "an open brief is never skipped because the reader asked nothing"
    assert out["brief"]["object"] == "drone"          # the memory is kept
    r = out["out"]
    assert r["brief"]["source"] == "fallback questions (AI unavailable)"
    assert r["questions"] and all(q["fallback"] for q in r["questions"])


def test_with_no_ai_the_fixed_questions_are_asked_instead_of_designing_anyway(monkeypatch):
    _no_code(monkeypatch)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("IDRIS_LLM_ORDER", "claude")
    out = main.Handler._chat_brief(None, DRONE, None, _routed(), None, None)
    assert out["continue"] is False, "no AI must not mean 'design it anyway'"
    r = out["out"]
    assert r["questions"] and len(r["questions"]) <= 3
    assert r["brief"]["source"] == "fallback questions (AI unavailable)"
    assert "no AI provider is configured" in r["brief_error"]["error"]
