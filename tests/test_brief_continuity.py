"""Doc 2 (21 Nov evening): a brief that is open owns the conversation until the design starts.

The live failure: "design snow plow …" asked its questions, then the chip "+ add" sent
"add: Use a replaceable cutting edge on the bottom of the blade" and IDRIS answered that a cutting edge is not a
gearbox design parameter. The brief was lost and the old gearbox reader answered instead. These tests pin the four
rules that stop it: a chip fills a feature, a free answer is a brief fact, a question is answered as chat, and the
gearbox requirements prompt is used only when a gearbox is open.
"""
import json

import pytest

import brief
import cad_router
import llm

SNOW = "design a snow plow for a pickup truck"
ADD = "add: Use a replaceable cutting edge on the bottom of the blade"
FREE = "pick up truck, electric actuator, straight blade"


def _open_brief() -> dict:
    b = brief.blank("assembly")
    b.update({"object": "snow plow", "purpose": "clear snow from a pickup truck", "asked": True})
    return b


def _fake_ai(monkeypatch, answer):
    seen = {}
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.setenv("IDRIS_LLM_ORDER", "claude")
    monkeypatch.delenv("IDRIS_AI_ROUTER", raising=False)

    def fake(prov, system, user, max_tokens=500, timeout=None):
        seen.update(prov=prov, system=system, user=user, max_tokens=max_tokens)
        return (answer if isinstance(answer, str) else json.dumps(answer)), "claude-haiku-5-5"
    monkeypatch.setattr(llm, "call_provider", fake)
    return seen


# --------------------------------------------------------------------------------------------- chips
def test_an_add_chip_becomes_a_feature():
    chip = brief.chip_answer(ADD)
    assert chip["features"][0]["text"].startswith("Use a replaceable cutting edge")
    b = brief.merge(_open_brief(), chip, ADD)
    assert [f["text"] for f in b["features"]][0].startswith("Use a replaceable cutting edge")


def test_a_no_thanks_chip_is_recorded_not_routed():
    chip = brief.chip_answer("no thanks: add a rubber skirt")
    assert chip["declined"][0]["text"] == "add a rubber skirt"
    assert "features" not in chip


def test_the_chip_id_from_the_studio_fills_the_suggestion_idris_made():
    b = _open_brief()
    b["suggested"] = [{"id": "s1", "text": "Use a replaceable cutting edge"}]
    # a chip whose text is not prefixed (the Studio also sends the id): the id decides, and only a suggestion IDRIS
    # itself made can be added
    t = brief.turn(b, "yes, add that", {"brief_answer": {"type": "add", "suggestion_id": "s1"}})
    assert t["action"] == "chip" and t["merge"]["features"][0]["text"] == "Use a replaceable cutting edge"


def test_a_suggestion_id_idris_never_made_is_not_a_feature():
    b = _open_brief()
    b["suggested"] = [{"id": "s1", "text": "Use a replaceable cutting edge"}]
    t = brief.turn(b, "hmm", {"brief_answer": {"type": "add", "suggestion_id": "s9"}})
    assert t["action"] == "answer"


# --------------------------------------------------------------------------------------------- what a message is
def test_a_free_answer_is_a_brief_fact_and_a_question_is_chat():
    b = _open_brief()
    assert brief.turn(b, FREE)["action"] == "answer"
    assert brief.turn(b, "why is a replaceable edge better?")["action"] == "chat"
    assert brief.turn(b, "kenapa bilah lengkung lebih baik?")["action"] == "chat"


def test_a_ready_brief_no_longer_owns_the_conversation():
    b = _open_brief()
    b["ready"] = True
    assert brief.is_open({"brief": b}) is None
    assert brief.turn(None, FREE)["action"] == "none"


# --------------------------------------------------------------------------------------------- the sequence
def test_the_snow_plow_answer_lands_in_the_brief_with_no_gearbox_and_no_invented_number(monkeypatch):
    answer = {
        "understanding": "A snow plow for a pickup truck, raised by an electric actuator, with a straight blade.",
        "brief": {"object": "snow plow", "purpose": "clear snow from a pickup truck",
                  "constraints": "pick up truck, electric actuator, straight blade",
                  "key_sizes": "blade 2400 mm wide", "process": "welded steel"},
        "questions": [
            {"id": "q1", "text": "How wide is the blade?", "options": [
                {"label": "Matches the truck", "note": "legal width on the road"},
                {"label": "Wider than the truck", "note": "clears a path in one pass"}]},
            {"id": "q2", "text": "How does it mount to the truck?", "options": [
                {"label": "Receiver mount", "note": "removed in minutes"},
                {"label": "Bolt-on bracket", "note": "stays on the truck"}]}],
        "suggestions": [{"id": "s1", "text": "Use a replaceable cutting edge", "why": "the wear part is cheap"}],
    }
    seen = _fake_ai(monkeypatch, answer)
    b = _open_brief()
    w, err = brief.write_detail(FREE, b, brief.missing(b), None, "en")
    assert err is None and w
    # the brief prompt writes the questions, never the gearbox requirements prompt
    assert "senior mechanical engineer at IDRIS" in seen["system"]
    assert "power (kW" not in seen["system"]
    b2 = brief.merge(b, w["brief"], FREE)
    assert b2["constraints"] == "pick up truck, electric actuator, straight blade"
    assert "2400" not in (b2.get("key_sizes") or "")       # a size the engineer never wrote is dropped
    assert "wide" in w["questions"][0]["text"].lower()
    assert [s["id"] for s in w["suggestions"]] == ["s1"]
    # the brief is still open: the next question is about width / mounting, not about power or gear ratio
    text = json.dumps(w["questions"]).lower()
    assert "gear" not in text and "rpm" not in text


def test_the_brief_prompt_still_asks_about_the_object_it_was_given(monkeypatch):
    seen = _fake_ai(monkeypatch, {"brief": {}, "questions": []})
    b = _open_brief()
    brief.write_detail(ADD, b, brief.missing(b), None, "en")
    assert "snow plow" in seen["user"]                  # the brief travels with the message
    assert "Still missing" in seen["user"]


# --------------------------------------------------------------------------------------------- a gearbox is a new topic
def test_a_gearbox_request_replaces_the_open_brief():
    ctx = {"brief": {"object": "snow plow", "kind": "assembly"}}
    r = cad_router.topic_reset("now design a gearbox 11 kW ratio 6", ctx)
    assert r and r["to"] == "gearbox"
    assert r["previous"]["object"] == "snow plow"


def test_a_gearbox_sentence_does_not_reset_a_part_that_is_not_a_product():
    # review of PR #31: a sentence about a part of an electric motor never replaces an open housing
    ctx = {"current_cad": {"request": "gearbox housing 200 mm"}}
    assert cad_router.topic_reset("tambah dudukan motor listrik", ctx) is None


# --------------------------------------------------------------------------------------------- no gearbox prompt
def test_a_chat_without_a_gearbox_does_not_use_the_gearbox_prompt(monkeypatch):
    seen = _fake_ai(monkeypatch, {"intent": "discuss", "params": {}, "select": None, "compare": None,
                                  "answer": "A replaceable edge wears instead of the blade, so the blade lasts longer.",
                                  "language": "en"})
    res = llm.interpret(None, "why is a replaceable edge better?", None, None, None, {"brief": _open_brief()})
    assert "gearbox design tool" not in seen["system"]
    assert "No gearbox is open" in seen["system"]
    assert "not a gearbox design parameter" not in (res.get("answer") or "")


def test_a_chat_with_a_gearbox_keeps_the_gearbox_prompt(monkeypatch):
    seen = _fake_ai(monkeypatch, {"intent": "discuss", "params": {}, "select": None, "compare": None,
                                  "answer": "ok", "language": "en"})
    llm.interpret(None, "how hot is it?", {"power_kw": 11}, ["A"], None, None)
    assert "gearbox design tool" in seen["system"]
