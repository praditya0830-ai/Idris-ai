"""v0.40.1: the brief and the assembly plan survive a real AI failure.

Arya, 21 Nov: "buatkan saya design konsep sepeda" answered with
"the AI did not return structured data". The cause was the brief writer quietly returning None (a 800-token
answer cut off, prose, a fence) and the caller sliding back into the planner. These tests pin the new contract:

  * a cut-off object or a ```json fence is read (llm._close_json / salvage_json), so its questions survive;
  * prose, an empty answer, a timeout or a `{"questions": []}` answer produce the same fixed questions locally,
    marked `source: "fallback questions (AI unavailable)"`, with `brief_error` saying what happened;
  * in every one of those cases the engineer is asked something and no planner or codegen call is made;
  * the plan writer has room (4000 tokens), the same repair and one stricter retry.
"""
import json

import pytest

import ace_core
import assembly_plan
import brief as B
import cad_codegen
import llm
import main

DRONE = "buatkan aku design konsep drone"
BIKE = "buatkan saya design konsep sepeda"


def _ai(monkeypatch, replies):
    """Fake AI door: a list of replies (str/dict/Exception) is handed out one call at a time; the last repeats."""
    seen = []
    queue = list(replies) if isinstance(replies, list) else [replies]
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.setenv("IDRIS_LLM_ORDER", "claude")
    monkeypatch.delenv("IDRIS_AI_ROUTER", raising=False)

    def fake(prov, system, user, max_tokens=500, timeout=None):
        reply = queue.pop(0) if len(queue) > 1 else queue[0]
        seen.append({"prov": prov, "system": system, "user": user, "max_tokens": max_tokens, "timeout": timeout})
        if isinstance(reply, Exception):
            raise reply
        return (reply if isinstance(reply, str) else json.dumps(reply)), "claude-haiku-5-5"

    monkeypatch.setattr(llm, "call_provider", fake)
    return seen


def _no_design(monkeypatch):
    """Any codegen or planner call while the brief is open fails the whole feature."""
    def boom(*a, **k):
        raise AssertionError("no code and no plan may be written while the brief is open")
    for name in ("generate", "generate_concepts", "review"):
        if hasattr(cad_codegen, name):
            monkeypatch.setattr(cad_codegen, name, boom)
    monkeypatch.setattr(main.Handler, "_chat_assembly", boom, raising=False)
    monkeypatch.setattr(main.Handler, "_chat_cad", boom, raising=False)
    monkeypatch.setattr(main.Handler, "_chat_concepts", boom, raising=False)


def _routed():
    return {"route": "brief", "kind": "part", "brief": {}, "library_id": None, "query": {},
            "concepts": [], "reason": "not enough yet", "read_by": "fake"}


DRONE_ANSWER = {
    "understanding": "You want a drone concept, not a finished airframe yet.",
    "brief": {"object": "drone", "purpose": "aerial camera platform"},
    "questions": [
        {"id": "q1", "text": "What size class should it be?", "options": [
            {"label": "FPV 5-inch", "note": "fast and agile"},
            {"label": "450 mm camera", "note": "stable, carries a camera"}]},
        {"id": "q2", "text": "What must it carry?", "options": [
            {"label": "action camera", "note": "light mount"},
            {"label": "gimbal camera", "note": "vibration damping needed"}]}],
    "suggestions": [{"id": "s1", "text": "Folding landing gear", "why": "fits in a backpack"}],
}

BIKE_ANSWER = {
    "understanding": "You want a bicycle concept, not a final frame drawing yet.",
    "brief": {"object": "bicycle", "purpose": "everyday city riding"},
    "questions": [
        {"id": "q1", "text": "What kind of riding?", "options": [
            {"label": "city and commute", "note": "upright, comfortable"},
            {"label": "touring", "note": "loads and long days"}]},
        {"id": "q2", "text": "What frame material?", "options": [
            {"label": "steel", "note": "easy to repair"},
            {"label": "aluminium", "note": "lighter"}]}],
    "suggestions": [{"id": "s1", "text": "Rack mounts", "why": "carries shopping or panniers"}],
}

# The answers that produced the live error, plus the ones that must never be read as a finished brief.
FAILURES = [
    pytest.param('{"understanding":"A bicycle concept.","brief":{"object":"bicycle"},"questions":[{"id":"q1",'
                 '"text":"What kind of riding?","options":[{"label":"city","note":"upright', id="truncated-json"),
    pytest.param("I would build a bicycle with a steel frame and 700c wheels.", id="prose"),
    pytest.param("", id="empty-answer"),
    pytest.param(TimeoutError("timed out after 30 s"), id="timeout"),
    pytest.param({"understanding": "Sizes still unknown.", "brief": {}, "questions": [], "suggestions": []},
                 id="no-questions"),
    pytest.param({"questions": "What size?"}, id="questions-not-a-list"),
]


@pytest.mark.parametrize("reply", FAILURES)
@pytest.mark.parametrize("message", [BIKE, DRONE])
def test_a_failed_brief_always_asks_and_never_starts_a_design(monkeypatch, reply, message):
    _no_design(monkeypatch)
    _ai(monkeypatch, [reply, reply])                   # first call and the one stricter retry
    out = main.Handler._chat_brief(None, message, None, _routed(), None, None)
    assert out["continue"] is False, "a brief that could not be written must not fall through to design"
    r = out["out"]
    assert r["route"]["route"] == "brief" and r["ready"] is False
    assert 1 <= len(r["questions"]) <= 3
    for q in r["questions"]:
        assert q["text"] and len(q["options"]) >= 2 and q["allow_free_text"] is True
        for o in q["options"]:
            assert not any(ch.isdigit() for ch in o["label"] + o["note"]), "a fallback option invents no number"
    assert r["answer"], "the engineer gets a written reply, not an empty turn"
    assert "brief_error" in r
    err = r["brief_error"]
    assert set(("provider", "model", "stop_reason", "chars", "head", "error")) <= set(err)
    assert err["provider"] == "claude"
    assert isinstance(err["chars"], int) and len(err["head"]) <= 300


def test_the_cut_off_object_keeps_its_own_questions(monkeypatch):
    """A truncated answer is repaired, so the reader's question is asked instead of being thrown away."""
    _no_design(monkeypatch)
    _ai(monkeypatch, '{"understanding":"A bicycle concept.","brief":{"object":"bicycle"},"questions":[{"id":"q1",'
                     '"text":"What kind of riding?","options":[{"label":"city","note":"upright browsing"},'
                     '{"label":"touring","note":"long days with loads"}],"allow_free_text":true}]}')
    out = main.Handler._chat_brief(None, BIKE, None, _routed(), None, None)
    assert out["continue"] is False
    r = out["out"]
    assert "brief_error" not in r
    assert [q["text"] for q in r["questions"]] == ["What kind of riding?"]
    assert len(r["questions"][0]["options"]) == 2
    assert r["brief"]["object"] == "bicycle"


def test_a_json_fence_is_read_too(monkeypatch):
    _no_design(monkeypatch)
    _ai(monkeypatch, "```json\n" + json.dumps(DRONE_ANSWER) + "\n```")
    r = main.Handler._chat_brief(None, DRONE, None, _routed(), None, None)["out"]
    assert len(r["questions"]) == 2 and r["suggestions"][0]["text"] == "Folding landing gear"


def test_a_fallback_question_is_written_in_the_engineers_language(monkeypatch):
    _no_design(monkeypatch)
    _ai(monkeypatch, "I cannot answer in JSON right now.")
    r = main.Handler._chat_brief(None, "buatkan saya design konsep sepeda", None, _routed(), None, None)["out"]
    assert r["language"] == "id"
    assert any("Kira-kira sebesar apa" in q["text"] or "Untuk apa" in q["text"] for q in r["questions"])
    assert "Atau bilang" in r["answer"]


def test_a_successful_reader_answers_sepeda_and_drone(monkeypatch):
    for message, answer, word in ((BIKE, BIKE_ANSWER, "riding"), (DRONE, DRONE_ANSWER, "carry")):
        _no_design(monkeypatch)
        _ai(monkeypatch, answer)
        out = main.Handler._chat_brief(None, message, None, _routed(), None, None)
        assert out["continue"] is False
        r = out["out"]
        assert len(r["questions"]) >= 1 and len(r["suggestions"]) >= 1
        assert any(word in q["text"].lower() for q in r["questions"])
        assert r["route"]["route"] == "brief"


def test_the_brief_call_has_room_and_a_timeout_that_fits_it(monkeypatch):
    _no_design(monkeypatch)
    seen = _ai(monkeypatch, DRONE_ANSWER)
    monkeypatch.delenv("IDRIS_BRIEF_MAX_TOKENS", raising=False)
    monkeypatch.delenv("IDRIS_BRIEF_TIMEOUT_S", raising=False)
    main.Handler._chat_brief(None, DRONE, None, _routed(), None, None)
    assert seen[0]["max_tokens"] == 1800 and seen[0]["timeout"] == 30
    monkeypatch.setenv("IDRIS_BRIEF_MAX_TOKENS", "2600")
    monkeypatch.setenv("IDRIS_BRIEF_TIMEOUT_S", "45")
    assert B.max_tokens() == 2600 and B.timeout_s() == 45


def test_the_retry_asks_for_one_json_object(monkeypatch):
    _no_design(monkeypatch)
    seen = _ai(monkeypatch, ["Here is a friendly paragraph about bicycles.", DRONE_ANSWER])
    main.Handler._chat_brief(None, DRONE, None, _routed(), None, None)
    assert len(seen) == 2
    assert "Return ONLY one JSON object" in seen[1]["user"]
    assert "Return ONLY one JSON object" not in seen[0]["user"]


def test_the_brief_error_head_is_the_first_300_characters(monkeypatch):
    _no_design(monkeypatch)
    long_prose = "I would start with the frame. " * 40
    _ai(monkeypatch, long_prose)
    r = main.Handler._chat_brief(None, BIKE, None, _routed(), None, None)["out"]
    assert r["brief_error"]["chars"] == len(long_prose)
    assert r["brief_error"]["head"] == long_prose[:300]
    assert len(r["brief_error"]["attempts"]) == 2


# ------------------------------------------------------------------ the assembly plan, same failure, same repair
def _plan_call(*replies):
    queue = list(replies)

    def call(system, user):
        return queue.pop(0) if len(queue) > 1 else queue[0]

    return call


def _plan(*parts):
    return {"parts": list(parts), "interfaces": [], "frames": [], "sources": {}, "questions": []}


def test_a_cut_off_plan_is_repaired_instead_of_reported_as_no_structured_data():
    good = _plan({"id": "frame", "name": "Frame", "role": "carries the load", "qty": 1})
    text = json.dumps(good)[:-40]                     # the tail is gone: interfaces/frames never arrived
    out = assembly_plan.plan_for("whole bicycle", call=_plan_call(text, text))
    assert out["ok"] is True, out
    assert [p["id"] for p in out["plan"]["parts"]] == ["frame"]


def test_a_prose_plan_is_retried_once_and_then_explained():
    text = "I would split it into a frame and wheels."
    out = assembly_plan.plan_for("whole bicycle", call=_plan_call(text, text))
    assert out["ok"] is False
    assert out["plan_error"]["chars"] == len(text)
    assert out["plan_error"]["head"] == text[:300]
    assert len(out["plan_error"]["attempts"]) == 2
    assert "JSON" in out["plan_error"]["error"]


def test_the_plan_call_has_more_room_and_a_stricter_retry(monkeypatch):
    seen = _ai(monkeypatch, ["not JSON", json.dumps(_plan({"id": "frame", "name": "Frame"}))])
    out = assembly_plan.plan_for("whole bicycle")
    assert out["ok"] is True
    assert seen[0]["max_tokens"] == 4000 and seen[0]["timeout"] == 60
    assert "Return ONLY one JSON object" in seen[1]["user"]
    monkeypatch.setenv("IDRIS_PLAN_MAX_TOKENS", "6000")
    monkeypatch.setenv("IDRIS_PLAN_TIMEOUT_S", "90")
    assert assembly_plan.plan_max_tokens() == 6000 and assembly_plan.plan_timeout_s() == 90


def test_the_solver_reports_the_new_version():
    assert ace_core.SOLVER_VERSION == "0.41.0"
