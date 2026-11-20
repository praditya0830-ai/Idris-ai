"""UI v66 B: the engineer says how many concepts to compare, and only then is code written.

Rules under test: a fresh free design is asked first (no AI code call), a follow-up/repair/change is never asked,
a library or gearbox request is never asked, IDRIS_CONCEPTS=0 keeps the old one-design path, n concepts really are
n different shapes (written in stages of four), and a concept the time or money does not reach says so.
"""
import threading
import time

import ai_budget
import cad_codegen
import cad_router
import design_job as dj
import main

FREE = {"route": "free_design", "library_id": None, "query": {}, "reason": "a bracket", "read_by": "fake",
        "concepts": [{"name": "L-bracket", "idea": "two legs"}, {"name": "U-channel", "idea": "channel"}]}


def _concept(name, ok=True, error=None):
    return {"name": name, "idea": "shape", "ok": ok, "cad": {"title": name, "code": "return { body };"} if ok else None,
            "answer": "ok" if ok else (error or "no"), "error": error, "attempts": []}


# ------------------------------------------------------------------ the decision
def test_a_fresh_free_design_is_asked_before_any_code_is_written(monkeypatch):
    monkeypatch.setenv("IDRIS_CONCEPTS", "1")
    assert main._concept_decision(FREE, None, None) == "ask"
    assert main._concept_decision(FREE, {}, None) == "ask"


def test_the_answer_decides_the_path():
    assert main._concept_decision(FREE, None, 1) == "concepts"
    assert main._concept_decision(FREE, None, 4) == "concepts"
    assert main._concept_decision(FREE, None, 5) == "job"
    assert main._concept_decision(FREE, None, 10) == "job"


def test_a_follow_up_a_repair_and_an_open_part_are_never_asked():
    for key in ("pending_cad", "cad_error", "current_cad"):
        assert main._concept_decision(FREE, {key: {"x": 1}}, None) == "cad", key


def test_other_routes_are_never_asked():
    for route in ("library", "primitive", "gearbox", "assembly", "other", None):
        routed = None if route is None else {**FREE, "route": route}
        assert main._concept_decision(routed, None, None) == "cad", route


def test_concepts_can_be_switched_off_for_everyone(monkeypatch):
    monkeypatch.setenv("IDRIS_CONCEPTS", "0")
    assert main._concept_decision(FREE, None, None) == "cad"
    monkeypatch.setenv("IDRIS_CONCEPTS", "off")
    assert main._concept_decision(FREE, None, None) == "cad"


def test_the_question_is_free_and_names_the_options():
    out = main._ask_concept_count(FREE)
    assert out["ask"] == {"k": "concept_count", "options": [3, 5, 7, 10], "default": 3}
    assert out["answer"] == "How many concepts should I design?"
    assert out["cad"] is None and out["concepts"] == [] and out["route"] == FREE


def test_only_an_integer_from_one_to_ten_is_a_count():
    assert main._concept_count(None) == (None, None)
    assert main._concept_count("5") == (5, None)
    for bad in (0, 11, -3, "many", 3.5, True):
        count, err = main._concept_count(bad)
        assert count is None and "between 1 and 10" in err, bad


# ------------------------------------------------------------------ writing them
def test_five_concepts_are_written_even_when_the_router_proposed_two(monkeypatch):
    asked = {}

    def concepts_for(msg, count, provider=None):
        asked["count"] = count
        return [{"name": f"Shape {i}", "idea": "x"} for i in range(count)]

    monkeypatch.setattr(cad_router, "concepts_for", concepts_for)

    def fake_generate(message, concepts, provider=None, reply_language=None, deadline=None):
        assert asked["count"] == 5 and len(concepts) == 5
        return {"ok": True, "cad": {"title": "Shape 0"}, "requested": 5, "written": 5,
                "answer": "5 concepts", "read_by": "fake",
                "concepts": [_concept(c["name"]) for c in concepts]}

    monkeypatch.setattr(cad_codegen, "generate_concepts", fake_generate)
    out = main.Handler._chat_concepts(None, "bracket L 40", FREE["concepts"], None, None, count=5)
    assert out["concept_count"] == 5 and out["written"] == 5
    assert [c["ok"] for c in out["concepts"]] == [True] * 5
    assert len({c["name"] for c in out["concepts"]}) == 5


def test_at_most_four_concepts_are_written_at_the_same_time(monkeypatch):
    live, peak, lock = {"n": 0}, {"n": 0}, threading.Lock()

    def fake_generate(message, context=None, provider=None, reply_language=None):
        with lock:
            live["n"] += 1
            peak["n"] = max(peak["n"], live["n"])
        time.sleep(0.05)
        with lock:
            live["n"] -= 1
        return {"ok": True, "cad": {"title": "c"}, "answer": "", "attempts": []}

    monkeypatch.setattr(cad_codegen, "generate", fake_generate)
    concepts = [{"name": f"c{i}", "idea": ""} for i in range(10)]
    r = cad_codegen.generate_concepts("bracket", concepts)
    assert r["ok"] and r["written"] == 10 and peak["n"] <= 4


def test_a_concept_the_time_budget_cannot_reach_says_so(monkeypatch):
    calls = {"n": 0}
    monkeypatch.setattr(cad_codegen, "generate",
                        lambda *a, **k: calls.__setitem__("n", calls["n"] + 1) or
                        {"ok": True, "cad": {"title": "c"}, "answer": "", "attempts": []})
    r = cad_codegen.generate_concepts("bracket", [{"name": "a", "idea": ""}, {"name": "b", "idea": ""}],
                                      deadline=time.time() - 1)
    assert calls["n"] == 0
    assert [c["error"] for c in r["concepts"]] == ["time budget used up"] * 2
    assert not r["ok"] and "time budget used up" in r["concepts"][0]["answer"]


def test_a_concept_the_ai_budget_cannot_pay_for_is_reported_not_dropped(monkeypatch):
    monkeypatch.setattr(ai_budget, "gate", lambda *a, **k: "the daily AI budget is spent")
    r = cad_codegen.generate_concepts("bracket", [{"name": "a", "idea": ""}, {"name": "b", "idea": ""}])
    assert r["ok"] is False and len(r["concepts"]) == 2
    assert all(c["ok"] is False and "daily AI budget" in c["error"] for c in r["concepts"])


# ------------------------------------------------------------------ the background job
def test_a_concept_job_returns_every_concept_and_promises_nothing_it_did_not_check(monkeypatch):
    monkeypatch.setattr(cad_router, "concepts_for",
                        lambda msg, count, provider=None: [{"name": f"Shape {i}", "idea": ""} for i in range(count)])
    monkeypatch.setattr(cad_codegen, "generate_concepts",
                        lambda message, concepts, provider=None, reply_language=None, deadline=None: {
                            "ok": True, "cad": {"title": "Shape 0"}, "read_by": "fake", "requested": len(concepts),
                            "written": len(concepts), "answer": "5 concepts",
                            "concepts": [_concept(c["name"]) for c in concepts]})
    dj._STORE.clear()
    res = dj.run("bracket L 40", concept_count=5, job_id="cj1")
    assert res["concept_count"] == 5 and len(res["concepts"]) == 5
    assert res["label"] == "Not validated" and res["checks"] == []
    assert any("kernel" in n for n in res["not_checked"]), "no concept was measured here"
    assert [e["step"] for e in res["events"]] == ["plan", "code", "code"]
    dj._STORE.clear()
