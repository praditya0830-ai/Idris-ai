"""Review of PR #5-#8, items 1-5: the fixes that came out of reading the live code.

No AI call and no CAD kernel: every rule under test is a server-side rule, and the two jobs run with a fake
designer. Item 6 (the DFM sources) has its own file, tests/test_dfm_sources.py.
"""
import threading
import time

import pytest

import assembly_ext
import assembly_plan
import cad_fallback
import main
from ace_core import objectives as obj

FRAMES = [{"part_id": "a", "origin_mm": [0, 0, 0], "axis": [0, 0, 1]},
          {"part_id": "b", "origin_mm": [1, 0, 0], "axis": [0, 0, 1]}]
PARTS = [{"id": "a", "name": "A", "role": "one"}, {"id": "b", "name": "B", "role": "two"}]


def _raw(source: str | None = "user") -> dict:
    iface = {"between": ["a", "b"], "kind": "press", "params": {"d": 44.5}}
    if source:
        iface["source"] = source
    return {"parts": PARTS, "interfaces": [iface], "frames": FRAMES, "questions": []}


@pytest.fixture(autouse=True)
def _clean_jobs():
    assembly_ext._JOBS.clear()
    assembly_ext._RUNNING["n"] = 0
    yield
    assembly_ext._JOBS.clear()
    assembly_ext._RUNNING["n"] = 0


def _ok(prompt, provider=None):
    return {"ok": True, "cad": {"title": "part", "code": "return { body };", "params": []},
            "fidelity": "full", "validated": False, "not_checked": []}


# ------------------------------------------------------------------ item 1: one time budget for the chat ladder
def test_a_design_that_used_the_budget_does_not_start_a_second_ai_call(monkeypatch):
    import cad_codegen
    monkeypatch.setenv("IDRIS_CAD_BUDGET_S", "0")
    calls = []

    def gen(msg, ctx, prov, lang):
        calls.append(msg)
        return {"ok": False, "cad": None, "answer": "the AI did not answer in time", "attempts": []}

    monkeypatch.setattr(cad_codegen, "generate", gen)
    out = main.Handler._chat_cad(None, "plat 120 mm x 80 mm x 6 mm", None)
    assert len(calls) == 1, "the budget was used up, so the simplified call is not started"
    assert out["fidelity"] == "placeholder", "the placeholder rung has no AI and always runs"
    assert [p["value"] for p in out["cad"]["params"]] == [120.0, 80.0, 6.0]


def test_the_chat_ladder_still_asks_for_the_simplified_version_while_there_is_time(monkeypatch):
    import cad_codegen
    monkeypatch.setenv("IDRIS_CAD_BUDGET_S", "85")
    calls = []

    def gen(msg, ctx, prov, lang):
        calls.append(msg)
        return {"ok": False, "cad": None, "answer": "no", "attempts": []}

    monkeypatch.setattr(cad_codegen, "generate", gen)
    out = main.Handler._chat_cad(None, "plat 120 mm x 80 mm", None)
    assert len(calls) == 2 and "simplest valid version" in calls[1].lower()
    assert out["fidelity"] == "placeholder"


# ------------------------------------------------------------------ item 2: honest verdicts
def test_a_quiet_motor_is_not_doubtful_it_is_not_checkable_yet():
    tc = cad_fallback.target_check("motor listrik yang senyap")
    assert tc["verdict"] == "not_checkable_yet"
    assert tc["verdict"] in cad_fallback.VERDICTS
    assert "no acoustic solver" in tc["reason"].lower()
    assert tc["what_is_checkable"], "what can be measured is still stated"


def test_a_thermal_target_is_not_checkable_yet_too():
    tc = cad_fallback.target_check("housing lebih dingin")
    assert tc["verdict"] == "not_checkable_yet"
    assert "no thermal solver" in tc["reason"].lower()


def test_a_target_physics_opposes_keeps_its_own_verdict():
    tc = cad_fallback.target_check("sayap lebih cepat 50% dan hemat bahan bakar 50%")
    assert tc["verdict"] == "not_achievable_by_this_part", "a law of physics still wins over 'not checkable'"


# ------------------------------------------------------------------ item 3: one assembly at a time
def test_one_assembly_designs_parts_at_a_time_and_the_next_ones_wait(monkeypatch):
    monkeypatch.setenv("IDRIS_ASSEMBLY_MAX_JOBS", "1")
    gate, started = threading.Event(), threading.Event()

    def designer(prompt, provider=None):
        started.set()
        gate.wait(5)
        return _ok(prompt, provider)

    plan, errs = assembly_plan.validate(_raw())
    assert not errs
    first = assembly_ext.start_job("assembly one", plan, designer=designer)
    assert started.wait(5), "the first job must be designing"
    second = assembly_ext.start_job("assembly two", plan, designer=designer)
    time.sleep(0.2)
    try:
        assert assembly_ext.status(first)["status"] in ("running", "awaiting_review", "done")
        assert assembly_ext.status(second)["status"] == "queued", "the second job waits for the free slot"
    finally:
        gate.set()
    deadline = time.time() + 10
    while time.time() < deadline and assembly_ext.status(second)["status"] in ("queued", "running"):
        time.sleep(0.05)
    assert assembly_ext.status(second)["status"] == "awaiting_review"   # Batch 1: the gate, then assembly
    assert assembly_ext.running_jobs() == 0, "the slot is handed back"


def test_a_third_waiting_assembly_is_refused_with_503(monkeypatch):
    monkeypatch.setenv("IDRIS_ASSEMBLY_MAX_JOBS", "1")
    gate, started = threading.Event(), threading.Event()

    def designer(prompt, provider=None):
        started.set()
        gate.wait(5)
        return _ok(prompt, provider)

    plan, _ = assembly_plan.validate(_raw())
    first = assembly_ext.start_job("one", plan, designer=designer)
    assert started.wait(5)
    assembly_ext.start_job("two", plan, designer=designer)
    assembly_ext.start_job("three", plan, designer=designer)        # two are waiting now
    try:
        status, body = assembly_ext.handle_post("/assembly/design",
                                                {"request": "four", "plan": _raw(), "provider": None})
        assert status == 503 and "queue is full" in body["error"]
        assert body["max_jobs"] == 1
    finally:
        gate.set()
    deadline = time.time() + 10
    while time.time() < deadline and assembly_ext.running_jobs() or assembly_ext.queued_jobs():
        if time.time() > deadline:
            break
        time.sleep(0.05)
    assert assembly_ext.status(first)["status"] == "awaiting_review"


# ------------------------------------------------------------------ item 4: a number needs a source
def test_a_size_without_a_source_is_marked_and_asked_back():
    plan, errs = assembly_plan.validate(_raw(source=None))
    assert not errs
    assert plan["unsourced"] == [{"between": ["a", "b"], "param": "d", "value": 44.5, "key": "a-b.d"}]
    assert plan["interfaces"][0]["param_sources"]["d"] == "unsourced"
    assert any("Confirm d = 44.5 mm for a-b?" in q for q in plan["questions"])


def test_the_job_does_not_start_until_the_size_is_confirmed(monkeypatch):
    started = []
    monkeypatch.setattr(assembly_ext, "start_job", lambda *a, **k: started.append(a) or "job123")
    status, body = assembly_ext.handle_post("/assembly/design", {"request": "bike", "plan": _raw(None)})
    assert status == 200 and body["need"] == "confirmation"
    assert "job_id" not in body and started == [], "no job is started on an unsourced size"
    status, body = assembly_ext.handle_post("/assembly/design",
                                            {"request": "bike", "plan": _raw(None), "confirm": ["a-b.d"]})
    assert status == 200 and body["job_id"] == "job123" and started
    assert body["plan"]["interfaces"][0]["param_sources"]["d"] == "user (confirmed)"
    assert not body["plan"]["unsourced"]


def test_a_size_with_a_standard_source_needs_no_question():
    plan, _ = assembly_plan.validate(_raw(source="standard: ISO 5775"))
    assert plan["unsourced"] == [] and plan["questions"] == []
    assert plan["interfaces"][0]["param_sources"]["d"] == "standard: ISO 5775"


# ------------------------------------------------------------------ item 5: solvers inside their range
def test_a_resonance_gap_is_not_claimed_from_the_gearbox_housing_solver():
    q = {x["key"]: x for x in obj.measurable("motor listrik senyap")["quantities"]}["resonance_gap"]
    assert q["solver"] is None and q["status"] == "not_available"
    assert q["measured_by"] is None


def test_the_plate_solver_is_used_on_a_plate_and_not_on_a_block():
    plate = {"params": [{"k": "L", "value": 200}, {"k": "W", "value": 100},
                        {"k": "t", "value": 3, "wall": True}]}
    block = {"params": [{"k": "L", "value": 50}, {"k": "W", "value": 50},
                        {"k": "H", "value": 50, "wall": True}]}
    hot = {x["key"]: x for x in obj.measurable("plat lebih dingin", plate)["quantities"]}["max_temperature"]
    assert hot["status"] == "solver", "a 200 x 100 x 3 plate is a plate"
    cold = {x["key"]: x for x in obj.measurable("plat lebih dingin", block)["quantities"]}["max_temperature"]
    assert cold["status"] == "not_available" and "not a plate" in cold["reason"]


def test_the_plate_solver_is_not_used_when_the_part_is_unknown():
    q = {x["key"]: x for x in obj.measurable("plat lebih dingin")["quantities"]}["max_temperature"]
    assert q["status"] == "not_available", "without the parameters IDRIS does not assume a plate"


def test_variants_are_compared_only_on_what_every_variant_measured():
    measured = {q["key"]: q for q in obj.measurable("plat ringan dan kaku")["quantities"]}
    out = obj.rank([
        {"name": "A", "values": {"mass": 1.0, "max_deflection": 0.9}, "checks": {"pass": True}},
        {"name": "B", "values": {"mass": 3.0}, "checks": {"pass": True}},
    ], None, measured)
    assert out["compared_on"] == ["mass"], "deflection is not measured for B, so it cannot decide the ranking"
    assert [r["name"] for r in out["ranking"]] == ["A", "B"]
    assert "max_deflection" in out["ranking"][0]["not_compared"]


def test_variants_with_no_shared_quantity_are_not_ranked_at_all():
    measured = {q["key"]: q for q in obj.measurable("plat ringan dan kaku")["quantities"]}
    out = obj.rank([
        {"name": "A", "values": {"mass": 1.0}, "checks": {"pass": True}},
        {"name": "B", "values": {"max_deflection": 0.4}, "checks": {"pass": True}},
    ], None, measured)
    assert out["ranking"] == []
    assert {r["name"] for r in out["not_ranked"]} == {"A", "B"}
    assert "cannot be compared" in out["summary"] or "nothing is ranked" in out["summary"]
