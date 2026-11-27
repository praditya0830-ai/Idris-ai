"""Bagian E tests (AI is fake): the plan is validated, interface sizes reach both parts that meet, one failed
part does not stop the assembly, the estimate is honest, and the local return repair happens before another AI
call.
"""
import json
import time

import assembly_ext
import assembly_plan


def _call(payload):
    return lambda system, user: json.dumps(payload)


BICYCLE = {
    "parts": [
        {"id": "frame", "name": "Frame", "role": "carries the rider", "qty": 1, "needs": ["tube diameter"],
         "standard": False},
        {"id": "fork", "name": "Fork", "role": "holds the front wheel", "qty": 1, "needs": ["steerer diameter"],
         "standard": False},
        {"id": "wheel", "name": "Wheel", "role": "rolls", "qty": 2, "needs": ["rim diameter"], "standard": False},
        {"id": "bolt_m6", "name": "M6 bolt", "role": "fastens", "qty": 8, "needs": [], "standard": True},
    ],
    "interfaces": [
        {"between": ["frame", "fork"], "kind": "press", "params": {"steerer_d_mm": 44.5},
         "source": "standard: ISO 5775 BSD 622 mm"},
    ],
    "frames": [{"part_id": "frame", "origin_mm": [0, 0, 0], "axis": [0, 0, 1]},
               {"part_id": "fork", "origin_mm": [400, 0, 0], "axis": [0, 0, 1]},
               {"part_id": "wheel", "origin_mm": [400, 0, -350], "axis": [1, 0, 0]},
               {"part_id": "bolt_m6", "origin_mm": [0, 40, 0], "axis": [0, 1, 0]}],
    "sources": {"steerer_d_mm": "standard: ISO 5775"},
    "questions": [],
}


# ------------------------------------------------------------------ E2: the plan
def test_a_whole_bicycle_becomes_a_validated_plan_with_an_estimate():
    out = assembly_plan.plan_for("whole bicycle 170 cm, 700c", call=_call(BICYCLE))
    assert out["ok"] is True
    plan = out["plan"]
    assert [p["id"] for p in plan["parts"]] == ["frame", "fork", "wheel", "bolt_m6"]
    assert plan["interfaces"][0]["params"]["steerer_d_mm"] == 44.5
    est = out["estimate"]
    assert est["parts_to_design"] == 3, "a standard part needs no AI"
    assert est["parts_reused"] == 1, "the two wheels are one design"
    assert est["standard_parts"] == 1
    assert "minute" in est["text"]


def test_an_interface_that_names_a_missing_part_is_refused():
    bad = dict(BICYCLE, interfaces=[{"between": ["frame", "ghost"], "kind": "press", "params": {}}])
    out = assembly_plan.plan_for("whole bicycle", call=_call(bad))
    assert out["ok"] is False
    assert "not in the plan" in out["error"]


def test_a_frame_without_numbers_is_refused():
    bad = dict(BICYCLE, frames=[{"part_id": "frame", "origin_mm": [0, 0], "axis": [0, 0, 1]}])
    out = assembly_plan.plan_for("whole bicycle", call=_call(bad))
    assert out["ok"] is False
    assert "three numbers" in out["error"]


def test_too_many_parts_asks_for_sub_assemblies(monkeypatch):
    monkeypatch.setenv("IDRIS_ASSEMBLY_MAX_PARTS", "2")
    out = assembly_plan.plan_for("whole bicycle", call=_call(BICYCLE))
    assert out["ok"] is False
    assert "IDRIS_ASSEMBLY_MAX_PARTS" in out["error"]
    assert "sub-assemblies" in out["error"]


def test_a_broken_reply_is_a_clear_message_not_a_crash():
    out = assembly_plan.plan_for("whole bicycle", call=lambda s, u: "I would split it into a frame and wheels")
    assert out["ok"] is False and out["plan"] is None
    assert "structured data" in out["error"]


# ------------------------------------------------------------------ E3: the parts
def _ok_design(prompt, provider=None):
    return {"ok": True, "cad": {"title": "part", "code": "return { body };", "params": []},
            "fidelity": "full", "validated": False, "not_checked": []}


def test_both_parts_that_meet_get_the_same_interface_value():
    plan, errs = assembly_plan.validate(BICYCLE)
    assert not errs
    seen = {}

    def designer(prompt, provider=None):
        for pid in ("frame", "fork"):
            if plan["parts"][[p["id"] for p in plan["parts"]].index(pid)]["name"] in prompt:
                seen[pid] = prompt
        return _ok_design(prompt, provider)

    parts, asm = assembly_ext.design_parts(plan, designer=designer)
    assert "44.5" in seen["frame"] and "44.5" in seen["fork"], "one skeleton value, both parts"
    assert [p["status"] for p in parts if p["id"] != "bolt_m6"] == ["done", "done", "done"]
    assert asm["frames_complete"] is True
    assert "kinematics" in asm["not_checked"]


def test_one_failed_part_does_not_stop_the_assembly():
    plan, _ = assembly_plan.validate(BICYCLE)

    def designer(prompt, provider=None):
        if "Fork" in prompt:
            raise RuntimeError("provider timed out")
        return _ok_design(prompt, provider)

    parts, asm = assembly_ext.design_parts(plan, designer=designer)
    by_id = {p["id"]: p for p in parts}
    assert by_id["fork"]["status"] == "failed"
    assert "provider timed out" in by_id["fork"]["error"]
    assert by_id["frame"]["status"] == "done", "the other parts still finish"
    assert asm["parts"]


def test_a_placeholder_from_the_ladder_is_kept_as_a_part():
    plan, _ = assembly_plan.validate(BICYCLE)

    def designer(prompt, provider=None):
        if "Fork" in prompt:
            return {"ok": True, "fidelity": "placeholder",
                    "cad": {"title": "Cylinder envelope", "code": "return { body };", "params": []}}
        return _ok_design(prompt, provider)

    parts, _asm = assembly_ext.design_parts(plan, designer=designer)
    assert {p["id"]: p for p in parts}["fork"]["fidelity"] == "placeholder"


# ------------------------------------------------------------------ E6: local repair, no extra AI call
def test_a_missing_return_is_added_locally():
    code = "const base = makeBox([0, 0, 0], [10, 10, 10]);"
    fixed, changed = assembly_plan.repair_return(code)
    assert changed is True and "return { body: base }" in fixed


def test_code_that_already_returns_is_left_alone():
    code = "const body = makeBox([0, 0, 0], [1, 1, 1]);\nreturn { body };"
    fixed, changed = assembly_plan.repair_return(code)
    assert changed is False and fixed == code


def test_a_failed_part_with_code_is_repaired_and_flagged():
    plan, _ = assembly_plan.validate(BICYCLE)

    def designer(prompt, provider=None):
        return {"ok": False, "fidelity": None, "answer": "code must end with return { body }",
                "cad": {"title": "part", "code": "const body = makeBox([0,0,0],[1,1,1]);", "params": []}}

    parts, _asm = assembly_ext.design_parts(plan, designer=designer)
    fork = {p["id"]: p for p in parts}["fork"]
    assert fork["status"] == "failed" and fork["repaired"] is True
    assert "return { body: body }" in fork["cad"]["code"]
    assert "added the missing return statement locally" in fork["error"]


# ------------------------------------------------------------------ E3/E4b: the job and the estimate
def test_the_job_reports_status_parts_and_eta():
    plan, _ = assembly_plan.validate(BICYCLE)
    job_id = assembly_ext.start_job("whole bicycle", plan, designer=_ok_design, review_gate=True)
    deadline = time.time() + 10
    st = assembly_ext.status(job_id)
    while st and st["status"] in ("queued", "running") and time.time() < deadline:
        time.sleep(0.05)
        st = assembly_ext.status(job_id)
    # Batch 1: a job stops at the review gate, and the assembly is placed only after approve.
    assert st["status"] == "awaiting_review"
    assert st["eta_s"] == 0.0
    assert st["elapsed_s"] >= 0
    assert len(st["parts"]) == 4
    assert st["assembly"] is None and st["review"]["parts"]
    assert st["not_checked"]
    code, out = assembly_ext.handle_post(f"/assembly/job/{job_id}/review", {"action": "approve"})
    assert code == 200 and out["status"] == "done"
    assert out["assembly"]["parts"]


def test_an_unknown_job_is_a_404_not_a_crash():
    status_, body = assembly_ext.handle_get("/assembly/design/deadbeef")
    assert status_ == 404 and "unknown assembly job" in body["error"]


def test_the_design_route_refuses_a_bad_plan_without_calling_ai():
    plan = {"parts": [{"id": "a", "name": "A"}],
            "interfaces": [{"between": ["a", "missing"], "kind": "press", "params": {}}]}
    status_, body = assembly_ext.handle_post("/assembly/design", {"request": "whole bicycle", "plan": plan})
    assert status_ == 422 and "not in the plan" in body["error"]
