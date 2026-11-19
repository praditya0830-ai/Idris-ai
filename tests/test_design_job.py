"""Bagian G: the design runs as a job, in rounds, and stops only when it passes or the rounds run out.

No AI and no kernel: the code writer is a fake, the measurements come from the browser (passed in), and FEA is
a fake that is only used when a load was given. The rules under test are the ones that keep the log honest and
keep an engineer's size out of reach of a repair.
"""
import time

import design_job as dj
from physics import dfm_rules


def _cad(wall=3.0, user_length=40.0, req=None):
    return {"title": "L bracket", "kind": "functional", "process": "cnc", "material": "aluminium",
            "note": "", "code": ("let body = base();\n// step: base plate\nbody = plate(body);\n"
                                 "// step: holes\nbody = holes(body);\nreturn { body };"),
            "params": [{"k": "L", "label": "Length", "label_id": "Panjang", "unit": "mm", "value": user_length,
                        "min": 10, "max": 200, "source": "user"},
                       {"k": "t", "label": "Wall", "label_id": "Tebal", "unit": "mm", "value": wall,
                        "min": 0.4, "max": 12, "source": "default", "wall": True}],
            "expect": [{"label": "Length", "axis": "x", "value": user_length, "tol": 0.5}],
            "holes": [{"x": 20, "y": 10, "z_top": 3, "axis": "z", "d": 4.5}],
            "variants": [], "missing": [], "requirements": req, "read_by": "fake (model)"}


def _ok(wall=3.0, user_length=40.0):
    return lambda msg, ctx, prov: {"ok": True, "cad": _cad(wall, user_length), "answer": "", "attempts": []}


# ------------------------------------------------------------------ G1: the limits are the loose ones
def test_the_default_limits_are_the_loose_g_ones():
    assert dj.setting("IDRIS_CAD_CALL_S") == 120
    assert dj.setting("IDRIS_PART_BUDGET_S") == 600
    assert dj.setting("IDRIS_CAD_MAX_LINES") == 200
    assert dj.setting("IDRIS_DESIGN_ROUNDS") == 4


def test_the_deadline_comes_from_the_env_budget_not_the_old_85_s(monkeypatch):
    """A design is no longer cut by the HTTP limit: the loop stops only when the job budget is used up."""
    monkeypatch.setenv("IDRIS_PART_BUDGET_S", "0.05")
    calls = []

    def slow(msg, ctx, prov):
        calls.append(1)
        time.sleep(0.12)
        return {"ok": True, "cad": _cad(wall=0.5)}          # a wall the DFM table rejects: the loop wants round 2

    res = dj.run("bracket L 40", cad_fn=slow)
    assert len(calls) == 1, "round 2 must not start a new AI call once the budget is gone"
    assert res["budget_s"] == 0.05
    assert any("budget" in (e.get("message") or "") for e in res["events"]), res["events"]


# ------------------------------------------------------------------ G2: the circle stops when it passes
def test_the_loop_stops_as_soon_as_every_check_passes():
    calls = []

    def cad_fn(msg, ctx, prov):
        calls.append(1)
        return {"ok": True, "cad": _cad()}

    res = dj.run("bracket L 40", cad_fn=cad_fn)
    assert len(calls) == 1 and res["rounds"] == 1
    assert res["label"] == "Geometry only", res["checks"]
    wall = [c for c in res["checks"] if c["id"] == "min_wall"][0]
    assert wall["status"] == "pass" and wall["source"].startswith("Machining guideline")


def test_a_repair_may_never_change_a_size_the_engineer_gave():
    seq = [_cad(wall=0.5, user_length=40.0), _cad(wall=3.0, user_length=50.0)]

    def cad_fn(msg, ctx, prov):
        return {"ok": True, "cad": seq.pop(0)}

    res = dj.run("bracket L 40", cad_fn=cad_fn)
    assert res["label"] == "Not validated"
    assert any(c["id"] == "engineer_sizes" and c["status"] == "fail" for c in res["checks"])
    assert "engineer size" in res["answer"]
    assert any(e["step"] == "repair" and e["status"] == "fail" for e in res["events"])


def test_a_load_without_fea_is_never_a_pass():
    req = {"loads": [{"force_N": [0, 0, -100], "where": {"plane": "z", "side": "min"}, "label": "Load"}],
           "supports": [{"where": {"plane": "z", "side": "min"}}], "design_factor": 2.0}
    res = dj.run("bracket 100 N", cad_fn=lambda m, c, p: {"ok": True, "cad": _cad(req=req)})
    fea = [c for c in res["checks"] if c["id"] == "fea"][0]
    assert fea["status"] == "not_checked" and res["label"] == "Not validated"


# ------------------------------------------------------------------ G2 step 6: the proposal uses real FEA numbers
def test_a_proposal_carries_the_numbers_of_an_fea_that_really_ran():
    req = {"loads": [{"force_N": [0, 0, -100], "where": {"plane": "z", "side": "min"}, "label": "Load"}],
           "supports": [{"where": {"plane": "z", "side": "min"}}], "design_factor": 2.0}
    ran = []

    def fea(cad):
        ran.append("fea")
        return {"sf": 1.3, "source": "fea_part_3d (bench: cantilever)"}

    def trial(cad, target):
        ran.append("trial")
        return {"param": "wall t", "from": 4, "to": 6, "sf": 2.1}

    res = dj.run("bracket 100 N", cad_fn=lambda m, c, p: {"ok": True, "cad": _cad(req=req)},
                 fea=fea, fea_trial=trial)
    assert ran == ["fea", "trial"]
    assert res["proposal"] and "2.1" in res["proposal"] and "1.3" in res["proposal"]
    step = [e for e in res["events"] if e["step"] == "fea"][0]
    assert step["numbers"]["sf"] == 1.3 and step["numbers"]["target"] == 2.0
    assert res["label"] == "Not validated"


# ------------------------------------------------------------------ G4: the build order comes from the code
def test_cad_steps_follow_the_step_markers():
    code = "let a = 1;\n// step: base plate\nlet b = 2;\n// step: holes\nreturn { body };"
    steps = dj.split_steps(code)
    assert [s["name"] for s in steps] == ["setup", "base plate", "holes"]
    assert steps[-1]["code_until_here"] == code
    assert "// step: holes" not in steps[1]["code_until_here"]
    assert dj.split_steps("return { body };")[0]["name"] == "part"


# ------------------------------------------------------------------ G2 step 4: DFM with its source
def test_dfm_rejects_a_half_mm_wall_for_cnc_aluminium_with_the_rule_source():
    out = dfm_rules.check({"process": "cnc", "material": "aluminium",
                           "params": [{"k": "t", "value": 0.5, "wall": True, "label": "wall"}]})
    fail = [c for c in out["checks"] if c["status"] == "fail"]
    assert fail and fail[0]["id"] == "min_wall" and fail[0]["limit"] == 0.8
    assert "Machining guideline" in fail[0]["source"]


def test_dfm_never_reports_a_pass_for_something_it_cannot_see():
    out = dfm_rules.check({"process": "cnc", "material": "aluminium",
                           "params": [{"k": "t", "value": 3.0, "wall": True, "label": "wall"}]})
    assert out["ok"] is True
    assert any(c["id"] == "inside_corner_radius" and c["status"] == "not_checked" for c in out["checks"])


# ------------------------------------------------------------------ G1/G3: the job is pollable
def test_a_job_is_pollable_and_keeps_its_events_and_label():
    job_id = dj.submit("bracket L 40", cad_fn=_ok())
    assert job_id
    st = None
    for _ in range(400):
        st = dj.status(job_id)
        if st and st["status"] in ("done", "failed"):
            break
        time.sleep(0.02)
    assert st and st["status"] == "done", st
    assert [e["step"] for e in st["events"]][:3] == ["plan", "code", "build"]
    assert st["result"]["label"] == "Geometry only"
    assert st["result"]["cad"]["steps"][0]["name"] == "setup"


def test_an_unknown_job_is_none_not_an_error():
    assert dj.status("nope") is None
