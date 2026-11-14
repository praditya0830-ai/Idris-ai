"""ACE exploration for free-form parts: plan -> (browser builds, /part-check) -> rank -> model -> wider search.
The 'FEA' here is the closed-form cantilever, so the true optimum is known and the search can be judged."""
import json
import math

import cad_codegen as cg
import cad_explore as ex

CAD = {"title": "Cantilever arm", "params": [
    {"k": "L", "label": "Length", "unit": "mm", "value": 100, "min": 50, "max": 200, "source": "user"},
    {"k": "b", "label": "Width", "unit": "mm", "value": 20, "min": 10, "max": 40, "source": "default"},
    {"k": "t", "label": "Thickness", "unit": "mm", "value": 5, "min": 2, "max": 12, "source": "default"}],
    "variants": [{"name": "Sturdy", "params": {"t": 8}}]}
F, YIELD, RHO = 50.0, 50.0, 1.24e-6          # N, MPa, kg/mm3 (PLA)


def fea(params):                              # sigma = 6 F L / (b t^2), tip load on a cantilever
    L, b, t = 100.0, params["b"], params["t"]
    return {"min_safety_factor": YIELD / (6 * F * L / (b * t * t)), "mass_kg": RHO * L * b * t,
            "max_displacement_mm": 4 * F * L ** 3 / (2300 * b * t ** 3)}


def run_rounds(n_rounds=3, **kw):
    results, last = [], None
    for rnd in range(1, n_rounds + 1):
        last = ex.plan(CAD, {"design_factor": 2}, results or None, 8, rnd, **kw)
        if last["done"] or not last["variants"]:
            break
        results += [{"name": v["name"], "params": v["params"], **fea(v["params"])} for v in last["variants"]]
    return ex.plan(CAD, {"design_factor": 2}, results, 8, ex.MAX_ROUNDS), results


def test_round_one_keeps_engineer_sizes_and_spreads_the_rest():
    r = ex.plan(CAD, {"design_factor": 2})
    assert [v["k"] for v in r["variables"]] == ["b", "t"] and r["fixed"][0]["k"] == "L"
    assert r["fixed"][0]["why"] == "given by the engineer"
    names = [v["name"] for v in r["variants"]]
    assert names[0] == "Current" and "Sturdy" in names and len(names) == 8
    assert all(set(v["params"]) == {"b", "t"} for v in r["variants"])
    assert all(10 <= v["params"]["b"] <= 40 and 2 <= v["params"]["t"] <= 12 for v in r["variants"])
    assert len({json.dumps(v["params"], sort_keys=True) for v in r["variants"]}) == 8      # no duplicates
    assert not r["ranking"] and "Latin hypercube" in r["method"]


def test_ranking_passes_first_then_lightest_and_gives_reasons():
    res = [{"name": "A", "params": {"t": 6}, "min_safety_factor": 1.6, "mass_g": 32.4},
           {"name": "B", "params": {"t": 14}, "min_safety_factor": 3.8, "mass_g": 75.6},
           {"name": "C", "params": {"t": 10}, "min_safety_factor": 2.1, "mass_g": 54.0},
           {"name": "D", "params": {"t": 3}, "status": "not checked", "reason": "support face not found"},
           {"name": "E", "params": {"t": 2}, "geometry_ok": False}]
    r = ex.rank(res, {"design_factor": 2})
    assert [x["name"] for x in r["ranking"]] == ["C", "B", "A", "D", "E"] and r["best"] == "C"
    by = {x["name"]: x for x in r["ranking"]}
    assert by["C"]["score"] == 100 and 50 < by["B"]["score"] < 100 and by["A"]["score"] < 50 and by["D"]["score"] is None
    assert "40 % heavier" in by["B"]["reasons"][-1] and "below the design factor" in by["A"]["reasons"][0]
    assert r["counts"] == {"pass": 2, "fail": 1, "not_checked": 2} and "Best: C" in r["summary"]
    r = ex.rank(res[:1], {"design_factor": 2})
    assert r["best"] is None and "none meets" in r["summary"]
    r = ex.rank(res[:3], {"design_factor": 2, "max_deflection_mm": 1.0})        # no deflection sent -> not judged on it
    assert r["best"] == "C"


def test_model_search_finds_the_known_optimum():
    """True optimum on the 0.5 mm grid: b = 10, t = 11 (SF 2.02, b*t = 110). Round 1 alone is far from it."""
    r1 = ex.plan(CAD, {"design_factor": 2})
    first = ex.rank([{"name": v["name"], "params": v["params"], **fea(v["params"])} for v in r1["variants"]], {"design_factor": 2})
    final, results = run_rounds(3)
    best = next(x for x in final["ranking"] if x["name"] == final["best"])
    bt = best["params"]["b"] * best["params"]["t"]
    assert best["min_safety_factor"] >= 2 and bt <= 110 * 1.06, (best, bt)
    first_best = next((x for x in first["ranking"] if x["name"] == first["best"]), None)
    assert first_best is None or best["mass_kg"] <= first_best["mass_kg"]
    assert len(results) <= 24                                   # a handful of real FEA runs, not thousands


def test_model_reports_what_it_did_and_never_poses_as_a_result():
    r1 = ex.plan(CAD, {"design_factor": 2})
    res = [{"name": v["name"], "params": v["params"], **fea(v["params"])} for v in r1["variants"]]
    r2 = ex.plan(CAD, {"design_factor": 2}, res, 8, 2)
    m = r2["model"]
    assert m["candidates_scored"] == ex.CANDIDATES and m["use"].startswith("chooses what to check next")
    assert math.isclose(m["safety_factor_exponents"]["t"], 2.0, abs_tol=0.02)          # SF ~ b t^2: the model recovers it
    assert math.isclose(m["safety_factor_exponents"]["b"], 1.0, abs_tol=0.02)
    assert math.isclose(m["mass_exponents"]["t"], 1.0, abs_tol=0.02)
    assert r2["variants"] and all(v["predicted"]["label"] == "model prediction, not a result" for v in r2["variants"])
    assert all("predicted" not in x for x in r2["ranking"])


def test_too_few_results_means_more_spread_not_a_model():
    res = [{"name": "Current", "params": {"b": 20, "t": 5}, **fea({"b": 20, "t": 5})}]
    r = ex.plan(CAD, {"design_factor": 2}, res, 6, 2)
    assert "model" not in r and r["variants"] and "needs 4" in r["method"]


def test_nothing_free_is_said_plainly():
    cad = {"params": [{"k": "L", "value": 100, "min": 50, "max": 200, "source": "user"}]}
    r = ex.plan(cad, {})
    assert r["done"] and not r["variants"] and "free_user_params" in r["notes"][0]
    assert ex.plan(cad, {}, free_user_params=True)["variants"]


# ------------------------------------------------------------------ requirements in the cad contract
RAW = {"title": "Hook", "kind": "functional", "process": "fdm", "material": "pla",
       "params": [{"k": "t", "label": "Thickness", "unit": "mm", "value": 5, "min": 3, "max": 10, "source": "user"}],
       "code": "const { t } = P;\nconst body = makeBaseBox(30, t, 60);\nreturn { body };", "expect": [], "missing": [],
       "variants": [], "holes": []}


def test_stated_load_is_kept_with_its_conversion():
    raw = {**RAW, "requirements": {"loads": [{"force_N": [0, 0, -19.62], "where": {"plane": "x", "side": "max"}, "label": "Towel"}],
                                   "supports": [{"where": {"plane": "x", "side": "min"}, "label": "Wall"}],
                                   "design_factor": 3, "max_deflection_mm": 9}}
    cad, errs, warns = cg.validate(raw, "wall hook 5 mm thick, must hold 2 kg, safety factor 3")
    assert not errs
    req = cad["requirements"]
    assert req["loads"][0]["from"] == "2 kg x 9.81" and req["supports"][0]["where"] == {"plane": "x", "side": "min"}
    assert req["design_factor"] == 3 and req["max_deflection_mm"] is None            # 9 mm was never said
    assert "Load to check: Towel 19.6 N (2 kg x 9.81)" in cg.requirements_text(cad)


def test_invented_load_is_dropped_and_idris_asks_instead():
    raw = {**RAW, "requirements": {"loads": [{"force_N": [0, 0, -50], "where": {"plane": "x", "side": "max"}}],
                                   "supports": [{"where": {"plane": "x", "side": "min"}}]}}
    cad, errs, warns = cg.validate(raw, "wall hook 5 mm thick")
    assert not errs and cad["requirements"] is None
    assert any("does not guess loads" in w for w in warns)
    assert "tell me the load" in cg.requirements_text(cad)
    cad, _, _ = cg.validate(dict(RAW), "wall hook 5 mm thick")
    assert cad["requirements"] is None


def test_selectors_match_part_check_forms():
    assert cg._selector({"plane": "z"}) == {"plane": "z", "side": "min"}
    assert cg._selector({"box": [[0, 0, 0], [1, 2, 3]]}) == {"box": [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]]}
    assert cg._selector({"cylinder": {"center": [1, 2, 3], "radius": 2.25, "axis": "x", "length": 5}})["cylinder"]["axis"] == "x"
    assert cg._selector({"face": "top"}) is None and cg._selector({"box": [[0, 0], [1, 1]]}) is None
    assert cg._stated_force(500, "carries 500 N") == "500 N" and cg._stated_force(2000, "load 2 kN") == "2 kN x 1000"
    assert cg._stated_force(123, "carries 500 N") is None


def test_follow_up_about_an_open_part_reaches_the_ai_with_that_part(monkeypatch):
    seen = {}
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    monkeypatch.setenv("IDRIS_CAD_ORDER", "gemini")
    monkeypatch.setenv("IDRIS_MODEL_GEMINI", "g")
    raw = {**RAW, "requirements": {"loads": [{"force_N": [0, 0, -19.62], "where": {"plane": "x", "side": "max"}}],
                                   "supports": [{"where": {"plane": "x", "side": "min"}}]}}

    def fake(prov, system, user, max_tokens=500, timeout=None):
        seen["user"], seen["system"] = user, system
        return json.dumps(raw), "g"
    import llm
    monkeypatch.setattr(llm, "call_provider", fake)
    ctx = {"current_cad": {"request": "wall hook 5 mm thick", "title": "Hook", "params": RAW["params"], "code": RAW["code"],
                           "requirements": None}}
    r = cg.generate("it must hold 2 kg, screwed to the wall", ctx)
    assert r["ok"] and r["cad"]["requirements"]["loads"][0]["from"] == "2 kg x 9.81"
    assert "EXISTING design" in seen["user"] and "makeBaseBox" in seen["user"] and '"requirements"' in seen["system"]
    assert [p["source"] for p in r["cad"]["params"]] == ["user"]            # 5 mm from the original request stays the engineer's
