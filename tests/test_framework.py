"""Tests for the ACE solver framework and every registered solver's benchmarks."""
from ace_core.http_routes import handle_get, handle_post
from ace_core.schema import validate
from ace_core import registry
import physics.fem2d as fem


def test_all_benchmarks_pass():
    rep = registry.verify()
    failed = [(r["solver"], r["benchmark"]) for r in rep["report"] if not r["pass"]]
    assert not failed, failed


def test_all_benchmarks_pass_without_scipy():
    old = fem.HAVE_SCIPY
    fem.HAVE_SCIPY = False
    try:
        assert handle_get("/verify/fea_plane")[1]["all_pass"]
    finally:
        fem.HAVE_SCIPY = old


def test_every_solver_has_benchmark_and_example():
    for s in registry.catalog():
        assert s["benchmarks"], s["name"]
        assert s["example"], s["name"]
        code, res = handle_post(f"/solve/{s['name']}", s["example"])
        assert code == 200 and res["ok"], (s["name"], res.get("errors"))
        assert res["contract"] == "1.0" and res["verdict"] in ("PASS", "FAIL", "OK")


def test_schema_rejects_unknown_and_bad_types():
    _, errs = validate({"a": {"type": "number", "required": True}}, {"a": "x", "b": 1})
    assert any("unknown" in e for e in errs) and any("number" in e for e in errs)
    _, errs = validate({"a": {"type": "number"}}, {"a": True})
    assert errs


def test_unknown_solver_404():
    assert handle_post("/solve/does_not_exist", {})[0] == 404


def test_solver_crash_is_contained():
    code, res = handle_post("/solve/fea_plane", {
        "case": "custom", "nodes_mm": [[0, 0], [10, 0], [10, 10], [0, 10]], "elements": [[0, 3, 2, 1]],
        "supports": [{"node": 0}], "loads": [{"node": 1, "Fx_N": 1}]})
    assert code == 400 and not res["ok"] and res["errors"]


def test_llm_tools_cover_all_solvers():
    tools = handle_get("/solvers/tools")[1]["function_declarations"]
    assert {t["name"] for t in tools} == {f"ace_{s['name']}" for s in registry.catalog()}


def test_fea_mesh_convergence_reported():
    res = handle_post("/solve/fea_plane", {"case": "plate_with_hole", "hole_radius_mm": 5,
                                           "half_width_mm": 50, "stress_MPa": 100,
                                           "include_field": False})[1]
    mc = res["outputs"]["mesh_convergence"]
    assert mc["relative_change"] < 0.05
    assert 2.9 < res["outputs"]["Kt_gross"] < 3.2


def test_verify_page_is_non_blocking():
    import time
    t0 = time.time()
    code, body = handle_get("/verify?refresh=1")
    assert code == 200 and body["status"] in ("running", "done")
    assert time.time() - t0 < 1.0
    for _ in range(300):
        code, body = handle_get("/verify")
        if body["status"] == "done":
            break
        time.sleep(0.1)
    assert body["status"] == "done" and body["all_pass"], body
