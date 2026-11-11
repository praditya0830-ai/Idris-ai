import math
import time

import numpy as np

import solvers  # noqa: F401
from ace_core import analysis, jobs
from ace_core.registry import verify
from physics import part_fea as pf
from solvers.part_fea import BOX_EX, _box_mesh


def test_benchmarks_pass():
    r = verify("fea_part_3d")
    assert r["all_pass"] and r["passed"] == 2


def test_box_volume_exact_and_equilibrium():
    o = pf.run(dict(BOX_EX, element_size_mm=2.5))
    assert o["volume"]["error_pct"] == 0.0 and o["equilibrium_error"] < 1e-6
    assert np.allclose(o["reaction_N"], [0, -1000, 0], atol=1e-3)          # reaction = applied load
    assert math.isclose(o["mass_kg"], 10000e-9 * pf.MATERIALS["aluminium"]["rho"])


def test_stl_input_and_box_selector():
    from physics.simple_part import write_stl
    from assembly3d import build_box
    import base64, os, tempfile
    v, f = build_box(0, 60, 0, 20, 0, 5)
    p = os.path.join(tempfile.mkdtemp(), "plate.stl")
    write_stl(v, f, p)
    b64 = base64.b64encode(open(p, "rb").read()).decode()
    o = pf.run({"stl_base64": b64, "material": "pla", "element_size_mm": 2.5,
                "supports": [{"where": {"box": [[0, 0, 0], [10, 20, 0]]}}],          # bottom pad under one end
                "loads": [{"force_N": [0, 0, -20], "where": {"box": [[55, 0, 5], [60, 20, 5]]}}]})
    assert o["applied_loads"][0]["area_mm2"] > 0 and o["max_displacement_mm"] > 0
    assert o["material"]["name"].startswith("PLA")


def test_errors_are_clear():
    for bad, word in ((dict(BOX_EX, supports=[{"where": {"box": [[500, 500, 500], [501, 501, 501]]}}]), "selects no surface"),
                      (dict(BOX_EX, element_size_mm=0.1), "elements > limit"),
                      ({k: v for k, v in BOX_EX.items() if k not in ("positions", "indices")}, "positions")):
        try:
            pf.run(dict(bad, _max_elements=60000))
            assert False, word
        except ValueError as e:
            assert word in str(e), (word, e)


def test_part_check_job_endpoint():
    code, body = analysis.handle_post("/part-check?sync=1", dict(BOX_EX, element_size_mm=2.5, material="steel"))
    assert code == 200 and body["status"] == "done"
    r = body["result"]
    assert {"surface", "summary", "volume", "checks", "limitations"} <= set(r)
    assert len(r["surface"]["positions"]) == 3 * len(r["surface"]["von_mises"])
    code, body = analysis.handle_post("/part-check", dict(BOX_EX, element_size_mm=2.5))
    assert code == 202
    for _ in range(100):
        c, st = analysis.handle_get("/part-check/" + body["job_id"])
        if st["status"] in ("done", "failed"):
            break
        time.sleep(0.1)
    assert st["status"] == "done"


def test_convergence_study_reports_gci():
    o = __import__("solvers.part_fea", fromlist=["_run"])._run(dict(BOX_EX, element_size_mm=2.0, convergence_study=True),
                                                               lambda *a, **k: None)["outputs"]
    assert set(o["mesh_convergence"]) == {"max_displacement_mm", "von_mises_p995_MPa"}
