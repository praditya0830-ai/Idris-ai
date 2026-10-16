"""Tests for the 3D FEA engine, the housing solver and the /fea job endpoint."""
import time
import numpy as np
from ace_core.http_routes import handle_get, handle_post, encode_json
from physics.fem3d import box_model, VoxelModel
from physics.housing import build_layout, build_model, bearing_loads, fixture_nodes
from solvers._example import EXAMPLE_VARIANT

REQ = {"analysis": "static", **EXAMPLE_VARIANT, "fixture": "base", "mesh": "coarse"}


def test_3d_engine_benchmarks():
    rep = handle_get("/verify/fea_solid_block")[1]
    assert rep["all_pass"], rep


def test_brick_grid_matches_cube_grid():
    # same cantilever meshed with cubes and with a rectilinear grid (non-uniform spacing)
    E, nu, P = 200000.0, 0.3, 1000.0
    a = box_model(100, 10, 10, 2.5)
    xs = np.r_[np.linspace(0, 50, 11), np.linspace(50, 100, 41)[1:]]   # 5 mm then 1.25 mm cells
    b = VoxelModel(np.ones((len(xs) - 1, 4, 4), bool), planes=(xs, np.linspace(0, 10, 5), np.linspace(0, 10, 5)))
    tips = []
    for m in (a, b):
        x = m.nodes
        end = np.where(np.abs(x[:, 0] - 100) < 1e-9)[0]
        loads = np.zeros((m.n_nodes, 3))
        loads[end, 1] = -P / len(end)
        r = m.solve(E, nu, np.where(x[:, 0] < 1e-9)[0], loads)
        tips.append(-r["u"][end, 1].mean())
    assert abs(tips[0] / tips[1] - 1) < 0.02


def test_surface_is_closed_and_outward():
    m = box_model(20, 10, 6, 2)
    ids, tris = m.surface()
    P = m.nodes[ids]
    vol = np.einsum("ij,ij->i", P[tris[:, 0]], np.cross(P[tris[:, 1]], P[tris[:, 2]])).sum() / 6
    assert abs(vol - 20 * 10 * 6) < 1e-6
    edges = np.sort(np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    assert set(counts) == {2}


def test_layout_matches_assembly3d():
    from assembly3d import build_assembly
    v = EXAMPLE_VARIANT
    asm = build_assembly(v["design"], v["kinematics"], v["shaft"], v["housing_params"])
    L = build_layout(v["design"], v["kinematics"], {}, v["shaft"], v["housing_params"])
    assert np.allclose(asm["layout"]["axis_a_mm"], [L["x_a"], L["yc"]])
    assert np.allclose(asm["layout"]["axis_b_mm"], [L["x_b"], L["yc"]])
    housing = np.vstack([p["verts"] for p in asm["parts"] if p["kind"] == "housing"])     # body + end cover
    m = build_model(L, "coarse")
    assert np.allclose(m.nodes.min(0), housing.min(0), atol=1e-6)
    assert np.allclose(m.nodes.max(0), housing.max(0), atol=1e-6)
    # tangential force of the last stage equals the variant's loads.tangential_force_n
    assert abs(L["meshes"][-1]["Wt"] - v["loads"]["tangential_force_n"]) / v["loads"]["tangential_force_n"] < 0.01


def test_housing_equilibrium():
    v = EXAMPLE_VARIANT
    L = build_layout(v["design"], v["kinematics"], {}, v["shaft"], v["housing_params"])
    m = build_model(L, "coarse")
    loads, applied = bearing_loads(m, L)
    assert len(applied) == 4
    assert np.allclose(loads.sum(0), 0, atol=1e-6)
    fx = fixture_nodes(m, L, "base")
    r = m.solve(71700, 0.33, fx, loads)
    assert np.allclose(r["reactions"][fx].sum(0), 0, atol=1e-3 * np.abs(loads).sum())


def test_fea_job_flow():
    code, b = handle_post("/fea", REQ)
    assert code == 202 and b["status"] == "queued" and b["job_id"]
    t0 = time.time()
    while True:
        s = handle_get(f"/fea/{b['job_id']}")[1]
        if s["status"] in ("done", "failed") or time.time() - t0 > 120:
            break
        assert s["stage"] in ("Queued", "Meshing", "Applying loads", "Solving", "Post-processing")
        time.sleep(0.2)
    assert s["status"] == "done", s
    r = s["result"]
    sf = r["surface"]
    n = len(sf["von_mises"])
    assert len(sf["positions"]) == 3 * n and len(sf["displacement"]) == 3 * n
    assert len(sf["indices"]) % 3 == 0 and max(sf["indices"]) < n
    P = np.array(sf["positions"]).reshape(-1, 3)
    hp = EXAMPLE_VARIANT["housing_params"]
    assert P.min() >= -1e-6 and np.allclose(P.max(0), [hp["envelope_width"], hp["envelope_height"], hp["envelope_depth"]])
    assert r["yield_mpa"] == 503 and r["mesh"]["elements"] > 0 and r["applied_loads"]


def test_fea_errors_are_readable():
    c, b = handle_post("/fea?sync=1", dict(REQ, housing_material="Unobtainium"))
    assert c == 400 and not b["success"] and "Available" in b["error"]
    assert handle_get("/fea/doesnotexist")[0] == 404


def test_material_aliases():
    c, b = handle_post("/fea?sync=1", dict(REQ, housing_material="Steel4140"))
    assert c == 200 and b["result"]["material"] == "Steel-4140"


def test_flange_fixture():
    c, b = handle_post("/fea?sync=1", dict(REQ, fixture="flange"))
    assert c == 200 and b["result"]["summary"]["max_von_mises_mpa"] > 0


def test_gzip_encoding():
    data, h = encode_json({"a": list(range(5000))}, "gzip, deflate")
    assert h.get("Content-Encoding") == "gzip"
    assert encode_json({"a": 1}, "")[1] == {}
